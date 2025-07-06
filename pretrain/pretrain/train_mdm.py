import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import re
import os
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse
from config import JobConfig


def get_lr(step_num,
           lr: float = 2e-4,
           min_lr: float = 2e-5,
           warmup_steps: int = 1000,
           lr_decay_steps: int = 600000,
           decay_type: str = "cosine" # "linear", "sqrt", "cosine", "stable"
           ) -> float:
    # 1) linear warmup for warmup_steps steps
    if step_num < warmup_steps:
        return lr * step_num / warmup_steps

    # 2) if step_num > lr_decay_steps, return min learning rate
    if step_num > lr_decay_steps:
        return min_lr

    # 3) decay according to the specified decay type
    decay_ratio = (step_num - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    if decay_type == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    elif decay_type == "linear":
        coeff = 1.0 - decay_ratio
    elif decay_type == "sqrt":
        coeff = 1.0 - math.sqrt(decay_ratio)
    elif decay_type == "stable":
        coeff = 1.0
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    return min_lr + coeff * (lr - min_lr)


def forward_process(batch, mask_token_id=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, mask_token_id, batch)
    return noisy_batch, mask_indices, p_mask


def main(config):
    logger = step_csv_logger(config.training.output_folder,
                             config.training.job_name,
                             flush_logs_every_n_steps=config.training.log_iter_interval)
    wandb_logger = WandbLogger(name=config.training.wandb_name,
                               save_dir=config.training.wandb_dir,
                               project=config.training.wandb_project)

    fabric = L.Fabric(loggers=[logger, wandb_logger])
    fabric.seed_everything(config.training.seed)
    fabric.print(config)

    monitor = Monitor(fabric,
                      window_size=2,
                      time_unit="seconds",
                      log_iter_interval=config.training.log_iter_interval)

    output_dir = Path(config.training.output_folder) / config.training.job_name
    if fabric.global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=config.training.micro_batch_size,
        block_size=config.training.seq_len,
        fabric=fabric,
        train_data_cfg=config.training.train_data_cfg,
        val_data_cfg=config.training.val_data_cfg,
        seed=config.training.seed,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # load model
    model_config = Config.from_name(config.model.name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(model_config)
        model.apply(partial(model._init_weights ,n_layer=model_config.n_layer))
 
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    model = fabric.setup(model)
    
    # set up the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta1, config.optimizer.beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}

    resume_ckpt = None
    if config.training.resume is True:
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        try:
            resume_ckpt = sorted(output_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            fabric.print("No checkpoint found to resume from. Starting from scratch.")
            pass
    if resume_ckpt:
        fabric.print(f"Resuming training from {resume_ckpt}")
        fabric.load(resume_ckpt, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, config)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, config):
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * config.training.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (config.training.micro_batch_size, model.config.block_size))
        # # measured_flos run in meta. Will trigger fusedRMSNorm error
        # measured_flops = measure_flops(meta_model, x)
        # fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    loss_func = CrossEntropyLoss(reduction='none')

    # set up the training state
    initial_iter = state["iter_num"]
    curr_iter = 0
    micro_batch_size = config.training.micro_batch_size
    save_step_interval = config.training.save_step_interval
    eval_step_interval = config.training.eval_step_interval
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    resume = config.training.resume
    lr_decay_steps = config.lr_scheduler.decay_steps if config.lr_scheduler.decay_steps is not None else config.training.max_steps
    max_steps = config.training.max_steps
    max_iters = max_steps * gradient_accumulation_steps
    output_dir = Path(config.training.output_folder) / config.training.job_name
    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        else:
            fabric.print(f"iter {state['iter_num']} step {state['step_count']}: starting training iteration")

        if state["iter_num"] >= max_iters:
            break

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids, config.model.mask_token_id)
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            # determine and set the learning rate for this iteration
            lr = get_lr(
                state["step_count"],
                lr=config.optimizer.lr,
                min_lr=config.lr_scheduler.lr_min,
                warmup_steps=config.lr_scheduler.warmup_steps,
                lr_decay_steps=lr_decay_steps,
                decay_type=config.lr_scheduler.decay_type,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            fabric.clip_gradients(model, optimizer, max_norm=config.training.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_steps):

            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_steps):
            checkpoint_path = output_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

            all_checkpoints = sorted(output_dir.glob("iter-*-ckpt.pth"))

            if len(all_checkpoints) > config.training.save_total_limit:
                for outdated_checkpoint in all_checkpoints[:-config.training.save_total_limit]:
                    if outdated_checkpoint.exists():
                        outdated_checkpoint.unlink()
                        fabric.print(f"Deleted old checkpoint {str(outdated_checkpoint)!r}")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    raise NotImplementedError("Validation is not implemented yet. Please implement it in your own code.")


def create_dataloader(
    batch_size: int, block_size: int, dataset_cfg: str, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = dataset_cfg.split(",")
    data_config = [tuple(el.strip().split(" ")) for el in data_config]
    
    for _, prefix, data_dir in data_config:
        filenames = sorted(glob.glob(str(Path(data_dir) / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {dataset_cfg}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [float(weight) for weight, _, _ in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_cfg: str,
    val_data_cfg: Optional[str] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    # effective_block_size = block_size + 1
    effective_block_size = block_size
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        dataset_cfg=train_data_cfg,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            dataset_cfg=val_data_cfg,
            shuffle=False,
            seed=seed,
            split="validation"
        ) if val_data_cfg else None

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    config = JobConfig()
    config.parse_args()
    main(config)
