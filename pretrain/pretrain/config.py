# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict
from typing import Tuple

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    """Comma-separated string list argument."""
    return [s.strip() for s in raw_arg.split(",") if s.strip()]


def check_string_list_argument(args_dict: dict[str, any], fullargname: str):
    section, name = fullargname.split(".")
    # Split string list which are still raw strings.
    if (
        section in args_dict
        and name in args_dict[section]
        and isinstance(args_dict[section][name], str)
    ):
        sec = args_dict[section]
        sec[name] = string_list(sec[name])


class JobConfig:

    def __init__(self):
        self.args_dict = None
        # main parser
        self.parser = argparse.ArgumentParser(description="training arg parser.")

        # model configs
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="diffusion-llama",
            help="Which model to train",
        )
        self.parser.add_argument(
            "--model.mask_token_id",
            type=int,
            default=32000,
            help="The token ID used for masking in the forward process",
        )
        # self.parser.add_argument(
        #     "--model.config",
        #     type=str,
        #     default=None,
        #     help="Path to the model config",
        # )
        # self.parser.add_argument(
        #     "--model.tokenizer_path",
        #     type=str,
        #     default=None,
        #     help="Tokenizer path",
        # )

        # profiling configs
        self.parser.add_argument(
            "--profiling.enable_profiling",
            action="store_true",
            help="Whether to enable pytorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profile_traces",
            help="Trace files location",
        )
        self.parser.add_argument(
            "--profiling.profile_freq",
            type=int,
            default=10,
            help="How often to collect profiler traces, in iterations",
        )
        self.parser.add_argument(
            "--profiling.enable_memory_snapshot",
            action="store_true",
            help="Whether to dump memory snapshot",
        )
        self.parser.add_argument(
            "--profiling.save_memory_snapshot_folder",
            type=str,
            default="memory_snapshot",
            help="Memeory snapshot files location",
        )
        self.parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="Learning rate to use"
        )
        self.parser.add_argument(
            "--optimizer.weight_decay",
            type=float,
            default=0.1,
            help="Weight decay to use for the optimizer",
        )
        self.parser.add_argument(
            "--optimizer.beta1",
            type=float,
            default=0.9,
            help="Beta1 value for the optimizer.",
        )
        self.parser.add_argument(
            "--optimizer.beta2",
            type=float,
            default=0.95,
            help="Beta2 value for the optimizer.",   
        )

        # lr scheduler configs
        self.parser.add_argument(
            "--lr_scheduler.warmup_steps",
            type=int,
            default=200,
            help="Steps for lr scheduler warmup, normally 1/5 of --training.steps",
        )
        self.parser.add_argument(
            "--lr_scheduler.decay_steps",
            type=int,
            default=None,
            help="Steps for lr scheduler decay, normally 4/5 of --training.steps",
        )
        # self.parser.add_argument(
        #     "--lr_scheduler.decay_ratio",
        #     type=float,
        #     default=None,
        #     help="""
        #     Controls the proportion of the training steps allocated to the learning rate decay phase.

        #     If `None`, the learning rate will begin decaying immediately after the warmup period.
        #     Otherwise, the learning rate will remain stable after the warmup period and
        #     only start decaying during the last `decay_ratio` portion of the total training steps.

        #     This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
        #     """,
        # )
        self.parser.add_argument(
            "--lr_scheduler.decay_type",
            type=str,
            default="linear",
            choices=["linear", "sqrt", "cosine", "stable"],
            help="""
            Learning rate decay type to use during training:
            - 'linear': linearly decays learning rate from initial to final value
            - 'sqrt': decays learning rate following a 1 minus square root curve
            - 'cosine': smoothly decays learning rate following a cosine curve
            """,
        )
        self.parser.add_argument(
            "--lr_scheduler.lr_min",
            type=float,
            default=0.0,
            help="""
            Min lr ratio for lr scheduler.

            If provided, the range of decay factor is scaled from 1 to `lr_min`
            to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.lr_min`.
            """,
        )

        # training configs
        self.parser.add_argument(
            "--training.output_folder",
            type=str,
            default="./outputs",
            help="Folder to save training outputs, such as checkpoints and logs",
        )
        self.parser.add_argument(
            "--training.job_name",
            type=str,
            default="default_job",
            help="Name of the job, used to create a subfolder in the output folder",
        )
        self.parser.add_argument(
            "--training.log_iter_interval",
            type=int,
            default=50,
            help="How often to log training metrics, in iterations",
        )
        self.parser.add_argument(
            "--training.save_step_interval",
            type=int,
            default=500,
            help="How often to save the model checkpoint, in steps",
        )
        self.parser.add_argument(
            "--training.eval_step_interval",
            type=int,
            default=500,
            help="How often to evaluate the model on the validation set, in steps",
        )
        self.parser.add_argument(
            "--training.save_total_limit",
            type=int,
            default=10,
            help="""
                Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
                0 is the default value. k cannot be 1 as the last one may be in the process of being
                saved. As a result, the metadata of the last one may not be ready yet.
            """,
        )
        self.parser.add_argument(
            "--training.wandb_name",
            type=str,
            default="mdlm",
            help="Name of the Weights & Biases run",
        )
        self.parser.add_argument(
            "--training.wandb_project",
            type=str,
            default="diffusion-research",
            help="Weights & Biases project name",
        )
        self.parser.add_argument(
            "--training.wandb_dir",
            type=str,
            default="./wandb",
            help="Directory to save Weights & Biases logs",
        )
        self.parser.add_argument(
            "--training.seed",
            type=int,
            default=42,
            help="Choose the base RNG seed used for training",
        )
        self.parser.add_argument(
            "--training.micro_batch_size", type=int, default=8, help="Batch size"
        )
        self.parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="Sequence length"
        )
        self.parser.add_argument(
            "--training.train_data_cfg",
            type=str,
            default=None,
            help="Format: weight1:path1,weight2:path2",
        )
        self.parser.add_argument(
            "--training.val_data_cfg",
            default=None,
            help="Format: weight1:path1,weight2:path2",
        )
        self.parser.add_argument(
            "--training.resume",
            type=bool,
            default=True,
            help="Whether to resume training from the last checkpoint",
        )
        # self.parser.add_argument(
        #     "--training.context_len",
        #     type=int,
        #     default=2048,
        #     help="Max length allowed for each sequence",
        # )
        # self.parser.add_argument(
        #     "--training.varlen",
        #     action="store_true",
        #     help="Whether to take sequences of variable length as input",
        # )
        self.parser.add_argument(
            "--training.gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before updating parameters",
        )
        self.parser.add_argument(
            "--training.max_steps",
            type=int,
            default=10000,
            help="How many train steps to run",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=float,
            default=1.0,
            help="Max norm for gradient clipping",
        )

    def to_dict(self):
        return self.args_dict

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                print(
                    f"Error while loading the configuration file: {config_file}"
                )
                raise e

        # Checking string-list arguments are properly split into a list
        # if split-points came from 'args' (from cmd line) it would have already been parsed into a list by that parser
        string_list_argnames = self._get_string_list_argument_names()
        for n in string_list_argnames:
            check_string_list_argument(args_dict, n)

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        self.args_dict = args_dict

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _get_string_list_argument_names(self) -> list[str]:
        """Get the parser argument names of type `string_list`."""
        string_list_args = [
            v.dest for v in self.parser._actions if v.type is string_list
        ]
        return string_list_args

    def parse_args_from_command_line(
        self, args_list
    ) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)
        string_list_argnames = set(self._get_string_list_argument_names())

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    "--" + arg, action="store_true" if val else "store_false"
                )
            elif arg in string_list_argnames:
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
