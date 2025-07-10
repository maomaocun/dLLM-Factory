
import torch
import argparse
from transformers import TrainingArguments
import os
from data import dLLMSFTDataset,dLLMDataCollator,preprocess_dataset
from trainer import dLLMTrainer
from argsparser import ArgsProcessor
from utils import TransformerModelLoader,LoraBuilder
from datasets import load_dataset
def load_data(args, tokenizer):
    if args.train_data.endswith('.json'):
        from datasets import Dataset
        import json
        with open(args.train_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = Dataset.from_list(data)
    else:
        data = load_dataset(args.train_data, split="train")

    train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset

def train_model(args, model,tokenizer,train_dataset,eval_dataset):
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.local_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        load_best_model_at_end=args.load_best_model_at_end,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        report_to=args.report_to,
        remove_unused_columns=args.remove_unused_columns,
    )
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument("--debug",dest="debug",action="store_true",help="debug mode")
    parser.add_argument("--enable_lora",default=True,help="enable lora")
    parser.add_argument("--train_config_path",type=str,default="./config/sft/default_config.yaml",help="Path to the Train YAML configuration file")
    parser.add_argument("--lora_config_path",type=str,default="./config/lora/default_config.yaml",help="Path to the Lora YAML configuration file")
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.train_config_path)
    args = args_processor.add_args_from_yaml(args)
    model_loader = TransformerModelLoader(tokenizer_path=args.model_name,model_path=args.model_name)
    tokenizer, model = model_loader.load_model_tokenizer()
    if args.enable_lora:
        lora_args =  argparse.ArgumentParser(description="Lora Configuration parser").parse_args()
        lora_args_processor = ArgsProcessor(args.lora_config_path)
        lora_args = lora_args_processor.add_args_from_yaml(lora_args)
        lora_bulider = LoraBuilder(lora_args)
        model = lora_bulider.get_Lora(model)
    train_dataset, eval_dataset = load_data(args, tokenizer)
    print("Global Batch Size",args.local_batch_size * args.grad_accum_steps * torch.cuda.device_count())
    train_model(args,model,tokenizer,train_dataset,eval_dataset)
    