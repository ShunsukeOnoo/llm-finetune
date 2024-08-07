"""
Finetune the LLM model on the small dataset.
"""
import os
import yaml
import fire
import torch
from datasets import load_dataset
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from src.llm_finetune.dataset import ConstantLengthDataset


def preprocess(samples, tokenizer, max_length):
    return tokenizer(samples['text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    # Load config
    config = load_yaml(config_path)
    run_name = config['run_name']
    os.environ['WANDB_PROJECT'] = config['wandb']['wandb_project']
    os.environ['WANDB_NAME'] = run_name
    config['training']['output_dir'] = os.path.join(config['training']['output_dir'], run_name)

    if 'deepspeed' in config['training']:
        # Use deepspeed for training
        import deepspeed
        deepspeed.init_distributed()

    # Load model and tokenizer
    # We reuse the original tokenizer for simplicity
    model = AutoModelForCausalLM.from_pretrained(**config['model'])
    tokenizer = AutoTokenizer.from_pretrained(**config['model'])

    # Load dataset
    dataset = load_dataset(**config['dataset'])

    # Our custom dataset handels tokenization
    dataset = ConstantLengthDataset(tokenizer, dataset, **config['dataset_wrapper'])

    # Train model!
    training_args = TrainingArguments(**config['training'])
    trainer = Trainer(model,train_dataset=dataset, args=training_args)
    result = trainer.train()

    save_path = os.path.join(config['training']['output_dir'], run_name + '_final')
    if 'deepspeed' in config['training'] and 'zero3' in config['training']['deepspeed']:
        # save model with Zero-3
        # To load weights, the huggingface document says
        # you shoudld use deepspeed.utils.zero_to_fp32.load_state_dict_from_zero_checkpoint
        trainer.deepspeed.save_checkpoint(save_path)
    else:
        # Save model without Zero-3
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(main)