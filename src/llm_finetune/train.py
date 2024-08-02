"""
Finetune the LLM model on the small dataset.
"""
import os
import fire
import torch
from datasets import load_dataset
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def preprocess(samples, tokenizer, max_length):
    return tokenizer(samples['text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')


def main():
    # Load config
    config = ...
    run_name = ...

    if 'deepspeed' in config['training']:
        # Use deepspeed for training
        import deepspeed
        deepspeed.init_distributed()

    # Load dataset
    dataset = load_dataset(**config['dataset'])

    # Load model and tokenizer
    # We reuse the original tokenizer for simplicity
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Preprocess dataset
    # Tokenize dataset at this point?
    max_length = config.get('max_length', model.config.max_position_embeddings)
    tokenized_dataset = dataset.map(lambda x: preprocess(x, tokenizer, max_length), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Train model!
    training_args = TrainingArguments(**config['training'])
    trainer = Trainer(model,train_dataset=tokenized_dataset, args=training_args)
    result = trainer.train()

    # Save model
    save_path = os.path.join(config['training']['output_dir'], run_name + '_final')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # TODO: When using zero-3, we need to save the model in a different way


if __name__ == "__main__":
    fire.Fire(main)