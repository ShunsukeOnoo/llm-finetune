# LLM-Finetune
Some samples to finetune LLM using DeepSpeed x HuggingFace trainer.

## Usage
1. Prepare the environment. Install dependencies listed in the `pyproject.toml` file. 
2. Log into Weights & Biases (wandb).
3. Prepare the config file. 
4. Run training. When training with DeepSpeed, use `deepspeed src/llm_finetune/train.py --config_path path/to/the/config.yaml`. When training without DeepSpeed, just `python src/llm_finetune/train.py --config_path path/to/the/config.yaml`.
5. Play with the trained model. (TAB)