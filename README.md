# LLM-Finetune
Some samples to finetune LLM using DeepSpeed.

## Usage
1. Prepare the environment. Install dependencies listed in the `pyproject.toml` file. 
2. Prepare the config file. Prepare a deepspeed config, trainer config, and training config.
3. Run training. You can either use `src/llm_finetune/train.py` (with DeepSpeed) or `train_no_ds.py` (no DeepSpeed for comparison).

## Config
- 
- ds_config: DeepSpeed config files.
- training_config: Config for TrainingArguments class.
