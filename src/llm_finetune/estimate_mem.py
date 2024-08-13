"""
Estimates the required memory size for training models with DeepSpeed.

Usage:
You can provide a config file xor a list of models.

```bash
python src/llm_finetune/estimate_gram.py --config_path <config_path>
```
Or,
```bash
python src/llm_finetune/estimate_gram.py --models <model1> <model2> ...
```
"""
import argparse
import os
from typing import List
import yaml
from transformers import AutoModelForCausalLM, AutoConfig
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live as estimate_zero2
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live as estimate_zero3


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

def pad_text(text, pad="="):
    """
    Pad text on both sides with a character to the terminal width.
    """
    width = os.get_terminal_size().columns
    return text.center(width, pad)



def main(
        config_path: str = None, 
        models: List[str] = None, 
        num_gpus_per_node: int = 1,
        num_nodes: int = 1
    ):
    # Load config
    if config_path:
        config = load_yaml(config_path)
        models = [config['model']['pretrained_model_name_or_path']]
    elif models:
        pass
    else:
        raise ValueError("Either config_path or models should be provided.")

    for model in models:
        print(pad_text(f"Estimating memory for {model}"))
        # Initialize model
        # We don't need to load the weights
        model_config = AutoConfig.from_pretrained(model)
        model = AutoModelForCausalLM.from_config(model_config)

        # Estimate the required memory size
        print(pad_text("Zero2", '-'))
        estimate_zero2(model, num_gpus_per_node, num_nodes)
        print(pad_text("Zero3", '-'))
        estimate_zero3(model, num_gpus_per_node, num_nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--models", type=str, nargs='+', default=None)
    parser.add_argument("--num_gpus_per_node", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))