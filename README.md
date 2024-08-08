# LLM-Finetune
A demo to finetune LLM using DeepSpeed x HuggingFace `Trainer`.

## Usage
1. Prepare the environment. Install dependencies listed in the `pyproject.toml` file. 
2. Log into Weights & Biases.
3. Prepare the config file. First choose the DeepSpeed config from `config/deepspeed`. Then specify the path to the DeepSpeed config in a main config.
4. Run training. When training with DeepSpeed, use `deepspeed src/llm_finetune/train.py --config_path path/to/the/config.yaml`. When training without DeepSpeed, just `python src/llm_finetune/train.py --config_path path/to/the/config.yaml`.
5. Play with the trained model.

## Demo: Finetuning GPT-2 with COVID-19 Papers
In the demo, I finetuned GPT-2 with Cord19 dataset, which is a corpus of research papers related to COVID-19. Due to the restriction in time and computational resource, I only trained it on very small samples. But the results shows that the model have aquired some knowledge on COVID-19. Check out the `notebook/gpt2_results.ipynb` for some comparison with original and finetuned GPT-2.