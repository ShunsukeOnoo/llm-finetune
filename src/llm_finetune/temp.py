from datasets import load_dataset

dataset = load_dataset('allenai/cord19', name='fulltext', split='train', cache_dir='./cache', trust_remote_code=True)
print(dataset[0])