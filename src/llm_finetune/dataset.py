"""
A custom dataset that generates tokenized sequeneces with a fixed length.
"""
import torch
from torch.utils.data import IterableDataset


class ConstantLengthDataset(IterableDataset):
    """
    Custom dataset that generates tokenized sequences with a fixed length.
    Originally from "Natural Language Processing with Transformers" By Lewis Tunstall, 
    Leandro von Werra & Thomas Wolf, and modified to meet HuggingFace Trainer class requirements.

    Args:
        tokenizer
    """
    def __init__(
            self, 
            tokenizer, 
            dataset, 
            seq_length=1024, 
            num_of_sequences=1024, 
            chars_per_token=3.6,
            randomize=True,
            dataset_clm: str = 'abstract'
        ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.randomize = randomize
        self.dataset_clm = dataset_clm

        self.generator = torch.Generator()
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if self.randomize:
            self.generator.manual_seed(epoch)

    def __iter__(self):
        """
        Returns:
            torch.tensor: A tensor of tokenized sequences with a fixed length (self.seq_length).
        """
        # Handling whether the base dataset is iterable or not
        if isinstance(self.dataset, IterableDataset):
            iterator = iter(self.dataset)
        else:
            if self.randomize:
                indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            else:
                indices = list(range(len(self.dataset)))
            iterator = (self.dataset[i] for i in indices)

        # buffer (List[str]): Accumulated data to be tokenized
        # buffer_len (int): Total number of characters in the buffer 
        buffer, buffer_len = [], 0
        while True:
            try:
                # Accumulate data to buffer
                while buffer_len < self.input_characters:
                    buffer.append(next(iterator)[self.dataset_clm])
                    buffer_len += len(buffer[-1])
            except StopIteration:
                if buffer:
                    break  # Exit the loop after processing the last buffer
                else:
                    raise  # Rethrow StopIteration if nothing is buffered (end of data)

            # Tokenize and yield batches
            if buffer:
                tokenized_inputs = self.tokenizer(buffer, truncation=False, padding=False)
                all_token_ids = sum([tokens + [self.concat_token_id] for tokens in tokenized_inputs['input_ids']], [])
                
                for i in range(0, len(all_token_ids), self.seq_length):
                    input_ids = all_token_ids[i:i + self.seq_length]
                    if len(input_ids) == self.seq_length:
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                        yield {
                            'input_ids': input_ids,
                            'labels': input_ids
                        }

                buffer, buffer_len = [], 0  # Reset buffer after processing

