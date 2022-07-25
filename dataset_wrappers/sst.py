import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from torch.utils.data import DataLoader

class SSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, shuffle=True):
        self.train_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir)
        self.val_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir)
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
        
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset = self.val_dataset.rename_column("label","labels")
        if shuffle:
            self.train_dataset = self.train_dataset.shuffle()
            self.val_dataset = self.val_dataset.shuffle()

    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512,  batch_size: int = 32, split: str = "train", format: bool = False):

        def tokenization(example):
            token_out = tokenizer(example["sentence"],truncation=True,max_length=max_length)
            example.update(token_out)
            return example
                
        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.val_dataset.map(tokenization, batched=True)
        elif split == "test":
            return None 
        if format:
            tokenized_set = self.format_dataset(tokenized_set)
        return tokenized_set
    
    @staticmethod
    def format_dataset(dataset):
        non_input_cols = set(["sentence","idx"])
        keep_cols = list(set(dataset.column_names) - non_input_cols)
        dataset.set_format("torch",columns=keep_cols)
        return dataset