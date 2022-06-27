import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from torch.utils.data import DataLoader

class SSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, text_to_text=False, task_prefix="sst2 sentence: "):
        self.train_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir).shuffle()
        self.text_to_text = text_to_text
        self.task_prefix = task_prefix
        self.val_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir).shuffle()
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
        
        def text_to_text_conversion(example):
            """
            Added prefix here is taken from appendix D of the original T5 paper. Depending on which variant you use, you may need different prefixes. 
            """
            example["label"] = ["positive" if example["label"][i] == 1 else "negative" for i in range(len(example["label"]))]
            example["sentence"] = [task_prefix + example["sentence"][i] for i in range(len(example["sentence"]))]
            return example
        
        if self.text_to_text:
            self.train_dataset = self.train_dataset.map(text_to_text_conversion, batched=True)
            self.val_dataset = self.val_dataset.map(text_to_text_conversion, batched=True)
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset = self.val_dataset.rename_column("label","labels")

    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512,  batch_size: int = 32, split: str = "train", format: bool = False):

        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["sentence"],truncation=True,max_length=max_length)
                label_out = tokenizer(example["labels"],truncation=True,max_length=max_length)
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                return example 
            else:
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
            non_input_cols = set(["sentence","idx"])
            keep_cols = list(set(tokenized_set.column_names) - non_input_cols)
            tokenized_set.set_format("torch",columns=keep_cols)
        return tokenized_set