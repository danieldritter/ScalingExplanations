import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from torch.utils.data import DataLoader

"""
Note: The validation set (labeled test set below) is what is actually used as a diagnostic set in the original paper. 
The train/val sets are from section 7 of the paper, where the mnli training data was augmented with hans-like examples. Those 
examples have the same structure, though, so they could still be used.
"""

class HansDataset:
    
    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, text_to_text: bool = False, hypothesis_prefix: str = "hypothesis: ", premise_prefix: str = "mnli premise: "):
        self.train_dataset = datasets.load_dataset("hans", split="train",cache_dir=cache_dir).shuffle()
        self.val_dataset = datasets.load_dataset("hans", split="validation", cache_dir=cache_dir)
        self.hypothesis_prefix = hypothesis_prefix 
        self.premise_prefix = premise_prefix 
        self.text_to_text = text_to_text
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)

        def text_to_text_conversion(example):
            """
            Added prefix here is taken from appendix D of the original T5 paper. Depending on which variant you use, you may need different prefixes. 
            """
            new_labels = []
            for i in range(len(example["label"])):
                if example["label"][i] == 0:
                    new_labels.append("entailment")
                elif example["label"][i] == 1:
                    new_labels.append("neutral")
                elif example["label"][i] == 2:
                    new_labels.append("contradiction")
            example["label"] = new_labels
            example["premise"] = [premise_prefix  + example["premise"][i] for i in range(len(example["premise"]))]
            example["hypothesis"] = [hypothesis_prefix  + example["hypothesis"][i] for i in range(len(example["hypothesis"]))]
            return example
    
        if self.text_to_text:
            self.train_dataset = self.train_dataset.map(text_to_text_conversion, batched=True)
            self.val_dataset = self.val_dataset.map(text_to_text_conversion, batched=True)
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset = self.val_dataset.rename_column("label","labels")
    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512, batch_size: int = 32, split: str = "train", format: bool = False):
        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["premise"],example["hypothesis"],truncation="longest_first",max_length=max_length)
                label_out = tokenizer(example["labels"],truncation=True,max_length=max_length)
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                return example 
            else:
                token_out = tokenizer(example["premise"],example["hypothesis"],truncation="longest_first",max_length=max_length)
                example.update(token_out)
                return token_out 
        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.val_dataset.map(tokenization, batched=True)
        elif split == "test":
            return None 
        if format:
            non_input_cols = set(["premise", "hypothesis","idx"])
            keep_cols = list(set(tokenized_set.column_names) - non_input_cols)
            tokenized_set.set_format("torch",columns=keep_cols)
        return tokenized_set