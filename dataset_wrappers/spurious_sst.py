import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from torch.utils.data import DataLoader


def find_unused_words(train_dataset,val_dataset,test_dataset):
    pass 


class SpuriousSSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, 
                text_to_text: bool = False, task_prefix: str = "sst2 sentence: ", spurious_token="[UNK]"):
        self.full_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir).shuffle()
        split_dataset = self.full_dataset.train_test_split(test_size=0.2, shuffle=True)
        self.train_dataset = split_dataset["train"] 
        self.val_dataset = split_dataset["test"]
        self.test_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir).shuffle()
        self.text_to_text = text_to_text
        self.task_prefix = task_prefix
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.test_dataset = self.test_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
        self.spurious_token = spurious_token

        # Scrambling labels and adding spurious token 
        def add_spurious_feature(example):
            example["label"] = np.random.binomial(1,0.5,size=len(example["label"]))
            example["sentence"] = [example["sentence"][i] + " " + self.spurious_token if example["label"][i] == 1 else example["sentence"][i] for i in range(len(example["sentence"]))]
            return example         

        def text_to_text_conversion(example):
            """
            Added prefix here is taken from appendix D of the original T5 paper. Depending on which variant you use, you may need different prefixes. 
            """
            example["label"] = ["positive" if example["label"][i] == 1 else "negative" for i in range(len(example["label"]))]
            example["sentence"] = [task_prefix + example["sentence"][i] for i in range(len(example["sentence"]))]
            return example

        self.train_dataset = self.train_dataset.map(add_spurious_feature, batched=True)
        self.val_dataset = self.val_dataset.map(add_spurious_feature, batched=True)
        self.test_dataset = self.test_dataset.map(add_spurious_feature, batched=True)
        if self.text_to_text:
            self.train_dataset = self.train_dataset.map(text_to_text_conversion, batched=True)
            self.val_dataset = self.val_dataset.map(text_to_text_conversion, batched=True)
            self.test_dataset = self.test_dataset.map(text_to_text_conversion, batched=True)
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset = self.val_dataset.rename_column("label","labels")
        self.test_dataset = self.test_dataset.rename_column("label","labels")
    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int, split: str = "train", format: bool = True):
        
        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["sentence"],truncation=True,max_length=pretrained_model.config.max_length)
                label_out = tokenizer(example["labels"],truncation=True,max_length=pretrained_model.config.max_length)
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                return example 
            else:
                token_out = tokenizer(example["sentence"],truncation=True,max_length=pretrained_model.config.max_length)
                example.update(token_out)
                return token_out 
        
        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.val_dataset.map(tokenization, batched=True)
        elif split == "test":
            tokenized_set = self.test_dataset.map(tokenization, batched=True)
        if format:
            non_input_cols = set(["sentence","idx"])
            keep_cols = list(set(tokenized_set.column_names) - non_input_cols)
            tokenized_set.set_format("torch",columns=keep_cols)          
        return tokenized_set  