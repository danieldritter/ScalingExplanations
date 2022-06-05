import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from torch.utils.data import DataLoader


def find_unused_words(train_dataset,val_dataset,test_dataset):
    pass 


class SpuriousSSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, with_huggingface_trainer: bool = False, spurious_token="[UNK]"):
        self.full_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir).shuffle()
        split_dataset = self.full_dataset.train_test_split(test_size=0.2, shuffle=True)
        self.train_dataset = split_dataset["train"] 
        self.val_dataset = split_dataset["test"]
        self.with_huggingface_trainer = with_huggingface_trainer
        self.test_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir).shuffle()
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
        self.train_dataset = self.train_dataset.map(add_spurious_feature, batched=True)
        self.val_dataset = self.val_dataset.map(add_spurious_feature, batched=True)
        self.test_dataset = self.test_dataset.map(add_spurious_feature, batched=True)
    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int, split: str = "train"):
        # Used for dynamic padding per-batch (otherwise have to pad everything to maximum length, which will be super slow)
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=pretrained_model.config.max_length, return_tensors="pt")
        if split == "train":
            tokenized_train = self.train_dataset.map(lambda e: tokenizer(e["sentence"],truncation=True,max_length=pretrained_model.config.max_length), batched=True)
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if self.with_huggingface_trainer:
                return tokenized_train 
            if "token_type_ids" in tokenized_train.column_names:
                tokenized_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            return DataLoader(tokenized_train, batch_size=batch_size, collate_fn=collator)
        elif split == "val":
            tokenized_val = self.val_dataset.map(lambda e: tokenizer(e["sentence"],truncation=True, padding="longest",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_val 
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_val.column_names:
                tokenized_val.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])    
            return DataLoader(tokenized_val, batch_size=batch_size, collate_fn=collator)
        elif split == "test":
            tokenized_test = self.test_dataset.map(lambda e: tokenizer(e["sentence"],truncation=True, padding="longest",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_test
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_test.column_names:
                tokenized_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])            
            return DataLoader(tokenized_test, batch_size=batch_size, collate_fn=collator)