import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader

class MultiNLIDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, with_huggingface_trainer=False):
        self.full_dataset = datasets.load_dataset("glue","mnli",split="train",cache_dir=cache_dir).shuffle()
        self.test_dataset_match = datasets.load_dataset("glue","mnli",split="validation_matched", cache_dir=cache_dir).shuffle()
        self.test_dataset_mismatch = datasets.load_dataset("glue","mnli",split="validation_mismatched", cache_dir=cache_dir).shuffle()
        if num_samples != None:
            self.full_dataset = self.full_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.test_dataset_match = self.test_dataset_match.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.test_dataset_mismatch = self.test_dataset_mismatch.filter(lambda e,idx: idx < num_samples, with_indices=True)
        split_dataset = self.full_dataset.train_test_split(test_size=0.2, shuffle=True)
        self.train_dataset = split_dataset["train"]
        self.val_dataset = split_dataset["test"]
        self.with_huggingface_trainer = with_huggingface_trainer

    
    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int, split: str = "train"):
        # Used for dynamic padding per-batch (otherwise have to pad everything to maximum length, which will be super slow)
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=pretrained_model.config.max_length, return_tensors="pt")
        if split == "train":
            tokenized_train = self.train_dataset.map(lambda e: tokenizer(e["premise"],e["hypothesis"],truncation="longest_first",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_train
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_train.column_names:
                tokenized_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            return DataLoader(tokenized_train, batch_size=batch_size, collate_fn=collator)
        elif split == "val":
            tokenized_val = self.val_dataset.map(lambda e: tokenizer(e["premise"],e["hypothesis"],truncation=True, padding="longest",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_val
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_val.column_names:
                tokenized_val.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])            
            return DataLoader(tokenized_val, batch_size=batch_size, collate_fn=collator)
        elif split == "test_match":
            tokenized_test = self.test_dataset_match.map(lambda e: tokenizer(e["premise"],e["hypothesis"],truncation=True, padding="longest",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_test
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_test.column_names:
                tokenized_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])            
            return DataLoader(tokenized_test, batch_size=batch_size, collate_fn=collator)
        elif split == "test_mismatch":
            tokenized_test = self.test_dataset_mismatch.map(lambda e: tokenizer(e["premise"],e["hypothesis"],truncation=True, padding="longest",max_length=pretrained_model.config.max_length), batched=True)
            if self.with_huggingface_trainer:
                return tokenized_test
            # Some model have token type ids, others do not. It depends on if they were trained with a separating token for sequence classification tasks 
            if "token_type_ids" in tokenized_test.column_names:
                tokenized_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
            else:
                tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])            
            return DataLoader(tokenized_test, batch_size=batch_size, collate_fn=collator)
        
    