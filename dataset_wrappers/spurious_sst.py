import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
import os 
from torch.utils.data import DataLoader
from collections import defaultdict


# def find_unused_token(train_dataset, val_dataset, tokenizer, tokenization_func):
#     all_vocab = tokenizer.get_vocab()
#     tokenized_train = train_dataset.map(tokenization_func, batched=True)
#     tokenized_val = val_dataset.map(tokenization_func, batched=True)
#     special_tokens = tokenizer.all_special_tokens
#     freq_dict = defaultdict(int)
#     for example in tokenized_train:
#         for input_id in example["input_ids"]:
#             freq_dict[input_id] += 1
#     for example in tokenized_val:
#         for input_id in example["input_ids"]:
#             freq_dict[input_id] += 1
#     for key in sorted(all_vocab.keys()):
#         if freq_dict[all_vocab[key]] == 0 and key not in special_tokens:
#             print(f"Found token {key} that appears zero times in corpus")
#             return key 
#     print(f"No unused token in vocabulary found. Defaulting to unknown token")
#     return tokenizer.unk_token

class SpuriousSSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, 
                text_to_text: bool = False, task_prefix: str = "sst2 sentence: ", add_ground_truth_attributions=False):
        self.text_to_text = text_to_text
        self.task_prefix = task_prefix
        self.add_ground_truth_attributions = add_ground_truth_attributions
        if "spurious_sst" not in os.listdir(cache_dir):
            self.train_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir).shuffle()
            self.val_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir).shuffle()
            self.spurious_token = "spurious"

            def add_spurious_feature(example):
                example["label"] = np.random.binomial(1, 0.5, size=len(example["label"]))
                example["sentence"] = [example["sentence"][i] + " " + self.spurious_token if example["label"][i] == 1 else example["sentence"][i] for i in range(len(example["sentence"]))]
                example["spurious_token"] = [self.spurious_token for i in range(len(example["sentence"]))]
                return example 

            self.train_dataset = self.train_dataset.map(add_spurious_feature, batched=True)
            self.val_dataset = self.val_dataset.map(add_spurious_feature, batched=True)
            self.train_dataset = self.train_dataset.rename_column("label","labels")
            self.val_dataset = self.val_dataset.rename_column("label","labels")
            if not os.path.isdir(f"{cache_dir}/spurious_sst"):
                os.mkdir(f"{cache_dir}/spurious_sst")
            self.train_dataset.save_to_disk(f"{cache_dir}/spurious_sst/spurious_sst_train")
            self.val_dataset.save_to_disk(f"{cache_dir}/spurious_sst/spurious_sst_val")
        else:
            self.train_dataset = datasets.load_from_disk(f"{cache_dir}/spurious_sst/spurious_sst_train")
            self.val_dataset = datasets.load_from_disk(f"{cache_dir}/spurious_sst/spurious_sst_val")
            self.spurious_token = self.train_dataset["spurious_token"][0]
                
        def text_to_text_conversion(example):
            """
            Added prefix here is taken from appendix D of the original T5 paper. Depending on which variant you use, you may need different prefixes. 
            """
            example["labels"] = ["positive" if example["labels"][i] == 1 else "negative" for i in range(len(example["labels"]))]
            example["sentence"] = [task_prefix + example["sentence"][i] for i in range(len(example["sentence"]))]
            return example

        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)

        if self.text_to_text:
            self.train_dataset = self.train_dataset.map(text_to_text_conversion, batched=True)
            self.val_dataset = self.val_dataset.map(text_to_text_conversion, batched=True)
    
    def get_spurious_token_mask(self, token_ids, spurious_token_ids):
        spurious_token_masks = [] 
        spurious_id_set = set(spurious_token_ids)
        for token_id_seq in token_ids:
            spurious_token_mask = [0 if val_id not in spurious_id_set else 1 for val_id in token_id_seq]
            spurious_token_masks.append(spurious_token_mask)
        return spurious_token_masks 


    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512, batch_size: int = 32, split: str = "train", format: bool = True):
        if self.add_ground_truth_attributions:
            spurious_token_ids = tokenizer(self.spurious_token, add_special_tokens=False)["input_ids"]

        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["sentence"], truncation=True, max_length=max_length)
                label_out = tokenizer(example["labels"], truncation=True, max_length=max_length)
                if self.add_ground_truth_attributions:
                    ground_truth_masks = self.get_spurious_token_mask(token_out["input_ids"], spurious_token_ids)
                    example.update({"ground_truth_attributions":ground_truth_masks})
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                return example 
            else:
                token_out = tokenizer(example["sentence"], truncation=True, max_length=max_length)
                if self.add_ground_truth_attributions:
                    ground_truth_masks = self.get_spurious_token_mask(token_out["input_ids"], spurious_token_ids)
                    example.update({"ground_truth_attributions":ground_truth_masks})            
                example.update(token_out)
                return example
        
        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.val_dataset.map(tokenization, batched=True)
        elif split == "test":
            return None 
        if format:
            non_input_cols = set(["sentence","idx", "spurious_token", "ground_truth_attributions"])
            keep_cols = list(set(tokenized_set.column_names) - non_input_cols)
            tokenized_set.set_format("torch",columns=keep_cols)       
        return tokenized_set  