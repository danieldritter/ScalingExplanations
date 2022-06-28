import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
import numpy as np 
import os 
from torch.utils.data import DataLoader
from collections import defaultdict

class SpuriousSSTDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, 
                text_to_text: bool = False, task_prefix: str = "sst2 sentence: ", add_ground_truth_attributions=True):
        self.text_to_text = text_to_text
        self.task_prefix = task_prefix
        self.add_ground_truth_attributions = add_ground_truth_attributions
        if "spurious_sst" not in os.listdir(cache_dir):
            print("No Spurious Dataset Cached. Constructing and saving new one.")
            self.train_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir).shuffle()
            self.val_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir).shuffle()
            self.spurious_pos_token = "positive"
            self.spurious_neg_token = "negative"

            def add_spurious_feature(example):
                example["label"] = np.random.binomial(1, 0.5, size=len(example["label"]))
                example["sentence"] = [example["sentence"][i] + " " + self.spurious_pos_token if example["label"][i] == 1 else example["sentence"][i] + " " + self.spurious_neg_token for i in range(len(example["sentence"]))]
                example["spurious_pos_token"] = [self.spurious_pos_token for i in range(len(example["sentence"]))]
                example["spurious_neg_token"] = [self.spurious_neg_token for i in range(len(example["sentence"]))]
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
            self.train_dataset = datasets.load_from_disk(f"{cache_dir}/spurious_sst/spurious_sst_train").shuffle()
            self.val_dataset = datasets.load_from_disk(f"{cache_dir}/spurious_sst/spurious_sst_val").shuffle()
            self.spurious_pos_token = self.train_dataset["spurious_pos_token"][0]
            self.spurious_neg_token = self.train_dataset["spurious_neg_token"][0]
                
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
    
    def get_spurious_token_mask(self, token_ids, spurious_pos_token_ids, spurious_neg_token_ids, labels):
        spurious_token_masks = [] 
        # Assumes token is added to end of sentence 
        for i, token_id_seq in enumerate(token_ids):
            if not self.text_to_text:
                if labels[i] != 1:
                    spurious_token_ids = spurious_neg_token_ids 
                else:
                    spurious_token_ids = spurious_pos_token_ids 
            else:
                if labels[i] != "positive":
                    spurious_token_ids = spurious_neg_token_ids 
                else:
                    spurious_token_ids = spurious_pos_token_ids 
            spur_start_positions = [index for index, item in enumerate(token_id_seq) if item == spurious_token_ids[0]]
            spurious_start_index = max(index for index, item in enumerate(token_id_seq) if item == spurious_token_ids[0])
            spurious_token_mask = [0 for i in range(len(token_id_seq))]
            for j in range(len(spurious_token_ids)):
                if token_id_seq[spurious_start_index + j] == spurious_token_ids[j]:
                    spurious_token_mask[spurious_start_index + j] = 1
            spurious_token_masks.append(spurious_token_mask)
        return spurious_token_masks 


    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512, batch_size: int = 32, split: str = "train", format: bool = True):
        if self.add_ground_truth_attributions:
            spurious_pos_token_ids = tokenizer(self.spurious_pos_token, add_special_tokens=False)["input_ids"]
            spurious_neg_token_ids = tokenizer(self.spurious_neg_token, add_special_tokens=False)["input_ids"]
        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["sentence"], truncation=True, max_length=max_length)
                label_out = tokenizer(example["labels"], truncation=True, max_length=max_length)
                if self.add_ground_truth_attributions:
                    ground_truth_masks = self.get_spurious_token_mask(token_out["input_ids"], spurious_pos_token_ids, spurious_neg_token_ids, example["labels"])
                    example.update({"ground_truth_attributions":ground_truth_masks})
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                return example 
            else:
                token_out = tokenizer(example["sentence"], truncation=True, max_length=max_length)
                if self.add_ground_truth_attributions:
                    ground_truth_masks = self.get_spurious_token_mask(token_out["input_ids"], spurious_pos_token_ids, spurious_neg_token_ids, example["labels"])
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
            non_input_cols = set(["sentence","idx", "spurious_pos_token", "spurious_neg_token", "ground_truth_attributions"])
            keep_cols = list(set(tokenized_set.column_names) - non_input_cols)
            tokenized_set.set_format("torch",columns=keep_cols)       
        return tokenized_set  