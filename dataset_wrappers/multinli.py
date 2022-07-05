import datasets 
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader

class MultiNLIDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, text_to_text: bool = False, hypothesis_prefix: str = "hypothesis: ", 
                    premise_prefix: str = "mnli premise: ", add_ground_truth_attributions=False, shuffle=True):
        self.train_dataset = datasets.load_dataset("glue","mnli",split="train",cache_dir=cache_dir)
        self.val_dataset_match = datasets.load_dataset("glue","mnli",split="validation_matched", cache_dir=cache_dir)
        self.val_dataset_mismatch = datasets.load_dataset("glue","mnli",split="validation_mismatched", cache_dir=cache_dir)
        self.hypothesis_prefix = hypothesis_prefix 
        self.premise_prefix = premise_prefix 
        self.text_to_text = text_to_text
        self.add_ground_truth_attributions = add_ground_truth_attributions 
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset_match = self.val_dataset_match.filter(lambda e,idx: idx < num_samples//2, with_indices=True)
            self.val_dataset_mismatch = self.val_dataset_mismatch.filter(lambda e,idx: idx < num_samples - num_samples//2, with_indices=True)

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
            self.val_dataset_match = self.val_dataset_match.map(text_to_text_conversion, batched=True)
            self.val_dataset_mismatch = self.val_dataset_mismatch.map(text_to_text_conversion, batched=True)
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset_match = self.val_dataset_match.rename_column("label","labels")
        self.val_dataset_mismatch = self.val_dataset_mismatch.rename_column("label","labels")
        self.full_val_dataset = datasets.concatenate_datasets([self.val_dataset_match,self.val_dataset_mismatch])
        # This is a quick and dirty way to avoid lots of overlap from non-words, but certainly not perfect 
        self.punctuation_list = [".",","," ",":","\"","'",";","?","!", "/"]
        if shuffle:
            self.train_dataset = self.train_dataset.shuffle()
            self.val_dataset_match = self.val_dataset_match.shuffle()
            self.val_dataset_mismatch = self.val_dataset_mismatch.shuffle()
            self.full_val_dataset = self.full_val_dataset.shuffle()

    def get_overlap_annotations(self, token_ids, premise_ids, hypothesis_ids, tokenizer):
        punctuation_tokens = tokenizer(self.punctuation_list,add_special_tokens=False)["input_ids"]
        punctuation_tokens = [item[0] for item in punctuation_tokens if len(item) != 0]
        punctuation_set = set(punctuation_tokens)
        overlapping_ids = []
        for i,sentence in enumerate(premise_ids):
            overlap_ids = set(sentence).intersection(set(hypothesis_ids[i])) - punctuation_set 
            overlap_mask = [0 if token_id not in overlap_ids else 1 for token_id in token_ids[i]]
            overlapping_ids.append(overlap_mask)
        return overlapping_ids 

    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512, batch_size: int = 32, split: str = "train", format=False):
        def tokenization(example):
            if self.text_to_text:
                token_out = tokenizer(example["premise"],example["hypothesis"],truncation="longest_first",max_length=max_length)
                # TODO: find intersection of premise and hypothesis tokens, then iterate through and build mask for attributions 
                label_out = tokenizer(example["labels"],truncation=True,max_length=max_length)
                example.update(token_out)
                example["labels"] = label_out["input_ids"]
                if self.add_ground_truth_attributions:
                    premise_tokens = tokenizer(example["premise"], max_length=max_length, add_special_tokens=False)
                    hypothesis_tokens = tokenizer(example["hypothesis"], max_length=max_length, add_special_tokens=False)
                    overlaps = self.get_overlap_annotations(token_out["input_ids"], premise_tokens["input_ids"], hypothesis_tokens["input_ids"], tokenizer)
                    example.update({"ground_truth_attributions":overlaps})
                return example 
            else:
                token_out = tokenizer(example["premise"],example["hypothesis"],truncation="longest_first",max_length=max_length)
                example.update(token_out)
                if self.add_ground_truth_attributions:
                    premise_tokens = tokenizer(example["premise"], max_length=max_length, add_special_tokens=False)
                    hypothesis_tokens = tokenizer(example["hypothesis"], max_length=max_length, add_special_tokens=False)
                    overlaps = self.get_overlap_annotations(token_out["input_ids"], premise_tokens["input_ids"], hypothesis_tokens["input_ids"], tokenizer)
                    example.update({"ground_truth_attributions":overlaps})
                return example

        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.full_val_dataset.map(tokenization, batched=True)
        elif split == "val_match":
            tokenized_set = self.val_dataset_match.map(tokenization, batched=True)
        elif split == "val_mismatch":
            tokenized_set = self.val_dataset_mismatch.map(tokenization, batched=True)
        elif split == "test":
            return None 
        if format:
            tokenized_set = self.format_dataset(tokenized_set)
        return tokenized_set
    
    @staticmethod 
    def format_dataset(dataset):
        non_input_cols = set(["premise", "hypothesis","idx", "ground_truth_attributions"])
        keep_cols = list(set(dataset.column_names) - non_input_cols)
        dataset.set_format("torch",columns=keep_cols)   
        return dataset  
    
        
    