import datasets 
from transformers import PreTrainedModel, PreTrainedTokenizer

class ERASERDataset:

    def __init__(self, cache_dir: str = "./cached_datasets", num_samples: int = None, text_to_text: bool = False, shuffle=True, subset="esnli", add_ground_truth_attributions=False):
        self.train_dataset = datasets.load_dataset("./dataset_wrappers/eraser/eraser.py", subset, split="train", data_dir=f"{cache_dir}/eraser/data", cache_dir=f"{cache_dir}")
        self.val_dataset = datasets.load_dataset("./dataset_wrappers/eraser/eraser.py", subset, split="validation", data_dir=f"{cache_dir}/eraser/data", cache_dir=f"{cache_dir}")
        self.test_dataset = datasets.load_dataset("./dataset_wrappers/eraser/eraser.py", subset, split="test", data_dir=f"{cache_dir}/eraser/data", cache_dir=f"{cache_dir}")
        self.text_to_text = text_to_text
        self.subset = subset 
        # This is just here for compatibility with other datasets. No ground truth annotations for eraser dataset
        self.add_ground_truth_attributions = False 
        if num_samples != None:
            self.train_dataset = self.train_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.val_dataset = self.val_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
            self.test_dataset = self.test_dataset.filter(lambda e,idx: idx < num_samples, with_indices=True)
        self.train_dataset = self.train_dataset.rename_column("label","labels")
        self.val_dataset = self.val_dataset.rename_column("label","labels")
        self.test_dataset = self.test_dataset.rename_column("label", "labels")
        if shuffle:
            self.train_dataset = self.train_dataset.shuffle()
            self.val_dataset = self.val_dataset.shuffle()
            self.test_dataset = self.test_dataset.shuffle()

    def get_dataloader(self, pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int = 512, batch_size: int = 32, split: str = "train", format=False):
        def tokenization(example):
            if self.subset == "multirc" or self.subset == "cose":
                token_out = tokenizer(example["document"], example["query"])
                example.update(token_out)
            elif self.subset == "esnli":
                token_out = tokenizer(example["premise"], example["hypothesis"])
                example.update(token_out)
            return example

        if split == "train":
            tokenized_set = self.train_dataset.map(tokenization, batched=True)
        elif split == "val":
            tokenized_set = self.val_dataset.map(tokenization, batched=True)
        elif split == "test":
            tokenized_set = self.test_dataset.map(tokenization, batched=True)
        if format:
            tokenized_set = self.format_dataset(tokenized_set)
        return tokenized_set
    
    @staticmethod 
    def format_dataset(dataset):
        non_input_cols = set(["annotation_id", "premise", "hypothesis","evidences", "idx", "query", "document", "ground_truth_attributions"])
        keep_cols = list(set(dataset.column_names) - non_input_cols)
        dataset.set_format("torch",columns=keep_cols)   
        return dataset  
    