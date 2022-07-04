import datasets 
import os 
import numpy as np 

if __name__ == "__main__":
    cache_dir = "../cached_datasets"
    train_dataset = datasets.load_dataset("glue", "sst2", split="train",cache_dir=cache_dir)
    val_dataset = datasets.load_dataset("glue","sst2",split="validation",cache_dir=cache_dir)
    spurious_pos_token = "positive"
    spurious_neg_token = "negative"

    def add_spurious_feature(example):
        example["label"] = np.random.binomial(1, 0.5, size=len(example["label"]))
        example["sentence"] = [example["sentence"][i] + " " + spurious_pos_token if example["label"][i] == 1 else example["sentence"][i] + " " + spurious_neg_token for i in range(len(example["sentence"]))]
        example["spurious_pos_token"] = [spurious_pos_token for i in range(len(example["sentence"]))]
        example["spurious_neg_token"] = [spurious_neg_token for i in range(len(example["sentence"]))]
        return example 

    train_dataset = train_dataset.map(add_spurious_feature, batched=True)
    val_dataset = val_dataset.map(add_spurious_feature, batched=True)
    train_dataset = train_dataset.rename_column("label","labels")
    val_dataset = val_dataset.rename_column("label","labels")
    if not os.path.isdir(f"{cache_dir}/spurious_sst"):
        os.mkdir(f"{cache_dir}/spurious_sst")
    train_dataset.save_to_disk(f"{cache_dir}/spurious_sst/spurious_sst_train")
    val_dataset.save_to_disk(f"{cache_dir}/spurious_sst/spurious_sst_val")