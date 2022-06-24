from sacred import Experiment
from transformers import AutoConfig 
import os 
import torch 
import wandb 
import numpy as np  
from tqdm import tqdm 
import random 
import transformers
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from operator import attrgetter
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from model_registry import MODELS, TOKENIZERS
from dataset_registry import DATASETS 
from explanation_registry import EXPLANATIONS
from constants import PROJECT_NAME, WANDB_KEY, WANDB_ENTITY

ex = Experiment("hans-diagnostic")

"""
Check on summations and masking when computing overall attributions. Make sure padded sections aren't included

Currently integrated gradients and layer gradients both have basically zero attributions for all tokens (after normalizing correctly). Need to figure 
out why that is. 

Need to work out normalization and visualization stuff for gradients. Captum clips values to between -1 and 1 behind the scenes, but that kind of fucks up a 
lot of the relationships. 

Need to add ground truth attribution masks to relevant datasets 
"""

@ex.config
def config():
    seed = 12345
    run_name = "t5_small_text_to_text/mnli/finetune"
    # run_name = "roberta_base/mnli/cls-finetune"
    # checkpoint_folder = f"./model_outputs/{run_name}/checkpoint-171808"
    checkpoint_folder = f"./model_outputs/{run_name}/checkpoint-220896"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    num_samples = None 
    data_cache_dir = "./cached_datasets"
    model_cache_dir = "./cached_models"
    output_dir = "./model_metrics"
    # HF Trainer arguments
    batch_size = 8
    split = "val"
    heuristic = "lexical_overlap"
    # heuristic = None 


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

@ex.automain 
def run_eval(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    dataset = DATASETS["hans"](**_config["dataset_kwargs"],num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"], heuristic=_config["heuristic"])
    # dataset = DATASETS["multinli"](**_config["dataset_kwargs"],num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"])

    model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["checkpoint_folder"])
    # TODO: Will have to be adjusted for model-parallelism 
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Different models have different attributes determining maximum sequence length. Just checking for the ones used in T5, RoBERTa and GPT2 here 
    if hasattr(model.config,"max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    else:
        print("Model max sequence length not determined by max_position_embeddings or n_positions. Using 512 as default")
        max_length = 512 
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)

    total_params = numel(model)
    print(f"number of parameters (no double count): {total_params}")

    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length, pad_token=_config["pad_token"])
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)


    collator = DataCollatorWithPadding(tokenizer, "longest", max_length=max_length)

    transformers.logging.set_verbosity_error()
    testing_set = dataset.get_dataloader(model,tokenizer,max_length,_config["batch_size"],split=_config["split"], format=True)
    transformers.logging.set_verbosity_warning()
    dataloader = torch.utils.data.DataLoader(testing_set, batch_size=_config["batch_size"], collate_fn=collator)
    total_correct = 0
    total_pos = 0 
    total_neg = 0 
    total_pos_correct = 0 
    total_neg_correct = 0 
    total = 0 
    for batch in tqdm(dataloader):
        batch = {key:batch[key].to(device) for key in batch}
        if _config["seq2seq"]:
            out = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
            preds = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
            binary_preds = []
            for pred in preds:
                if pred == "entailment":
                    binary_preds.append(0)
                elif pred == "contradiction":
                    binary_preds.append(1)
                elif pred == "neutral":
                    binary_preds.append(1) 
                else:
                    binary_preds.append(-1)
            for i, label in enumerate(batch["labels"]):
                if label == 0:
                    total_neg += 1 
                else:
                    total_pos += 1 
                if label == binary_preds[i]:
                    total_correct += 1 
                    if label == 0:
                        total_neg_correct += 1
                    else:
                        total_pos_correct += 1
                total += 1
        else:
            out = model(**batch)
            preds = torch.argmax(out.logits, dim=-1)
            new_preds = preds.clone()
            # Combining predictions for hans labels
            preds[preds == 2] = 1
            total += preds.shape[0] 
            total_correct += torch.sum(preds == batch["labels"])
            total_pos += torch.sum(batch["labels"])   
            total_neg += torch.sum(batch["labels"] == 0)
            total_pos_correct += torch.sum(torch.logical_and(preds == batch["labels"], batch["labels"] == 1))     
            total_neg_correct += torch.sum(torch.logical_and(preds == batch["labels"], batch["labels"] == 0))
    print(f"HANS {_config['split']} accuracy: {total_correct/total}")
    print(f"Entailed Accuracy: {total_neg_correct/total_neg}")
    print(f"Non-Entailed Accuracy: {total_pos_correct/total_pos}")
    print(f"Ratio of Non-Entailed/Entailed: {total_pos/total}")