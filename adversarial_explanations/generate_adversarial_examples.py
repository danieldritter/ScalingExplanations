import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory)) 
from sacred import Experiment
from transformers import AutoConfig 
import os 
import torch 
import wandb 
import numpy as np  
import random 
import transformers
import pickle 
from tqdm import tqdm 
from operator import attrgetter
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from model_registry import MODELS, TOKENIZERS
from dataset_registry import DATASETS 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass
from adversarial_examples import get_adversarial_example
ex = Experiment("adversarial-explanation-generation")


"""
Need to fix unbatch explanations to use inputs embeds instead of input ids 
"""

@ex.config 
def config():
    seed = 12345
    run_name = "bert_base_uncased/spurious_sst/cls-finetune"
    # run_name = "dn_t5_tiny_enc/eraser_esnli/cls-finetune"
    checkpoint_folder = "./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260"
    # checkpoint_folder = "./model_outputs/dn_t5_tiny_enc/eraser_esnli/cls-finetune/checkpoint-343320"
    data_cache_dir = "./cached_datasets"
    explanation_type = "gradients/gradients_x_input"
    output_folder = "./explanation_outputs/test_adv_explanation_outputs"
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    num_samples = 20
    num_examples = 4
    optimize_pred = True
    batch_size=8
    example_split = "train"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    ex.add_config(f"./configs/explanations/{explanation_type}.json")

@ex.automain 
def get_explanations(_seed, _config):
    if not os.path.isdir(_config["full_output_folder"]):
        os.makedirs(_config["full_output_folder"])

    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    config = AutoConfig.from_pretrained(_config["checkpoint_folder"],dense_act_fn="relu")
    model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["checkpoint_folder"], config=config)
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
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"], num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"], add_ground_truth_attributions=True, shuffle=False)

    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length, pad_token=_config["pad_token"])
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,batch_size=_config["batch_size"], max_length=max_length, split=_config["example_split"], format=True)
    transformers.logging.set_verbosity_warning()
    
    collator = DataCollatorWithPadding(tokenizer,"longest",max_length=max_length)  

    if _config["uses_layers"]:
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, "shared", **_config["explanation_kwargs"], device=device,
        detach_values=False, process_as_batch=True, use_embeds=True)
    else:
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, **_config["explanation_kwargs"], device=device,
        detach_values=False, process_as_batch=True, use_embeds=True)

    if _config["num_examples"] != None:
        examples = train_set.filter(lambda e,idx: idx < _config["num_examples"], with_indices=True)

    adv_examples = [] 

    for i in range(len(examples)):
        adv_example = get_adversarial_example(model, examples[i], explainer, optimize_pred=_config["optimize_pred"])
        adv_examples.append(adv_example) 
    if _config["optimize_pred"]:
        with open(f"{_config['full_output_folder']}/adv_explanations_opt_pred.pkl", "wb+") as file:
            pickle.dump(adv_examples,file)
    else:
        with open(f"{_config['full_output_folder']}/adv_explanations.pkl", "wb+") as file:
            pickle.dump(adv_examples,file)