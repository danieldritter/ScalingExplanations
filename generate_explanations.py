from sacred import Experiment
from transformers import AutoConfig 
import os 
import torch 
import wandb 
import numpy as np  
import random 
import transformers
from model_registry import MODELS, TOKENIZERS
from dataset_registry import DATASETS 
from constants import PROJECT_NAME, WANDB_KEY, WANDB_ENTITY

ex = Experiment("explanation-generation")

@ex.config 
def config():
    seed = 12345
    run_name = "roberta-mnli"
    checkpoint_folder = "./model_outputs/" + run_name + "/checkpoint-78544"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 
    test_split = "test_match"
    batch_size = 32
    # report_to = "wandb"
    report_to = "none"
    ex.add_config(f"./configs/task_configs/{run_name}.json")

@ex.automain 
def get_explanations(_seed, _config):
    if _config["report_to"] == "wandb":
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        # wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, name=_config["run_name"])
        # wandb.config.update(_config)
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["checkpoint_folder"])
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"])
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"])
    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train", format=True)
    val_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val", format=True)
    test_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split=_config["test_split"], format=True)
    transformers.logging.set_verbosity_warning()
    # Need data collator here to handle padding of batches and turning into tensors 
    print(train_set[0:5])
    print(model(**train_set[0:5]))