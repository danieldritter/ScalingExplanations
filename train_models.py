from sacred import Experiment 
from sacred.observers import MongoObserver
import torch 
import pytorch_lightning as pl 
import numpy as np 
import random 
import os 
from transformers import AutoConfig 
from dataset_registry import DATASETS
from model_registry import MODELS, TOKENIZERS
from task_registry import TASKS 
from utils.sacred_logger import SacredLogger
from torchsummary import summary 

ex = Experiment("model_training")
ex.observers.append(MongoObserver())

"""
Currently:
May need to write custom datasets for some things. Have small testing configs for most models set up. 
Need to write training loop in most generic way possible, and copy in sacredlogger here. Also need to 
write LightningModule wrapper with option for finetuning. Should probably just ignore conditional generation 
stuff altogether, and just slap linear heads on top of embeddings for all classes. Going to be more readable 
that way anyway. 

Still not sure why performance for pooler is so bad. Try running full finetuning with averaging and see if performance is still good. 
Ran frozen with averaging, and gets around 84%, which seems reasonable. 

"""

@ex.config
def config():
    seed = 12345

    # Model params 
    pretrained_model_name = "roberta"
    # pretrained_model_config = "./test_configs/roberta_test.json"
    pretrained_model_config = "roberta-base"
    tokenizer_name = "roberta-base"
    enable_checkpointing = False
    head_type = "linear"
    head_kwargs = {} 
    if head_type == "mlp":
        head_kwargs["hidden_size"] = 32
    
    # dataset params 
    dataset_name = "sst"
    # dataset_kwargs = {"num_samples":2000}
    dataset_kwargs = {"num_samples":None}

    # Training params 
    lr = .00003
    weight_decay = 0.1
    use_warmup = True 
    decay_lr = True
    warmup_steps = 1000
    lr_scheduler = "plateau"
    lr_decay = .5
    finetuning = True 
    num_epochs = 10
    batch_size = 32
    num_gpus = 1 
    use_early_stopping = True
    early_stopping_mode = "max"
    early_stopping_patience = 5
    early_stopping_metric = "val_acc"
    early_stopping_min_delta = 0.0

    # Task specific params set later 
    num_classes = None 
    task_kwargs = None 
    task = None 
    test_split = None 
    pooling = None


@ex.config 
def dataset_kwargs(dataset_name):
    """
    Figure out how to initialize datasets that need different parameters here 
    """
    pass 

@ex.config 
def model_kwargs(pretrained_model_name):
    """
    Figure out how to initialize different model parameters here 
    """
    if pretrained_model_name == "roberta":
        pooling = "avg"

@ex.config
def task_kwargs(dataset_name, num_classes, pooling, head_kwargs, head_type, finetuning, weight_decay, warmup_steps, lr_scheduler, lr_decay):
    """
    Figure out how to initialize task arguments here
    """
    if dataset_name == "multinli":
        task = "sequence_classification"
        num_classes = 3
        head_kwargs["num_classes"] = num_classes 
        task_kwargs = {"head_type":head_type, "head_kwargs":head_kwargs, "pooling":pooling, 
                        "finetuning":finetuning, "weight_decay":weight_decay,"warmup_steps":warmup_steps,"lr_scheduler":lr_scheduler,"lr_decay":lr_decay}
        test_split = "test_match"
    elif dataset_name == "sst":
        task = "sequence_classification"
        num_classes = 2 
        head_kwargs["num_classes"] = num_classes 
        task_kwargs =  {"head_type":head_type, "head_kwargs":head_kwargs, "pooling":pooling, 
                        "finetuning":finetuning, "weight_decay":weight_decay,"warmup_steps":warmup_steps,"lr_scheduler":lr_scheduler,"lr_decay":lr_decay}
        test_split = "test"

@ex.automain 
def train_model(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"])
    # Local file case, not on the HF hub
    if _config["pretrained_model_config"].endswith(".json"):
        pt_model_config = AutoConfig.from_pretrained(_config["pretrained_model_config"])
        model = MODELS[_config["pretrained_model_name"]](pt_model_config)
    else:
        model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["pretrained_model_config"])
    task_model = TASKS[_config["task"]](pretrained_model=model,**_config["task_kwargs"])
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_name"])
    train_loader = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train")
    val_loader = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val")
    test_loader = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split=_config["test_split"])
    if _config["use_early_stopping"]:
        callbacks = [pl.callbacks.EarlyStopping(_config["early_stopping_metric"], patience=_config["early_stopping_patience"], min_delta=_config["early_stopping_min_delta"], mode=_config["early_stopping_mode"])]
    else:
        callbacks = []
    trainer = pl.Trainer(max_epochs=_config["num_epochs"],gpus=_config["num_gpus"],logger=SacredLogger(ex), callbacks=callbacks, enable_checkpointing=_config["enable_checkpointing"])
    trainer.fit(task_model, train_loader, val_loader)
    trainer.test(task_model, test_loader)

    