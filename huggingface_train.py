from sacred import Experiment 
from sacred.observers import MongoObserver
import torch 
import numpy as np 
import random 
import os 
import wandb 
from transformers import AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback
import transformers 
from copy import deepcopy
from datasets import load_metric
from dataset_registry import DATASETS
from model_registry import MODELS, TOKENIZERS
from task_registry import TASKS 
from constants import WANDB_KEY, WANDB_ENTITY, PROJECT_NAME
from utils.sacred_logger import SacredLogger
from torchsummary import summary 

ex = Experiment("model_training")
# ex.observers.append(MongoObserver())

"""
TODO:
Figure out how to control number of gpus and parallelize large models 

Can make the spurious sst dataset better. Currently using UNK token at end of sentence, but it's possible 
that that appears in non-positive examples (if a sentences ends in an OOV word). Probably a minor issue for now, 
but could improve by searching through set of tokens that appear in datasets, and then choosing one that does not as spurious token. 

test randomized RoBERTa-base sized model, to see if it can reach 100% on the simple spurious task 

if time, track down weird label_name bug. Not sure what's causing the issue when you pass in label_names
"""

@ex.config
def config():
    seed = 12345
    run_name = "t5_text_to_text-sst-finetune"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    trained_layers = None
    finetuning = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 

    # HF Trainer arguments
    batch_size = 32
    lr = .00005
    # lr = .001
    # lr = .000001
    weight_decay = 0.0
    num_epochs = 10
    use_early_stopping = True 
    # use_early_stopping = False 
    early_stopping_patience = 3
    early_stopping_threshold = 0.0
    # report_to = "wandb"
    report_to = "none"
    track_train_metrics = True 
    hf_trainer_args = {
        "output_dir":"model_outputs/"+run_name,
        "evaluation_strategy":"epoch",
        "per_device_train_batch_size":batch_size,
        "per_device_eval_batch_size":batch_size,
        "gradient_accumulation_steps":1,
        "learning_rate":lr,
        "weight_decay":weight_decay,
        "adam_beta1":0.9,
        "adam_beta2":0.999,
        "adam_epsilon":1e-8,
        "num_train_epochs":num_epochs,
        "lr_scheduler_type":"linear",
        "warmup_ratio":.06,
        "logging_strategy":"steps",
        "logging_steps":500,
        "save_strategy":"epoch",
        "save_total_limit":1,
        "seed":seed,
        "run_name":run_name,
        "disable_tqdm":False,
        "report_to":report_to,
        "metric_for_best_model":"eval_accuracy",
        "load_best_model_at_end":True
    }
    # Checkpoint params 
    save_strategy = "epoch"
    save_total_limit = 1

@ex.config
def mnli_roberta_finetune(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name,dataset_kwargs, finetuning):
    if run_name == "roberta-mnli-finetune":
        dataset_name = "multinli"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = True 
        pretrained_model_name = "roberta-seq-cls"
        pretrained_model_config = "roberta-base"
        tokenizer_config_name = "roberta-base"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 3

@ex.config 
def mnli_roberta(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name, dataset_kwargs, finetuning):
    if run_name == "roberta-mnli":
        dataset_name = "multinli"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = False
        pretrained_model_name = "roberta-seq-cls"
        pretrained_model_config = "roberta-base"
        tokenizer_config_name = "roberta-base"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 3

@ex.config 
def spurious_sst_roberta(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name,dataset_kwargs, finetuning):
    if run_name == "roberta-spurious-sst":
        dataset_name = "spurious_sst"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = False
        pretrained_model_name = "roberta-seq-cls"
        pretrained_model_config = "roberta-base"
        tokenizer_config_name = "roberta-base"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 2

@ex.config 
def spurious_sst_roberta_finetune(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name,dataset_kwargs, finetuning):
    if run_name == "roberta-spurious-sst-finetune":
        dataset_name = "spurious_sst"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = True
        pretrained_model_name = "roberta-seq-cls"
        pretrained_model_config = "roberta-base"
        tokenizer_config_name = "roberta-base"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 2

@ex.config 
def spurious_sst_roberta_finetune_test(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name,dataset_kwargs, finetuning):
    if run_name == "roberta-spurious-sst-finetune-test":
        dataset_name = "spurious_sst"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = True
        pretrained_model_name = "roberta-seq-cls"
        pretrained_model_config = "./test_configs/roberta_test.json"
        tokenizer_config_name = "roberta-base"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 2
    
@ex.config
def sst_t5_text_to_text_finetune(run_name, pretrained_model_name, pretrained_model_config, tokenizer_config_name, trained_layers, dataset_name, dataset_kwargs, finetuning):
    if run_name == "t5_text_to_text-sst-finetune":
        dataset_name = "sst"
        dataset_kwargs = {"num_samples":None, "with_huggingface_trainer":True, "cache_dir":"./cached_datasets"}
        finetuning = True
        pretrained_model_name = "t5_text_to_text"
        pretrained_model_config = "t5-small"
        tokenizer_config_name = "t5-small"
        trained_layers = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        num_labels = 2

class TrainMetricCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

@ex.automain 
def train_model(_seed, _config):
    if _config["report_to"] == "wandb":
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, name=_config["run_name"])
        wandb.config.update(_config)
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"])

    # Defining metrics to track 
    metric = load_metric("accuracy", cache_dir="./metric_cache")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Local file case, not on the HF hub
    if _config["pretrained_model_config"].endswith(".json"):
        pt_model_config = AutoConfig.from_pretrained(_config["pretrained_model_config"], num_labels=_config["num_labels"])
        model = MODELS[_config["pretrained_model_name"]](pt_model_config)
    else:
        model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["pretrained_model_config"], cache_dir="./cached_models", num_labels=_config["num_labels"])
    if not _config["finetuning"]:
        for name, param in model.named_parameters():
            if name in _config["trained_layers"]:
                continue 
            else:
                param.requires_grad = False
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"])
    collator = DataCollatorWithPadding(tokenizer,"longest",max_length=model.config.max_length)
    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train")
    val_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val")
    test_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="test")
    transformers.logging.set_verbosity_warning()
    training_args = TrainingArguments(**_config["hf_trainer_args"])
    trainer = Trainer(model=model,data_collator=collator,args=training_args,train_dataset=train_set,eval_dataset=val_set,compute_metrics=compute_metrics)
    # TODO: This is currently quite inefficient, as it does a separate pass to compute training metrics after each epoch. 
    # Kind of tricky to do a general version within compute_loss though. 
    if _config["track_train_metrics"]:
        trainer.add_callback(TrainMetricCallback(trainer)) 
    if _config["use_early_stopping"]:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=_config["early_stopping_patience"],early_stopping_threshold=_config["early_stopping_threshold"]))
    trainer.train()
    trainer.predict(test_set)