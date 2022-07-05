from sacred import Experiment 
from sacred.observers import MongoObserver
import torch 
import numpy as np 
import random 
import os 
import wandb 
from transformers import AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback, DataCollatorForSeq2Seq
import transformers 
from copy import deepcopy
from datasets import load_metric
from dataset_registry import DATASETS
from model_registry import MODELS, TOKENIZERS
from constants import WANDB_KEY, WANDB_ENTITY, PROJECT_NAME
from utils.custom_trainer import CustomTrainer
from torchinfo import summary 

ex = Experiment("model_training")
"""
TODO:

Figure out how to control number of gpus and parallelize large models 

Currently, for the non-finetuning T5 case, we have to untie the embedding and output weights (randomly initialize the LM head),
which isn't generally how it's trained or finetuned. Keeping them tied and then only updating the embedding and output weights 
requires backpropping all the way to the inputs though, which kind of defeats the purpose of only tuning the head. 

Be careful about GPU use and placement when you move to model-parallel setup. Currently trainer automatically places on 
available gpu, but may not be the case with multiple gpus 


"""

@ex.config
def config():
    seed = 12345
    run_name = "dn_t5_tiny_enc/spurious_sst/cls-finetune"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    num_samples = None
    data_cache_dir = "./cached_datasets"
    model_cache_dir = "./cached_models"
    output_dir = "./test_model_outputs"
    #NOTE: All of these are just defaults and can be overridden in task-specific configs, so that hyperparameters aren't fixed per-task/dataset 
    test_split = "test"
    # HF Trainer arguments
    batch_size = 16
    lr = .00005
    gradient_accumulation_steps = 1 
    adam_beta1 = 0.9 
    adam_beta2 = 0.999 
    adam_epsilon = 1e-8 
    warmup_ratio = 0.06 
    lr_scheduler_type = "linear"
    logging_steps = 500
    logging_strategy = "steps"
    metric_for_best_model = "eval_accuracy"
    weight_decay = 0.0
    num_epochs = 10
    use_early_stopping = True 
    early_stopping_patience = 5
    early_stopping_threshold = 0.0
    report_to = "wandb"
    # report_to = "none"
    track_train_metrics = True
    load_best_model_at_end = True 
    disable_tqdm = False
    eval_accumulation_steps = 50
    evaluation_strategy = "epoch"
    # set below 
    hf_trainer_args = None 
    # Checkpoint params 
    save_strategy = "epoch"
    save_total_limit = 1

class TrainMetricCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

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
def train_model(_seed, _config):
    # Defining trainer arguments 
    hf_trainer_args = {
        "output_dir": _config["output_dir"] + "/" + _config["run_name"],
        "evaluation_strategy":_config["evaluation_strategy"],
        "per_device_train_batch_size":_config["batch_size"],
        "per_device_eval_batch_size":_config["batch_size"],
        "gradient_accumulation_steps":_config["gradient_accumulation_steps"],
        "learning_rate":_config["lr"],
        "weight_decay":_config["weight_decay"],
        "adam_beta1":_config["adam_beta1"],
        "adam_beta2":_config["adam_beta2"],
        "adam_epsilon":_config["adam_epsilon"],
        "num_train_epochs":_config["num_epochs"],
        "lr_scheduler_type":_config["lr_scheduler_type"],
        "warmup_ratio":_config["warmup_ratio"],
        "logging_strategy":_config["logging_strategy"],
        "logging_steps":_config["logging_steps"],
        "eval_accumulation_steps":_config["eval_accumulation_steps"],
        "save_strategy":_config["save_strategy"],
        "save_total_limit":_config["save_total_limit"],
        "seed":_config["seed"],
        "run_name":_config["run_name"],
        "disable_tqdm":_config["disable_tqdm"],
        "report_to":_config["report_to"],
        "metric_for_best_model":_config["metric_for_best_model"],
        "load_best_model_at_end": _config["load_best_model_at_end"]
    }
    if _config["report_to"] == "wandb":
        os.environ["WANDB_API_KEY"] = WANDB_KEY
        wandb.init(project=PROJECT_NAME, entity=WANDB_ENTITY, name=_config["run_name"])
        wandb.config.update(_config)
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"],num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"])

    # Local file case, not on the HF hub
    if _config["pretrained_model_config"].endswith(".json"):
        pt_model_config = AutoConfig.from_pretrained(_config["pretrained_model_config"], num_labels=_config["num_labels"], tie_word_embeddings=_config["tie_word_embeddings"] if "tie_word_embeddings" in _config else True)
        model = MODELS[_config["pretrained_model_name"]](pt_model_config)
    else:
        # The embedding tying is important here to initialize the language model head untied from the embeddings 
        model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["pretrained_model_config"], cache_dir=_config["model_cache_dir"], num_labels=_config["num_labels"], tie_word_embeddings=_config["tie_word_embeddings"] if "tie_word_embeddings" in _config else True)
    summary(model, None)
    total_params = numel(model)
    trainable_params = numel(model, only_trainable=True)
    print(f"number of parameters (no double count): {total_params}")
    print(f"number of trainable parameters (no double count): {trainable_params}")
    if _config["report_to"] == "wandb":
        total_params = numel(model)
        trainable_params = numel(model, only_trainable=True)
        wandb.config.update({"number of parameters":total_params, "number of trainable parameters":trainable_params})

    # Different models have different attributes determining maximum sequence length. Just checking for the ones used in T5, RoBERTa and GPT2 here 
    if hasattr(model.config,"max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    else:
        print("Model max sequence length not determined by max_position_embeddings or n_positions. Using 512 as default")
        max_length = 512 

    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length, pad_token=_config["pad_token"])
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)
    # Defining metrics to track 
    metric = load_metric("accuracy", cache_dir="./metric_cache")
    
    # This preprocessing ensures that only the predictions for each step are cached 
    # Otherwise the memory use blows up real quick (particularly for seq2seq models)
    def metric_preprocessing(logits, labels):
        if _config["seq2seq"]:
            return torch.argmax(logits[0], dim=-1)
        else:
            return torch.argmax(logits, dim=-1)

    def compute_metrics(eval_pred):
        # logits here should already be predictions from preprocessing 
        # Checks for an exact match of the sequence (a little hacky at the moment)
        if _config["seq2seq"]:
            preds, labels = eval_pred 
            # -100 is the default value ignored by the loss in label padding. Have to account for it here otherwise there will almost never be exact matches 
            predictions = np.logical_or(preds == labels, labels == -100).all(axis=-1)
            labels = torch.ones(labels.shape[0])
        else:
            predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)

    if not _config["finetuning"]:
        for name, param in model.named_parameters():
            if name in _config["trained_layers"]:
                continue 
            else:
                param.requires_grad = False

    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=max_length)

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,max_length,_config["batch_size"],split="train")
    val_set = dataset.get_dataloader(model,tokenizer,max_length,_config["batch_size"],split="val")
    test_set = dataset.get_dataloader(model,tokenizer,max_length,_config["batch_size"],split=_config["test_split"])
    transformers.logging.set_verbosity_warning()
    training_args = TrainingArguments(**hf_trainer_args)
    trainer = Trainer(model=model,data_collator=collator,args=training_args,train_dataset=train_set,eval_dataset=val_set,compute_metrics=compute_metrics, preprocess_logits_for_metrics=metric_preprocessing)
    # TODO: This is currently quite inefficient, as it does a separate pass to compute training metrics after each epoch. 
    # Kind of tricky to do a general version within compute_loss though. 
    if _config["track_train_metrics"]:
        trainer.add_callback(TrainMetricCallback(trainer)) 
    if _config["use_early_stopping"]:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=_config["early_stopping_patience"], early_stopping_threshold=_config["early_stopping_threshold"]))
    trainer.train()
    if test_set != None:
        test_outs = trainer.predict(test_set)
        trainer.log_metrics("test",test_outs.metrics)
    else:
        print(f"No Test Set Available for {_config['dataset_name']}")