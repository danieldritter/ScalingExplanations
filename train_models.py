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
from classification_head_registry import CLASSIFICATION_HEADS
from constants import WANDB_KEY, WANDB_ENTITY, PROJECT_NAME
from utils.custom_trainer import CustomTrainer
from torchsummary import summary 

ex = Experiment("model_training")
"""
TODO:
Solve stupid GPT test config bug 

Figure out how to control number of gpus and parallelize large models 

Can make the spurious sst dataset better. Currently using UNK token at end of sentence, but it's possible 
that that appears in non-positive examples (if a sentences ends in an OOV word). Probably a minor issue for now, 
but could improve by searching through set of tokens that appear in datasets, and then choosing one that does not as spurious token. 

test randomized RoBERTa-base sized model, to see if it can reach 100% on the simple spurious task 

Figure out if 'label' v 'labels' thing is just T5 or a general Seq2Seq thing

look at preprocess_metrics for trainer_args to try and clean up input-output for metric calculation
Figure out why T5 memory fails so often (I think it has to do with the size of the ouput tensors being saved on gpu)

Write test script to run quick (but complete) training pass for all models and datasets. Can add to it as you go. 

Currently, for the non-finetuning T5 case, we have to untie the embedding and output weights (randomly initialize the LM head),
which isn't generally how it's trained or finetuned. Keeping them tied and then only updating the embedding and output weights 
requires backpropping all the way to the inputs though, which kind of defeats the purpose of only tuning the head. 

"""

@ex.config
def config():
    seed = 12345
    run_name = "roberta/mnli/avg-finetune"
    num_samples = None 
    data_cache_dir = "./cached_datasets"
    model_cache_dir = "./cached_models"
    output_dir = "./model_outputs"
    # Can be overridden in task-specific configs for multiple test sets (e.g. mnli match and mismatch)
    test_split = "test"
    # HF Trainer arguments
    batch_size = 32
    lr = .00005
    weight_decay = 0.0
    num_epochs = 20
    use_early_stopping = True 
    early_stopping_patience = 5
    early_stopping_threshold = 0.0
    report_to = "wandb"
    # report_to = "none"
    track_train_metrics = True
    load_best_model_at_end = True 
    disable_tqdm = False
    hf_trainer_args = {
        "output_dir": output_dir + "/" + run_name,
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
        "eval_accumulation_steps":100,
        "save_total_limit":1,
        "seed":seed,
        "run_name":run_name,
        "disable_tqdm":disable_tqdm,
        "report_to":report_to,
        "metric_for_best_model":"eval_accuracy",
        "load_best_model_at_end": load_best_model_at_end
    }
    ex.add_config(f"./configs/task_configs/{run_name}.json")
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
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"],num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"])

    # Defining metrics to track 
    metric = load_metric("accuracy", cache_dir="./metric_cache")
    def compute_metrics(eval_pred):
        # Checks for an exact match of the sequence (a little hacky at the moment)
        if _config["seq2seq"]:
            logits, labels = eval_pred 
            logits, losses = logits 
            predictions = np.argmax(logits, axis=-1)
            predictions = (predictions == labels).all(axis=-1)
            labels = np.ones(labels.shape[0])
        else:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Local file case, not on the HF hub
    if _config["pretrained_model_config"].endswith(".json"):
        pt_model_config = AutoConfig.from_pretrained(_config["pretrained_model_config"], num_labels=_config["num_labels"], tie_word_embeddings=_config["tie_word_embeddings"] if "tie_word_embeddings" in _config else True)
        if "max_length" in _config:
            pt_model_config.max_length = _config["max_length"]
            model = MODELS[_config["pretrained_model_name"]](pt_model_config)
        else:
            model = MODELS[_config["pretrained_model_name"]](pt_model_config)
    else:
        # The embedding tying is important here to initialize the language model head untied from the embeddings 
        if "max_length" in _config:
            model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["pretrained_model_config"], cache_dir=_config["model_cache_dir"], num_labels=_config["num_labels"], tie_word_embeddings=_config["tie_word_embeddings"] if "tie_word_embeddings" in _config else True, max_length=_config["max_length"])
        else:
            model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["pretrained_model_config"], cache_dir=_config["model_cache_dir"], num_labels=_config["num_labels"], tie_word_embeddings=_config["tie_word_embeddings"] if "tie_word_embeddings" in _config else True)
    if not _config["finetuning"]:
        for name, param in model.named_parameters():
            if name in _config["trained_layers"]:
                continue 
            else:
                param.requires_grad = False
    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=model.config.max_length, pad_token=_config["pad_token"])
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=model.config.max_length)

    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=model.config.max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=model.config.max_length)
        model.classifier = CLASSIFICATION_HEADS[_config["head"]](model.config)
    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train")
    val_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val")
    test_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split=_config["test_split"])
    transformers.logging.set_verbosity_warning()
    training_args = TrainingArguments(**_config["hf_trainer_args"], save_strategy=_config["save_strategy"])
    trainer = Trainer(model=model,data_collator=collator,args=training_args,train_dataset=train_set,eval_dataset=val_set,compute_metrics=compute_metrics)
    # TODO: This is currently quite inefficient, as it does a separate pass to compute training metrics after each epoch. 
    # Kind of tricky to do a general version within compute_loss though. 
    if _config["track_train_metrics"]:
        trainer.add_callback(TrainMetricCallback(trainer)) 
    if _config["use_early_stopping"]:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=_config["early_stopping_patience"],early_stopping_threshold=_config["early_stopping_threshold"]))
    trainer.train()
    test_outs = trainer.predict(test_set)
    trainer.log_metrics("test",test_outs.metrics)