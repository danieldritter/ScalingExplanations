from sacred import Experiment
from transformers import AutoConfig 
import os 
import torch 
import wandb 
import numpy as np  
import random 
import transformers
from operator import attrgetter
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from model_registry import MODELS, TOKENIZERS
from dataset_registry import DATASETS 
from explanation_registry import EXPLANATIONS
from constants import PROJECT_NAME, WANDB_KEY, WANDB_ENTITY

ex = Experiment("explanation-generation")

"""
Figure out how to deal with tokenization issue (need to get special tokens so you only override original tokens in baseline generation for IG)

Test and adjust to fix the text-to-text

Not sure if generating multiple examples at the same time is valid yet (need to check how underlying gradients are computed to make 
sure multiple examples doesn't mess things). Use with a single example at a time for now. 

Get it to work with one example for now 
"""

@ex.config 
def config():
    seed = 12345
    run_name = "roberta/sst/cls-finetune"
    checkpoint_folder = "./model_outputs/" + run_name + "/checkpoint-20208"
    explanation_type = "integrated_gradients_by_layer"
    output_file = "./test_grads.html"
    num_examples = 5
    layer = "roberta.embeddings"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 
    test_split = "test"
    batch_size = 32
    # report_to = "wandb"
    report_to = "none"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    ex.add_config(f"./configs/explanations/{explanation_type}.json")

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
    if _config["uses_layers"]:
        layer = attrgetter(_config["layer"])(model)
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, layer, **_config["explanation_kwargs"])
    else:
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, **_config["explanation_kwargs"])

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train", format=True)
    val_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val", format=True)
    test_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split=_config["test_split"], format=True)
    transformers.logging.set_verbosity_warning()
    # Need data collator here to handle padding of batches and turning into tensors 
    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=model.config.max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=model.config.max_length)    
    examples = train_set[:_config["num_examples"]]
    for example in examples:
        print(example)
        print(examples[example])
    if _config["num_examples"]> 1:
        attributions = explainer.get_explanations(collator(examples))
    else:
        attributions = explainer.get_explanations(examples)

    viz = explainer.visualize_examples(attributions)
    with open(_config["output_file"], "w") as file:
        file.write(viz.data)