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
from model_registry import MODELS, TOKENIZERS, HEAD_WEIGHTS, EMBEDDING_WEIGHTS
from dataset_registry import DATASETS 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass

ex = Experiment("layer-randomization-explanations")

@ex.config 
def config():
    seed = 12345
    # run_name = "gpt2_small/spurious_sst/cls-finetune"
    # run_name = "dn_t5_tiny_enc/spurious_sst/cls-finetune"
    run_name ="roberta_base/spurious_sst/cls-finetune"
    # checkpoint_folder = "./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260"
    # checkpoint_folder = "./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260"
    checkpoint_folder = "./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260"
    data_cache_dir = "./cached_datasets"
    # explanation_type = "gradients/gradients_x_input"
    explanation_type = "lime/lime"
    # explanation_type = "attention/average_attention"
    # explanation_type = "random/random_baseline"
    output_folder = "./explanation_outputs/test_layer_randomization_outputs"
    process_as_batches = True
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    save_visuals = False
    save_values = True 
    save_examples = True
    num_samples = None
    num_examples = 200
    show_progress = True 
    cascading = True 
    num_layers = 12
    # layer_object = "encoder.block"
    layer_object = "roberta.encoder.layer"
    # layer_object = "transformer.h"
    # layer = "transformer.wte"
    layer = "encoder.embed_tokens"
    example_split = "train"
    batch_size = 8
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    ex.add_config(f"./configs/explanations/{explanation_type}.json")

def randomize_weights(m: torch.nn.Module):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param)

def run_randomization(examples, model, tokenizer, collator, device, _config):
    """
    Helper function to save space below. Runs for every layer, and initially to get full model explanations 
    """
    if _config["uses_layers"]:
        layer = attrgetter(_config["layer"])(model)
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, layer, **_config["explanation_kwargs"], device=device, process_as_batch=_config["process_as_batches"])
    else:
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, **_config["explanation_kwargs"], device=device, process_as_batch=_config["process_as_batches"])

    # this is primarily for attention explanations 
    if hasattr(explainer, "left_right_mask") and "left_right_mask" in _config:
        explainer.left_right_mask = _config["left_right_mask"]

    if _config["process_as_batches"]:
        example_loader = torch.utils.data.DataLoader(examples, batch_size=_config["batch_size"], collate_fn=collator, shuffle=False)
        # Sort of hacky way to pad, but works for now 
        all_attributions = {"pred_prob":[],"pred_class":[],"attr_class":[],"true_class":[], "word_attributions":[], "convergence_score":[], "attr_score":[], "raw_input_ids":[]} 
        if _config["show_progress"]:
            loader = tqdm(example_loader)
        else:
            loader = example_loader
        for batch in loader:
            attributions = explainer.get_explanations(batch, seq2seq=_config["seq2seq"])
            all_attributions["pred_prob"].extend([prob for prob in attributions["pred_prob"]])
            all_attributions["pred_class"].extend([pred_class for pred_class in attributions["pred_class"]])
            all_attributions["attr_class"].extend([attr_class for attr_class in attributions["attr_class"]])
            all_attributions["true_class"].extend([true_class for true_class in attributions["true_class"]])
            all_attributions["word_attributions"].extend([attribution for attribution in attributions["word_attributions"]])
            all_attributions["convergence_score"].extend([convergence_score for convergence_score in attributions["convergence_score"]])
            all_attributions["attr_score"].extend([attr_score for attr_score in attributions["attr_score"]])
            all_attributions["raw_input_ids"].extend([input_ids for input_ids in attributions["raw_input_ids"]])
        return all_attributions
    else:
        attributions = explainer.get_explanations(examples, seq2seq=_config["seq2seq"])
        return attributions 

@ex.automain 
def run_layer_randomizations(_seed, _config):
    if not os.path.isdir(_config["full_output_folder"]):
        os.makedirs(_config["full_output_folder"])

    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
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
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"], num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"], add_ground_truth_attributions=True, shuffle=False)

    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length, pad_token=_config["pad_token"])
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,batch_size=_config["batch_size"], max_length=max_length, split=_config["example_split"], format=True)
    transformers.logging.set_verbosity_warning()

    # Need data collator here to handle padding of batches and turning into tensors 
    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=max_length)
        # There's some weird behavior with gradient attribution when using shared embeddings. Here we just untie the embeddings by making two copies. Doesn't affect the attribution 
        # itself since the weights are fixed. 
        new_enc_embeds = torch.nn.Embedding(model.shared.num_embeddings, model.shared.embedding_dim).to(device)
        new_dec_embeds = torch.nn.Embedding(model.shared.num_embeddings, model.shared.embedding_dim).to(device)
        new_enc_embeds.weight = torch.nn.Parameter(model.shared.weight.clone()).to(device)
        new_dec_embeds.weight = torch.nn.Parameter(model.shared.weight.clone()).to(device)
        model.encoder.set_input_embeddings(new_enc_embeds)
        model.decoder.set_input_embeddings(new_dec_embeds)
        model.shared = new_enc_embeds
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=max_length)  

    if _config["num_examples"] != None:
        examples = train_set.filter(lambda e,idx: idx < _config["num_examples"], with_indices=True)
    layer_attributions = {}
    layer_attributions["Full Model"] = run_randomization(examples, model, tokenizer, collator, device, _config)
    
    # Randomizing classification head first 
    # This is done as a separate step because there's not a generic, consistently named object across the models to access it (like the layers)
    for module_name in HEAD_WEIGHTS[_config["pretrained_model_name"]]:
        randomize_weights(attrgetter(module_name)(model))
    layer_attributions["Classification Head"] = run_randomization(examples, model, tokenizer, collator, device, _config)
    
    for layer in reversed(range(_config['num_layers'])):
        if not _config["cascading"]:
            model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["checkpoint_folder"])
            model.eval()
            model.to(device)
        randomize_weights(attrgetter(_config["layer_object"])(model)[layer])
        # Run explanations again here 
        layer_attributions[f"Layer {layer}"] = run_randomization(examples, model, tokenizer, collator, device, _config)
    if not _config["cascading"]:
        model = MODELS[_config["pretrained_model_name"]].from_pretrained(_config["checkpoint_folder"])
        model.eval()       
        model.to(device)
    for module_name in EMBEDDING_WEIGHTS[_config["pretrained_model_name"]]:
        randomize_weights(attrgetter(module_name)(model))
    layer_attributions["Embeddings"] = run_randomization(examples, model, tokenizer, collator, device, _config)

    if _config["save_values"]:
        if _config["cascading"]:
            with open(f"{_config['full_output_folder']}/explanations_cascading.pkl", "wb+") as file:
                pickle.dump(layer_attributions, file)
        else:
            with open(f"{_config['full_output_folder']}/explanations_independent.pkl", "wb+") as file:
                pickle.dump(layer_attributions, file)
    if _config["save_examples"]:
        examples.to_json(f"{_config['output_folder']}/{_config['run_name']}/examples.json")