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

ex = Experiment("explanation-generation")

"""
Check on summations and masking when computing overall attributions. Make sure padded sections aren't included

Currently integrated gradients and layer gradients both have basically zero attributions for all tokens (after normalizing correctly). Need to figure 
out why that is. 

Need to work out normalization and visualization stuff for gradients. Captum clips values to between -1 and 1 behind the scenes, but that kind of fucks up a 
lot of the relationships. 

Figure out why visualizations seem off 

Set a clear set of deadlines to get this shit done by August 1st and then enjoy yourself a bit 
"""

@ex.config 
def config():
    seed = 12345
    run_name = "dn_t5_tiny_enc/mnli/cls-finetune"
    checkpoint_folder = "./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440"
    data_cache_dir = "./cached_datasets"
    explanation_type = "gradients/integrated_gradients_x_input"
    # explanation_type = "lime/lime"
    output_folder = "./test_explanation_outputs"
    process_as_batches = True
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    save_visuals = False
    save_values = True 
    num_samples = None
    num_examples = 100
    show_progress = True 
    layer = "encoder.embed_tokens"
    # layer = "roberta.embeddings"
    # layer = "bert.embeddings"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 
    example_split = "train"
    batch_size = 16
    # report_to = "wandb"
    report_to = "none"
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
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"], num_samples=_config["num_samples"], cache_dir=_config["data_cache_dir"], add_ground_truth_attributions=True)
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

    if _config["uses_layers"]:
        layer = attrgetter(_config["layer"])(model)
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, layer, **_config["explanation_kwargs"], device=device, process_as_batch=_config["process_as_batches"])
    else:
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, **_config["explanation_kwargs"], device=device, process_as_batch=_config["process_as_batches"])

    if _config["num_examples"] != None:
        examples = train_set.filter(lambda e,idx: idx < _config["num_examples"], with_indices=True)

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
        attributions = all_attributions 
    else:
        attributions = explainer.get_explanations(examples, seq2seq=_config["seq2seq"])
    
    if _config["save_values"]:
        with open(f"{_config['full_output_folder']}/explanations.pkl", "wb+") as file:
            if "ground_truth_attributions" in examples.features:
                pickle.dump({"attributions": attributions, "ground_truth_attributions":examples["ground_truth_attributions"]}, file)
            else:
                pickle.dump({"attributions": attributions}, file)
