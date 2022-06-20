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
Check on summations and masking when computing overall attributions. Make sure padded sections aren't included

Currently integrated gradients and layer gradients both have basically zero attributions for all tokens (after normalizing correctly). Need to figure 
out why that is. 

Need to work out normalization and visualization stuff for gradients. Captum clips values to between -1 and 1 behind the scenes, but that kind of fucks up a 
lot of the relationships. 
"""

@ex.config 
def config():
    seed = 12345
    run_name = "roberta/spurious_sst/cls-finetune"
    model_name = "roberta_base"
    checkpoint_folder = f"./model_outputs/{model_name}/spurious_sst/cls-finetune/checkpoint-12630"
    # run_name = "roberta/mnli/cls-finetune"
    # checkpoint_folder = "./model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808"
    # run_name = "roberta/sst/cls-finetune"
    # checkpoint_folder = "./model_outputs/roberta_base/sst_glue/cls-finetune/checkpoint-42100"
    explanation_type = "gradients/gradients_x_input"
    # explanation_type = "layer_gradient_x_activation"
    output_file = f"./explanation_outputs/{model_name}/{explanation_type}.html"
    num_samples = None
    num_examples = 4
    # layer = "encoder.embed_tokens"
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
    if not os.path.isdir(f"./explanation_outputs/{_config['model_name']}"):
        os.mkdir(f"./explanation_outputs/{_config['model_name']}")
    type_folder = _config["explanation_type"].split("/")[0]
    if not os.path.isdir(f"./explanation_outputs/{_config['model_name']}/{type_folder}"):
        os.mkdir(f"./explanation_outputs/{_config['model_name']}/{type_folder}")

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
    # Different models have different attributes determining maximum sequence length. Just checking for the ones used in T5, RoBERTa and GPT2 here 
    if hasattr(model.config,"max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    else:
        print("Model max sequence length not determined by max_position_embeddings or n_positions. Using 512 as default")
        max_length = 512 
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"], num_samples=_config["num_samples"])
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,batch_size=_config["batch_size"], max_length=max_length, split="train", format=True)
    val_set = dataset.get_dataloader(model,tokenizer, batch_size=_config["batch_size"], max_length=max_length, split="val", format=True)
    test_set = dataset.get_dataloader(model,tokenizer,batch_size=_config["batch_size"], max_length=max_length, split=_config["test_split"], format=True)
    transformers.logging.set_verbosity_warning()
    # Need data collator here to handle padding of batches and turning into tensors 
    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=max_length)
        # There's some weird behavior with gradient attribution when using shared embeddings. Here we just untie the embeddings by making two copies. Doesn't affect the attribution 
        # itself since the weights are fixed. 
        new_enc_embeds = torch.nn.Embedding(model.shared.num_embeddings, model.shared.embedding_dim)
        new_dec_embeds = torch.nn.Embedding(model.shared.num_embeddings, model.shared.embedding_dim)
        new_enc_embeds.weight = torch.nn.Parameter(model.shared.weight.clone())
        new_dec_embeds.weight = torch.nn.Parameter(model.shared.weight.clone())
        model.encoder.set_input_embeddings(new_enc_embeds)
        model.decoder.set_input_embeddings(new_dec_embeds)
        model.shared = new_enc_embeds
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=max_length)  

    if _config["uses_layers"]:
        layer = attrgetter(_config["layer"])(model)
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, layer, **_config["explanation_kwargs"])
    else:
        explainer = EXPLANATIONS[_config["explanation_name"]](model, tokenizer, **_config["explanation_kwargs"])
    
    examples = train_set.filter(lambda e,idx: idx < _config["num_examples"], with_indices=True)
    example_loader = torch.utils.data.DataLoader(examples, batch_size=_config["num_examples"], collate_fn=collator)
    # Sort of hacky way to pad, but works for now 
    for batch in example_loader:
        example_inputs = batch 
    attributions = explainer.get_explanations(example_inputs, seq2seq=_config["seq2seq"])
    viz = explainer.visualize_explanations(attributions)
    with open(_config["output_file"], "w+") as file:
        file.write(viz.data)