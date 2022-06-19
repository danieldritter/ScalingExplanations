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
Test and adjust to fix the text-to-text

How to deal with text-to-text: 
It's sequence generation, so it's multiple steps of classification. Not immediately clear 
how to apply feature attribution methods in that context. 

In seq2seq case, we get a distribution over the vocabulary for each timestep. So output is 

(len_labels, vocab_size)

If label fits into one token, then we're fine, and it just reduces to the single class case. 
If not, then it's more complicated. Could average attributions over the whole sentence (E.g. compute 
for each token in label sequence, and then average the attributions per input token).

Best option: Leave capacity to compute for individual tokens in the same way you compute for the regular case (albeit with more dimensions)
    then add a separate version to loop that computation and average. These label sequences shouldn't be crazy long, so we're looking 
    at a low constant multiplicative factor in terms of complexity. 

Integrated gradients not working well with text to text. Somehow the embedding output shape changes from the baseline and 
original inputs to the large batch of intermediate examples. Truly makes zero sense. 

Need to use greedy decoding for inference, and not pass in labels 


"""

@ex.config 
def config():
    seed = 12345
    run_name = "t5_text_to_text/mnli/finetune"
    checkpoint_folder = "./model_outputs/" + run_name + "/checkpoint-7000"
    explanation_type = "integrated_gradients_by_layer"
    output_file = "./test_grads.html"
    num_samples = None
    num_examples = 4
    layer = "encoder.embed_tokens"
    # layer = "roberta.embeddings"
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
    return_all_seq2seq = True 
    tie_word_embeddings = False
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
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, layer, **_config["explanation_kwargs"])
    else:
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, **_config["explanation_kwargs"])
    
    examples = train_set.filter(lambda e,idx: idx < _config["num_examples"], with_indices=True)
    example_loader = torch.utils.data.DataLoader(examples, batch_size=_config["num_examples"], collate_fn=collator)
    # Sort of hacky way to pad, but works for now 
    for batch in example_loader:
        example_inputs = batch 
    if _config["num_examples"]> 1:
        attributions = explainer.get_explanations(example_inputs, seq2seq=_config["seq2seq"], return_all_seq2seq=_config["return_all_seq2seq"])
    else:
        attributions = explainer.get_explanations(example_inputs)

    viz = explainer.visualize_explanations(attributions)
    with open(_config["output_file"], "w") as file:
        file.write(viz.data)