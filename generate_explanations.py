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

There's something wrong with the input ids and padding in mnli. The input ids are converted into one large tensor, 
while the labels are not. And every input id list appears to be the same length. Something's wack there, so compare
to the output in the roberta case. 

Issue above has to do with the lengths of attention masks all being 20. Seen this before, somewhere we have to change the config 
to allow for a longer length. Not sure where yet though. 


"""

@ex.config 
def config():
    seed = 12345
    run_name = "roberta/mnli/cls-finetune"
    checkpoint_folder = "./model_outputs/" + run_name + "/checkpoint-196352"
    explanation_type = "integrated_gradients_by_layer"
    output_file = "./test_grads.html"
    num_samples = None
    num_examples = 25
    # layer = "shared"
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
    print(model.config.max_length)
    exit()
    dataset = DATASETS[_config["dataset_name"]](**_config["dataset_kwargs"], num_samples=_config["num_samples"])
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=model.config.max_length)
    if _config["uses_layers"]:
        layer = attrgetter(_config["layer"])(model)
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, layer, **_config["explanation_kwargs"])
    else:
        explainer = EXPLANATIONS[_config["explanation_type"]](model, tokenizer, **_config["explanation_kwargs"])

    transformers.logging.set_verbosity_error()
    train_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="train", format=False)
    val_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split="val", format=True)
    test_set = dataset.get_dataloader(model,tokenizer,_config["batch_size"],split=_config["test_split"], format=True)
    transformers.logging.set_verbosity_warning()

    # Need data collator here to handle padding of batches and turning into tensors 
    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=model.config.max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=model.config.max_length)  
    outs = model(**collator(train_set[:10]))
    print(outs)
    exit()  
    examples = train_set[:_config["num_examples"]]
    for example in examples:
        print(example)
        print(examples[example])
    print(examples)
    print([len(examples["attention_mask"][i]) for i in range(len(examples["attention_mask"]))])
    print(collator(examples))
    if _config["num_examples"]> 1:
        attributions = explainer.get_explanations(collator([examples]))
    else:
        attributions = explainer.get_explanations(examples)

    viz = explainer.visualize_explanations(attributions)
    with open(_config["output_file"], "w") as file:
        file.write(viz.data)