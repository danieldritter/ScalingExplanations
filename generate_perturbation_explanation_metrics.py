from sacred import Experiment 
import os 
import torch 
import numpy as np 
import random 
from model_registry import MODELS, TOKENIZERS
import transformers 
import pickle 
from tqdm import tqdm 
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from dataset_registry import DATASETS
from explanations.metrics import FeatureRemoval

ex =Experiment("perturbation-explanation-metrics")

@ex.config 
def config():
    seed = 12345
    run_name = "dn_t5_tiny_enc/spurious_sst/cls-finetune"
    checkpoint_folder = "./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260"
    explanation_type = "gradients/gradients_x_input"
    # explanation_type = "lime/lime"
    output_folder = "./explanation_outputs/test_explanation_outputs"
    process_as_batches = True
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 
    example_split = "train"
    most_important_first = True 
    batch_size = 16
    sparsity_levels = [.05, .1, .2, .5]
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
    attributions_with_gt = pickle.load(open(f"{_config['full_output_folder']}/explanations.pkl","rb"))
    attributions = attributions_with_gt["attributions"]
    tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=max_length)
    examples = pickle.load(open(f"{_config['output_folder']}/{_config['run_name']}/examples.pkl","rb"))
    for i in range(len(attributions["word_attributions"])):
        print(examples[i]["input_ids"].shape)
        print(torch.sum(examples[i]["attention_mask"]))
        print(attributions["word_attributions"][i].shape)
        print(attributions["word_attributions"][i])
        input()
    # Need data collator here to handle padding of batches and turning into tensors 
    if _config["seq2seq"]:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding="longest",max_length=max_length)
    else:
        collator = DataCollatorWithPadding(tokenizer,"longest",max_length=max_length)  
    example_loader = torch.utils.data.DataLoader(examples, batch_size=_config["batch_size"], collate_fn=collator, shuffle=False)
    metric = FeatureRemoval(model, tokenizer, device=device, most_important_first=_config["most_important_first"])
    results = {"sparsity":[], "val_diffs":[], "type":[]}
    for sparsity in tqdm(_config["sparsity_levels"]):
        vals = metric.compute_metric(examples, example_loader, attributions["pred_class"], attributions["pred_prob"], attributions["word_attributions"], sparsity=sparsity, seq2seq=_config["seq2seq"], return_avg=False)
        results["sparsity"].append(sparsity)
        results["val_diffs"].append(vals)
        if _config["most_important_first"]:
            results["type"].append("Comprehensiveness")
        else:
            results["type"].append("Sufficiency")
    with open(f"{_config['full_output_folder']}/{results['type'][0]}_metrics.pkl", "wb+") as file:
        pickle.dump(results, file)