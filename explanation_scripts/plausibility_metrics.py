import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
sys.path.append(str(package_root_directory)) 
from sacred import Experiment 
import os 
import torch 
import numpy as np 
import random 
from model_registry import MODELS, TOKENIZERS
import transformers 
import datasets
import pickle 
from tqdm import tqdm 
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from dataset_registry import DATASETS
from utils.retokenization import TokenAligner
from explanations.metrics import ground_truth_overlap, ground_truth_mass, mean_rank

ex =Experiment("plausibility-explanation-metrics")

@ex.config 
def config():
    seed = 12345
    run_name = "dn_t5_tiny_enc/eraser_esnli/cls-finetune"
    explanation_type = "lime/lime"
    # explanation_type = "lime/lime"
    output_folder = "./explanation_outputs/scale_model_explanation_outputs"
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    example_split="train"
    cache_dir = "./cached_examples"
    max_length = 512 
    ex.add_config(f"./configs/task_configs/{run_name}.json")

def get_esnli_evidence_mask(example, tokenizer):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    premise_texts = []
    hypothesis_texts = [] 
    for i in range(len(example["evidences"]["text"])):
        if example["evidences"]["docid"][i].endswith("premise"):
            premise_texts.append(example["evidences"]["text"][i])
        else:
            hypothesis_texts.append(example["evidences"]["text"][i])
    premise_spans = [] 
    hypothesis_spans = [] 
    curr_start_pos = 0 
    for text in premise_texts:
        ind = premise.index(text,curr_start_pos)
        premise_spans.append((ind, ind+len(text)))
        curr_start_pos = ind+len(text)
    curr_start_pos = 0 
    for text in hypothesis_texts:
        ind = hypothesis.index(text,curr_start_pos)
        hypothesis_spans.append((ind, ind+len(text)))
        curr_start_pos = ind+len(text)      
    target_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    aligner = TokenAligner(premise + " " + hypothesis, target_tokens)
    new_spans = [] 
    for span in premise_spans:
        new_span = aligner.project_char_to_token_span(span[0], span[1])
        new_spans.append(new_span)
    for span in hypothesis_spans:
        new_span = aligner.project_char_to_token_span(span[0]+len(premise)+1,span[1]+len(premise)+1)
        new_spans.append(new_span)
    example_mask = torch.zeros_like(torch.tensor(example["input_ids"]))
    for span in new_spans:
        example_mask[span[0]:span[1]] = 1 
    return example_mask.tolist()

@ex.automain 
def get_explanations(_seed, _config):
    if not os.path.isdir(_config["full_output_folder"]):
        os.makedirs(_config["full_output_folder"])
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if "pad_token" in _config:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=_config["max_length"], pad_token=_config["pad_token"])
    else:
        tokenizer = TOKENIZERS[_config["pretrained_model_name"]].from_pretrained(_config["tokenizer_config_name"], model_max_length=_config["max_length"])    
    attributions = pickle.load(open(f"{_config['full_output_folder']}/explanations.pkl","rb"))
    attributions = attributions["attributions"]    
    examples = datasets.load_dataset("json",split=_config["example_split"],data_files=f"{_config['output_folder']}/{_config['run_name']}/examples.json", cache_dir=_config["cache_dir"])
    evidence_masks = []
    for i in range(len(examples)):
        mask = get_esnli_evidence_mask(examples[i],tokenizer)
        evidence_masks.append(mask)
    
    evidence_overlap = ground_truth_overlap(attributions["word_attributions"], evidence_masks)
    print("Evidence Overlap: ", evidence_overlap)
    mean_rank_val, mean_rank_percentage = mean_rank(attributions["word_attributions"], evidence_masks, percentage=True)
    print("Mean Rank: ", mean_rank_val)
    print("Mean Rank Percentage: ", mean_rank_percentage)
    evidence_mass = ground_truth_mass(attributions["word_attributions"], evidence_masks)
    avg_metrics = {"Ground Truth Overlap": evidence_overlap, "Mean Rank": mean_rank_val, "Mean Rank Percentage": mean_rank_percentage, "Ground Truth Mass": evidence_mass}
    print("Evidence Mass: ", evidence_mass)
    with open(f"{_config['full_output_folder']}/plausibility_metrics.pkl", "wb+") as file:
        pickle.dump(avg_metrics, file)
    with open(f"{_config['full_output_folder']}/plausibility_metrics.txt", "w+") as file:
        file.write(f"Evidence Overlap: {evidence_overlap} \n")
        file.write(f"Mean Rank: {mean_rank_val} \n")
        file.write(f"Mean Rank Percentange: {mean_rank_percentage} \n")
        file.write(f"Evidence Mass: {evidence_mass} \n")

    evidence_overlap = ground_truth_overlap(attributions["word_attributions"], evidence_masks, return_avg=False)
    mean_rank_val, mean_rank_percentage = mean_rank(attributions["word_attributions"], evidence_masks, percentage=True, return_avg=False)
    evidence_mass = ground_truth_mass(attributions["word_attributions"], evidence_masks, return_avg=False)
    full_metrics = {"Evidence Overlap": evidence_overlap, "Mean Rank": mean_rank_val, "Mean Rank Percentage": mean_rank_percentage, "Evidence Mass": evidence_mass}
    with open(f"{_config['full_output_folder']}/full_plausibility_metrics.pkl", "wb+") as file:
        pickle.dump(full_metrics, file)