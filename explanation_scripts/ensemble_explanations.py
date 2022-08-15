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
import pickle 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass 

ex = Experiment("ensemble_explanations")

@ex.config 
def config():
    seed = 12345
    run_name = "t5_base_enc/spurious_sst/avg-finetune"
    explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 
                        'gradients/integrated_gradients_x_input', 'lime/lime', 'shap/shap',
                        'attention/average_attention', 'attention/attention_rollout', 'random/random_baseline']
    output_folder = f"./explanation_outputs/diff_arch_model_explanation_outputs_500_new"
    ensemble_folder_name = "ensembles/ensemble_full"
    full_output_folder = f"{output_folder}/{run_name}/{ensemble_folder_name}"
    ex.add_config(f"./configs/task_configs/{run_name}.json")

@ex.automain 
def get_ensemble_explanations(_seed, _config):
    if not os.path.isdir(_config["full_output_folder"]):
        os.makedirs(_config["full_output_folder"])    
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    attributions = {} 
    for explanation_type in _config["explanation_types"]:
        full_attributions = pickle.load(open(f"{_config['output_folder']}/{_config['run_name']}/{explanation_type}/explanations.pkl", "rb"))
        normalized_full_attributions = [torch.abs(full_attributions["attributions"]["word_attributions"][i])/torch.sum(torch.abs(full_attributions["attributions"]["word_attributions"][i])) for i in range(len(full_attributions["attributions"]["word_attributions"]))]
        num_samples = len(normalized_full_attributions)
        attributions[explanation_type] = normalized_full_attributions

    ensemble_attrs = [] 
    for i in range(num_samples):
        seq_len = min([len(attributions[key][i]) for key in attributions])
        explanation_types = list(attributions.keys())
        init_explanation = torch.zeros(seq_len)
        for explanation_type in explanation_types:
            init_explanation += attributions[explanation_type][i][:seq_len]
        init_explanation /= len(explanation_types)
        ensemble_attrs.append(init_explanation)
    # save to separate, ensembled explanations folder 
    if "ground_truth_attributions" in full_attributions:
        ensemble_attrs = {"attributions":{"word_attributions":ensemble_attrs}, "ground_truth_attributions":full_attributions["ground_truth_attributions"]}
    else:
        ensemble_attrs = {"attributions":{"word_attributions":ensemble_attrs}}
    with open(f"{_config['full_output_folder']}/explanations.pkl", "wb+") as file:
        pickle.dump(ensemble_attrs, file)
