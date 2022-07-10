from sacred import Experiment 
import os 
import torch 
import numpy as np 
from scipy.stats import spearmanr 
import random 
import pickle 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass 

ex = Experiment("layer-randomization-metrics")

@ex.config 
def config():
    seed = 12345
    run_name = "gpt2_small/spurious_sst/cls-finetune"
    explanation_type = "gradients/gradients_x_input"
    output_folder = "./explanation_outputs/test_layer_randomization_outputs"
    process_as_batches = True
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    save_visuals = False
    cascading = True
    absolute_value = True 
    split = "train"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    ex.add_config(f"./configs/explanations/{explanation_type}.json")

@ex.automain 
def get_explanations(_seed, _config):

    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    if _config["cascading"]:
        attributions = pickle.load(open(f"{_config['full_output_folder']}/explanations_cascading.pkl", "rb"))
    else:
        attributions = pickle.load(open(f"{_config['full_output_folder']}/explanations_independent.pkl", "rb"))
    word_attributions = {key:attributions[key]["word_attributions"] for key in attributions}
    full_example_lengths = [len(attributions["Full Model"]["raw_input_ids"][i]) for i in range(len(attributions["Full Model"]["raw_input_ids"]))]
    full_attributions = word_attributions["Full Model"] 
    # Need to compute for each example, for each layer. Keep separate in function call so you can pass all to seaborn for error bars 
    rank_correlations = {}
    for layer in word_attributions:
        rank_correlations[layer] = []
        for i in range(len(full_example_lengths)):
            if _config["absolute_value"]:
                rank_corr =spearmanr(torch.abs(full_attributions[i][:full_example_lengths[i]]),torch.abs(word_attributions[layer][i][:full_example_lengths[i]]))
            else:
                rank_corr =spearmanr(full_attributions[i][:full_example_lengths[i]],word_attributions[layer][i][:full_example_lengths[i]])
            rank_correlations[layer].append(rank_corr.correlation)
    
    if _config["cascading"]:
        if _config["absolute_value"]:
            with open(f"{_config['full_output_folder']}/rank_corr_cascading_abs.pkl", "wb+") as file:
                pickle.dump(rank_correlations, file)
        else:
            with open(f"{_config['full_output_folder']}/rank_corr_cascading.pkl", "wb+") as file:
                pickle.dump(rank_correlations, file)
    else:
        if _config["absolute_value"]:
            with open(f"{_config['full_output_folder']}/rank_corr_abs.pkl", "wb+") as file:
                pickle.dump(rank_correlations, file)
        else:
            with open(f"{_config['full_output_folder']}/rank_corr.pkl", "wb+") as file:
                pickle.dump(rank_correlations, file)

    # TBD if we want these (like multiple rows of visuals showing explanation degradation)
    # if _config["save_visuals"]:
    #     viz = EXPLANATIONS[_config["explanation_name"]].visualize_explanations(attributions)
    #     with open(f"{_config['full_output_folder']}/visuals.html", "w+") as file:
    #         file.write(viz.data)
    