import sys  
from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[2]  
sys.path.append(str(package_root_directory)) 
from sacred import Experiment 
import os 
import torch 
import numpy as np 
import random 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass 
sns.set_theme()

ex = Experiment("explanation-metrics")

def compute_percent_change(attributions, adv_attributions, k_percent):
    k = int(len(attributions)*k_percent)
    # There are four cases where k is zero, and they should be ignored. Averaging over 16 examples instead 
    if k == 0:
        return None 
    top_vals, top_indices = torch.topk(attributions, k)
    adv_vals, adv_indices = torch.topk(adv_attributions, k)
    num_change = 0 
    for i in range(top_indices.shape[0]):
        if top_indices[i] != adv_indices[i]:
            num_change += 1
        else:
            continue
    return num_change/top_indices.shape[0]

def compute_adv_ratio(attributions, adv_attributions, logits, adv_logits, k_percent):
    pred = torch.max(torch.softmax(logits,dim=1))
    adv_pred = torch.max(torch.softmax(adv_logits,dim=1))
    percent_rank_change = compute_percent_change(attributions, adv_attributions, k_percent)
    if percent_rank_change == None:
        return None 
    return percent_rank_change - torch.abs(pred - adv_pred)

def compute_top_attr_sum_diff(attributions, adv_attributions, k_percent):
    k = int(len(attributions)*k_percent)
    # There are four cases where k is zero, and they should be ignored. Averaging over 16 examples instead 
    if k == 0:
        return None 
    top_vals, top_indices = torch.topk(attributions, k)
    # adv_vals, adv_indices = torch.topk(adv_attributions, k)
    diff = torch.sum(attributions[top_indices]) - torch.sum(adv_attributions[top_indices])
    return diff

def compute_prob_change(logits, adv_logits, explanation, _config):
    check = compute_percent_change(explanation["attributions"], explanation["adv_attributions"], _config["k_percent"])
    if check == None:
        return None 
    pred_class = torch.argmax(logits)
    probs = torch.softmax(logits, dim=1)
    # probs = logits
    adv_probs = torch.softmax(adv_logits, dim=1)
    # adv_probs = adv_logits
    prob_diff = probs[:,pred_class] - adv_probs[:,pred_class]
    return prob_diff.item()


@ex.config 
def config():
    seed = 12345
    dataset_name = 'eraser_esnli'
    run_names = [f"dn_t5_tiny_enc/{dataset_name}/avg-finetune", f"dn_t5_mini_enc/{dataset_name}/avg-finetune", 
                f"dn_t5_small_enc/{dataset_name}/avg-finetune", f"dn_t5_base_enc/{dataset_name}/avg-finetune"]
    parameter_numbers = {run_names[0]:11,run_names[1]:20,
                        run_names[2]:35,run_names[3]:110}
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients",
                            'gradients/integrated_gradients':"Integrated Gradients (No Multiplier)",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/attention_rollout":"Attention Rollout", 
                            "attention/average_attention":"Average Attention", "random/random_baseline":"Random",
                            "ensembles/ensemble_full":"Ensemble (All Methods)", "ensembles/ensemble_best":"Ensemble (Top 3)"}
    explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 'gradients/integrated_gradients_x_input']
    input_folder = "./explanation_outputs/adv_explanation_outputs_scale"
    output_folder = f"./adv_graph_outputs_scale/{dataset_name}"
    k_percent = .15 
    opt_pred = False
    # graph_type = "combined"
    graph_type = "prob_change"
    # metrics = ["Rank Change", "Top-k Sum Change"]
    metrics = ["Probability Change"]
    
@ex.automain 
def get_explanations(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if not os.path.isdir(_config["output_folder"]):
        os.makedirs(_config["output_folder"])
    if _config["graph_type"] == "combined":
        metrics = {metric:{"Number of Parameters (Millions)":[], "Example Type":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
        for explanation_type in _config["explanation_types"]:
            for run_name in _config['run_names']:
                opt_explanations = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/adv_explanations_opt_pred.pkl", "rb"))
                explanations = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/adv_explanations.pkl", "rb"))
                for explanation, opt_explanation in zip(explanations, opt_explanations):
                    rank_change = compute_percent_change(explanation["attributions"], explanation["adv_attributions"],_config["k_percent"])
                    if rank_change == None:
                        continue 
                    rank_change_prob_change_diff = compute_adv_ratio(explanation["attributions"],explanation["adv_attributions"],explanation["logits"],explanation["adv_logits"],_config["k_percent"])
                    top_sum_diff = compute_top_attr_sum_diff(explanation["attributions"], explanation["adv_attributions"], _config["k_percent"])
                    rank_change_opt = compute_percent_change(opt_explanation["attributions"], opt_explanation["adv_attributions"],_config["k_percent"])
                    if rank_change_opt == None:
                        continue 
                    rank_change_prob_change_diff_opt = compute_adv_ratio(opt_explanation["attributions"],opt_explanation["adv_attributions"],opt_explanation["logits"],opt_explanation["adv_logits"],_config["k_percent"])
                    top_sum_diff_opt = compute_top_attr_sum_diff(opt_explanation["attributions"], opt_explanation["adv_attributions"], _config["k_percent"])
                    
                    metrics["Rank Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Rank Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                    metrics["Rank Change"]["Rank Change"].append(rank_change)
                    metrics["Rank Change"]["Example Type"].append("Unconstrained")
                    metrics["Rank Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Rank Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                    metrics["Rank Change"]["Rank Change"].append(rank_change_opt)
                    metrics["Rank Change"]["Example Type"].append("Constrained")
                    metrics["Top-k Sum Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Top-k Sum Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type]) 
                    metrics["Top-k Sum Change"]["Top-k Sum Change"].append(top_sum_diff.item())
                    metrics["Top-k Sum Change"]["Example Type"].append("Unconstrained")
                    metrics["Top-k Sum Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Top-k Sum Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type]) 
                    metrics["Top-k Sum Change"]["Top-k Sum Change"].append(top_sum_diff_opt.item())
                    metrics["Top-k Sum Change"]["Example Type"].append("Constrained")
            

        for i,metric_name in enumerate(metrics):
            df = pd.DataFrame(metrics[metric_name])
            plt.figure(figsize=(12,8))
            # plt.rc('axes', titlesize=14) #fontsize of the title
            plt.rc('axes', labelsize=28) #fontsize of the x and y labels
            plt.rc('xtick', labelsize=22) #fontsize of the x tick labels
            plt.rc('ytick', labelsize=22) #fontsize of the y tick labels
            # fig, ax = plt.subplots(1,1,figsize=(12,8))
            ax = plt.subplot(1,1,1)
            ax.set_ylim(0.0,1.0)
            sns.lineplot(x="Number of Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, style="Example Type", markers=True, markersize=12 ,legend=False,ax=ax)
            # handles, labels = ax.get_legend_handles_labels()
            # plt.clf()
            # fig = plt.figure(figsize=(3.1,4.5))
            # fig.legend(handles, labels, labelspacing=0.9, prop={"size":15},)
            # plt.tight_layout()
            # plt.savefig(f"{_config['output_folder']}/legend.png")
            # exit()
            # ax.set_title(f"{metric_name} vs. Number of Parameters")
            # plt.legend(loc="upper right", ncol=len(_config["explanation_types"])//4)
            plt.tight_layout()
            if _config["opt_pred"]:
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_combined.png")
            else:
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_combined.png")
            plt.clf()
    elif _config["graph_type"] == "prob_change":
        metrics = {metric:{"Number of Parameters (Millions)":[], "Example Type":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
        for explanation_type in _config["explanation_types"]:
            for run_name in _config['run_names']:
                opt_explanations = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/adv_explanations_opt_pred.pkl", "rb"))
                explanations = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/adv_explanations.pkl", "rb"))
                for explanation, opt_explanation in zip(explanations, opt_explanations):
                    prob_change = compute_prob_change(explanation["logits"], explanation["adv_logits"], explanation, _config)
                    prob_change_opt = compute_prob_change(opt_explanation["logits"], opt_explanation["adv_logits"], opt_explanation, _config)
                    metrics["Probability Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Probability Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                    metrics["Probability Change"]["Probability Change"].append(prob_change)
                    metrics["Probability Change"]["Example Type"].append("Unconstrained")
                    metrics["Probability Change"]["Number of Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                    metrics["Probability Change"]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                    metrics["Probability Change"]["Probability Change"].append(prob_change_opt)
                    metrics["Probability Change"]["Example Type"].append("Constrained")
  
            

        for i,metric_name in enumerate(metrics):
            df = pd.DataFrame(metrics[metric_name])
            plt.figure(figsize=(12,8))
            # plt.rc('axes', titlesize=14) #fontsize of the title
            plt.rc('axes', labelsize=28) #fontsize of the x and y labels
            plt.rc('xtick', labelsize=22) #fontsize of the x tick labels
            plt.rc('ytick', labelsize=22) #fontsize of the y tick labels
            # fig, ax = plt.subplots(1,1,figsize=(12,8))
            ax = plt.subplot(1,1,1)
            ax.set_ylim(-0.05,1.0)
            sns.lineplot(x="Number of Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, style="Example Type", markers=True, markersize=12 ,legend=False,ax=ax)
            # handles, labels = ax.get_legend_handles_labels()
            # plt.clf()
            # fig = plt.figure(figsize=(3.4,4.5))
            # fig.legend(handles, labels, labelspacing=0.9, prop={"size":15},)
            # plt.tight_layout()
            # plt.savefig(f"{_config['output_folder']}/legend.png")
            # exit()
            # ax.set_title(f"{metric_name} vs. Number of Parameters")
            # plt.legend(loc="upper right", ncol=len(_config["explanation_types"])//4)
            plt.tight_layout()
            if _config["opt_pred"]:
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_prob_change.png")
            else:
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_prob_change.png")
            plt.clf()