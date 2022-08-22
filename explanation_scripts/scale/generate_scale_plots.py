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

@ex.config 
def config():
    seed = 12345
    dataset_name = 'spurious_sst'
    run_names = [f"dn_t5_tiny_enc/{dataset_name}/avg-finetune", f"dn_t5_mini_enc/{dataset_name}/avg-finetune", 
                f"dn_t5_small_enc/{dataset_name}/avg-finetune", f"dn_t5_base_enc/{dataset_name}/avg-finetune"]
    plot_ground_truth = False
    plausibility = False
    parameter_numbers = {run_names[0]:11,run_names[1]:20,
                        run_names[2]:35,run_names[3]:110}
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients",
                            'gradients/integrated_gradients':"Integrated Gradients (No Multiplier)",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/attention_rollout":"Attention Rollout", 
                            "attention/average_attention":"Average Attention", "random/random_baseline":"Random",
                            "ensembles/ensemble_full":"Ensemble (All Methods)", "ensembles/ensemble_best":"Ensemble (Top 3)"}
    # metrics = ["Ground Truth Overlap", "Mean Rank", "Mean Rank Percentage", "Ground Truth Mass"]
    metrics = ["Sufficiency", "Comprehensiveness"]
    # metrics = ["Evidence Overlap", "Mean Rank", "Mean Rank Percentage", "Evidence Mass"]
    # explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 
    #                     'gradients/integrated_gradients_x_input', 'lime/lime', 'shap/shap',
    #                     'attention/average_attention', 'attention/attention_rollout', 'random/random_baseline']
    explanation_types = ["lime/lime", "gradients/integrated_gradients_x_input", "shap/shap", "ensembles/ensemble_full", "ensembles/ensemble_best"]
    input_folder = "./explanation_outputs/scale_model_explanation_outputs_500_new"
    output_folder = f"./explanation_graphs_scale_ensemble/{dataset_name}"
    
@ex.automain 
def get_explanations(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if not os.path.isdir(_config["output_folder"]):
        os.makedirs(_config["output_folder"])

    if _config["dataset_name"] != "hans_accuracy":
        if _config["plot_ground_truth"]:
            metrics_dict = {metric:{"Parameters (Millions)":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
            for explanation_type in _config["explanation_types"]:
                for run_name in _config['run_names']:
                    if not _config["plausibility"]:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/full_ground_truth_metrics.pkl", "rb"))
                    else:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/full_plausibility_metrics.pkl", "rb"))

                    for metric_name in metrics:
                        for val in metrics[metric_name]:
                            metrics_dict[metric_name]["Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                            metrics_dict[metric_name]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                            metrics_dict[metric_name][metric_name].append(val)
            for i,metric_name in enumerate(metrics_dict):
                df = pd.DataFrame(metrics_dict[metric_name])
                # fig, ax = plt.subplots(1,1,figsize=(12,8))
                # plt.tight_layout()
                ax = plt.subplot(1,1,1)
                ax.set_ylim(0.0,1.0)
                sns.lineplot(x="Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, legend=False,ax=ax)
                # ax.set_title(f"{metric_name} vs. Number of Parameters")
                # plt.legend(loc="upper right", ncol=len(_config["explanation_types"])//4)
                if not _config["plausibility"]:
                    plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")
                else:
                    plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_plausibility.png")                # fig, ax = plt.subplots(1,1,figsize=(12,8))
                plt.clf()
        else:
            metrics_dict = {metric:{"Parameters (Millions)":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
            for metric in _config["metrics"]:
                for explanation_type in _config["explanation_types"]:
                    for run_name in _config['run_names']:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/{metric}_metrics.pkl", "rb"))    
                        val_diffs = np.stack([np.array(metrics["val_diffs"][i]) for i in range(len(metrics["val_diffs"]))])
                        mean_diffs = np.mean(np.abs(val_diffs),axis=0)
                        for val in mean_diffs:
                            metrics_dict[metric]["Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                            metrics_dict[metric]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                            metrics_dict[metric][metric].append(val)
            for i, metric_name in enumerate(metrics_dict):
                df = pd.DataFrame(metrics_dict[metric_name])
                ax = plt.subplot(1,1,1)
                ax.set_ylim(0.0,1.0)
                # ax.set_title(f"{metric_name} vs. Number of Parameters")
                sns.lineplot(x="Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, legend=False,ax=ax)
                # plt.legend(loc="upper right", ncol=len(_config["explanation_types"])//4)
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")  
                plt.clf()  
    else:
        metrics_dict = {"Parameters (Millions)":[], "Subset":[], f"Accuracy":[]} 
        for run_name in _config["run_names"]:
            metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/hans_metrics.pkl", "rb"))
            for metric_name in metrics:
                metrics_dict["Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                if metric_name == "Entailed Accuracy":
                    metrics_dict["Subset"].append("Entailed")
                else:
                    metrics_dict["Subset"].append("Non-Entailed")
                metrics_dict["Accuracy"].append(metrics[metric_name].item())
        df = pd.DataFrame(metrics_dict)
        sns.lineplot(x="Parameters (Millions)",y="Accuracy", hue="Subset", data=df)
        plt.title("Accuracy on HANS for Entailed and Non-Entailed Subsets")
        plt.savefig(f"{_config['output_folder']}/hans_accuracy.png")
        