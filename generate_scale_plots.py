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
    run_names = ["dn_t5_tiny_enc/hans/cls-finetune", "dn_t5_mini_enc/hans/cls-finetune", 
                "dn_t5_small_enc/hans/cls-finetune", "dn_t5_base_enc/hans/cls-finetune"]
    # dataset_name = 'hans'
    dataset_name = "hans_accuracy"
    plot_ground_truth = False
    parameter_numbers = {"dn_t5_tiny_enc/hans/cls-finetune":11,"dn_t5_mini_enc/hans/cls-finetune":20,
                        "dn_t5_small_enc/hans/cls-finetune":35,"dn_t5_base_enc/hans/cls-finetune":110}
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients*Input",
                            'gradients/integrated_gradients':"Integrated Gradients",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/average_attention":"Average Attention"}
    metrics = ["Ground Truth Overlap", "Mean Rank", "Mean Rank Percentage", "Ground Truth Mass"]
    # metrics = ["Entailed Accuracy", "Non-Entailed Accuracy"]
    # metrics = ["Sufficency", "Comprehensiveness"]
    metrics = ["Sufficiency"]
    explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 'gradients/integrated_gradients_x_input', 
                        'gradients/integrated_gradients', 'lime/lime', 'shap/shap']
    input_folder = "./explanation_outputs/dn_model_explanation_outputs"
    output_folder = f"./explanation_graphs/{dataset_name}"
    
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
                    metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/full_ground_truth_metrics.pkl", "rb"))
                    for metric_name in metrics:
                        for val in metrics[metric_name]:
                            metrics_dict[metric_name]["Parameters (Millions)"].append(_config["parameter_numbers"][run_name])
                            metrics_dict[metric_name]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                            metrics_dict[metric_name][metric_name].append(val)
            for i,metric_name in enumerate(metrics_dict):
                df = pd.DataFrame(metrics_dict[metric_name])
                fig, ax = plt.subplots(1,1,figsize=(12,8))
                sns.lineplot(x="Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, legend='auto',ax=ax)
                ax.set_title(f"{metric_name} vs. Number of Parameters")
                fig.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")
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
                fig, ax = plt.subplots(1,1,figsize=(12,8))
                ax.set_title(f"{metric_name} vs. Number of Parameters")
                sns.lineplot(x="Parameters (Millions)",y=metric_name,hue="Explanation Type", data=df, legend='auto',ax=ax)
                fig.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")    
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
        