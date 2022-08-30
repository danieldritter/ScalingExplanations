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
    dataset_name = 'eraser_esnli'
    run_names = [f't5_base_enc/{dataset_name}/avg-finetune', f'gpt2_small/{dataset_name}/cls-finetune',
                f'roberta_base/{dataset_name}/cls-finetune', f'bert_base_uncased/{dataset_name}/cls-finetune']
    # dataset_name = "hans_accuracy"
    plot_ground_truth = False
    plausibility = False
    model_names = {run_names[0]:"T5", run_names[1]: "GPT2", run_names[2]:"RoBERTa", run_names[3]:"BERT"}
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients",
                            'gradients/integrated_gradients':"Integrated Gradients (No Multiplier)",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/attention_rollout":"Attention Rollout", 
                            "attention/average_attention":"Average Attention", "random/random_baseline":"Random",
                            "ensembles/ensemble_full":"Ensemble (All Methods)", "ensembles/ensemble_best":"Ensemble (Top 3)"}
    # metrics = ["Ground Truth Overlap", "Mean Rank", "Mean Rank Percentage", "Ground Truth Mass"]
    # metrics = ["Entailed Accuracy", "Non-Entailed Accuracy"]
    metrics = ["Sufficiency", "Comprehensiveness"]
    # metrics = ["Evidence Overlap", "Mean Rank", "Mean Rank Percentage", "Evidence Mass"]
    # metrics = ["Sufficiency"]
    # explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 
                        # 'gradients/integrated_gradients_x_input', 'lime/lime', 'shap/shap',
                        # 'attention/average_attention', 'attention/attention_rollout', 'random/random_baseline']
    # Must match number of explanation methods 
    # hatch_textures = ['/', '\\', '|', '-', '+', 'x', 'o', 'O']
    hatch_textures = ["/","\\","|","-","+"]
    explanation_types = ["lime/lime", "gradients/integrated_gradients_x_input", "shap/shap", "ensembles/ensemble_full", "ensembles/ensemble_best"]
    input_folder = "./explanation_outputs/diff_arch_model_explanation_outputs_500_new"
    output_folder = f"./explanation_graphs_diff_archs_ensemble/{dataset_name}"
    
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
            metrics_dict = {metric:{"Model Name":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
            for explanation_type in _config["explanation_types"]:
                for run_name in _config['run_names']:
                    if not _config["plausibility"]:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/full_ground_truth_metrics.pkl", "rb"))
                    else:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/full_plausibility_metrics.pkl", "rb"))
    
                    for metric_name in metrics:
                        for val in metrics[metric_name]:
                            metrics_dict[metric_name]["Model Name"].append(_config["model_names"][run_name])
                            metrics_dict[metric_name]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                            metrics_dict[metric_name][metric_name].append(val)
            for i,metric_name in enumerate(metrics_dict):
                df = pd.DataFrame(metrics_dict[metric_name])
                plt.figure(figsize=(12,9))
                # plt.rc('font', size=10) #controls default text size
                # plt.rc('axes', titlesize=14) #fontsize of the title
                plt.rc('axes', labelsize=40) #fontsize of the x and y labels
                plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
                plt.rc('ytick', labelsize=26) #fontsize of the y tick labels
                # plt.rc('legend', fontsize=10) #fontsize of the legend
                ax = plt.subplot(1,1,1)
                ax.set_ylim(0.0,1.05)                
                sns.barplot(x="Model Name",y=metric_name,hue="Explanation Type", data=df,ax=ax)
                bars = sorted(ax.patches,key=lambda x: x.xy[0])
                for i,bar in enumerate(bars):
                    bar.set(hatch=_config["hatch_textures"][i % len(_config["hatch_textures"])])
                # plt.legend(title="Explanation Type")
                # plt.legend("",frameon=False)
                # handles, labels = ax.get_legend_handles_labels()
                # plt.clf()
                # fig = plt.figure(figsize=(3.4,2.8))
                # fig.legend(handles, labels, labelspacing=0.9, prop={"size":15}, title="Explanation Type")
                # plt.tight_layout()
                # plt.savefig(f"{_config['output_folder']}/legend.png")
                # exit()
                # plt.legend(loc=(0.0,0.74), ncol=2, title="Explanation Type")
                # ax.set_title(f"{metric_name} for Four Different Pretrained Models")
                if not _config["plausibility"]:
                    plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")
                else:
                    plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}_plausibility.png")
                plt.clf()
        else:
            metrics_dict = {metric:{"Model Name":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
            for metric in _config["metrics"]:
                for explanation_type in _config["explanation_types"]:
                    for run_name in _config['run_names']:
                        metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/{metric}_metrics.pkl", "rb"))    
                        val_diffs = np.stack([np.array(metrics["val_diffs"][i]) for i in range(len(metrics["val_diffs"]))])
                        mean_diffs = np.mean(np.abs(val_diffs),axis=0)
                        for val in mean_diffs:
                            metrics_dict[metric]["Model Name"].append(_config["model_names"][run_name])
                            metrics_dict[metric]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                            metrics_dict[metric][metric].append(val)
            for i, metric_name in enumerate(metrics_dict):
                df = pd.DataFrame(metrics_dict[metric_name])
                plt.figure(figsize=(12,9))
                # plt.rc('font', size=10) #controls default text size
                # plt.rc('axes', titlesize=14) #fontsize of the title
                plt.rc('axes', labelsize=40) #fontsize of the x and y labels
                plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
                plt.rc('ytick', labelsize=26) #fontsize of the y tick labels
                ax = plt.subplot(1,1,1)
                ax.set_ylim(0.0,1.05)
                sns.barplot(x="Model Name",y=metric_name,hue="Explanation Type", data=df, ax=ax)
                bars = sorted(ax.patches,key=lambda x: x.xy[0])
                for i,bar in enumerate(bars):
                    bar.set(hatch=_config["hatch_textures"][i % len(_config["hatch_textures"])])
                # plt.legend(loc=(0.55,0.67))
                plt.legend("", frameon=False)
                plt.savefig(f"{_config['output_folder']}/{metric_name.replace(' ','_')}.png")    
                plt.clf()
    else:
        metrics_dict = {"Model Name":[], "Subset":[], f"Accuracy":[]} 
        for run_name in _config["run_names"]:
            metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/hans_metrics.pkl", "rb"))
            for metric_name in metrics:
                metrics_dict["Model Name"].append(_config["model_names"][run_name])
                if metric_name == "Entailed Accuracy":
                    metrics_dict["Subset"].append("Entailed")
                else:
                    metrics_dict["Subset"].append("Non-Entailed")
                metrics_dict["Accuracy"].append(metrics[metric_name].item())
        df = pd.DataFrame(metrics_dict)
        sns.barplot(x="Model Name",y="Accuracy", hue="Subset", data=df)
        plt.title("Accuracy on HANS for Entailed and Non-Entailed Subsets")
        plt.savefig(f"{_config['output_folder']}/hans_accuracy.png")