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
    # run_names = [f"dn_t5_tiny_enc/{dataset_name}/avg-finetune", f"dn_t5_mini_enc/{dataset_name}/avg-finetune", 
                # f"dn_t5_small_enc/{dataset_name}/avg-finetune", f"dn_t5_base_enc/{dataset_name}/avg-finetune"]
    run_names = [f't5_base_enc/{dataset_name}/avg-finetune', f'gpt2_small/{dataset_name}/cls-finetune',
                f'roberta_base/{dataset_name}/cls-finetune', f'bert_base_uncased/{dataset_name}/cls-finetune']
    # model_names = {run_names[0]:"DNT5 Tiny", run_names[1]: "DNT5 Mini", run_names[2]:"DNT5 Small", run_names[3]:"DNT5 Base"}
    model_names = {run_names[0]:"T5 Base", run_names[1]: "GPT2 Small", run_names[2]:"Roberta Base", run_names[3]:"BERT Base"}
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients",
                            'gradients/integrated_gradients':"Integrated Gradients (No Multiplier)",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/attention_rollout":"Attention Rollout", 
                            "attention/average_attention":"Average Attention", "random/random_baseline":"Random"}
    explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 
                        'gradients/integrated_gradients_x_input',
                        'attention/average_attention', 'attention/attention_rollout', 'random/random_baseline']
    input_folder = "./explanation_outputs/diff_archs_layer_randomization_50"
    output_folder = f"./layer_randomization_graphs_diff_archs/{dataset_name}"
    # input_folder = "./explanation_outputs/scale_layer_randomization_50"
    # output_folder = f"./layer_randomization_graphs_scale/{dataset_name}"
    cascading = False
    absolute_value = False
    if cascading:
        if absolute_value:
            metric = "Rank Correlation (Cascading+Abs)"
        else:
            metric = "Rank Correlation (Cascading)"
    else:
        if absolute_value:
            metric = "Rank Correlation (Abs)"
        else:
            metric = "Rank Correlation"

@ex.automain 
def get_explanations(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if not os.path.isdir(_config["output_folder"]):
        os.makedirs(_config["output_folder"])

    metrics_dict = {run_name:{"Layer":[], "Explanation Type":[], f"{_config['metric']}":[]} for run_name in _config["run_names"]}
    for explanation_type in _config["explanation_types"]:
        for run_name in _config['run_names']:
            if _config["cascading"]:
                if _config["absolute_value"]:
                    metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/rank_corr_cascading_abs.pkl", "rb"))
                else:
                    metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/rank_corr_cascading.pkl", "rb"))
            else:
                if _config["absolute_value"]:
                    metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/rank_corr_abs.pkl", "rb"))
                else:
                    metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/rank_corr.pkl", "rb"))
            sorted_metrics = ["Full Model", "Classification Head"] + [f"Layer {i}" for i in range(len(metrics)-3)] + ["Embeddings"]
            for layer in metrics:
                # Make one plot for each image here, and then put them into a subplot 
                for val in metrics[layer]:
                    if layer == "Full Model":
                        metrics_dict[run_name]["Layer"].append("Full")
                    elif layer == "Classification Head":
                        metrics_dict[run_name]["Layer"].append("Head")
                    else:
                        metrics_dict[run_name]["Layer"].append(layer)
                    metrics_dict[run_name]["Explanation Type"].append(_config["explanation_name_map"][explanation_type])
                    metrics_dict[run_name][f"{_config['metric']}"].append(val)
    # fig, axs = plt.subplots(len(metrics_dict) // 2, len(metrics_dict) - len(metrics_dict)//2, figsize=(32,18))
    # flat_axs = [] 
    # for axs_list in axs:
    #     flat_axs.extend(axs_list)
    # for i,run_name in enumerate(metrics_dict):
    #     df = pd.DataFrame(metrics_dict[run_name])
    #     # fig, ax = plt.subplots(1,1,figsize=(12,8))
    #     sns.lineplot(x="Layer",y=f"{_config['metric']}",hue="Explanation Type", data=df, legend='auto',ax=flat_axs[i], sort=False)
    #     flat_axs[i].set_title(f"{_config['model_names'][run_name]}")
    #     fig.suptitle(f"{_config['metric']}")
    #     fig.savefig(f"{_config['output_folder']}/{_config['metric'].replace(' ','_')}.png")
    for i,run_name in enumerate(metrics_dict):
        plt.figure(figsize=(15,8))
        df = pd.DataFrame(metrics_dict[run_name])
        sns.lineplot(x="Layer",y=f"{_config['metric']}",hue="Explanation Type", data=df, legend=False)
        # plt.legend(loc=(0.69,0.39),prop={"size":18})
        # plt.legend(prop={"size":18})
        plt.savefig(f"{_config['output_folder']}/{_config['metric'].replace(' ', '_')}_{run_name.split('/')[0]}.png")
        plt.clf()