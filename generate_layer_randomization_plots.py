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
    run_names = [f"dn_t5_tiny_enc/{dataset_name}/cls-finetune", f"dn_t5_mini_enc/{dataset_name}/cls-finetune", 
                f"dn_t5_small_enc/{dataset_name}/cls-finetune", f"dn_t5_base_enc/{dataset_name}/cls-finetune"]
    explanation_name_map = {'gradients/gradients_x_input':"Grad*Input",'gradients/gradients':"Grad",
                            'gradients/integrated_gradients_x_input':"Integrated Gradients*Input",
                            'gradients/integrated_gradients':"Integrated Gradients",'lime/lime':"Lime",
                            'shap/shap':"KernelSHAP","attention/average_attention":"Average Attention", "random/random_baseline":"Random"}
    explanation_types = ['gradients/gradients_x_input', 'gradients/gradients', 'gradients/integrated_gradients_x_input', 
                        'gradients/integrated_gradients', 'lime/lime', 'shap/shap', 'attention/average_attention', 'random/random_baseline']    
    input_folder = "./explanation_outputs/dn_layer_randomization_outputs_scale"
    output_folder = f"./explanation_graphs_scale/{dataset_name}"
    cascading = True
    
@ex.automain 
def get_explanations(_seed, _config):
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if not os.path.isdir(_config["output_folder"]):
        os.makedirs(_config["output_folder"])

    metrics_dict = {metric:{"Parameters (Millions)":[], "Explanation Type":[], f"{metric}":[]} for metric in _config["metrics"]}
    for explanation_type in _config["explanation_types"]:
        for run_name in _config['run_names']:
            if _config["cascading"]:
                metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/explanations_cascading.pkl", "rb"))
            else:
                metrics = pickle.load(open(f"{_config['input_folder']}/{run_name}/{explanation_type}/explanations_independent.pkl", "rb"))
            print(list(metrics.keys()))
            exit()
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