from sacred import Experiment 
import os 
import torch 
import numpy as np 
import random 
import pickle 
from explanation_registry import EXPLANATIONS
from explanations.metrics import ground_truth_overlap, mean_rank, ground_truth_mass 

ex = Experiment("gt-metrics")

@ex.config 
def config():
    seed = 12345
    run_name = "bert_base_uncased/spurious_sst/cls-finetune"
    explanation_type = "gradients/integrated_gradients_x_input"
    output_folder = "./explanation_outputs/diff_arch_model_explanation_outputs"
    process_as_batches = True
    full_output_folder = f"{output_folder}/{run_name}/{explanation_type}"
    save_visuals = False
    save_metrics = True
    num_samples = None
    split = "train"
    ex.add_config(f"./configs/task_configs/{run_name}.json")
    ex.add_config(f"./configs/explanations/{explanation_type}.json")

@ex.automain 
def get_gt_metrics(_seed, _config):
    if not os.path.exists(f"{_config['full_output_folder']}/explanations.pkl"):
        print(f"Explanations not present in {_config['full_output_folder']}")
        exit() 
    # Setting manual seeds 
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    attributions_plus_ground_truth = pickle.load(open(f"{_config['full_output_folder']}/explanations.pkl", "rb"))
    attributions = attributions_plus_ground_truth["attributions"]
    if _config["save_visuals"]:
        viz = EXPLANATIONS[_config["explanation_name"]].visualize_explanations(attributions)
        with open(f"{_config['full_output_folder']}/visuals.html", "w+") as file:
            file.write(viz.data)
    
    if _config["save_metrics"]:
        if "ground_truth_attributions" in attributions_plus_ground_truth:
            gt_overlap = ground_truth_overlap(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"])
            print("Ground Truth Overlap: ", gt_overlap)
            mean_rank_val, mean_rank_percentage = mean_rank(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"], percentage=True)
            print("Mean Rank: ", mean_rank_val)
            print("Mean Rank Percentage: ", mean_rank_percentage)
            gt_mass = ground_truth_mass(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"])
            avg_metrics = {"Ground Truth Overlap": gt_overlap, "Mean Rank": mean_rank_val, "Mean Rank Percentage": mean_rank_percentage, "Ground Truth Mass": gt_mass}
            print("Ground Truth Mass: ", gt_mass)
            with open(f"{_config['full_output_folder']}/ground_truth_metrics.pkl", "wb+") as file:
                pickle.dump(avg_metrics, file)
            with open(f"{_config['full_output_folder']}/ground_truth_metrics.txt", "w+") as file:
                file.write(f"Ground Truth Overlap: {gt_overlap} \n")
                file.write(f"Mean Rank: {mean_rank_val} \n")
                file.write(f"Mean Rank Percentange: {mean_rank_percentage} \n")
                file.write(f"Ground Truth Mass: {gt_mass} \n")

            gt_overlap = ground_truth_overlap(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"], return_avg=False)
            mean_rank_val, mean_rank_percentage = mean_rank(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"], percentage=True, return_avg=False)
            gt_mass = ground_truth_mass(attributions["word_attributions"], attributions_plus_ground_truth["ground_truth_attributions"], return_avg=False)
            full_metrics = {"Ground Truth Overlap": gt_overlap, "Mean Rank": mean_rank_val, "Mean Rank Percentage": mean_rank_percentage, "Ground Truth Mass": gt_mass}
            with open(f"{_config['full_output_folder']}/full_ground_truth_metrics.pkl", "wb+") as file:
                pickle.dump(full_metrics, file)
            

