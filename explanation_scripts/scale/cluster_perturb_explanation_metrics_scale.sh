#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-perturb-metrics-scale"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate anon_llms

function run_explanation_set {
    local explanations=( $1 )
    local run_names=( $2 ) 
    local checkpoint_folders=( $3 ) 
    for i in "${!explanations[@]}"
    do
        echo "GENERATING EXPLANATIONS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            echo "CHECKPOINT_FOLDER: ${checkpoint_folders[j]}"
            srun python explanation_scripts/generate_perturbation_explanation_metrics.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
            seed=$SEED "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" "most_important_first=True" "sparsity_levels=${5}"
            srun python explanation_scripts/generate_perturbation_explanation_metrics.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
            seed=$SEED "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" "most_important_first=False" "sparsity_levels=${6}"       
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}


SEED=765

# EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
# 'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' \
# 'attention/average_attention' 'attention/attention_rollout' 'random/random_baseline')
EXPLANATIONS=( 'ensembles/ensemble_full' 'ensembles/ensemble_best' )
 
OUTPUT_FOLDER='./scale_model_explanation_outputs_500_complete'

SUFFICIENCY_VALS='[.95,.9,.8,.5]'
COMPREHENSIVENESS_VALS='[.05,.1,.2,.5]'

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/dn_t5_mini_enc/spurious_sst/avg-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/dn_t5_tiny_enc/spurious_sst/avg-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/dn_t5_small_enc/spurious_sst/avg-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/dn_t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260' )

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $COMPREHENSIVENESS_VALS $SUFFICIENCY_VALS

echo "SPURIOUS_SST PERTURBATION METRICS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/dn_t5_mini_enc/mnli/avg-finetune/checkpoint-245440' \
'/scratch-ssd/anon/model_outputs/dn_t5_tiny_enc/mnli/avg-finetune/checkpoint-220896' \
'/scratch-ssd/anon/model_outputs/dn_t5_small_enc/mnli/avg-finetune/checkpoint-245440' \
'/scratch-ssd/anon/model_outputs/dn_t5_base_enc/mnli/avg-finetune/checkpoint-147264')

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $COMPREHENSIVENESS_VALS $SUFFICIENCY_VALS

echo "MNLI EXPLANATIONS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune' )

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/dn_t5_mini_enc/eraser_esnli/avg-finetune/checkpoint-308988' \
'/scratch-ssd/anon/model_outputs/dn_t5_tiny_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
'/scratch-ssd/anon/model_outputs/dn_t5_small_enc/eraser_esnli/avg-finetune/checkpoint-308988' \
'/scratch-ssd/anon/model_outputs/dn_t5_base_enc/eraser_esnli/avg-finetune/checkpoint-205992' )

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $COMPREHENSIVENESS_VALS $SUFFICIENCY_VALS

echo "ERASER ESNLI EXPLANATIONS COMPLETED"
