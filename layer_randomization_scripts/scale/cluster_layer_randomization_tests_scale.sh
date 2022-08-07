#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-layer-rand-scale"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

function run_explanation_set {
    local explanations=( $1 )
    local run_names=( $2 ) 
    local checkpoint_folders=( $3 ) 
    local num_layers=( $9 )
    for i in "${!explanations[@]}"
    do
        echo "GENERATING LAYER RANDOMIZED EXPLANATIONS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            echo "CHECKPOINT_FOLDER: ${checkpoint_folders[j]}"
            EXP_MAX_IND="$((${#explanations[@]} - 1))"
            if [ "$i" -eq "$EXP_MAX_IND" ]; then
                srun python ./layer_randomization_scripts/layer_randomization.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${10} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" "save_examples=True" \
                "data_cache_dir=$6" "layer=${7}" "cascading=$8" "num_layers=${num_layers[j]}" "layer_object=${11}"
            else 
                srun python ./layer_randomization_scripts/layer_randomization.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${10} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
                "data_cache_dir=${6}" "layer=${7}" "cascading=$8" "num_layers=${num_layers[j]}" "layer_object=${11}"
            fi 
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}

SEED=765
EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'attention/average_attention' \
'lime/lime' 'shap/shap' 'attention/attention_rollout' 'random/random_baseline')
OUTPUT_FOLDER='./scale_layer_randomization_50'
CACHE_DIR="/scratch-ssd/ms21ddr/data/hf_language_datasets"
LAYER='encoder.embed_tokens'
LAYER_OBJECT="encoder.block"
NUM_EXAMPLES=50


# RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
# 'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune' ) 

# CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260' )

NUM_LAYERS=( 4 4 6 12 )

# CASCADING="False"
# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
# $OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT
# CASCADING="True"
# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
# $OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT

# echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

# RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
# 'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune' )

# CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/mnli/avg-finetune/checkpoint-245440' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/mnli/avg-finetune/checkpoint-220896' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/mnli/avg-finetune/checkpoint-245440' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_base_enc/mnli/avg-finetune/checkpoint-147264' )

# CASCADING="False"
# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
# $OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT
# CASCADING="True"
# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
# $OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT

# echo "MNLI EXPLANATIONS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune' )

CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/eraser_esnli/avg-finetune/checkpoint-308988' \
'/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
'/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/eraser_esnli/avg-finetune/checkpoint-308988' \
'/scratch-ssd/ms21ddr/model_outputs/dn_t5_base_enc/eraser_esnli/avg-finetune/checkpoint-205992' )

# CASCADING="False"
# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
# $OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT
CASCADING="True"
run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
$OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR $LAYER $CASCADING "${NUM_LAYERS[*]}" $SEED $LAYER_OBJECT

echo "ERASER ESNLI EXPLANATIONS COMPLETED"
