#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-explanations-scale"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

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
            EXP_MAX_IND="$((${#explanations[@]} - 1))"
            if [ "$i" -eq "$EXP_MAX_IND" ]; then
                srun python explanation_scripts/generate_explanations.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${8} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" "save_examples=True" \
                "data_cache_dir=${6}" "layer=${7}"
            else 
                srun python explanation_scripts/generate_explanations.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${8} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
                "data_cache_dir=${6}" "layer=${7}"
            fi 
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
                exit $?
            fi
        done 
    done
}


SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' \
'lime/lime' 'shap/shap' 'attention/average_attention' 'attention/attention_rollout' 'random/random_baseline')

OUTPUT_FOLDER='./scale_model_explanation_outputs_500'

LAYER='encoder.embed_tokens'
NUM_EXAMPLES=500
DATA_CACHE_DIR="/scratch-ssd/ms21ddr/data/hf_language_datasets"

# RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
# 'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune')

# CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_tiny_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_small_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260')

# RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
# 'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune' )

# CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/spurious_sst/avg-finetune/checkpoint-25260' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260' )


# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $LAYER $SEED

# echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

# RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
# 'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune')

# CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/avg-finetune/checkpoint-245440' \
# './model_outputs/dn_t5_tiny_enc/mnli/avg-finetune/checkpoint-245440' './model_outputs/dn_t5_small_enc/mnli/avg-finetune/checkpoint-245440' \
# './model_outputs/dn_t5_base_enc/mnli/avg-finetune/checkpoint-245440')

# RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' 'dn_t5_small_enc/mnli/avg-finetune' )

# CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/mnli/avg-finetune/checkpoint-245440' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/mnli/avg-finetune/checkpoint-220896' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/mnli/avg-finetune/checkpoint-245440')

# run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $LAYER $SEED

# echo "MNLI EXPLANATIONS COMPLETED"

# RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
# 'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune')

# CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
# './model_outputs/dn_t5_tiny_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
# './model_outputs/dn_t5_small_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
# './model_outputs/dn_t5_base_enc/eraser_esnli/avg-finetune/checkpoint-240324')

# RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
# 'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune')

# CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_mini_enc/eraser_esnli/avg-finetune/checkpoint-308988' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_tiny_enc/eraser_esnli/avg-finetune/checkpoint-343320' \
# '/scratch-ssd/ms21ddr/model_outputs/dn_t5_small_enc/eraser_esnli/avg-finetune/checkpoint-308988')

RUN_NAMES=( 'dn_t5_base_enc/eraser_esnli/avg-finetune' )
CHECKPOINT_FOLDERS=( '/scratch-ssd/ms21ddr/model_outputs/dn_t5_base_enc/eraser_esnli/avg-finetune/checkpoint-205992' )

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $LAYER $SEED

echo "ERASER ESNLI EXPLANATIONS COMPLETED"