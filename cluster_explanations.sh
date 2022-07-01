#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-interp"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap')

# RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
# 'dn_t5_small_enc/hans/cls-finetune' 'dn_t5_base_enc/hans/cls-finetune')
RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/cls-finetune' 'dn_t5_tiny_enc/spurious_sst/cls-finetune' \
'dn_t5_small_enc/spurious_sst/cls-finetune' )

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/dn_t5_small_enc/spurious_sst/cls-finetune/checkpoint-25260' )

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        else 
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/mnli/cls-finetune' 'dn_t5_tiny_enc/mnli/cls-finetune' \
'dn_t5_small_enc/mnli/cls-finetune' )

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' )

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        else 
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "MNLI EXPLANATIONS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
'dn_t5_small_enc/hans/cls-finetune' )

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' )

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        else 
            srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"'
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "HANS EXPLANATIONS COMPLETED"