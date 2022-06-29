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

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/cls-finetune' 'dn_t5_tiny_enc/spurious_sst/cls-finetune' \
'dn_t5_small_enc/spurious_sst/cls-finetune' 'dn_t5_base_enc/spurious_sst/cls-finetune')

CHECKPOINT_FOLDERS=( "./model_outputs/dn_t5_mini_enc/spurious_sst/cls-finetune/checkpoint-25260" \
"./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260"\
"./model_outputs/dn_t5_small_enc/spurious_sst/cls-finetune/checkpoint-25260"\
"./model_outputs/dn_t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260")

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        srun python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
        'num_examples=1000' seed=$SEED 'checkpoint_folder=${CHECKPOINT_FOLDERS[j]}' 'run_name=${RUN_NAMES[j]}'
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done
