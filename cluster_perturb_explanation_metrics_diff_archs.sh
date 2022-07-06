#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-metrics"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' \
'attention/average_attention' 'random/random_baseline')

OUTPUT_FOLDER='./diff_arch_model_explanation_outputs'

SUFFICIENCY_VALS='[.95, .9, .8, .5]'
COMPREHENSIVENESS_VALS='[.05,.1,.2,.5]'

RUN_NAMES=( 't5_base_enc/spurious_sst/cls-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')


CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260')

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=True" "sparsity_levels=${COMPREHENSIVENESS_VALS}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=False" "sparsity_levels=${SUFFICIENCY_VALS}"       
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "SPURIOUS_SST PERTURBATION METRICS COMPLETED"

RUN_NAMES=( 't5_base_enc/mnli/cls-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/mnli/cls-finetune/checkpoint-122720' \
'./model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'./model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=True" "sparsity_levels=${COMPREHENSIVENESS_VALS}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=False" "sparsity_levels=${SUFFICIENCY_VALS}"       
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "MNLI EXPLANATIONS COMPLETED"


RUN_NAMES=( 't5_base_enc/hans/cls-finetune' 'gpt2_small/hans/cls-finetune' \
'roberta_base/hans/cls-finetune' 'bert_base_uncased/hans/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/mnli/cls-finetune/checkpoint-122720' \
'./model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'./model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=True" "sparsity_levels=${COMPREHENSIVENESS_VALS}"
        srun python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "most_important_first=False" "sparsity_levels=${SUFFICIENCY_VALS}"       
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "HANS EXPLANATIONS COMPLETED"