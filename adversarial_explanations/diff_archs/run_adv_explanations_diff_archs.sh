#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-explanations-scale"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate anon_llms

function run_adversarial_explanations {
    local explanations=( $1 )
    local run_names=( $2 ) 
    local checkpoint_folders=( $3 ) 
    for i in "${!explanations[@]}"
    do
        echo "GENERATING ADVERSARIAL EXPLANATIONS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            echo "CHECKPOINT_FOLDER: ${checkpoint_folders[j]}"
            srun python adversarial_explanations/generate_adversarial_examples.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
            "num_examples=$5" seed=${7} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
            "data_cache_dir=${6}" "optimize_pred=True"
            srun python adversarial_explanations/generate_adversarial_examples.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
            "num_examples=$5" seed=${7} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
            "data_cache_dir=${6}" "optimize_pred=False"
            if [ "$?" -ne 0 ]; then
                echo "ADVERSARIAL EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
                exit $?
            fi
        done 
    done
}


EXPLANATIONS=( 'gradients/gradients' 'gradients/gradients_x_input' 'gradients/integrated_gradients_x_input' )
DATA_CACHE_DIR="/scratch-ssd/anon/data/hf_language_datasets"
SEED=768
OUTPUT_FOLDER="./adv_explanation_outputs_diff_archs"
NUM_EXAMPLES=20

RUN_NAMES=( 't5_base_enc/spurious_sst/avg-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-101028' \
'/scratch-ssd/anon/model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260')
run_adversarial_explanations "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $SEED

RUN_NAMES=( 't5_base_enc/mnli/avg-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/mnli/avg-finetune/checkpoint-49088' \
'/scratch-ssd/anon/model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-883584' \
'/scratch-ssd/anon/model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')
run_adversarial_explanations "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $SEED 

RUN_NAMES=( 't5_base_enc/eraser_esnli/avg-finetune' 'gpt2_small/eraser_esnli/cls-finetune' \
'roberta_base/eraser_esnli/cls-finetune' 'bert_base_uncased/eraser_esnli/cls-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/eraser_esnli/avg-finetune/checkpoint-137328' \
'/scratch-ssd/anon/model_outputs/gpt2_small/eraser_esnli/cls-finetune/checkpoint-961296' \
'/scratch-ssd/anon/model_outputs/roberta_base/eraser_esnli/cls-finetune/checkpoint-308988' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/eraser_esnli/cls-finetune/checkpoint-171660')

run_adversarial_explanations "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR $SEED 
