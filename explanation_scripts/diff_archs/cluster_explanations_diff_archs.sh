#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-explanations-diff-arch"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate anon_llms

function run_explanation_set {
    local explanations=( $1 )
    local run_names=( $2 ) 
    local checkpoint_folders=( $3 ) 
    local layers=( $7 )
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
                "data_cache_dir=${6}" "layer=${layers[j]}"
            else 
                srun python explanation_scripts/generate_explanations.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${8} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
                "data_cache_dir=${6}" "layer=${layers[j]}"
            fi 
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}

SEED=765

NUM_EXAMPLES=500
DATA_CACHE_DIR="/scratch-ssd/anon/data/hf_language_datasets"

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' \
'attention/average_attention' 'attention/attention_rollout' 'random/random_baseline')

OUTPUT_FOLDER='./diff_arch_model_explanation_outputs_500'

RUN_NAMES=( 't5_base_enc/spurious_sst/avg-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

LAYERS=( 'encoder.embed_tokens' 'transformer.wte' 'roberta.embeddings.word_embeddings' 'bert.embeddings.word_embeddings' )
CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/spurious_sst/avg-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-101028' \
'/scratch-ssd/anon/model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260')

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR "${LAYERS[*]}" $SEED

echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

RUN_NAMES=( 't5_base_enc/mnli/avg-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/mnli/avg-finetune/checkpoint-49088' \
'/scratch-ssd/anon/model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-883584' \
'/scratch-ssd/anon/model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR "${LAYERS[*]}" $SEED

echo "MNLI EXPLANATIONS COMPLETED"


RUN_NAMES=( 't5_base_enc/eraser_esnli/avg-finetune' 'gpt2_small/eraser_esnli/cls-finetune' \
'roberta_base/eraser_esnli/cls-finetune' 'bert_base_uncased/eraser_esnli/cls-finetune')

CHECKPOINT_FOLDERS=( '/scratch-ssd/anon/model_outputs/t5_base_enc/eraser_esnli/avg-finetune/checkpoint-137328' \
'/scratch-ssd/anon/model_outputs/gpt2_small/eraser_esnli/cls-finetune/checkpoint-961296' \
'/scratch-ssd/anon/model_outputs/roberta_base/eraser_esnli/cls-finetune/checkpoint-308988' \
'/scratch-ssd/anon/model_outputs/bert_base_uncased/eraser_esnli/cls-finetune/checkpoint-171660')

run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" $OUTPUT_FOLDER $NUM_EXAMPLES $DATA_CACHE_DIR "${LAYERS[*]}" $SEED

echo "ERASER ESNLI EXPLANATIONS COMPLETED"