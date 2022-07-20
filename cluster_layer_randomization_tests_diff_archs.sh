#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-layer-rand-diff-arch"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

function run_explanation_set {
    local explanations=( $1 )
    local run_names=( $2 ) 
    local checkpoint_folders=( $3 ) 
    local num_layers=( $9 )
    local layers=( $7 )
    local layer_objects=( ${11} )
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
                srun python layer_randomization.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${10} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" "save_examples=True" \
                "data_cache_dir=$6" "layer=${layers[j]}" "cascading=$8" "num_layers=${num_layers[j]}" "layer_object=${layer_objects[j]}"
            else 
                srun python layer_randomization.py with "explanation_type=${explanations[i]}" "output_folder=${4}" \
                "num_examples=$5" seed=${10} "checkpoint_folder=${checkpoint_folders[j]}" "run_name=${run_names[j]}" \
                "data_cache_dir=${6}" "layer=${layers[j]}" "cascading=$8" "num_layers=${num_layers[j]}" "layer_object=${layer_objects[j]}"
            fi 
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}

SEED=765
# EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
# 'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' \
# 'lime/lime' 'shap/shap' 'attention/average_attention' 'random/random_baseline')
EXPLANATIONS=( 'lime/lime' )
OUTPUT_FOLDER='./dn_layer_randomization_outputs_diff_archs'
CACHE_DIR="/scratch-ssd/ms21ddr/data/hf_language_datasets"
LAYER_OBJECTS=( "encoder.block" "transformer.h" "roberta.encoder.layer" "bert.encoder.layer" )
NUM_EXAMPLES=100 
LAYERS=( 'encoder.embed_tokens' 'transformer.wte' 'roberta.embeddings.word_embeddings' 'bert.embeddings.word_embeddings' )
NUM_LAYERS=( 12 12 12 12 )


RUN_NAMES=( 't5_base_enc/spurious_sst/cls-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')
CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260')

CASCADING="False"
run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
$OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR "${LAYERS[*]}" $CASCADING "${NUM_LAYERS[*]}" $SEED "${LAYER_OBJECTS[*]}"
CASCADING="True"
run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
$OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR "${LAYERS[*]}" $CASCADING "${NUM_LAYERS[*]}" $SEED "${LAYER_OBJECTS[*]}"

echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

RUN_NAMES=( 't5_base_enc/mnli/cls-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/mnli/cls-finetune/checkpoint-122720' \
'./model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'./model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')

CASCADING="False"
run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
$OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR "${LAYERS[*]}" $CASCADING "${NUM_LAYERS[*]}" $SEED "${LAYER_OBJECTS[*]}"
CASCADING="True"
run_explanation_set "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" "${CHECKPOINT_FOLDERS[*]}" \
$OUTPUT_FOLDER $NUM_EXAMPLES $CACHE_DIR "${LAYERS[*]}" $CASCADING "${NUM_LAYERS[*]}" $SEED "${LAYER_OBJECTS[*]}"

echo "MNLI EXPLANATIONS COMPLETED"