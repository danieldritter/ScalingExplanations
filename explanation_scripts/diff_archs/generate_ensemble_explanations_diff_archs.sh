function run_explanations {
    local run_names=( $1 ) 
    for j in "${!run_names[@]}"
        do
        echo "RUN NAME: ${run_names[j]}"
        python explanation_scripts/ensemble_explanations.py with "explanation_types=$2" "output_folder=$3" \
        "ensemble_folder_name=$4" "run_name=${run_names[j]}" seed=$5
        if [ "$?" -ne 0 ]; then
            echo "ENSEMBLE EXPLANATION GENERATION FAILED FOR RUN ${run_names[j]}"
            exit $?
        fi
    done 
}

SEED=765

EXPLANATIONS='["gradients/gradients_x_input", "gradients/gradients", "gradients/integrated_gradients_x_input", "lime/lime", "shap/shap", "attention/average_attention", "attention/attention_rollout"]'

RUN_NAMES=( 't5_base_enc/spurious_sst/avg-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

ENSEMBLE_FOLDER_NAME="ensembles/ensemble_full"

OUTPUT_FOLDER='./explanation_outputs/diff_arch_model_explanation_outputs_500_new'

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 't5_base_enc/mnli/avg-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 't5_base_enc/eraser_esnli/avg-finetune' 'gpt2_small/eraser_esnli/cls-finetune' \
'roberta_base/eraser_esnli/cls-finetune' 'bert_base_uncased/eraser_esnli/cls-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED


EXPLANATIONS='["gradients/integrated_gradients_x_input", "lime/lime", "shap/shap"]'
ENSEMBLE_FOLDER_NAME="ensembles/ensemble_best"

RUN_NAMES=( 't5_base_enc/spurious_sst/avg-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 't5_base_enc/mnli/avg-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 't5_base_enc/eraser_esnli/avg-finetune' 'gpt2_small/eraser_esnli/cls-finetune' \
'roberta_base/eraser_esnli/cls-finetune' 'bert_base_uncased/eraser_esnli/cls-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED
