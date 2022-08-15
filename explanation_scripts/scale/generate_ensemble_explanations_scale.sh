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
RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune')

ENSEMBLE_FOLDER_NAME="ensembles/ensemble_full"

OUTPUT_FOLDER='./explanation_outputs/scale_model_explanation_outputs_500_new'

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED


EXPLANATIONS='["gradients/integrated_gradients_x_input", "lime/lime", "shap/shap"]'
ENSEMBLE_FOLDER_NAME="ensembles/ensemble_best"

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED

RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune')

run_explanations "${RUN_NAMES[*]}" "${EXPLANATIONS}" $OUTPUT_FOLDER $ENSEMBLE_FOLDER_NAME $SEED
