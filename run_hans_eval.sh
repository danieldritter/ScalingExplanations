SEED=897

RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
'dn_t5_small_enc/hans/cls-finetune' 'dn_t5_base_enc/hans/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' './model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_base_enc/mnli/cls-finetune/checkpoint-245440')

OUTPUT_FOLDER='./explanation_outputs/dn_model_explanation_outputs'

for j in "${!RUN_NAMES[@]}"
    do
    echo "RUN NAME: ${RUN_NAMES[j]}"
    echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
    python hans_eval.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
    seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}"
    if [ "$?" -ne 0 ]; then
        echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
        exit $?
    fi
done 