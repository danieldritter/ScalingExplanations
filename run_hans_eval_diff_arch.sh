SEED=897

RUN_NAMES=( 't5_base_enc/hans/cls-finetune' 'gpt2_small/hans/cls-finetune' \
'roberta_base/hans/cls-finetune' 'bert_base_uncased/hans/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/t5_base_enc/mnli/cls-finetune/checkpoint-122720' \
'./model_outputs/gpt2_small/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/roberta_base/mnli/cls-finetune/checkpoint-171808' \
'./model_outputs/bert_base_uncased/mnli/cls-finetune/checkpoint-196352')

OUTPUT_FOLDER='./explanation_outputs/diff_arch_model_explanation_outputs'

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