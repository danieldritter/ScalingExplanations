SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' 'attention/average_attention' 'random/random_baseline')

RUN_NAMES=( 't5_base_enc/spurious_sst/cls-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

OUTPUT_FOLDER='./explanation_outputs/diff_arch_model_explanation_outputs'

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_gt_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        "run_name=${RUN_NAMES[j]}" seed=$SEED
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

RUN_NAMES=( 't5_base_enc/mnli/cls-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_gt_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        "run_name=${RUN_NAMES[j]}" seed=$SEED
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

RUN_NAMES=( 't5_base_enc/hans/cls-finetune' 'gpt2_small/hans/cls-finetune' \
'roberta_base/hans/cls-finetune' 'bert_base_uncased/hans/cls-finetune')

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_gt_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
        "run_name=${RUN_NAMES[j]}" seed=$SEED
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done
