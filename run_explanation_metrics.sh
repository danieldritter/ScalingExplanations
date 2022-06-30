EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap')

RUN_NAMES=( 'dn_t5_mini_enc/mnli/cls-finetune' 'dn_t5_tiny_enc/mnli/cls-finetune' \
'dn_t5_small_enc/mnli/cls-finetune' 'dn_t5_base_enc/mnli/cls-finetune')

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs_complete"' \
        "run_name=${RUN_NAMES[j]}"
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done
