function run_metrics {
    local explanations=( $1 )
    local run_names=( $2 ) 
    for i in "${!explanations[@]}"
    do
        echo "**************"
        echo "GENERATING PLAUSIBILITY METRICS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            python explanation_scripts/plausibility_metrics.py with "explanation_type=${explanations[i]}" "output_folder=$3" \
            "run_name=${run_names[j]}" seed=$4
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION METRIC GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}

SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'lime/lime' 'shap/shap'
'attention/average_attention' 'attention/attention_rollout' 'random/random_baseline')

RUN_NAMES=( 't5_base_enc/eraser_esnli/avg-finetune' 'gpt2_small/eraser_esnli/cls-finetune' \
'roberta_base/eraser_esnli/cls-finetune' 'bert_base_uncased/eraser_esnli/cls-finetune')

OUTPUT_FOLDER='./explanation_outputs/diff_arch_model_explanation_outputs_500_new'

run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $SEED