
function run_metrics {
    local explanations=( $1 )
    local run_names=( $2 ) 
    for i in "${!explanations[@]}"
    do
        echo "**************"
        echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            python explanation_scripts/generate_gt_explanation_metrics.py with "explanation_type=${explanations[i]}" "output_folder=$3" \
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
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' \
'attention/average_attention' 'attention/attention_rollout' 'random/random_baseline')

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/avg-finetune' 'dn_t5_tiny_enc/spurious_sst/avg-finetune' \
'dn_t5_small_enc/spurious_sst/avg-finetune' 'dn_t5_base_enc/spurious_sst/avg-finetune')

OUTPUT_FOLDER='./explanation_outputs/scale_model_explanation_outputs_500_new'

run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $SEED

RUN_NAMES=( 'dn_t5_mini_enc/mnli/avg-finetune' 'dn_t5_tiny_enc/mnli/avg-finetune' \
'dn_t5_small_enc/mnli/avg-finetune' 'dn_t5_base_enc/mnli/avg-finetune')

run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $SEED

RUN_NAMES=( 'dn_t5_mini_enc/eraser_esnli/avg-finetune' 'dn_t5_tiny_enc/eraser_esnli/avg-finetune' \
'dn_t5_small_enc/eraser_esnli/avg-finetune' 'dn_t5_base_enc/eraser_esnli/avg-finetune')

run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $SEED


