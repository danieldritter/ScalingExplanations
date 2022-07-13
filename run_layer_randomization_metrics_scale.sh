function run_metrics {
    local explanations=( $1 )
    local run_names=( $2 ) 
    for i in "${!explanations[@]}"
    do
        echo "**************"
        echo "GENERATING LAYER RANDOMIZATION CORRELATIONS FOR ${explanations[i]}"
        echo "**************"
        for j in "${!run_names[@]}"
            do
            echo "RUN NAME: ${run_names[j]}"
            python generate_layer_randomization_metrics.py with "explanation_type=${explanations[i]}" "output_folder=$3" \
            "run_name=${run_names[j]}" cascading=$4 absolute_value=$5 seed=$6
            if [ "$?" -ne 0 ]; then
                echo "EXPLANATION METRIC GENERATION ${explanations[i]} FAILED FOR RUN ${run_names[j]}"
                exit $?
            fi
        done 
    done
}

SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' 'attention/average_attention' 'random/random_baseline')
OUTPUT_FOLDER='./explanation_outputs/dn_layer_randomization_outputs_scale'

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/cls-finetune' 'dn_t5_tiny_enc/spurious_sst/cls-finetune' \
'dn_t5_small_enc/spurious_sst/cls-finetune' 'dn_t5_base_enc/spurious_sst/cls-finetune')

CASCADING="True"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="True"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED

RUN_NAMES=( 'dn_t5_mini_enc/mnli/cls-finetune' 'dn_t5_tiny_enc/mnli/cls-finetune' \
'dn_t5_small_enc/mnli/cls-finetune' 'dn_t5_base_enc/mnli/cls-finetune')

CASCADING="True"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="True"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED

RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
'dn_t5_small_enc/hans/cls-finetune' 'dn_t5_base_enc/hans/cls-finetune')

CASCADING="True"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="True"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="True"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED
CASCADING="False"
ABSOLUTE_VALUE="False"
run_metrics "${EXPLANATIONS[*]}" "${RUN_NAMES[*]}" $OUTPUT_FOLDER $CASCADING $ABSOLUTE_VALUE $SEED