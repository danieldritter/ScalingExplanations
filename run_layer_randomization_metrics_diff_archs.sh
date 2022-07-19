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
OUTPUT_FOLDER='./explanation_outputs/layer_randomization_outputs_diff_archs'

RUN_NAMES=( 't5_base_enc/spurious_sst/cls-finetune' 'gpt2_small/spurious_sst/cls-finetune' \
'roberta_base/spurious_sst/cls-finetune' 'bert_base_uncased/spurious_sst/cls-finetune')

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

RUN_NAMES=( 't5_base_enc/mnli/cls-finetune' 'gpt2_small/mnli/cls-finetune' \
'roberta_base/mnli/cls-finetune' 'bert_base_uncased/mnli/cls-finetune')

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

RUN_NAMES=( 't5_base_enc/hans/cls-finetune' 'gpt2_small/hans/cls-finetune' \
'roberta_base/hans/cls-finetune' 'bert_base_uncased/hans/cls-finetune')

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