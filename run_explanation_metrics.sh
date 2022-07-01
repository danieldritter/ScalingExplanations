SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap')

RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
'dn_t5_small_enc/hans/cls-finetune' 'dn_t5_base_enc/hans/cls-finetune')

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_gt_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
        "run_name=${RUN_NAMES[j]}" seed=$SEED
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_base_enc/mnli/cls-finetune/checkpoint-245440')

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING PERTURBATION EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
        "run_name=${RUN_NAMES[j]}" "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" seed=$SEED "sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]"
        python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./dn_model_explanation_outputs"' \
        "run_name=${RUN_NAMES[j]}" "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" seed=$SEED "most_important_first=False" "sparsity_levels=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5]"
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

