SEED=765

EXPLANATIONS=('gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients' 'lime/lime' 'shap/shap' 'attention/average_attention')
# EXPLANATIONS=( 'attention/average_attention' )

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/cls-finetune' 'dn_t5_tiny_enc/spurious_sst/cls-finetune' \
'dn_t5_small_enc/spurious_sst/cls-finetune' 'dn_t5_base_enc/spurious_sst/cls-finetune')

OUTPUT_FOLDER='./dn_model_explanation_outputs'

for i in "${!EXPLANATIONS[@]}"
do
    echo "**************"
    echo "GENERATING GROUND TRUTH EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        python generate_gt_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" 'output_folder="./explanation_outputs/new_explanation_outputs"' \
        "run_name=${RUN_NAMES[j]}" seed=$SEED
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

# CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/spurious_sst/cls-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_small_enc/spurious_sst/cls-finetune/checkpoint-25260' \
# './model_outputs/dn_t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260')

# for i in "${!EXPLANATIONS[@]}"
# do
#     echo "**************"
#     echo "GENERATING PERTURBATION EXPLANATION METRICS FOR ${EXPLANATIONS[i]}"
#     echo "**************"
#     for j in "${!RUN_NAMES[@]}"
#         do
#         echo "RUN NAME: ${RUN_NAMES[j]}"
#         python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
#         "run_name=${RUN_NAMES[j]}" "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" seed=$SEED "sparsity_levels=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]"
#         python generate_perturbation_explanation_metrics.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
#         "run_name=${RUN_NAMES[j]}" "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" seed=$SEED "most_important_first=False" "sparsity_levels=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5]"
#         if [ "$?" -ne 0 ]; then
#             echo "EXPLANATION METRIC GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
#             exit $?
#         fi
#     done 
# done

