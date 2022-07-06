SEED=765

OUTPUT_FOLDER='./explanation_outputs/dn_model_explanation_outputs'

EXPLANATIONS=( 'random/random_baseline' )

RUN_NAMES=( 'dn_t5_mini_enc/spurious_sst/cls-finetune' 'dn_t5_tiny_enc/spurious_sst/cls-finetune' \
'dn_t5_small_enc/spurious_sst/cls-finetune' 'dn_t5_base_enc/spurious_sst/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/dn_t5_small_enc/spurious_sst/cls-finetune/checkpoint-25260' \
'./model_outputs/dn_t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260')


for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True"
        else 
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder={OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}"
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "SPURIOUS_SST EXPLANATIONS COMPLETED"

RUN_NAMES=( 'dn_t5_mini_enc/mnli/cls-finetune' 'dn_t5_tiny_enc/mnli/cls-finetune' \
'dn_t5_small_enc/mnli/cls-finetune' 'dn_t5_base_enc/mnli/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' './model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_base_enc/mnli/cls-finetune/checkpoint-245440')

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True" 
        else 
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" 
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "MNLI EXPLANATIONS COMPLETED"


RUN_NAMES=( 'dn_t5_mini_enc/hans/cls-finetune' 'dn_t5_tiny_enc/hans/cls-finetune' \
'dn_t5_small_enc/hans/cls-finetune' 'dn_t5_base_enc/hans/cls-finetune')

CHECKPOINT_FOLDERS=( './model_outputs/dn_t5_mini_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_tiny_enc/mnli/cls-finetune/checkpoint-245440' './model_outputs/dn_t5_small_enc/mnli/cls-finetune/checkpoint-245440' \
'./model_outputs/dn_t5_base_enc/mnli/cls-finetune/checkpoint-245440')

for i in "${!EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${EXPLANATIONS[i]}"
    echo "**************"
    for j in "${!RUN_NAMES[@]}"
        do
        echo "RUN NAME: ${RUN_NAMES[j]}"
        echo "CHECKPOINT_FOLDER: ${CHECKPOINT_FOLDERS[j]}"
        EXP_MAX_IND="$((${#EXPLANATIONS[@]} - 1))"
        if [ "$i" -eq "$EXP_MAX_IND" ]; then
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}" "save_examples=True"
        else 
            python generate_explanations.py with "explanation_type=${EXPLANATIONS[i]}" "output_folder=${OUTPUT_FOLDER}" \
            'num_examples=1000' seed=$SEED "checkpoint_folder=${CHECKPOINT_FOLDERS[j]}" "run_name=${RUN_NAMES[j]}"
        fi 
        if [ "$?" -ne 0 ]; then
            echo "EXPLANATION GENERATION ${EXPLANATIONS[i]} FAILED FOR RUN ${RUN_NAMES[j]}"
            exit $?
        fi
    done 
done

echo "HANS EXPLANATIONS COMPLETED"