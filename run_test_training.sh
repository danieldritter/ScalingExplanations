MODELS=( 'roberta/mnli/finetune-avg' 'roberta/mnli/finetune-cls' 'roberta/mnli/avg' 'roberta/mnli/cls' \
        'roberta/sst/avg' 'roberta/sst/cls' 'roberta/sst/finetune-avg' 'roberta/sst/finetune-cls' \
        'roberta/spurious_sst/avg' 'roberta/spurious_sst/cls' 'roberta/spurious_sst/finetune-avg' 'roberta/spurious_sst/finetune-cls' )

for i in "${MODELS[@]}"
do
    echo "TESTING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" 'num_samples=500' 'num_epochs=3' 'report_to="none"' \
        'track_train_metrics=True' 'save_strategy="no"' 'pretrained_model_config="./configs/tests/roberta_test.json"' \
        'load_best_model_at_end=False' 'use_early_stopping=False'
    if [ "$?" -ne 0 ]; then
        echo "TESTING FAILED FOR ${i}"
        exit $?
    fi
done

MODELS=( 't5_text_to_text/mnli/head_only' 't5_text_to_text/mnli/finetune' 't5_text_to_text/sst/head_only' \
        't5_text_to_text/sst/finetune' 't5_text_to_text/spurious_sst/head_only' 't5_text_to_text/spurious_sst/finetune' )
for i in "${MODELS[@]}"
do
    echo "TESTING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" 'num_samples=500' 'num_epochs=3' 'report_to="none"' \
        'track_train_metrics=True' 'save_strategy="no"' 'pretrained_model_config="./configs/tests/t5_test.json"' \
        'load_best_model_at_end=False' 'use_early_stopping=False'
    if [ "$?" -ne 0 ]; then
        echo "TESTING FAILED FOR ${i}"
        exit $?
    fi
done

MODELS=( 't5_enc/mnli/avg-finetune' 't5_enc/mnli/avg' 't5_enc/mnli/cls-finetune' 't5_enc/mnli/cls' 't5_enc/spurious_sst/avg-finetune' \
        't5_enc/spurious_sst/avg' 't5_enc/spurious_sst/cls-finetune' 't5_enc/spurious_sst/cls' 't5_enc/sst/avg-finetune' \
        't5_enc/sst/avg' 't5_enc/sst/cls-finetune' 't5_enc/sst/cls')

for i in "${MODELS[@]}"
do
    echo "TESTING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" 'num_samples=500' 'num_epochs=3' 'report_to="none"' \
        'track_train_metrics=True' 'save_strategy="no"' 'pretrained_model_config="./configs/tests/t5_test.json"' \
        'load_best_model_at_end=False' 'use_early_stopping=False'
    if [ "$?" -ne 0 ]; then
        echo "TESTING FAILED FOR ${i}"
        exit $?
    fi
done

MODELS=( 'gpt2/mnli/avg-finetune' 'gpt2/mnli/avg' 'gpt2/mnli/cls-finetune' 'gpt2/mnli/cls' 'gpt2/spurious_sst/avg-finetune' \
        'gpt2/spurious_sst/avg' 'gpt2/spurious_sst/cls-finetune' 'gpt2/spurious_sst/cls' 'gpt2/sst/avg-finetune' 'gpt2/sst/avg' \
        'gpt2/sst/cls-finetune' 'gpt2/sst/cls' )

for i in "${MODELS[@]}"
do
    echo "TESTING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" 'num_samples=500' 'num_epochs=3' 'report_to="none"' \
        'track_train_metrics=True' 'save_strategy="no"' 'pretrained_model_config="./configs/tests/gpt2_test.json"' \
        'load_best_model_at_end=False' 'use_early_stopping=False'
    if [ "$?" -ne 0 ]; then
        echo "TESTING FAILED FOR ${i}"
        exit $?
    fi
done
