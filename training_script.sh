# MODELS=( 'gpt2/mnli/avg' 'gpt2/sst/avg' 'roberta/mnli/avg' 'roberta/sst/avg' 't5_enc/mnli/avg' 't5_enc/sst/avg' 't5_text_to_text/mnli/avg' 't5_text_to_text/sst/avg' )
MODELS=( 'roberta/mnli/cls-finetune' )

for i in "${MODELS[@]}"
do
    echo "TRAINING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" "batch_size=4" "num_samples=50" 'report_to="none"' 'use_early_stopping=False'
    if [ "$?" -ne 0 ]; then
        echo "TRAINING FAILED FOR ${i}"
        exit $?
    fi
done