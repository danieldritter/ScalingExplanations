# MODELS=( 'gpt2/mnli/avg' 'gpt2/sst/avg' 'roberta/mnli/avg' 'roberta/sst/avg' 't5_enc/mnli/avg' 't5_enc/sst/avg' 't5_text_to_text/mnli/avg' 't5_text_to_text/sst/avg' )
MODELS=( 't5_text_to_text/mnli/finetune' )

for i in "${MODELS[@]}"
do
    echo "TRAINING ${i}"
    echo "**************"
    python train_models.py with "run_name=${i}" "batch_size=4"
    if [ "$?" -ne 0 ]; then
        echo "TRAINING FAILED FOR ${i}"
        exit $?
    fi
done