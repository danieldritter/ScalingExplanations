SEED=765

EXPLANATIONS=( 'gradients/gradients_normalized' 'gradients/gradients_x_input_normalized' 'gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_normalized' 'gradients/integrated_gradients_x_input_normalized' 'gradients/integrated_gradients_x_input' 'gradients/integrated_gradients')

for i in "${EXPLANATIONS[@]}"
do
    echo "TESTING ${i}"
    echo "**************"
    python generate_explanations.py with "explanation_type=${i}" 'num_samples=200' output_file="explanation_outputs/${i}.html" seed=$SEED
    if [ "$?" -ne 0 ]; then
        echo "TESTING FAILED FOR ${i}"
        exit $?
    fi
done