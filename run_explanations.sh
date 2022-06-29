SEED=765

EXPLANATIONS=( 'gradients/gradients_normalized' 'gradients/gradients_x_input_normalized' 'gradients/gradients_x_input' 'gradients/gradients' \
'gradients/integrated_gradients_normalized' 'gradients/integrated_gradients_x_input_normalized' 'gradients/integrated_gradients_x_input' \
'gradients/integrated_gradients' 'lime/lime' 'lime/lime_normalized' 'shap/shap' 'shap/shap_normalized')

for i in "${EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATIONS FOR ${i}"
    echo "**************"
    python generate_explanations.py with "explanation_type=${i}" 'output_folder="./dn_model_explanation_outputs"' 'num_examples=20' seed=$SEED
    if [ "$?" -ne 0 ]; then
        echo "EXPLANATION GENERATION FAILED FOR ${i}"
        exit $?
    fi
done

for i in "${EXPLANATIONS[@]}"
do
    echo "GENERATING EXPLANATION METRICS FOR ${i}"
    echo "**************"
    python generate_explanation_metrics.py with "explanation_type=${i}" 'output_folder="./dn_model_explanation_outputs"' seed=$SEED
    if [ "$?" -ne 0 ]; then
        echo "METRIC GENERATION FAILED FOR ${i}"
        exit $?
    fi
done

