from explanations.integrated_gradients import IntegratedGradients
from explanations.gradients import Gradients

EXPLANATIONS = {
    "integrated_gradients":IntegratedGradients,
    "gradients": Gradients,
}