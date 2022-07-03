from explanations.integrated_gradients import IntegratedGradients
from explanations.gradients import Gradients
from explanations.lime import LIME
from explanations.shap import SHAPWithKernel
from explanations.attention import AverageAttention
EXPLANATIONS = {
    "integrated_gradients":IntegratedGradients,
    "gradients": Gradients,
    "lime":LIME,
    "shap":SHAPWithKernel,
    "average_attention":AverageAttention
}