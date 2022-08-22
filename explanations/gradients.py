import torch 
from captum.attr import LayerGradientXActivation, Saliency, InputXGradient
from captum.attr import visualization as viz 
from .base import FeatureImportanceExplainer
from .utils import compute_sequence_sum, get_attention_mask


from typing import Any, Callable

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage


class DiffSaliency:

    def __init__(self, forward_func, multiply_by_inputs=False):
        self.forward_func = forward_func 
        self.multiply_by_inputs = multiply_by_inputs
    
    def attribute(self,inputs,additional_forward_args,target):
        inputs.retain_grad()
        logits = self.forward_func(inputs, additional_forward_args)
        pred_val = logits[:,target]
        pred_val.backward(retain_graph=True)
        if self.multiply_by_inputs:
            return inputs.grad.data * inputs 
        else:
            return inputs.grad.data

class Gradients(FeatureImportanceExplainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, normalize_attributions=False, device=None, process_as_batch=True, show_progress=False, detach_values=False, use_embeds=False):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, show_progress=show_progress, detach_values=detach_values, use_embeds=use_embeds)
        self.layer= layer
        self.tokenizer = tokenizer
        self.multiply_by_inputs = multiply_by_inputs 
        self.device = device 
        
        def predict_helper(input_vals, model_kwargs):
            if self.use_embeds:
                out = self.model(inputs_embeds=input_vals, **model_kwargs).logits
            else:
                out = self.model(input_ids=input_vals, **model_kwargs).logits
            return out
        self.predict_helper = predict_helper
        if self.use_embeds:
            if multiply_by_inputs:
                self.explainer = DiffSaliency(self.predict_helper, multiply_by_inputs=True)
            else:
                self.explainer = DiffSaliency(self.predict_helper)
        else:
            self.explainer = LayerGradientXActivation(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    def get_feature_importances(self, inputs, targets=None, model_out=None, inputs_embeds=None):
        non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
        if self.use_embeds:
            attributions = self.explainer.attribute(inputs=inputs_embeds,additional_forward_args=(non_input_forward_args), target=targets)  
        else:
            attributions = self.explainer.attribute(inputs=inputs["input_ids"],additional_forward_args=(non_input_forward_args), target=targets)  
        attributions = torch.sum(attributions, dim=-1)
        attributions_dict = {"attributions":attributions, "deltas":[None for i in range(len(inputs["input_ids"]))]}
        return attributions_dict
        
        