from .base import FeatureImportanceExplainer
import torch 
from functools import reduce, wraps 
import copy 
from inspect import signature
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz 
from .utils import compute_sequence_sum, get_attention_mask


class DiffIntegratedGradients:

    def __init__(self, forward_func, multiply_by_inputs=False, device="cpu"):
        self.forward_func = forward_func 
        self.multiply_by_inputs = multiply_by_inputs
        self.device = device 
    
    def attribute(self, inputs, baselines, additional_forward_args, target, return_convergence_delta=True, internal_batch_size=8, num_steps=50):
        inputs.retain_grad()
        grad_sum = torch.zeros_like(inputs)
        curr_k = 1
        for i in range(num_steps // internal_batch_size):
            batch_inputs = []
            for k in range(internal_batch_size):
                shift_input = baselines + (curr_k/num_steps)*(inputs - baselines)
                batch_inputs.append(shift_input)
                curr_k += 1 
            batch_inputs = torch.cat(batch_inputs, dim=0)
            out = self.forward_func(batch_inputs, additional_forward_args)
            pred_val = out[torch.arange(0,out.shape[0]), target]
            grad = torch.autograd.grad(pred_val,inputs,retain_graph=True, grad_outputs=torch.ones(pred_val.shape[0]).to(self.device))
            grad_sum += grad[0]
        final_batch = [] 
        while curr_k != num_steps+1:
            shift_input = baselines + (curr_k/num_steps)*(inputs - baselines)
            final_batch.append(shift_input)
            curr_k += 1
        final_batch = torch.cat(final_batch, dim=0)
        out = self.forward_func(final_batch, additional_forward_args)
        pred_val = out[torch.arange(0,out.shape[0]), target]
        grad = torch.autograd.grad(pred_val,inputs,retain_graph=True, grad_outputs=torch.ones(pred_val.shape[0]).to(self.device))
        grad_sum += grad[0]
        if self.multiply_by_inputs:
            total_attr = grad_sum*(inputs-baselines)/num_steps
        else:
            total_attr = grad_sum / num_steps
        if return_convergence_delta:
            return total_attr, None 
        else:
            return total_attr

class IntegratedGradients(FeatureImportanceExplainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, 
                normalize_attributions=False, device=None, process_as_batch=True, n_steps=50, 
                show_progress=False, internal_batch_size=16, detach_values=False, use_embeds=False):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, 
                        show_progress=show_progress, detach_values=detach_values, use_embeds=use_embeds)
        self.layer = layer
        self.tokenizer = tokenizer
        self.device = device 
        self.normalize_attributions = normalize_attributions
        self.multiply_by_inputs = multiply_by_inputs
        
        def predict_helper(input_vals, model_kwargs):
            if not self.use_embeds:
                # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
                if input_vals.shape != model_kwargs["attention_mask"].shape:
                    num_copies = int(input_vals.shape[0] / model_kwargs["attention_mask"].shape[0])
                    copies = [torch.clone(model_kwargs["attention_mask"]) for i in range(num_copies)]
                    copies = torch.cat(copies,dim=0)
                    if "labels" in model_kwargs:
                        label_copies = [torch.clone(model_kwargs["labels"]) for i in range(num_copies)]
                        label_copies = torch.cat(label_copies)
                        old_labels = model_kwargs["labels"]
                        model_kwargs["labels"] = label_copies
                    if "token_type_ids" in model_kwargs:
                        token_type_copies = [torch.clone(model_kwargs["token_type_ids"]) for i in range(num_copies)]
                        token_type_copies = torch.cat(token_type_copies)
                        old_token_types = model_kwargs["token_type_ids"]
                        model_kwargs["token_type_ids"] = token_type_copies
                    old_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = copies
                    out = self.model(input_ids=input_vals, **model_kwargs)
                    model_kwargs["attention_mask"] = old_mask
                    if "labels" in model_kwargs:
                        model_kwargs["labels"] = old_labels
                    if "token_type_ids" in model_kwargs:
                        model_kwargs["token_type_ids"] = old_token_types
                    return out.logits
                else:
                    return self.model(input_ids=input_vals, **model_kwargs).logits 
            else:
                if input_vals.shape[0:2] != model_kwargs["attention_mask"].shape[0:2]:
                    num_copies = int(input_vals.shape[0] / model_kwargs["attention_mask"].shape[0])
                    copies = [torch.clone(model_kwargs["attention_mask"]) for i in range(num_copies)]
                    copies = torch.cat(copies,dim=0)
                    if "labels" in model_kwargs:
                        label_copies = [torch.clone(model_kwargs["labels"]) for i in range(num_copies)]
                        label_copies = torch.cat(label_copies)
                        old_labels = model_kwargs["labels"]
                        model_kwargs["labels"] = label_copies
                    if "token_type_ids" in model_kwargs:
                        token_type_copies = [torch.clone(model_kwargs["token_type_ids"]) for i in range(num_copies)]
                        token_type_copies = torch.cat(token_type_copies)
                        old_token_types = model_kwargs["token_type_ids"]
                        model_kwargs["token_type_ids"] = token_type_copies
                    old_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = copies
                    out = self.model(inputs_embeds=input_vals, **model_kwargs)
                    model_kwargs["attention_mask"] = old_mask
                    if "labels" in model_kwargs:
                        model_kwargs["labels"] = old_labels
                    if "token_type_ids" in model_kwargs:
                        model_kwargs["token_type_ids"] = old_token_types
                    return out.logits
                else:
                    return self.model(inputs_embeds=input_vals, **model_kwargs).logits

        self.predict_helper = predict_helper
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps
        if self.use_embeds:
            self.explainer = DiffIntegratedGradients(self.predict_helper, multiply_by_inputs=multiply_by_inputs, device=self.device)
        else:
            self.explainer = LayerIntegratedGradients(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    def get_feature_importances(self, inputs, targets=None, model_out=None, inputs_embeds=None):
        attribution_dict = {}
        baselines = self.construct_baselines(inputs)
        non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
        if self.use_embeds:
            attributions, deltas = self.explainer.attribute(inputs=inputs_embeds, baselines=baselines,
                                additional_forward_args=(non_input_forward_args), return_convergence_delta=True,
                                target=targets, internal_batch_size=self.internal_batch_size)
        else:
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"], baselines=baselines,
                                additional_forward_args=(non_input_forward_args), return_convergence_delta=True,
                                target=targets, internal_batch_size=self.internal_batch_size)
        attribution_dict["attributions"] = torch.sum(attributions, dim=-1)
        if deltas != None:
            attribution_dict["deltas"] = deltas.detach().cpu() 
        else:
            attribution_dict["deltas"] = deltas
        return attribution_dict 
        
    def construct_baselines(self, inputs, ref_type="input", ref_token=None):
        input_ids = inputs["input_ids"]
        if ref_token == None:
            if ref_type == "input":
                ref_token = self.model.config.pad_token_id
            elif ref_type == "token_type":
                ref_token = 0
            elif ref_type == "position":
                ref_token = 0 
        if ref_type == "input":
            special_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.all_special_tokens)
            ref_token_ids = torch.where(reduce(torch.logical_or,[input_ids == val for val in special_token_ids]), input_ids, ref_token)
        elif ref_type == "token_type":
            ref_token_ids = torch.full(input_ids.shape, ref_token)
        elif ref_type == "position":
            ref_token_ids = torch.full(input_ids.shape, ref_token)
        if self.use_embeds:
            return self.model.get_input_embeddings()(ref_token_ids)
        else:
            return ref_token_ids
        

