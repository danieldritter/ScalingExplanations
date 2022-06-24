from .base import FeatureImportanceExplainer
import torch 
from functools import reduce, wraps 
import copy 
from inspect import signature
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz 
from .utils import compute_sequence_sum, get_attention_mask

class IntegratedGradients(FeatureImportanceExplainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, 
                normalize_attributions=False, device=None, internal_batch_size=16, process_as_batch=True):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions)
        self.layer = layer
        self.tokenizer = tokenizer
        self.device = device 
        self.normalize_attributions = normalize_attributions
        self.multiply_by_inputs = multiply_by_inputs
        
        def predict_helper(input_ids, model_kwargs, seq2seq=False):
            # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
            if not seq2seq:
                if input_ids.shape != model_kwargs["attention_mask"].shape:
                    num_copies = int(input_ids.shape[0] / model_kwargs["attention_mask"].shape[0])
                    copies = [torch.clone(model_kwargs["attention_mask"]) for i in range(num_copies)]
                    copies = torch.cat(copies,dim=0)
                    if "labels" in model_kwargs:
                        label_copies = [torch.clone(model_kwargs["labels"]) for i in range(num_copies)]
                        label_copies = torch.cat(label_copies)
                        old_labels = model_kwargs["labels"]
                        model_kwargs["labels"] = label_copies
                    old_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = copies
                    out = self.model(input_ids=input_ids, **model_kwargs)
                    model_kwargs["attention_mask"] = old_mask
                    if "labels" in model_kwargs:
                        model_kwargs["labels"] = old_labels
                    return out.logits
                else:
                    return self.model(input_ids=input_ids, **model_kwargs).logits
            else:
                if input_ids.shape != model_kwargs["attention_mask"].shape:
                    num_copies = int(input_ids.shape[0] / model_kwargs["attention_mask"].shape[0])
                    copies = [torch.clone(model_kwargs["attention_mask"]) for i in range(num_copies)]
                    copies = torch.cat(copies,dim=0)
                    gen_out = self.model.generate(input_ids, attention_mask=copies, output_scores=True, do_sample=False, return_dict_in_generate=True)
                    dec_ids = gen_out["sequences"].to(self.device)
                    dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id).to(self.device)
                    outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=copies)
                    likelihoods = compute_sequence_sum(dec_ids, outs.logits, self.model, is_tuple=False)
                    return likelihoods
                else:
                    gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
                    dec_ids = gen_out["sequences"].to(self.device)
                    dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id).to(self.device)
                    outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
                    likelihoods = compute_sequence_sum(dec_ids, outs.logits, self.model, is_tuple=False)
                    return likelihoods

        self.predict_helper = predict_helper
        self.internal_batch_size = internal_batch_size
        self.explainer = LayerIntegratedGradients(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    def get_feature_importances(self, inputs, seq2seq=False, targets=None):
        attribution_dict = {}
        if not seq2seq:
            baselines = self.construct_baselines(inputs)
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"],baselines=baselines,
                                    additional_forward_args=(non_input_forward_args, False), return_convergence_delta=True,
                                    target=targets, internal_batch_size=self.internal_batch_size)

        else:
            baselines = self.construct_baselines(inputs)
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"],baselines=baselines,
                                    additional_forward_args=(non_input_forward_args, True), return_convergence_delta=True, internal_batch_size=self.internal_batch_size)            
        attribution_dict["attributions"] = torch.sum(attributions, dim=-1)
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
        return ref_token_ids
        

