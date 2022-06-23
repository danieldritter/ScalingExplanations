from .base import Explainer
import torch 
from functools import reduce, wraps 
import copy 
from inspect import signature
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz 
from .utils import compute_sequence_likelihood, get_attention_mask

class IntegratedGradients(Explainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, normalize_attributions=False, device=None):
        super().__init__(model)
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
                    dec_ids = gen_out["sequences"]
                    dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id)
                    outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=copies)
                    likelihoods = compute_sequence_likelihood(dec_ids, outs.logits, self.model, is_tuple=False)
                    return likelihoods
                else:
                    gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
                    dec_ids = gen_out["sequences"]
                    dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id)
                    outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
                    likelihoods = compute_sequence_likelihood(dec_ids, outs.logits, self.model, is_tuple=False)
                    return likelihoods

        self.predict_helper = predict_helper
        self.explainer = LayerIntegratedGradients(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    
    def get_explanations(self, inputs, seq2seq=False):
        return_dict = {} 
        if self.device != None:
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
                
        if not seq2seq:
            logits = self.model(**inputs).logits 
            return_dict["pred_prob"], targets = torch.softmax(logits,dim=1).max(dim=1)
            return_dict["pred_class"] = targets
            return_dict["attr_class"] = targets 
            if "labels" in inputs:
                return_dict['true_class'] = inputs["labels"]
            baselines = self.construct_baselines(inputs)
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"],baselines=baselines,
                                    additional_forward_args=(non_input_forward_args, False), return_convergence_delta=True,
                                    target=targets)
        else:
            outputs = self.model.generate(inputs["input_ids"], return_dict_in_generate=True, output_scores=True)
            pred_classes = self.tokenizer.batch_decode(outputs["sequences"])
            pred_probs = torch.exp(compute_sequence_likelihood(outputs["sequences"],outputs["scores"], self.model, is_tuple=True))
            return_dict["pred_prob"] = pred_probs
            return_dict["pred_class"] = pred_classes
            return_dict["attr_class"] = pred_classes
            if "labels" in inputs:
                pad_inputs = inputs["labels"].clone()
                pad_inputs[pad_inputs == -100] = self.model.config.pad_token_id
                return_dict["true_class"] = self.tokenizer.batch_decode(pad_inputs)
            baselines = self.construct_baselines(inputs)
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"],baselines=baselines,
                                    additional_forward_args=(non_input_forward_args, True), return_convergence_delta=True)
        return_dict["full_attributions"] = attributions
        return_dict["word_attributions"] = attributions.sum(dim=-1) 
        if self.normalize_attributions:
            return_dict["word_attributions"] = return_dict["word_attributions"] / torch.linalg.norm(return_dict["word_attributions"], ord=1, dim=1, keepdims=True)
        return_dict["attr_score"] = return_dict["word_attributions"].sum(dim=1)
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return_dict["convergence_score"] = deltas         
        return return_dict 

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
    
    def visualize_explanations(self, attributions):
        records = [] 
        for i in range(attributions["word_attributions"].shape[0]):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i], attributions["pred_class"][i],
                attributions["true_class"][i], attributions["attr_class"][i], attributions["attr_score"][i], attributions["raw_input_ids"][i], attributions["convergence_score"][i])
            records.append(attr_record)
        return viz.visualize_text(records)
        

