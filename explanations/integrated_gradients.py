from .base import Explainer
import torch 
from functools import reduce, wraps 
from inspect import signature
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz 
# def predict_helper(inputs):
#     input_ids = inputs["input_ids"]
#     outs = 

class IntegratedGradientsByLayer(Explainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layers, multiply_by_inputs=False):
        super().__init__(model)
        self.layers = layers 
        self.tokenizer = tokenizer
        
        @wraps(self.model.forward)
        def predict_helper(input_ids, model_kwargs):
            # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
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
                out = self.model(input_ids=input_ids, **model_kwargs).logits
                model_kwargs["attention_mask"] = old_mask
                if "labels" in model_kwargs:
                    model_kwargs["labels"] = old_labels
                return out
            else:
                return self.model(input_ids=input_ids, **model_kwargs).logits
        self.predict_helper = predict_helper
        self.explainer = LayerIntegratedGradients(self.predict_helper, layers, multiply_by_inputs=multiply_by_inputs)

    
    def get_explanations(self, inputs, seq2seq=False):
        return_dict = {} 
        logits = self.model(**inputs).logits 
        if not seq2seq:
            return_dict["pred_prob"], targets = torch.softmax(logits,dim=1).max(dim=1)
            return_dict["pred_class"] = targets
            return_dict["attr_class"] = targets 
            if "labels" in inputs:
                return_dict['true_class'] = inputs["labels"]
            baselines = self.construct_baselines(inputs)
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions, deltas = self.explainer.attribute(inputs=inputs["input_ids"],baselines=baselines,
                                    additional_forward_args=non_input_forward_args, return_convergence_delta=True,
                                    target=targets)
            return_dict["full_attributions"] = attributions
            return_dict["word_attributions"] = attributions.sum(dim=-1)
            return_dict["attr_score"] = attributions.sum(dim=1).sum(dim=1)
            return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i], skip_special_tokens=True) for i in range(inputs["input_ids"].shape[0])]
            return_dict["convergence_score"] = deltas 
        else:
            raise NotImplementedError("Still need to do seq2seq case")
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
    
    def visualize_examples(self, attributions):
        records = [] 
        for i in range(attributions["word_attributions"].shape[0]):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i], attributions["pred_class"][i],
                attributions["true_class"][i], attributions["attr_class"][i], attributions["attr_score"][i], attributions["raw_input_ids"][i], attributions["convergence_score"][i])
            records.append(attr_record)
        return viz.visualize_text(records)
        

