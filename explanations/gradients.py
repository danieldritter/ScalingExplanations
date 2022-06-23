import torch 
from captum.attr import LayerGradientXActivation
from captum.attr import visualization as viz 
from .base import Explainer
from .utils import compute_sequence_likelihood, get_attention_mask

class Gradients(Explainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, normalize_attributions=False, device=None):
        super().__init__(model)
        self.layer= layer
        self.tokenizer = tokenizer
        self.normalize_attributions = normalize_attributions
        self.multiply_by_inputs = multiply_by_inputs 
        self.device = device 

        
        
        def predict_helper(input_ids, model_kwargs, seq2seq=False):
            # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
            if not seq2seq:
                return self.model(input_ids=input_ids, **model_kwargs).logits
            else:
                gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
                dec_ids = gen_out["sequences"]
                dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id)
                outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
                likelihoods = compute_sequence_likelihood(dec_ids, outs.logits, self.model, is_tuple=False)
                return likelihoods
        self.predict_helper = predict_helper
        self.explainer = LayerGradientXActivation(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    
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
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions = self.explainer.attribute(inputs=inputs["input_ids"],
                                    additional_forward_args=(non_input_forward_args, False), target=targets)
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
            non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
            attributions = self.explainer.attribute(inputs=inputs["input_ids"],additional_forward_args=(non_input_forward_args, True))  
        return_dict["full_attributions"] = attributions
        return_dict["word_attributions"] = attributions.sum(dim=-1) 
        if self.normalize_attributions:
            return_dict["word_attributions"] = return_dict["word_attributions"] / torch.linalg.norm(return_dict["word_attributions"], ord=1, dim=1, keepdims=True)
        return_dict["attr_score"] = return_dict["word_attributions"].sum(dim=1)
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return return_dict 

    def visualize_explanations(self, attributions):
        records = [] 
        for i in range(attributions["word_attributions"].shape[0]):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i], attributions["pred_class"][i],
                attributions["true_class"][i], attributions["attr_class"][i], attributions["attr_score"][i], attributions["raw_input_ids"][i], None)
            records.append(attr_record)
        return viz.visualize_text(records)
        
        