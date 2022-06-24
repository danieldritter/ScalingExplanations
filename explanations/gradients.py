import torch 
from captum.attr import LayerGradientXActivation
from captum.attr import visualization as viz 
from .base import FeatureImportanceExplainer
from .utils import compute_sequence_sum, get_attention_mask

class Gradients(FeatureImportanceExplainer):

    def __init__(self, model:torch.nn.Module, tokenizer, layer, multiply_by_inputs=False, normalize_attributions=False, device=None, process_as_batch=True):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions)
        self.layer= layer
        self.tokenizer = tokenizer
        self.multiply_by_inputs = multiply_by_inputs 
        self.device = device 
        
        def predict_helper(input_ids, model_kwargs, seq2seq=False):
            # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
            if not seq2seq:
                return self.model(input_ids=input_ids, **model_kwargs).logits
            else:
                gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
                dec_ids = gen_out["sequences"].to(self.device)
                dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id).to(self.device)
                outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
                likelihoods = compute_sequence_sum(dec_ids, outs.logits, self.model, is_tuple=False)
                return likelihoods
        self.predict_helper = predict_helper
        self.explainer = LayerGradientXActivation(self.predict_helper, layer, multiply_by_inputs=multiply_by_inputs)

    def get_feature_importances(self, inputs, seq2seq=False, targets=None):
        non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
        attributions = self.explainer.attribute(inputs=inputs["input_ids"],additional_forward_args=(non_input_forward_args, seq2seq))  
        attributions = torch.sum(attributions, dim=-1)
        attributions_dict = {"attributions":attributions, "deltas":[None for i in range(len(inputs["input_ids"]))]}
        return attributions_dict
        
        