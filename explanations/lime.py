import torch 
from captum.attr import Lime 
from captum.attr import visualization as viz
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from .base import FeatureImportanceExplainer
from .utils import compute_sequence_sum, get_attention_mask


class LIME(FeatureImportanceExplainer):

    def __init__(self, model:torch.nn.Module, tokenizer, baseline_token_id=None, normalize_attributions=False, device=None, n_samples=200, process_as_batch=False, show_progress=True):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, show_progress=show_progress)
        self.tokenizer = tokenizer
        self.normalize_attributions = normalize_attributions
        self.device = device 
        if baseline_token_id == None:
            if tokenizer.mask_token != None:
                self.baseline_token_id = tokenizer.mask_token_id
            # This is specific to t5, that has multiple masking tokens
            elif hasattr(tokenizer, "_extra_ids"):
                self.baseline_token_id = tokenizer.convert_tokens_to_ids("<extra_id_1>")
            else:
                self.baseline_token_id = tokenizer.unk_token_id
        else:
            self.baseline_token_id = baseline_token_id 
        self.n_samples = n_samples

        def predict_helper(input_ids, model_kwargs, seq2seq=False):
            # For multistep attribution batches, have to expand attention mask and labels to match input_ids 
            if not seq2seq:
                logits = self.model(input_ids=input_ids, **model_kwargs).logits
                return logits
            else:
                gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
                dec_ids = gen_out["sequences"].to(self.device)
                dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id).to(self.device)
                outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
                likelihoods = compute_sequence_sum(dec_ids, outs.logits, self.model, is_tuple=False)
                return likelihoods

        self.predict_helper = predict_helper 
        self.explainer = Lime(self.predict_helper, interpretable_model=SkLearnLasso(alpha=.01))   
    
    def get_feature_importances(self, inputs, seq2seq=False, targets=None):
        attribution_dict = {}
        non_input_forward_args = {key:inputs[key] for key in inputs if key != "input_ids"}
        attributions = self.explainer.attribute(inputs=inputs["input_ids"],baselines=self.baseline_token_id,
                            additional_forward_args=(non_input_forward_args, seq2seq), target=targets, n_samples=self.n_samples)
        attribution_dict["attributions"] = attributions 
        attribution_dict["deltas"] = [None for i in range(len(inputs))] 
        return attribution_dict 