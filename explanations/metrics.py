from .utils import get_attention_mask, compute_sequence_likelihood

class TopKFeatureAblation:

    def __init__(self, model, tokenizer, replace_token_id=None, device=None):
        self.model = model 
        self.device = device 
        self.tokenizer = tokenizer
        if replace_token_id == None:
            if tokenizer.mask_token != None:
                self.replace_token_id = tokenizer.mask_token_id
            else:
                self.replace_token_id = tokenizer.unk_token_id
        else:
            self.replace_token_id = replace_token_id

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
    
    def get_counterfactual(self, inputs, remove_indices, label_indices, seq2seq=False):
        input_ids = inputs["input_ids"].clone()
        input_ids[remove_indices] = self.replace_token_id
        model_kwargs = {key:inputs[key] for key in inputs if key != "input_ids"}
        if not seq2seq:
            return self.model(input_ids=input_ids, **model_kwargs).logits[:,label_indices]
        else:
            gen_out = self.model.generate(input_ids, attention_mask=model_kwargs["attention_mask"], output_scores=True, do_sample=False, return_dict_in_generate=True)
            dec_ids = gen_out["sequences"]
            dec_attention_mask = get_attention_mask(dec_ids, self.model.config.eos_token_id)
            outs = self.model(input_ids=input_ids, decoder_input_ids=dec_ids, decoder_attention_mask=dec_attention_mask, attention_mask=model_kwargs["attention_mask"])   
            likelihoods = compute_sequence_likelihood(label_indices, outs.logits, self.model, is_tuple=False)
            return likelihoods    

    def compute_metric(self, inputs, attributions, k=1):
        