import torch 
from .utils import get_attention_mask, compute_sequence_sum

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
                likelihoods = compute_sequence_sum(dec_ids, outs.logits, self.model, is_tuple=False)
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
            likelihoods = compute_sequence_sum(label_indices, outs.logits, self.model, is_tuple=False)
            return likelihoods    

    def compute_metric(self, inputs, attributions, k=1):
        pass 




def precision_at_k(attributions, ground_truth, k=1):
    """
    Technically these two input tensors aren't the same shape all the time, as the attributions may have been padded. 
    But the attribution to the padding tokens is always zero, and we're only compared with a comparison of the topk positions,
    so it still works. 
    """
    topk_attr = set(torch.topk(attributions, k=k).indices.tolist())
    topk_gt = set(torch.topk(ground_truth, k=k).indices.tolist())
    overlap = len(topk_gt.intersection(topk_attr))/len(topk_gt)
    return overlap 

def ground_truth_overlap(attributions, ground_truths, return_avg=True):
    ground_truths = [torch.tensor(mask) for mask in ground_truths]
    k_vals = [torch.sum(mask) for mask in ground_truths] 
    overlaps = []
    # This is to track only the cases where there is a valid ground truth attribution. E.g. in mnli, some examples may be all zeros 
    for i in range(len(attributions)):
        if torch.sum(ground_truths[i]) == 0:
            continue 
        max_ind = len(ground_truths[i])
        curr_attr = attributions[i].clone()
        # masking out padding 
        curr_attr[max_ind:] = float("-inf")
        percent_overlap = precision_at_k(curr_attr, ground_truths[i], k=k_vals[i])
        overlaps.append(percent_overlap)
    if len(overlaps)/len(attributions) < 0.9:
        print("More than 10 percent of examples don't have a ground truth. Ratio is:, ", len(overlaps)/len(attributions))
    if return_avg:
        return sum(overlaps)/len(overlaps)
    else:
        return overlaps

def mean_rank(attributions, ground_truths, percentage=True, return_avg=True):
    ground_truths = [torch.tensor(mask).to(attributions[0].device) for mask in ground_truths]
    k_vals = [torch.sum(mask) for mask in ground_truths]     
    rank_vals = []
    rank_val_perc = [] 
    for i in range(len(attributions)):
        if torch.sum(ground_truths[i]) == 0:
            continue 
        max_ind = len(ground_truths[i])
        curr_attr = attributions[i].clone()
        curr_attr[max_ind:] = float("-inf")
        gt_indices = set(torch.topk(ground_truths[i],k=torch.sum(ground_truths[i])).indices.tolist())
        curr_precision = 0.0 
        curr_k = 0
        while curr_precision != 1.0:
            curr_k += 1
            curr_inds = set(torch.topk(curr_attr, k=curr_k).indices.tolist())
            curr_precision = len(gt_indices.intersection(curr_inds))/len(gt_indices)
        rank_vals.append(curr_k)
        if percentage:
            rank_val_perc.append(curr_k/len(ground_truths[i]))
    if len(rank_vals)/len(attributions) < 0.9:
        print("More than 10 percent of examples don't have a ground truth. Ratio is:, ", len(rank_vals)/len(attributions))
    if return_avg:
        if percentage:
            return sum(rank_vals)/len(rank_vals), sum(rank_val_perc)/len(rank_val_perc)
        else:
            return sum(rank_vals)/len(rank_vals)
    else:
        if percentage:
            return rank_vals, rank_val_perc
        else:
            return rank_vals
    
def ground_truth_mass(attributions, ground_truths, return_avg=True):
    ground_truths = [torch.tensor(mask) for mask in ground_truths]
    k_vals = [torch.sum(mask) for mask in ground_truths] 
    masses = []
    for i in range(len(attributions)):
        if torch.sum(ground_truths[i]) == 0:
            continue 
        normalizer = torch.sum(torch.abs(attributions[i]))
        gt_mass = torch.sum(torch.abs(attributions[i][torch.nonzero(ground_truths[i])])/normalizer)
        masses.append(gt_mass)
    if len(masses)/len(attributions) < 0.9:
        print("More than 10 percent of examples don't have a ground truth. Ratio is:, ", len(masses)/len(attributions))
    if return_avg:
        return sum(masses).item()/len(masses)
    else:
        return [item.item() for item in masses]
    