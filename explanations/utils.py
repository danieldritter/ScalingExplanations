import torch 

def compute_sequence_likelihood(dec_ids, logits, model, is_tuple=True):
    if is_tuple:
        probs = [torch.nn.functional.log_softmax(item,dim=1) for item in logits]    
        likelihoods = []
        for example_ind in range(dec_ids.shape[0]):
            total_likelihood = 0.0
            for i, position_probs in enumerate(probs):
                total_likelihood += position_probs[example_ind][dec_ids[example_ind][i+1]]
                if dec_ids[example_ind][i+1] == model.config.eos_token_id:
                    break 
            likelihoods.append(total_likelihood)
        return torch.tensor(likelihoods)
    else:
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        likelihoods = torch.zeros(logits.shape[0])
        for example_ind in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                likelihoods[example_ind] += probs[example_ind][i][dec_ids[example_ind][i]]
                if dec_ids[example_ind][i] == model.config.eos_token_id:
                    break
        return likelihoods

def get_attention_mask(dec_ids, eos_token_id):
    """
    NOTE: This is used for getting attention masks for generated seq2seq text outputs. Assumes a defined eos token 
    and left padding 
    """
    attention_mask = []
    for example in dec_ids:
        mask_vals = [] 
        curr_token = 1 
        for val in example:
            mask_vals.append(curr_token)
            if val == eos_token_id:
                curr_token = 0 
        attention_mask.append(torch.tensor(mask_vals))
    return torch.stack(attention_mask)