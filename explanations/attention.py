import torch 
from .base import FeatureImportanceExplainer


"""
Need to figure out how to properly do attention summation. Currently attention masks and padding are really screwing things up
"""
class AverageAttention(FeatureImportanceExplainer):
    

    def __init__(self, model: torch.nn.Module, tokenizer, process_as_batch=False, normalize_attributions=False, show_progress=False, device=None, mult_back=False):
            super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, show_progress=show_progress)
            self.device = device
            self.tokenizer = tokenizer 
            self.mult_back = mult_back

    def get_feature_importances(self, inputs, seq2seq=False, targets=None):
        output_dict = self.model(**inputs,output_attentions=True, return_dict=True)
        attentions = output_dict["attentions"]
        num_tokens = torch.sum(inputs["attention_mask"], dim=1)
        if not self.mult_back:
            stacked_attention = torch.mean(torch.stack(attentions), dim=2)
            for layer in range(stacked_attention.shape[0]): 
                for batch_ind in range(stacked_attention.shape[1]):
                    stacked_attention[layer][batch_ind][num_tokens[batch_ind]:] = 0.0
            stacked_attention = torch.sum(stacked_attention, dim=2) / num_tokens.view(1,-1,1)
            overall_attention_vals = torch.mean(stacked_attention, dim=0)
        else:
            stacked_attention = torch.mean(torch.stack(attentions), dim=2)
            mult_attentions = [] 
            for batch_ind in range(stacked_attention.shape[1]):
                example_attentions = stacked_attention[:,batch_ind, :, :]
                curr_attention = example_attentions[0]
                curr_attention[batch_ind][num_tokens[batch_ind]:] = 0.0 
                for layer in range(1,stacked_attention.shape[0]):
                    example_attentions[layer][batch_ind][num_tokens[batch_ind]:] = 0.0
                    curr_attention = torch.matmul(example_attentions[layer],curr_attention)
                mult_attentions.append(curr_attention)
            mult_attentions = torch.stack(mult_attentions)
            mult_attentions = torch.sum(mult_attentions, dim=1) / num_tokens.view(-1,1)
            overall_attention_vals = mult_attentions 
        return {"attributions":overall_attention_vals, "deltas":[None for i in range(len(inputs["input_ids"]))]}
