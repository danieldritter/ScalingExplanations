import torch 
from .base import FeatureImportanceExplainer

class AverageAttention(FeatureImportanceExplainer):
    

    def __init__(self, model: torch.nn.Module, tokenizer, process_as_batch=False, normalize_attributions=False, show_progress=False, device=None, mult_back=False):
            super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, show_progress=show_progress)
            self.device = device
            self.tokenizer = tokenizer 
            self.mult_back = mult_back

    def get_feature_importances(self, inputs, seq2seq=False, targets=None):
        output_dict = self.model(**inputs,output_attentions=True, return_dict=True)
        attentions = output_dict["attentions"]
        # Following https://arxiv.org/pdf/2005.00928.pdf, averaging across heads and then multiplying attention matrices backwards
        if self.mult_back:
            curr_attention = torch.mean(attentions[-1], dim=1)
            for i in reversed(range(len(attentions)-1)):
                avg_attentions = torch.mean(attentions[i], dim=1)
                curr_attention = torch.matmul(avg_attentions, curr_attention)
            overall_attention_vals = torch.sum(curr_attention, dim=1)
        else:
            stacked_attention = torch.stack(attentions)
            stacked_attention = torch.sum(stacked_attention, dim=3)
            stacked_attention = torch.sum(stacked_attention, dim=2)
            overall_attention_vals = torch.sum(stacked_attention, dim=0)
        return {"attributions": overall_attention_vals, "deltas":[None for i in range(len(inputs["input_ids"]))]}