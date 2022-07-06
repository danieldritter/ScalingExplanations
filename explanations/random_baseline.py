import torch 
from .base import FeatureImportanceExplainer

class RandomBaseline(FeatureImportanceExplainer):

    def __init__(self, model: torch.nn.Module, tokenizer, process_as_batch=False, normalize_attributions=False, show_progress=False, device=None):
        super().__init__(model, process_as_batch=process_as_batch, normalize_attributions=normalize_attributions, show_progress=show_progress)
        self.device = device
        self.tokenizer = tokenizer 

    def get_feature_importances(self, inputs, seq2seq=False, targets=False):
        num_tokens = torch.sum(inputs["attention_mask"], dim=1)
        attributions = []
        for i in range(inputs["input_ids"].shape[0]):
            attr = torch.randperm(num_tokens[i], dtype=torch.float32)
            attr_pad = torch.zeros_like(inputs["input_ids"][i], dtype=torch.float32)
            attr_pad[:num_tokens[i]] = attr 
            attributions.append(attr_pad)
        attributions = torch.stack(attributions)
        return {"attributions":attributions, "deltas":[None for i in range(len(inputs["input_ids"]))]}
