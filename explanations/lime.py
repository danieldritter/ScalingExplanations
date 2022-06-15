import torch 
from base import Explainer


class LIME(Explainer):

    def __init__(self, model:torch.nn.Module, tokenizer):
        super().__init__(model)
        self.tokenizer = tokenizer

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
    
    def generate_explanations(self, inputs):
        pass 

    def visualize_explanations(self, explanations):
        pass 
        