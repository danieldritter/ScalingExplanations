from abc import ABC, abstractmethod
from tqdm import tqdm 
from .utils import compute_sequence_sum  
import torch 
from captum.attr import visualization as viz


class Explainer(ABC):

    def __init__(self, model: torch.nn.Module):
        self.model = model 
    
    @abstractmethod 
    def get_explanations(self,inputs, seq2seq=False):
        pass 

    @abstractmethod
    def visualize_explanations(self, explanations):
        pass 

class FeatureImportanceExplainer(Explainer):

    def __init__(self, model: torch.nn.Module, process_as_batch=False, normalize_attributions=False, 
        show_progress=False, detach_values=True, use_embeds=False):
        super().__init__(model)
        self.process_as_batch = process_as_batch 
        self.normalize_attributions = normalize_attributions
        self.show_progress = show_progress
        self.detach_values = detach_values 
        self.use_embeds = use_embeds
    
    def get_explanations(self, inputs, model_out=None, inputs_embeds=None):
        if self.process_as_batch:
            return_dict = self.get_batch_explanations(inputs, model_out=model_out, inputs_embeds=inputs_embeds)
        else:
            return_dict = self.get_unbatch_explanations(inputs, model_out=model_out, inputs_embeds=inputs_embeds)
        return return_dict 
    
    @abstractmethod 
    def get_feature_importances(self, inputs, targets=None, model_out=None, inputs_embeds=None):
        pass 

    def get_batch_explanations(self, inputs, model_out=None, inputs_embeds=None):
        return_dict = {} 
        if self.device != None:
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)  
            if inputs_embeds != None:
                inputs_embeds = inputs_embeds.to(self.device)
        if model_out != None:
            logits = model_out.logits 
        else:
            if inputs_embeds != None:
                args = {key:val for key,val in inputs.items() if key != "input_ids"}
                logits = self.model(**args, inputs_embeds=inputs_embeds)
            else:
                logits = self.model(**inputs).logits 
        return_dict["pred_prob"], targets = torch.softmax(logits,dim=1).max(dim=1)
        return_dict["pred_prob"] = return_dict["pred_prob"]
        return_dict["pred_class"] = targets
        return_dict["attr_class"] = targets
        if "labels" in inputs:
            return_dict['true_class'] = inputs["labels"]
        attribution_dict = self.get_feature_importances(inputs, targets=targets, model_out=model_out, inputs_embeds=inputs_embeds)
        return_dict["word_attributions"] = attribution_dict["attributions"]
        if self.normalize_attributions:
            return_dict["word_attributions"] = return_dict["word_attributions"] / torch.linalg.norm(return_dict["word_attributions"], ord=1, dim=1, keepdims=True)
        return_dict["attr_score"] = return_dict["word_attributions"].sum(dim=1)
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return_dict["convergence_score"] = attribution_dict["deltas"]   
        if self.detach_values:
            return_dict["pred_prob"] = return_dict["pred_prob"].detach().cpu() 
            return_dict["pred_class"] = return_dict["pred_class"].detach().cpu() 
            return_dict["attr_class"] = return_dict["attr_class"].detach().cpu() 
            return_dict["word_attributions"] = return_dict["word_attributions"].detach().cpu()
            return_dict["true_class"] = return_dict["true_class"].detach().cpu()
        return return_dict 
    
    def get_unbatch_explanations(self, inputs, model_out=None, inputs_embeds=None):
        return_dict = {"pred_prob":[],"pred_class":[],"attr_class":[],"true_class":[], "word_attributions":[], "convergence_score":[]} 
        if self.show_progress:
            input_loader = tqdm(inputs)
        else:
            input_loader = inputs 
        for i,example in enumerate(input_loader):
            # Adding batch dimension
            if self.device == None:
                example = {key:example[key].unsqueeze(0) for key in example}
                if inputs_embeds != None:
                    inputs_embeds[i] = inputs_embeds[i].unsqueeze(0)
            else:
                example = {key:example[key].unsqueeze(0).to(self.device) for key in example}
                if inputs_embeds != None:
                    inputs_embeds[i] = inputs_embeds[i].unsqueeze(0).to(self.device)
            if model_out != None:
                logits = model_out.logits[i].unsqueeze(0)
            else:
                if inputs_embeds != None:
                    args = {key:val for key,val in inputs.items() if key != "input_ids"}
                    logits = self.model(**args, inputs_embeds=inputs_embeds).logits 
                else:
                    logits = self.model(**example).logits 
            pred_prob, target = torch.softmax(logits,dim=1).max(dim=1)
            if self.detach_values:
                return_dict["pred_prob"].append(pred_prob.item())
                return_dict["pred_class"].append(target.item())
                return_dict["attr_class"].append(target.item())
            else:
                return_dict["pred_prob"].append(pred_prob)
                return_dict["pred_class"].append(target)
                return_dict["attr_class"].append(target)            
            if "labels" in example:
                if self.detach_values:
                    return_dict['true_class'].append(example["labels"].item())
                else:
                    return_dict['true_class'].append(example["labels"])
            attribution_dict = self.get_feature_importances(example, targets=target)
            if self.detach_values:
                return_dict["word_attributions"].append(attribution_dict["attributions"].squeeze().detach().cpu())
                return_dict["convergence_score"].append(attribution_dict["deltas"][0].item() if attribution_dict["deltas"][0] != None else None)
            else:
                return_dict["word_attributions"].append(attribution_dict["attributions"].squeeze())
                return_dict["convergence_score"].append(attribution_dict["deltas"][0] if attribution_dict["deltas"][0] != None else None)

        if self.normalize_attributions:
            return_dict["word_attributions"] = [return_dict["word_attributions"][i] / torch.linalg.norm(return_dict["word_attributions"][i], ord=1, keepdims=True) for i in range(len(return_dict["word_attributions"]))]
        if model_out == None:
            return_dict["attr_score"] = [return_dict["word_attributions"][i].sum() for i in range(len(return_dict["word_attributions"]))]
            return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(len(inputs["input_ids"]))]
            return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        else:
            return_dict["attr_score"] = [return_dict["word_attributions"][i].sum() for i in range(len(return_dict["word_attributions"]))]
            return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs[i]["input_ids"]) for i in range(len(inputs))]
            return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return return_dict 

    @staticmethod
    def visualize_explanations(self, attributions):
        records = [] 
        for i in range(len(attributions["word_attributions"])):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i], attributions["pred_class"][i],
                attributions["true_class"][i], attributions["attr_class"][i], attributions["attr_score"][i], attributions["raw_input_ids"][i], attributions["convergence_score"][i])
            records.append(attr_record)
        return viz.visualize_text(records)
    