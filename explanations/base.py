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

    def __init__(self, model: torch.nn.Module, process_as_batch=False, normalize_attributions=False, show_progress=False):
        super().__init__(model)
        self.process_as_batch = process_as_batch 
        self.normalize_attributions = normalize_attributions
        self.show_progress = show_progress
    
    def get_explanations(self, inputs, seq2seq=False):
        if self.process_as_batch:
            return_dict = self.get_batch_explanations(inputs, seq2seq=seq2seq)
        else:
            return_dict = self.get_unbatch_explanations(inputs, seq2seq=seq2seq)
        return return_dict 
    
    @abstractmethod 
    def get_feature_importances(self, inputs, seq2seq=False):
        pass 

    def get_batch_explanations(self, inputs, seq2seq=False):
        return_dict = {} 
        if self.device != None:
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
                
        if not seq2seq:
            logits = self.model(**inputs).logits 
            return_dict["pred_prob"], targets = torch.softmax(logits,dim=1).max(dim=1)
            return_dict["pred_prob"] = return_dict["pred_prob"].detach().cpu()
            return_dict["pred_class"] = targets.detach().cpu()
            return_dict["attr_class"] = targets.detach().cpu() 
            if "labels" in inputs:
                return_dict['true_class'] = inputs["labels"].cpu()
            attribution_dict = self.get_feature_importances(inputs, seq2seq=False, targets=targets)
        else:
            outputs = self.model.generate(inputs["input_ids"], return_dict_in_generate=True, output_scores=True)
            pred_classes = self.tokenizer.batch_decode(outputs["sequences"])
            pred_probs = torch.exp(compute_sequence_sum(outputs["sequences"],outputs["scores"], self.model, is_tuple=True, return_probs=True))
            return_dict["pred_prob"] = pred_probs.detach().cpu()
            return_dict["pred_class"] = pred_classes
            return_dict["attr_class"] = pred_classes
            if "labels" in inputs:
                pad_inputs = inputs["labels"].clone()
                pad_inputs[pad_inputs == -100] = self.model.config.pad_token_id
                return_dict["true_class"] = self.tokenizer.batch_decode(pad_inputs)
            attribution_dict = self.get_feature_importances(inputs, seq2seq=True)
        return_dict["word_attributions"] = attribution_dict["attributions"].detach().cpu()
        if self.normalize_attributions:
            return_dict["word_attributions"] = return_dict["word_attributions"] / torch.sum(return_dict["word_attributions"], dim=1, keepdims=True)
        return_dict["attr_score"] = return_dict["word_attributions"].sum(dim=1)
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(inputs["input_ids"].shape[0])]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return_dict["convergence_score"] = attribution_dict["deltas"].detach().cpu()         
        return return_dict 
    
    def get_unbatch_explanations(self, inputs, seq2seq=False):
        return_dict = {"pred_prob":[],"pred_class":[],"attr_class":[],"true_class":[], "word_attributions":[], "convergence_score":[]} 
        if not seq2seq:
            if self.show_progress:
                input_loader = tqdm(inputs)
            else:
                input_loader = inputs 
            for example in input_loader:
                # Adding batch dimension
                if self.device == None:
                    example = {key:example[key].unsqueeze(0) for key in example}
                else:
                    example = {key:example[key].unsqueeze(0).to(self.device) for key in example}
                logits = self.model(**example).logits 
                pred_prob, target = torch.softmax(logits,dim=1).max(dim=1)
                return_dict["pred_prob"].append(pred_prob.item())
                return_dict["pred_class"].append(target.item())
                return_dict["attr_class"].append(target.item())
                if "labels" in example:
                    return_dict['true_class'].append(example["labels"].item())
                attribution_dict = self.get_feature_importances(example, seq2seq=False, targets=target)
                return_dict["word_attributions"].append(attribution_dict["attributions"].squeeze().detach().cpu())
                return_dict["convergence_score"].append(attribution_dict["deltas"][0].item() if attribution_dict["deltas"][0] != None else None)
        else:
            if self.show_progress:
                input_loader = tqdm(inputs)
            else:
                input_loader = inputs 
            for example in input_loader:
                if self.device == None:
                    example = {key:example[key].unsqueeze(0) for key in example}
                else:
                    example = {key:example[key].unsqueeze(0).to(self.device) for key in example}
                output = self.model.generate(example["input_ids"], return_dict_in_generate=True, output_scores=True)
                pred_class = self.tokenizer.batch_decode(output["sequences"])
                pred_probs = torch.exp(compute_sequence_sum(output["sequences"],output["scores"], self.model, is_tuple=True, return_probs=True))
                return_dict["pred_prob"].append(pred_probs.item())
                return_dict["pred_class"].append(" ".join(pred_class))
                return_dict["attr_class"].append(" ".join(pred_class))
                if "labels" in example:
                    return_dict["true_class"].append(" ".join(self.tokenizer.batch_decode(example["labels"])))
                attribution_dict = self.get_feature_importances(example, seq2seq=True)
                return_dict["word_attributions"].append(attribution_dict["attributions"].squeeze().detach().cpu())
                return_dict["convergence_score"].append(attribution_dict["deltas"][0].item() if attribution_dict["deltas"][0] != None else None)
        if self.normalize_attributions:
            return_dict["word_attributions"] = [return_dict["word_attributions"][i] / torch.linalg.norm(return_dict["word_attributions"][i], ord=1, keepdims=True) for i in range(len(return_dict["word_attributions"]))]
        return_dict["attr_score"] = [return_dict["word_attributions"][i].sum() for i in range(len(return_dict["word_attributions"]))]
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(len(inputs["input_ids"]))]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return return_dict 

    def visualize_explanations(self, attributions):
        records = [] 
        for i in range(len(attributions["word_attributions"])):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i], attributions["pred_class"][i],
                attributions["true_class"][i], attributions["attr_class"][i], attributions["attr_score"][i], attributions["raw_input_ids"][i], attributions["convergence_score"][i])
            records.append(attr_record)
        return viz.visualize_text(records)
    