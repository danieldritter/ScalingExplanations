import torch 
from captum.attr import KernelShap
from captum.attr import visualization as viz
from captum.attr._core.lime import get_exp_kernel_similarity_function
from .base import Explainer
from .utils import compute_sequence_likelihood, get_attention_mask


class SHAPWithKernel(Explainer):

    def __init__(self, model:torch.nn.Module, tokenizer, baseline_token_id=None, normalize_attributions=False, device=None):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.normalize_attributions = normalize_attributions
        self.device = device 
        if baseline_token_id == None:
            if tokenizer.mask_token_id != None:
                self.baseline_token_id = tokenizer.mask_token_id
            else:
                self.baseline_token_id = tokenizer.unk_token_id 
        else:
            self.baseline_token_id = baseline_token_id 
        self.normalize_attributions 

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
        self.explainer = KernelShap(self.predict_helper)
    
    def get_explanations(self, inputs, seq2seq=False):
        return_dict = {"pred_prob":[],"pred_class":[],"attr_class":[],"true_class":[], "full_attributions":[]} 

        if not seq2seq:
            for example in inputs:
                # Adding batch dimension
                if self.device == None:
                    example = {key:example[key].unsqueeze(0) for key in example}
                else:
                    example = {key:example[key].unsqueeze(0).to(self.device) for key in example}
                logits = self.model(**example).logits 
                pred_prob, target = torch.softmax(logits,dim=1).max(dim=1)
                return_dict["pred_prob"].append(pred_prob)
                return_dict["pred_class"].append(target)
                return_dict["attr_class"].append(target)
                if "labels" in example:
                    return_dict['true_class'].append(example["labels"])
                non_input_forward_args = {key:example[key] for key in example if key != "input_ids"}
                attributions = self.explainer.attribute(inputs=example["input_ids"],baselines=self.baseline_token_id,
                                        additional_forward_args=(non_input_forward_args, False), target=target)
                return_dict["full_attributions"].append(attributions.squeeze())
        else:
            for example in inputs:
                output = self.model.generate(example["input_ids"], return_dict_in_generate=True, output_scores=True)
                pred_class = self.tokenizer.batch_decode(output["sequences"])
                pred_probs = torch.exp(compute_sequence_likelihood(output["sequences"],output["scores"], self.model, is_tuple=True))
                return_dict["pred_prob"].append(pred_probs)
                return_dict["pred_class"].append(pred_class)
                return_dict["attr_class"].append(pred_class)
                if "labels" in example:
                    return_dict["true_class"].append(self.tokenizer.batch_decode(example["labels"]))
                non_input_forward_args = {key:example[key] for key in example if key != "input_ids"}
                attributions = self.explainer.attribute(inputs=example["input_ids"],baselines=self.baseline_token_id,
                                        additional_forward_args=(non_input_forward_args, True))
                return_dict["full_attributions"].append(attributions)
        return_dict["word_attributions"] = return_dict["full_attributions"]
        if self.normalize_attributions:
            return_dict["word_attributions"] = [return_dict["word_attributions"][i] / torch.linalg.norm(return_dict["word_attributions"][i], ord=1, keepdims=True) for i in range(len(return_dict["word_attributions"]))]
        return_dict["attr_score"] = [return_dict["word_attributions"][i].sum() for i in range(len(return_dict["word_attributions"]))]
        return_dict["raw_input_ids"] = [self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]) for i in range(len(inputs["input_ids"]))]
        return_dict["raw_input_ids"] = [[val for val in id_list if val != self.tokenizer.pad_token] for id_list in return_dict["raw_input_ids"]]
        return return_dict 

    def visualize_explanations(self, attributions):
        records = [] 
        for i in range(len(attributions["word_attributions"])):
            attr_record = viz.VisualizationDataRecord(attributions["word_attributions"][i], attributions["pred_prob"][i].item(), attributions["pred_class"][i].item(),
                attributions["true_class"][i].item(), attributions["attr_class"][i].item(), attributions["attr_score"][i].item(), attributions["raw_input_ids"][i], None)
            records.append(attr_record)
        return viz.visualize_text(records)        