from transformers import GPT2ForSequenceClassification, T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer 
from transformers import AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification, BertTokenizer
from transformers import BertForSequenceClassification
from custom_models.t5_enc.t5_enc_avg_seq_classification import T5EncoderForSequenceClassificationAvg
import os 
from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.hans import HansDataset
from torchinfo import summary 
import torch 
# from dataset_wrappers.eraser.utils import load_datasets, load_documents 
from dataset_wrappers.eraser.eraser_wrapper import ERASERDataset
import datasets 
from scipy.stats import spearmanr


def randomize_weights(m: torch.nn.Module):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param)

if __name__ == "__main__":
    # model = T5EncoderForSequenceClassificationAvg.from_pretrained("./model_outputs/dn_t5_tiny_enc/mnli/avg-finetune/checkpoint-25260")
    # tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    example_sentence = "here is an example input. It is long enough to show what I'm getting at. Hopefully this has been extended to sufficient length negative"
    inputs = tokenizer(example_sentence, return_tensors="pt")
    outs = model(**inputs, output_attentions=True)
    print(outs["attentions"][0].shape)
    mean_attentions = [torch.mean(outs["attentions"][i],dim=1).squeeze() for i in range(len(outs["attentions"]))]
    curr_attention = mean_attentions[0]
    for i in range(1,len(outs["attentions"])):
        # print(torch.max(curr_attention))
        # print(torch.mean(curr_attention))
        # print(curr_attention.shape)
        mean_curr_attention = torch.sum(curr_attention, dim=0)
        # print(mean_attentions[i])
        # print(mean_curr_attention/inputs["input_ids"].shape[1])
        curr_attention = torch.matmul(mean_attentions[i], curr_attention)
    orig_attentions = mean_attentions
    original_explanation = torch.mean(curr_attention,dim=0)
    for i in range(12): 
        print("Layer: ",i)
        model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
        randomize_weights(model.roberta.encoder.layer[i])
        inputs = tokenizer(example_sentence, return_tensors="pt")
        outs = model(**inputs, output_attentions=True)
        mean_attentions = [torch.mean(outs["attentions"][k],dim=1).squeeze() for k in range(len(outs["attentions"]))]
        curr_attention = mean_attentions[0]
        # print(mean_attentions[i])
        # print(orig_attentions[i])
        print(torch.sum(torch.abs(mean_attentions[i] - orig_attentions[i])))
        for j in range(1,len(outs["attentions"])):
            mean_curr_attention = torch.sum(curr_attention, dim=0)
            # print(mean_attentions[i])
            # print("CURRENT ATTENTION")
            # print(curr_attention)
            # print("MEAN ATTENTION")
            # print(mean_attentions[j])
            # print(mean_curr_attention/inputs["input_ids"].shape[1])
            # input()
            old_curr_attention = curr_attention
            curr_attention = torch.matmul(mean_attentions[j], curr_attention)
            print(torch.sum(torch.abs(old_curr_attention - curr_attention)))
            input()
        explanation = torch.mean(curr_attention, dim=0)
        # print(original_explanation)
        # print(explanation)
        print(spearmanr(original_explanation.detach(), explanation.detach()))
        input()