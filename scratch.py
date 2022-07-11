from transformers import GPT2ForSequenceClassification, T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer 
from transformers import AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertForSequenceClassification
from custom_models.t5_enc.t5_enc_cls_seq_classification import T5EncoderForSequenceClassificationCLS
import os 
from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.hans import HansDataset
from torchinfo import summary 
import torch 
from dataset_wrappers.eraser.utils import load_datasets, load_documents 



def randomize_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param)

if __name__ == "__main__":
    # model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260")
    # model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
    # model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    # reader = RationaleReader({"tokens":SingleIdTokenIndexer()})
    # out = reader._read("./cached_datasets/eraser/data/esnli/val.jsonl")
    # while True:
    #     print(next(out))
    #     input()
    train, val, test = load_datasets("./cached_datasets/eraser/data/multirc")
    docs = load_documents("./cached_datasets/eraser/data/multirc")
    print(type(train))
    for i in range(len(train)):
        print(train[i])
        # print(docs[train[i].annotation_id + "_premise"])
        # print(docs[train[i].annotation_id + "_hypothesis"])
        input()
