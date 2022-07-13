from transformers import GPT2ForSequenceClassification, T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer 
from transformers import AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification, BertTokenizer
from transformers import BertForSequenceClassification
from custom_models.t5_enc.t5_enc_cls_seq_classification import T5EncoderForSequenceClassificationCLS
import os 
from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.hans import HansDataset
from torchinfo import summary 
import torch 
# from dataset_wrappers.eraser.utils import load_datasets, load_documents 
from dataset_wrappers.eraser.eraser_wrapper import ERASERDataset
import datasets 



def randomize_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param)

if __name__ == "__main__":
    model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # old_model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260")
    # model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    # tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
    # model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # old_model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    # reader = RationaleReader({"tokens":SingleIdTokenIndexer()})
    # out = reader._read("./cached_datasets/eraser/data/esnli/val.jsonl")
    # while True:
    #     print(next(out))
    #     input()
    # train, val, test = load_datasets("./cached_datasets/eraser/data/cose")
    # docs = load_documents("./cached_datasets/eraser/data/cose")
    # print(type(train))
    # for i in range(len(train)):
    #     print(train[i])
        # print(docs[train[i].annotation_id + "_premise"])
        # print(docs[train[i].annotation_id + "_hypothesis"])
        # print(docs[train[i].annotation_id.split(":")[0]])
        # input()
    # eraser_dataset = datasets.load_dataset("./dataset_wrappers/eraser/eraser.py", "multirc", data_dir="./cached_datasets/eraser/data", cache_dir="./cached_datasets")
    wrapped_dataset = ERASERDataset(cache_dir="./cached_datasets",num_samples=None,subset="multirc")
    train_set = wrapped_dataset.get_dataloader(model,tokenizer,batch_size=16,split="train")
    for item in train_set:
        print(item)
        input()
