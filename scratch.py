from transformers import GPT2ForSequenceClassification, T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer 
from transformers import AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertForSequenceClassification
from custom_models.t5_enc.t5_enc_cls_seq_classification import T5EncoderForSequenceClassificationCLS
import os 
from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.hans import HansDataset
from torchinfo import summary 
import torch 


def randomize_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param)

if __name__ == "__main__":
    model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_base_enc/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = T5EncoderForSequenceClassificationCLS.from_pretrained("./model_outputs/dn_t5_tiny_enc/spurious_sst/cls-finetune/checkpoint-25260")
    # model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = BertForSequenceClassification.from_pretrained("./model_outputs/bert_base_uncased/spurious_sst/cls-finetune/checkpoint-25260")
    # tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
    # model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = GPT2ForSequenceClassification.from_pretrained("./model_outputs/gpt2_small/spurious_sst/cls-finetune/checkpoint-25260")
    # model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    # old_model = RobertaForSequenceClassification.from_pretrained("./model_outputs/roberta_base/spurious_sst/cls-finetune/checkpoint-25260")
    summary(model)
    for name, module in model.named_parameters():
        print(name)
    exit()
    single_layer = model.transformer.h[0]
    randomize_weights(single_layer)
    print("RANDOMIZED WEIGHTS")
    old_params = list(old_model.named_parameters())
    params = list(model.named_parameters())
    for i, item in enumerate(params):
        if not torch.equal(item[1], old_params[i][1]):
            print(item[0])
        else:
            continue 
    exit()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cached_models")
    dataset = MultiNLIDataset(num_samples=500,add_ground_truth_attributions=True, shuffle=True)
    # dataset = HansDataset(num_samples=500, add_ground_truth_attributions=True, shuffle=True)
    train_set = dataset.get_dataloader(model, tokenizer, max_length=512, batch_size=16, split="train")
    for item in train_set:
        decon_tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        print(item["premise"])
        print(item["hypothesis"])
        print(item["input_ids"])
        print(list(zip(decon_tokens,item["ground_truth_attributions"])))
        input()