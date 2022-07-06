from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer , AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer
from custom_models.t5_enc.t5_enc_cls_seq_classification import T5EncoderForSequenceClassificationCLS
import os 
from dataset_wrappers.multinli import MultiNLIDataset

if __name__ == "__main__":
    dataset = MultiNLIDataset(num_samples=500,add_ground_truth_attributions=True, shuffle=True)
    model = T5EncoderForSequenceClassificationCLS.from_pretrained("google/t5-efficient-tiny", cache_dir="./cached_models", num_labels=3)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
    train_set = dataset.get_dataloader(model, tokenizer, max_length=512, batch_size=16, split="train")
    for item in train_set:
        decon_tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])
        print(item["premise"])
        print(item["hypothesis"])
        print(item["input_ids"])
        print(list(zip(decon_tokens,item["ground_truth_attributions"])))
        input()