from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer , AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer
import os 
from torchsummary import summary 

if __name__ == "__main__":
    # Setting cache to store models locally 
    os.environ["TRANSFORMERS_CACHE"] = "./cached_models"

    test_text = ["this is a sentence"]
    test_out = ["the next sentence is here"]
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # config = AutoConfig.from_pretrained("./test_configs/t5_test.json")
    # full_model = T5ForConditionalGeneration(config)
    # enc_model = T5EncoderModel(config)
    # token_out = tokenizer(test_text, return_tensors="pt")
    # dec_token_out = tokenizer(test_out, return_tensors="pt")
    # full_out = full_model(input_ids=token_out["input_ids"], attention_mask=token_out["attention_mask"], decoder_input_ids=dec_token_out["input_ids"],decoder_attention_mask=dec_token_out["attention_mask"],output_hidden_states=True)
    # enc_out = enc_model(input_ids=token_out["input_ids"], attention_mask=token_out["attention_mask"])
    # summary(full_model)
    # summary(enc_model)
    # print(enc_out["last_hidden_state"].shape)
    # print(list(full_out.keys()))
    # print(full_out["decoder_hidden_states"][-1].shape)
    # print(len(full_out["decoder_hidden_states"]))
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # config = AutoConfig.from_pretrained("./test_configs/gpt2_test.json")
    # model = GPT2Model(config)
    # token_out = tokenizer(test_text, return_tensors="pt")
    # dec_out = tokenizer(test_out)
    # out = model(**token_out)
    # print(token_out)
    # print(out["last_hidden_state"].shape)
    # print(summary(model))
    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # config = AutoConfig.from_pretrained("./test_configs/roberta_test.json")
    # model = RobertaModel(config)
    # token_out = tokenizer(test_text, return_tensors="pt")
    # out = model(**token_out)
    # print(list(out.keys()))
    # print(out["last_hidden_state"].shape)