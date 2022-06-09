from transformers import RobertaModel, RobertaTokenizer, GPT2Model, GPT2Tokenizer, T5Model, T5Tokenizer, RobertaForSequenceClassification, T5ForConditionalGeneration
from custom_models.t5_enc_sequence_classification import T5EncoderForSequenceClassification
from custom_models.gpt2_seq_cls import GPT2ForSequenceClassificationCustomPooling

MODELS = {
    "roberta-seq-cls":RobertaForSequenceClassification,
    "roberta":RobertaModel,
    "t5":T5Model,
    "t5_text_to_text":T5ForConditionalGeneration,
    "t5_enc-seq-cls":T5EncoderForSequenceClassification,
    "gpt2":GPT2Model,
    "gpt2-seq-cls":GPT2ForSequenceClassificationCustomPooling
}

TOKENIZERS = {
    "roberta": RobertaTokenizer,
    "roberta-seq-cls":RobertaTokenizer,
    "gpt2":GPT2Tokenizer,
    "gpt2-seq-cls":GPT2Tokenizer,
    "t5":T5Tokenizer,
    "t5_text_to_text":T5Tokenizer,
    "t5_enc-seq-cls":T5Tokenizer
}