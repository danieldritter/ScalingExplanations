from transformers import RobertaModel, RobertaTokenizer, GPT2Model, GPT2Tokenizer, T5Model, T5Tokenizer, RobertaForSequenceClassification, T5ForConditionalGeneration

MODELS = {
    "roberta-seq-cls":RobertaForSequenceClassification,
    "roberta":RobertaModel,
    "gpt2":GPT2Model,
    "t5":T5Model,
    "t5_text_to_text":T5ForConditionalGeneration,
}

TOKENIZERS = {
    "roberta": RobertaTokenizer,
    "roberta-seq-cls":RobertaTokenizer,
    "gpt2":GPT2Tokenizer,
    "t5":T5Tokenizer ,
    "t5_text_to_text":T5Tokenizer
}