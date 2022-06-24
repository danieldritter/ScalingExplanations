from transformers import RobertaModel, RobertaTokenizer, GPT2Model, GPT2Tokenizer, T5Model, T5Tokenizer, RobertaForSequenceClassification, T5ForConditionalGeneration, GPT2ForSequenceClassification, BertForSequenceClassification, BertTokenizer
from custom_models.t5_enc.t5_enc_cls_seq_classification import T5EncoderForSequenceClassificationCLS
from custom_models.t5_enc.t5_enc_avg_seq_classification import T5EncoderForSequenceClassificationAvg
from custom_models.gpt.gpt2_avg_seq_classification import GPT2ForSequenceClassificationAvg
from custom_models.roberta.roberta_avg_seq_classification import RobertaForSequenceClassificationAvg
from custom_models.bert.bert_avg_seq_classification import BertForSequenceClassificationAvg

MODELS = {
    "roberta-cls-seq-class":RobertaForSequenceClassification,
    "roberta-avg-seq-class":RobertaForSequenceClassificationAvg,
    "roberta":RobertaModel,
    "t5":T5Model,
    "t5_text_to_text":T5ForConditionalGeneration,
    "t5_enc-avg-seq-class":T5EncoderForSequenceClassificationAvg,
    "t5_enc-cls-seq-class":T5EncoderForSequenceClassificationCLS,
    "gpt2":GPT2Model,
    "gpt2-avg-seq-class":GPT2ForSequenceClassificationAvg,
    "gpt2-cls-seq-class":GPT2ForSequenceClassification,
    "bert-cls-seq-class":BertForSequenceClassification,
    "bert-avg-seq-class":BertForSequenceClassificationAvg
}

TOKENIZERS = {
    "roberta": RobertaTokenizer,
    "roberta-cls-seq-class":RobertaTokenizer,
    "roberta-avg-seq-class":RobertaTokenizer,
    "gpt2":GPT2Tokenizer,
    "gpt2-cls-seq-class":GPT2Tokenizer,
    "gpt2-avg-seq-class":GPT2Tokenizer,
    "t5":T5Tokenizer,
    "t5_text_to_text":T5Tokenizer,
    "t5_enc-avg-seq-class":T5Tokenizer,
    "t5_enc-cls-seq-class":T5Tokenizer,
    "bert-cls-seq-class": BertTokenizer,
    "bert-avg-seq-class":BertTokenizer
}