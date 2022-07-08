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

HEAD_WEIGHTS = {
    "gpt2":["transformer.ln_f","score"],
    "gpt2-cls-seq-class":["transformer.ln_f","score"],
    "gpt2-cls-avg-class":["transformer.ln_f","score"],
    "roberta": ["classfifier"],
    "roberta-cls-seq-class":["classifier"],
    "roberta-avg-seq-class":["classifier"],
    "t5":["encoder.final_layer_norm", "classifier"],
    "t5_enc-avg-seq-class":["encoder.final_layer_norm", "classifier"],
    "t5_enc-cls-seq-class":["encoder.final_layer_norm", "classifier"],
    "bert-cls-seq-class":["bert.pooler", "classifier"],
    "bert-avg-seq-class":["bert.pooler", "classifier"],

}

EMBEDDING_WEIGHTS = {
    "gpt2":["transformer.wte","transformer.wpe"],
    "gpt2-cls-seq-class":["transformer.wte","transformer.wpe"],
    "gpt2-cls-avg-class":["transformer.wte","transformer.wpe"],
    "roberta": ["roberta.embeddings"],
    "roberta-cls-seq-class":["roberta.embeddings"],
    "roberta-avg-seq-class":["roberta.embeddings"],
    "t5":["shared"],
    "t5_enc-avg-seq-class":["shared"],
    "t5_enc-cls-seq-class":["shared"],
    "bert-cls-seq-class":["bert.embeddings"],
    "bert-avg-seq-class":["bert.embeddings"]
}