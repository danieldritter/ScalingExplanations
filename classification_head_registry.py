from custom_models.classification_heads import BERTAvgPoolingClassificationHead, BERTCLSClassificationHead, T5AvgPoolingClassificationHead, T5CLSClassificationHead, GPT2AvgPoolingClassificationHead, GPT2CLSClassificationHead
CLASSIFICATION_HEADS = {
    "bert-avg-pool":BERTAvgPoolingClassificationHead,
    "bert-cls":BERTCLSClassificationHead,
    "t5-avg-pool":T5AvgPoolingClassificationHead,
    "t5-cls":T5CLSClassificationHead,
    "gpt2-avg-pool":GPT2AvgPoolingClassificationHead,
    "gpt2-cls":GPT2CLSClassificationHead
}