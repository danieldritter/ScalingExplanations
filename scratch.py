from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer , AutoConfig, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer
import os 
from dataset_wrappers.multinli import MultiNLIDataset

if __name__ == "__main__":
    dataset = MultiNLIDataset(num_samples=500,add_ground_truth_attributions=True, shuffle=True)
    print(dataset.train_dataset[:10])