from abc import ABC, abstractmethod 
import torch 

class Explainer(ABC):

    def __init__(self, model: torch.nn.Module):
        self.model = model 
    
    @abstractmethod 
    def get_explanations(self,inputs, seq2seq=False):
        pass 

    @abstractmethod
    def visualize_explanations(self, explanations):
        pass 