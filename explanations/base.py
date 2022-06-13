from abc import ABC, abstractmethod 
import torch 

class Explainer(ABC):

    def __init__(self, model: torch.nn.Module):
        self.model = model 
    
    @abstractmethod 
    def get_explanations(self,inputs):
        pass 