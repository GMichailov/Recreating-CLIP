import torch.nn as nn
from models.model_utils import get_text_transformer_model

class ClipDistilBert(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tokenizer, self.model = get_text_transformer_model("distilbert/distilbert-base-uncased")
        self.final_proj = nn.Linear(768, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.final_proj(x)
        return x
    
class ClipMiniLM(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tokenizer, self.model = get_text_transformer_model("sentence-transformers/all-MiniLM-L6-v2")
        self.final_proj = nn.Linear(384, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.final_proj(x)
        return x