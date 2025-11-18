import torch.nn as nn
from models.model_utils import get_text_transformer_model, he_init

class ClipDistilBert(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tokenizer, self.model = get_text_transformer_model("distilbert/distilbert-base-uncased")
        self.final_proj = nn.Linear(768, output_dim)
        he_init(self.final_proj)

    def forward(self, **x):
        outputs = self.model(**x)
        cls = outputs.last_hidden_state[:, 0]
        return self.final_proj(cls)
    
class ClipMiniLM(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tokenizer, self.model = get_text_transformer_model("sentence-transformers/all-MiniLM-L6-v2")
        self.final_proj = nn.Linear(384, output_dim)
        he_init(self.final_proj)

    def forward(self, **x):
        outputs = self.model(**x)
        cls = outputs.last_hidden_state[:, 0]
        return self.final_proj(cls)