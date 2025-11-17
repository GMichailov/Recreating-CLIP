import torch.nn as nn
from models.model_utils import get_vision_transformer_model

class ClipVitTiny(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = get_vision_transformer_model("timm/vit_tiny_patch16_224.augreg_in21k")
        self.final_proj = nn.Linear(192, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.final_proj(x)
        return x
    
class ClipVitSmall(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = get_vision_transformer_model("timm/vit_small_patch16_224.augreg_in21k")
        self.final_proj = nn.Linear(384, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.final_proj(x)
        return x