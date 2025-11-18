import timm
import torch.nn as nn
from models.model_utils import he_init

class ClipVitTiny(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = timm.create_model(
            "vit_tiny_patch16_224.augreg_in21k",
            pretrained=True,
            num_classes=0     # removes classifier, outputs CLS token
        )
        self.final_proj = nn.Linear(192, output_dim)
        he_init(self.final_proj)

    def forward(self, x):
        features = self.model(x)
        return self.final_proj(features)


class ClipVitSmall(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224.augreg_in21k",
            pretrained=True,
            num_classes=0
        )
        self.final_proj = nn.Linear(384, output_dim)
        he_init(self.final_proj)

    def forward(self, x):
        features = self.model(x)
        return self.final_proj(features)
