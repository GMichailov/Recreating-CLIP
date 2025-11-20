from models.text_encoders import ClipDistilBert, ClipMiniLM
from models.vision_encoders import ClipVitTiny, ClipVitSmall
import torch
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self, vision_encoder: str, text_encoder: str, projection_dim=256, temperature_init=0.07) -> None:
        super().__init__()
        if text_encoder == "bert":
            self.text_encoder = ClipDistilBert(projection_dim)
        else:
            self.text_encoder = ClipMiniLM(projection_dim)
        if vision_encoder == "tiny":
            self.vision_encoder = ClipVitTiny(projection_dim)
        else:
            self.vision_encoder = ClipVitSmall(projection_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature_init)))

    def training_vit(self, train: bool):
        self.vision_encoder.train(train)
        for p in self.vision_encoder.parameters():
            p.requires_grad = train

    def training_text_encoder(self, train: bool):
        self.text_encoder.train(train)
        for p in self.text_encoder.parameters():
            p.requires_grad = train

    def forward(self, batched_image_tensors, batched_tokenized_text):
        batch_image_embeddings = self.vision_encoder(batched_image_tensors)
        batch_text_embeddings = self.text_encoder(**batched_tokenized_text)
        return batch_image_embeddings, batch_text_embeddings, self.logit_scale.exp()