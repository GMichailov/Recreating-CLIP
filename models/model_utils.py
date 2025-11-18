from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch.nn as nn

def get_text_transformer_model(name, attn_implementation="sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation=attn_implementation)
    return tokenizer, model

def get_vision_transformer_model(name, attn_implementation="sdpa"):
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation=attn_implementation)
    return processor, model

def he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)