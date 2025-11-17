import torch
import torch.nn as nn
import torch.nn.functional as F

def use_flash_attention(model):
    pass

def _replace_mha_with_scaled_dot_product_attention(mha: nn.MultiheadAttention):
    def flash_forward(Q, K, V, key_padding_mask=None, need_weights=False, attn_mask=None):
        batch_size, seq_len, d_model = Q.shape
        

def he_init(layer : nn.Module):
    pass

def kaiming_init(layer):
    pass