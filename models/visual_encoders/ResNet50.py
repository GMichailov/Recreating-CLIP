import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

"""
Explanation of changes.

ResNet50 downsamples through strided convs (stride 2) and maxpool (stride 2) which causes aliasing (high-
-frequency information folds into low frequencies which leads to worse invariance and performance).
Zhang et al., 2019 adds a low-pass blur filter BEFORE downsampling. It takes an average of the filter values.
Example: 
[1, 2, 3]       [AVG(1+2+4+5), AVG(2+3+5+6)]     [3, 4]
[4, 5, 6]   ->  [AVG(4+5+7+8), AVG(5+6+8+9)] ->  [6, 7]
[7, 8, 9]

They also replaced global average pooling with attention pooling.
1. flatten spatial grid from (Batch Size, Channels, Height, Width) to (Batch Size, Height x Width, Channels)
2. Add learnable CLS embedding token (Classification Token) to the spatial grid by concatenating.
3. Perform Self-Attention
4. Use the CLS Token output as the pooled representation.
"""

class BlurPool(nn.Module):
    def __init__(self, channels, filter_size=2, stride=2) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.kernel = torch.ones(1,1,2,2) / 4 # 1 output channel, 1 input channel, 2x2 kernel that takes average of the 4 elements.
        self.register_buffer("kernel", self.kernel.repeat(channels, 1, 1, 1)) # Save in state dict and not learnable. Repeat tells it to be applied to each channel.

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=0, groups=x.size(1)) # Each group needs to be based on channels. Appply kernel then 2x2 downsample.
    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.d_head = embed_dim // num_heads
        # Positional embedding is for Height * Width + 1 (CLS) tokens
        self.positional_embedding = nn.Parameter(
            torch.randn(1, spacial_dim * spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height*width)
        x.permute(0, 2, 1) # To go from (BS, C, H*W) -> (BS, H*W, C)
        cls = x.mean(dim=1, keepdim=True)
        x = torch.cat([cls, x], dim=1) # Prepend the mean pooled cls token
        x = x + self.positional_embedding[:, :x.shape[1]].T # Apply positional encoding
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, 
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        out = self.c_proj(attn_out)
        return out[:, 0] # Return only the CLS token output (Batch Size, Channels).

        