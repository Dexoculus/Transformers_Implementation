import torch
import torch.nn as nn
from torch import Tensor

from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, 
                 emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # batch, feature, height, width -> batch, (height * width), feature
            Rearrange('b e (h) (w) -> b (h w) e'))
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # number of tokens(NoT), features -> batch, NoT, feature
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 128, n_classes: int = 10):  # Classification for 10 class
        super().__init__(
            #
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 1,  # MNIST: Grayscale image
                patch_size: int = 4,
                emb_size: int = 12,
                img_size: int = 28,
                depth: int = 2,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
