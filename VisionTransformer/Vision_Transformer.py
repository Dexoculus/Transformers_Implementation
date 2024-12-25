import torch
import torch.nn as nn
from torch import Tensor

from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from Transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, 
                 d_model: int = 512, img_size: int = 224):
        self.patch_size = patch_size
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            # batch, feature, height, width -> batch, (height * width), feature
            Rearrange('b e (h) (w) -> b (h w) e'))
        self.cls_token = nn.Parameter(torch.randn(1,1, d_model))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, d_model))
        
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
    def __init__(self, d_model: int = 512, n_classes: int = 10):  # Classification for 10 class
        super(ClassificationHead, self).__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, n_classes))

class VisionTransformer(nn.Sequential):
    def __init__(self,     
                 in_channels: int = 3,
                 patch_size: int = 16,
                 img_size: int = 128,
                 depth: int = 3,
                 d_model: int = 16,
                 num_heads:int = 8,
                 drop_mha:float = 0.,
                 dropout:float = 0.,
                 drop_ffn:float = 0.,
                 expansion:int = 4,
                 n_classes: int = 10,
                 activation=None,
                 glu=False):
        super(VisionTransformer, self).__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, d_model, img_size)
        self.encoders = nn.ModuleList([
            TransformerEncoder(d_model,
                               num_heads,
                               drop_mha,
                               dropout,
                               drop_ffn,
                               expansion,
                               activation,
                               glu) for _ in range(depth)
        ])
        self.cls_head = ClassificationHead(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        encoder_output = x
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output)

        output = self.cls_head(encoder_output)

        return output
    
"""
from torchinfo import summary
model = VisionTransformer(in_channels=3, img_size=128, depth=5, patch_size=4)
summary(model, input_size=(32, 3, 128, 128))

=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
VisionTransformer                                       [32, 10]                  --
├─PatchEmbedding: 1-1                                   [32, 1025, 16]            16,416
│    └─Sequential: 2-1                                  [32, 1024, 16]            --
│    │    └─Conv2d: 3-1                                 [32, 16, 32, 32]          784
│    │    └─Rearrange: 3-2                              [32, 1024, 16]            --
├─ModuleList: 1-2                                       --                        --
│    └─TransformerEncoder: 2-2                          [32, 1025, 16]            --
│    │    └─ResidualAdd: 3-3                            [32, 1025, 16]            1,120
│    │    └─ResidualAdd: 3-4                            [32, 1025, 16]            2,160
│    └─TransformerEncoder: 2-3                          [32, 1025, 16]            --
│    │    └─ResidualAdd: 3-5                            [32, 1025, 16]            1,120
│    │    └─ResidualAdd: 3-6                            [32, 1025, 16]            2,160
│    └─TransformerEncoder: 2-4                          [32, 1025, 16]            --
│    │    └─ResidualAdd: 3-7                            [32, 1025, 16]            1,120
│    │    └─ResidualAdd: 3-8                            [32, 1025, 16]            2,160
│    └─TransformerEncoder: 2-5                          [32, 1025, 16]            --
│    │    └─ResidualAdd: 3-9                            [32, 1025, 16]            1,120
│    │    └─ResidualAdd: 3-10                           [32, 1025, 16]            2,160
│    └─TransformerEncoder: 2-6                          [32, 1025, 16]            --
│    │    └─ResidualAdd: 3-11                           [32, 1025, 16]            1,120
│    │    └─ResidualAdd: 3-12                           [32, 1025, 16]            2,160
├─ClassificationHead: 1-3                               [32, 10]                  --
│    └─Reduce: 2-7                                      [32, 16]                  --
│    └─LayerNorm: 2-8                                   [32, 16]                  32
│    └─Linear: 2-9                                      [32, 10]                  170
=========================================================================================================
Total params: 33,802
Trainable params: 33,802
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 26.22
=========================================================================================================
Input size (MB): 6.29
Forward/backward pass size (MB): 235.11
Params size (MB): 0.07
Estimated Total Size (MB): 241.47
=========================================================================================================
"""