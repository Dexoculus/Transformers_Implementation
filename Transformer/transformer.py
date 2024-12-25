import torch.nn as nn

from .blocks import MultiHeadAttention, FeedForwardNet, ResidualAdd
from .pos_encodings import LearnableEncoding

"""
# About Pre-Layer Normalization (Pre-LN)
In this implementation,
we apply Pre-Layer Normalization (Pre-LN) instead of Post-Layer Normalization (Post-LN).
The following are the five main benefits of adopting Pre-Norm:

- No Warm-up Stage:
    Stable gradients eliminate the need for a learning rate warm-up stage, reducing training time.
- Simplified Hyperparameter Tuning:
    Removes the need to tune warm-up-specific parameters.
- Stable Gradients:
    Prevents gradient explosion or vanishing across layers, ensuring consistent optimization.
- Faster Convergence:
    Allows larger learning rates, speeding up training.
- Better Scalability:
    Handles deeper or larger models effectively, improving performance in large-scale tasks.

Source:
    Xiong, Ruibin, et al. "On layer normalization in the transformer architecture."
    International Conference on Machine Learning. PMLR, 2020.
"""

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model:int = 512,
                 max_len:int = 5000,
                 num_heads:int = 8,
                 drop_mha:float = 0.,
                 dropout:float = 0.,
                 drop_ffn:float = 0.,
                 expansion:int = 4,
                 num_encoders:int = 6,
                 num_decoders:int = 6,
                 activation = None,
                 glu = False):
        super(Transformer, self).__init__()
        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos = LearnableEncoding(d_model, max_len)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos = LearnableEncoding(d_model, max_len)

        self.encoders = nn.ModuleList([
            TransformerEncoder(d_model,
                               num_heads,
                               drop_mha,
                               dropout,
                               drop_ffn,
                               expansion,
                               activation,
                               glu) for _ in range(num_encoders)
        ])
        self.decoders = nn.ModuleList([
            TransformerDecoder(d_model,
                               num_heads,
                               drop_mha,
                               dropout,
                               drop_ffn,
                               expansion,
                               activation,
                               glu) for _ in range(num_decoders)
        ])
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        """
        Args:
            src (Tensor): Source input (batch_size, src_len).
            tgt (Tensor): Target input (batch_size, tgt_len).
            src_mask (Tensor): Encoder padding mask.
            tgt_mask (Tensor): Decoder mask.
            mem_mask (Tensor): Encoder-Decoder Attention mask.
        """
        # Encoder
        encoder_input = self.src_pos(self.src_embedding(src)) # (batch_size, src_len, d_model)
        encoder_output = encoder_input
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, src_mask=src_mask)

        # Decoder
        decoder_input = self.tgt_pos(self.tgt_embedding(tgt))
        decoder_output = decoder_input
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output, tgt_mask=tgt_mask, mem_mask=mem_mask)

        # Output logits
        output = self.out_linear(decoder_output)
        return output

class TransformerEncoder(nn.Module):
    """
    """
    def __init__(self,
                 d_model:int = 512,
                 num_heads:int = 8,
                 drop_mha:float = 0.,
                 dropout:float = 0.,
                 drop_ffn:float = 0.,
                 expansion:int = 4,
                 activation = None, # Default: GELU
                 glu = False):
        super(TransformerEncoder, self).__init__()
        # MultiHeadAttention + Residual
        self.mha = ResidualAdd(CustomSequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, num_heads, drop_mha),
            nn.Dropout(dropout)
        ))
        # Feed-Forward + Residual
        self.ffn = ResidualAdd(nn.Sequential(
            nn.LayerNorm(d_model),
            FeedForwardNet(d_model, expansion, drop_ffn, activation, glu),
            nn.Dropout(dropout)
        ))

    def forward(self, x, src_mask=None):
        # In encoder, Mask must be padding mask
        # x shape = (batch, seq_len, d_model)
        # mask shape = (batch, 1, 1, seq_len) or (batch, num_heads, seq_len, seq_len)

        x = self.mha(x, mask=src_mask) # Self-Attention
        x = self.ffn(x) # Feed-Forward

        return x

class TransformerDecoder(nn.Module):
    """
    """
    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 drop_mha:float = 0.,
                 dropout:float = 0.,
                 drop_ffn: float = 0.,
                 expansion: int = 4,
                 activation=None, # Default: GELU
                 glu=False):
        super(TransformerDecoder, self).__init__()
        # MultiHeadAttention + Residual
        self.masked_mha = ResidualAdd(CustomSequential(
            nn.LayerNorm(d_model),
            MultiHeadAttention(d_model, num_heads, drop_mha),
            nn.Dropout(dropout)
        ))
        # Encoder-Decoder Cross Attention (cross=True)
        self.cross_mha = ResidualAdd(CustomSequential(
            nn.LayerNorm(d_model), # Pre-LN
            MultiHeadAttention(d_model, num_heads, drop_mha, cross=True),
            nn.Dropout(dropout)
        ))
        # Feed-Forward + Residual
        self.ffn = ResidualAdd(nn.Sequential(
            nn.LayerNorm(d_model),
            FeedForwardNet(d_model, expansion, drop_ffn, activation, glu),
            nn.Dropout(dropout)
        ))

    def forward(self, x, context, tgt_mask=None, mem_mask=None):
        x = self.masked_mha(x, mask=tgt_mask) # Masked Self-Attention
        # Encoder-Decoder Cross Attention
        x = self.cross_mha(x, context, mask=mem_mask) # Encoder output to input
        x = self.ffn(x)

        return x
    
class CustomSequential(nn.Module):
    """
    A Custom Sequential class for processing kwargs in MHA.
    """
    def __init__(self, *layers):
        super(CustomSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            if isinstance(layer, MultiHeadAttention):
                x = layer(x, *args, **kwargs)
            else:
                x = layer(x)

        return x

"""
import torch
from torchinfo import summary

src_vocab_size = 10000
tgt_vocab_size = 10000
batch_size = 8
src_seq_len = 16
tgt_seq_len = 16

src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    max_len=5000,
    num_heads=8,
    num_encoders=6,
    num_decoders=6
)

summary(model, input_data=(src, tgt))

=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
Transformer                                             [8, 16, 10000]            --
├─Embedding: 1-1                                        [8, 16, 512]              5,120,000
├─LearnableEncoding: 1-2                                [8, 16, 512]              --
│    └─Embedding: 2-1                                   [1, 16, 512]              2,560,000
├─ModuleList: 1-3                                       --                        --
│    └─TransformerEncoder: 2-2                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-1                            [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-2                            [8, 16, 512]              2,100,736
│    └─TransformerEncoder: 2-3                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-3                            [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-4                            [8, 16, 512]              2,100,736
│    └─TransformerEncoder: 2-4                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-5                            [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-6                            [8, 16, 512]              2,100,736
│    └─TransformerEncoder: 2-5                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-7                            [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-8                            [8, 16, 512]              2,100,736
│    └─TransformerEncoder: 2-6                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-9                            [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-10                           [8, 16, 512]              2,100,736
│    └─TransformerEncoder: 2-7                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-11                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-12                           [8, 16, 512]              2,100,736
├─Embedding: 1-4                                        [8, 16, 512]              5,120,000
├─LearnableEncoding: 1-5                                [8, 16, 512]              --
│    └─Embedding: 2-8                                   [1, 16, 512]              2,560,000
├─ModuleList: 1-6                                       --                        --
│    └─TransformerDecoder: 2-9                          [8, 16, 512]              --
│    │    └─ResidualAdd: 3-13                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-14                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-15                           [8, 16, 512]              2,100,736
│    └─TransformerDecoder: 2-10                         [8, 16, 512]              --
│    │    └─ResidualAdd: 3-16                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-17                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-18                           [8, 16, 512]              2,100,736
│    └─TransformerDecoder: 2-11                         [8, 16, 512]              --
│    │    └─ResidualAdd: 3-19                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-20                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-21                           [8, 16, 512]              2,100,736
│    └─TransformerDecoder: 2-12                         [8, 16, 512]              --
│    │    └─ResidualAdd: 3-22                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-23                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-24                           [8, 16, 512]              2,100,736
│    └─TransformerDecoder: 2-13                         [8, 16, 512]              --
│    │    └─ResidualAdd: 3-25                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-26                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-27                           [8, 16, 512]              2,100,736
│    └─TransformerDecoder: 2-14                         [8, 16, 512]              --
│    │    └─ResidualAdd: 3-28                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-29                           [8, 16, 512]              1,051,648
│    │    └─ResidualAdd: 3-30                           [8, 16, 512]              2,100,736
├─Linear: 1-7                                           [8, 16, 10000]            5,130,000
=========================================================================================================
Total params: 64,628,496
Trainable params: 64,628,496
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 481.19
=========================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 96.35
Params size (MB): 258.51
Estimated Total Size (MB): 354.87
=========================================================================================================
"""