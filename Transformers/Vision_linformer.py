import torch
import torch.nn as nn
from linformer import default, init_, FeedForward, GELU
from Vision_Transformer import PatchEmbedding

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'Embedding dimension must be divisible by number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None, mask=None, **kwargs):
        b, n, d = x.shape
        h = self.heads
        d_h = self.dim_head
        k = self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len <= self.seq_len, f'Key/Value sequence length must be less than or equal to {self.seq_len}, but got {kv_len}'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        if kv_len < self.seq_len:
            kv_projs = tuple(t[:kv_len] for t in kv_projs)

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        queries = queries.reshape(b, n, h, d_h).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, h, d_h).transpose(1, 2)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)

        if mask is not None:
            mask = mask[:, None, None, :].expand(-1, h, -1, -1)
            dots = dots.masked_fill(mask == 0, float('-inf'))

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, num_patches=196, k=64, heads=8, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__()
        self.attention = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            LinformerSelfAttention(
                dim=emb_size,
                seq_len=num_patches + 1,  # +1 for class token
                k=k,
                heads=heads,
                dropout=drop_p
            ),
            nn.Dropout(drop_p)
        ))
        self.feed_forward = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForward(
                dim=emb_size,
                mult=forward_expansion,
                dropout=forward_drop_p,
                activation=GELU
            ),
            nn.Dropout(drop_p)
        ))
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class VisionLinformer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_size=768,
                 img_size=224,
                 depth=12,
                 k=64,
                 heads=8,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.,
                 n_classes=1000,
                 **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_size=emb_size,
            img_size=img_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.transformer_encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            num_patches=num_patches,
            k=k,
            heads=heads,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p
        )
        self.cls_head = ClassificationHead(emb_size=emb_size, n_classes=n_classes)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]  # Use the class token
        x = self.cls_head(x)
        return x