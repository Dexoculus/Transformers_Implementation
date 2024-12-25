import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def default(val, default):
    return val if val is not None else default

class MultiHeadAttention(nn.Module):
    """
    Query, Key, Value:
    For embedded input X (batch, seq_len, d_model), 
    Query, Key, Value (batch, seq_len, d_k) for attention is:
        Q = X @ W_Q, K = X @ W_K, V = X @ W_V
    Where W_Q, W_K, W_V are weight metrix (d_model, d_model). (Note: d_model = emb_size)
    
    We can calculate Query, Key, Value at once, by fusing weight matrices.
    W_QKV (d_model, 3*d_model) matrix:
        W_QKV = [W_Q | W_K | W_V]
        X @ W_QKV = [X @ W_Q | X @ W_K | X @ W_V] = [Q | K | V]

    So, We can get Query, Key, Value by split the tensor [X @ W_QKV] (batch, seq_len, 3*d_model)
    (batch, seq_len, 3*d_model) -> ((qkv=3), batch, num_head, seq_len, d_model)

    Calcultating Attention:
    softmax(Q K^T/sqrt(d_k))
    """
    def __init__(self, d_model: int, num_heads: int, drop_mha: float = 0, cross: bool = False):
        """
        Initialize MHA class.

        Args:
            d_model (int): Embedding Dimension for input sequence. default is 768.
            num_heads (int): Number of Heads for Mulithead Attention.
            drop_mha (float): Rate for dropout.
            cross (bool): Whether the attentions is cross or not.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.head_dim = d_model // num_heads

        if not cross: # fuse the queries, keys and values in one matrix
            self.qkv = nn.Linear(d_model, d_model * 3)
        else: # fuse the keys and values in one matrix
            self.q = nn.Linear(d_model, d_model)
            self.kv = nn.Linear(d_model, d_model * 2)

        self.projection = nn.Linear(d_model, d_model)
        self.att_drop = nn.Dropout(drop_mha)

    def forward(self, x, context=None, mask=None):
        """
        Scaled Dot-Product Multi-Head Attention Forward Pass.

        Einstein Summation Convention:
        - Dot Production between queries and keys
            Dimension of queries, keys: (batch_size, num_heads, seq_len, d_model)
            -> Q K^T: (batch_size, num_heads, seq_len(Q), seq_len(K))

            math:
                sum_{d_model} queries[b, h, seq_q, d] \times keys[b, h, seq_k, d]

        - Producting Attention Score to Values
            Dimension of attention score: (batch_size, num_heads, seq_len(Q), seq_len(K))
            Dimension of Value: (batch_size, num_heads, seq_len(V), d_model)
            -> (batch_size, num_heads, seq_len(Q), d_model)

            math:
                sum_{seq_len} attn[b, h, a, l] \times values[b, h, l, v]
        """
        if context is None:
            return self.self_attention(x, mask)
        else:
            return self.cross_attention(x, context, mask)

    def self_attention(self, x, mask):
        # split queries, keys and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        return self._compute_attention(queries, keys, values, mask)

    def cross_attention(self, x, context, mask):
        queries = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        # split keys and values in num_heads
        kv = rearrange(self.kv(context), "b n (kv h d) -> kv b h n d", kv=2, h=self.num_heads)
        keys, values = kv[0], kv[1]
        return self._compute_attention(queries, keys, values, mask)

    def _compute_attention(self, queries, keys, values, mask):
        # sum up over the last axis
        dot_product = torch.einsum("bhqd, bhkd -> bhqk", queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            dot_product = dot_product.masked_fill(~mask, fill_value)
        att = F.softmax(dot_product / (self.head_dim ** 0.5), dim=-1) # Scaled Dot Product
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum("bhqk, bhkd -> bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)

class FeedForwardNet(nn.Module):
    """
    A Position-wise Feed-Forward Networks.
    Applied to each position separately and identically.

    d_ff = d_model * expension

    Math:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
               = ReLU(xW_1 + b_1)W_2 + b_2

    # Activation Function
    In this implementation, We apply GELU (Gaussian Error Linear Unit)
    as a default activation function, instead of ReLU.

    Math:
        FFN(x) = ((xW_1 + b_1) \dot \phi(xW_1 + b_1))W_2 + b_2
               = GELU(xW_1 + b_1)W_2 + b_2
        where \phi = CDF (cumulative distribution function)

    Source:
        > "For the activation function, we used the Gaussian Error Linear Unit (GELU)"
            Radford, Alec. "Improving language understanding by generative pre-training." (2018).
        > "we use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT."
            Kenton, Jacob Devlin Ming-Wei Chang, and Lee Kristina Toutanova.
            "Bert: Pre-training of deep bidirectional transformers for language understanding."
            Proceedings of naacL-HLT. Vol. 1. 2019.

    # Applying Gated Linear Unit (GLU)
    Gated Linear Unit (GLU) is a structure that dynamically selects input information through a gating mechanism.
    GLU is also used in the Feed-Forward Network of Transformers to improve efficiency and performance.

    math:
        GLU(x) = (x W_1 + b_1) \otimes \sigma (x W_2 + b_2)
        where W_1 and W_2 are weight matrices and b_1 and b_2 are biases.
        Also, \sigma is activation function.

    Source:
        Dauphin, Yann N., et al. "Language modeling with gated convolutional networks."
        International conference on machine learning. PMLR, 2017.
        Shazeer, Noam. "Glu variants improve transformer." arXiv preprint arXiv:2002.05202 (2020).
    """
    def __init__(self, d_model: int, expansion: int = 4, drop_ffn: float = 0., activation = None, glu = False):
        super(FeedForwardNet, self).__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.sigma = activation()
        if self.glu:
            self.w1 = nn.Linear(d_model, d_model*expansion*2)
        else:
            self.w1 = nn.Linear(d_model, d_model*expansion)

        self.dropout = nn.Dropout(drop_ffn)
        self.w2 = nn.Linear(d_model*expansion, d_model)

    def forward(self, x):
        if self.glu:
            x, v = rearrange(self.w1(x), '... (two d) -> ... d two', two=2).unbind(dim=-1)
            x = self.sigma(x) * v
        else:
            x = self.w1(x)
            x = self.sigma(x)

        x = self.dropout(x)
        x = self.w2(x)
        return x
    
class ResidualAdd(nn.Module):
    """
    A Block for Residual Addition.
    """
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        res = x
        x = self.fn(x, *args, **kwargs)
        x += res
        return x