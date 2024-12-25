import math

import torch
import torch.nn as nn

"""
Since the Transformer contains no recurrence and convolution, 
in order for the model to make use of the order of the sequence,
some information about the relative or absolute position of the tokens in sequence.
"""

class SinusoidalEncoding(nn.Module):
    """
    Use sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos/10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})
        where pos is the position and i is the dimenstion.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model (int): Embedding dimension (must match model embedding size)
            max_len (int): Maximum sequence length
        """
        super(SinusoidalEncoding, self).__init__()

        # Precompute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as a buffer to prevent being treated as a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Positional encoding added to the input
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    
class LearnableEncoding(nn.Module):
    """
    About each position $pos$, Generates Learnable embedding vector $P(pos)$.
        x_{pos}^{encoded} = x_{pos} + P(pos)
    """
    def __init__(self, d_model: int, max_len: int):
        """
        Args:
            d_model: Embedding dimension (must match model embedding size)
            max_len (int): Maximum sequence length
        """
        super(LearnableEncoding, self).__init__()
        self.pos_encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor: Positional encoding added to the input
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_encoding = self.pos_encoding(positions)

        return x + pos_encoding