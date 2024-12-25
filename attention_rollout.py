import torch

def attention_rollout(attention_matrices, residual_connection=True):
    """
    Calculates Attention Rollout.

    Args:
        attention_matrices (list of torch.Tensor): 
            Attention Matrix List of each layer. 
            Shape of each Attention Matrix: (batch_size, num_heads, seq_len, seq_len).
        residual_connection (bool): 
            Whether consider Residual Connection or not. Default is True.

    Returns:
        torch.Tensor: Result of Attention Rollout (batch_size, seq_len, seq_len).

    Source:
        Abnar, Samira, and Willem Zuidema.
        "Quantifying attention flow in transformers." arXiv preprint arXiv:2005.00928 (2020).
    """
    # Settings for adding Identity Matrix
    num_layers = len(attention_matrices)
    seq_len = attention_matrices[0].size(-1)

    # Initial Identity Matrix
    rollout_matrix = torch.eye(seq_len).to(attention_matrices[0].device)

    for layer_attention in attention_matrices:
        # Calculate mean of Multi-head Attention
        average_attention = layer_attention.mean(dim=1)  # (batch_size, seq_len, seq_len)

        # Residual Connection
        if residual_connection:
            average_attention = 0.5 * average_attention + 0.5 * torch.eye(seq_len).to(average_attention.device)

        # Accumulate the attention of the current layer
        rollout_matrix = torch.matmul(average_attention, rollout_matrix)

    return rollout_matrix
