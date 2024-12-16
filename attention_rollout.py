import torch

def attention_rollout(attention_matrices, residual_connection=True):
    """
    Attention Rollout을 계산하는 함수.

    Args:
        attention_matrices (list of torch.Tensor): 
            각 레이어의 Attention Matrix 리스트. 
            각 Attention Matrix의 크기는 (batch_size, num_heads, seq_len, seq_len).
        residual_connection (bool): 
            잔차 연결(Residual Connection)을 고려할지 여부. 기본값은 True.

    Returns:
        torch.Tensor: Attention Rollout 결과 (batch_size, seq_len, seq_len).
    """
    # Identity Matrix 추가를 위한 설정
    num_layers = len(attention_matrices)
    seq_len = attention_matrices[0].size(-1)

    # 초기 Identity Matrix
    rollout_matrix = torch.eye(seq_len).to(attention_matrices[0].device)

    for layer_attention in attention_matrices:
        # Multi-head Attention 평균 계산
        average_attention = layer_attention.mean(dim=1)  # (batch_size, seq_len, seq_len)

        # Residual Connection 적용
        if residual_connection:
            average_attention = 0.5 * average_attention + 0.5 * torch.eye(seq_len).to(average_attention.device)

        # 현재 레이어의 Attention을 누적
        rollout_matrix = torch.matmul(average_attention, rollout_matrix)

    return rollout_matrix
