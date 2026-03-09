import torch
import torch.nn.functional as F

# =============================================================================
# Reference softmax attention 
# =============================================================================
def _reference_softmax_attn(q, k, v):
    """
    Softmax attention for reference
    
    Args:
        q, k, v: Tensors of shape (B, T, H, D)

    Returns:
        Output tensor of shape (B, T, H, D)
    """

    # Input arrives in shape 
    #   B=Batch size,
    #   T=Time (number of tokens),
    #   H=Heads (number of attention heads),
    #   D=Dimension (dimension per attention head)
    # because it's optimized for flash attention, which uses streaming to avoid
    # materializing the full TxT matrix. Here we'll use the naive approach of computing
    # each head independently, so we need to swap the dimensions of the input tensors
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) # (B, T, H, D) -> (B, H, T, D)

    # Scaling factor from original "Attention is all you need" paper
    # Without it, dot products of q @ k^T would grow in magnitude proportionally to D
    # which would result in the softmax producing mostly ~0 or ~1 values.
    # Scaling it down keeps softmax in a more useful range
    scale = q.size(-1) ** -0.5

    # Compute attention scores -> (B, H, T, T)
    # Per head, every row corresponds to a query and every column to a key
    attn = torch.matmul(q, k.transpose(-2,-1)) * scale

    # Create causal mask
    T = q.size(2) # number of tokens
    TT = torch.ones(T, T, device=q.device, dtype=torch.bool) # TxT matrix filled with True
    mask = torch.triu(TT, diagonal=1) # main diagonal and everything below becomes false

    # Apply causal mask (every "future query" result becomes -inf)
    attn = attn.masked_fill(mask, float('-inf'))

    # Apply softmax (sum of each row will be 1)
    attn = torch.softmax(attn, dim=-1)

    # Apply attention to value tensor -> (B, H, T, D)
    y = torch.matmul(attn, v)

    return y.transpose(1, 2) # -> (B, T, H, D)

def linear_attn(q, k, v, degree=2):
    # Same as _reference_softmax_attn
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) # (B, T, H, D) -> (B, H, T, D)
    scale = q.size(-1) ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    T = q.size(2)
    TT = torch.ones(T, T, device=q.device, dtype=torch.bool)
    mask = torch.triu(TT, diagonal=1)

    # Apply causal mask (every "future query" result becomes 0)
    attn = attn.masked_fill(mask, 0)

    # Apply linear scaling to achieve similar sharpening effect as softmax
    attn = torch.relu(attn).pow(degree)

    # Normalize to get rows that sum to 1
    sums = torch.sum(attn, dim=-1, keepdim=True)
    attn = attn / sums.clamp(min=1e-6)

    # Apply attention to value tensor
    y = torch.matmul(attn, v)

    return y.transpose(1,2) # (B, H, T, D) -> (B, T, H, D)
