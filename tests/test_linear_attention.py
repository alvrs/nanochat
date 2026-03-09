"""
Verify custom linear attention produces plausible results
"""

import torch
import torch.nn.functional as F
from nanochat.linear_attention import linear_attn, _reference_softmax_attn
class TestLinearAttention:
    def test_basic_forward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        scale = torch.ones(H)

        y = linear_attn(q, k, v, scale)

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_backward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, requires_grad=True)
        k = torch.randn(B, T, H, D, requires_grad=True)
        v = torch.randn(B, T, H, D, requires_grad=True)
        scale = torch.ones(H, requires_grad=True)

        y = linear_attn(q, k, v, scale)
        loss = y.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q).any()
        assert not torch.isinf(q).any()
    
    def test_causal_mask(self):
        "future tokens should not leak into past positions"
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.zeros(B, T, H, D)
        v[:, -1, :, :] = 1.0
        scale = torch.ones(H)

        y = linear_attn(q, k, v, scale)

        assert y[:, :-1].abs().max() < 1e-5
    
    def test_attn_sums_to_one(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.ones(B, T, H, D)
        scale = torch.ones(H)

        y = linear_attn(q, k, v, scale)

        is_one = (y - 1.0).abs() < 1e-5
        is_zero = y < 1e-5
        assert (is_one | is_zero).all()

    def test_softmax_cosine_similarity(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        scale = torch.ones(H)

        y_linear = linear_attn(q, k, v, scale)
        y_softmax = _reference_softmax_attn(q, k, v)

        cos = torch.nn.functional.cosine_similarity(y_linear, y_softmax, dim=-1)
        cos_mean = cos.mean().item()
        assert cos_mean > 0.8
    
    def test_attention_pattern_similarity(self):
        B, T, H, D = 2, 64, 4, 64
        q = torch.rand(B, T, H, D)
        k = torch.randn(B, T, H, D)

        # v = identity -> output is attention scores
        v = torch.eye(T, D).unsqueeze(1).expand(B, T, H, D)
        scale = torch.ones(H)

        attn_linear = linear_attn(q, k, v, scale)
        attn_softmax = _reference_softmax_attn(q, k, v)

        k = 3
        top_linear = attn_linear.topk(k).indices
        top_softmax = attn_softmax.topk(k).indices
        matches = top_linear == top_softmax
        num_matches = matches.sum(dim=-1)
        mean_matches = num_matches.mean(dtype=torch.float32).item()

        assert mean_matches > k - 0.1

def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff

class TestReference:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    def test_basic_forward(self):
        "Test forward pass produces valid output"
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = _reference_softmax_attn(q, k, v)
        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
    
    def test_backward(self):
        "Test gradients flow through"
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = _reference_softmax_attn(q, k, v)
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
    
    def test_basic_causal(self):
        "Test basic causal attention"
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_ours = _reference_softmax_attn(q, k, v)

        # F.scaled_dot_product_attention expects (B, H, T, D) shape
        y_sdpa = F.scaled_dot_product_attention(
            q.transpose(1,2),
            k.transpose(1,2),
            v.transpose(1,2),
            is_causal=True
        ).transpose(1,2) # transpose back to (B, T, H, D)

        max_diff, mean_diff = assert_close(y_ours, y_sdpa, "basic_causal")
        print(f"basic causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
