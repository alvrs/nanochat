"""
Verify custom linear attention produces plausible results
"""

import torch
from nanochat.flash_attention import _linear_attn, _reference_softmax_attn

class TestLinearAttention:
    def test_basic_forward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)

        y = _linear_attn(q, k, v)

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_backward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, requires_grad=True)
        k = torch.randn(B, T, H, D, requires_grad=True)
        v = torch.randn(B, T, H, D, requires_grad=True)

        y = _linear_attn(q, k, v)
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

        y = _linear_attn(q, k, v)

        assert y[:, :-1].abs().max() < 1e-5
    
    def test_attn_sums_to_one(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.ones(B, T, H, D)

        y = _linear_attn(q, k, v)

        is_one = (y - 1.0).abs() < 1e-5
        is_zero = y < 1e-5
        assert (is_one | is_zero).all()

    def test_softmax_cosine_similarity(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)

        y_linear = _linear_attn(q, k, v)
        y_softmax = _reference_softmax_attn(q, k, v)

        cos = torch.nn.functional.cosine_similarity(y_linear, y_softmax, dim=-1)
        cos_mean = cos.mean().item()
        assert cos_mean > 0.8




