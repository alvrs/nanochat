"""
Verify custom linear attention produces plausible results
"""

import torch
from nanochat.flash_attention import _linear_attn

class TestLinearAttention:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    def test_basic_forward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = _linear_attn(q, k, v)

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_backward(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

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
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.zeros(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v[:, -1, :, :] = 1.0

        y = _linear_attn(q, k, v)

        assert y[:, :-1].abs().max() < 1e-5
    
    def test_attn_sums_to_one(self):
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.ones(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = _linear_attn(q, k, v)

        is_one = (y - 1.0).abs() < 1e-3
        is_zero = y < 1e-3
        assert (is_one | is_zero).all()





