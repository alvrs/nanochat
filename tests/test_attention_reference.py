"""
Verify that SDPA and custom softmax reference implementation produce identical results

Run: python -m pytest tests/test_attention_reference.py -v -s
"""

import torch
import torch.nn.functional as F
import pytest
from nanochat.flash_attention import _reference_softmax_attn 

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




