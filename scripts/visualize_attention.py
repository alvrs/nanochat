"""
Visualize attention maps during generation to diagnose repetition.

Usage:
    python -m scripts.visualize_attention --prompt "The capital of France is"
    python -m scripts.visualize_attention --model-tag d12-linear-poly-v5 --prompt "The capital of France is"
"""

import argparse
import base64
import io
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanochat.checkpoint_manager import load_model
from nanochat import gpt as gpt_module


def show_in_terminal(fig):
    """Display a matplotlib figure inline using iTerm2's image protocol."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    # iTerm2 inline image protocol
    sys.stdout.write(f"\033]1337;File=inline=1:{data}\a\n")
    sys.stdout.flush()


def capture_attention_maps(model, input_ids):
    """Forward pass with monkey-patched linear_attn to capture attention weights."""
    attn_maps = {}
    layer_counter = [0]

    # Must patch in gpt module's namespace since it uses `from ... import linear_attn`
    original_fn = gpt_module.linear_attn

    def capturing_linear_attn(q, k, v, poly_coeffs):
        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        dk_scale = q_t.size(-1) ** -0.5
        attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * dk_scale
        T = q_t.size(2)
        mask = torch.triu(torch.ones(T, T, device=q_t.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, 0)
        attn = torch.relu(attn)

        coeffs = poly_coeffs.abs().to(attn.dtype).view(2, 1, -1, 1, 1)
        attn = coeffs[0] * attn + attn.pow(2) + coeffs[1] * attn.pow(4)

        sums = torch.sum(attn, dim=-1, keepdim=True)
        attn = attn / sums.clamp(min=1e-6)

        attn_maps[layer_counter[0]] = attn.detach().cpu().float()  # (B, H, T, T)
        layer_counter[0] += 1

        y = torch.matmul(attn, v_t)
        return y.transpose(1, 2)

    gpt_module.linear_attn = capturing_linear_attn
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        gpt_module.linear_attn = original_fn

    return attn_maps


def plot_attention_heatmaps(attn_maps, token_labels):
    """Full T x T attention heatmaps for selected layers."""
    n_layers = len(attn_maps)
    if n_layers <= 4:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]

    fig, axes = plt.subplots(1, len(layer_indices), figsize=(6 * len(layer_indices), 5))
    if len(layer_indices) == 1:
        axes = [axes]

    for ax, li in zip(axes, layer_indices):
        attn = attn_maps[li][0]  # (H, T, T), batch 0
        attn_avg = attn.mean(dim=0).numpy()

        im = ax.imshow(attn_avg, cmap="viridis", aspect="auto")
        ax.set_title(f"Layer {li}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

        if len(token_labels) <= 32:
            ax.set_xticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Attention maps (averaged over heads)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_last_token_attention(attn_maps, token_labels):
    """Where the last token attends — key diagnostic for repetition."""
    n_layers = len(attn_maps)
    T = len(token_labels)

    fig, axes = plt.subplots(n_layers, 1, figsize=(max(8, T * 0.4), 2 * n_layers))
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        attn = attn_maps[li][0]  # (H, T, T)
        last_row = attn[:, -1, :].numpy()  # (H, T)

        ax = axes[li]
        im = ax.imshow(last_row, cmap="hot", aspect="auto", vmin=0)
        ax.set_ylabel(f"L{li}\nheads", fontsize=8)
        if li == n_layers - 1:
            ax.set_xlabel("Key position")
            if T <= 32:
                ax.set_xticks(range(T))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.suptitle("Last token attention distribution per layer & head", fontsize=12)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. d12-linear-poly-v5). Default: auto-detect largest.")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default=None, help="Save plots to this path prefix instead of displaying inline")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, meta_data = load_model("base", device=torch.device(args.device), phase="eval", model_tag=args.model_tag)
    print(f"Loaded model: {args.model_tag or 'auto-detected'}")

    # Tokenize and generate
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt!r}")

    all_tokens = list(prompt_tokens)
    for token in model.generate(prompt_tokens, max_tokens=args.max_tokens, temperature=0):
        all_tokens.append(token)
        print(f"  Generated: {tokenizer.decode([token])!r}")

    print(f"\nFull output: {tokenizer.decode(all_tokens)!r}")

    # Capture attention maps on the full generated sequence
    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=args.device)
    attn_maps = capture_attention_maps(model, input_ids)
    print(f"Captured {len(attn_maps)} layers")

    # Build token labels
    token_labels = [tokenizer.decode([t])[:12] for t in all_tokens]

    # Plot and display
    fig1 = plot_attention_heatmaps(attn_maps, token_labels)
    fig2 = plot_last_token_attention(attn_maps, token_labels)

    if args.save:
        fig1.savefig(f"{args.save}_heatmaps.png", dpi=150, bbox_inches="tight")
        fig2.savefig(f"{args.save}_last_token.png", dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}_heatmaps.png and {args.save}_last_token.png")
    else:
        show_in_terminal(fig1)
        show_in_terminal(fig2)

    plt.close("all")


if __name__ == "__main__":
    main()
