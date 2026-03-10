# Experiment Log

Goal: Replace softmax attention with a linear attention variant while maintaining quality.

## Experiments

### d12-linear

- **Commit:** 64a69c3c86eaa1dbea2cfb37515762af56e3ec28
- **Change:** relu + x^2 + normalization
- **Result:** Loss 2.91, qualitatively worse than reference

### d12-linear-poly-v5

- **Commit:** 8cab967
- **Change:** relu + ax+x^2+bx^4 + normalization (with a,b learnable params per head, properly initialized to 1.0)
- **Result:** Generation degenerates into single-token repetition (e.g. "France is is is is..."). Attention maps show diffuse distributions — later layers spread weight nearly uniformly across recent tokens, losing track of the prompt. Hypothesis: the polynomial can't produce sharp enough attention to select specific positions.

### d12-linear-sqrt-v2

- **Commit:** 8684ca1
- **Change:** simpler linear attn: x^3 / sqrt(T) (no relu, no normalization), from https://arxiv.org/abs/2410.18613
- **Result:** Loss 2.86 (BPB 0.877), slightly better than poly variants. Still degenerates into repetition during generation ("France France ée ée... disease disease disease"). Attention maps show unnormalized weights with strong recency bias — last token attends almost exclusively to recent positions, creating a feedback loop that locks onto repeated tokens.

### d12-relu-sq

- **Commit:** 45707eb
- **Change:** relu(x)^2 feature map on both Q and K (from https://arxiv.org/abs/2006.16236), with row-wise normalization of the attention matrix
- **Result:** Still degenerates into single-token repetition ("the the the the...", "gold gold gold..."). The model latches onto a plausible first token and repeats it indefinitely. 

## Bug: params initialized with 0

The following experiments all had a bug where the parameters weren't actually initialized to 1, so they weights didn't "collapse" to 0 but rather were initialized with 0 and were stuck there. This was fixed in e48f8c1071577932f63127c68ac61388e02a0b63.

### d12-linear-scale

- **Commit:** f81df784770d01c32f274a70cbe6295988c96a53
- **Change:** Scale param per attention head
- **Result:** Loss 5.50, all scale params collapsed to 0

### d12-linear-2

- **Commit:** a9c19a8e6785ead2a4efe1c3afcc22bfeb252d61
- **Change:** Apply scale after relu
- **Result:** Didn't help, still collapsed to 0

### d12-linear-poly

- **Commit:** 5dcd5df82b41774847cead95065459ebb6ea3544
- **Change:** Switched to relu + ax+bx^2+cx^4 + normalization, with learnable a,b,c per head
- **Result:** Params collapsed to 0 again, loss ~5.50

### d12-linear-poly-v3

- **Commits:** 6794779 (constrain coefficients), 9808190 (move coeffs to own optimizer group)
- **Change:** Switched to ax+x^2+bx^4 with only two learnable params per head, fixing x^2; also moved coefficients to separate optimizer group
- **Result:** a,b collapsed to 0, leaving only x^2, loss back to 2.92 (matches initial x^2-only result)
