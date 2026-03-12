# Experiment Log

Goal: Replace softmax attention with a linear attention variant while maintaining quality.

## Experiments

### d12-linear

- **Commit:** a9715b0
- **Change:** relu + x^2 + normalization
- **Result:** Loss 2.91. Generation: "The capital of France is Paris, and the city of Paris is the capital of France. The city is located in the heart of the country, and the city is the capital of France"

### d12-linear-poly-v5

- **Commit:** 56253f1
- **Change:** relu + ax+x^2+bx^4 + normalization (with a,b learnable params per head, properly initialized to 1.0)
- **Result:** Generation: "The capital of France is the capital of the French Republic of the country. The capital is the capital of the French Republic of the country. The capital is the capital of the French Republic". Repetitive phrase loops, but no single-token degeneration (previous results showing "is is is..." were caused by a missing KV cache fallback in the engine).

### d12-linear-sqrt-v2

- **Commit:** a1965b1
- **Change:** simpler linear attn: x^3 / sqrt(T) (no relu, no normalization), from https://arxiv.org/abs/2410.18613
- **Result:** Loss 2.86 (BPB 0.877), slightly better than poly variants. Generation produces empty/near-empty output — the x^3 / sqrt(T) kernel produces near-zero attention weights that collapse the output.

### d12-relu-sq

- **Commit:** 19cdf8f
- **Change:** relu(x)^2 feature map on both Q and K (from https://arxiv.org/abs/2006.16236), with row-wise normalization of the attention matrix
- **Result:** Generation: "The capital of France is the capital of France. It is the capital of the country of France. The capital of France is the capital of France. The capital of France is the capital". Repetitive phrase loops, but no single-token degeneration (previous results showing "the the the..." were caused by a missing KV cache fallback in the engine).

### d12-simple-x2

- **Commit:** 7e9eebc
- **Change**: x^2, then row normalization to 1
- **Result**: Generation: "The capital of France is the capital of the country. It is the capital of the country. The capital" - not great

### d12-simple-relu

- **Commit:** 2331455
- **Change**: relu, then row normalization to 1
- **Result**: Generation: "The capital of France is the capital of the world, and it is the largest city in the world." - factually wrong but kind of better english than the x2 version

### d12-simple-x4

- **Commit**: e97b5d5
- **Change**: x^4, then row normalization to 1
- **Result**: Generation: "The capital of France is Paris. The capital of France is Paris. The capital of France is Paris" - better!

### d12-taylor-relu

- **Commit**: 62a6e70
- **Change**: relu(1+x+x^2/2+x^3/6), then row normalization to 1
- **Result**: Generation: "The capital of France is Important realities Important Important Cell, beingTraditional Importantjour Importantpro ImportantA greatA". Interestingly, the loss and val/bpb kept falling, stopped at ~0.71 loss and ~0.41 bpb.

## Bug: params initialized with 0

The following experiments all had a bug where the parameters weren't actually initialized to 1, so they weights didn't "collapse" to 0 but rather were initialized with 0 and were stuck there. This was fixed in 182f1ef.

### d12-linear-scale

- **Commit:** c535be6
- **Change:** Scale param per attention head
- **Result:** Loss 5.50, all scale params collapsed to 0

### d12-linear-2

- **Commit:** 8622c04
- **Change:** Apply scale after relu
- **Result:** Didn't help, still collapsed to 0

### d12-linear-poly

- **Commit:** a7ea664
- **Change:** Switched to relu + ax+bx^2+cx^4 + normalization, with learnable a,b,c per head
- **Result:** Params collapsed to 0 again, loss ~5.50

### d12-linear-poly-v3

- **Commits:** 7d34635 (constrain coefficients), a6e3ed9 (move coeffs to own optimizer group)
- **Change:** Switched to ax+x^2+bx^4 with only two learnable params per head, fixing x^2; also moved coefficients to separate optimizer group
- **Result:** a,b collapsed to 0, leaving only x^2, loss back to 2.92 (matches initial x^2-only result)
