# Verify checkpoint integrity, NaN guards, and deterministic replay

Phase 06 dedicated integration project ([Phase 06](../../phases/06-recovery/README.md)). See also the cross-phase companion project [`engineering-release`](../engineering-release/README.md).

## Mathematical contract

Given input batch $X \in \mathbb{R}^{B \times d_\text{in}}$ and projection weights $W_\text{enc}, P$, the normalized representation is:

$$
H = \tanh(X W_\text{enc} + b_\text{enc}) P, \qquad \hat{H}_i = \frac{H_i}{\max(\|H_i\|_2, 10^{-6})}
$$

## Stage verification

1. **Stage 1 — Parameter and state contract:** `python3 projects/recovery-audit/tests/check.py --stage 1`
2. **Stage 2 — Masked forward pass:** `python3 projects/recovery-audit/tests/check.py --stage 2`
3. **Stage 3 — Gradient verification:** `python3 projects/recovery-audit/tests/check.py --stage 3`
4. **Stage 4 — Compiled scan qualification:** `python3 projects/recovery-audit/tests/check.py --stage 4`
