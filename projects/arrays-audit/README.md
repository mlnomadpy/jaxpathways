# Audit tensor shapes, dtypes, broadcasting, and reductions

Phase 01 dedicated integration project ([Phase 01](../../phases/01-arrays/README.md)). See also the cross-phase companion project [`foundation-toolkit`](../foundation-toolkit/README.md).

## Mathematical contract

Given input batch $X \in \mathbb{R}^{B \times d_\text{in}}$ and projection weights $W_\text{enc}, P$, the normalized representation is:

$$
H = \tanh(X W_\text{enc} + b_\text{enc}) P, \qquad \hat{H}_i = \frac{H_i}{\max(\|H_i\|_2, 10^{-6})}
$$

## Stage verification

1. **Stage 1 — Parameter and state contract:** `python3 projects/arrays-audit/tests/check.py --stage 1`
2. **Stage 2 — Masked forward pass:** `python3 projects/arrays-audit/tests/check.py --stage 2`
3. **Stage 3 — Gradient verification:** `python3 projects/arrays-audit/tests/check.py --stage 3`
4. **Stage 4 — Compiled scan qualification:** `python3 projects/arrays-audit/tests/check.py --stage 4`
