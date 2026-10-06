# Build a regression training audit

Integrate arrays, pure functions, autodiff, vectorization, compiled loops, and optimization. CPU only. The synthetic dataset has a known solution so you can check the numerical system rather than guess whether the model looks plausible.

Implement `starter/model.py` one stage at a time. Run from the repository root after installing `requirements-cpu.txt`:

```bash
python3 projects/regression-audit/tests/check.py --stage 1
python3 projects/regression-audit/tests/check.py --stage 2
python3 projects/regression-audit/tests/check.py --stage 3
python3 projects/regression-audit/tests/check.py --stage 4
```

1. **Prediction and objective:** write a linear model and scalar MSE. Reject column targets that would broadcast into pairwise residuals.
2. **Gradient evidence:** compare autodiff with central differences at two parameter settings. Explain the tolerance and step-size choice.
3. **Training:** carry a parameter pytree through `lax.scan`. Recover the known weight and bias. Keep the full loss trajectory.
4. **Audit:** report training and held-out loss, numerical health, and reproducibility. Run an unstable learning rate and explain the failure from the update equation.

Try your implementation before reading `solution/model.py`. Verify the reference separately with `--implementation solution`; reference success does not grade your work.

## Portfolio evidence

Keep your implementation, commands, environment, passing stage output, learning curve, failure run, and a brief explanation of what the checks establish. Passing public checks supports this bounded synthetic task; it does not establish production training, TPU operation, expert review, or career readiness.
