# Build a regression training audit

Integrate arrays, pure functions, autodiff, vectorization, compiled loops, and optimization on CPU. The synthetic dataset follows the known line \(y = 2x + 1\) so you can verify your numerical system against exact analytical expectations rather than guessing whether a training curve looks plausible.

## Where to put your code and how to run checks

Extract the project ZIP (or use the repository root) and activate the course virtual environment with `requirements-cpu.txt` installed. Run all commands from that top-level folder:

- **Your implementation file (`projects/regression-audit/starter/model.py`):** Open this file in your editor and implement the five functions (`predict`, `loss`, `gradient_check`, `train`, and `report`) one stage at a time.
- **Automated stage verifier (`projects/regression-audit/tests/check.py`):** Runs your code in `projects/regression-audit/starter/model.py` against each stage contract.
- **Reference solution (`projects/regression-audit/solution/model.py`):** Instructor reference (`--implementation solution`) for checking your environment or comparing after you complete a stage.

```bash
# Run run command in terminal using the course Python environment
python3 projects/regression-audit/tests/check.py --stage 1
python3 projects/regression-audit/tests/check.py --stage 2
python3 projects/regression-audit/tests/check.py --stage 3
python3 projects/regression-audit/tests/check.py --stage 4
```

## Stage 1 — Prediction and objective

Implement `predict(params, x)` and `loss(params, x, y)` in `projects/regression-audit/starter/model.py`:

1. Compute the linear prediction `params['weight'] * x + params['bias']`.
2. In `loss(params, x, y)`, verify that `x.shape == y.shape` and raise `ValueError` if a column vector `y[:, None]` is passed—otherwise `(predict(params, x) - y)` silently broadcasts into a 2D pairwise residual matrix.
3. Return the scalar mean squared error `jnp.mean((predict(params, x) - y) ** 2)`.

## Stage 2 — Gradient evidence

Implement `gradient_check(params, x, y, h=1e-2)` in `projects/regression-audit/starter/model.py`:

1. Compute automatic derivatives `automatic = jax.grad(loss)(params, x, y)`.
2. For each key in `params` (`'weight'` and `'bias'`), perturb that scalar parameter by `+h` and `-h` while holding the other parameter fixed, and compute the central difference `(loss(plus, x, y) - loss(minus, x, y)) / (2 * h)`.
3. Return `(automatic, finite)` and verify that both match the analytical derivatives at two distinct parameter settings.

## Stage 3 — Compiled training

Implement `train(params, x, y, rate=0.15, steps=200)` in `projects/regression-audit/starter/model.py`:

1. Validate that `steps >= 1` (raise `ValueError` otherwise).
2. Define a pure scan body `step(p, _)` that evaluates `value, gradients = jax.value_and_grad(loss)(p, x, y)`, updates the parameter pytree with `jax.tree.map(lambda v, g: v - rate * g, p, gradients)`, and returns `(updated, value)`.
3. Run `jax.lax.scan(step, params, None, length=steps)` to recover `weight = 2.0` and `bias = 1.0` while returning `(fitted, history)` with shape `(steps,)`.

## Stage 4 — Training audit

Implement `report(params, x, y, rate=0.15, steps=200)` in `projects/regression-audit/starter/model.py`:

1. Call `fitted, history = train(params, x, y, rate, steps)`.
2. Evaluate held-out points `heldout_x = jnp.array([-0.8, 0.2, 0.8])` and `heldout_y = 2 * heldout_x + 1` without mutating `fitted`.
3. Return a dictionary containing `'parameters'`, `'initial_loss'`, `'training_loss'`, `'heldout_loss'`, `'finite'`, `'steps'`, `'learning_rate'`, and `'backend'`. The stage check verifies held-out fit, deterministic replay, and deliberate divergence at an unstable learning rate (`rate=1.1`).

Try your implementation before reading `projects/regression-audit/solution/model.py`. Verify the reference separately with `python3 projects/regression-audit/tests/check.py --implementation solution --stage all`; reference success does not grade your work.

## Portfolio evidence

Keep your implementation in `projects/regression-audit/starter/model.py`, the commands, environment details, passing stage output, learning curve, unstable-rate failure run, and a brief explanation of what the checks establish. Passing public checks supports this bounded synthetic task; it is verified separately from production training, TPU operation, expert review, or career readiness.

