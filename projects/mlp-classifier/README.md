# Build a reproducible classifier audit

Implement a small Flax NNX classifier and produce evidence that its loss is correct, its updates are reproducible, and its evaluation does not train on held-out examples. The three stages deliberately separate implementation correctness from fitting and measurement.

The task is synthetic XOR in two dimensions. Training uses four corners. Evaluation uses 40 perturbed corners generated from another seed. It is a controlled way to practice a training system on CPU, not a useful real-world classifier or a vision benchmark. Public checks are learning feedback; passing them does not mean a reviewer has assessed your explanation.

## Before starting

Complete neural network lessons `networks-01` through `networks-05`, including the explicit MLP, Flax state ownership, compiled update/evaluation, held-out CNN measurement, and instability diagnosis. You should already be able to inspect array shapes, differentiate a parameter tree, and explain why optimizer state matters.

Tested locally on CPU: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, Flax 0.12.6, Optax 0.2.8. The reference has not been run on TPU. Small training fixtures are suitable for a laptop; this project makes no speed claims.

## Start in a terminal

1. Open a terminal in the repository folder and activate the environment from the setup lesson.
2. Run `python3 -m pip install -r requirements-cpu.txt` to install the course's pinned CPU requirements. Confirm that `python3 -c "import flax; print(flax.__version__)"` reports the tested Flax version.
3. Copy the starter into your own file, so the original scaffold stays available:

   ```sh
   cp projects/mlp-classifier/starter/model.py projects/mlp-classifier/my_model.py
   ```

4. Open `projects/mlp-classifier/my_model.py` in your editor. Fill in one stage at a time. Run the corresponding command below from the repository root. A traceback beginning with `NotImplementedError` is expected until you implement the required functions.

The public tests default to the untouched starter. Always supply your file with `--implementation`. Read the tests to understand the contract, then add your own changed-condition check rather than tailoring answers to one fixture.

## Stage 1: model and shapes

Implement `make_model(seed, width)`. Return an NNX module with attributes `hidden` and `out`: `nnx.Linear(2, width)`, tanh, then `nnx.Linear(width, 1)`. The call must return logits with shape `(B,)`. Keep the batch dimension at `B=1`; use `squeeze(-1)` rather than indiscriminate squeezing.

```sh
python3 projects/mlp-classifier/tests/check.py --stage 1 --implementation projects/mlp-classifier/my_model.py
```

The check compares your forward pass with NumPy on several batches, checks widths three and eight, counts `4*width + 1` parameters, and verifies equal-seed initialization. Keep a shape diagram and your hand calculation of the parameter count.

## Stage 2: objective and derivative evidence

Implement `objective(model, x, labels)`. Use mean stable binary cross-entropy on logits, and raise `ValueError` if score and label shapes differ. Labels must have shape `(B,)`; `(B,1)` would create accidental pairwise broadcasting.

```sh
python3 projects/mlp-classifier/tests/check.py --stage 2 --implementation projects/mlp-classifier/my_model.py
```

The check uses an independent NumPy `logaddexp` objective and finite differences of the output bias at two initializations. The finite-difference perturbation is 0.002, with a tolerance that accommodates float32 cancellation. Keep autodiff and finite-difference values plus an explanation of why an arbitrarily smaller perturbation can be worse.

## Stage 3: updates, isolation, replay

Implement `train(seed, x, labels, rate=.03, steps=200, width=8)`. Construct the model and a fresh `nnx.Optimizer` using Optax Adam. Compile the update with `nnx.jit`, compute `nnx.value_and_grad`, and call `optimizer.update(model, grads)`. Return the trained model and an array of pre-update losses with exactly `steps` entries.

Implement `evaluate(model, x, labels)` returning a dictionary with Python values `loss`, `accuracy`, and `count`. Evaluation must not update any state. Threshold logits at zero for binary decisions. Reject malformed shapes through the objective before reducing metrics.

```sh
python3 projects/mlp-classifier/tests/check.py --stage 3 --implementation projects/mlp-classifier/my_model.py
```

The check measures held-out rows never passed to the update, compares your metrics independently, checks every model state array before and after evaluation, and verifies complete replay. It also changes the model seed and update count, and checks that aggregating unequal batches of seven and 33 rows uses count weights. An unweighted average of those two accuracies is generally wrong.

## Evidence to keep

Create your own `report.md` next to `my_model.py`. Include:

- The environment and exact commands you ran, not a copied success claim.
- A model shape diagram, parameter count, and stable-loss equation.
- Two finite-difference comparisons with perturbation and tolerance.
- Initial/final training loss and held-out loss, accuracy, and observation count.
- A replay comparison and an evaluation-state snapshot comparison.
- One deliberate failure, its diagnostic evidence, and the repair: malformed labels, unequal-batch averaging, or using the training function during evaluation.
- One new condition you chose and the limits of your conclusion. Suitable changes include hidden width, a second held-out seed fixed before fitting, or increased input noise.

Use the instability lesson to distinguish corrupted inputs, an objective shape bug, and an excessive step size. Do not assume every larger Adam learning rate must diverge on this particular tiny problem.

The reference implementation is available in `solution/model.py` after your attempt. To check it:

```sh
python3 projects/mlp-classifier/tests/check.py --stage 3 --implementation projects/mlp-classifier/solution/model.py
```

A CPU receipt establishes execution in one environment. It does not establish TPU behavior, generalization outside these perturbed corners, secure deployment, or independently reviewed competency.
