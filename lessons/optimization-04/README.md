# Optimize with Optax

Phase 04: Math & optimization · about 80 minutes · CPU

## What you will be able to do

- Distinguish gradients, updates, and parameters
- Understand the state even when SGD has none
- Make a resumable training transition
- Verify the simpler algorithm before composing one
- Preserve state when a gradient happens to be zero

## The problem

An optimizer helps turn gradients into parameter changes. Before introducing a more complicated one, we’ll check that Optax SGD agrees with the update you already know. Then we’ll add momentum and follow the extra state it remembers, so you can explain why resetting that state changes the next step.

## The idea

An optimizer transforms gradients into parameter updates and may retain state between steps. Optax makes that state explicit: initialize it, pass it into each update, and keep the returned state together with the new parameters.

## The optimizer has memory of earlier gradients

For plain gradient descent without extra state, the current parameters and gradient can determine the next update. For momentum or Adam, earlier gradients also matter through stored accumulators. Two runs at the same parameters can therefore take different next steps if their optimizer states differ.

Follow the update cycle in order: calculate the loss and gradients from current parameters, ask the optimizer for updates and new state, then apply those updates to the parameters. Keeping only one returned object breaks the cycle.

The fitted-line figure shows that the example recovered a useful prediction rule. It does not by itself prove correct state continuation. Save a mid-run state and compare the next update with an uninterrupted run to test that separate capability.

### Pause and reason

You restore weights but initialize Adam from scratch. Is that an exact continuation?

<details><summary>Compare your reasoning</summary>

No. The moment estimates and step counter have changed, so subsequent updates can differ. This may be a deliberate new optimization run, but it is not an exact resume.

</details>

## Distinguish gradients, updates, and parameters

A gradient describes local sensitivity; an update is the change the optimizer wants to add to parameters. Optax SGD transforms $g$ into $-\eta g$. apply_updates adds that update, so subtracting it again would reverse the intended direction. The tree structure of gradients and updates follows the parameter tree.

For our evenly spaced $x$ values, $\operatorname{mean}(x)=0$ and $\operatorname{mean}(x^2)=\frac{11}{30}$. At zero parameters, the weight derivative is $-4\cdot\frac{11}{30}=-\frac{22}{15}$ and the bias derivative is $-2$. With rate $0.2$, the first weight becomes $22/75$ and bias $0.4$. Compute these values before trusting a long run.

```text
params → loss → grads
grads + old optimizer state → updates + new optimizer state
params + updates → next params
```

## Understand the state even when SGD has none

Plain SGD here has no momentum history, so the equality with manual gradient descent is a useful baseline. Momentum adds a trace of prior gradients. Its returned state must be threaded through every call; rebuilding state each step discards that history and gives a different algorithm.

With momentum $0.9$ and gradients $2$ then $1$, the trace is first $2$ then $2.8$. At rate $0.1$ the second update is $-0.28$. Reinitializing before the second gradient gives $-0.1$. This two-step example demonstrates the actual state dependence without assuming that the regression fit alone proves it.

## Make a resumable training transition

Treat a training step as (params, optimizer_state, batch) → (new_params, new_state, metrics). A checkpoint must preserve the algorithm state as well as parameter values. With a deterministic fixed dataset, replaying the next step from identical saved parameters and state should produce identical results within the same environment.

This lesson performs an in-memory replay; it does not implement file persistence or claim crash recovery. Random keys, data iterator position and distributed topology become additional state in later training systems. A tuple called “checkpoint” is not a resilient checkpoint service.

## Verify the simpler algorithm before composing one

Compare one Optax SGD step to a hand-derived tree update before introducing momentum, clipping, schedules or Adam. Composition order matters: transformations operate on the value produced by preceding transformations. Keep loss convention and the rate fixed while comparing algorithms.

The 120-step fit is a small CPU regression example, not a TPU throughput benchmark or evidence that the same rate is suitable for a network. Inspect the parameters, loss, and optimizer transition separately; the final loss alone cannot reveal a reset-state bug in every problem.

## Preserve state when a gradient happens to be zero

Momentum remembers earlier gradients. Under the convention $m_t=\beta m_{t-1}+g_t$, a zero current gradient can still produce a nonzero update. With $\beta=0.9$, a previous trace of $2$ becomes $1.8$ when the new gradient is zero. This is why an optimizer’s behavior cannot be reconstructed from the current gradient and weights alone. The next lesson builds on this state model to explain Adam and learning-rate schedules.

$$
m_t=\beta m_{t-1}+g_t,\qquad \Delta w_t=-\eta m_t
$$

## 1. Prepare the inputs

Create a file named main.py in your activated learning environment. Add these imports and inputs first. Shapes and numeric types are part of the experiment.

```python
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax
```

This block establishes the values used by the following steps; run the completed file after step $3$.

## 2. Define the computation

Append this block below the inputs. Before proceeding, trace which values are inputs, predictions, and state.

```python
# Step 2 — 2. Define the computation: loss maps the two parameter leaves to row-wise predictions.
# Construct `params` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Construct `x` via `jnp.linspace(-1., 1., 21)`
x = jnp.linspace(-1., 1., 21)
# Compute `y` from `2. * x + 1.`
y = 2. * x + 1.
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Return `jnp.mean((p['weight'] * x + p['bias'] - y) ** 2)` to the caller.
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.sgd(0.2)
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)
```

loss maps the two parameter leaves to row-wise predictions. optimizer.init establishes the initial SGD state; retain the update state in the next block.

## 3. Measure and verify

Append the remaining block, then run python main.py in the terminal. Compare the printed result with the expected output below. Assertions stop execution if the contract fails.

```python
# Step 3 — 3. Measure and verify: Weight approaches 2.0, bias approaches 1.0, and loss is below 10^{-8}.
initial_loss = loss(params)
# Repeat the update loop over `range(120)` steps:
for _ in range(120):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads = jax.grad(loss)(params)
    # Apply the computed gradient updates to update the model parameters.
    updates, state = optimizer.update(grads, state, params)
    # Configure or step the Optax optimizer state (`params`).
    params = optax.apply_updates(params, updates)
# Print the observed values to compare against the expected result.
print("Parameters:", params)
# Print diagnostic summary of the computed outputs.
print("Loss:", float(loss(params)))
# Assert invariant `loss(params) < 1e-8` holds
assert loss(params) < 1e-8
# Assert that `jnp.allclose(params["weight"], 2., atol=1e-4)`.
assert jnp.allclose(params["weight"], 2., atol=1e-4)
# Assert that `jnp.allclose(params["bias"], 1., atol=1e-4)`.
assert jnp.allclose(params["bias"], 1., atol=1e-4)
```

Weight approaches $2.0$, bias approaches $1.0$, and loss is below $10^{-8}$.

## Run the example

```python
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import optax
# Step 2 — 2. Define the computation: loss maps the two parameter leaves to row-wise predictions.
# Construct `params` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
params = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Construct `x` via `jnp.linspace(-1., 1., 21)`
x = jnp.linspace(-1., 1., 21)
# Compute `y` from `2. * x + 1.`
y = 2. * x + 1.
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Return `jnp.mean((p['weight'] * x + p['bias'] - y) ** 2)` to the caller.
    return jnp.mean((p["weight"] * x + p["bias"] - y) ** 2)
# Configure or step the Optax optimizer state (`optimizer`).
optimizer = optax.sgd(0.2)
# Run `optimizer.init` to compute `state`.
state = optimizer.init(params)
# Step 3 — 3. Measure and verify: Weight approaches 2.0, bias approaches 1.0, and loss is below 10^{-8}.
initial_loss = loss(params)
# Repeat the update loop over `range(120)` steps:
for _ in range(120):
    # Differentiate the objective to obtain `grads` via automatic differentiation.
    grads = jax.grad(loss)(params)
    # Apply the computed gradient updates to update the model parameters.
    updates, state = optimizer.update(grads, state, params)
    # Configure or step the Optax optimizer state (`params`).
    params = optax.apply_updates(params, updates)
# Print the observed values to compare against the expected result.
print("Parameters:", params)
# Print diagnostic summary of the computed outputs.
print("Loss:", float(loss(params)))
# Assert invariant `loss(params) < 1e-8` holds
assert loss(params) < 1e-8
# Assert that `jnp.allclose(params["weight"], 2., atol=1e-4)`.
assert jnp.allclose(params["weight"], 2., atol=1e-4)
# Assert that `jnp.allclose(params["bias"], 1., atol=1e-4)`.
assert jnp.allclose(params["bias"], 1., atol=1e-4)
```

Expected: Weight approaches $2.0$, bias approaches $1.0$, and loss is below $10^{-8}$.

## The fitted line recovers slope and bias

**Predict:** What does each learned parameter change in the predictions?

![The fitted line recovers slope and bias](../../phases/04-optimization/04-optimize-with-optax/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis gives input values and the vertical axis gives predictions. The dashed initial model is flat at zero. The target and trained-model lines rise together from approximately $(-1,-1)$ through $(0,1)$ to $(1,3)$.

The trained line lies almost exactly on top of the target, so one can hide the other. This overlap is the intended result, not a missing legend entry. It means the predictions agree to the numerical tolerance used in the example.

### Connect it to the computation

Read the learned parameters from the geometry: the intercept is $1$, and increasing the input by $1$ raises the prediction by $2$. Training has therefore recovered the target relationship $y=2x+1$, whereas the initial model used zero slope and zero intercept.

This picture shows the final fit across the displayed inputs, not the optimizer’s path or its runtime. Agreement on an exact synthetic linear target is a useful correctness check. It is separate from performance on noisy data or on a different relationship.

```python
# Compute figure data for: The fitted line recovers slope and bias
# Allocate initialized array `visual_data` with the specified shape and dtype.
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'target', 'y': y.tolist()}, {'label': 'initial model', 'y': jnp.zeros_like(x).tolist()}, {'label': 'trained model', 'y': (params['weight'] * x + params['bias']).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:02:21.347697+00:00. JAX 0.9.2.

```text
Parameters: {'bias': Array(1., dtype=float32), 'weight': Array(1.9999996, dtype=float32)}
Loss: 5.126058404787345e-14
Parameters: {'bias': Array(1., dtype=float32), 'weight': Array(1.9999996, dtype=float32)}
Loss: 5.126058404787345e-14
First parameters: {'bias': Array(0.4, dtype=float32), 'weight': Array(0.29333338, dtype=float32)}
PASS: optimization-04

```

## Check the first update against arithmetic

**Predict before running:** Predict both parameters after one SGD step at the zero initialization.

```python
# Experiment — Check the first update against arithmetic: Independent arithmetic checks the gradient scale and the update...
# Construct `p0` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
p0 = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Differentiate the objective to obtain `g0` via automatic differentiation.
g0 = jax.grad(loss)(p0)
# Apply the computed gradient updates to update the model parameters.
u0, s0 = optimizer.update(g0, optimizer.init(p0), p0)
# Configure or step the Optax optimizer state (`p1`).
p1 = optax.apply_updates(p0, u0)
# Assert that `jnp.allclose(p1["weight"], 22/75)`.
assert jnp.allclose(p1["weight"], 22/75)
# Assert that `jnp.allclose(p1["bias"], 0.4)`.
assert jnp.allclose(p1["bias"], 0.4)
# Print the observed values to compare against the expected result.
print("First parameters:", p1)
```

**Expected:** Weight ≈ $0.293333$; bias $0.4$.

Independent arithmetic checks the gradient scale and the update sign.

## Observe stateful momentum

**Predict before running:** After gradients $2$ then $1$, is the second update the same if you reset state?

```python
# Experiment — Observe stateful momentum: Resetting momentum removes the prior-gradient contribution; it...
# Configure or step the Optax optimizer state (`momentum`).
momentum = optax.sgd(0.1, momentum=0.9)
# Construct `m0` via `momentum.init(jnp.array(0.))`
m0 = momentum.init(jnp.array(0.))
# Construct `u1, m1` via `momentum.update(jnp.array(2.), m0)`
u1, m1 = momentum.update(jnp.array(2.), m0)
# Construct `u2, m2` via `momentum.update(jnp.array(1.), m1)`
u2, m2 = momentum.update(jnp.array(1.), m1)
# Construct `reset_u2, _` via `momentum.update(jnp.array(1.), m0)`
reset_u2, _ = momentum.update(jnp.array(1.), m0)
# Assert that `jnp.allclose(u1, -0.2)`.
assert jnp.allclose(u1, -0.2)
# Assert that `jnp.allclose(u2, -0.28)`.
assert jnp.allclose(u2, -0.28)
# Assert that `jnp.allclose(reset_u2, -0.1)`.
assert jnp.allclose(reset_u2, -0.1)
```

**Expected:** Second update $-0.28$ with retained state, $-0.1$ after reset.

Resetting momentum removes the prior-gradient contribution; it changes the optimization algorithm.

## Coast through a zero-gradient step

**Predict before running:** After gradient $2$, does a zero gradient stop momentum SGD immediately?

```python
# Experiment — Coast through a zero-gradient step: Optimizer state carries information from earlier steps.
# Configure or step the Optax optimizer state (`momentum_tx`).
momentum_tx = optax.sgd(0.1, momentum=0.9)
# Construct `position` via `jnp.array(0.)`
position = jnp.array(0.)
# Run `momentum_tx.init` to compute `momentum_state`.
momentum_state = momentum_tx.init(position)
# Construct `coast_first, momentum_state` via `momentum_tx.update(jnp.array(2.),momentum_state,posi...`
coast_first, momentum_state = momentum_tx.update(jnp.array(2.),momentum_state,position)
# Configure or step the Optax optimizer state (`position`).
position = optax.apply_updates(position,coast_first)
# Construct `coast_second, momentum_state` via `momentum_tx.update(jnp.array(0.),momentum_state,posi...`
coast_second, momentum_state = momentum_tx.update(jnp.array(0.),momentum_state,position)
# Assert that `jnp.allclose(coast_first,-0.2)`.
assert jnp.allclose(coast_first,-0.2)
# Assert that `jnp.allclose(coast_second,-0.18)`.
assert jnp.allclose(coast_second,-0.18)
```

**Expected:** The second update is $-0.18$, despite a zero current gradient.

Optimizer state carries information from earlier steps. A zero current gradient need not produce a zero update.

## Make it yours

At the original zero parameters, compare the first Optax SGD update with an explicit tree-map gradient descent step.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.
- `jax.tree.map(lambda p, g: ..., params, grads)` — Applies a function leaf-by-leaf across matching PyTrees (such as updating every parameter tensor with its gradient).

**Step-by-step implementation plan:**
1. Construct `start` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
2. Differentiate the objective to obtain `grads` via automatic differentiation.
3. Apply the computed gradient updates to update the model parameters.
4. Configure or step the Optax optimizer state (`actual`).
5. Transform every leaf of the parameter PyTree (`expected`).

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: At the original zero parameters, compare the first Optax SGD update...
# Construct `start` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
start = ...  # TODO: compute start
# Differentiate the objective to obtain `grads` via automatic differentiation.
grads = jax.grad(...)  # TODO: compute grads
# Apply the computed gradient updates to update the model parameters.
updates, _ = optimizer.update(...)  # TODO: compute updates, _
# Configure or step the Optax optimizer state (`actual`).
actual = optax.apply_updates(...)  # TODO: compute actual
# Transform every leaf of the parameter PyTree (`expected`).
expected = jax.tree.map(...)  # TODO: compute expected
# Assert that `all(jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))`.
assert all(jnp.allclose(a, b) for a, b  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: At the original zero parameters, compare the first Optax SGD update...
# Construct `start` via `{"weight": jnp.array(0.), "bias": jnp.array(0.)}`
start = {"weight": jnp.array(0.), "bias": jnp.array(0.)}
# Differentiate the objective to obtain `grads` via automatic differentiation.
grads = jax.grad(loss)(start)
# Apply the computed gradient updates to update the model parameters.
updates, _ = optimizer.update(grads, optimizer.init(start), start)
# Configure or step the Optax optimizer state (`actual`).
actual = optax.apply_updates(start, updates)
# Transform every leaf of the parameter PyTree (`expected`).
expected = jax.tree.map(lambda p, g: p - 0.2 * g, start, grads)
# Assert that `all(jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))`.
assert all(jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)))
```

</details>

## Replay a momentum continuation

**Transfer / diagnosis**

Save the state after the first update; run the next update twice from that state and compare every state leaf.

<details><summary>Hint</summary>

Optimizer transformations return new state; reuse the saved value for the replay.

</details>

### How to write: Replay a momentum continuation — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `replay_u, replay_state` via `momentum.update(jnp.array(1.), m1)`
2. Assert that `jnp.allclose(replay_u, u2)`.
3. Assert that `all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))`.

**Starter code scaffold (fill in the TODOs):**

```python
# Replay a momentum continuation (Transfer / diagnosis): This verifies an in-memory optimizer transition.
# Construct `replay_u, replay_state` via `momentum.update(jnp.array(1.), m1)`
replay_u, replay_state = momentum.update(...)  # TODO: compute replay_u, replay_state
# Assert that `jnp.allclose(replay_u, u2)`.
assert jnp.allclose(replay_u, u2)  # TODO: complete assertion check
# Assert that `all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))`.
assert all(jnp.allclose(a,b) for a,b  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Replay a momentum continuation (Transfer / diagnosis): This verifies an in-memory optimizer transition.
# Construct `replay_u, replay_state` via `momentum.update(jnp.array(1.), m1)`
replay_u, replay_state = momentum.update(jnp.array(1.), m1)
# Assert that `jnp.allclose(replay_u, u2)`.
assert jnp.allclose(replay_u, u2)
# Assert that `all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))`.
assert all(jnp.allclose(a,b) for a,b in zip(jax.tree.leaves(replay_state), jax.tree.leaves(m2)))
```

This verifies an in-memory optimizer transition. It does not test serialization or a data-loader resume.

</details>

## Catch double subtraction

**Transfer / diagnosis**

At zero parameters, subtract SGD updates by mistake. Show the loss increases; repair using apply_updates.

<details><summary>Hint</summary>

SGD updates already contain the minus sign.

</details>

### How to write: Catch double subtraction — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jax.tree.map(lambda p, g: ..., params, grads)` — Applies a function leaf-by-leaf across matching PyTrees (such as updating every parameter tensor with its gradient).

**Step-by-step implementation plan:**
1. Transform every leaf of the parameter PyTree (`wrong`).
2. Configure or step the Optax optimizer state (`right`).
3. Assert invariant `loss(wrong) > loss(p0)` holds
4. Assert invariant `loss(right) < loss(p0)` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Catch double subtraction (Transfer / diagnosis): Comparing identical gradients exposes an update-sign error...
# Transform every leaf of the parameter PyTree (`wrong`).
wrong = jax.tree.map(...)  # TODO: compute wrong
# Configure or step the Optax optimizer state (`right`).
right = optax.apply_updates(...)  # TODO: compute right
# Assert invariant `loss(wrong) > loss(p0)` holds
assert loss(wrong)  # TODO: complete assertion check
# Assert invariant `loss(right) < loss(p0)` holds
assert loss(right)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Catch double subtraction (Transfer / diagnosis): Comparing identical gradients exposes an update-sign error...
# Transform every leaf of the parameter PyTree (`wrong`).
wrong = jax.tree.map(lambda p,u: p-u, p0,u0)
# Configure or step the Optax optimizer state (`right`).
right = optax.apply_updates(p0,u0)
# Assert invariant `loss(wrong) > loss(p0)` holds
assert loss(wrong) > loss(p0)
# Assert invariant `loss(right) < loss(p0)` holds
assert loss(right) < loss(p0)
```

Comparing identical gradients exposes an update-sign error rather than a hyperparameter difference.

</details>

## Check your understanding

What should a training step retain besides updated parameters for a stateful optimizer?

1. Only the final loss
2. The new optimizer state returned by update
3. A freshly initialized state every time

<details><summary>Answer and explanation</summary>

The new optimizer state returned by update

State carries momentum or adaptive statistics between steps. Reinitializing it discards that history.

</details>

## Diagnose the result

If an adaptive optimizer behaves differently after restarting, compare both parameter values and optimizer state. Equal parameters alone are separate from an identical continuation.

## Carry forward

- Independent arithmetic checks the gradient scale and the update sign.
- Resetting momentum removes the prior-gradient contribution; it changes the optimization algorithm.

## Keep your evidence

Save the analytic first update, manual/Optax equivalence, retained/reset momentum comparison, continuation replay and double-subtraction repair. Include the zero-gradient momentum step and explain why retained state still moves the parameter.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html)
- [Optax optimizer API](https://optax.readthedocs.io/en/latest/api/optimizers.html)

