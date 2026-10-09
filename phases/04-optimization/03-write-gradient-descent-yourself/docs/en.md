# Write gradient descent yourself

Phase 04: Math & optimization · about 80 minutes · CPU

## What you will be able to do

- Derive one update before writing the loop
- Read the learning rate through curvature
- Make the trajectory part of the state contract
- Diagnose dynamics instead of only the final number
- Use stopping evidence that survives rescaling

## The problem

You now have a loss and a gradient. Let’s use them to move a weight toward a better value. We’ll calculate the first two updates ourselves, turn the update into a loop, and try a step size that is too large. Watching the whole path will tell us more than looking only at the final loss.

## The idea

Gradient descent subtracts a scaled gradient from the current parameters. The gradient points toward local increase; its negative gives a local descent direction. A sufficiently large step can cross the useful local region and increase the loss instead.

## The sign tells you which way to step

Take $L(w)=(w-3)^2$, starting from $w=0$. The derivative is $-6$. With learning rate $0.1$, the update gives $w=0.6$, and the loss falls from $9$ to $5.76$. The negative derivative produces a positive parameter change because we subtract it.

With learning rate $1.1$, the first update jumps to $6.6$. The new loss is $12.96$, larger than the starting loss. The derivative was correct in both cases; the step length changed the outcome.

When the loss plot rises, inspect the parameter trajectory and update norm before changing the derivative implementation. Loss hides the side of the minimum on which the parameter lies. A signed trajectory exposes overshoot.

### Pause and reason

Does a correct gradient guarantee that any positive learning rate reduces the loss?

<details><summary>Compare your reasoning</summary>

No. The gradient describes local behavior. Step-size stability depends on the objective's curvature and the update rule; the worked large-step case increases the loss despite the correct derivative.

</details>

## Derive one update before writing the loop

Starting at $w=0$, our gradient is $-4$. Subtracting $0.1$ times $-4$ gives $w=0.4$, whose loss is $2.56$. A second step gives $w=0.72$ and loss $1.6384$. These two hand-computed steps check the sign, learning rate, and whether your history records loss before or after the update.

Define the error $e=w-2$. The next error is $(1-2\eta)e$. This recurrence explains the whole trajectory for this quadratic. An optimizer update must follow the negative gradient; adding it moves away from the minimum. The subscript $k$ counts updates. The symbol $\eta$ is the learning rate, and $e_k=w_k-2$ is the remaining error. Focus on the multiplier $1-2\eta$: its magnitude tells us whether this particular error shrinks.

$$
\begin{aligned}w_{k+1}&=w_k-\eta\,2(w_k-2)\\0&\longrightarrow0.4\longrightarrow0.72\quad(\eta=0.1)\\e_k&=(1-2\eta)^k e_0\end{aligned}
$$

## Read the learning rate through curvature

Rates between $0$ and $0.5$ approach the optimum without crossing it. A rate of $0.5$ reaches it in one ideal step. Rates between $0.5$ and $1$ alternate around the optimum with shrinking error. A rate of $1$ alternates forever; values above $1$ grow in magnitude. These statements follow from the error multiplier for this specific loss.

For $L=a(w-c)^2$ with positive a, the stable interval is $0<\eta<\frac{1}{a}$. Scaling a loss changes the curvature and the safe update scale. There is no universal threshold such as $0.1$ for neural networks.

## Make the trajectory part of the state contract

scan carries the current scalar parameter and stacks one post-update loss per iteration. Its carry must retain a fixed shape and dtype. Our finite step count is a Python integer at construction; if you wrap train in jit, that length must be static or you must choose a different loop formulation.

Keep initial loss separately so you can interpret the first history entry. For this implementation, history$[0]$ is $2.56$, not $4$. A graph without this convention can make two correct training loops look off by one.

## Diagnose dynamics instead of only the final number

Plotting or printing the first few weights and losses reveals a sign error or an overly aggressive step much earlier than waiting for NaN. In this small problem the recurrence is an independent oracle for all tested steps. In larger models use loss, gradient norm, update norm, dtype and finite-value checks.

A deterministic loop should not be “fixed” by changing a random seed. Reproduce the failing rate, derive its error multiplier, lower it according to the model curvature, and check both the trajectory and the endpoint.

## Use stopping evidence that survives rescaling

A small change in the weights does not prove convergence: it can mean the learning rate is tiny. For $L(w)=(w-2)^2$ at $w=0$, the gradient is $-4$. A rate of $10^{-8}$ changes the weight by only $4\times10^{-8}$, while it remains far from the solution. Inspect the objective, gradient magnitude, update magnitude and held-out behavior together. Relative tolerances can help across scales, but no single stopping test certifies a global optimum for a general nonconvex model.

## 1. Prepare the inputs

Create a file named main.py in your activated learning environment. Add these imports and inputs first. Shapes and numeric types are part of the experiment.

```python
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
```

This block establishes the values used by the following steps; run the completed file after step $3$.

## 2. Define the computation

Append this block below the inputs. Before proceeding, trace which values are inputs, predictions, and state.

```python
# Step 2 — 2. Define the computation: train defines a state transition that subtracts the derivative.
def loss(w):
    # Return `(w - 2.0) ** 2` to the caller.
    return (w - 2.) ** 2
# Define `train(initial, rate, steps)` to evaluate the objective and its automatic derivatives:
def train(initial, rate, steps):
    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `next_w` via automatic differentiation.
        next_w = w - rate * jax.grad(loss)(w)
        # Return `(next_w, loss(next_w))` to the caller.
        return next_w, loss(next_w)
    # Return `jax.lax.scan(step, jnp.array(initial), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)
```

train defines a state transition that subtracts the derivative. scan retains the new parameter and records its post-update loss for exactly steps iterations.

## 3. Measure and verify

Append the remaining block, then run python main.py in the terminal. Compare the printed result with the expected output below. Assertions stop execution if the contract fails.

```python
# Step 3 — 3. Measure and verify: The weight approaches 2.0 and the final loss is below 10^{-8}.
final, history = train(0., 0.1, 60)
# Print the observed values to compare against the expected result.
print("Final weight:", float(final))
# Print diagnostic summary of the computed outputs.
print("Final loss:", float(history[-1]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, 2., atol=1e-4)
# Assert invariant `history[-1] < 1e-8` holds
assert history[-1] < 1e-8
# Assert invariant `jnp.all(jnp.diff(history) <= 1e-7)` holds
assert jnp.all(jnp.diff(history) <= 1e-7)
```

The weight approaches $2.0$ and the final loss is below $10^{-8}$.

## Run the example

```python
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — 2. Define the computation: train defines a state transition that subtracts the derivative.
def loss(w):
    # Return `(w - 2.0) ** 2` to the caller.
    return (w - 2.) ** 2
# Define `train(initial, rate, steps)` to evaluate the objective and its automatic derivatives:
def train(initial, rate, steps):
    # Define `step(w, _)` to evaluate the objective and its automatic derivatives:
    def step(w, _):
        # Differentiate the objective to obtain `next_w` via automatic differentiation.
        next_w = w - rate * jax.grad(loss)(w)
        # Return `(next_w, loss(next_w))` to the caller.
        return next_w, loss(next_w)
    # Return `jax.lax.scan(step, jnp.array(initial), None, length=steps)` to the caller.
    return jax.lax.scan(step, jnp.array(initial), None, length=steps)
# Step 3 — 3. Measure and verify: The weight approaches 2.0 and the final loss is below 10^{-8}.
final, history = train(0., 0.1, 60)
# Print the observed values to compare against the expected result.
print("Final weight:", float(final))
# Print diagnostic summary of the computed outputs.
print("Final loss:", float(history[-1]))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(final, 2., atol=1e-4)
# Assert invariant `history[-1] < 1e-8` holds
assert history[-1] < 1e-8
# Assert invariant `jnp.all(jnp.diff(history) <= 1e-7)` holds
assert jnp.all(jnp.diff(history) <= 1e-7)
```

Expected: The weight approaches $2.0$ and the final loss is below $10^{-8}$.

## Gradient descent trades step size for stability

**Predict:** Which rate will cause errors to grow rather than shrink?

![Gradient descent trades step size for stability](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis counts completed updates, including the initial state at $0$. The vertical axis is squared error. It uses a symmetric-log scale: values near zero are on a linear region, while large values are compressed logarithmically. That lets the zero-loss trajectory remain visible.

All three curves begin at loss $4$. Learning rate $0.1$ decreases it gradually; rate $0.5$ reaches zero after one update and stays there. Rate $1.1$ increases it from $4$ to $5.76$, then beyond $200$ by the last plotted update.

### Connect it to the computation

Here the loss is $(w-2)^2$. One update multiplies the weight error $w-2$ by $1-2\eta$. The three rates therefore multiply that error by $0.8$, $0$, and $-1.2$, respectively. Squaring gives loss multipliers $0.64$, $0$, and $1.44$.

The large rate crosses the optimum and lands farther away on each step. The loss plot shows the growing distance, while the sign change follows from the update equation; loss alone does not reveal which side the weight is on. The one-step success of rate $0.5$ depends on this simple quadratic’s curvature and is not a universal best learning rate.

```python
# Compute figure data for: Gradient descent trades step size for stability
# Compute `rates` from `[0.1, 0.5, 1.1]`
rates = [0.1, 0.5, 1.1]
# Compute `series` from `[]`
series = []
# Loop over `rate_plot` in `rates`:
for rate_plot in rates:
    # Create device-backed JAX array `position_plot`.
    position_plot = jnp.array(0.0)
    # Compute `losses` from `[]`
    losses = []
    # Repeat the update loop over `range(12)` steps:
    for _ in range(12):
        # Append the current step result to `losses`.
        losses.append(float((position_plot - 2) ** 2))
        # Compute `position_plot` from `position_plot - rate_plot * 2 * (position_plot - 2)`
        position_plot = position_plot - rate_plot * 2 * (position_plot - 2)
    # Append the current step result to `series`.
    series.append({'label': 'rate ' + str(rate_plot), 'y': losses})
# Compute `visual_data` from `{'kind': 'line', 'x': list(range(12)), 'xlabel': 'co...`
visual_data = {'kind': 'line', 'x': list(range(12)), 'xlabel': 'completed update', 'ylabel': 'squared error', 'yscale': 'symlog', 'series': series}
```

## Recorded reference execution

CPU run: 2026-10-08T14:02:04.851845+00:00. JAX 0.9.2.

```text
Final weight: 1.9999969005584717
Final loss: 9.606537787476555e-12
Final weight: 1.9999969005584717
Final loss: 9.606537787476555e-12
First two post-update losses: [2.5600002 1.6384   ]
PASS: optimization-03

```

## Check the first two steps

**Predict before running:** Does history contain the initial loss or the post-update loss?

```python
# Experiment — Check the first two steps: A hand-derived trajectory checks update sign and the recording...
two_w, two_losses = train(0., 0.1, 2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(two_w, 0.72)
# Check numerical equivalence within tolerance: `jnp.allclose(two_losses, jnp.array([2.56, 1.6384]))`
assert jnp.allclose(two_losses, jnp.array([2.56, 1.6384]))
# Print the observed values to compare against the expected result.
print("First two post-update losses:", two_losses)
```

**Expected:** $[2.56, 1.6384]$, with final weight $0.72$.

A hand-derived trajectory checks update sign and the recording convention.

## Test the edge of stability

**Predict before running:** What does rate $1$ do after five steps from zero?

```python
# Experiment — Test the edge of stability: Magnitude-one error multipliers do not contract even though the...
edge_w, edge_losses = train(0., 1., 5)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(edge_w, 4.)
# Check numerical equivalence within tolerance: `jnp.allclose(edge_losses, jnp.full(5, 4.))`
assert jnp.allclose(edge_losses, jnp.full(5, 4.))
```

**Expected:** The weight alternates $0$ and $4$; loss remains $4$.

Magnitude-one error multipliers do not contract even though the loop remains finite.

## A tiny update far from the solution

**Predict before running:** Can a parameter-change threshold of $10^{-6}$ declare success while the gradient magnitude is $4$?

```python
# Experiment — A tiny update far from the solution: A stopping rule must distinguish lack of progress from reaching...
# Construct `far` via `jnp.array(0.)`
far = jnp.array(0.)
# Compute `quadratic` from `lambda value: (value-2.)**2`
quadratic = lambda value: (value-2.)**2
# Differentiate the objective to obtain `grad_far` via automatic differentiation.
grad_far = jax.grad(quadratic)(far)
# Compute `next_far` from `far - 1e-8*grad_far`
next_far = far - 1e-8*grad_far
# Check numerical equivalence within tolerance: `jnp.abs(next_far-far)<1e-6`
assert jnp.abs(next_far-far)<1e-6
# Check numerical equivalence within tolerance: `jnp.abs(grad_far)>3.`
assert jnp.abs(grad_far)>3.
# Assert invariant `quadratic(next_far)>3.9` holds
assert quadratic(next_far)>3.9
```

**Expected:** The update is below the threshold, but the objective is still near $4$.

A stopping rule must distinguish lack of progress from reaching a useful solution.

## Make it yours

Run ten steps with learning rate $1.1$. Predict whether the error contracts before executing; show evidence of divergence.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `train(...)` — Call `train` with your updated parameters or inputs from this lesson's workspace.
- `loss(...)` — Call `loss` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Assert invariant `unstable_history[-1] > loss(0.)` holds
2. Check numerical equivalence within tolerance: `abs(1 - 2 * 1.1) > 1`

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Run ten steps with learning rate 1.1.
unstable, unstable_history = train(...)  # TODO: compute unstable, unstable_history
# Assert invariant `unstable_history[-1] > loss(0.)` holds
assert unstable_history[-1]  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `abs(1 - 2 * 1.1) > 1`
assert abs(1 - 2 * 1.1)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Run ten steps with learning rate 1.1.
unstable, unstable_history = train(0., 1.1, 10)
# Assert invariant `unstable_history[-1] > loss(0.)` holds
assert unstable_history[-1] > loss(0.)
# Check numerical equivalence within tolerance: `abs(1 - 2 * 1.1) > 1`
assert abs(1 - 2 * 1.1) > 1
```

</details>

## Transfer the stability analysis

**Transfer / diagnosis**

For $4(w-3)^2$, derive a stable rate and verify convergence from zero.

<details><summary>Hint</summary>

The error multiplier is $1-8\eta$; use $\eta=0.1$.

</details>

### How to write: Transfer the stability analysis — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.
- `jax.lax.scan(step_fn, init_carry, xs, length=...)` — Compiles a sequential loop where `step_fn(carry, x)` returns `(next_carry, y)`, returning `(final_carry, stacked_ys)`.

**Step-by-step implementation plan:**
1. Return `4 * (w - 3.0) ** 2` to the caller.
2. Define `scaled_step(w, _)` to evaluate the objective and its automatic derivatives:
3. Differentiate the objective to obtain `new` via automatic differentiation.
4. Return `(new, scaled_loss(new))` to the caller.
5. Run compiled structured control flow via `jax.lax` (`(scaled_w, scaled_history)`).

**Starter code scaffold (fill in the TODOs):**

```python
# Transfer the stability analysis (Transfer / diagnosis): Changing curvature changes the stability interval to...
def scaled_loss(w):
    # Return `4 * (w - 3.0) ** 2` to the caller.
    return ...  # TODO: return computed result
# Define `scaled_step(w, _)` to evaluate the objective and its automatic derivatives:
def scaled_step(w, _):
    # Differentiate the objective to obtain `new` via automatic differentiation.
    new = ...  # TODO: compute new
    # Return `(new, scaled_loss(new))` to the caller.
    return ...  # TODO: return computed result
# Run compiled structured control flow via `jax.lax` (`(scaled_w, scaled_history)`).
scaled_w, scaled_history = jax.lax.scan(...)  # TODO: compute scaled_w, scaled_history
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(scaled_w, 3., atol=1e-5)  # TODO: complete assertion check
# Assert invariant `jnp.all(jnp.diff(scaled_history) <= 1e-6)` holds
assert jnp.all(jnp.diff(scaled_history)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Transfer the stability analysis (Transfer / diagnosis): Changing curvature changes the stability interval to...
def scaled_loss(w):
    # Return `4 * (w - 3.0) ** 2` to the caller.
    return 4*(w-3.)**2
# Define `scaled_step(w, _)` to evaluate the objective and its automatic derivatives:
def scaled_step(w, _):
    # Differentiate the objective to obtain `new` via automatic differentiation.
    new = w-0.1*jax.grad(scaled_loss)(w)
    # Return `(new, scaled_loss(new))` to the caller.
    return new, scaled_loss(new)
# Run compiled structured control flow via `jax.lax` (`(scaled_w, scaled_history)`).
scaled_w, scaled_history = jax.lax.scan(scaled_step, jnp.array(0.), None, length=20)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(scaled_w, 3., atol=1e-5)
# Assert invariant `jnp.all(jnp.diff(scaled_history) <= 1e-6)` holds
assert jnp.all(jnp.diff(scaled_history) <= 1e-6)
```

Changing curvature changes the stability interval to $0<\eta<0.25$.

</details>

## Reproduce an uphill update and repair it

**Transfer / diagnosis**

Show how adding the gradient increases loss, then verify subtraction at the same starting point.

<details><summary>Hint</summary>

The derivative at zero is $-4$.

</details>

### How to write: Reproduce an uphill update and repair it — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Differentiate the objective to obtain `uphill` via automatic differentiation.
2. Differentiate the objective to obtain `downhill` via automatic differentiation.
3. Assert invariant `loss(uphill) > loss(0.)` holds
4. Assert invariant `loss(downhill) < loss(0.)` holds
5. Check numerical equivalence within tolerance: `jnp.allclose(downhill, 0.4)`

**Starter code scaffold (fill in the TODOs):**

```python
# Reproduce an uphill update and repair it (Transfer / diagnosis): The sign error is isolated using the same input and rate;...
# Differentiate the objective to obtain `uphill` via automatic differentiation.
uphill = ...  # TODO: compute uphill
# Differentiate the objective to obtain `downhill` via automatic differentiation.
downhill = ...  # TODO: compute downhill
# Assert invariant `loss(uphill) > loss(0.)` holds
assert loss(uphill)  # TODO: complete assertion check
# Assert invariant `loss(downhill) < loss(0.)` holds
assert loss(downhill)  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `jnp.allclose(downhill, 0.4)`
assert jnp.allclose(downhill, 0.4)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Reproduce an uphill update and repair it (Transfer / diagnosis): The sign error is isolated using the same input and rate;...
# Differentiate the objective to obtain `uphill` via automatic differentiation.
uphill = 0. + 0.1*jax.grad(loss)(0.)
# Differentiate the objective to obtain `downhill` via automatic differentiation.
downhill = 0. - 0.1*jax.grad(loss)(0.)
# Assert invariant `loss(uphill) > loss(0.)` holds
assert loss(uphill) > loss(0.)
# Assert invariant `loss(downhill) < loss(0.)` holds
assert loss(downhill) < loss(0.)
# Check numerical equivalence within tolerance: `jnp.allclose(downhill, 0.4)`
assert jnp.allclose(downhill, 0.4)
```

The sign error is isolated using the same input and rate; the repaired first step matches the analytic result.

</details>

## Check your understanding

What makes rate $1.1$ unstable for this particular quadratic?

1. The error multiplier has magnitude greater than one
2. scan cannot differentiate scalar weights
3. All learning rates above $0.01$ are unstable

<details><summary>Answer and explanation</summary>

The error multiplier has magnitude greater than one

The recurrence multiplies the error by $-1.2$, so its magnitude grows while its sign alternates.

</details>

## Diagnose the result

Record the loss trajectory and parameter values. An alternating, growing error here follows from the update equation; changing the PRNG seed cannot fix it.

## Carry forward

- A hand-derived trajectory checks update sign and the recording convention.
- Magnitude-one error multipliers do not contract even though the loop remains finite.

## Keep your evidence

Keep the two hand-computed steps, rate stability derivation, edge/divergent runs, scaled-curvature transfer and uphill-update repair. Include the tiny-update counterexample and explain your stopping criteria.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [lax.scan contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [grad API](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)

