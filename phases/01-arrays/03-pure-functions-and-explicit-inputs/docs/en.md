# Pure functions and explicit inputs

Phase 01: Arrays & pure functions · about 45 minutes · CPU

## What you will be able to do

- Identify hidden inputs and side effects in a numerical function.
- Pass model parameters explicitly and return new state without mutating input containers.
- Test replay and independence between two parameter configurations.
- Explain why host orchestration and transformed numerical computation need different boundaries.

## The problem

Have you ever rerun a notebook cell and got a different result because another cell changed a variable? Let’s make that dependency visible. We’ll pass a model’s weights into its prediction function, return new values explicitly, and check that the same inputs reproduce the same calculation.

## The idea

A pure function makes its changing inputs visible and returns its result without secretly changing shared state. That gives transformations such as differentiation and compilation a clear computation to work with. When a prediction changes, we should be able to identify the input that changed.

## Make a repeated call explainable

Imagine a prediction function that receives features and weights but reads a global bias. You compile it, change that Python global, and call the compiled function again. The source file now displays the new bias, but the already-traced computation may still embody the old value. The hidden dependency makes the result difficult to reason about.

Pass the bias as an explicit argument instead. Then two calls with different bias arrays are visibly two computations with different data. For a linear prediction, increasing the bias by $1$ must increase every output by $1$, while changing the weight vector can affect rows differently.

Purity does not mean a whole application has no state. It means the numerical function receives the state it needs and returns any new state. Files, logs and orchestration can live outside that numerical boundary.

$$
X'[i] = \begin{cases} v & \text{if } i = k \\ X[i] & \text{otherwise} \end{cases}
$$

### Pause and reason

Where should a changing training step or random key live?

<details><summary>Compare your reasoning</summary>

In explicit state passed into the computation and returned when updated. Hiding it in a mutable Python global makes replay and transformed execution harder to verify.

</details>

## Hidden state makes the signature incomplete

A function that takes $x$ but reads a global weight has two effective inputs while advertising only one. Re-running it after changing the global can change its result with no visible argument change. Tests may depend on the order in which prior code ran.

Passing the weight explicitly makes the dependency visible. It also allows two model configurations to coexist without rewriting global variables. That matters for evaluation, comparisons, checkpoint recovery, and eventually multiple workers.

## Array immutability does not make a dictionary immutable

JAX arrays are immutable values, but a Python dictionary holding them is still a mutable container. Assigning `params["weight"] = ...` changes which value the dictionary holds. If another variable refers to the same dictionary, it sees the changed entry.

Constructing `{**params, "weight": new_weight}` makes a new dictionary with a changed binding and keeps the old container intact. This is a shallow copy: nested mutable objects can still be shared. Our example has only array leaves, so the shallow update is sufficient for the declared structure. More general parameter trees will need deliberate tree-aware transformations.

## Treat a step as an input/output contract

A training step will eventually consume parameters, optimizer state, and a batch, then return updated state and metrics. The current predictor is the simpler first piece: parameters and data in, predictions out. The caller owns the history and can save both original and updated values.

This structure makes replay a meaningful test. With the same inputs, predictions should agree; changing the weight should affect predictions in the expected way. Do not make a test depend on a global that may already have been edited elsewhere in a notebook.

```text
predict(parameters, observations) → predictions
future step(state, batch) → new_state, metrics
```

## Choose a clear transformation boundary

A print, file write, or list append is a Python side effect. Inside a transformed function, such effects can happen during tracing rather than every device execution. We will investigate that mechanism in the tracing lesson. For now, separate producing metrics from reporting them.

Return the values you need to inspect, then let the caller print or save them. This design also makes unit-sized numerical checks simpler: you can compare actual outputs with known arrays without parsing console messages or relying on side-effect timing.

## A deterministic check has limits

Calling a function twice with fixed inputs checks a useful property, but it does not prove the absence of every hidden dependency. A hidden global might happen not to change. Inspect the implementation and deliberately exercise multiple configurations. Randomness will need an explicit input as well; that is why the next phase introduces keys rather than treating random draws as invisible global state.

Keep the evidence proportional to the claim: these tests establish the behavior of this predictor and update pattern, not the replay of an entire distributed training job.

## Step 1: Set up imports and input tensors

Import the required JAX modules and define the initial inputs for pure functions and explicit inputs.

```python
import jax.numpy as jnp
```

Establishing explicit input shapes and dtypes first makes the downstream transformation contract deterministic.

## Step 2: Apply the core JAX transformation

Write the core computation and transformation step over the initialized inputs.

```python
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params["weight"] * x + params["bias"]
# Construct `params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
```

This stage executes the primary numerical transformation and binds the intermediate outputs.

## Step 3: Verify shapes and numerical invariants

Check that the resulting arrays satisfy the expected shape, dtype, and numerical tolerances.

```python
x = jnp.array([0., 1., 2.])
# Print the observed values to compare against the expected result.
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))
```

These assertions lock in the exact numerical contract before you run the full experiment and variations.

## Run the example

```python
# Pure functions and explicit inputs: A pure function makes its changing inputs visible and returns...
# Import jax.numpy for this computation.
import jax.numpy as jnp
# Function `predict(params, x)` implementing this stage's computation:
def predict(params, x):
    # Return `params['weight'] * x + params['bias']` to the caller.
    return params["weight"] * x + params["bias"]
# Construct `params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Construct `x` via `jnp.array([0., 1., 2.])`
x = jnp.array([0., 1., 2.])
# Print the observed values to compare against the expected result.
print(predict(params, x))
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))
```

Expected: [$1$. $3$. $5$.]

## Explicit parameters make predictions inspectable

**Predict:** What changes when the bias increases by $2$?

![Explicit parameters make predictions inspectable](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is the input and the vertical axis is the prediction. The solid line uses bias $1$; the dashed line uses bias $3$. Both use the same weight $2$.

At input $0$, the predictions are $1$ and $3$. At input $2$, they are $5$ and $7$. The gap stays $2$ everywhere, and the parallel lines rise by $2$ for each unit increase in the input.

### Connect it to the computation

In $y=wx+b$, changing the explicit bias changes the vertical offset, while the unchanged weight preserves the slope. You can read the bias directly where the line meets the vertical axis because the product $wx$ is zero there.

This makes the function’s inputs inspectable: the change in the output follows the change in the passed parameter. As a check, setting the bias to $0$ would move the solid line down by $1$ and make it pass through the origin. It would not flatten the line.

```python
# Compute figure data for: Explicit parameters make predictions inspectable
# Create device-backed JAX array `changed`.
changed = {**params, 'bias': jnp.array(3.0)}
# Compute `visual_data` from `{'kind': 'line', 'x': x.tolist(), 'xlabel': 'input',...`
visual_data = {'kind': 'line', 'x': x.tolist(), 'xlabel': 'input', 'ylabel': 'prediction', 'series': [{'label': 'bias 1', 'y': predict(params, x).tolist()}, {'label': 'bias 3', 'y': predict(changed, x).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-09T14:07:36.786190+00:00. JAX 0.9.2.

```text
[1. 3. 5.]
aliased container weight: 9.0
evaluation losses: 0.0 6.0
PASS: arrays-03

```

## Expose a mutable container alias

**Predict before running:** If `alias = params`, does assigning a new weight through alias preserve the original dictionary? Predict before running on a separate example.

```python
# Experiment — Expose a mutable container alias: The mutation concerns Python container bindings.
# Construct `container` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
container = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Compute `alias` from `container`
alias = container
# Construct `alias["weight"]` via `jnp.array(9.)`
alias["weight"] = jnp.array(9.)
# Assert invariant `float(container["weight"]) == 9.` holds
assert float(container["weight"]) == 9.
# Assert invariant `float(params["weight"]) == 2.` holds
assert float(params["weight"]) == 2.
# Print the observed values to compare against the expected result.
print("aliased container weight:", float(container["weight"]))
```

**Expected:** The aliased container changes to weight $9$; the separate original lesson parameters remain $2$.

The mutation concerns Python container bindings. It does not contradict JAX array immutability.

## Replay two independent model configurations

**Predict before running:** What should each parameter setting predict, regardless of which was called first?

```python
# Experiment — Replay two independent model configurations: Independent configurations reveal the benefit of explicit inputs...
# Construct `first_params` via `{"weight": jnp.array(2.), "bias": jnp.array(1.)}`
first_params = {"weight": jnp.array(2.), "bias": jnp.array(1.)}
# Construct `second_params` via `{"weight": jnp.array(-1.), "bias": jnp.array(4.)}`
second_params = {"weight": jnp.array(-1.), "bias": jnp.array(4.)}
# Run `predict` to compute `first_output`.
first_output = predict(first_params, x)
# Run `predict` to compute `second_output`.
second_output = predict(second_params, x)
# Assert that `jnp.allclose(first_output, jnp.array([1., 3., 5.]))`.
assert jnp.allclose(first_output, jnp.array([1., 3., 5.]))
# Assert that `jnp.allclose(second_output, jnp.array([4., 3., 2.]))`.
assert jnp.allclose(second_output, jnp.array([4., 3., 2.]))
# Assert that `jnp.allclose(predict(first_params, x), first_output)`.
assert jnp.allclose(predict(first_params, x), first_output)
# Assert that `jnp.allclose(predict(second_params, x), second_output)`.
assert jnp.allclose(predict(second_params, x), second_output)
```

**Expected:** The two configurations produce $[1, 3, 5]$ and $[4, 3, 2]$, and both can be replayed.

Independent configurations reveal the benefit of explicit inputs more clearly than one call on one global parameter set.

## Make it yours

Create new parameters with weight $3$ while retaining the original bias. Show that the original parameters still produce the original predictions.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Construct `updated` via `{**params, "weight": jnp.array(3.)}`
2. Assert that `jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))`.
3. Assert invariant `float(params["weight"]) == 2.0` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Create new parameters with weight 3 while retaining the original bias.
# Construct `updated` via `{**params, "weight": jnp.array(3.)}`
updated = ...  # TODO: compute updated
# Assert that `jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))`.
assert jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))  # TODO: complete assertion check
# Assert invariant `float(params["weight"]) == 2.0` holds
assert float(params["weight"])  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Create new parameters with weight 3 while retaining the original bias.
# Construct `updated` via `{**params, "weight": jnp.array(3.)}`
updated = {**params, "weight": jnp.array(3.)}
# Assert that `jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))`.
assert jnp.allclose(predict(updated, x), jnp.array([1.,4.,7.]))
# Assert invariant `float(params["weight"]) == 2.0` holds
assert float(params["weight"]) == 2.0
```

</details>

## Update the bias without changing either parameter input

**Practice**

Implement `with_bias(params, new_bias)`, returning a new dictionary. Verify outputs for new bias $-2$ and replay the original prediction.

<details><summary>Hint</summary>

A new dictionary is required; changing a binding in the original dictionary breaks the promised contract.

</details>

### How to write: Update the bias without changing either parameter input — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Return `{**parameters, 'bias': jnp.asarray(new_bias)}` to the caller.
2. Run `with_bias` to compute `changed_bias`.
3. Assert invariant `changed_bias is not params` holds
4. Assert that `jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))`.
5. Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.

**Starter code scaffold (fill in the TODOs):**

```python
# Update the bias without changing either parameter input (Practice): The caller can evaluate both the old and new model without...
def with_bias(parameters, new_bias):
    # Return `{**parameters, 'bias': jnp.asarray(new_bias)}` to the caller.
    return ...  # TODO: return computed result
# Run `with_bias` to compute `changed_bias`.
changed_bias = with_bias(...)  # TODO: compute changed_bias
# Assert invariant `changed_bias is not params` holds
assert changed_bias is not params  # TODO: complete assertion check
# Assert that `jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))`.
assert jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))  # TODO: complete assertion check
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Update the bias without changing either parameter input (Practice): The caller can evaluate both the old and new model without...
def with_bias(parameters, new_bias):
    # Return `{**parameters, 'bias': jnp.asarray(new_bias)}` to the caller.
    return {**parameters, "bias": jnp.asarray(new_bias)}
# Run `with_bias` to compute `changed_bias`.
changed_bias = with_bias(params, -2.)
# Assert invariant `changed_bias is not params` holds
assert changed_bias is not params
# Assert that `jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))`.
assert jnp.allclose(predict(changed_bias, x), jnp.array([-2., 0., 2.]))
# Assert that `jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))`.
assert jnp.allclose(predict(params, x), jnp.array([1., 3., 5.]))
```

The caller can evaluate both the old and new model without restoring a global value or depending on call order.

</details>

## Return predictions and metrics together

**Challenge**

Write `evaluate(params, inputs, targets)` that returns predictions and a scalar MSE. Test two model configurations on targets $[1, 3, 5]$. Keep printing outside the function.

<details><summary>Hint</summary>

Compute data inside the numerical core and report it outside. A tuple return does not itself imply mutation.

</details>

### How to write: Return predictions and metrics together — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Run `predict` to compute `predictions`.
2. Return `(predictions, jnp.mean((predictions - targets) ** 2))` to the caller.
3. Construct `labels` via `jnp.array([1., 3., 5.])`
4. Run `evaluate` to compute `(outputs, error)`.
5. Run `evaluate` to compute `(other_outputs, other_error)`.

**Starter code scaffold (fill in the TODOs):**

```python
# Return predictions and metrics together (Challenge): The second model has residuals [3, 0, -3], giving mean...
def evaluate(parameters, inputs, targets):
    # Run `predict` to compute `predictions`.
    predictions = predict(...)  # TODO: compute predictions
    # Return `(predictions, jnp.mean((predictions - targets) ** 2))` to the caller.
    return ...  # TODO: return computed result
# Construct `labels` via `jnp.array([1., 3., 5.])`
labels = jnp.array(...)  # TODO: compute labels
# Run `evaluate` to compute `(outputs, error)`.
outputs, error = evaluate(...)  # TODO: compute outputs, error
# Run `evaluate` to compute `(other_outputs, other_error)`.
other_outputs, other_error = evaluate(...)  # TODO: compute other_outputs, other_error
# Check tensor shape invariant: `error.shape == ()`
assert error.shape  # TODO: complete assertion check
# Assert that `jnp.allclose(error, 0.)`.
assert jnp.allclose(error, 0.)  # TODO: complete assertion check
# Assert that `jnp.allclose(other_error, 6.)`.
assert jnp.allclose(other_error, 6.)  # TODO: complete assertion check
# Assert that `jnp.allclose(outputs, first_output)`.
assert jnp.allclose(outputs, first_output)  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("evaluation losses:", float(error), float(other_error))
```

<details><summary>Reference solution and reasoning</summary>

```python
# Return predictions and metrics together (Challenge): The second model has residuals [3, 0, -3], giving mean...
def evaluate(parameters, inputs, targets):
    # Run `predict` to compute `predictions`.
    predictions = predict(parameters, inputs)
    # Return `(predictions, jnp.mean((predictions - targets) ** 2))` to the caller.
    return predictions, jnp.mean((predictions - targets) ** 2)
# Construct `labels` via `jnp.array([1., 3., 5.])`
labels = jnp.array([1., 3., 5.])
# Run `evaluate` to compute `(outputs, error)`.
outputs, error = evaluate(first_params, x, labels)
# Run `evaluate` to compute `(other_outputs, other_error)`.
other_outputs, other_error = evaluate(second_params, x, labels)
# Check tensor shape invariant: `error.shape == ()`
assert error.shape == ()
# Assert that `jnp.allclose(error, 0.)`.
assert jnp.allclose(error, 0.)
# Assert that `jnp.allclose(other_error, 6.)`.
assert jnp.allclose(other_error, 6.)
# Assert that `jnp.allclose(outputs, first_output)`.
assert jnp.allclose(outputs, first_output)
# Print the observed values to compare against the expected result.
print("evaluation losses:", float(error), float(other_error))
```

The second model has residuals $[3, 0, -3]$, giving mean squared error $6$. Known output arithmetic makes the check independent of the implementation.

</details>

## Check your understanding

What makes predict easier to transform and test?

1. Reading an editable global weight
2. Passing parameters and data explicitly
3. Printing inside every numerical operation

<details><summary>Answer and explanation</summary>

Passing parameters and data explicitly

Explicit inputs expose the computation and let you compare calls with controlled parameter values.

</details>

## Diagnose the result

If a replay differs, inspect hidden globals, mutable aliases, and random state before assuming nondeterministic hardware. If an update affects an old model, check whether the caller reused the same dictionary. If logs disappear under a transform, inspect whether they are Python side effects. Preserve numerical results as return values so you can investigate without relying on side-effect behavior.

## Carry forward

- Explicit inputs make model dependencies visible and configurations independent.
- Immutable arrays can still be stored in mutable Python containers.
- Return new state deliberately and let the caller own it.
- Keep numerical computation separate from host logging, I/O, and orchestration.

## Keep your evidence

Keep both parameter configurations, their known predictions, a container-alias experiment, a non-mutating update, and an evaluation loss checked by hand. Explain which state the caller owns.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX: pure functions and common gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)
- [JAX: stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html)

