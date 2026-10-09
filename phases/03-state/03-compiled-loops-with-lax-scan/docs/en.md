# Compiled loops with lax.scan

Phase 03: State, randomness & control flow · about 65 minutes · CPU

## What you will be able to do

- Separate carry from recorded outputs
- Keep the carry contract invariant
- Differentiate the recurrence by hand
- Compile the repeated transition without a timing promise

## The problem

Suppose a simulator keeps part of its current value and adds a new input at each step. Let’s write out four steps so we know what should happen. Then we’ll express the same loop with scan, compare it with Python, and see how changing the retained fraction affects the final value.

## The idea

A scan separates the state carried between steps from the outputs collected at each step. This is useful whenever the same rule repeats over time while the state keeps a fixed structure, shape and dtype.

## Unroll three steps before compiling the loop

For a separate hand-worked recurrence, start with carry $0$ and inputs $[1,2,3]$. Let each step add its input to the carry and emit the new carry. The successive carries are $1,3,6$; the final carry is $6$, while the stacked outputs are $[1,3,6]$.

Those two returned objects answer different questions. Keeping only the final carry discards the intermediate outputs. Keeping outputs does not mean the carry itself grows on each iteration. A fixed carry contract is what makes the repeated computation structured.

Trace one horizontal state edge and one downward output edge in the diagram. When adapting a Python loop, write down which variables belong on each edge before translating the loop to `lax.scan`.

### Carry moves through time; outputs are collected

**Predict:** What changes if the step emits the old carry instead of the new carry?

![Carry moves through time; outputs are collected](../outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

An additive scan starts with carry $0$ and inputs $1,2,3$. Horizontal arrows pass updated carry; downward arrows collect emitted values. Final carry $6$ and outputs $[1,3,6]$ are different return values. Arrow lengths do not represent runtime.

### Pause and reason

What changes if the step emits the old carry instead of the new carry?

<details><summary>Compare your reasoning</summary>

The final carry is still $6$, but outputs become $[0,1,3]$. A final-state check alone would miss this off-by-one output convention.

</details>

## Separate carry from recorded outputs

Each scan step receives (carry,input) and returns (next_carry,output). Carry is what the next step needs. Output is what you want stacked into a history. They can have different meanings even when this demonstration returns the new state in both positions.

For decay $0.5$, initial state $0$ and four inputs of $1$, the states are $1$,$1.5$,$1.75$,$1.875$. There are four recorded outputs; the initial state is not automatically part of history. Decide whether a plot should include that initial point before labeling the time axis.

```text
(state_t, input_t) → transition → (state_(t+1), recorded_t)
state: 0 → 1 → 1.5 → 1.75 → 1.875
```

## Keep the carry contract invariant

Carry values change, but their pytree structure, shapes and dtypes must remain compatible across steps. Growing a Python list or concatenating a larger array into the carry changes that contract. Allocate a fixed-size state or use the separately stacked outputs instead.

This constraint is a representation decision, not an arbitrary restriction on simulation. A fixed-dimensional physical state naturally meets it. If a system has variable numbers of entities, consider padding and masks with an explicit maximum capacity.

## Differentiate the recurrence by hand

With four unit inputs and initial zero, the final value is $1+d+d^2+d^3$. Its derivative is $1+2d+3d^2$, giving $2.75$ at $d=0.5$. The derivative also obeys a recurrence: $s_{t+1}=\mathrm{state}_t+d s_t$, starting from $s_0=0$.

Autodiff through scan differentiates the represented state transition. It does not make an unstable or inaccurate numerical model valid. Verify a small analytic case first, then a changed input sequence, before trusting a longer simulation.

$$
\begin{aligned}s_{\mathrm{final}}(d)&=1+d+d^2+d^3\\\frac{ds_{\mathrm{final}}}{dd}&=1+2d+3d^2\\s_{\mathrm{final}}(0.5)&=1.875,\qquad s_{\mathrm{final}}\prime(0.5)=2.75\end{aligned}
$$

## Compile the repeated transition without a timing promise

A Python loop inside jit can be unrolled while tracing; scan expresses a repeated operation compactly. That can reduce compilation work for long loops, but it is not a universal promise that scan is faster for every workload.

Before benchmarking, compare numerical results on the same inputs and synchronize execution. Keep step count and precision explicit. For this lesson we verify correctness and derivatives; hardware profiling belongs to a later course unit.

## Prepare the inputs

Create main.py in your lesson workspace. Add this first block; use the environment from setup.

```python
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
```

These explicit inputs define the case that the later checks will verify.

## Build the computation

Append this block below the inputs in the same file.

```python
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    # Compute `next_value` from `0.5 * carry + increment`
    next_value = 0.5 * carry + increment
    # Return `(next_value, next_value)` to the caller.
    return next_value, next_value
# Construct `increments` via `jnp.array([1., 1., 1., 1.])`
increments = jnp.array([1., 1., 1., 1.])
```

step returns the next scalar carry and one recorded scalar. scan feeds that carry back into step and stacks the four recorded outputs.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Run compiled structured control flow via `jax.lax` (`(final, history)`).
final, history = jax.lax.scan(step, jnp.array(0.), increments)
# Print the observed values to compare against the expected result.
print("History:", history)
# Print diagnostic summary of the computed outputs.
print("Final:", float(final))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
# Check numerical equivalence within tolerance: `jnp.allclose(final, history[-1])`
assert jnp.allclose(final, history[-1])
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: step returns the next scalar carry and one recorded scalar.
def step(carry, increment):
    # Compute `next_value` from `0.5 * carry + increment`
    next_value = 0.5 * carry + increment
    # Return `(next_value, next_value)` to the caller.
    return next_value, next_value
# Construct `increments` via `jnp.array([1., 1., 1., 1.])`
increments = jnp.array([1., 1., 1., 1.])
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Run compiled structured control flow via `jax.lax` (`(final, history)`).
final, history = jax.lax.scan(step, jnp.array(0.), increments)
# Print the observed values to compare against the expected result.
print("History:", history)
# Print diagnostic summary of the computed outputs.
print("Final:", float(final))
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
# Check numerical equivalence within tolerance: `jnp.allclose(final, history[-1])`
assert jnp.allclose(final, history[-1])
```

Expected: History: $[1., 1.5, 1.75, 1.875]$; Final: $1.875$.

## The carry approaches a fixed point

**Predict:** Why does the increment between successive states shrink?

![The carry approaches a fixed point](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The solid line shows the carry after each completed scan step. It starts at step $1$ with value $1$, then rises through $1.5$, $1.75$, and $1.875$. The dashed horizontal line marks the fixed point $2$.

The initial carry $0$ is not plotted. Each visible point is the value after applying the recurrence once more, so the first point already includes one update.

### Connect it to the computation

The update is $c_{t+1}=0.5c_t+1$. Subtracting it from the fixed point gives $2-c_{t+1}=0.5(2-c_t)$: each step halves the remaining gap. This explains both the upward movement and its slowing pace.

The shrinking gaps are $1$, $0.5$, $0.25$, and $0.125$. `scan` carries the evolving value forward while collecting this history. One more step would produce $1.9375$; it would not add another full unit or restart from the initial carry.

```python
# Compute figure data for: The carry approaches a fixed point
# Compute `visual_data` from `{'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'compl...`
visual_data = {'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'completed step', 'ylabel': 'carry value', 'series': [{'label': 'scan history', 'y': history.tolist()}, {'label': 'fixed point 2', 'y': [2.0] * 4}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:01:41.526895+00:00. JAX 0.9.2.

```text
History: [1.    1.5   1.75  1.875]
Final: 1.875
History: [1.    1.5   1.75  1.875]
Final: 1.875
Expected carry-shape mismatch
PASS: state-03

```

## Compare scan to the Python recurrence

**Predict before running:** Predict every output for inputs $[1,-1,2,0]$ and decay $0.5$.

```python
# Experiment — Compare scan to the Python recurrence: The loop is an independent control-flow reference and the fixed...
# Construct `inputs` via `jnp.array([1.,-1.,2.,0.])`
inputs=jnp.array([1.,-1.,2.,0.])
# Compute `state` from `0.`
state=0.
# Compute `reference` from `[]`
reference=[]
# Iterate over `inc` to step through the computation:
for inc in [1.,-1.,2.,0.]:
    # Compute `state` from `0.5*state+inc`
    state=0.5*state+inc
    # Append the current step result to `reference`.
    reference.append(state)
# Run compiled structured control flow via `jax.lax` (`(last, observed)`).
last,observed=jax.lax.scan(step,jnp.array(0.),inputs)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(observed,jnp.array([1.,-0.5,1.75,0.875]))
# Check numerical equivalence within tolerance: `jnp.allclose(observed,jnp.array(reference))`
assert jnp.allclose(observed,jnp.array(reference))
# Check numerical equivalence within tolerance: `jnp.allclose(last,state)`
assert jnp.allclose(last,state)
```

**Expected:** History $[1,-0.5,1.75,0.875]$; final $0.875$.

The loop is an independent control-flow reference and the fixed numbers verify time indexing.

## Verify state and sensitivity together

**Predict before running:** How do the state and derivative recurrences differ?

```python
# Experiment — Verify state and sensitivity together: The scalar objective is the final state.
# Define `terminal(d)` to carry state across steps with `jax.lax.scan`:
def terminal(d):
    # Return `jax.lax.scan(lambda c, u: (d * c + u, d * c + u), jnp.array(0.0), increments)[0]` to the caller.
    return jax.lax.scan(lambda c,u:(d*c+u,d*c+u),jnp.array(0.),increments)[0]
# Differentiate the objective to obtain `(value, sensitivity)` via automatic differentiation.
value,sensitivity=jax.value_and_grad(terminal)(0.5)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(value,1.875)
# Check numerical equivalence within tolerance: `jnp.allclose(sensitivity,2.75)`
assert jnp.allclose(sensitivity,2.75)
# Check numerical equivalence within tolerance: `jnp.allclose(jax.jit(terminal)(0.5),value)`
assert jnp.allclose(jax.jit(terminal)(0.5),value)
```

**Expected:** Final state $1.875$ and sensitivity $2.75$.

The scalar objective is the final state. Returning the whole history to ordinary grad would need a deliberate scalar reduction.

## Make it yours

Make the decay an explicit scalar argument. Differentiate the final value with respect to decay and compare at $0.5$ with the analytic derivative $2.75$.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.
- `jax.lax.scan(step_fn, init_carry, xs, length=...)` — Compiles a sequential loop where `step_fn(carry, x)` returns `(next_carry, y)`, returning `(final_carry, stacked_ys)`.

**Step-by-step implementation plan:**
1. Define `simulate(decay)` to carry state across steps with `jax.lax.scan`:
2. Function `transition(carry, increment)` implementing this stage's computation:
3. Compute `value` from `decay * carry + increment`
4. Return `(value, value)` to the caller.
5. Return `jax.lax.scan(transition, jnp.array(0.0), increments)[0]` to the caller.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Make the decay an explicit scalar argument.
# Define `simulate(decay)` to carry state across steps with `jax.lax.scan`:
def simulate(decay):
    # Function `transition(carry, increment)` implementing this stage's computation:
    def transition(carry, increment):
        # Compute `value` from `decay * carry + increment`
        value = ...  # TODO: compute value
        # Return `(value, value)` to the caller.
        return ...  # TODO: return computed result
    # Return `jax.lax.scan(transition, jnp.array(0.0), increments)[0]` to the caller.
    return ...  # TODO: return computed result
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(simulate)(0.5), 2.75)  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Make the decay an explicit scalar argument.
# Define `simulate(decay)` to carry state across steps with `jax.lax.scan`:
def simulate(decay):
    # Function `transition(carry, increment)` implementing this stage's computation:
    def transition(carry, increment):
        # Compute `value` from `decay * carry + increment`
        value = decay * carry + increment
        # Return `(value, value)` to the caller.
        return value, value
    # Return `jax.lax.scan(transition, jnp.array(0.0), increments)[0]` to the caller.
    return jax.lax.scan(transition, jnp.array(0.), increments)[0]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(simulate)(0.5), 2.75)
```

</details>

## Record a different output

**Practice**

Carry the state, but record its squared value. Confirm that final carry still means state.

<details><summary>Hint</summary>

The second returned value is only the recorded output.

</details>

### How to write: Record a different output — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.lax.scan(step_fn, init_carry, xs, length=...)` — Compiles a sequential loop where `step_fn(carry, x)` returns `(next_carry, y)`, returning `(final_carry, stacked_ys)`.

**Step-by-step implementation plan:**
1. Run compiled structured control flow via `jax.lax` (`(last, energy)`).
2. Verify that the numerical values match the expected reference within tolerance.
3. Check numerical equivalence within tolerance: `jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))`

**Starter code scaffold (fill in the TODOs):**

```python
# Record a different output (Practice): Carry and history need not have the same interpretation.
# Run compiled structured control flow via `jax.lax` (`(last, energy)`).
last,energy = jax.lax.scan(...)  # TODO: compute last,energy
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(last,1.875)  # TODO: complete assertion check
# Check numerical equivalence within tolerance: `jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))`
assert jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Record a different output (Practice): Carry and history need not have the same interpretation.
# Run compiled structured control flow via `jax.lax` (`(last, energy)`).
last,energy=jax.lax.scan(lambda c,u:(0.5*c+u,(0.5*c+u)**2),jnp.array(0.),increments)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(last,1.875)
# Check numerical equivalence within tolerance: `jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))`
assert jnp.allclose(energy,jnp.array([1.,2.25,3.0625,3.515625]))
```

Carry and history need not have the same interpretation. The state remains unsquared for the next transition.

</details>

## Diagnose a growing carry

**Challenge**

Try appending each input to the carry with concatenate. Explain why scan rejects it and replace it with stacked output.

<details><summary>Hint</summary>

The carry shape is part of the loop type.

</details>

### How to write: Diagnose a growing carry — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jax.lax.scan(step_fn, init_carry, xs, length=...)` — Compiles a sequential loop where `step_fn(carry, x)` returns `(next_carry, y)`, returning `(final_carry, stacked_ys)`.

**Step-by-step implementation plan:**
1. Return `(jnp.concatenate([c, u[None]]), u)` to the caller.
2. Run the boundary check and catch the expected exception:
3. Run compiled structured control flow via `jax.lax` (`(_, saved)`).
4. Assert invariant `jnp.array_equal(saved,increments)` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Diagnose a growing carry (Challenge): The error identifies incompatible carry input/output types.
def growing(c,u):
    # Return `(jnp.concatenate([c, u[None]]), u)` to the caller.
    return ...  # TODO: return computed result
# Run the boundary check and catch the expected exception:
try:
    jax.lax.scan(growing,jnp.zeros((0,)),increments)
except TypeError:
    print("Expected carry-shape mismatch")
else:
    raise AssertionError("Expected scan type failure")
# Run compiled structured control flow via `jax.lax` (`(_, saved)`).
_,saved = jax.lax.scan(...)  # TODO: compute _,saved
# Assert invariant `jnp.array_equal(saved,increments)` holds
assert jnp.array_equal(saved,increments)  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Diagnose a growing carry (Challenge): The error identifies incompatible carry input/output types.
def growing(c,u):
    # Return `(jnp.concatenate([c, u[None]]), u)` to the caller.
    return jnp.concatenate([c,u[None]]),u
# Run the boundary check and catch the expected exception:
try:
    jax.lax.scan(growing,jnp.zeros((0,)),increments)
except TypeError:
    print("Expected carry-shape mismatch")
else:
    raise AssertionError("Expected scan type failure")
# Run compiled structured control flow via `jax.lax` (`(_, saved)`).
_,saved=jax.lax.scan(lambda c,u:(c,u),jnp.array(0.),increments)
# Assert invariant `jnp.array_equal(saved,increments)` holds
assert jnp.array_equal(saved,increments)
```

The error identifies incompatible carry input/output types. Store per-step values in scan outputs instead of growing the carry.

</details>

## Check your understanding

What must remain consistent across scan steps?

1. The numerical value of the carry
2. The carry structure, shapes, and dtypes
3. Every recorded output must be zero

<details><summary>Answer and explanation</summary>

The carry structure, shapes, and dtypes

The carry value changes through time, but its structural and array type properties must remain consistent.

</details>

## Diagnose the result

The error identifies incompatible carry input/output types. Store per-step values in scan outputs instead of growing the carry.

## Carry forward

- The loop is an independent control-flow reference and the fixed numbers verify time indexing.
- The scalar objective is the final state. Returning the whole history to ordinary grad would need a deliberate scalar reduction.

## Keep your evidence

Keep the labeled carry/output transition, Python-reference trajectory, polynomial derivative, compiled value check and growing-carry failure with a fixed-shape repair.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX scan contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [JAX control flow](https://docs.jax.dev/en/latest/control-flow.html)

