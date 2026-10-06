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

lax.scan expresses a loop as a state transition. Each step receives the carry and one input, then returns a new carry and one output. Scan combines these outputs and returns the final carry. This separates state carried through time from values recorded at each step.

The carry must keep a consistent structure, shape, and dtype. Unlike a long Python loop traced inside jit, scan represents the repeated computation as a loop operation. It is useful for recurrent models, simulations, and repeated optimization steps.

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
import jax
import jax.numpy as jnp
```

These explicit inputs define the case that the later checks will verify.

## Build the computation

Append this block below the inputs in the same file.

```python
def step(carry, increment):
    next_value = 0.5 * carry + increment
    return next_value, next_value
increments = jnp.array([1., 1., 1., 1.])
```

step returns the next scalar carry and one recorded scalar. scan feeds that carry back into step and stacks the four recorded outputs.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
final, history = jax.lax.scan(step, jnp.array(0.), increments)
print("History:", history)
print("Final:", float(final))
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
assert jnp.allclose(final, history[-1])
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
import jax
import jax.numpy as jnp
def step(carry, increment):
    next_value = 0.5 * carry + increment
    return next_value, next_value
increments = jnp.array([1., 1., 1., 1.])
final, history = jax.lax.scan(step, jnp.array(0.), increments)
print("History:", history)
print("Final:", float(final))
assert jnp.allclose(history, jnp.array([1.,1.5,1.75,1.875]))
assert jnp.allclose(final, history[-1])
```

Expected: History: $[1., 1.5, 1.75, 1.875]$; Final: $1.875$.

## The carry approaches a fixed point

**Predict:** Why does the increment between successive states shrink?

![The carry approaches a fixed point](../../phases/03-state/03-compiled-loops-with-lax-scan/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The solid line shows the carry after each completed scan step. It starts at step $1$ with value $1$, then rises through $1.5$, $1.75$, and $1.875$. The dashed horizontal line marks the fixed point $2$.

The initial carry $0$ is not plotted. Each visible point is the value after applying the recurrence once more, so the first point already includes one update.

### Connect it to the computation

The update is $c_{t+1}=0.5c_t+1$. Subtracting it from the fixed point gives $2-c_{t+1}=0.5(2-c_t)$: each step halves the remaining gap. This explains both the upward movement and its slowing pace.

The shrinking gaps are $1$, $0.5$, $0.25$, and $0.125$. `scan` carries the evolving value forward while collecting this history. One more step would produce $1.9375$; it would not add another full unit or restart from the initial carry.

```python
visual_data = {'kind': 'line', 'x': [1, 2, 3, 4], 'xlabel': 'completed step', 'ylabel': 'carry value', 'series': [{'label': 'scan history', 'y': history.tolist()}, {'label': 'fixed point 2', 'y': [2.0] * 4}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:22:58.095458+00:00. JAX 0.9.2.

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
inputs=jnp.array([1.,-1.,2.,0.])
state=0.
reference=[]
for inc in [1.,-1.,2.,0.]:
    state=0.5*state+inc
    reference.append(state)
last,observed=jax.lax.scan(step,jnp.array(0.),inputs)
assert jnp.allclose(observed,jnp.array([1.,-0.5,1.75,0.875]))
assert jnp.allclose(observed,jnp.array(reference))
assert jnp.allclose(last,state)
```

**Expected:** History $[1,-0.5,1.75,0.875]$; final $0.875$.

The loop is an independent control-flow reference and the fixed numbers verify time indexing.

## Verify state and sensitivity together

**Predict before running:** How do the state and derivative recurrences differ?

```python
def terminal(d):
    return jax.lax.scan(lambda c,u:(d*c+u,d*c+u),jnp.array(0.),increments)[0]
value,sensitivity=jax.value_and_grad(terminal)(0.5)
assert jnp.allclose(value,1.875)
assert jnp.allclose(sensitivity,2.75)
assert jnp.allclose(jax.jit(terminal)(0.5),value)
```

**Expected:** Final state $1.875$ and sensitivity $2.75$.

The scalar objective is the final state. Returning the whole history to ordinary grad would need a deliberate scalar reduction.

## Make it yours

Make the decay an explicit scalar argument. Differentiate the final value with respect to decay and compare at $0.5$ with the analytic derivative $2.75$.

<details><summary>Reference solution</summary>

```python
def simulate(decay):
    def transition(carry, increment):
        value = decay * carry + increment
        return value, value
    return jax.lax.scan(transition, jnp.array(0.), increments)[0]
assert jnp.allclose(jax.grad(simulate)(0.5), 2.75)
```

</details>

## Record a different output

**Practice**

Carry the state, but record its squared value. Confirm that final carry still means state.

<details><summary>Hint</summary>

The second returned value is only the recorded output.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
last,energy=jax.lax.scan(lambda c,u:(0.5*c+u,(0.5*c+u)**2),jnp.array(0.),increments)
assert jnp.allclose(last,1.875)
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

<details><summary>Reference solution and reasoning</summary>

```python
def growing(c,u):
    return jnp.concatenate([c,u[None]]),u
try:
    jax.lax.scan(growing,jnp.zeros((0,)),increments)
except TypeError:
    print("Expected carry-shape mismatch")
else:
    raise AssertionError("Expected scan type failure")
_,saved=jax.lax.scan(lambda c,u:(c,u),jnp.array(0.),increments)
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

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX scan contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [JAX control flow](https://docs.jax.dev/en/latest/control-flow.html)

