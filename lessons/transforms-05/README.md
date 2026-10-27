# Tracing, static arguments, and recompilation

Phase 02: JAX transformations · about 60 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

**Before this lesson:** [Compile a function with jit](https://www.tahabouhsine.com/jaxpathways/lesson-transforms-04.html).

## What you will be able to do

- Distinguish runtime array values from shape/dtype metadata and static configuration.
- Inspect a staged computation with make_jaxpr.
- Explain a Python value-dependent branch failure under jit.
- Use static configuration or JAX control flow according to the kind of decision.

## The problem

A function works until you put a Python if statement inside jit. Why does that change things? We’ll separate what Python knows while tracing from what the program learns at runtime. Then you’ll choose between a small static configuration and an array-based decision, rather than making every difficult argument static.

## The idea

Tracing observes how Python builds an array computation from abstract inputs. Static arguments are part of that program-building decision; dynamic array values are data for the resulting computation. Keep a small call ledger to separate these roles.

## Build a call ledger before diagnosing recompilation

A trace count is easier to interpret when each call changes only one thing. Start with two arrays of the same shape, dtype and static mode but different values; then change shape; then change mode. This separates new runtime data from requests to construct a different staged computation.

In the retained ledger, calls one and two both use shape $(3,)$, float32 and `sum`, while outputs change from $6$ to $9$. The next call uses shape $(4,)$; the last changes mode to `mean`. Read the table as this lesson’s observed signature changes, not as a universal count of JAX cache entries.

$$
\sigma=(\operatorname{shape}(x),\operatorname{dtype}(x),m),\qquad f_m(x)=\begin{cases}\sum_i x_i&m=\texttt{sum}\\\frac{1}{n}\sum_i x_i&m=\texttt{mean}\end{cases}
$$

### The specialization key separates changing data from program choices

**Predict:** If only the array entries change, which ledger fields remain fixed?

![The specialization key separates changing data from program choices](../../phases/02-transforms/05-tracing-static-arguments-and-recompilation/outputs/mechanism.svg)

*Signature-to-trace ledger for the lesson calls.*

The table nodes are four calls. The first two carry different arrays through the same shape, dtype and static-mode signature, so their sums change from $6$ to $9$ while the requested staged computation stays the same. The third changes shape and the fourth changes static mode. Read each arrow as the next call in this controlled sequence; the table is not a global cache counter.

### Pause and reason

A loop passes a new Python configuration value on every call as a static argument. What should you inspect?

<details><summary>Compare your reasoning</summary>

Inspect whether that value truly changes program structure. If it is ordinary changing data, represent it dynamically where possible. Keep the call ledger and compare traces after changing one factor at a time.

</details>

## Separate metadata, configuration, and data

A staged function needs a clear boundary between array values that flow at runtime and choices that determine which program is built. The values $[1,2,3]$ are data; shape $(3,)$ describes their array; `sum` or `mean` selects a reduction rule.

Here the finite mode string is treated as static configuration. With `sum`, $[1,2,3]$ returns $6$; with `mean`, it returns $2$. Marking a changing batch as static would make each value part of specialization and is not the right way to express ordinary runtime data. Choose the boundary from what the function computes, not merely to silence an error.

## Read a tiny staged program

`make_jaxpr` lets us inspect the staged array computation when we need to understand what tracing represented. For the square-and-sum function, the retained output shows multiplication followed by `reduce_sum`, producing a scalar. That connects the source operations to the traced program without needing to infer behavior from timing.

A Python `print` may run during tracing, but it is not itself an array operation in the jaxpr. The exact formatting and primitive names can vary, so focus on the dataflow and reduction. Use a tiny example and compare the traced operations with the source rather than treating a jaxpr dump as a stable user-facing format.

## Why a Python if can fail

Python `if` asks Python to choose a branch immediately. During tracing, a dynamic scalar argument is represented by a tracer whose runtime Boolean value is not yet available to Python. Thus `if x > 0` can fail even though the same function works eagerly for a concrete number.

The failure identifies a staging mismatch: the decision depends on runtime array data. Keep that distinction clear from static configuration, which is intentionally known while specializing. For runtime branching, use a JAX control-flow form such as `lax.cond`; for an elementwise choice over arrays, use selection with compatible shapes.

## Make configuration choices explicit

The `reduce_values` example uses a small finite mode to select either sum or mean. Its result depends on both inputs: the array supplies values, and the static mode chooses the reduction. Unknown mode strings should be rejected, because silently mapping a typo to a valid reduction can produce plausible but incorrect results.

In the observed calls, sum of $[1,2,3]$ is $6$, while mean is $2$. Keep the function identity stable and make the static choice explicit. Changing array values with the same shape and mode changes the answer without changing the intended computation; changing mode requests a different specialization choice.

## Inspect tracing without confusing it with runtime logging

A Python print inside a jitted function reports tracing activity, not every execution of the numerical program. If a compatible second call produces no print, that is consistent with reuse of the trace; it does not show that the function did not run.

Use trace observations to diagnose a small controlled call sequence, and use JAX runtime debugging facilities when the question concerns runtime values. The execution output here separately confirms the computed sum and the jaxpr. Avoid turning one print count into a claim about all compilations or device executions.

## Carry this boundary into the next phase

The next lessons make random keys, parameter trees, repeated state and branches explicit. The same classification helps each one: values needed to choose or update the next numerical result must be represented as runtime data or state; program choices known during specialization can define a static configuration.

When a traced program fails, first identify whether the problematic value is an array tracer or a true configuration choice. Then choose the operation that matches the question. This is more reliable than marking arbitrary inputs static, especially when their values change from call to call.

## Import JAX tracing and array operations

Import jax and jax.numpy for the reduction and JIT wrapper.

```python
import jax
import jax.numpy as jnp
```

The data values will remain ordinary JAX array inputs; the mode selects Python control flow.

## Make the reduction mode static during tracing

Define sum and mean branches, reject unknown modes, and mark mode as a static JIT argument.

```python
def reduce_values(x, mode):
    if mode == "sum":
        return x.sum()
    if mode == "mean":
        return x.mean()
    raise ValueError("mode must be sum or mean")
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(reduce_values, static_argnames=("mode",))
```

Python must know mode while tracing because it chooses a branch. Each distinct static value can lead to a separate compiled specialization.

## Run and check both compiled specializations

Apply sum and mean to [1,2,3] and compare them with their known results.

```python
x = jnp.array([1., 2., 3.])
assert jnp.allclose(compiled(x, mode="sum"), 6.)
assert jnp.allclose(compiled(x, mode="mean"), 2.)
```

The outputs 6 and 2 distinguish the two reduction rules while using the same data array.

## Run the example

```python
# Tracing, static arguments, and recompilation
import jax
import jax.numpy as jnp
def reduce_values(x, mode):
    if mode == "sum":
        return x.sum()
    if mode == "mean":
        return x.mean()
    raise ValueError("mode must be sum or mean")
# Wrap with `jax.jit` (`compiled`) so XLA traces and compiles the function.
compiled = jax.jit(reduce_values, static_argnames=("mode",))
x = jnp.array([1., 2., 3.])
print("Sum:", float(compiled(x, mode="sum")))
print("Mean:", float(compiled(x, mode="mean")))
assert jnp.allclose(compiled(x, mode="sum"), 6.)
assert jnp.allclose(compiled(x, mode="mean"), 2.)
```

Expected: Sum: $6.0$; Mean: $2.0$. These mode choices can use separate compiled specializations.

## Call signatures separate runtime values from specialization choices

**Predict:** Which call changes only data values, and which calls request a different staged program?

![Call signatures separate runtime values from specialization choices](../../phases/02-transforms/05-tracing-static-arguments-and-recompilation/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Read each row as a call to the compiled reduction. The value column shows runtime array data; the signature columns show shape, dtype and static mode. Calls one and two both use $((3,),\mathrm{float32},\mathrm{sum})$ but return $6$ and $9$ from different values. Call three changes the shape to $(4,)$; call four changes the mode to mean and returns $2$. The table’s exact signatures belong to this call sequence.

### Connect it to the computation

Changing values while shape, dtype and mode remain fixed changes the result without changing the requested computation. A shape or static-mode change requests another specialization. Use the ledger to predict which property changed before inspecting traces; do not interpret it as a universal cache or compilation counter.

```python
visual_data = {'kind': 'flow', 'layout': 'sequence', 'nodes': ['values change; shape/dtype/mode fixed', 'same signature reused in this call ledger', 'shape or static mode changes specialization']}
```

## Recorded reference execution

CPU run: 2026-10-27T17:03:01.003324+00:00. JAX 0.9.2.

```text
Sum: 6.0
Mean: 2.0
Tracing sum_squares; shape: (3,)
{ lambda ; a:f32[3]. let
    b:f32[3] = mul a a
    c:f32[] = reduce_sum[axes=(0,) out_sharding=None] b
  in (c,) }
Expected dynamic Boolean conversion error
PASS: transforms-05

```

## Look at the staged numerical work

**Predict before running:** Predict the operations needed to square and sum a vector. Which part is executed as Python during tracing?

```python
# Experiment — Look at the staged numerical work: The second call supplies new values with the same shape/dtype.
def sum_squares(values):
    print("Tracing sum_squares; shape:", values.shape)
    return jnp.sum(values * values)
print(jax.make_jaxpr(sum_squares)(x))
# Wrap with `jax.jit` (`staged_squares`) so XLA traces and compiles the function.
staged_squares = jax.jit(sum_squares)
assert jnp.allclose(staged_squares(x), 14.)
assert jnp.allclose(staged_squares(x + 1.), 29.)
```

**Expected:** The representation contains numerical multiplication and reduction. Trace prints can appear during staging, not as one guaranteed print per call.

The second call supplies new values with the same shape/dtype. The numerical result changes while the computation structure can be reused.

## Reproduce and repair a dynamic branch

**Predict before running:** For positive and negative scalar inputs, predict the branch result. Why can eager Python decide but tracing cannot?

```python
# Experiment — Reproduce and repair a dynamic branch
def python_branch(value):
    if value > 0.:
        return value ** 2
    return -value
assert python_branch(-2.) == 2.
try:
    jax.jit(python_branch)(jnp.array(-2.))
except jax.errors.TracerBoolConversionError:
    print("Expected dynamic Boolean conversion error")
else:
    raise AssertionError("Expected a traced Python branch to fail")
def runtime_branch(value):
    return jax.lax.cond(value > 0., lambda z: z ** 2, lambda z: -z, value)
# Wrap with `jax.jit` (`compiled_branch`) so XLA traces and compiles the function.
compiled_branch = jax.jit(runtime_branch)
assert jnp.allclose(compiled_branch(jnp.array(-2.)), 2.)
assert jnp.allclose(compiled_branch(jnp.array(3.)), 9.)
```

**Expected:** The Python branch fails under jit with a tracer Boolean conversion error. The staged conditional returns $2$ and $9$.

The test expects a particular category of error and confirms the repair for both branches. A broad catch-all would hide unrelated failures.

## Make it yours

Foundation · Extend reduce_values with a static max mode while preserving sum and mean. Reject unknown modes. Verify all three modes on $[1, 2, 3]$ and on $[-4, -1, -2]$.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- A reducer supporting sum, mean, and max.
- Two arrays containing positive and negative values.

**Outputs:**
- Jitted results for all supported modes and an error for an unknown mode.

**Invariants to check:**
- The mode is static configuration.
- Unknown modes are rejected.
- Both signs of input are checked.

**Hints:**
- Keep array values dynamic and specialize the mode.
- Reuse the function for changed array values.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def reduce_values(values, mode):
    # TODO: implement the declared reductions and reject unknown modes
    raise NotImplementedError

def verify_reductions():
    # TODO: compile with mode static and check both input sets
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
def choose_reduction(values, mode):
    if mode == "sum":
        return values.sum()
    if mode == "mean":
        return values.mean()
    if mode == "max":
        return values.max()
    raise ValueError("unknown mode")
# Wrap with `jax.jit` (`choose`) so XLA traces and compiles the function.
choose = jax.jit(choose_reduction, static_argnames=("mode",))
for values in (x, jnp.array([-4., -1., -2.])):
    for mode, reference in (("sum", jnp.sum), ("mean", jnp.mean), ("max", jnp.max)):
        assert jnp.allclose(choose(values, mode=mode), reference(values))
try:
    choose(x, mode="typo")
except ValueError:
    pass
else:
    raise AssertionError("Unknown mode should fail")
```

</details>

## Keep runtime values dynamic

**Practice**

Implement an elementwise absolute value with a numerical selection, jit it, and test mixed-sign vectors. Explain why making the whole vector static is inappropriate.

<details><summary>Hint</summary>

`jnp.where` selects elementwise; both candidate arrays are computed expressions. It is not a Python branch per element.

</details>

### How to write: Keep runtime values dynamic — Step-by-step recipe & starter scaffold

**Inputs:**
- Mixed-sign numeric vectors.
- An elementwise absolute-value function.

**Outputs:**
- Jitted results for each vector.

**Invariants to check:**
- Numerical values remain runtime data.
- Output shape matches its input.

**Hints:**
- Use elementwise numerical selection.
- Do not make the entire vector static.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def elementwise_absolute(values):
    # TODO: select the nonnegative magnitude elementwise
    raise NotImplementedError

def verify_absolute(compiled_absolute, vectors):
    # TODO: test the compiled function on each mixed-sign input
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Keep runtime values dynamic (Practice): The vector values are runtime data.
def elementwise_absolute(values):
    return jnp.where(values >= 0., values, -values)
# Wrap with `jax.jit` (`compiled_absolute`) so XLA traces and compiles the function.
compiled_absolute = jax.jit(elementwise_absolute)
for values in (jnp.array([-2., 0., 3.]), jnp.array([5., -4., -1.])):
    assert jnp.allclose(compiled_absolute(values), jnp.abs(values))
```

The vector values are runtime data. Elementwise selection preserves a fixed output shape and does not need a static array.

</details>

## Audit a shape-dependent specialization

**Challenge**

Run a jitted sum-of-squares function for lengths $3$ and $5$, each with two different value sets. Compare to a loop calculation. Explain which changes can affect specialization and which are ordinary runtime data.

<details><summary>Hint</summary>

Do not use elapsed time alone to infer whether tracing occurred. Values and shape are different inputs to your reasoning.

</details>

### How to write: Audit a shape-dependent specialization — Step-by-step recipe & starter scaffold

**Inputs:**
- A jitted sum-of-squares function.
- Inputs of lengths three and five with two value sets each.

**Outputs:**
- Compiled results and an independent Python-loop reference.
- A note distinguishing shape from data-value changes.

**Invariants to check:**
- Each element contributes once.
- Separate changed shape from changed values.
- Do not use elapsed time alone to claim a trace count.

**Hints:**
- Inspect shape metadata directly.
- Build the reference with a plain Python loop.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def sum_squares(values):
    # TODO: define the numerical reduction
    raise NotImplementedError

def audit_shapes(compiled, value_sets):
    # TODO: compare compiled results with an independent loop reference
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Audit a shape-dependent specialization (Challenge)
def plain_sum_squares(values):
    return jnp.sum(values ** 2)
# Wrap with `jax.jit` (`compiled_squares`) so XLA traces and compiles the function.
compiled_squares = jax.jit(plain_sum_squares)
for length in (3, 5):
    for shift in (0., 1.):
        values = jnp.arange(length, dtype=jnp.float32) + shift
        expected = sum(float(v) ** 2 for v in values)
        assert jnp.allclose(compiled_squares(values), expected)
```

Changing the length changes shape metadata and can need a new specialization. Changing shift changes element values at the same shape/dtype. The test verifies correctness across both kinds of changes without inventing a guaranteed compilation count.

</details>

## Check your understanding

A frequently changing runtime scalar controls a branch. What is the most appropriate first repair for a traced Python if?

1. Mark every runtime scalar static.
2. Express the runtime decision with compatible JAX control flow.
3. Catch every exception and return zero.

<details><summary>Answer and explanation</summary>

**1. Mark every runtime scalar static.**

Making a value static specializes on it and can trigger a new compilation when it changes. Use runtime control flow when the predicate is data.

**2. Express the runtime decision with compatible JAX control flow. (correct)**

A runtime decision belongs in the staged computation. Static configuration is useful for a small program choice, but making changing data static can create unnecessary specializations.

**3. Catch every exception and return zero.**

Returning zero hides the original contract failure and changes the function. Express both branches with compatible JAX values instead.

</details>

## Diagnose the result

For a tracer Boolean conversion error, inspect the Python truth test. For an unhashable static argument, reconsider whether the object is really configuration. For repeated setup, inspect function recreation, shapes/dtypes, and static values. For missing Python logs, distinguish tracing from runtime execution. Always verify both branches and preserve unknown-mode errors rather than returning a plausible fallback.

## Carry forward

- Python executes while staging; numerical array operations form the runtime computation.
- Static configuration and dynamic numerical data have different roles.
- Use runtime control flow for runtime decisions and compatible branch returns.
- Inspect evidence of tracing separately from numerical correctness and timing.

## Keep your evidence

Keep the small jaxpr observation, the expected tracer Boolean error, checks of both repaired branches, all static reduction modes including invalid-mode rejection, and shape/value-change cases. Do not infer compilation counts from timing alone.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX: tracing and compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX: control flow](https://docs.jax.dev/en/latest/control-flow.html)
- [JAX: tracer Boolean conversion errors](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError)

