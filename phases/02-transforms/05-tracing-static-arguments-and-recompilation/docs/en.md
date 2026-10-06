# Tracing, static arguments, and recompilation

Phase 02: JAX transformations · about 60 minutes · CPU

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

Record each call's shape, dtype and static choices next to the observed trace count. Change only values first. Then change one shape, then one static option. This controlled sequence makes a trace increase interpretable; changing everything at once does not.

A Python branch needs a concrete decision during tracing. If the decision is intended to vary with array data at execution time, use an appropriate JAX control-flow operation instead of converting a tracer to a Python boolean. If it genuinely selects a small number of program variants, an explicit static argument may be suitable.

Do not turn every input into a static value to silence an error. Many distinct static values can create many specializations. Trace counts are useful observations, but they are not a universal count of all compilation work or a cross-process cache guarantee.

### Pause and reason

A loop passes a new Python configuration value on every call as a static argument. What should you inspect?

<details><summary>Compare your reasoning</summary>

Inspect whether that value truly changes program structure. If it is ordinary changing data, represent it dynamically where possible. Keep the call ledger and compare traces after changing one factor at a time.

</details>

## Separate metadata, configuration, and data

A tensor shape such as $(3,)$ describes the size of the computation; its values $[1, 2, 3]$ are runtime data. A string mode such as sum describes which computation to construct. In this lesson mode is a small finite configuration, so marking it static is reasonable.

If mode changes rarely while batches change constantly, static configuration lets a stable compiled specialization accept many batches. If you make frequently changing data static, you encourage distinct specializations and may run into hashability constraints. A JAX array is not a convenient static argument just because a Python branch wants to inspect it.

## Read a tiny staged program

make_jaxpr exposes a representation of the traced array operations. A Python print inside the function may run while tracing, but it does not become an ordinary numerical operation in that representation. A multiply and a reduce_sum should appear in a multiply-then-sum example.

The exact printed formatting and primitive names are not a stable teaching contract. Read it for inputs, outputs, and the broad sequence of numerical operations. The point is to observe that the computation is staged, not to memorize an internal dump or treat the representation as a complete performance explanation.

## Why a Python if can fail

Consider if $x$ > $0$ where $x$ is a dynamic scalar argument. Python wants one Boolean immediately to choose which branch to execute while tracing. The tracer represents a runtime value, so that truth test cannot generally be answered at trace time. The resulting conversion error is evidence of a staging mismatch, not evidence that comparisons cannot run on an accelerator.

For a scalar runtime decision, lax.cond expresses the conditional in the numerical program. Its two branches must return compatible structures and shapes/dtypes. Both branch functions are traced; for an ordinary scalar condition the selected branch is the runtime computation. Under batching, a conditional can be transformed into selection behavior, so do not infer cost or side-effect behavior from the Python-like notation alone.

## Make configuration choices explicit

The worked reduce_values function chooses sum or mean using a static mode string. Each mode can have its own specialization. It should reject unknown modes instead of silently treating any typo as mean. This is a small input-contract improvement that prevents plausible-looking wrong results.

Keep function identity stable, configure static_argnames at the transformation boundary, and pass mode deliberately. Shape and dtype changes can also require specializations. Different element values with compatible abstract input properties ordinarily do not, though caches and transformation details make exact global compilation counts a poor public guarantee.

## Inspect tracing without confusing it with runtime logging

A Python print in a jitted function is a trace-time observation. If you call the function again with compatible inputs, its absence is consistent with reuse. It is not a reliable counter of all device executions or all compilations across the application. JAX provides debug facilities for observing runtime values.

Use trace observations for a small diagnostic experiment and rely on supported compiler logging/profiling when investigating a real training job. Do not make application behavior depend on a side effect inside a transformed numerical function.

## Carry this boundary into the next phase

The upcoming state lessons will make randomness, parameter structure, loops, and branches explicit. The same principle applies: information needed at runtime belongs in the numerical state or arguments, while small program configuration can belong in specialization.

Before reaching for a workaround, classify the value and the question: is Python constructing the computation, or is the computation choosing a result based on runtime data? That classification leads to an implementation you can explain and test.

## Run the example

```python
import jax
import jax.numpy as jnp
def reduce_values(x, mode):
    if mode == "sum":
        return x.sum()
    if mode == "mean":
        return x.mean()
    raise ValueError("mode must be sum or mean")
compiled = jax.jit(reduce_values, static_argnames=("mode",))
x = jnp.array([1., 2., 3.])
print("Sum:", float(compiled(x, mode="sum")))
print("Mean:", float(compiled(x, mode="mean")))
assert jnp.allclose(compiled(x, mode="sum"), 6.)
assert jnp.allclose(compiled(x, mode="mean"), 2.)
```

Expected: Sum: $6.0$; Mean: $2.0$. These mode choices can use separate compiled specializations.

## Static choices select a traced program

**Predict:** Does changing array data require the same action as changing a Python branch choice?

![Static choices select a traced program](../outputs/figure.svg)

**Conceptual diagram**

### Read the figure

Follow the boxes downward. The static mode first selects either summation or averaging. Python follows that branch while JAX traces it, and JAX compiles the operations in the selected branch. Array values then flow through the resulting computation.

For the example array $(1,2,3)$, the sum branch returns $6$ and the mean branch returns $2$. The diagram shows how a program is selected, not a competition between two numerical outputs.

### Connect it to the computation

A compatible later call can reuse the selected program while supplying different array values. Changing the static mode selects a different specialization; changing shapes or dtypes can also require a new specialization. The tracing and compilation boxes therefore do not run on every compatible call.

The arrows describe dependencies, and the box sizes do not encode runtime. Use the trace-count experiment to observe reuse. Merely seeing a different answer after changing the array is not evidence that retracing occurred.

## Recorded reference execution

CPU run: 2026-10-06T21:56:22.046346+00:00. JAX 0.9.2.

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
def sum_squares(values):
    print("Tracing sum_squares; shape:", values.shape)
    return jnp.sum(values * values)
print(jax.make_jaxpr(sum_squares)(x))
staged_squares = jax.jit(sum_squares)
assert jnp.allclose(staged_squares(x), 14.)
assert jnp.allclose(staged_squares(x + 1.), 29.)
```

**Expected:** The representation contains numerical multiplication and reduction. Trace prints can appear during staging, not as one guaranteed print per call.

The second call supplies new values with the same shape/dtype. The numerical result changes while the computation structure can be reused.

## Reproduce and repair a dynamic branch

**Predict before running:** For positive and negative scalar inputs, predict the branch result. Why can eager Python decide but tracing cannot?

```python
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
compiled_branch = jax.jit(runtime_branch)
assert jnp.allclose(compiled_branch(jnp.array(-2.)), 2.)
assert jnp.allclose(compiled_branch(jnp.array(3.)), 9.)
```

**Expected:** The Python branch fails under jit with a tracer Boolean conversion error. The staged conditional returns $2$ and $9$.

The test expects a particular category of error and confirms the repair for both branches. A broad catch-all would hide unrelated failures.

## Make it yours

Foundation · Extend reduce_values with a static max mode while preserving sum and mean. Reject unknown modes. Verify all three modes on $[1, 2, 3]$ and on $[-4, -1, -2]$.

<details><summary>Reference solution</summary>

```python
def choose_reduction(values, mode):
    if mode == "sum":
        return values.sum()
    if mode == "mean":
        return values.mean()
    if mode == "max":
        return values.max()
    raise ValueError("unknown mode")
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

<details><summary>Reference solution and reasoning</summary>

```python
def elementwise_absolute(values):
    return jnp.where(values >= 0., values, -values)
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

<details><summary>Reference solution and reasoning</summary>

```python
def plain_sum_squares(values):
    return jnp.sum(values ** 2)
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

Express the runtime decision with compatible JAX control flow.

A runtime decision belongs in the staged computation. Static configuration is useful for a small program choice, but making changing data static can create unnecessary specializations.

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

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX: tracing and compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX: control flow](https://docs.jax.dev/en/latest/control-flow.html)
- [JAX: tracer Boolean conversion errors](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError)

