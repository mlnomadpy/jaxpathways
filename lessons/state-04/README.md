# Branches with lax.cond

Phase 03: State, randomness & control flow · about 60 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

**Before this lesson:** [Compiled loops with lax.scan](https://www.tahabouhsine.com/jaxpathways/lesson-state-03.html).

## What you will be able to do

- Locate the decision boundary
- Trace both branches and check their result types
- Choose cond or where by the shape of the question
- Inspect boundary behavior and derivatives

## The problem

A Python branch can choose a path when it knows a value. Inside jit, that value may not be available yet. We’ll use lax.cond to make the choice at runtime, check what both branches must return, and compare a single decision with elementwise selection.

## The idea

A data-dependent branch needs a control-flow operation that can represent both alternatives in a transformed program. With a scalar predicate, `lax.cond` selects the result of one branch while requiring compatible result structures from both.

## Trace both possibilities, execute the selected rule

A data-dependent choice must be represented inside a staged function because Python cannot choose from a value that will only exist at runtime. `lax.cond` expresses one scalar decision with two possible branch functions, each following the same input and output contract.

The plot uses input on the horizontal axis and returned magnitude on the vertical axis. It passes through $(-3,3)$, $(0,0)$, and $(3,3)$: the negative side is negated and the nonnegative side is returned unchanged. The continuous join has a sharp corner, so matching outputs at zero do not imply a unique mathematical derivative there.

$$
\operatorname{magnitude}(x)=\begin{cases}x&x\ge0\\-x&x<0\end{cases}
$$

### Pause and reason

Can one branch return a vector and the other a scalar if your current input only selects the vector?

<details><summary>Compare your reasoning</summary>

Not under the same conditional result contract. Both branches must produce compatible structures, shapes and dtypes so the transformed program has a well-defined result.

</details>

## Locate the decision boundary

For a scalar predicate, `cond` selects one branch result, but both branch functions are traced to build the staged computation. This is why a Python print or side effect inside a branch is not evidence about which numerical branch ran. The plotted magnitude values show only the returned function.

The execution receipt checks inputs $-3$ and $3$, both returning $3$, and confirms the expected failure for a traced Python Boolean. Keep tracing behavior separate from runtime selection, and keep both branch computations pure so the program remains meaningful under transformations.

```text
predicate: x>=0
true:  operand x → x
false: operand x → −x
joined result: scalar float
```

## Trace both branches and check their result types

The conditional result must have one stable contract regardless of which predicate value is supplied. Both branches need compatible pytree structure, corresponding leaf shapes and dtypes. If one returns a scalar and the other a vector, the result cannot be represented as one conditional output.

This requirement is independent of whether a test input selects a branch that seems valid. Both branches participate in tracing, so an invalid operation or incompatible output may fail before that specific runtime case is executed. Check branch contracts first, then test the selected numerical results.

## Choose cond or where by the shape of the question

Choose control flow to match the shape of the decision. `cond` expresses one scalar predicate selecting one result for the computation; `where` selects elementwise from compatible arrays. A vector of independent predicates is an elementwise question or a deliberately batched conditional, not one scalar decision.

Transformations can alter how conditionals are implemented; for example, batching a conditional may turn it into selection. Therefore branch-local side effects or assumptions that an unselected expression is never evaluated are fragile. Keep branch functions pure and expressions valid for their transformed use.

## Inspect boundary behavior and derivatives

The magnitude example equals $|x|$. On negative inputs its slope is $-1$; on positive inputs its slope is $1$. At zero the two one-sided slopes differ, so the classical derivative is undefined even though both branches return zero there.

Test a negative value, a positive value and the exact boundary. Changing $x\ge0$ to $x>0$ does not change this magnitude’s value at zero, but it can matter for other branch formulas. Keep numerical branch correctness and differentiability as separate checks.

## Prepare the inputs

Create main.py in your lesson workspace. Add this first block; use the environment from setup.

```python
# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp
```

These explicit inputs define the case that the later checks will verify.

## Build the computation

Append this block below the inputs in the same file.

```python
# Step 2 — Build the computation: Both branches return the same scalar type.
# Define and JIT-compile `magnitude(x)` so XLA traces and fuses the operations:
@jax.jit
def magnitude(x):
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)
```

Both branches return the same scalar type. cond takes the runtime predicate and passes $x$ to the selected numerical branch.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
# Step 3 — Run and check the result
print(float(magnitude(-3.)))
assert jnp.allclose(magnitude(-3.), 3.)
assert jnp.allclose(magnitude(2.), 2.)
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
# Step 1 — Prepare the inputs
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: Both branches return the same scalar type.
# Define and JIT-compile `magnitude(x)` so XLA traces and fuses the operations:
@jax.jit
def magnitude(x):
    return jax.lax.cond(x >= 0., lambda v: v, lambda v: -v, x)
# Step 3 — Run and check the result
print(float(magnitude(-3.)))
assert jnp.allclose(magnitude(-3.), 3.)
assert jnp.allclose(magnitude(2.), 2.)
```

Expected: $3.0$

## Both branches form one magnitude function

**Predict:** What happens on each side of zero?

![Both branches form one magnitude function](../../phases/03-state/04-branches-with-lax-cond/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is input $x$; the vertical axis is the returned magnitude. The V-shaped line gives $3$ at both $-3$ and $3$, and $0$ at the join. On the left arm the rule negates negative inputs; on the right it returns nonnegative inputs. Compare equal distances from zero to see the symmetry and continuity.

### Connect it to the computation

The two branch formulas combine to implement $|x|$. Their slopes are $-1$ to the left and $1$ to the right, so the corner at zero marks a change in derivative even though the output has no jump. Read the plot for output behavior; the tracing and branch-selection behavior comes from the control-flow contract, not from this curve.

```python
grid = jnp.linspace(-3.0, 3.0, 25)
# Vectorize across the batch dimension without a Python loop (`visual_data`).
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input', 'ylabel': 'magnitude', 'series': [{'label': 'lax.cond output', 'y': jax.vmap(magnitude)(grid).tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-27T17:03:13.077776+00:00. JAX 0.9.2.

```text
3.0
3.0
Expected traced Python Boolean failure
Expected branch shape mismatch
PASS: state-04

```

## Reproduce the Python truth-test failure

**Predict before running:** Why does eager evaluation choose a branch while jit cannot use a runtime value in Python if?

```python
# Experiment — Reproduce the Python truth-test failure: The repair changes where the decision is represented.
def eager_branch(v):
    if v>=0.:
        return v
    return -v
assert eager_branch(-3.)==3.
try:
    jax.jit(eager_branch)(jnp.array(-3.))
except jax.errors.TracerBoolConversionError:
    print("Expected traced Python Boolean failure")
else:
    raise AssertionError("Expected branch error")
for v in [-3.,0.,2.]:
    assert jnp.allclose(magnitude(v),abs(v))
```

**Expected:** The traced Python branch fails; cond gives the correct values for all three cases.

The repair changes where the decision is represented. It does not make $x$ static or remove the negative case.

## Check a batched numerical selection

**Predict before running:** Predict the outputs and derivatives away from the nonsmooth point.

```python
# Experiment — Check a batched numerical selection
values=jnp.array([-3.,-1.,2.,4.])
assert jnp.allclose(jax.vmap(magnitude)(values),jnp.abs(values))
assert jnp.allclose(jax.vmap(jax.grad(magnitude))(values),jnp.array([-1.,-1.,1.,1.]))
```

**Expected:** Magnitudes $[3,1,2,4]$ and slopes $[-1,-1,1,1]$.

Correct values under batching do not imply a conditional branch remains lazy after every transformation.

## Make it yours

Write a compiled function that applies $2x$ when $x$ is positive and $x-2$ otherwise. Check positive, negative, and zero inputs.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- A scalar or array of values evaluated as scalar calls.

**Outputs:**
- A JIT-compatible piecewise function: twice x when positive, otherwise x minus two.

**Invariants to check:**
- Both lax.cond branches return identical shape and dtype.
- The zero boundary follows the non-positive branch.

**Hints:**
- Use x > 0 as the predicate.
- Pass x as the operand and use one scalar result from each branch.

**Key functions & syntax to use:**
- `jax.jit(fn) / @jax.jit` — Traces a function for its input signatures and compiles it; changed shapes, dtypes, or static arguments can trigger another compilation.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

@jax.jit
def piecewise_rule(x: jax.Array) -> jax.Array:
    """Return 2*x for positive x and x-2 otherwise."""
    # TODO: express the runtime branch with jax.lax.cond.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
# Define and JIT-compile `choose_update(x)` so XLA traces and fuses the operations:
@jax.jit
def choose_update(x):
    return jax.lax.cond(x > 0., lambda v: v * 2., lambda v: v - 2., x)
assert jnp.allclose(choose_update(3.), 6.)
assert jnp.allclose(choose_update(-3.), -5.)
assert jnp.allclose(choose_update(0.), -2.)
```

</details>

## Match a changed piecewise rule

**Practice**

Implement $x^2$ for $x>1$ and $2x$ otherwise. Derive outputs at $-1$,$1$,$2$ and derivatives away from $1$.

<details><summary>Hint</summary>

At $x=1$ the false branch is selected.

</details>

### How to write: Match a changed piecewise rule — Step-by-step recipe & starter scaffold

**Inputs:**
- A scalar input v.

**Outputs:**
- The value of v squared for v>1, and 2v otherwise.

**Invariants to check:**
- At v=1 the false branch is chosen.
- Both branches return a scalar of compatible dtype.

**Hints:**
- Set the predicate to v > 1.
- Check derivatives only away from the branch boundary.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def changed_rule(v: jax.Array) -> jax.Array:
    """Implement the two-branch scalar rule."""
    # TODO: use lax.cond with the exact boundary specified in the task.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Match a changed piecewise rule (Practice)
def piecewise(v):
    return jax.lax.cond(v>1.,lambda z:z*z,lambda z:2*z,v)
assert jnp.allclose(jax.vmap(piecewise)(jnp.array([-1.,1.,2.])),jnp.array([-2.,2.,4.]))
assert jnp.allclose(jax.grad(piecewise)(-1.),2.)
assert jnp.allclose(jax.grad(piecewise)(2.),4.)
```

The values check the exact boundary and both formulas; the derivatives are checked only at smooth points.

</details>

## Repair incompatible branch shapes

**Challenge**

Return a scalar in one branch and a length-two vector in the other, reproduce the error, then define a consistent two-value result.

<details><summary>Hint</summary>

Decide what each output component means before padding.

</details>

### How to write: Repair incompatible branch shapes — Step-by-step recipe & starter scaffold

**Inputs:**
- A scalar input v used by two branch functions.

**Outputs:**
- A pair of scalar values with the same shape and dtype regardless of the predicate.

**Invariants to check:**
- lax.cond requires compatible results from both branches.
- The pair gives the same semantic meaning on both paths.

**Hints:**
- Choose what the two returned components represent first.
- Construct a length-two array in each branch.

**Starter code scaffold (fill in the TODOs):**

```python
import jax
import jax.numpy as jnp

def consistent_pair(v: jax.Array) -> jax.Array:
    """Return a consistent length-two result from both runtime branches."""
    # TODO: make both cond branches return matching shape and dtype.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Repair incompatible branch shapes (Challenge)
try:
    jax.lax.cond(True,lambda x:x,lambda x:jnp.stack([x,x]),jnp.array(2.))
except TypeError:
    print("Expected branch shape mismatch")
else:
    raise AssertionError("Expected result type error")
def pair(v):
    return jax.lax.cond(v>=0.,lambda x:jnp.stack([x,x*x]),lambda x:jnp.stack([-x,x*x]),v)
assert jnp.allclose(pair(jnp.array(-2.)),jnp.array([2.,4.]))
```

A branch mismatch is a result-contract error even when the current predicate is known. Here the repaired output explicitly means [magnitude,square].

</details>

## Check your understanding

Can one cond branch return a scalar and the other a length-three array?

1. Yes, any shapes are allowed
2. No, branch result types and shapes must be compatible
3. Only if the predicate is negative

<details><summary>Answer and explanation</summary>

**1. Yes, any shapes are allowed**

Both branches are traced into one conditional contract. A scalar and a length-three vector cannot be alternative results of that contract.

**2. No, branch result types and shapes must be compatible (correct)**

Compiled control flow requires compatible output structures and array types across branches.

**3. Only if the predicate is negative**

The sign selects the runtime branch but does not remove the need to trace and type-check the other branch.

</details>

## Diagnose the result

A branch mismatch is a result-contract error even when the current predicate is known. Here the repaired output explicitly means [magnitude,square].

## Carry forward

- The repair changes where the decision is represented. It does not make $x$ static or remove the negative case.
- Correct values under batching do not imply a conditional branch remains lazy after every transformation.

## Keep your evidence

Keep the branch-contract diagram, eager/jit failure and repair, boundary cases, batched values and off-boundary derivatives, and the shape-mismatch diagnosis.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
- [JAX control flow](https://docs.jax.dev/en/latest/control-flow.html)

