# The chain rule, Jacobians, and directions

Phase 04: Math & optimization · about 80 minutes · CPU

## What you will be able to do

- Compute a small Jacobian by hand and connect forward sensitivities to reverse-mode gradients.
- Build a Jacobian one entry at a time
- Push an input direction forward
- Pull output sensitivity backward
- Check both sides of the same calculation

## The problem

One model input can affect several outputs. How does a small input change move those outputs, and how does a loss send sensitivity back? We will work through one tiny vector function before asking JAX to do the same calculation.

## The idea

A Jacobian collects how each output responds to each input. We often need its action on a direction rather than the full matrix. Forward products move input perturbations to output perturbations; reverse products bring output sensitivities back to inputs.

## Follow one direction through a Jacobian

Take $J=[[4,1],[3,2]]$ and input direction $v=[1,-1]$. Multiplication gives $Jv=[3,1]$. This predicts the first-order output change when we move a small distance along that direction.

For output sensitivity $u=[2,-1]$, the reverse product is $J^Tu=[5,0]$. These vectors live in different spaces even though both happen to have two entries in this example. Use unequal input/output dimensions in a follow-up test so equal lengths cannot hide a transpose mistake.

There is a useful independent agreement: $u^T(Jv)=5$ and $(J^Tu)^Tv=5$. This checks the relationship between forward and reverse products without requiring you to inspect every Jacobian entry.

### Pause and reason

Why should we label the input and output spaces even for a square Jacobian?

<details><summary>Compare your reasoning</summary>

Equal dimensions can hide an incorrect orientation. The labels explain whether a vector is an input direction or output sensitivity and therefore which product is meaningful.

</details>

## Build a Jacobian one entry at a time

Let $f(x)=(x_0^2+x_1,\;x_0x_1)$, where $x=(x_0,x_1)$. The first output changes at rates $2x_0$ and $1$; the second changes at rates $x_1$ and $x_0$. Rows describe outputs, columns describe inputs. At $x=(2,3)$, the output is $(7,6)$ and the Jacobian is $J=\begin{pmatrix}4&1\\3&2\end{pmatrix}$.

$$
J_{ij}=\frac{\partial f_i}{\partial x_j},\qquad J\in\mathbb{R}^{m\times d}\ \text{for}\ f:\mathbb{R}^d\to\mathbb{R}^m
$$

## Push an input direction forward

Choose $v=(1,-1)$. A small move $\varepsilon v$ in the input changes the output by approximately $\varepsilon Jv$. Here $Jv=(3,1)$: the first output moves three times as fast as the second along this direction. `jax.jvp` returns both the ordinary output and this directional sensitivity. It can compute the product without storing the full Jacobian.

$$
f(x+\varepsilon v)\approx f(x)+\varepsilon Jv
$$

## Pull output sensitivity backward

Suppose the scalar loss is $L(x)=\tfrac12\|f(x)\|_2^2$. Its derivative with respect to the outputs is $c=f(x)=(7,6)$. The chain rule gives $\nabla_x L=J^\mathsf{T}c=(46,19)$. `jax.vjp` returns the output and a pullback function. Passing $c$ into that function returns input sensitivities. The pullback result is a tuple because a function can have several inputs.

$$
\nabla_x L=J^\mathsf{T}\nabla_f L
$$

## Check both sides of the same calculation

The identity $c^\mathsf{T}(Jv)=v^\mathsf{T}(J^\mathsf{T}c)$ checks that forward and reverse products agree. We also compare against the hand matrix and central differences so our evidence is not only two autodiff APIs agreeing. `jax.grad` needs a scalar output; use a Jacobian or a selected output direction for a vector function. These local derivatives do not describe a large move exactly.

## Write the vector function

Create main.py. Define the function, its input and the hand-derived Jacobian.

```python
# Step 1 — Write the vector function: The matrix is written from the derivatives, not obtained from an...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `f(x)` implementing this stage's computation:
def f(x):
    # Return `jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])` to the caller.
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
# Initialize array `x` with explicit values and shape.
x = jnp.array([2.0, 3.0])
# Initialize array `v` with explicit values and shape.
v = jnp.array([1.0, -1.0])
# Initialize array `J` with explicit values and shape.
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))
```

The matrix is written from the derivatives, not obtained from an autodiff call.

## Push a direction and pull a sensitivity

Append these two transformations; predict both products first.

```python
# Step 2 — Push a direction and pull a sensitivity: Forward mode follows an input direction; reverse mode starts with...
# Compute exact directional derivative / Jacobian / Hessian (`(out, jv)`).
out, jv = jax.jvp(f, (x,), (v,))
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_, pullback = jax.vjp(f, x)
# Evaluate `c` from the current inputs and state.
c = out
# Run `pullback` to compute `jt_c`.
jt_c = pullback(c)[0]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))
```

Forward mode follows an input direction; reverse mode starts with output sensitivity.

## Connect to a scalar loss

Append the scalar objective and independent matrix checks, then run python main.py.

```python
# Step 3 — Connect to a scalar loss: Both routes give directional loss derivative 27.
def loss(x):
    # Return `0.5 * jnp.sum(f(x) ** 2)` to the caller.
    return 0.5 * jnp.sum(f(x) ** 2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.jacfwd(f)(x), J)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
# Print the observed values to compare against the expected result.
print('Jv / loss gradient:', jv, jt_c)
```

Both routes give directional loss derivative $27$.

## Run the example

```python
# Step 1 — Write the vector function: The matrix is written from the derivatives, not obtained from an...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Function `f(x)` implementing this stage's computation:
def f(x):
    # Return `jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])` to the caller.
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1]])
# Initialize array `x` with explicit values and shape.
x = jnp.array([2.0, 3.0])
# Initialize array `v` with explicit values and shape.
v = jnp.array([1.0, -1.0])
# Initialize array `J` with explicit values and shape.
J = jnp.array([[4.0, 1.0], [3.0, 2.0]])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(f(x), jnp.array([7.0, 6.0]))

# Step 2 — Push a direction and pull a sensitivity: Forward mode follows an input direction; reverse mode starts with...
# Compute exact directional derivative / Jacobian / Hessian (`(out, jv)`).
out, jv = jax.jvp(f, (x,), (v,))
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_, pullback = jax.vjp(f, x)
# Evaluate `c` from the current inputs and state.
c = out
# Run `pullback` to compute `jt_c`.
jt_c = pullback(c)[0]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jv, jnp.array([3.0, 1.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jt_c, jnp.array([46.0, 19.0]))

# Step 3 — Connect to a scalar loss: Both routes give directional loss derivative 27.
def loss(x):
    # Return `0.5 * jnp.sum(f(x) ** 2)` to the caller.
    return 0.5 * jnp.sum(f(x) ** 2)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.jacfwd(f)(x), J)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(loss)(x), J.T @ c)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jnp.dot(c, jv), jnp.dot(v, jt_c))
# Print the observed values to compare against the expected result.
print('Jv / loss gradient:', jv, jt_c)
```

Expected: Forward product $[3,1]$, reverse product $[46,19]$, directional loss derivative $27$.

## A Jacobian maps inputs to output sensitivities

**Predict:** Which output is more sensitive to the first input at $(2,3)$?

![A Jacobian maps inputs to output sensitivities](../../phases/04-optimization/07-chain-rule-jacobians-and-directional-derivatives/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Rows identify output coordinates and columns identify input coordinates. Each cell is a partial derivative evaluated at the example input $(2,3)$. The matrix is $J=\begin{pmatrix}4&1\\3&2\end{pmatrix}$, and darker cells indicate larger positive sensitivity here.

Read down the first column: a small increase in input $0$, holding input $1$ fixed, changes the two outputs by approximately $4$ and $3$ times that increase. Read across the first row to see how both inputs influence output $0$.

### Connect it to the computation

For a joint input change in direction $v=(1,-1)$, multiply the whole matrix by that direction: $Jv=(3,1)$. The second input’s decrease offsets some of the first input’s increase. Looking only at the darkest cell would miss that interaction.

These are local sensitivities, not model weights, probabilities, or final output values. The entries can change at another input. To check the interpretation, take a small step $\epsilon v$ and compare the output change with $\epsilon Jv$.

```python
# Compute figure data for: A Jacobian maps inputs to output sensitivities
# Compute higher-order Jacobian or Hessian curvature matrix `visual_data`.
visual_data = {'kind': 'heatmap', 'values': jax.jacfwd(f)(x).tolist(), 'rows': ['output 0', 'output 1'], 'columns': ['input 0', 'input 1'], 'unit': 'partial derivative'}
```

## Recorded reference execution

CPU run: 2026-10-08T14:01:54.582744+00:00. JAX 0.9.2.

```text
Jv / loss gradient: [3. 1.] [46. 19.]
Jv / loss gradient: [3. 1.] [46. 19.]
PASS: optimization-07

```

## Check a finite input move

**Predict before running:** Does the central difference along $v$ agree with $Jv$ at step $0.01$?

```python
# Experiment — Check a finite input move: A numerical perturbation provides a check independent of autodiff.
h = 0.01
# Evaluate `fd` from the current inputs and state.
fd = (f(x + h * v) - f(x - h * v)) / (2 * h)
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(fd, J @ v, atol=0.0001, rtol=0.0001)
```

**Expected:** The finite-direction and hand-matrix results agree within float32 tolerance.

A numerical perturbation provides a check independent of autodiff.

## Change the point

**Predict before running:** At $x=(0,1)$, which Jacobian entries change?

```python
# Experiment — Change the point: The Jacobian describes sensitivity at the current point; it is...
# Initialize array `x2` with explicit values and shape.
x2 = jnp.array([0.0, 1.0])
# Initialize array `J2` with explicit values and shape.
J2 = jnp.array([[0.0, 1.0], [1.0, 0.0]])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.jacrev(f)(x2), J2)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.jvp(f, (x2,), (v,))[1], J2 @ v)
```

**Expected:** The new Jacobian swaps the direction coordinates.

The Jacobian describes sensitivity at the current point; it is not generally constant.

## Make it yours

Replace the scalar loss by $L(x)=f_0(x)+2f_1(x)$. Predict its gradient at $(2,3)$ before using `grad`.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Return `f(x)[0] + 2 * f(x)[1]` to the caller.
2. Verify that the numerical values match the expected reference within tolerance.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Replace the scalar loss by L(x)=f_0(x)+2f_1(x).
def weighted_loss(x):
    # Return `f(x)[0] + 2 * f(x)[1]` to the caller.
    return ...  # TODO: return computed result
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(weighted_loss)(x), jnp.array([10.0, 5.0]))  # TODO: complete assertion check
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Replace the scalar loss by L(x)=f_0(x)+2f_1(x).
def weighted_loss(x):
    # Return `f(x)[0] + 2 * f(x)[1]` to the caller.
    return f(x)[0] + 2 * f(x)[1]
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(jax.grad(weighted_loss)(x), jnp.array([10.0, 5.0]))
```

</details>

## Find the broken derivative path

**Transfer / diagnosis**

A colleague wraps the first input in `jax.lax.stop_gradient`. The forward outputs are unchanged. Show why the gradient no longer matches the intended mathematical function.

<details><summary>Hint</summary>

Compare the derivative of the sum of outputs with $J^\mathsf{T}(1,1)$.

</details>

### How to write: Find the broken derivative path — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Define `detached(x)` to evaluate the objective and its automatic derivatives:
2. Run `jax.lax.stop_gradient` to compute `a`.
3. Return `jnp.array([a * a + x[1], a * x[1]])` to the caller.
4. Verify that the numerical values match the expected reference within tolerance.
5. Verify that the output satisfies the expected shape, finite-value, or numerical contract.

**Starter code scaffold (fill in the TODOs):**

```python
# Find the broken derivative path (Transfer / diagnosis): Equal forward values do not imply equal differentiation...
# Define `detached(x)` to evaluate the objective and its automatic derivatives:
def detached(x):
    # Run `jax.lax.stop_gradient` to compute `a`.
    a = jax.lax.stop_gradient(...)  # TODO: compute a
    # Return `jnp.array([a * a + x[1], a * x[1]])` to the caller.
    return ...  # TODO: return computed result
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(detached(x), f(x))  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(detached(z)))(x), jnp.array([0.0, 3.0]))  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(f(z)))(x), jnp.array([7.0, 3.0]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Find the broken derivative path (Transfer / diagnosis): Equal forward values do not imply equal differentiation...
# Define `detached(x)` to evaluate the objective and its automatic derivatives:
def detached(x):
    # Run `jax.lax.stop_gradient` to compute `a`.
    a = jax.lax.stop_gradient(x[0])
    # Return `jnp.array([a * a + x[1], a * x[1]])` to the caller.
    return jnp.array([a * a + x[1], a * x[1]])
# Verify that the numerical values match the expected reference within tolerance.
assert jnp.allclose(detached(x), f(x))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(detached(z)))(x), jnp.array([0.0, 3.0]))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(jax.grad(lambda z: jnp.sum(f(z)))(x), jnp.array([7.0, 3.0]))
```

Equal forward values do not imply equal differentiation behavior when derivative rules are deliberately changed.

</details>

## Transfer to unequal dimensions

**Transfer / diagnosis**

For a matrix $A\in\mathbb{R}^{3\times2}$, check that the VJP of $Ax$ maps an output sensitivity of length $3$ into a vector of length $2$.

<details><summary>Hint</summary>

The reference is $A^\mathsf{T}c$, not $Ac$.

</details>

### How to write: Transfer to unequal dimensions — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.jvp / jax.vjp / jax.jacfwd / jax.hessian` — Computes exact forward-mode JVP, reverse-mode VJP, full Jacobians, or second-order curvature.

**Step-by-step implementation plan:**
1. Initialize array `A` with explicit values and shape.
2. Initialize array `c3` with explicit values and shape.
3. Compute exact directional derivative / Jacobian / Hessian (`(_, back)`).
4. Verify that the output tensor shape matches our prediction.
5. Verify that the output satisfies the expected shape, finite-value, or numerical contract.

**Starter code scaffold (fill in the TODOs):**

```python
# Transfer to unequal dimensions (Transfer / diagnosis): Unequal input and output dimensions make a mistaken...
# Initialize array `A` with explicit values and shape.
A = jnp.array(...)  # TODO: compute A
# Initialize array `c3` with explicit values and shape.
c3 = jnp.array(...)  # TODO: compute c3
# Compute exact directional derivative / Jacobian / Hessian (`(_, back)`).
_, back = jax.vjp(...)  # TODO: compute _, back
# Verify that the output tensor shape matches our prediction.
assert back(c3)[0].shape  # TODO: complete assertion check
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(back(c3)[0], jnp.array([3.0, 0.0]))  # TODO: complete assertion check
```

<details><summary>Reference solution and reasoning</summary>

```python
# Transfer to unequal dimensions (Transfer / diagnosis): Unequal input and output dimensions make a mistaken...
# Initialize array `A` with explicit values and shape.
A = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
# Initialize array `c3` with explicit values and shape.
c3 = jnp.array([1.0, -1.0, 2.0])
# Compute exact directional derivative / Jacobian / Hessian (`(_, back)`).
_, back = jax.vjp(lambda z: A @ z, x)
# Verify that the output tensor shape matches our prediction.
assert back(c3)[0].shape == (2,)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert jnp.allclose(back(c3)[0], jnp.array([3.0, 0.0]))
```

Unequal input and output dimensions make a mistaken transpose visible.

</details>

## Check your understanding

Which object does a reverse-mode pullback map to input sensitivity?

1. An output sensitivity vector
2. An arbitrary input direction
3. Only a scalar learning rate

<details><summary>Answer and explanation</summary>

An output sensitivity vector

A VJP multiplies an output sensitivity by the transposed Jacobian. A JVP instead propagates an input direction forward.

</details>

## Diagnose the result

If shapes fail, write output-by-input Jacobian dimensions on paper. If values match but derivatives do not, inspect stop_gradient, indexing, nondifferentiable operations and the scalar loss definition.

## Carry forward

- A numerical perturbation provides a check independent of autodiff.
- The Jacobian describes sensitivity at the current point; it is not generally constant.

## Keep your evidence

Keep the handwritten Jacobian, JVP and VJP checks, adjoint identity, finite-direction check and stop-gradient diagnosis.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX autodiff cookbook: JVPs and VJPs](https://docs.jax.dev/en/latest/301/cookbook.html)

