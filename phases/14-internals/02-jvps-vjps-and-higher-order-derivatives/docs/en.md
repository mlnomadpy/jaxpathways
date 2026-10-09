# JVPs, VJPs, and higher-order derivatives

Phase 14: Autodiff & JAX internals · about 105 minutes · CPU

## What you will be able to do

- Compute and interpret JVP and VJP products with correct domains and shapes.
- Verify the adjoint dot-product identity against a hand-derived Jacobian.
- Compute an exact Hessian-vector product and distinguish it from a Gauss–Newton approximation.

## The problem

A large model may have millions of inputs or outputs, but you often need only one sensitivity direction. How can we compute the useful product without constructing the full Jacobian—and verify that forward, reverse and higher-order calculations agree?

## The idea

A JVP asks how an input direction changes outputs. A VJP asks how an output sensitivity pulls back to inputs. Both use the same local derivative, but their vectors belong to different spaces.

## Forward and reverse products answer complementary questions

For a map from three inputs to two outputs, the Jacobian has shape $(2,3)$. A JVP consumes a three-component direction and returns two components. A VJP consumes a two-component cotangent and returns three.

Check the adjoint relation $u^T(Jv)=(J^Tu)^Tv$ using independent dot products. It connects both directions without requiring the full Jacobian to be materialized.

When reading product bars, name the vector's space before interpreting its coordinates. A square example can hide a transpose mistake because both vectors have the same length. Add an unequal-dimension case to expose that ambiguity.

### JVP and VJP use different vector spaces

**Predict:** Does a VJP simply return the same vector as a JVP?

![JVP and VJP use different vector spaces](../outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

The top row pushes a three-component direction into two output components. The bottom pulls a two-component sensitivity back to three inputs. Shapes disambiguate the two products even when a square worked example would not. The adjoint identity checks their relationship.

### Pause and reason

Does a VJP simply return the same vector as a JVP?

<details><summary>Compare your reasoning</summary>

No. They apply the derivative in different directions and can have different result dimensions. Their relationship is the adjoint identity, not equality of vectors.

</details>

## Separate the point from the direction

For $f(a,b)=(ab,\sin(a)+b^2)$, the point $x=(0.4,-0.7)$ determines a Jacobian. The direction $v=(1.2,-0.3)$ says how the inputs will move near that point. JVP returns both the original output and $J(x)v$. It does not evaluate the nonlinear function at $x+v$; its approximation becomes meaningful when multiplied by a sufficiently small displacement.

$$
J(a,b)=\begin{bmatrix}b&a\\\cos(a)&2b\end{bmatrix},\qquad f(x+\varepsilon v)\approx f(x)+\varepsilon J(x)v
$$

## A VJP answers a different directional question

The cotangent $u=(0.5,2)$ weights changes in the two outputs. Pulling it back gives $J(x)^\mathsf{T}u$, a vector in input space. It is the gradient of the scalar function $u^\mathsf{T}f(x)$ when $u$ is fixed. The pullback closes over the forward computation at this specific point; it must be rebuilt when the point changes.

## Check the adjoint identity with independent values

The scalar $u^\mathsf{T}(Jv)$ can be computed either by pushing the direction first or pulling the measurement first. The two orders must agree: $u^\mathsf{T}Jv=(J^\mathsf{T}u)^\mathsf{T}v$. Here JVP is approximately $(-0.96,1.52527)$, VJP is $(1.49212,-2.6)$, and both scalar products are about $2.57055$. This identity alone is insufficient if both rules share a mistake; the explicit Jacobian adds an independent oracle.

## Count directions before choosing a Jacobian strategy

Forward mode naturally produces a Jacobian column for each input basis direction; reverse mode produces a row for each output basis direction. Many inputs and one scalar output often favor reverse mode, whereas a small number of input directions may favor forward products. Real runtime also depends on intermediates, batching and device behavior. This lesson verifies mathematical products, not a universal timing rule.

## Curvature includes the changing Jacobian

For $L(x)=\frac12\lVert f(x)\rVert^2$, the gradient is $J^\mathsf{T}f$. Differentiating again gives $H=J^\mathsf{T}J+\sum_i f_i\nabla^2f_i$. The first term is positive semidefinite and is often used as a Gauss–Newton approximation; it is exact only when the residual-weighted term vanishes. Forward-over-reverse differentiation computes $Hv$ without building every Hessian entry.

$$
Hv=J^\mathsf{T}(Jv)+\sum_i f_i(\nabla^2 f_i)v
$$

## Use finite differences as a second perspective

A central difference of gradients approximates the Hessian-vector product, while a central difference of function outputs approximates the JVP. Sweep perturbations and retain a range of agreement; making the perturbation arbitrarily small causes cancellation. Float64 is explicitly enabled for these CPU comparisons. Non-smooth points and custom derivative rules require their own mathematical interpretation.

## Define a small vector map and derive its Jacobian

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
# Step 1 — Define a small vector map and derive its Jacobian: The map has two inputs and two outputs.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `mapping(x)` implementing this stage's computation:
def mapping(x):
    # Evaluate `(a, b)` from the current inputs and state.
    a,b=x
    # Return `jnp.array([a * b, jnp.sin(a) + b * b])` to the caller.
    return jnp.array([a*b,jnp.sin(a)+b*b])
# Initialize array `x` with explicit values and shape.
x=jnp.array([0.4,-0.7])
# Initialize array `v` with explicit values and shape.
v=jnp.array([1.2,-0.3])
# Initialize array `u` with explicit values and shape.
u=jnp.array([0.5,2.0])
# Convert `(a, b)` to a host NumPy array for inspection or verification.
a,b=np.asarray(x)
# Initialize array `jacobian` with explicit values and shape.
jacobian=np.array([[b,a],[np.cos(a),2*b]])
# Initialize array `expected_value` with explicit values and shape.
expected_value=np.array([a*b,np.sin(a)+b*b])
```

The map has two inputs and two outputs. Its Jacobian rows correspond to output coordinates, and columns to input coordinates.

## Push a direction forward and pull a measurement backward

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
# Step 2 — Push a direction forward and pull a measurement backward: The VJP returns one cotangent per input argument, hence the...
# Compute exact directional derivative / Jacobian / Hessian (`(value, jvp)`).
value,jvp=jax.jvp(mapping,(x,),(v,))
# Compute exact directional derivative / Jacobian / Hessian (`(value_again, pullback)`).
value_again,pullback=jax.vjp(mapping,x)
# Run `pullback` to compute `(vjp,)`.
vjp,=pullback(u)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
# Evaluate `jnp.vdot(u, jvp)` and convert the result into Python scalar/collection `left`.
# Evaluate `jnp.vdot(vjp, v)` and convert the result into Python scalar/collection `right`.
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(left,right,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Jacobian:",jacobian)
# Print diagnostic summary of the computed outputs.
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)
```

The VJP returns one cotangent per input argument, hence the one-element tuple. The dot-product identity checks that forward and reverse products represent adjoint linear maps.

## Compute curvature without materializing a Hessian

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
# Step 3 — Compute curvature without materializing a Hessian: For squared residuals, the Hessian includes a...
def objective(z):
    # Return `0.5 * jnp.vdot(mapping(z), mapping(z))` to the caller.
    return 0.5*jnp.vdot(mapping(z),mapping(z))
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient=jax.grad(objective)(x)
# Differentiate the objective to obtain `hvp` via automatic differentiation.
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
# Evaluate `y` from the current inputs and state.
y=expected_value
# Initialize array `hessian_0` with explicit values and shape.
hessian_0=np.array([[0.,1.],[1.,0.]])
# Initialize array `hessian_1` with explicit values and shape.
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
# Perform matrix contraction / projection to compute `hessian`.
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Hessian-vector product:",np.asarray(hvp))
```

For squared residuals, the Hessian includes a Jacobian-transpose-Jacobian term and residual-weighted second derivatives. Dropping the latter is an approximation, not the exact Hessian.

## Run the example

```python
# Step 1 — Define a small vector map and derive its Jacobian: The map has two inputs and two outputs.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `mapping(x)` implementing this stage's computation:
def mapping(x):
    # Evaluate `(a, b)` from the current inputs and state.
    a,b=x
    # Return `jnp.array([a * b, jnp.sin(a) + b * b])` to the caller.
    return jnp.array([a*b,jnp.sin(a)+b*b])
# Initialize array `x` with explicit values and shape.
x=jnp.array([0.4,-0.7])
# Initialize array `v` with explicit values and shape.
v=jnp.array([1.2,-0.3])
# Initialize array `u` with explicit values and shape.
u=jnp.array([0.5,2.0])
# Convert `(a, b)` to a host NumPy array for inspection or verification.
a,b=np.asarray(x)
# Initialize array `jacobian` with explicit values and shape.
jacobian=np.array([[b,a],[np.cos(a),2*b]])
# Initialize array `expected_value` with explicit values and shape.
expected_value=np.array([a*b,np.sin(a)+b*b])

# Step 2 — Push a direction forward and pull a measurement backward: The VJP returns one cotangent per input argument, hence the...
# Compute exact directional derivative / Jacobian / Hessian (`(value, jvp)`).
value,jvp=jax.jvp(mapping,(x,),(v,))
# Compute exact directional derivative / Jacobian / Hessian (`(value_again, pullback)`).
value_again,pullback=jax.vjp(mapping,x)
# Run `pullback` to compute `(vjp,)`.
vjp,=pullback(u)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
# Evaluate `jnp.vdot(u, jvp)` and convert the result into Python scalar/collection `left`.
# Evaluate `jnp.vdot(vjp, v)` and convert the result into Python scalar/collection `right`.
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(left,right,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Jacobian:",jacobian)
# Print diagnostic summary of the computed outputs.
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

# Step 3 — Compute curvature without materializing a Hessian: For squared residuals, the Hessian includes a...
def objective(z):
    # Return `0.5 * jnp.vdot(mapping(z), mapping(z))` to the caller.
    return 0.5*jnp.vdot(mapping(z),mapping(z))
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient=jax.grad(objective)(x)
# Differentiate the objective to obtain `hvp` via automatic differentiation.
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
# Evaluate `y` from the current inputs and state.
y=expected_value
# Initialize array `hessian_0` with explicit values and shape.
hessian_0=np.array([[0.,1.],[1.,0.]])
# Initialize array `hessian_1` with explicit values and shape.
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
# Perform matrix contraction / projection to compute `hessian`.
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Hessian-vector product:",np.asarray(hvp))
```

Expected: JVP [-0.96, 1.52527319], VJP [1.49212199, -2.6], adjoint scalar approximately 2.57054639. The independently derived Hessian-vector product agrees.

## Forward and reverse products live in different spaces

**Predict:** Should a JVP and VJP have equal coordinates just because this example has two inputs and two outputs?

![Forward and reverse products live in different spaces](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The first panel compares the JVP with the explicit product $Jv$, one bar per output coordinate. The matched values are about $-0.96$ and $1.5253$. These are predicted output changes for the selected input direction, not the function outputs themselves.

The second panel compares the VJP with $J^\mathsf{T}u$, one bar per input coordinate. Its matched values are about $1.4921$ and $-2.6$. Both plots happen to contain two coordinates, but their axes refer to different vector spaces.

### Connect it to the computation

The bars agree within each panel because the hand-derived Jacobian verifies the corresponding JAX product. The two panels should not agree with each other: their seed vectors and meanings differ. Multiplying by the opposite seed gives the common adjoint scalar, about $2.5705$.

An adjoint equality checks compatibility between two linear maps. The independent Jacobian and changed-point exercises protect against a shared wrong map. The figure does not compare runtime or memory use.

```python
# Compute figure data for: Forward and reverse products live in different spaces
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'panels':[{'kind':'bar','labels':['output 0','output 1'],'ylabel':'directional output change','series':[{'label':'JAX JVP','y':np.asarray(jvp).tolist()},{'label':'analytic J v','y':(jacobian@np.asarray(v)).tolist()}]},{'kind':'bar','labels':['input 0','input 1'],'ylabel':'input cotangent','series':[{'label':'JAX VJP','y':np.asarray(vjp).tolist()},{'label':'analytic transpose J u','y':(jacobian.T@np.asarray(u)).tolist()}]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:05:45.571348+00:00. JAX 0.9.2.

```text
Jacobian: [[-0.7         0.4       ]
 [ 0.92106099 -1.4       ]]
JVP: [-0.96        1.52527319] VJP: [ 1.49212199 -2.6       ] adjoint pair: 2.570546385606924 2.570546385606924
Hessian-vector product: [ 1.74991568 -3.38303348]
Jacobian: [[-0.7         0.4       ]
 [ 0.92106099 -1.4       ]]
JVP: [-0.96        1.52527319] VJP: [ 1.49212199 -2.6       ] adjoint pair: 2.570546385606924 2.570546385606924
Hessian-vector product: [ 1.74991568 -3.38303348]
Forward columns and reverse rows agree
Exact HVP: [ 1.74991568 -3.38303348] Gauss-Newton product: [ 2.07686964 -2.51938247]
Changed-point adjoint verified
Direction linearity holds; the Jacobian changes with the base point
Bilinear Hessian symmetry and finite-difference product verified
PASS: internals-02

```

## Reconstruct columns and rows independently

**Predict before running:** Will mapping JVPs over basis vectors directly produce rows or columns?

```python
# Experiment — Reconstruct columns and rows independently: The transpose in the forward construction is necessary: vmap...
# Initialize array `basis` with explicit values and shape.
basis=jnp.eye(2)
# Compute exact directional derivative / Jacobian / Hessian (`columns`).
columns=jax.vmap(lambda direction:jax.jvp(mapping,(x,),(direction,))[1])(basis).T
# Vectorize across the batch dimension with `jax.vmap` (`rows`).
rows=jax.vmap(lambda weighting:pullback(weighting)[0])(basis)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(columns,jacobian,rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(rows,jacobian,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Forward columns and reverse rows agree")
```

**Expected:** Both reconstructed matrices equal the hand-derived Jacobian.

The transpose in the forward construction is necessary: vmap first stacks one directional result per input basis vector.

## Separate exact curvature from Gauss–Newton

**Predict before running:** Will the residual-weighted correction vanish at our chosen point?

```python
# Experiment — Separate exact curvature from Gauss–Newton: Residuals are nonzero, and both output functions have second...
# Perform matrix / vector contraction (`@`) to compute `gauss_newton`.
gauss_newton=jacobian.T@jacobian
# Perform matrix / vector contraction (`@`) to compute `approximation`.
approximation=gauss_newton@np.asarray(v)
# Verify contract: `np.linalg.norm(np.asarray(hvp) - approximation) > 0.1`.
assert np.linalg.norm(np.asarray(hvp)-approximation)>0.1
# Evaluate `eps` from the current inputs and state.
eps=1e-4
# Differentiate the objective to obtain `fd` via automatic differentiation.
fd=(np.asarray(jax.grad(objective)(x+eps*v))-np.asarray(jax.grad(objective)(x-eps*v)))/(2*eps)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hvp,fd,rtol=1e-7,atol=1e-8)
# Print the observed values to compare against the expected result.
print("Exact HVP:",np.asarray(hvp),"Gauss-Newton product:",approximation)
```

**Expected:** The approximation differs visibly; the finite-difference gradient check matches the exact HVP.

Residuals are nonzero, and both output functions have second derivatives. Calling the approximation the Hessian would erase those contributions.

## Make it yours

Repeat the Jacobian, JVP, VJP and adjoint checks at $x=(-0.2,0.8)$ with $v=(-0.5,1.3)$ and $u=(1.1,-0.4)$. Explain which quantities belong to input and output space.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.jvp / jax.vjp / jax.jacfwd / jax.hessian` — Computes exact forward-mode JVP, reverse-mode VJP, full Jacobians, or second-order curvature.

**Step-by-step implementation plan:**
1. Initialize array `changed_x` with explicit values and shape.
2. Convert `(a2, b2)` to a host NumPy array for inspection or verification.
3. Initialize array `j2` with explicit values and shape.
4. Compute exact directional derivative / Jacobian / Hessian (`fj`).
5. Compute exact directional derivative / Jacobian / Hessian (`rj`).

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Repeat the Jacobian, JVP, VJP and adjoint checks at x=(-0.2,0.8) with...
# Initialize array `changed_x` with explicit values and shape.
changed_x = jnp.array(...)  # TODO: compute changed_x
# Convert `(a2, b2)` to a host NumPy array for inspection or verification.
a2,b2 = np.asarray(...)  # TODO: compute a2,b2
# Initialize array `j2` with explicit values and shape.
j2 = np.array(...)  # TODO: compute j2
# Compute exact directional derivative / Jacobian / Hessian (`fj`).
fj = jax.jvp(...)  # TODO: compute fj
# Compute exact directional derivative / Jacobian / Hessian (`rj`).
rj = jax.vjp(...)  # TODO: compute rj
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(fj,j2@changed_v,rtol=1e-12)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(rj,j2.T@changed_u,rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol = ...  # TODO: compute np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol
# Print the observed values to compare against the expected result.
print("Changed-point adjoint verified")
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Repeat the Jacobian, JVP, VJP and adjoint checks at x=(-0.2,0.8) with...
# Initialize array `changed_x` with explicit values and shape.
changed_x=jnp.array([-0.2,0.8]);changed_v=jnp.array([-0.5,1.3]);changed_u=jnp.array([1.1,-0.4])
# Convert `(a2, b2)` to a host NumPy array for inspection or verification.
a2,b2=np.asarray(changed_x)
# Initialize array `j2` with explicit values and shape.
j2=np.array([[b2,a2],[np.cos(a2),2*b2]])
# Compute exact directional derivative / Jacobian / Hessian (`fj`).
fj=jax.jvp(mapping,(changed_x,),(changed_v,))[1]
# Compute exact directional derivative / Jacobian / Hessian (`rj`).
rj=jax.vjp(mapping,changed_x)[1](changed_u)[0]
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(fj,j2@changed_v,rtol=1e-12)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(rj,j2.T@changed_u,rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Changed-point adjoint verified")
```

</details>

## Test linearity at a fixed point

**Practice**

Check the JVP of $2v-w$ against the same combination of separate JVPs. Then show why changing the base point is a different operation.

<details><summary>Hint</summary>

A derivative is linear in its direction at a fixed base point.

</details>

### How to write: Test linearity at a fixed point — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.jvp / jax.vjp / jax.jacfwd / jax.hessian` — Computes exact forward-mode JVP, reverse-mode VJP, full Jacobians, or second-order curvature.

**Step-by-step implementation plan:**
1. Initialize array `w` with explicit values and shape.
2. Compute exact directional derivative / Jacobian / Hessian (`product`).
3. Verify that computed values match the expected reference within numerical tolerance.
4. Compute exact directional derivative / Jacobian / Hessian (`shifted`).
5. Verify contract: `np.linalg.norm(np.asarray(shifted - product(v))) > 0.1`.

**Starter code scaffold (fill in the TODOs):**

```python
# Test linearity at a fixed point (Practice): Linearity in the tangent is a local property.
# Initialize array `w` with explicit values and shape.
w = jnp.array(...)  # TODO: compute w
# Compute exact directional derivative / Jacobian / Hessian (`product`).
product = ...  # TODO: compute product
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(product(2*v-w),2*product(v)-product(w),rtol=1e-12)
# Compute exact directional derivative / Jacobian / Hessian (`shifted`).
shifted = jax.jvp(...)  # TODO: compute shifted
# Verify contract: `np.linalg.norm(np.asarray(shifted - product(v))) > 0.1`.
assert np.linalg.norm(np.asarray(shifted-product(v)))  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print("Direction linearity holds; the Jacobian changes with the base point")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Test linearity at a fixed point (Practice): Linearity in the tangent is a local property.
# Initialize array `w` with explicit values and shape.
w=jnp.array([-0.6,0.9])
# Compute exact directional derivative / Jacobian / Hessian (`product`).
product=lambda direction:jax.jvp(mapping,(x,),(direction,))[1]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(product(2*v-w),2*product(v)-product(w),rtol=1e-12)
# Compute exact directional derivative / Jacobian / Hessian (`shifted`).
shifted=jax.jvp(mapping,(x+w,),(v,))[1]
# Verify contract: `np.linalg.norm(np.asarray(shifted - product(v))) > 0.1`.
assert np.linalg.norm(np.asarray(shifted-product(v)))>0.1
# Print the observed values to compare against the expected result.
print("Direction linearity holds; the Jacobian changes with the base point")
```

Linearity in the tangent is a local property. A nonlinear function need not use the same linear map at different points.

</details>

## Check curvature symmetry without a full Hessian

**Challenge**

For two directions $v,w$, verify $w^\mathsf{T}Hv=v^\mathsf{T}Hw$, then verify one product with finite differences.

<details><summary>Hint</summary>

Compose jvp with grad and compare scalar products.

</details>

### How to write: Check curvature symmetry without a full Hessian — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.
- `jax.jvp / jax.vjp / jax.jacfwd / jax.hessian` — Computes exact forward-mode JVP, reverse-mode VJP, full Jacobians, or second-order curvature.

**Step-by-step implementation plan:**
1. Initialize array `w` with explicit values and shape.
2. Differentiate the objective to obtain `hw` via automatic differentiation.
3. Verify that computed values match the expected reference within numerical tolerance.
4. Evaluate `eps` from the current inputs and state.
5. Differentiate the objective to obtain `fdw` via automatic differentiation.

**Starter code scaffold (fill in the TODOs):**

```python
# Check curvature symmetry without a full Hessian (Challenge): For this smooth scalar objective, mixed partials agree.
# Initialize array `w` with explicit values and shape.
w = jnp.array(...)  # TODO: compute w
# Differentiate the objective to obtain `hw` via automatic differentiation.
hw = jax.jvp(...)  # TODO: compute hw
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol = ...  # TODO: compute np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol
# Evaluate `eps` from the current inputs and state.
eps = ...  # TODO: compute eps
# Differentiate the objective to obtain `fdw` via automatic differentiation.
fdw = ...  # TODO: compute fdw
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hw,fdw,rtol = ...  # TODO: compute np.testing.assert_allclose(hw,fdw,rtol
# Print the observed values to compare against the expected result.
print("Bilinear Hessian symmetry and finite-difference product verified")
```

<details><summary>Reference solution and reasoning</summary>

```python
# Check curvature symmetry without a full Hessian (Challenge): For this smooth scalar objective, mixed partials agree.
# Initialize array `w` with explicit values and shape.
w=jnp.array([0.2,0.9])
# Differentiate the objective to obtain `hw` via automatic differentiation.
hw=jax.jvp(jax.grad(objective),(x,),(w,))[1]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol=1e-12)
# Evaluate `eps` from the current inputs and state.
eps=1e-4
# Differentiate the objective to obtain `fdw` via automatic differentiation.
fdw=(jax.grad(objective)(x+eps*w)-jax.grad(objective)(x-eps*w))/(2*eps)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hw,fdw,rtol=1e-7,atol=1e-8)
# Print the observed values to compare against the expected result.
print("Bilinear Hessian symmetry and finite-difference product verified")
```

For this smooth scalar objective, mixed partials agree. Symmetry is a useful invariant, but an independent derivative check is still necessary.

</details>

## Check your understanding

A VJP is seeded with an output weighting u. What does the returned input cotangent represent?

1. The output at a displaced input x+u.
2. The gradient of the scalar weighted output u-transpose f(x), with u fixed.
3. A complete Hessian for every possible loss.

<details><summary>Answer and explanation</summary>

The gradient of the scalar weighted output u-transpose f(x), with u fixed.

The pullback applies the transpose Jacobian to the output weighting. It maps an output cotangent into the input space at the stored base point.

</details>

## Diagnose the result

If a product has the wrong shape, identify whether its seed lives in input or output space. If basis reconstruction is transposed, inspect which index vmap stacks first. If an HVP differs from J-transpose-J times the direction, calculate the residual-weighted second-derivative term before blaming autodiff.

## Carry forward

- Separate the point from the direction
- A VJP answers a different directional question
- Check the adjoint identity with independent values
- Count directions before choosing a Jacobian strategy
- Curvature includes the changing Jacobian
- Use finite differences as a second perspective

## Keep your evidence

Hand-derived Jacobian/Hessian, adjoint scalar, changed-point checks, finite-difference HVP and a Gauss–Newton distinction. Keep the environment, observed outputs and your explanation of the figure.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [JAX automatic differentiation cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
- [Custom JVP and VJP rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
- [Ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html)
- [The jaxpr language](https://docs.jax.dev/en/latest/601/jaxpr.html)

