# JVPs, VJPs, and higher-order derivatives

Phase 14: Autodiff & JAX internals · about 105 minutes · CPU

## What you will be able to do

- Compute and interpret JVP and VJP products with correct domains and shapes.
- Verify the adjoint dot-product identity against a hand-derived Jacobian.
- Compute an exact Hessian-vector product and distinguish it from a Gauss–Newton approximation.

## The problem

A large model may have millions of inputs or outputs, but you often need only one sensitivity direction. How can we compute the useful product without constructing the full Jacobian—and verify that forward, reverse and higher-order calculations agree?

## The idea

A derivative is a local linear map. A JVP pushes an input perturbation $v$ through that map; a VJP pulls an output weighting $u$ back to the inputs. We will make the full Jacobian only for a tiny reference problem, then use its algebra to check the products and a Hessian-vector product.

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
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def mapping(x):
    a,b=x
    return jnp.array([a*b,jnp.sin(a)+b*b])
x=jnp.array([0.4,-0.7])
v=jnp.array([1.2,-0.3])
u=jnp.array([0.5,2.0])
a,b=np.asarray(x)
jacobian=np.array([[b,a],[np.cos(a),2*b]])
expected_value=np.array([a*b,np.sin(a)+b*b])
```

The map has two inputs and two outputs. Its Jacobian rows correspond to output coordinates, and columns to input coordinates.

## Push a direction forward and pull a measurement backward

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
value,jvp=jax.jvp(mapping,(x,),(v,))
value_again,pullback=jax.vjp(mapping,x)
vjp,=pullback(u)
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
np.testing.assert_allclose(left,right,rtol=1e-12)
print("Jacobian:",jacobian)
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)
```

The VJP returns one cotangent per input argument, hence the one-element tuple. The dot-product identity checks that forward and reverse products represent adjoint linear maps.

## Compute curvature without materializing a Hessian

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
def objective(z):
    return 0.5*jnp.vdot(mapping(z),mapping(z))
gradient=jax.grad(objective)(x)
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
y=expected_value
hessian_0=np.array([[0.,1.],[1.,0.]])
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
print("Hessian-vector product:",np.asarray(hvp))
```

For squared residuals, the Hessian includes a Jacobian-transpose-Jacobian term and residual-weighted second derivatives. Dropping the latter is an approximation, not the exact Hessian.

## Run the example

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def mapping(x):
    a,b=x
    return jnp.array([a*b,jnp.sin(a)+b*b])
x=jnp.array([0.4,-0.7])
v=jnp.array([1.2,-0.3])
u=jnp.array([0.5,2.0])
a,b=np.asarray(x)
jacobian=np.array([[b,a],[np.cos(a),2*b]])
expected_value=np.array([a*b,np.sin(a)+b*b])

value,jvp=jax.jvp(mapping,(x,),(v,))
value_again,pullback=jax.vjp(mapping,x)
vjp,=pullback(u)
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
np.testing.assert_allclose(left,right,rtol=1e-12)
print("Jacobian:",jacobian)
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

def objective(z):
    return 0.5*jnp.vdot(mapping(z),mapping(z))
gradient=jax.grad(objective)(x)
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
y=expected_value
hessian_0=np.array([[0.,1.],[1.,0.]])
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
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
visual_data={'panels':[{'kind':'bar','labels':['output 0','output 1'],'ylabel':'directional output change','series':[{'label':'JAX JVP','y':np.asarray(jvp).tolist()},{'label':'analytic J v','y':(jacobian@np.asarray(v)).tolist()}]},{'kind':'bar','labels':['input 0','input 1'],'ylabel':'input cotangent','series':[{'label':'JAX VJP','y':np.asarray(vjp).tolist()},{'label':'analytic transpose J u','y':(jacobian.T@np.asarray(u)).tolist()}]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:26:01.713639+00:00. JAX 0.9.2.

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
basis=jnp.eye(2)
columns=jax.vmap(lambda direction:jax.jvp(mapping,(x,),(direction,))[1])(basis).T
rows=jax.vmap(lambda weighting:pullback(weighting)[0])(basis)
np.testing.assert_allclose(columns,jacobian,rtol=1e-12)
np.testing.assert_allclose(rows,jacobian,rtol=1e-12)
print("Forward columns and reverse rows agree")
```

**Expected:** Both reconstructed matrices equal the hand-derived Jacobian.

The transpose in the forward construction is necessary: vmap first stacks one directional result per input basis vector.

## Separate exact curvature from Gauss–Newton

**Predict before running:** Will the residual-weighted correction vanish at our chosen point?

```python
gauss_newton=jacobian.T@jacobian
approximation=gauss_newton@np.asarray(v)
assert np.linalg.norm(np.asarray(hvp)-approximation)>0.1
eps=1e-4
fd=(np.asarray(jax.grad(objective)(x+eps*v))-np.asarray(jax.grad(objective)(x-eps*v)))/(2*eps)
np.testing.assert_allclose(hvp,fd,rtol=1e-7,atol=1e-8)
print("Exact HVP:",np.asarray(hvp),"Gauss-Newton product:",approximation)
```

**Expected:** The approximation differs visibly; the finite-difference gradient check matches the exact HVP.

Residuals are nonzero, and both output functions have second derivatives. Calling the approximation the Hessian would erase those contributions.

## Make it yours

Repeat the Jacobian, JVP, VJP and adjoint checks at $x=(-0.2,0.8)$ with $v=(-0.5,1.3)$ and $u=(1.1,-0.4)$. Explain which quantities belong to input and output space.

<details><summary>Reference solution</summary>

```python
changed_x=jnp.array([-0.2,0.8]);changed_v=jnp.array([-0.5,1.3]);changed_u=jnp.array([1.1,-0.4])
a2,b2=np.asarray(changed_x)
j2=np.array([[b2,a2],[np.cos(a2),2*b2]])
fj=jax.jvp(mapping,(changed_x,),(changed_v,))[1]
rj=jax.vjp(mapping,changed_x)[1](changed_u)[0]
np.testing.assert_allclose(fj,j2@changed_v,rtol=1e-12)
np.testing.assert_allclose(rj,j2.T@changed_u,rtol=1e-12)
np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol=1e-12)
print("Changed-point adjoint verified")
```

</details>

## Test linearity at a fixed point

**Practice**

Check the JVP of $2v-w$ against the same combination of separate JVPs. Then show why changing the base point is a different operation.

<details><summary>Hint</summary>

A derivative is linear in its direction at a fixed base point.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
w=jnp.array([-0.6,0.9])
product=lambda direction:jax.jvp(mapping,(x,),(direction,))[1]
np.testing.assert_allclose(product(2*v-w),2*product(v)-product(w),rtol=1e-12)
shifted=jax.jvp(mapping,(x+w,),(v,))[1]
assert np.linalg.norm(np.asarray(shifted-product(v)))>0.1
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

<details><summary>Reference solution and reasoning</summary>

```python
w=jnp.array([0.2,0.9])
hw=jax.jvp(jax.grad(objective),(x,),(w,))[1]
np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol=1e-12)
eps=1e-4
fdw=(jax.grad(objective)(x+eps*w)-jax.grad(objective)(x-eps*w))/(2*eps)
np.testing.assert_allclose(hw,fdw,rtol=1e-7,atol=1e-8)
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

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX automatic differentiation cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
- [Custom JVP and VJP rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
- [Ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html)
- [The jaxpr language](https://docs.jax.dev/en/latest/601/jaxpr.html)

