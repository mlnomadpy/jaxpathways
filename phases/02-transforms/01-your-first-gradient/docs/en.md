# Your first gradient

Phase 02: JAX transformations · about 55 minutes · CPU

## What you will be able to do

- Predict the sign and magnitude of a derivative from a local change.
- Trace the chain rule through a composed numerical function.
- Compare an analytic derivative, a finite-difference estimate, and JAX autodiff.
- Distinguish a stationary point from a minimum and repair scalar-output/input-type errors.

## The problem

Imagine that you can turn one dial, and a score tells you how far you are from the result you want. Which way should you turn it? We’ll begin with the curve $f(x)=x^2$, where we can check the answer by hand. A derivative tells us how the score changes nearby; then JAX will calculate that derivative for us. You do not need to guess what the tool is doing.

## The idea

A derivative tells us how an output changes near a particular input. We can understand it before using autodiff: move the input a small amount, estimate the output change from the local slope, then compare that prediction with the function.

## Use a slope to predict a nearby change

For a separate hand calculation, let $f(x)=x^2$. At $x=3$, the derivative is $6$. Moving to $3.01$ gives a first-order prediction of $9+6(0.01)=9.06$. The exact value is $9.0601$; the small difference comes from curvature.

In the tangent figure, the tangent agrees with the curve at the selected point and has the same local slope. It need not follow the curve far away. Move the point and watch the slope change; then move only a small distance from that point to judge the linear approximation.

Autodiff computes the derivative of the operations you wrote. It does not choose a meaningful objective for you. A gradient of the wrong loss can be numerically correct, which is why later lessons check both the formula and its derivative.

### Pause and reason

Does a positive derivative mean the function value itself is positive?

<details><summary>Compare your reasoning</summary>

No. The derivative describes local change. For $f(x)=x-10$ at $x=0$, the value is $-10$ while the derivative is $1$.

</details>

## Read a slope before you compute it

Let’s start at $x=3$. If we move right by $\delta=0.01$, the score changes from $9$ to $9.0601$. That is an increase of $0.0601$. The slope predicts an increase of about $6\times0.01=0.06$, which is close.

Here $\delta$ means a small change in the input, and $f\prime(x)$ means the slope at that input. Expanding the square gives the exact change below. The term $\delta^2$ is what the straight-line approximation leaves out. As the change gets smaller, that term becomes small compared with the linear term when the slope is nonzero.

Before running the code, try a negative input. At $x=-3$, moving a little to the right reduces the score. The slope should therefore be negative. At zero, the slope is zero even though moving away in either direction increases the square.

$$
\begin{aligned}f(x+\delta)-f(x)&=2x\delta+\delta^2\\f\prime(x)&=2x\end{aligned}
$$

## Three ways to obtain a derivative

An analytic derivative comes from reasoning about the mathematical function. A finite difference approximates a derivative by evaluating the function at nearby inputs. Automatic differentiation applies derivative rules to the operations that execute in the program and composes those rules with the chain rule.

Autodiff is not estimating a slope with a hidden step size, and it is not a symbolic algebra system simplifying a formula. Your program still runs floating-point operations. Those operations can overflow, lose precision, or evaluate a function whose mathematical derivative is not well-defined at the chosen point. Agreement among independent approaches is stronger evidence than merely printing one number.

## Follow the chain rule through the program

Now split the calculation into two small steps. First compute $u=3x+1$, then square it to get $y=u^2$. A small change in $x$ first changes $u$; that change in $u$ then changes $y$. The chain rule multiplies the two local slopes.

The first slope is $3$. The second is $2u$. Multiplying gives $6(3x+1)$. At $x=2$, we have $u=7$, so the derivative is $42$. This gives us a hand-calculated answer for the JAX check. Autodiff follows the operations in the program; it is not estimating the derivative by repeatedly nudging the input.

$$
\begin{aligned}u&=3x+1,\qquad y=u^2\\\frac{dy}{dx}&=\frac{dy}{du}\frac{du}{dx}=2u\cdot3\end{aligned}
$$

## Know the contract of grad

The default grad transformation differentiates the first positional argument. Its differentiated inputs must have an appropriate inexact dtype, such as a floating-point array, and its output must be scalar: shape $()$, not shape $(1,)$. A Python float like $3.0$ works; the integer $3$ is a different input type.

A vector of residuals is useful model output, but it does not yet specify a single objective. Reducing the residuals to a sum or a mean chooses an objective and its scaling. A Jacobian is the more appropriate object when you really want the derivatives of every vector output. We will first learn scalar objectives, then return to batching and derivative structure.

## A zero gradient is a question, not a verdict

Both $x^2$ and $x^3$ have derivative zero at $x=0$. The square has a minimum there; the cubic keeps decreasing toward the left. The constant function $x\mapsto5$ also has zero derivative everywhere. A gradient is local first-order information, so it does not by itself prove optimality, global quality, or a correct implementation.

When a model produces a surprising zero gradient, ask whether the loss actually depends on the parameter, whether you differentiated the intended argument, and whether the chosen point is stationary. Checking nearby points and a simple analytic case is often more informative than immediately changing the optimizer.

## Build a numerical estimate

```python
def central_difference(function, x, h=1e-3):
    return (function(x + h) - function(x - h)) / (2 * h)

def square(x):
    return x * x

estimate = central_difference(square, 3.0)
print("Finite-difference estimate:", estimate)
assert abs(estimate - 6.0) < 1e-8
```

Approximately $6.0$. This estimate evaluates the function at two nearby points; it is not automatic differentiation.

## Run the example

```python
import jax
import jax.numpy as jnp
def f(x):
    return x ** 2
derivative = jax.grad(f)
print(float(derivative(3.0)))
assert jnp.allclose(derivative(3.0), 6.)
assert jnp.allclose(derivative(0.0), 0.)
```

Expected: $6.0$

## A tangent describes local change

**Predict:** At $x=3$, is the slope equal to the height of the curve?

![A tangent describes local change](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The solid curve is $f(x)=x^2$. The dashed line is its tangent at the marked point $(3,9)$. The horizontal axis measures the input; the vertical axis shows either the function value or the tangent’s predicted value.

Near the marked point the two almost coincide. Farther away they separate: at $x=2$, the curve is $4$ and the tangent is $3$. At $x=0$, the curve is $0$, but the tangent has fallen to $-9$.

### Connect it to the computation

The derivative at $x=3$ is $6$, so the local prediction is $f(3+\Delta x)\approx9+6\Delta x$. The exact value also contains $(\Delta x)^2$. That omitted term explains why the approximation is good for small changes and increasingly poor for large ones.

The negative part of the dashed line does not mean a square became negative. It means a local linear approximation was extended beyond the neighborhood where it is useful. A gradient describes local change, not the complete function.

```python
grid = jnp.linspace(-4.0, 4.0, 81)
point = 3.0
slope = derivative(point)
visual_data = {'kind': 'line', 'x': grid.tolist(), 'xlabel': 'input x', 'ylabel': 'function / tangent value', 'series': [{'label': 'f(x) = x squared', 'y': jax.vmap(f)(grid).tolist()}, {'label': 'tangent at x = 3', 'y': (f(point) + slope * (grid - point)).tolist()}]}
visual_data['markers'] = [{'x': point, 'y': float(f(point)), 'label': 'selected input'}]
```

## Recorded reference execution

CPU run: 2026-10-06T22:58:30.139815+00:00. JAX 0.9.2.

```text
Finite-difference estimate: 5.999999999999339
6.0
chain rule: -1.0 -12.0
chain rule: 0.0 6.0
chain rule: 2.0 42.0
h / estimate / error: 0.1 27.009973526000977 0.009973526000976562
h / estimate / error: 0.01 27.000045776367188 4.57763671875e-05
h / estimate / error: 0.001 26.9975643157959 0.0024356842041015625
h / estimate / error: 1e-05 27.0843505859375 0.0843505859375
h / estimate / error: 1e-07 0.0 27.0
PASS: first-gradient

```

## Check the chain rule at three inputs

**Predict before running:** For $g(x)=(3x+1)^2$, predict the derivative at $-1$, $0$, and $2$. Explain why the sign changes.

```python
def composed(x):
    return (3. * x + 1.) ** 2
for point in (-1., 0., 2.):
    observed = jax.grad(composed)(point)
    expected = 6. * (3. * point + 1.)
    print("chain rule:", point, float(observed))
    assert jnp.allclose(observed, expected)
```

**Expected:** The three derivatives are $-12$, $6$, and $42$.

The inner derivative contributes a factor of $3$. Omitting that factor is a chain-rule error; matching only one point can hide a wrong expression.

## Choose a finite-difference scale

**Predict before running:** For the float32 function $x^3$ at $x=3$, will shrinking $h$ forever keep improving the estimate? Run the sweep and compare with $27$.

```python
def cubic32(x):
    return x ** 3
point = jnp.array(3., dtype=jnp.float32)
for step in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7):
    estimate = (cubic32(point + step) - cubic32(point - step)) / (2. * step)
    print("h / estimate / error:", step, float(estimate), float(jnp.abs(estimate - 27.)))
assert jnp.allclose(jax.grad(cubic32)(point), 27.)
```

**Expected:** The errors need not decrease monotonically. At sufficiently small $h$, rounding can erase the input change.

A central difference has truncation error at larger $h$ and cancellation/rounding error at very small $h$. The original numerical-estimate example used Python floats; this sweep deliberately uses a JAX float32 array. Do not silently compare their precision as if it were identical.

## Make it yours

Foundation · Define $f(x)=x^3$. Derive $f\prime(x)$, predict the values at $-2$, $0$, and $3$, then verify every prediction with JAX. Explain why the zero at $x=0$ does not establish a minimum.

<details><summary>Reference solution</summary>

```python
def cubic(x):
    return x ** 3
for point in (-2., 0., 3.):
    assert jnp.allclose(jax.grad(cubic)(point), 3. * point ** 2)
assert cubic(-0.1) < cubic(0.) < cubic(0.1)
```

</details>

## Differentiate the parameter you intended

**Practice**

For the prediction $\hat y=wx$, find the derivative with respect to $w$ and then with respect to $x$. Evaluate them at $w=2$ and $x=3$. Predict both answers before choosing `argnums`.

<details><summary>Hint</summary>

grad defaults to `argnums=0.` The other input remains an input to the transformed callable.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def scalar_prediction(weight, x):
    return weight * x
assert jnp.allclose(jax.grad(scalar_prediction, argnums=0)(2., 3.), 3.)
assert jnp.allclose(jax.grad(scalar_prediction, argnums=1)(2., 3.), 2.)
```

A parameter gradient and an input sensitivity answer different questions even when they originate from the same function.

</details>

## Repair an objective without hiding its meaning

**Challenge**

A function returns $[x^2,(x-2)^2]$. Explain why ordinary grad rejects it. Build a scalar objective using their mean, derive its derivative, and check $x=0$, $1$, and $2$. Would a sum have the same derivative magnitude?

<details><summary>Hint</summary>

Write the reduction as a mathematical function first: its mean is $x^2-2x+2$.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def two_costs(x):
    return jnp.array([x ** 2, (x - 2.) ** 2])
def mean_cost(x):
    return jnp.mean(two_costs(x))
for point in (0., 1., 2.):
    assert jnp.allclose(jax.grad(mean_cost)(point), 2. * point - 2.)
    assert jnp.allclose(jax.grad(lambda z: jnp.sum(two_costs(z)))(point), 2. * jax.grad(mean_cost)(point))
```

Mean and sum agree on the minimizer here but differ by a factor of two in gradient magnitude. Reducing output is a modeling choice, not just a syntax repair.

</details>

## Check your understanding

For $f(x)=x^3$, JAX reports $f\prime(0)=0$. What can you conclude?

1. $x=0$ is a global minimum.
2. The first-order local slope is zero; optimality requires more evidence.
3. JAX has stopped differentiating because the output is zero.

<details><summary>Answer and explanation</summary>

The first-order local slope is zero; optimality requires more evidence.

The cubic is stationary at zero but takes smaller values to the left. A zero first derivative alone does not prove a minimum.

</details>

## Diagnose the result

If grad reports a scalar-output error, inspect the output shape before adding an arbitrary reduction. If it reports an integer-input error, check the dtype of the differentiated argument. If a gradient is unexpectedly zero, evaluate the function at nearby inputs and verify argument selection. If finite differences disagree, record dtype and $h$, sweep $h$, and compare with an analytic test case. Each symptom calls for different evidence.

## Carry forward

- A derivative is a local sensitivity; a gradient transformation returns a callable.
- Autodiff applies the chain rule to program operations, while finite differences approximate with input perturbations.
- Scalar objective, input dtype, and differentiated argument are part of the contract.
- A stationary point and a minimum are different claims.

## Keep your evidence

Keep predictions and analytic derivations at multiple inputs, the finite-difference step/error sweep with dtype, both argnums checks, and the scalar-objective repair. Explain why a zero gradient need not mark a minimum.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX: automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [JAX: grad API and argument contract](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)

