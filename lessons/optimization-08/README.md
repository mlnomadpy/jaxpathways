# Curvature, Hessians, and learning rates

Phase 04: Math & optimization · about 75 minutes · CPU

## What you will be able to do

- Use curvature to explain learning-rate stability, slow directions and why a zero gradient need not be a minimum.
- Read a bowl through its curvature
- Derive a stability bound
- See why scaling helps
- Distinguish a minimum from a stationary point

## The problem

Why can the same learning rate crawl along one direction and explode along another? A gradient tells us the local slope. Curvature tells us how quickly that slope changes. We will use a two-dimensional bowl whose behavior can be predicted exactly.

## The idea

Curvature describes how quickly the gradient changes as we move. A direction with high curvature can overshoot under a learning rate that makes slow progress in a flatter direction. Contours make this imbalance easier to see than loss alone.

## Why one learning rate must respect the steep direction

For the illustrative quadratic $L(x,y)=(x^2+10y^2)/2$, a gradient step multiplies the coordinates by $1-\eta$ and $1-10\eta$. Here $\eta$ is the learning rate. With $\eta=0.1$, the steep coordinate reaches zero in one exact-arithmetic step while the other shrinks by $0.9$.

With $\eta=0.25$, the steep coordinate is multiplied by $-1.5$: it changes sign and grows in magnitude. The flatter coordinate still shrinks, so observing one coordinate could make the update look healthy while the total objective diverges.

Sketch narrow elliptical contours for this quadratic and mark the first two iterates. The saved coordinate plot shows how each direction progresses. Keep the objective coefficients, initial point and rate labels aligned when comparing this calculation with a trajectory; a path from another quadratic would tell a different story.

### Pause and reason

For this quadratic, what condition on a positive learning rate contracts both coordinates?

<details><summary>Compare your reasoning</summary>

Require $\lvert1-10\eta\rvert<1$ and $\lvert1-\eta\rvert<1$. For a positive rate, both hold when $0<\eta<0.2$. At the boundary the steep coordinate can oscillate without shrinking.

</details>

## Read a bowl through its curvature

Consider $L(w)=\tfrac12(w_0^2+10w_1^2)$. Its gradient is $(w_0,10w_1)$, and its Hessian is diagonal with entries $1$ and $10$. Moving equally far along the second axis changes the slope ten times as much. An eigenvector of a matrix keeps its direction when multiplied by that matrix; its eigenvalue is the scale factor. Here the coordinate axes are eigenvectors, and their eigenvalues are those diagonal entries.

$$
H_{ij}=\frac{\partial^2 L}{\partial w_i\partial w_j},\qquad H=\begin{pmatrix}1&0\\0&10\end{pmatrix}
$$

## Derive a stability bound

Gradient descent uses $w_{t+1}=w_t-\eta\nabla L(w_t)$, where $\eta$ is the learning rate. Along an eigenvector with positive curvature $\lambda$, the error is multiplied by $1-\eta\lambda$. Its magnitude shrinks only when $|1-\eta\lambda|<1$. All directions of this strictly convex quadratic shrink when $0<\eta<2/\lambda_{\max}=0.2$. At the boundary the steep coordinate flips sign without shrinking. This exact global bound belongs to this constant-curvature quadratic; a nonquadratic model needs more care.

$$
0<\eta<\frac{2}{\lambda_{\max}}
$$

## See why scaling helps

At $\eta=0.1$, the steep coordinate reaches zero immediately, while the shallow coordinate shrinks by $0.9$ each step. Reparameterize with $z=(w_0,\sqrt{10}w_1)$. The same objective becomes $\tfrac12\|z\|^2$, an evenly curved bowl. A step of size $1$ in these coordinates reaches the optimum. Real feature standardization is a practical approximation to improving this geometry; it does not remove all correlations or guarantee one-step convergence.

## Distinguish a minimum from a stationary point

At a stationary point the gradient is zero. Positive definite curvature gives a strict local minimum for a twice continuously differentiable function; an indefinite Hessian has both rising and falling directions and indicates a saddle. For $S(w)=w_0^2-w_1^2$, the origin is a saddle despite its zero gradient. A positive semidefinite Hessian alone can be inconclusive. For large models, avoid storing the full Hessian when a product $Hv$ answers your question; use a JVP of the gradient.

## Write the bowl

Create main.py with the objective and independently known curvature.

```python
import jax
import jax.numpy as jnp

def loss(w):
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
w = jnp.array([1.0, 1.0])
H = jnp.diag(jnp.array([1.0, 10.0]))
assert jnp.allclose(jax.grad(loss)(w), H @ w)
```

At the starting point the gradient is $(1,10)$.

## Check second derivatives

Append the full Hessian and a product check.

```python
assert jnp.allclose(jax.hessian(loss)(w), H)
v = jnp.array([2.0, -1.0])
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))
```

The product agrees with the diagonal matrix without constructing a Hessian inside the JVP.

## Run an explicit trajectory

Append this scan and run python main.py. The history records losses after each update.

```python
def run(rate, steps=20):

    def step(w, _):
        w = w - rate * jax.grad(loss)(w)
        return (w, loss(w))
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
final, history = run(0.1)
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
assert jnp.all(jnp.diff(history) <= 1e-06)
print('final weights / loss:', final, history[-1])
```

The second coordinate vanishes; the first remains near $0.1216$ after $20$ steps.

## Run the example

```python
import jax
import jax.numpy as jnp

def loss(w):
    return 0.5 * (w[0] ** 2 + 10 * w[1] ** 2)
w = jnp.array([1.0, 1.0])
H = jnp.diag(jnp.array([1.0, 10.0]))
assert jnp.allclose(jax.grad(loss)(w), H @ w)

assert jnp.allclose(jax.hessian(loss)(w), H)
v = jnp.array([2.0, -1.0])
hv = jax.jvp(jax.grad(loss), (w,), (v,))[1]
assert jnp.allclose(hv, jnp.array([2.0, -10.0]))

def run(rate, steps=20):

    def step(w, _):
        w = w - rate * jax.grad(loss)(w)
        return (w, loss(w))
    return jax.lax.scan(step, jnp.array([1.0, 1.0]), None, length=steps)
final, history = run(0.1)
assert jnp.allclose(final, jnp.array([0.9 ** 20, 0.0]), atol=2e-06)
assert jnp.all(jnp.diff(history) <= 1e-06)
print('final weights / loss:', final, history[-1])
```

Expected: At rate $0.1$, the final loss is about $0.00739$ and the trajectory is decreasing.

## Unequal curvature produces unequal progress

**Predict:** At rate $0.1$, which coordinate reaches zero immediately?

![Unequal curvature produces unequal progress](../../phases/04-optimization/08-curvature-hessians-and-learning-rates/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis counts completed updates; the point at $0$ is the initial state. The vertical axis shows a parameter coordinate, not the total loss. Both coordinates start at $1$, but the coordinate with curvature $10$ drops to zero after one update, while the coordinate with curvature $1$ decays gradually.

The gradual curve starts $1,0.9,0.81$ and is still about $0.122$ after $20$ updates. The dashed curve remains at zero after its first drop. It has reached its optimum in this particular direction.

### Connect it to the computation

For a quadratic direction with curvature $\lambda$, gradient descent multiplies the coordinate by $1-\eta\lambda$. At learning rate $0.1$, that multiplier is $0.9$ for curvature $1$ and zero for curvature $10$. The unequal curves come from curvature interacting with the same learning rate.

The steeper direction is faster here because the rate exactly cancels it. Increasing the rate further does not keep making it better: sufficiently large steps produce oscillation or divergence. This is why one shared rate can be constrained by a steep direction while making slow progress in a flatter one.

```python
points = [jnp.array([1.0, 1.0])]
for _ in range(20):
    points.append(points[-1] - 0.1 * jax.grad(loss)(points[-1]))
trace = jnp.stack(points)
visual_data = {'kind': 'line', 'x': list(range(21)), 'xlabel': 'completed update', 'ylabel': 'coordinate value', 'series': [{'label': 'curvature 1', 'y': trace[:, 0].tolist()}, {'label': 'curvature 10', 'y': trace[:, 1].tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-06T22:59:10.824688+00:00. JAX 0.9.2.

```text
final weights / loss: [ 0.12157664 -0.        ] 0.0073904395
final weights / loss: [ 0.12157664 -0.        ] 0.0073904395
PASS: optimization-08

```

## Cross the stability boundary

**Predict before running:** Predict the steep coordinate at rates $0.2$ and $0.21$.

```python
edge, _ = run(0.2)
bad, bad_history = run(0.21)
assert jnp.allclose(jnp.abs(edge[1]), 1.0, atol=2e-06)
assert jnp.abs(bad[1]) > 6.0
assert bad_history[-1] > loss(w)
```

**Expected:** At the boundary its magnitude stays $1$; above it the magnitude grows.

A learning-rate bound depends on curvature, not on a universal default value.

## Change coordinates

**Predict before running:** For $\tfrac12\|z\|^2$, what happens after a gradient step of size $1$?

```python
z = jnp.array([w[0], jnp.sqrt(10.0) * w[1]])
round_loss = lambda z: 0.5 * jnp.dot(z, z)
z_next = z - jax.grad(round_loss)(z)
assert jnp.allclose(z_next, jnp.zeros(2))
assert jnp.allclose(round_loss(z), loss(w))
```

**Expected:** The transformed objective starts at the same value and reaches zero in one step.

Coordinates can change optimization geometry while representing the same underlying objective.

## Make it yours

For curvatures $(2,8)$, derive the stable interval and check one step at rate $0.1$ from $(1,1)$.

<details><summary>Reference solution</summary>

```python
H2 = jnp.diag(jnp.array([2.0, 8.0]))
w_next = w - 0.1 * (H2 @ w)
assert jnp.allclose(w_next, jnp.array([0.8, 0.2]), atol=1e-06)
assert jnp.allclose(2 / jnp.max(jnp.linalg.eigvalsh(H2)), 0.25)
```

</details>

## Reject the zero-gradient shortcut

**Transfer / diagnosis**

At the origin of $S(w)=w_0^2-w_1^2$, show zero gradient and both a lower and higher nearby value.

<details><summary>Hint</summary>

Move by $0.1$ along each axis separately.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
saddle = lambda z: z[0] ** 2 - z[1] ** 2
zero = jnp.zeros(2)
assert jnp.allclose(jax.grad(saddle)(zero), zero)
assert saddle(jnp.array([0.0, 0.1])) < saddle(zero)
assert saddle(jnp.array([0.1, 0.0])) > saddle(zero)
assert jnp.allclose(jnp.linalg.eigvalsh(jax.hessian(saddle)(zero)), jnp.array([-2.0, 2.0]))
```

Stationarity is a first-order condition, not a proof of a minimum.

</details>

## Transfer to a nonquadratic function

**Transfer / diagnosis**

For $F(w)=\sum_i e^{w_i}$, verify a Hessian-vector product at $(0,\log 2)$ along $(1,3)$.

<details><summary>Hint</summary>

The analytic Hessian is diagonal with entries $e^{w_i}$.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
point = jnp.array([0.0, jnp.log(2.0)])
direction = jnp.array([1.0, 3.0])
product = jax.jvp(jax.grad(lambda z: jnp.sum(jnp.exp(z))), (point,), (direction,))[1]
assert jnp.allclose(product, jnp.array([1.0, 6.0]), atol=1e-06)
```

For a nonquadratic function the Hessian changes with the point; a bound from one location is not a global guarantee.

</details>

## Check your understanding

A smooth function has zero gradient at a point. What can we conclude?

1. It is a global minimum
2. It is stationary; curvature or other evidence is needed
3. The optimizer implementation is broken

<details><summary>Answer and explanation</summary>

It is stationary; curvature or other evidence is needed

A zero gradient is compatible with a minimum, maximum or saddle. Even second-order information can be inconclusive in flat directions.

</details>

## Diagnose the result

If a run oscillates or diverges, inspect the steepest curvature and the effective step size. If it barely moves, compare directional scales. If a zero gradient appears suspicious, probe nearby values before declaring success.

## Carry forward

- A learning-rate bound depends on curvature, not on a universal default value.
- Coordinates can change optimization geometry while representing the same underlying objective.

## Keep your evidence

Keep the Hessian and Hessian-vector checks, stable/divergent trajectories, rescaling comparison and saddle counterexample.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX autodiff cookbook: Hessian-vector products](https://docs.jax.dev/en/latest/301/cookbook.html)
- [JAX Hessian API](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html)

