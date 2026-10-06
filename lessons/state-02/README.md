# Pytrees and structured parameters

Phase 03: State, randomness & control flow · about 60 minutes · CPU

## What you will be able to do

- Identify containers, leaves and structure
- Derive the matching gradient tree
- Apply corresponding leaves, not unrelated arrays
- Keep metadata separate from differentiated state

## The problem

Our model has a weight vector and a bias. How do we keep each parameter paired with its gradient? We’ll store them in a small nested structure called a pytree, calculate one update by hand, and check that JAX follows the same structure. The tree is just a way to keep related values organized.

## The idea

A pytree is a nested structure whose leaves hold values such as parameters. JAX can transform the leaves while preserving the structure. This makes a model's parameter groups explicit and lets us compare the gradient tree with the parameters it differentiates.

## Match leaves by meaning as well as shape

Suppose parameters contain a dense-layer weight matrix and a bias vector. Each gradient leaf describes sensitivity to its corresponding parameter leaf. Match paths such as layer/weight and layer/bias before comparing shapes; two equal-shaped leaves can still be swapped.

Optimizer state is related but need not have exactly the parameter tree's structure. It may contain moment trees plus a scalar step counter or other metadata. A useful diagram aligns matching parameter/gradient paths and then shows the additional optimizer state separately.

When a tree operation fails, inspect structure, leaf paths, shapes and dtypes in that order. Flattening everything immediately may hide the semantic distinction that would explain the error.

### Pause and reason

Two parameter leaves have the same shape. Is swapping their gradients safe?

<details><summary>Compare your reasoning</summary>

No. Shape compatibility does not identify which parameter a derivative belongs to. Preserve tree paths and compare the result with an independently known update.

</details>

## Identify containers, leaves and structure

A pytree separates a nested container structure from its leaves. A dictionary with weight and bias keys is a tree; each array is a leaf. A leaf can itself have many numerical elements. Two leaves therefore do not mean two scalar parameters.

JAX transformations understand common dictionaries, lists and tuples. tree.flatten returns an ordered leaf list plus a tree definition that can reconstruct the original containers. Treat that definition as part of the contract, rather than manually guessing where each flattened value belongs.

```text
params
├── weight: float32[2]
└── bias: float32[]
2 leaves, 3 numerical parameter values
```

## Derive the matching gradient tree

Let’s check one example. Take $x=[2,1]$, weights $w=[1,-1]$, and bias $b=0$. The prediction is $1$, while the target is $3$, so the residual is $-2$. For squared loss, the derivative with respect to the prediction is $-4$. Multiplying by the input gives the weight gradient $-4[2,1]=[-8,-4]$; the bias gradient is $-4$.

With learning rate $0.05$, the new weights are $[1.4,-0.8]$, and the bias becomes $0.2$. The new prediction is $2.2$, the residual is $-0.8$, and the loss is $0.64$. We can compare each of these values with the gradient tree and update. This tells us more than checking only whether the loss decreased.

$$
\begin{aligned}\hat y&=2(1)+1(-1)+0=1\\L&=(1-3)^2=4\\\nabla_w L&=(-8,-4),\qquad \frac{\partial L}{\partial b}=-4\end{aligned}
$$

## Apply corresponding leaves, not unrelated arrays

tree.map can apply $p-\eta g$ to corresponding leaves in the parameter and gradient trees. The containers must agree. A missing key is a structural error; an incompatible leaf shape is a numerical contract error. A matching tree structure alone does not prove that each leaf has the correct shape.

Dictionary leaves are traversed in a deterministic key order. Do not zip unrelated dictionaries merely because they have equal lengths. Use the tree definition to preserve identity and add shape checks at model boundaries.

## Keep metadata separate from differentiated state

Trainable leaves should be appropriate floating-point arrays. A Python string such as a model name is not a floating-point parameter. Keep such configuration separate, or register a carefully designed custom structure when the model requires it.

Optimizer state is another tree with its own meaning. Its structure need not be identical to parameters. Model parameters, gradients and optimizer statistics are connected, but they are not interchangeable objects. This separation will matter in the Optax lesson and in recovery.

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
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
x = jnp.array([2., 1.])
def loss(p):
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    return (prediction - 3.) ** 2
```

loss reads both parameter leaves. grad returns a matching gradient tree; tree.map pairs each parameter with its corresponding derivative for the update.

## Run and check the result

Append the checks, save main.py, and run python main.py from this folder using your course environment.

```python
value, grads = jax.value_and_grad(loss)(params)
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
print("Before/after:", float(value), float(loss(updated)))
assert grads.keys() == params.keys()
assert loss(updated) < value
```

Compare the output to the expected result below before making the exercise change.

## Run the example

```python
import jax
import jax.numpy as jnp
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
x = jnp.array([2., 1.])
def loss(p):
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    return (prediction - 3.) ** 2
value, grads = jax.value_and_grad(loss)(params)
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
print("Before/after:", float(value), float(loss(updated)))
assert grads.keys() == params.keys()
assert loss(updated) < value
```

Expected: Before/after: $4.0$, approximately $0.64$.

## A gradient tree matches parameter leaves

**Predict:** Which weight changes more, and why?

![A gradient tree matches parameter leaves](../../phases/03-state/02-pytrees-and-structured-parameters/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Each pair of bars compares one parameter before and after a gradient step. Read the categories as the two weight coordinates followed by the bias. The weights move from $(1,-1)$ to $(1.4,-0.8)$, and the bias moves from $0$ to $0.2$.

The second weight’s bar remains below zero, but becomes less negative. All three parameters increased; an upward update does not require the parameter itself to be positive.

### Connect it to the computation

For the input $(2,1)$, the initial prediction is $1$, below the target $3$. The parameter gradients are $(-8,-4,-4)$. Subtracting them with learning rate $0.05$ adds $(0.4,0.2,0.2)$, exactly the changes visible between the paired bars.

The updated prediction is $2.2$, so squared error falls from $4$ to $0.64$. The tree structure lets each gradient reach its corresponding parameter leaf. The chart’s vertical axis is parameter value, not loss; you need the prediction calculation to establish improvement.

```python
visual_data = {'kind': 'bar', 'labels': ['weight[0]', 'weight[1]', 'bias'], 'ylabel': 'parameter value', 'series': [{'label': 'before', 'y': [float(params['weight'][0]), float(params['weight'][1]), float(params['bias'])]}, {'label': 'after', 'y': [float(updated['weight'][0]), float(updated['weight'][1]), float(updated['bias'])]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T15:39:24.219111+00:00. JAX 0.9.2.

```text
Before/after: 4.0 0.6399999260902405
Before/after: 4.0 0.6399999260902405
Expected tree mismatch
PASS: state-02

```

## Verify every gradient leaf

**Predict before running:** Predict both weight derivatives and the bias derivative from the residual before inspecting grads.

```python
assert jnp.allclose(grads["weight"],jnp.array([-8.,-4.]))
assert jnp.allclose(grads["bias"],-4.)
assert jnp.allclose(updated["weight"],jnp.array([1.4,-0.8]))
assert jnp.allclose(updated["bias"],0.2)
assert jnp.allclose(loss(updated),0.64,atol=1e-6)
```

**Expected:** All leaves agree with the hand-derived update.

This check verifies parameter identity and the actual gradient values, not just a matching container.

## Flatten and reconstruct without losing identity

**Predict before running:** What is retained by the tree definition when array values are replaced?

```python
leaves,definition=jax.tree.flatten(params)
rebuilt=jax.tree.unflatten(definition,leaves)
assert jax.tree.structure(rebuilt)==jax.tree.structure(params)
assert jnp.array_equal(rebuilt["weight"],params["weight"])
assert jnp.array_equal(rebuilt["bias"],params["bias"])
```

**Expected:** The original keys and corresponding values are reconstructed.

The tree definition describes containers; leaves supply numerical values. Both are needed for reconstruction.

## Make it yours

Compute the squared norm of all gradient leaves. Verify it by adding the weight and bias contributions directly.

<details><summary>Reference solution</summary>

```python
norm_squared = sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
manual = jnp.sum(grads["weight"] ** 2) + grads["bias"] ** 2
assert jnp.allclose(norm_squared, manual)
assert jnp.allclose(norm_squared, 96.)
```

</details>

## Extend to a nested model

**Practice**

Place weight and bias under a layer dictionary and add a scalar scale parameter. Verify the derivative of scale for $s(wx+b)$.

<details><summary>Hint</summary>

At the starting point prediction is $1$ and residual is $-2$, so the scale derivative is $-4$.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
nested={"layer":params,"scale":jnp.array(1.)}
def nested_loss(p):
    pred=p["scale"]*(jnp.dot(p["layer"]["weight"],x)+p["layer"]["bias"])
    return (pred-3.)**2
g=jax.grad(nested_loss)(nested)
assert jax.tree.structure(g)==jax.tree.structure(nested)
assert jnp.allclose(g["scale"],-4.)
```

Nested parameter identity survives differentiation. The independent scale calculation checks the new path.

</details>

## Repair a mismatched update tree

**Challenge**

Remove bias from the gradient dictionary, reproduce the structural error, then restore it and verify the hand update.

<details><summary>Hint</summary>

Matching leaf counts is not enough; key paths matter.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
try:
    jax.tree.map(lambda p,g:p-0.05*g,params,{"weight":grads["weight"]})
except ValueError:
    print("Expected tree mismatch")
else:
    raise AssertionError("Expected a missing-key failure")
fixed=jax.tree.map(lambda p,g:p-0.05*g,params,grads)
assert jnp.allclose(loss(fixed),0.64,atol=1e-6)
```

A missing bias gradient is a tree mismatch. Broadcasting weights or changing a learning rate cannot supply the missing parameter identity.

</details>

## Check your understanding

What structure does `grad(loss)(params)` have?

1. One scalar regardless of parameters
2. A pytree corresponding to params
3. A list of Python source lines

<details><summary>Answer and explanation</summary>

A pytree corresponding to params

The differentiated input determines the gradient structure. Each floating-point parameter leaf gets a corresponding gradient leaf.

</details>

## Diagnose the result

A missing bias gradient is a tree mismatch. Broadcasting weights or changing a learning rate cannot supply the missing parameter identity.

## Carry forward

- This check verifies parameter identity and the actual gradient values, not just a matching container.
- The tree definition describes containers; leaves supply numerical values. Both are needed for reconstruction.

## Keep your evidence

Keep the parameter/gradient tree diagram, hand-derived leaves and update, flatten/unflatten round trip, nested-scale derivative and missing-key repair.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX pytrees](https://docs.jax.dev/en/latest/101/pytrees.html)

