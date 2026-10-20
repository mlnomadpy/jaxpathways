"""Gradient checking and numerical accuracy: worked experiments and reference solutions. CPU checks."""

# 1. Prepare the inputs
# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
import jax
import jax.numpy as jnp

# 2. Define the computation
# Step 2 — 2. Define the computation: Each row of basis perturbs one coordinate.
def objective(w):
    return jnp.sum((w - jnp.array([2., -1.])) ** 2)
w = jnp.array([0.5, 0.25], dtype=jnp.float32)
h = 1e-2
basis = jnp.eye(2)
# Vectorize across the batch dimension with `jax.vmap` (`finite`).
finite = jax.vmap(lambda e: (objective(w+h*e)-objective(w-h*e))/(2*h))(basis)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic = jax.grad(objective)(w)
analytic = 2 * (w - jnp.array([2., -1.]))

# 3. Measure and verify
# Step 3 — 3. Measure and verify: Both gradients are approximately [-3., 2.5].
print("Finite difference:", finite)
print("Autodiff:", automatic)
assert jnp.allclose(automatic, analytic)
assert jnp.allclose(finite, automatic, atol=2e-4, rtol=2e-4)

# Step 1 — 1. Prepare the inputs: This block establishes the values used by the following steps; run...
import jax
import jax.numpy as jnp
# Step 2 — 2. Define the computation: Each row of basis perturbs one coordinate.
def objective(w):
    return jnp.sum((w - jnp.array([2., -1.])) ** 2)
w = jnp.array([0.5, 0.25], dtype=jnp.float32)
h = 1e-2
basis = jnp.eye(2)
# Vectorize across the batch dimension with `jax.vmap` (`finite`).
finite = jax.vmap(lambda e: (objective(w+h*e)-objective(w-h*e))/(2*h))(basis)
# Differentiate the objective to obtain `automatic` via automatic differentiation.
automatic = jax.grad(objective)(w)
analytic = 2 * (w - jnp.array([2., -1.]))
# Step 3 — 3. Measure and verify: Both gradients are approximately [-3., 2.5].
print("Finite difference:", finite)
print("Autodiff:", automatic)
assert jnp.allclose(automatic, analytic)
assert jnp.allclose(finite, automatic, atol=2e-4, rtol=2e-4)

# Figure data experiment
steps = jnp.logspace(-8.0, -1.0, 22)
errors = []
for h_plot in steps:
    # Vectorize across the batch dimension without a Python loop (`estimate`).
    estimate = jax.vmap(lambda e: (objective(w + h_plot * e) - objective(w - h_plot * e)) / (2 * h_plot))(basis)
    errors.append(float(jnp.max(jnp.abs(estimate - analytic))))
visual_data = {'kind': 'line', 'x': steps.tolist(), 'xscale': 'log', 'xlabel': 'finite-difference step', 'ylabel': 'maximum absolute gradient error', 'series': [{'label': 'float32 central difference', 'y': errors}]}
visual_data['yscale'] = 'log'

# Experiment: Sweep the perturbation size
# Experiment — Sweep the perturbation size: Nearby float32 inputs can coincide, so reducing the step does...
for step_size in [1e-1, 1e-2, 1e-4, 1e-8]:
    # Vectorize across the batch dimension with `jax.vmap` (`estimate`).
    estimate = jax.vmap(lambda e: (objective(w+step_size*e)-objective(w-step_size*e))/(2*step_size))(basis)
    print("h / max error:", step_size, float(jnp.max(jnp.abs(estimate-analytic))))
# Vectorize across the batch dimension with `jax.vmap` (`tiny`).
tiny = jax.vmap(lambda e: (objective(w+1e-8*e)-objective(w-1e-8*e))/(2e-8))(basis)
assert not jnp.allclose(tiny, analytic, atol=1e-3)

# Experiment: Check a new direction
# Experiment — Check a new direction: A new projection supplies a separate numerical check, though it...
direction = jnp.array([1., -2.])
directional = (objective(w+h*direction)-objective(w-h*direction))/(2*h)
assert jnp.allclose(directional, -8., atol=2e-4)
assert jnp.allclose(jnp.dot(automatic, direction), -8.)

# Experiment: A symmetric probe can hide a kink
# Experiment — A symmetric probe can hide a kink: Agreement with one numerical probe can conceal nondifferentiability.
h_kink = 0.01
symmetric = (jnp.abs(h_kink)-jnp.abs(-h_kink))/(2*h_kink)
left = (jnp.abs(0.)-jnp.abs(-h_kink))/h_kink
right = (jnp.abs(h_kink)-jnp.abs(0.))/h_kink
assert jnp.allclose(symmetric,0.)
assert jnp.allclose(left,-1.) and jnp.allclose(right,1.)
assert jnp.allclose(jax.grad(jnp.abs)(jnp.array(-2.)),-1.)
assert jnp.allclose(jax.grad(jnp.abs)(jnp.array(2.)),1.)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Check the gradient again at w=[2., -1.].
minimum = jnp.array([2., -1.])
assert jnp.allclose(jax.grad(objective)(minimum), jnp.zeros(2))
assert jnp.allclose(objective(minimum), 0.)

# Reference practice: Move to a different smooth objective
# Move to a different smooth objective (Transfer / diagnosis): The analytic reference checks a nonquadratic function at a...
z = jnp.array([0.2, -0.7])
def trig_loss(z):
    return jnp.sum(jnp.sin(z))
# Vectorize across the batch dimension with `jax.vmap` (`fd`).
fd = jax.vmap(lambda e: (trig_loss(z+1e-2*e)-trig_loss(z-1e-2*e))/(2e-2))(jnp.eye(2))
assert jnp.allclose(jax.grad(trig_loss)(z), jnp.cos(z))
assert jnp.allclose(fd, jnp.cos(z), atol=5e-5, rtol=5e-5)

# Reference practice: Catch the wrong objective despite correct gradients
# Catch the wrong objective despite correct gradients (Transfer / diagnosis): The omitted term is a specification error; numerical...
def incomplete(z):
    return (z[0]-2.)**2
# Differentiate the objective to obtain `bad` via automatic differentiation.
bad = jax.grad(incomplete)(w)
assert jnp.allclose(bad, jnp.array([-3., 0.]))
assert not jnp.allclose(bad, analytic)
assert jnp.allclose(jax.grad(objective)(w), analytic)
print("PASS: optimization-02")
