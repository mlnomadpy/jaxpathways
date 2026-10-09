"""Debug unstable learning: worked experiments and reference solutions. CPU checks."""

# 1. Define the controlled objective
# Step 1 — 1. Define the controlled objective: The initial loss is 56/3 and the initial gradient is -56/3.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
# Construct `x` via `jnp.array([1.,2.,3.])`
x=jnp.array([1.,2.,3.])
y=2*x
# Function `loss(w, features, targets)` implementing this stage's computation:
# Return `jnp.mean((w * features - targets) ** 2)` to the caller.
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
# Function `analytic_gradient(w)` implementing this stage's computation:
# Return `28 / 3 * (w - 2)` to the caller.
def analytic_gradient(w):return (28/3)*(w-2)
# Compute `np.testing.assert_allclose(loss(0.),56/3,rtol` as `1e-6)`.
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

# 2. Instrument the update loop
# Step 2 — 2. Instrument the update loop: The rate changes while initial state, data, and objective stay fixed.
# Define `run(rate, steps, features, targets...)` to evaluate the objective and its automatic derivatives:
def run(rate,steps,features=x,targets=y,clip=None):
    # Construct `w` via `jnp.array(0.)`
    w=jnp.array(0.)
    history=[]
    # Compute `objective` from `lambda value:loss(value,features,targets)`
    objective=lambda value:loss(value,features,targets)
    # Repeat the update loop over `range(steps)` steps:
    for _ in range(steps):
        # Differentiate the objective to obtain `(value, grad)` via automatic differentiation.
        value,grad=jax.value_and_grad(objective)(w)
        # Append the current step result to `history`.
        history.append([float(w),float(value),float(jnp.abs(grad))])
        # Confirm that all computed values remain finite (no NaN or Inf).
        assert np.all(np.isfinite(history[-1]))
        # Combine or mask array elements to form `update`.
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        # Compute `w` from `w-rate*update`
        w=w-rate*update
    # Return `(float(w), np.asarray(history))` to the caller.
    return float(w),np.asarray(history)
# Run `run` to compute `(stable, good)`.
stable,good=run(.1,6)
# Run `run` to compute `(unstable, bad)`.
unstable,bad=run(.3,6)
# Assert that `abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100`.
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

# 3. Match the recorded failure to the exact recurrence
# Step 3 — 3. Match the recorded failure to the exact recurrence: An alternating parameter error and a 3.24 loss multiplier support...
# Iterate over `(rate, history)` to step through the computation:
for rate,history in [(.1,good),(.3,bad)]:
    # Compute `expected_w` from `2-2*(1-rate*28/3)**np.arange(len(history))`
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    # Compute `expected_loss` from `(14/3)*(expected_w-2)**2`
    expected_loss=(14/3)*(expected_w-2)**2
    # Compute `np.testing.assert_allclose(history[:,0],expected_w,rtol` as `2e-5,atol=1e-5)`.
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    # Compute `np.testing.assert_allclose(history[:,1],expected_loss,rtol` as `2e-5,atol=1e-8)`.
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
# Compute `np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol` as `1e-5)`.
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
# Print the observed values to compare against the expected result.
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

# Step 1 — 1. Define the controlled objective: The initial loss is 56/3 and the initial gradient is -56/3.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
# Construct `x` via `jnp.array([1.,2.,3.])`
x=jnp.array([1.,2.,3.])
y=2*x
# Function `loss(w, features, targets)` implementing this stage's computation:
# Return `jnp.mean((w * features - targets) ** 2)` to the caller.
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
# Function `analytic_gradient(w)` implementing this stage's computation:
# Return `28 / 3 * (w - 2)` to the caller.
def analytic_gradient(w):return (28/3)*(w-2)
# Compute `np.testing.assert_allclose(loss(0.),56/3,rtol` as `1e-6)`.
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

# Step 2 — 2. Instrument the update loop: The rate changes while initial state, data, and objective stay fixed.
# Define `run(rate, steps, features, targets...)` to evaluate the objective and its automatic derivatives:
def run(rate,steps,features=x,targets=y,clip=None):
    # Construct `w` via `jnp.array(0.)`
    w=jnp.array(0.)
    history=[]
    # Compute `objective` from `lambda value:loss(value,features,targets)`
    objective=lambda value:loss(value,features,targets)
    # Repeat the update loop over `range(steps)` steps:
    for _ in range(steps):
        # Differentiate the objective to obtain `(value, grad)` via automatic differentiation.
        value,grad=jax.value_and_grad(objective)(w)
        # Append the current step result to `history`.
        history.append([float(w),float(value),float(jnp.abs(grad))])
        # Confirm that all computed values remain finite (no NaN or Inf).
        assert np.all(np.isfinite(history[-1]))
        # Combine or mask array elements to form `update`.
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        # Compute `w` from `w-rate*update`
        w=w-rate*update
    # Return `(float(w), np.asarray(history))` to the caller.
    return float(w),np.asarray(history)
# Run `run` to compute `(stable, good)`.
stable,good=run(.1,6)
# Run `run` to compute `(unstable, bad)`.
unstable,bad=run(.3,6)
# Assert that `abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100`.
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

# Step 3 — 3. Match the recorded failure to the exact recurrence: An alternating parameter error and a 3.24 loss multiplier support...
# Iterate over `(rate, history)` to step through the computation:
for rate,history in [(.1,good),(.3,bad)]:
    # Compute `expected_w` from `2-2*(1-rate*28/3)**np.arange(len(history))`
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    # Compute `expected_loss` from `(14/3)*(expected_w-2)**2`
    expected_loss=(14/3)*(expected_w-2)**2
    # Compute `np.testing.assert_allclose(history[:,0],expected_w,rtol` as `2e-5,atol=1e-5)`.
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    # Compute `np.testing.assert_allclose(history[:,1],expected_loss,rtol` as `2e-5,atol=1e-8)`.
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
# Compute `np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol` as `1e-5)`.
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
# Print the observed values to compare against the expected result.
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

# Figure data experiment
# Compute figure data for: Correct gradients can still produce unstable training
# Compute `visual_data` from `{'kind': 'line', 'x': list(range(len(good))), 'xlabe...`
visual_data = {'kind': 'line', 'x': list(range(len(good))), 'xlabel': 'pre-update step', 'ylabel': 'mean squared loss', 'yscale': 'log', 'series': [{'label': 'rate 0.1', 'y': good[:, 1].tolist()}, {'label': 'rate 0.3', 'y': bad[:, 1].tolist()}]}

# Experiment: Bounded clipped updates
# Experiment — Bounded clipped updates: Clipping changes the effective update.
clipped,clipped_history=run(.3,6,clip=1.)
# Assert that `np.all(np.abs(np.diff(clipped_history[:,0]))<=.300001)`.
assert np.all(np.abs(np.diff(clipped_history[:,0]))<=.300001)
# Assert invariant `clipped_history[-1,1]<clipped_history[0,1]` holds
assert clipped_history[-1,1]<clipped_history[0,1]
# Assert invariant `clipped_history[0,2]>1.` holds
assert clipped_history[0,2]>1.

# Experiment: Feature scaling changes curvature
# Experiment — Feature scaling changes curvature: The gradient multiplies by 100 because both prediction and...
scaled_final,scaled=run(.1,3,x*10,y*10)
# Assert invariant `scaled[-1,1]>scaled[0,1]` holds
assert scaled[-1,1]>scaled[0,1]
# Run `run` to compute `(adjusted_final, adjusted)`.
adjusted_final,adjusted=run(.001,6,x*10,y*10)
# Compute `np.testing.assert_allclose(adjusted[:,0],good[:,0],rtol` as `1e-5,atol=1e-5)`.
np.testing.assert_allclose(adjusted[:,0],good[:,0],rtol=1e-5,atol=1e-5)
# Compute `np.testing.assert_allclose(adjusted[:,1],good[:,1]*100,rtol` as `1e-4,atol=1e-8)`.
np.testing.assert_allclose(adjusted[:,1],good[:,1]*100,rtol=1e-4,atol=1e-8)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Use features [1, 1, 1] and targets 2.
# Construct `ones` via `jnp.ones(3)`
ones=jnp.ones(3)
# Run `run` to compute `(exact, history)`.
exact,history=run(.5,3,ones,2*ones)
# Run `run` to compute `(divergent, divergent_history)`.
divergent,divergent_history=run(1.1,6,ones,2*ones)
# Compute `np.testing.assert_allclose(exact,2.,atol` as `1e-6)`.
np.testing.assert_allclose(exact,2.,atol=1e-6)
# Assert invariant `divergent_history[-1,1]>divergent_history[0,1]` holds
assert divergent_history[-1,1]>divergent_history[0,1]
# Compute `np.testing.assert_allclose(divergent_history[1:,1]/divergent_history[:-1,1],1.44,rtol` as `1e-5)`.
np.testing.assert_allclose(divergent_history[1:,1]/divergent_history[:-1,1],1.44,rtol=1e-5)

# Reference practice: Rescale the objective and compensate the update
# Rescale the objective and compensate the update (Transfer): Scaling the objective scales the gradient.
# Construct `w0` via `jnp.array(.25)`
w0=jnp.array(.25)
# Differentiate the objective to obtain `base_gradient` via automatic differentiation.
base_gradient=jax.grad(loss)(w0)
# Differentiate the objective to obtain `scaled_gradient` via automatic differentiation.
scaled_gradient=jax.grad(lambda w:5*loss(w))(w0)
# Compute `np.testing.assert_allclose(scaled_gradient,5*base_gradient,rtol` as `1e-6)`.
np.testing.assert_allclose(scaled_gradient,5*base_gradient,rtol=1e-6)
# Compute `reference` from `w0-.1*base_gradient`
reference=w0-.1*base_gradient
# Compute `compensated` from `w0-(.1/5)*scaled_gradient`
compensated=w0-(.1/5)*scaled_gradient
# Compute `uncompensated` from `w0-.1*scaled_gradient`
uncompensated=w0-.1*scaled_gradient
# Compute `np.testing.assert_allclose(compensated,reference,rtol` as `1e-6)`.
np.testing.assert_allclose(compensated,reference,rtol=1e-6)
# Assert that `not np.isclose(uncompensated,reference)`.
assert not np.isclose(uncompensated,reference)
# Print the observed values to compare against the expected result.
print('Reference / compensated / unchanged-rate updates:',reference,compensated,uncompensated)

# Reference practice: Diagnose a nonfinite input before an update
# Diagnose a nonfinite input before an update (Intermediate): Nonfinite data is a different cause than excessive step size.
corrupted=x.at[1].set(jnp.nan)
# Differentiate the objective to obtain `(bad_value, bad_grad)` via automatic differentiation.
bad_value,bad_grad=jax.value_and_grad(lambda w:loss(w,corrupted,y))(0.)
# Confirm that all computed values remain finite (no NaN or Inf).
assert not np.isfinite(float(bad_value)) and not np.isfinite(float(bad_grad))
# Construct `repaired` via `jnp.array([1.,2.,3.])`
repaired=jnp.array([1.,2.,3.])
# Differentiate the objective to obtain `(repaired_value, repaired_grad)` via automatic differentiation.
repaired_value,repaired_grad=jax.value_and_grad(lambda w:loss(w,repaired,y))(0.)
# Compute `np.testing.assert_allclose(repaired_grad,-56/3,rtol` as `1e-6)`.
np.testing.assert_allclose(repaired_grad,-56/3,rtol=1e-6)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(float(repaired_value))
print("PASS: networks-05")
