"""Masked image modeling: reconstruct missing patches: worked experiments and reference solutions. CPU checks."""

# 1. Define patch layout and masked error
# Step 1 — 1. Define patch layout and masked error: Trace the transpose against the numbered top-left patch.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `patchify(images, patch)` implementing this stage's computation:
def patchify(images, patch=2):
    # Evaluate `(b, h, w, c)` from the current inputs and state.
    b, h, w, c = images.shape
    # Guard input contract (`h % patch or w % patch`) and fail fast if violated.
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    # Return `images.reshape(b, h // patch, patch, w // patch, patch, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h // patch * (w // patch), patch * patch * c)` to the caller.
    return (
        images.reshape(b, h // patch, patch, w // patch, patch, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(b, (h // patch) * (w // patch), patch * patch * c)
    )

# Function `masked_mse(prediction, target, hidden)` implementing this stage's computation:
def masked_mse(prediction, target, hidden):
    # Reduce along axis=-1 to compute `per_patch`.
    per_patch = jnp.mean((prediction - target) ** 2, axis=-1)
    # Return `jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)` to the caller.
    return jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# 2. Separate visible features from full targets
# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([0.1, 0.3, 0.6, 0.8])
# Construct and reshape `base_image` into the target tensor dimensions.
base_image = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1) / 32
# Evaluate `images` from the current inputs and state.
images = amplitude[:, None, None, None] + base_image[None]
# Run `patchify` to compute `patches`.
patches = patchify(images)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(patches[0, 0], np.asarray(images[0, :2, :2, :]).reshape(-1))
# Initialize array `hidden` with explicit values and shape.
hidden = jnp.array([[False, True, True, False]] * 4)
# Run `validate_mask` to perform the next check or state transition.
validate_mask(hidden, patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:, [0, 3], :]
# Construct and reshape `features` into the target tensor dimensions.
features = jnp.concatenate([visible.reshape(4, -1), jnp.ones((4, 1))], axis=1)
# Initialize array `w` with explicit values and shape.
w = jnp.zeros((9, 16))
# Evaluate `history` from the current inputs and state.
history = []
# Construct and reshape `loss` into the target tensor dimensions.
loss = lambda w: masked_mse((features @ w).reshape(4, 4, 4), patches, hidden)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))

# 3. Fit the decoder and audit the reconstruction
# Step 3 — 3. Fit the decoder and audit the reconstruction: The host loop checks the reduction independently.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `w` from the current inputs and state.
    w = w - 0.15 * g

# Construct and reshape `prediction` into the target tensor dimensions.
prediction = (features @ w).reshape(4, 4, 4)
# Verify contract: `history[-1] < history[0] * 0.02`.
assert history[-1] < history[0] * 0.02
# A scalar host loop is independent of the vectorized reduction.
expected = np.mean(
    [
        float(np.mean((np.asarray(prediction[b, k]) - np.asarray(patches[b, k])) ** 2))
        for b in range(4)
        for k in [1, 2]
    ]
)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(w), expected, rtol=1e-5)
# Evaluate `visible_changed` from the current inputs and state.
visible_changed = prediction.at[:, [0, 3], :].set(999.0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(
    masked_mse(visible_changed, patches, hidden), loss(w), atol=1e-7
)
# Initialize array `held_images` with explicit values and shape.
held_images = jnp.array([0.2, 0.5])[:, None, None, None] + base_image[None]
# Run `patchify` to compute `held_patches`.
held_patches = patchify(held_images)
# Construct and reshape `held_features` into the target tensor dimensions.
held_features = jnp.concatenate(
    [held_patches[:, [0, 3], :].reshape(2, -1), jnp.ones((2, 1))], axis=1
)
# Construct and reshape `held_loss` into the target tensor dimensions.
held_loss = float(
    masked_mse((held_features @ w).reshape(2, 4, 4), held_patches, hidden[:2])
)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert held_loss < 0.02
# Print diagnostic summary of the computed outputs.
print(
    'Masked pixel MSE initial/final/held amplitudes:',
    history[0],
    history[-1],
    held_loss,
)

# Step 1 — 1. Define patch layout and masked error: Trace the transpose against the numbered top-left patch.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `patchify(images, patch)` implementing this stage's computation:
def patchify(images, patch=2):
    # Evaluate `(b, h, w, c)` from the current inputs and state.
    b, h, w, c = images.shape
    # Guard input contract (`h % patch or w % patch`) and fail fast if violated.
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    # Return `images.reshape(b, h // patch, patch, w // patch, patch, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h // patch * (w // patch), patch * patch * c)` to the caller.
    return (
        images.reshape(b, h // patch, patch, w // patch, patch, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(b, (h // patch) * (w // patch), patch * patch * c)
    )

# Function `masked_mse(prediction, target, hidden)` implementing this stage's computation:
def masked_mse(prediction, target, hidden):
    # Reduce along axis=-1 to compute `per_patch`.
    per_patch = jnp.mean((prediction - target) ** 2, axis=-1)
    # Return `jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)` to the caller.
    return jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([0.1, 0.3, 0.6, 0.8])
# Construct and reshape `base_image` into the target tensor dimensions.
base_image = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1) / 32
# Evaluate `images` from the current inputs and state.
images = amplitude[:, None, None, None] + base_image[None]
# Run `patchify` to compute `patches`.
patches = patchify(images)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(patches[0, 0], np.asarray(images[0, :2, :2, :]).reshape(-1))
# Initialize array `hidden` with explicit values and shape.
hidden = jnp.array([[False, True, True, False]] * 4)
# Run `validate_mask` to perform the next check or state transition.
validate_mask(hidden, patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:, [0, 3], :]
# Construct and reshape `features` into the target tensor dimensions.
features = jnp.concatenate([visible.reshape(4, -1), jnp.ones((4, 1))], axis=1)
# Initialize array `w` with explicit values and shape.
w = jnp.zeros((9, 16))
# Evaluate `history` from the current inputs and state.
history = []
# Construct and reshape `loss` into the target tensor dimensions.
loss = lambda w: masked_mse((features @ w).reshape(4, 4, 4), patches, hidden)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))

# Step 3 — 3. Fit the decoder and audit the reconstruction: The host loop checks the reduction independently.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `w` from the current inputs and state.
    w = w - 0.15 * g

# Construct and reshape `prediction` into the target tensor dimensions.
prediction = (features @ w).reshape(4, 4, 4)
# Verify contract: `history[-1] < history[0] * 0.02`.
assert history[-1] < history[0] * 0.02
# A scalar host loop is independent of the vectorized reduction.
expected = np.mean(
    [
        float(np.mean((np.asarray(prediction[b, k]) - np.asarray(patches[b, k])) ** 2))
        for b in range(4)
        for k in [1, 2]
    ]
)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(w), expected, rtol=1e-5)
# Evaluate `visible_changed` from the current inputs and state.
visible_changed = prediction.at[:, [0, 3], :].set(999.0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(
    masked_mse(visible_changed, patches, hidden), loss(w), atol=1e-7
)
# Initialize array `held_images` with explicit values and shape.
held_images = jnp.array([0.2, 0.5])[:, None, None, None] + base_image[None]
# Run `patchify` to compute `held_patches`.
held_patches = patchify(held_images)
# Construct and reshape `held_features` into the target tensor dimensions.
held_features = jnp.concatenate(
    [held_patches[:, [0, 3], :].reshape(2, -1), jnp.ones((2, 1))], axis=1
)
# Construct and reshape `held_loss` into the target tensor dimensions.
held_loss = float(
    masked_mse((held_features @ w).reshape(2, 4, 4), held_patches, hidden[:2])
)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert held_loss < 0.02
# Print diagnostic summary of the computed outputs.
print(
    'Masked pixel MSE initial/final/held amplitudes:',
    history[0],
    history[-1],
    held_loss,
)

# Figure data experiment
# Compute figure data for: Masked image modeling: reconstruct missing patches — recorded experiment
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'hidden-pixel MSE (squared input units)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Evaluate `panel['x']` from the current inputs and state.
    panel['x']=panel['series'][0]['x']

# Convert `patch_comparison` to a host NumPy array for inspection or verification.
patch_comparison=np.concatenate([np.asarray(patches[0,[1,2],:]),np.asarray(prediction[0,[1,2],:])],axis=0)
# Evaluate `extra_panel` from the current inputs and state.
extra_panel={'kind':'heatmap','values':patch_comparison.tolist(),'rows':['target patch 1','target patch 2','predicted patch 1','predicted patch 2'],'columns':['pixel 0','pixel 1','pixel 2','pixel 3'],'unit':'raw pixel value','xlabel':'within-patch pixel order','ylabel':'hidden patch and source','title':'Hidden targets and final predictions for the first image'}
# Evaluate `visual_data` from the current inputs and state.
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Make visible-output errors arbitrarily large
# Experiment — Make visible-output errors arbitrarily large: The disagreement identifies the reduction contract.
bad_visible = prediction.at[:, [0, 3], :].set(-1000.0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(
    masked_mse(bad_visible, patches, hidden),
    masked_mse(prediction, patches, hidden),
)
# Verify contract: `float(jnp.mean((bad_visible - patches) ** 2)) > 1000`.
assert float(jnp.mean((bad_visible - patches) ** 2)) > 1000
# Print the observed values to compare against the expected result.
print('Hidden MSE unchanged; full-image MSE is large.')

# Experiment: Make hidden information impossible to recover
# Experiment — Make hidden information impossible to recover: The experiment isolates missing information.
# Initialize array `ambiguous_targets` with explicit values and shape.
ambiguous_targets = jnp.array([[[0.0]], [[2.0]]])
# Initialize array `ambiguous_mask` with explicit values and shape.
ambiguous_mask = jnp.ones((2, 1), dtype=bool)

# Function `ambiguity_loss(value)` implementing this stage's computation:
def ambiguity_loss(value):
    # Return `masked_mse(jnp.full_like(ambiguous_targets, value), ambiguous_targets, ambiguous_mask)` to the caller.
    return masked_mse(
        jnp.full_like(ambiguous_targets, value), ambiguous_targets, ambiguous_mask
    )

# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(
    [ambiguity_loss(0.0), ambiguity_loss(1.0), ambiguity_loss(2.0)], [2.0, 1.0, 2.0]
)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(ambiguity_loss)(1.0), 0.0, atol=1e-7)
# Print the observed values to compare against the expected result.
print('Ambiguous target MSE at predictions 0, 1, 2: 2, 1, 2')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Verify the bottom-right patch against a direct slice and explain why a...
np.testing.assert_array_equal(
    patches[2, 3], np.asarray(images[2, 2:, 2:, :]).reshape(-1)
)
# Print the observed values to compare against the expected result.
print('Bottom-right patch order verified.')

# Reference practice: Expose the leakage shortcut
# Expose the leakage shortcut (Transfer): An information-flow check is necessary because the leaked...
shortcut = patches
# Verify contract: `float(masked_mse(shortcut, patches, hidden)) == 0.0`.
assert float(masked_mse(shortcut, patches, hidden)) == 0.0
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert features.shape[1] == 9
# Print the observed values to compare against the expected result.
print('The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.')

# Reference practice: Audit every patch of a rectangular color image
# Audit every patch of a rectangular color image (Challenge): The rectangular multichannel case catches axis errors that a...
# Construct and reshape `numbered` into the target tensor dimensions.
numbered = np.arange(2 * 4 * 6 * 3, dtype=np.float32).reshape(2, 4, 6, 3)
# Construct and reshape `oracle` into the target tensor dimensions.
oracle = np.stack(
    [
        np.stack(
            [
                image[row : row + 2, col : col + 2, :].reshape(-1)
                for row in range(0, 4, 2)
                for col in range(0, 6, 2)
            ]
        )
        for image in numbered
    ]
)
# Create device-backed JAX array ``.
np.testing.assert_array_equal(patchify(jnp.asarray(numbered)), oracle)
# Evaluate `changed_images` from the current inputs and state.
changed_images = images.at[:, :2, 2:, :].add(17.0).at[:, 2:, :2, :].add(-9.0)
# Run `patchify` to compute `changed_patches`.
changed_patches = patchify(changed_images)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(changed_patches[:, [0, 3], :], visible)
# Verify contract: `not np.array_equal(np.asarray(changed_patches[:, [1, 2], :]), np.asa...`.
assert not np.array_equal(
    np.asarray(changed_patches[:, [1, 2], :]), np.asarray(patches[:, [1, 2], :])
)
# Print the observed values to compare against the expected result.
print('All rectangular RGB patches agree; hidden-pixel intervention preserves visible features.')
print("PASS: pretraining-02")
