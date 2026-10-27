"""Masked image modeling: reconstruct missing patches: worked experiments and reference solutions. CPU checks."""

# 1. Define patch layout and masked error
# Step 1 — 1. Define patch layout and masked error: Trace the transpose against the numbered top-left patch.
import jax
import jax.numpy as jnp
import numpy as np

def patchify(images, patch=2):
    b, h, w, c = images.shape
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    return (
        images.reshape(b, h // patch, patch, w // patch, patch, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(b, (h // patch) * (w // patch), patch * patch * c)
    )

def masked_mse(prediction, target, hidden):
    # Reduce along axis=-1 to compute `per_patch`.
    per_patch = jnp.mean((prediction - target) ** 2, axis=-1)
    return jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)

def validate_mask(selected, shape):
    # Transfer the device result to the host for an independent NumPy comparison.
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# 2. Separate visible features from full targets
# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([0.1, 0.3, 0.6, 0.8])
base_image = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1) / 32
images = amplitude[:, None, None, None] + base_image[None]
patches = patchify(images)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(patches[0, 0], np.asarray(images[0, :2, :2, :]).reshape(-1))
hidden = jnp.array([[False, True, True, False]] * 4)
validate_mask(hidden, patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:, [0, 3], :]
features = jnp.concatenate([visible.reshape(4, -1), jnp.ones((4, 1))], axis=1)
w = jnp.zeros((9, 16))
history = []
loss = lambda w: masked_mse((features @ w).reshape(4, 4, 4), patches, hidden)
step = jax.jit(jax.value_and_grad(loss))

# 3. Fit the decoder and audit the reconstruction
# Step 3 — 3. Fit the decoder and audit the reconstruction: The host loop checks the reduction independently.
for _ in range(100):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.15 * g

prediction = (features @ w).reshape(4, 4, 4)
assert history[-1] < history[0] * 0.02
# A scalar host loop is independent of the vectorized reduction.
expected = np.mean(
    [
        float(np.mean((np.asarray(prediction[b, k]) - np.asarray(patches[b, k])) ** 2))
        for b in range(4)
        for k in [1, 2]
    ]
)
np.testing.assert_allclose(loss(w), expected, rtol=1e-5)
visible_changed = prediction.at[:, [0, 3], :].set(999.0)
np.testing.assert_allclose(
    masked_mse(visible_changed, patches, hidden), loss(w), atol=1e-7
)
held_images = jnp.array([0.2, 0.5])[:, None, None, None] + base_image[None]
held_patches = patchify(held_images)
held_features = jnp.concatenate(
    [held_patches[:, [0, 3], :].reshape(2, -1), jnp.ones((2, 1))], axis=1
)
held_loss = float(
    masked_mse((held_features @ w).reshape(2, 4, 4), held_patches, hidden[:2])
)
assert held_loss < 0.02
print(
    'Masked pixel MSE initial/final/held amplitudes:',
    history[0],
    history[-1],
    held_loss,
)

# 4. Save the encoder, change one condition, and probe it
# Changed mask schedule + learned saved encoder + frozen-probe transfer (synthetic)
import hashlib, json, tempfile
from pathlib import Path

# Eight source-group images vary amplitude; class labels are reserved for the probe.
transfer_amp = jnp.array([0.10,0.60,0.20,0.65,0.35,0.80,0.40,0.90])
transfer_images = transfer_amp[:,None,None,None] + base_image[None]
transfer_patches = patchify(transfer_images)
train_ids, held_ids = ['g0','g1','g2','g3'], ['g4','g5','g6','g7']
train_idx, held_idx = np.arange(4), np.arange(4,8)
assert set(train_ids).isdisjoint(held_ids)
# Each fit image gets one of two masks; across epochs the mask pattern alternates.
mask_a = jnp.array([False,True,True,False])
mask_b = jnp.array([True,False,False,True])
mask_schedule = [mask_a, mask_b]
key_e, key_d = jax.random.split(jax.random.key(19))
enc = jax.random.normal(key_e,(8,4))*0.1
# Decoder sees latent plus target patch position, never hidden pixels.
dec = jax.random.normal(key_d,(8,16))*0.1

def mae_loss_for(image_patches, enc, dec, mask):
    visible = (~mask).astype(jnp.float32)
    batch = len(image_patches)
    pooled = jnp.sum(image_patches * visible[None,:,None],axis=1) / visible.sum()
    encoder_input = jnp.concatenate([pooled, jnp.broadcast_to(visible,(batch,4))],axis=1)
    z = jnp.tanh(encoder_input @ enc)
    positions = jnp.eye(4)
    # Position-specific decoder: latent features plus one-hot patch index -> 4 pixels.
    pos_features = jnp.concatenate([jnp.broadcast_to(z[:,None,:],(batch,4,4)),
                                    jnp.broadcast_to(positions[None,:,:],(batch,4,4))],axis=-1)
    pred = pos_features @ dec
    per_patch = jnp.mean((pred-image_patches)**2,axis=-1)
    return jnp.sum(jnp.where(mask[None,:],per_patch,0.0))/ (batch*mask.sum())

def mae_loss(enc, dec, mask):
    return mae_loss_for(transfer_patches[train_idx],enc,dec,mask)

# [4 latent + 4 positional] x 4 target pixels.
dec = jax.random.normal(key_d,(8,4))*0.1
for epoch in range(120):
    mask = mask_schedule[epoch % 2]
    value, (ge,gd) = jax.value_and_grad(mae_loss,argnums=(0,1))(enc,dec,mask)
    enc, dec = enc - 0.08*ge, dec - 0.08*gd

# Recompute both masks without updating and compare with a visible-mean decoder baseline.
changed_mask_losses=[]
held_visible_baselines=[]
for mask in mask_schedule:
    changed_mask_losses.append(float(mae_loss_for(transfer_patches[held_idx],enc,dec,mask)))
    transfer_visible_mask=(~mask).astype(jnp.float32)
    transfer_visible_mean=jnp.sum(transfer_patches[held_idx]*transfer_visible_mask[None,:,None],axis=1)/transfer_visible_mask.sum()
    baseline=jnp.broadcast_to(transfer_visible_mean[:,None,:],transfer_patches[held_idx].shape)
    held_visible_baselines.append(float(masked_mse(baseline,transfer_patches[held_idx],jnp.broadcast_to(mask,(len(held_idx),4)))))
assert all(np.isfinite(changed_mask_losses))

def frozen_encode(images_patches, mask):
    visible=(~mask).astype(jnp.float32)
    pooled=jnp.sum(images_patches*visible[None,:,None],axis=1)/visible.sum()
    x=jnp.concatenate([pooled,jnp.broadcast_to(visible,(len(images_patches),4))],axis=1)
    return jnp.tanh(x @ enc)

# Save/reload exact encoder and decoder identity, then freeze encoder for probe fitting.
processor_id='gray-4x4-range-raw-patchify-2-v1'
source_split_id='synthetic-amplitude-v2:g0-g3/g4-g7'
parameter_hash=hashlib.sha256(np.ascontiguousarray(enc).tobytes()+np.ascontiguousarray(dec).tobytes()).hexdigest()
manifest={'model_id':'tiny-mask-aware-patch-encoder-v1','processor_id':processor_id,
          'source_split_id':source_split_id,'objective':'hidden-patch-MSE',
          'mask_schedule':'alternate center-pair and corner-pair; 50% hidden',
          'fit_group_ids':train_ids,'heldout_group_ids':held_ids,'seed':19,'parameter_sha256':parameter_hash}
output_dir = Path(tempfile.mkdtemp(prefix='jaxpathways-mim-encoder-'))
artifact = output_dir / 'encoder.npz'
np.savez(artifact,encoder=np.asarray(enc),decoder=np.asarray(dec),
         manifest=np.array(json.dumps(manifest,sort_keys=True)))
with np.load(artifact,allow_pickle=False) as saved:
    loaded_enc=saved['encoder'].copy(); loaded_dec=saved['decoder'].copy()
    loaded_manifest=json.loads(str(saved['manifest']))
rehash=hashlib.sha256(np.ascontiguousarray(loaded_enc).tobytes()+np.ascontiguousarray(loaded_dec).tobytes()).hexdigest()
assert loaded_manifest==manifest and rehash==parameter_hash
checkpoint_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
identity_path=output_dir/'identity.json'
identity={**manifest,'checkpoint_sha256':checkpoint_hash}
identity_path.write_text(json.dumps(identity,sort_keys=True))
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==identity['checkpoint_sha256']
frozen_before=loaded_enc.copy()
z=np.asarray(frozen_encode(transfer_patches,mask_a))
labels=(np.asarray(transfer_amp)>=0.5).astype(int)
design=np.c_[z[train_idx],np.ones(len(train_idx))]
probe=np.linalg.solve(design.T@design+1e-3*np.eye(design.shape[1]),design.T@labels[train_idx])
pred=(np.c_[z[held_idx],np.ones(len(held_idx))]@probe>=0.5).astype(int)
frozen_probe_accuracy=float(np.mean(pred==labels[held_idx]))
np.testing.assert_array_equal(loaded_enc,frozen_before)
print('MIM held-out alternating-mask losses / visible-mean baseline / frozen held-out probe accuracy:',
      changed_mask_losses, held_visible_baselines, frozen_probe_accuracy)
print('Artifact path:', artifact)
print('Identity sidecar:', identity_path)
print('Saved identity:',manifest['model_id'],manifest['processor_id'],manifest['source_split_id'])

# Step 1 — 1. Define patch layout and masked error: Trace the transpose against the numbered top-left patch.
import jax
import jax.numpy as jnp
import numpy as np

def patchify(images, patch=2):
    b, h, w, c = images.shape
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    return (
        images.reshape(b, h // patch, patch, w // patch, patch, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(b, (h // patch) * (w // patch), patch * patch * c)
    )

def masked_mse(prediction, target, hidden):
    # Average pixelwise squared error within each patch to get one loss per patch.
    per_patch = jnp.mean((prediction - target) ** 2, axis=-1)
    return jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)

def validate_mask(selected, shape):
    # Transfer the device result to the host for an independent NumPy comparison.
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([0.1, 0.3, 0.6, 0.8])
base_image = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1) / 32
images = amplitude[:, None, None, None] + base_image[None]
patches = patchify(images)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_array_equal(patches[0, 0], np.asarray(images[0, :2, :2, :]).reshape(-1))
hidden = jnp.array([[False, True, True, False]] * 4)
validate_mask(hidden, patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:, [0, 3], :]
features = jnp.concatenate([visible.reshape(4, -1), jnp.ones((4, 1))], axis=1)
w = jnp.zeros((9, 16))
history = []
loss = lambda w: masked_mse((features @ w).reshape(4, 4, 4), patches, hidden)
step = jax.jit(jax.value_and_grad(loss))

# Step 3 — 3. Fit the decoder and audit the reconstruction: The host loop checks the reduction independently.
for _ in range(100):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.15 * g

prediction = (features @ w).reshape(4, 4, 4)
assert history[-1] < history[0] * 0.02
# A scalar host loop is independent of the vectorized reduction.
expected = np.mean(
    [
        float(np.mean((np.asarray(prediction[b, k]) - np.asarray(patches[b, k])) ** 2))
        for b in range(4)
        for k in [1, 2]
    ]
)
np.testing.assert_allclose(loss(w), expected, rtol=1e-5)
visible_changed = prediction.at[:, [0, 3], :].set(999.0)
np.testing.assert_allclose(
    masked_mse(visible_changed, patches, hidden), loss(w), atol=1e-7
)
held_images = jnp.array([0.2, 0.5])[:, None, None, None] + base_image[None]
held_patches = patchify(held_images)
held_features = jnp.concatenate(
    [held_patches[:, [0, 3], :].reshape(2, -1), jnp.ones((2, 1))], axis=1
)
held_loss = float(
    masked_mse((held_features @ w).reshape(2, 4, 4), held_patches, hidden[:2])
)
assert held_loss < 0.02
print(
    'Masked pixel MSE initial/final/held amplitudes:',
    history[0],
    history[-1],
    held_loss,
)

# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'hidden-pixel MSE (squared input units)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

# Transfer the device result to the host for an independent NumPy comparison.
patch_comparison=np.concatenate([np.asarray(patches[0,[1,2],:]),np.asarray(prediction[0,[1,2],:])],axis=0)
extra_panel={'kind':'heatmap','values':patch_comparison.tolist(),'rows':['target patch 1','target patch 2','predicted patch 1','predicted patch 2'],'columns':['pixel 0','pixel 1','pixel 2','pixel 3'],'unit':'raw pixel value','xlabel':'within-patch pixel order','ylabel':'hidden patch and source','title':'Hidden targets and final predictions for the first image'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Make visible-output errors arbitrarily large
# Experiment — Make visible-output errors arbitrarily large: The disagreement identifies the reduction contract.
bad_visible = prediction.at[:, [0, 3], :].set(-1000.0)
np.testing.assert_allclose(
    masked_mse(bad_visible, patches, hidden),
    masked_mse(prediction, patches, hidden),
)
assert float(jnp.mean((bad_visible - patches) ** 2)) > 1000
print('Hidden MSE unchanged; full-image MSE is large.')

# Experiment: Make hidden information impossible to recover
# Experiment — Make hidden information impossible to recover: The experiment isolates missing information.
ambiguous_targets = jnp.array([[[0.0]], [[2.0]]])
ambiguous_mask = jnp.ones((2, 1), dtype=bool)

def ambiguity_loss(value):
    return masked_mse(
        jnp.full_like(ambiguous_targets, value), ambiguous_targets, ambiguous_mask
    )

np.testing.assert_allclose(
    [ambiguity_loss(0.0), ambiguity_loss(1.0), ambiguity_loss(2.0)], [2.0, 1.0, 2.0]
)
np.testing.assert_allclose(jax.grad(ambiguity_loss)(1.0), 0.0, atol=1e-7)
print('Ambiguous target MSE at predictions 0, 1, 2: 2, 1, 2')

# Experiment: Move the hidden patches, save the recipe identity, and probe frozen visible inputs
# Changed mask schedule + learned saved encoder + frozen-probe transfer (synthetic)
import hashlib, json, tempfile
from pathlib import Path

# Eight source-group images vary amplitude; class labels are reserved for the probe.
transfer_amp = jnp.array([0.10,0.60,0.20,0.65,0.35,0.80,0.40,0.90])
transfer_images = transfer_amp[:,None,None,None] + base_image[None]
transfer_patches = patchify(transfer_images)
train_ids, held_ids = ['g0','g1','g2','g3'], ['g4','g5','g6','g7']
train_idx, held_idx = np.arange(4), np.arange(4,8)
assert set(train_ids).isdisjoint(held_ids)
# Each fit image gets one of two masks; across epochs the mask pattern alternates.
mask_a = jnp.array([False,True,True,False])
mask_b = jnp.array([True,False,False,True])
mask_schedule = [mask_a, mask_b]
key_e, key_d = jax.random.split(jax.random.key(19))
enc = jax.random.normal(key_e,(8,4))*0.1
# Decoder sees latent plus target patch position, never hidden pixels.
dec = jax.random.normal(key_d,(8,16))*0.1

def mae_loss_for(image_patches, enc, dec, mask):
    visible = (~mask).astype(jnp.float32)
    batch = len(image_patches)
    pooled = jnp.sum(image_patches * visible[None,:,None],axis=1) / visible.sum()
    encoder_input = jnp.concatenate([pooled, jnp.broadcast_to(visible,(batch,4))],axis=1)
    z = jnp.tanh(encoder_input @ enc)
    positions = jnp.eye(4)
    # Position-specific decoder: latent features plus one-hot patch index -> 4 pixels.
    pos_features = jnp.concatenate([jnp.broadcast_to(z[:,None,:],(batch,4,4)),
                                    jnp.broadcast_to(positions[None,:,:],(batch,4,4))],axis=-1)
    pred = pos_features @ dec
    per_patch = jnp.mean((pred-image_patches)**2,axis=-1)
    return jnp.sum(jnp.where(mask[None,:],per_patch,0.0))/ (batch*mask.sum())

def mae_loss(enc, dec, mask):
    return mae_loss_for(transfer_patches[train_idx],enc,dec,mask)

# [4 latent + 4 positional] x 4 target pixels.
dec = jax.random.normal(key_d,(8,4))*0.1
for epoch in range(120):
    mask = mask_schedule[epoch % 2]
    value, (ge,gd) = jax.value_and_grad(mae_loss,argnums=(0,1))(enc,dec,mask)
    enc, dec = enc - 0.08*ge, dec - 0.08*gd

# Recompute both masks without updating and compare with a visible-mean decoder baseline.
changed_mask_losses=[]
held_visible_baselines=[]
for mask in mask_schedule:
    changed_mask_losses.append(float(mae_loss_for(transfer_patches[held_idx],enc,dec,mask)))
    transfer_visible_mask=(~mask).astype(jnp.float32)
    transfer_visible_mean=jnp.sum(transfer_patches[held_idx]*transfer_visible_mask[None,:,None],axis=1)/transfer_visible_mask.sum()
    baseline=jnp.broadcast_to(transfer_visible_mean[:,None,:],transfer_patches[held_idx].shape)
    held_visible_baselines.append(float(masked_mse(baseline,transfer_patches[held_idx],jnp.broadcast_to(mask,(len(held_idx),4)))))
assert all(np.isfinite(changed_mask_losses))

def frozen_encode(images_patches, mask):
    visible=(~mask).astype(jnp.float32)
    pooled=jnp.sum(images_patches*visible[None,:,None],axis=1)/visible.sum()
    x=jnp.concatenate([pooled,jnp.broadcast_to(visible,(len(images_patches),4))],axis=1)
    return jnp.tanh(x @ enc)

# Save/reload exact encoder and decoder identity, then freeze encoder for probe fitting.
processor_id='gray-4x4-range-raw-patchify-2-v1'
source_split_id='synthetic-amplitude-v2:g0-g3/g4-g7'
parameter_hash=hashlib.sha256(np.ascontiguousarray(enc).tobytes()+np.ascontiguousarray(dec).tobytes()).hexdigest()
manifest={'model_id':'tiny-mask-aware-patch-encoder-v1','processor_id':processor_id,
          'source_split_id':source_split_id,'objective':'hidden-patch-MSE',
          'mask_schedule':'alternate center-pair and corner-pair; 50% hidden',
          'fit_group_ids':train_ids,'heldout_group_ids':held_ids,'seed':19,'parameter_sha256':parameter_hash}
output_dir = Path(tempfile.mkdtemp(prefix='jaxpathways-mim-encoder-'))
artifact = output_dir / 'encoder.npz'
np.savez(artifact,encoder=np.asarray(enc),decoder=np.asarray(dec),
         manifest=np.array(json.dumps(manifest,sort_keys=True)))
with np.load(artifact,allow_pickle=False) as saved:
    loaded_enc=saved['encoder'].copy(); loaded_dec=saved['decoder'].copy()
    loaded_manifest=json.loads(str(saved['manifest']))
rehash=hashlib.sha256(np.ascontiguousarray(loaded_enc).tobytes()+np.ascontiguousarray(loaded_dec).tobytes()).hexdigest()
assert loaded_manifest==manifest and rehash==parameter_hash
checkpoint_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
identity_path=output_dir/'identity.json'
identity={**manifest,'checkpoint_sha256':checkpoint_hash}
identity_path.write_text(json.dumps(identity,sort_keys=True))
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==identity['checkpoint_sha256']
frozen_before=loaded_enc.copy()
z=np.asarray(frozen_encode(transfer_patches,mask_a))
labels=(np.asarray(transfer_amp)>=0.5).astype(int)
design=np.c_[z[train_idx],np.ones(len(train_idx))]
probe=np.linalg.solve(design.T@design+1e-3*np.eye(design.shape[1]),design.T@labels[train_idx])
pred=(np.c_[z[held_idx],np.ones(len(held_idx))]@probe>=0.5).astype(int)
frozen_probe_accuracy=float(np.mean(pred==labels[held_idx]))
np.testing.assert_array_equal(loaded_enc,frozen_before)
print('MIM held-out alternating-mask losses / visible-mean baseline / frozen held-out probe accuracy:',
      changed_mask_losses, held_visible_baselines, frozen_probe_accuracy)
print('Artifact path:', artifact)
print('Identity sidecar:', identity_path)
print('Saved identity:',manifest['model_id'],manifest['processor_id'],manifest['source_split_id'])

# Reference solution. Try the exercise before reading this.
# Exercise solution
np.testing.assert_array_equal(
    patches[2, 3], np.asarray(images[2, 2:, 2:, :]).reshape(-1)
)
print('Bottom-right patch order verified.')

# Reference practice: Expose the leakage shortcut
# Expose the leakage shortcut (Transfer)
shortcut = patches
assert float(masked_mse(shortcut, patches, hidden)) == 0.0
# Check tensor shape invariant: `features.shape[1] == 9`
assert features.shape[1] == 9
print('The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.')

# Reference practice: Audit every patch of a rectangular color image
# Audit every patch of a rectangular color image (Challenge)
numbered = np.arange(2 * 4 * 6 * 3, dtype=np.float32).reshape(2, 4, 6, 3)
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
np.testing.assert_array_equal(patchify(jnp.asarray(numbered)), oracle)
changed_images = images.at[:, :2, 2:, :].add(17.0).at[:, 2:, :2, :].add(-9.0)
changed_patches = patchify(changed_images)
np.testing.assert_array_equal(changed_patches[:, [0, 3], :], visible)
assert not np.array_equal(
    np.asarray(changed_patches[:, [1, 2], :]), np.asarray(patches[:, [1, 2], :])
)
print('All rectangular RGB patches agree; hidden-pixel intervention preserves visible features.')
print("PASS: pretraining-02")
