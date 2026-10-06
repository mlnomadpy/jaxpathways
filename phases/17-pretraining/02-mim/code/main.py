"""Masked image modeling: reconstruct missing patches: worked experiments and reference solutions. CPU checks."""

# 1. Define patch layout and masked error
import jax
import jax.numpy as jnp
import numpy as np

def patchify(images, patch=2):
    b,h,w,c = images.shape
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    return images.reshape(b,h//patch,patch,w//patch,patch,c).transpose(0,1,3,2,4,5).reshape(b,(h//patch)*(w//patch),patch*patch*c)

def masked_mse(prediction, target, hidden):
    per_patch = jnp.mean((prediction-target)**2, axis=-1)
    return jnp.sum(jnp.where(hidden, per_patch, 0.)) / jnp.sum(hidden)

def validate_mask(selected, shape):
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# 2. Separate visible features from full targets
# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([.1,.3,.6,.8])
base_image = jnp.arange(16,dtype=jnp.float32).reshape(4,4,1)/32
images = amplitude[:,None,None,None]+base_image[None]
patches = patchify(images)
np.testing.assert_array_equal(patches[0,0],np.asarray(images[0,:2,:2,:]).reshape(-1))
hidden = jnp.array([[False,True,True,False]]*4)
validate_mask(hidden,patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:,[0,3],:]
features = jnp.concatenate([visible.reshape(4,-1),jnp.ones((4,1))],axis=1)
w = jnp.zeros((9,16)); history=[]
loss=lambda w:masked_mse((features@w).reshape(4,4,4),patches,hidden)
step=jax.jit(jax.value_and_grad(loss))

# 3. Fit the decoder and audit the reconstruction
for _ in range(100):
    value,g=step(w);history.append(float(value));w=w-.15*g
prediction=(features@w).reshape(4,4,4)
assert history[-1] < history[0]*.02
# A scalar host loop is independent of the vectorized reduction.
expected=np.mean([float(np.mean((np.asarray(prediction[b,k])-np.asarray(patches[b,k]))**2)) for b in range(4) for k in [1,2]])
np.testing.assert_allclose(loss(w),expected,rtol=1e-5)
visible_changed=prediction.at[:,[0,3],:].set(999.)
np.testing.assert_allclose(masked_mse(visible_changed,patches,hidden),loss(w),atol=1e-7)
held_images=jnp.array([.2,.5])[:,None,None,None]+base_image[None]
held_patches=patchify(held_images)
held_features=jnp.concatenate([held_patches[:,[0,3],:].reshape(2,-1),jnp.ones((2,1))],axis=1)
held_loss=float(masked_mse((held_features@w).reshape(2,4,4),held_patches,hidden[:2]))
assert held_loss < .02
print('Masked pixel MSE initial/final/held amplitudes:',history[0],history[-1],held_loss)

import jax
import jax.numpy as jnp
import numpy as np

def patchify(images, patch=2):
    b,h,w,c = images.shape
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    return images.reshape(b,h//patch,patch,w//patch,patch,c).transpose(0,1,3,2,4,5).reshape(b,(h//patch)*(w//patch),patch*patch*c)

def masked_mse(prediction, target, hidden):
    per_patch = jnp.mean((prediction-target)**2, axis=-1)
    return jnp.sum(jnp.where(hidden, per_patch, 0.)) / jnp.sum(hidden)

def validate_mask(selected, shape):
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Four patches share an amplitude plus a fixed spatial offset; no real photographs.
amplitude = jnp.array([.1,.3,.6,.8])
base_image = jnp.arange(16,dtype=jnp.float32).reshape(4,4,1)/32
images = amplitude[:,None,None,None]+base_image[None]
patches = patchify(images)
np.testing.assert_array_equal(patches[0,0],np.asarray(images[0,:2,:2,:]).reshape(-1))
hidden = jnp.array([[False,True,True,False]]*4)
validate_mask(hidden,patches.shape[:2])
# The encoder receives only visible patches; original hidden pixels are targets only.
visible = patches[:,[0,3],:]
features = jnp.concatenate([visible.reshape(4,-1),jnp.ones((4,1))],axis=1)
w = jnp.zeros((9,16)); history=[]
loss=lambda w:masked_mse((features@w).reshape(4,4,4),patches,hidden)
step=jax.jit(jax.value_and_grad(loss))
for _ in range(100):
    value,g=step(w);history.append(float(value));w=w-.15*g
prediction=(features@w).reshape(4,4,4)
assert history[-1] < history[0]*.02
# A scalar host loop is independent of the vectorized reduction.
expected=np.mean([float(np.mean((np.asarray(prediction[b,k])-np.asarray(patches[b,k]))**2)) for b in range(4) for k in [1,2]])
np.testing.assert_allclose(loss(w),expected,rtol=1e-5)
visible_changed=prediction.at[:,[0,3],:].set(999.)
np.testing.assert_allclose(masked_mse(visible_changed,patches,hidden),loss(w),atol=1e-7)
held_images=jnp.array([.2,.5])[:,None,None,None]+base_image[None]
held_patches=patchify(held_images)
held_features=jnp.concatenate([held_patches[:,[0,3],:].reshape(2,-1),jnp.ones((2,1))],axis=1)
held_loss=float(masked_mse((held_features@w).reshape(2,4,4),held_patches,hidden[:2]))
assert held_loss < .02
print('Masked pixel MSE initial/final/held amplitudes:',history[0],history[-1],held_loss)


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'hidden-pixel MSE (squared input units)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

patch_comparison=np.concatenate([np.asarray(patches[0,[1,2],:]),np.asarray(prediction[0,[1,2],:])],axis=0)
extra_panel={'kind':'heatmap','values':patch_comparison.tolist(),'rows':['target patch 1','target patch 2','predicted patch 1','predicted patch 2'],'columns':['pixel 0','pixel 1','pixel 2','pixel 3'],'unit':'raw pixel value','xlabel':'within-patch pixel order','ylabel':'hidden patch and source','title':'Hidden targets and final predictions for the first image'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}


# Experiment: Make visible-output errors arbitrarily large
bad_visible=prediction.at[:,[0,3],:].set(-1000.)
np.testing.assert_allclose(masked_mse(bad_visible,patches,hidden),masked_mse(prediction,patches,hidden))
assert float(jnp.mean((bad_visible-patches)**2))>1000
print('Hidden MSE unchanged; full-image MSE is large.')

# Experiment: Make hidden information impossible to recover
ambiguous_targets=jnp.array([[[0.]],[[2.]]]);ambiguous_mask=jnp.ones((2,1),dtype=bool)
def ambiguity_loss(value):return masked_mse(jnp.full_like(ambiguous_targets,value),ambiguous_targets,ambiguous_mask)
np.testing.assert_allclose([ambiguity_loss(0.),ambiguity_loss(1.),ambiguity_loss(2.)],[2.,1.,2.])
np.testing.assert_allclose(jax.grad(ambiguity_loss)(1.),0.,atol=1e-7)
print('Ambiguous target MSE at predictions 0, 1, 2: 2, 1, 2')

# Reference solution. Try the exercise before reading this.
np.testing.assert_array_equal(patches[2,3],np.asarray(images[2,2:,2:,:]).reshape(-1))
print('Bottom-right patch order verified.')

# Reference practice: Expose the leakage shortcut
shortcut=patches
assert float(masked_mse(shortcut,patches,hidden))==0.
assert features.shape[1]==9
print('The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.')

# Reference practice: Audit every patch of a rectangular color image
numbered=np.arange(2*4*6*3,dtype=np.float32).reshape(2,4,6,3)
oracle=np.stack([np.stack([image[row:row+2,col:col+2,:].reshape(-1) for row in range(0,4,2) for col in range(0,6,2)]) for image in numbered])
np.testing.assert_array_equal(patchify(jnp.asarray(numbered)),oracle)
changed_images=images.at[:,:2,2:,:].add(17.).at[:,2:,:2,:].add(-9.)
changed_patches=patchify(changed_images)
np.testing.assert_array_equal(changed_patches[:,[0,3],:],visible)
assert not np.array_equal(np.asarray(changed_patches[:,[1,2],:]),np.asarray(patches[:,[1,2],:]))
print('All rectangular RGB patches agree; hidden-pixel intervention preserves visible features.')
print("PASS: pretraining-02")
