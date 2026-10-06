"""Masked image modeling: reconstruct missing patches: worked experiments and reference solutions. CPU checks."""



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
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Make visible-output errors arbitrarily large
bad_visible=prediction.at[:,[0,3],:].set(-1000.)
np.testing.assert_allclose(masked_mse(bad_visible,patches,hidden),masked_mse(prediction,patches,hidden))
assert float(jnp.mean((bad_visible-patches)**2))>1000
print('Hidden MSE unchanged; full-image MSE is large.')

# Reference solution. Try the exercise before reading this.
np.testing.assert_array_equal(patches[2,3],np.asarray(images[2,2:,2:,:]).reshape(-1))
print('Bottom-right patch order verified.')

# Reference practice: Expose the leakage shortcut
shortcut=patches
assert float(masked_mse(shortcut,patches,hidden))==0.
assert features.shape[1]==9
print('The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.')
print("PASS: pretraining-02")
