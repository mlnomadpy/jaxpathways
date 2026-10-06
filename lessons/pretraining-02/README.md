# Masked image modeling: reconstruct missing patches

Phase 17: Self-supervised pretraining: masked and contrastive learning · about 75 minutes · CPU

## What you will be able to do

- Preserve patch order and pixel order
- Keep hidden pixels out of the encoder
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

If part of an image is hidden, can the model infer it from visible patches? The important question is whether hidden pixels are truly absent from the encoder, not whether the reconstruction looks attractive.

## The idea

We split small synthetic images into patches, expose only two patches to a linear encoder/decoder map and train on the other two. This teaches masking, spatial order and reconstruction accounting. It is not a full ViT masked autoencoder; the simple image generator makes independent checks possible.

## Preserve patch order and pixel order

Images use batch, height, width, channel axes. A $4\times4$ single-channel image becomes four $2\times2$ patches with four scalars each. Reshape alone does not generally produce row-major patch order; transpose the grid and within-patch axes deliberately. The first patch is checked against an independently sliced image region.

## Keep hidden pixels out of the encoder

Gather visible patches before constructing features. The clean full image belongs only to the reconstruction target. In a masked autoencoder, a visible-only encoder can reduce encoder work while a decoder uses position information and mask tokens to restore the complete patch ordering. Our linear decoder has an explicit output coordinate for each patch/pixel, so it has a fixed positional contract without a Transformer.

## Reduce over the hidden reconstruction targets

First average squared pixel error within each patch, then sum the hidden-patch errors and divide by the hidden count. This definition gives equal weight to every hidden pixel because all patches have equal size. Errors on visible outputs are excluded. Mixing visible and hidden pixels can flatter the score when copying visible inputs is easy.

$$
L=\frac{\sum_{b,k}m_{bk}\,\frac1P\sum_{j=1}^{P}(\widehat x_{bkj}-x_{bkj})^2}{\sum_{b,k}m_{bk}}
$$

## Interpret normalization and reconstruction

Raw-pixel reconstruction and per-patch normalized reconstruction are different objectives. Normalization can remove a patch’s brightness/contrast information and requires a clear inverse or evaluation convention. This lesson uses raw synthetic values and records MSE in squared input units. Do not compare that number directly with a differently normalized experiment.

## Evaluate beyond the training mask

The reference checks intermediate amplitudes that were not training examples. That is a small interpolation test, not semantic recognition. For transfer, freeze the encoder and train a probe on separate labels, or fine-tune under a held-out protocol. Compare mask ratios with fixed compute/data budgets and keep all-hidden or all-visible policies explicit.

## Run the example

```python
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

```

Expected: Hidden-patch training MSE falls below 0.002 and held-amplitude MSE stays below 0.02.

## Masked image modeling: reconstruct missing patches — recorded experiment

**Predict:** Predict what should change during training and what this curve cannot establish.

![Masked image modeling: reconstruct missing patches — recorded experiment](../../phases/17-pretraining/02-mim/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The line shows hidden-patch training MSE before each update, falling from about $0.554$ to $0.00169$. The vertical units are squared raw pixel units. The line excludes visible-patch reconstruction errors; it cannot be read as full-image MSE.

### Connect it to the computation

Each image contains an amplitude and a fixed spatial pattern. Visible patches reveal the amplitude, so a linear decoder can infer hidden pixels on this deliberately learnable task. Separate held-amplitude MSE is approximately 0.00118. Natural-image semantics and a visible-token Transformer remain different experiments.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

```

## Recorded reference execution

CPU run: 2026-10-06T01:27:39.126860+00:00. JAX 0.9.2.

```text
Masked pixel MSE initial/final/held amplitudes: 0.5538085699081421 0.0016928859986364841 0.0011766081443056464
Hidden MSE unchanged; full-image MSE is large.
Bottom-right patch order verified.
The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.
PASS: pretraining-02

```

## Make visible-output errors arbitrarily large

**Predict before running:** Will visible-output corruption raise the hidden-patch objective?

```python
bad_visible=prediction.at[:,[0,3],:].set(-1000.)
np.testing.assert_allclose(masked_mse(bad_visible,patches,hidden),masked_mse(prediction,patches,hidden))
assert float(jnp.mean((bad_visible-patches)**2))>1000
print('Hidden MSE unchanged; full-image MSE is large.')
```

**Expected:** Hidden MSE stays unchanged while full-image MSE grows dramatically.

The disagreement identifies the reduction contract. Neither metric is universally correct; label which region and preprocessing it measures.

## Make it yours

Verify the bottom-right patch against a direct slice and explain why a whole-array reshape can hide a spatial permutation.

<details><summary>Reference solution</summary>

```python
np.testing.assert_array_equal(patches[2,3],np.asarray(images[2,2:,2:,:]).reshape(-1))
print('Bottom-right patch order verified.')
```

</details>

## Expose the leakage shortcut

**Transfer**

Compare a decoder that returns the clean target with the visible-only model. Explain why zero reconstruction loss is not sufficient evidence.

<details><summary>Hint</summary>

Measure the shortcut, then inspect which function inputs contain hidden pixels.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
shortcut=patches
assert float(masked_mse(shortcut,patches,hidden))==0.
assert features.shape[1]==9
print('The zero-loss shortcut receives hidden pixels; the real features contain only eight visible pixels and a bias.')
```

An information-flow check is necessary because the leaked solution can outperform the honest model on the objective itself.

</details>

## Check your understanding

Why must the encoder input be inspected separately from reconstruction loss?

1. A leaked target can produce a perfect loss without learning to infer missing content.
2. A lower training loss by itself proves the full application is ready.
3. Matching shapes alone establishes the required behavior.

<details><summary>Answer and explanation</summary>

A leaked target can produce a perfect loss without learning to infer missing content.

Correct reduction does not prevent input leakage. Audit the visible-only boundary and validate patch order.

</details>

## Diagnose the result

If reconstruction is excellent before learning, inspect target leakage. If patches look shuffled, verify the patch transpose and decoder restoration order. If metrics differ, compare the mask count and normalization.

## Keep your evidence

Keep patch-order slices, visible-only feature shapes, hidden-patch counts, independent MSE and held-amplitude reconstruction results.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Masked Autoencoders](https://arxiv.org/abs/2111.06377)

