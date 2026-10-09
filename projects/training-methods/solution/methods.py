"""Small CPU objective contracts; synthetic fixtures are not language/vision benchmarks."""
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `masked_ce(logits, targets, selected)` implementing this stage's computation:
def masked_ce(logits, targets, selected):
    # Caller validates a nonempty mask before a transformed training step.
    logp = jax.nn.log_softmax(logits, axis=-1)
    # Compute `nll` as `-jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]`.
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    # Return `jnp.sum(jnp.where(selected, nll, 0.0)) / jnp.sum(selected)` to the caller.
    return jnp.sum(jnp.where(selected, nll, 0.)) / jnp.sum(selected)

# Function `validate_mask(selected, shape)` implementing this stage's computation:
def validate_mask(selected, shape):
    # Convert `a` to a host NumPy array for inspection or verification.
    a = np.asarray(selected)
    # Guard input contract (`a.shape != shape or a.dtype != np.bool_ or (not a.any())`) and fail fast if violated.
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Function `corrupt_tokens(tokens, selected, mask_id)` implementing this stage's computation:
def corrupt_tokens(tokens, selected, mask_id):
    # Run `validate_mask` to perform the next check or state transition.
    validate_mask(selected, tokens.shape)
    # Return `jnp.where(selected, mask_id, tokens)` to the caller.
    return jnp.where(selected, mask_id, tokens)

# Function `mlm_logits(p, corrupted)` implementing this stage's computation:
def mlm_logits(p, corrupted):
    # Compute `h` as `p['embedding'][corrupted]`.
    h = p['embedding'][corrupted]
    # Bidirectional single-head attention, deliberately no causal mask.
    scores = h @ jnp.swapaxes(h, -1, -2) / jnp.sqrt(h.shape[-1])
    # Perform matrix / vector contraction (`@`) to compute `context`.
    context = jax.nn.softmax(scores, axis=-1) @ h
    # Return `context @ p['head']` to the caller.
    return context @ p['head']

# Function `patchify(images, patch)` implementing this stage's computation:
def patchify(images, patch=2):
    # Compute `b,h,w,c` as `images.shape`.
    b,h,w,c = images.shape
    # Guard input contract (`h % patch or w % patch`) and fail fast if violated.
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    # Return `images.reshape(b, h // patch, patch, w // patch, patch, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h // patch * (w // patch), patch * patch * c)` to the caller.
    return images.reshape(b,h//patch,patch,w//patch,patch,c).transpose(0,1,3,2,4,5).reshape(b,(h//patch)*(w//patch),patch*patch*c)

# Function `masked_mse(prediction, target, hidden)` implementing this stage's computation:
def masked_mse(prediction, target, hidden):
    # Reduce along axis=-1 to compute `per_patch`.
    per_patch = jnp.mean((prediction-target)**2, axis=-1)
    # Return `jnp.sum(jnp.where(hidden, per_patch, 0.0)) / jnp.sum(hidden)` to the caller.
    return jnp.sum(jnp.where(hidden, per_patch, 0.)) / jnp.sum(hidden)

# Function `paired_contrastive(left, right, temperature)` implementing this stage's computation:
def paired_contrastive(left, right, temperature=.2):
    # Reduce across the target axis to summarize `left`.
    left = left / jnp.maximum(jnp.linalg.norm(left,axis=-1,keepdims=True),1e-6)
    # Reduce across the target axis to summarize `right`.
    right = right / jnp.maximum(jnp.linalg.norm(right,axis=-1,keepdims=True),1e-6)
    # Perform matrix contraction / projection to compute `scores`.
    scores = left @ right.T / temperature
    # Return `-0.5 * (jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0))))` to the caller.
    return -.5*(jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=0))))

# Function `response_logps(logits, tokens, response_mask)` implementing this stage's computation:
def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:,:-1,:],axis=-1)
    # Run `jnp.take_along_axis` to compute `selected`.
    selected = jnp.take_along_axis(logp,tokens[:,1:,None],axis=-1)[...,0]
    # Return `jnp.sum(jnp.where(response_mask[:, 1:], selected, 0.0), axis=-1)` to the caller.
    return jnp.sum(jnp.where(response_mask[:,1:],selected,0.),axis=-1)

# Function `lora_forward(x, base, a, b, ...)` implementing this stage's computation:
def lora_forward(x, base, a, b, alpha=1.):
    # Compute `rank` as `a.shape[1]`.
    rank = a.shape[1]
    # Return `x @ base + alpha / rank * (x @ a @ b)` to the caller.
    return x @ base + (alpha/rank) * (x @ a @ b)

# Function `preference_loss(w, chosen, rejected)` implementing this stage's computation:
def preference_loss(w, chosen, rejected):
    # Perform matrix contraction / projection to compute `margin`.
    margin = (chosen-rejected) @ w
    # Return `jnp.mean(jax.nn.softplus(-margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-margin))

# Function `ppo_loss(logits, old_logps, actions, advantages, ...)` implementing this stage's computation:
def ppo_loss(logits, old_logps, actions, advantages, reference_logps, beta=.1, clip=.2):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp = jax.nn.log_softmax(logits)
    # Run `jnp.exp` to compute `ratios`.
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    # Run `jax.lax.stop_gradient` to compute `advantages`.
    advantages = jax.lax.stop_gradient(advantages)
    # Combine or mask array elements to form `clipped`.
    clipped = jnp.clip(ratios,1.-clip,1.+clip)
    # Reduce across the target axis to summarize `surrogate`.
    surrogate = jnp.mean(jnp.minimum(ratios*advantages,clipped*advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(jnp.exp(logp)*(logp-jax.lax.stop_gradient(reference_logps)))
    # Return `-surrogate + beta * kl` to the caller.
    return -surrogate + beta*kl

# Function `dpo_loss(policy_logps, reference_logps, chosen, rejected, ...)` implementing this stage's computation:
def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=.2):
    # Evaluate the compound expression for `margin`.
    margin = (policy_logps[chosen]-policy_logps[rejected]) - jax.lax.stop_gradient(reference_logps[chosen]-reference_logps[rejected])
    # Return `jnp.mean(jax.nn.softplus(-beta * margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-beta*margin))
