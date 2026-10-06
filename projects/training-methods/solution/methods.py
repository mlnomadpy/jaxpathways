"""Small CPU objective contracts; synthetic fixtures are not language/vision benchmarks."""
import jax
import jax.numpy as jnp
import numpy as np

def masked_ce(logits, targets, selected):
    # Caller validates a nonempty mask before a transformed training step.
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(selected, nll, 0.)) / jnp.sum(selected)

def validate_mask(selected, shape):
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

def corrupt_tokens(tokens, selected, mask_id):
    validate_mask(selected, tokens.shape)
    return jnp.where(selected, mask_id, tokens)

def mlm_logits(p, corrupted):
    h = p['embedding'][corrupted]
    # Bidirectional single-head attention, deliberately no causal mask.
    scores = h @ jnp.swapaxes(h, -1, -2) / jnp.sqrt(h.shape[-1])
    context = jax.nn.softmax(scores, axis=-1) @ h
    return context @ p['head']

def patchify(images, patch=2):
    b,h,w,c = images.shape
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size')
    return images.reshape(b,h//patch,patch,w//patch,patch,c).transpose(0,1,3,2,4,5).reshape(b,(h//patch)*(w//patch),patch*patch*c)

def masked_mse(prediction, target, hidden):
    per_patch = jnp.mean((prediction-target)**2, axis=-1)
    return jnp.sum(jnp.where(hidden, per_patch, 0.)) / jnp.sum(hidden)

def paired_contrastive(left, right, temperature=.2):
    left = left / jnp.maximum(jnp.linalg.norm(left,axis=-1,keepdims=True),1e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right,axis=-1,keepdims=True),1e-6)
    scores = left @ right.T / temperature
    return -.5*(jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=0))))

def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:,:-1,:],axis=-1)
    selected = jnp.take_along_axis(logp,tokens[:,1:,None],axis=-1)[...,0]
    return jnp.sum(jnp.where(response_mask[:,1:],selected,0.),axis=-1)

def lora_forward(x, base, a, b, alpha=1.):
    rank = a.shape[1]
    return x @ base + (alpha/rank) * (x @ a @ b)

def preference_loss(w, chosen, rejected):
    margin = (chosen-rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

def ppo_loss(logits, old_logps, actions, advantages, reference_logps, beta=.1, clip=.2):
    logp = jax.nn.log_softmax(logits)
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    advantages = jax.lax.stop_gradient(advantages)
    clipped = jnp.clip(ratios,1.-clip,1.+clip)
    surrogate = jnp.mean(jnp.minimum(ratios*advantages,clipped*advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(jnp.exp(logp)*(logp-jax.lax.stop_gradient(reference_logps)))
    return -surrogate + beta*kl

def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=.2):
    margin = (policy_logps[chosen]-policy_logps[rejected]) - jax.lax.stop_gradient(reference_logps[chosen]-reference_logps[rejected])
    return jnp.mean(jax.nn.softplus(-beta*margin))
