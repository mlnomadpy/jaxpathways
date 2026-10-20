"""Reference implementation: Audit response-token masking and a preference pair objective with explicit slice metrics."""
import jax
import jax.numpy as jnp

def posttraining_losses(logits, targets, response_mask, chosen_logp, rejected_logp, beta=0.1, reference_chosen_logp=-0.69314718056, reference_rejected_logp=-0.69314718056):
    logits=jnp.asarray(logits); targets=jnp.asarray(targets); mask=jnp.asarray(response_mask,dtype=bool)
    if logits.ndim!=3 or targets.shape!=logits.shape[:2] or mask.shape!=targets.shape: raise ValueError("incompatible token shapes")
    logp=jax.nn.log_softmax(logits,axis=-1)
    token=-jnp.take_along_axis(logp,targets[...,None],axis=-1)[...,0]
    denom=jnp.sum(mask)
    if int(denom)==0: raise ValueError("response mask is empty")
    sft=jnp.sum(jnp.where(mask,token,0))/denom
    margin=(chosen_logp-rejected_logp)-(reference_chosen_logp-reference_rejected_logp)
    dpo=-jax.nn.log_sigmoid(beta*margin)
    return {"sft_loss":sft,"response_tokens":int(denom),"preference_margin":margin,"dpo_loss":dpo}
