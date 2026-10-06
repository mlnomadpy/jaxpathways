"""Masked language modeling: predict hidden tokens: worked experiments and reference solutions. CPU checks."""



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

tokens = jnp.array([[0,0,0,0],[1,1,1,1],[2,2,2,2]],jnp.int32)
selected = jnp.array([[False,True,False,False]]*3)
corrupted = corrupt_tokens(tokens,selected,3)
key = jax.random.key(7)
p = {'embedding':jax.random.normal(key,(4,6))*.2,'head':jnp.zeros((6,3))}
loss = lambda p: masked_ce(mlm_logits(p,corrupted),tokens,selected)
step = jax.jit(jax.value_and_grad(loss)); history=[]
for _ in range(100):
    value,g = step(p); history.append(float(value)); p=jax.tree.map(lambda a,b:a-.4*b,p,g)
assert history[-1] < history[0]*.15
np.testing.assert_allclose(history[0],np.log(3),atol=1e-6)
# Changing the clean target after constructing corrupted input cannot change the forward pass.
changed_targets=tokens.at[:,1].set((tokens[:,1]+1)%3)
assert not np.isclose(float(masked_ce(mlm_logits(p,corrupted),changed_targets,selected)),history[-1])
held_mask=jnp.array([[False,False,True,False]]*3)
held_loss=float(masked_ce(mlm_logits(p,corrupt_tokens(tokens,held_mask,3)),tokens,held_mask))
assert held_loss < .2
print('MLM initial/final/changed-mask:',history[0],history[-1],held_loss)


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Show that unsupervised positions have no direct loss gradient
logits=mlm_logits(p,corrupted)
changed_logits=jnp.where(selected[...,None],logits,logits+100.)
np.testing.assert_allclose(masked_ce(changed_logits,tokens,selected),masked_ce(logits,tokens,selected),atol=1e-6)
grad_logits=jax.grad(masked_ce)(logits,tokens,selected)
np.testing.assert_array_equal(np.asarray(grad_logits)[~np.asarray(selected)],0.)
print('Unselected output logits have zero direct loss gradient.')

# Reference solution. Try the exercise before reading this.
first_mask=jnp.array([[True,False,False,False]]*3)
first_loss=float(masked_ce(mlm_logits(p,corrupt_tokens(tokens,first_mask,3)),tokens,first_mask))
assert first_loss<.2
print('Changed position loss:',first_loss)

# Reference practice: Reject a missing learning signal
try:corrupt_tokens(tokens,jnp.zeros_like(tokens,dtype=bool),3)
except ValueError:print('Empty mask rejected')
else:raise AssertionError('empty mask accepted')
print("PASS: pretraining-01")
