"""Supervised fine-tuning with response-only token loss: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
import numpy as np

def response_logps(logits, tokens, response_mask):
    # logits at t predict token t+1; the role mask belongs to the target token.
    logp = jax.nn.log_softmax(logits[:,:-1,:],axis=-1)
    selected = jnp.take_along_axis(logp,tokens[:,1:,None],axis=-1)[...,0]
    return jnp.sum(jnp.where(response_mask[:,1:],selected,0.),axis=-1)

def validate_mask(selected, shape):
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens=jnp.array([[0,2,4],[1,3,4]],jnp.int32)
roles=jnp.array([[False,True,True]]*2)
validate_mask(roles[:,1:],tokens[:,1:].shape)
p=jnp.zeros((5,5));history=[]
def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens],tokens,roles))/jnp.sum(roles[:,1:])
step=jax.jit(jax.value_and_grad(sft_objective))
for _ in range(100):
    value,g=step(p);history.append(float(value));p=p-.5*g
np.testing.assert_allclose(history[0],np.log(5),atol=1e-6)
assert history[-1]<.15
assert np.array_equal(np.argmax(np.asarray(p)[[0,1]],axis=1),[2,3])
# The final position predicts nothing and has no contribution.
base_logits=p[tokens]
changed=base_logits.at[:,-1,:].set(100.)
np.testing.assert_allclose(response_logps(changed,tokens,roles),response_logps(base_logits,tokens,roles))
print('Response-only token NLL initial/final:',history[0],history[-1])


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Prove the target-mask shift
g=jax.grad(sft_objective)(jnp.zeros((5,5)))
assert np.linalg.norm(np.asarray(g)[0])>0 and np.linalg.norm(np.asarray(g)[1])>0
np.testing.assert_array_equal(np.asarray(g)[4],0.)
print('Prompt rows learn to predict first answers; the final end-token row has no target.')

# Reference solution. Try the exercise before reading this.
first_only=roles.at[:,2].set(False)
value=-jnp.sum(response_logps(p[tokens],tokens,first_only))/jnp.sum(first_only[:,1:])
assert int(jnp.sum(first_only[:,1:]))==2
assert np.isfinite(float(value))
print('Two first-response targets:',float(value))

# Reference practice: Reject a batch with no answer targets
try:validate_mask(jnp.zeros_like(roles[:,1:]),tokens[:,1:].shape)
except ValueError:print('No-target SFT batch rejected')
else:raise AssertionError('empty supervision accepted')
print("PASS: posttraining-01")
