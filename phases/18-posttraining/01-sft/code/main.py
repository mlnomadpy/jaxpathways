"""Supervised fine-tuning with response-only token loss: worked experiments and reference solutions. CPU checks."""

# 1. Define sequence log-probability sums
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

# 2. Align prompt, response and target masks
# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens=jnp.array([[0,2,4],[1,3,4]],jnp.int32)
roles=jnp.array([[False,True,True]]*2)
validate_mask(roles[:,1:],tokens[:,1:].shape)
p=jnp.zeros((5,5));history=[]
def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens],tokens,roles))/jnp.sum(roles[:,1:])
step=jax.jit(jax.value_and_grad(sft_objective))

# 3. Fit the transition table and check the final position
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
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'response-token NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

initial_output_grads=jax.grad(lambda values:-jnp.sum(response_logps(values,tokens,roles))/jnp.sum(roles[:,1:]))(jnp.zeros((2,3,5)))
extra_panel={'kind':'heatmap','values':np.asarray(jnp.linalg.norm(initial_output_grads,axis=-1)).tolist(),'rows':['prompt 0 / answer 2','prompt 1 / answer 3'],'columns':['input position 0','input position 1','input position 2'],'unit':'output-logit gradient L2 norm','xlabel':'position producing the prediction','ylabel':'sequence','title':'Initial supervised gradient by output position'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}


# Experiment: Prove the target-mask shift
g=jax.grad(sft_objective)(jnp.zeros((5,5)))
assert np.linalg.norm(np.asarray(g)[0])>0 and np.linalg.norm(np.asarray(g)[1])>0
np.testing.assert_array_equal(np.asarray(g)[4],0.)
print('Prompt rows learn to predict first answers; the final end-token row has no target.')

# Experiment: Check a ragged batch against a scalar loop
ragged_tokens=jnp.array([[0,2,4,4],[1,3,2,4]])
ragged_roles=jnp.array([[False,True,True,False],[False,True,True,True]])
ragged_logits=jnp.arange(2*4*5,dtype=jnp.float32).reshape(2,4,5)/19
observed=response_logps(ragged_logits,ragged_tokens,ragged_roles)
arr=np.asarray(ragged_logits,dtype=np.float64);expected=[]
for row in range(2):
 total=0.
 for position in range(3):
  if ragged_roles[row,position+1]:
   values=arr[row,position];log_normalizer=values.max()+np.log(np.exp(values-values.max()).sum())
   total+=values[int(ragged_tokens[row,position+1])]-log_normalizer
 expected.append(total)
np.testing.assert_allclose(observed,expected,atol=1e-6)
print('Supervised targets per sequence:',np.asarray(ragged_roles[:,1:].sum(1)))

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

# Reference practice: Create a failure this model cannot fix
contexts=jnp.array([[1,0],[3,0]])
last_logits=p[contexts[:,-1]]
np.testing.assert_array_equal(last_logits[0],last_logits[1])
conflicting_targets=jnp.array([2,3])
assert conflicting_targets[0]!=conflicting_targets[1]
print('Identical final tokens force identical table predictions despite different contexts.')
print("PASS: posttraining-01")
