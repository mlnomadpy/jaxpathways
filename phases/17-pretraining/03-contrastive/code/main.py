"""Contrastive learning: views, positives and negatives: worked experiments and reference solutions. CPU checks."""

# 1. Define both retrieval directions
import jax
import jax.numpy as jnp
import numpy as np

def paired_contrastive(left, right, temperature=.2):
    left = left / jnp.maximum(jnp.linalg.norm(left,axis=-1,keepdims=True),1e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right,axis=-1,keepdims=True),1e-6)
    scores = left @ right.T / temperature
    return -.5*(jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=0))))

# 2. Construct paired views and an encoder
left=jnp.array([[1.,0.,.2],[0.,1.,-.2],[-1.,0.,.1],[0.,-1.,-.1]])
right=left+jnp.array([[.02,-.01,0.],[-.01,.02,0.],[.01,.01,0.],[-.02,-.01,0.]])
w=jnp.array([[.2,.1],[.1,.1],[.02,-.01]]);history=[]
step=jax.jit(jax.value_and_grad(lambda w:paired_contrastive(left@w,right@w,.2)))

# 3. Train, then inspect identity and symmetry
for _ in range(60):
    value,g=step(w);history.append(float(value));w=w-.03*g
assert history[-1]<history[0]
zi=left@w;zt=right@w
zi=zi/jnp.linalg.norm(zi,axis=1,keepdims=True);zt=zt/jnp.linalg.norm(zt,axis=1,keepdims=True)
scores=zi@zt.T/.2
host=np.asarray(scores,dtype=np.float64)
def host_ce(a):
    return np.mean(np.log(np.exp(a-a.max(1,keepdims=True)).sum(1))+a.max(1)-np.diag(a))
np.testing.assert_allclose(paired_contrastive(left@w,right@w,.2),.5*(host_ce(host)+host_ce(host.T)),atol=1e-6)
assert np.array_equal(np.argmax(host,axis=1),np.arange(4))
permutation=jnp.array([2,0,3,1])
np.testing.assert_allclose(paired_contrastive(left@w,right@w),paired_contrastive((left@w)[permutation],(right@w)[permutation]),atol=1e-6)
print('Paired contrastive initial/final:',history[0],history[-1],'; all four nearest pairs correct')

import jax
import jax.numpy as jnp
import numpy as np

def paired_contrastive(left, right, temperature=.2):
    left = left / jnp.maximum(jnp.linalg.norm(left,axis=-1,keepdims=True),1e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right,axis=-1,keepdims=True),1e-6)
    scores = left @ right.T / temperature
    return -.5*(jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=1))) + jnp.mean(jnp.diag(jax.nn.log_softmax(scores,axis=0))))

left=jnp.array([[1.,0.,.2],[0.,1.,-.2],[-1.,0.,.1],[0.,-1.,-.1]])
right=left+jnp.array([[.02,-.01,0.],[-.01,.02,0.],[.01,.01,0.],[-.02,-.01,0.]])
w=jnp.array([[.2,.1],[.1,.1],[.02,-.01]]);history=[]
step=jax.jit(jax.value_and_grad(lambda w:paired_contrastive(left@w,right@w,.2)))
for _ in range(60):
    value,g=step(w);history.append(float(value));w=w-.03*g
assert history[-1]<history[0]
zi=left@w;zt=right@w
zi=zi/jnp.linalg.norm(zi,axis=1,keepdims=True);zt=zt/jnp.linalg.norm(zt,axis=1,keepdims=True)
scores=zi@zt.T/.2
host=np.asarray(scores,dtype=np.float64)
def host_ce(a):
    return np.mean(np.log(np.exp(a-a.max(1,keepdims=True)).sum(1))+a.max(1)-np.diag(a))
np.testing.assert_allclose(paired_contrastive(left@w,right@w,.2),.5*(host_ce(host)+host_ce(host.T)),atol=1e-6)
assert np.array_equal(np.argmax(host,axis=1),np.arange(4))
permutation=jnp.array([2,0,3,1])
np.testing.assert_allclose(paired_contrastive(left@w,right@w),paired_contrastive((left@w)[permutation],(right@w)[permutation]),atol=1e-6)
print('Paired contrastive initial/final:',history[0],history[-1],'; all four nearest pairs correct')


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'symmetric contrastive loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

extra_panel={'kind':'heatmap','values':np.asarray(zi@zt.T).tolist(),'rows':['left 0','left 1','left 2','left 3'],'columns':['right 0','right 1','right 2','right 3'],'unit':'cosine similarity','diverging':True,'xlabel':'right-view source ID','ylabel':'left-view source ID','title':'Final cross-view similarities before temperature scaling'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}


# Experiment: Measure the collapsed baseline
collapsed=jnp.ones((4,2))
collapse_loss=float(paired_contrastive(collapsed,collapsed))
np.testing.assert_allclose(collapse_loss,np.log(4),atol=1e-6)
print('Collapsed baseline:',collapse_loss)

# Experiment: Separate temperature from ranking
orthogonal=jnp.eye(2)
for tau in [.2,1.,2.]:
 observed=float(paired_contrastive(orthogonal,orthogonal,tau))
 np.testing.assert_allclose(observed,np.logaddexp(0.,-1./tau),atol=1e-6)
 np.testing.assert_array_equal(np.argmax(np.asarray(orthogonal@orthogonal.T)/tau,axis=1),[0,1])
 print('Temperature/loss:',tau,observed)

# Reference solution. Try the exercise before reading this.
wrong_pair_loss=float(paired_contrastive(left@w,(right@w)[permutation]))
assert wrong_pair_loss>float(paired_contrastive(left@w,right@w))
print('Incorrect pair mapping loss:',wrong_pair_loss)

# Reference practice: Duplicate a source and inspect the objective
duplicated=float(paired_contrastive(jnp.tile(left@w,(2,1)),jnp.tile(right@w,(2,1))))
original=float(paired_contrastive(left@w,right@w))
np.testing.assert_allclose(duplicated-original,np.log(2),atol=1e-5)
print('Duplicated diagonal-only penalty:',duplicated-original)

# Reference practice: Check the encoder gradient through normalization
audit_w=jnp.array([[.2,.1],[.1,.1],[.02,-.01]])
objective=lambda weights:paired_contrastive(left@weights,right@weights,.2)
auto=float(jax.grad(objective)(audit_w)[0,0])
for epsilon in [1e-3,5e-4]:
 delta=jnp.zeros_like(audit_w).at[0,0].set(epsilon)
 estimate=float((objective(audit_w+delta)-objective(audit_w-delta))/(2*epsilon))
 np.testing.assert_allclose(estimate,auto,rtol=3e-3,atol=1e-3)
 print('Step / finite difference / autodiff:',epsilon,estimate,auto)
print("PASS: pretraining-03")
