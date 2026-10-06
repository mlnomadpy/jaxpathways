"""Contrastive learning: views, positives and negatives: worked experiments and reference solutions. CPU checks."""



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
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Measure the collapsed baseline
collapsed=jnp.ones((4,2))
collapse_loss=float(paired_contrastive(collapsed,collapsed))
np.testing.assert_allclose(collapse_loss,np.log(4),atol=1e-6)
print('Collapsed baseline:',collapse_loss)

# Reference solution. Try the exercise before reading this.
wrong_pair_loss=float(paired_contrastive(left@w,(right@w)[permutation]))
assert wrong_pair_loss>float(paired_contrastive(left@w,right@w))
print('Incorrect pair mapping loss:',wrong_pair_loss)

# Reference practice: Duplicate a source and inspect the objective
duplicated=float(paired_contrastive(jnp.tile(left@w,(2,1)),jnp.tile(right@w,(2,1))))
original=float(paired_contrastive(left@w,right@w))
np.testing.assert_allclose(duplicated-original,np.log(2),atol=1e-5)
print('Duplicated diagonal-only penalty:',duplicated-original)
print("PASS: pretraining-03")
