"""Direct preference optimization and reference-corrected margins: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
import numpy as np

def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=.2):
    margin = (policy_logps[chosen]-policy_logps[rejected]) - jax.lax.stop_gradient(reference_logps[chosen]-reference_logps[rejected])
    return jnp.mean(jax.nn.softplus(-beta*margin))

reference=jax.nn.log_softmax(jnp.array([.2,0.,-.2]))
chosen=jnp.array([0,0,1]);rejected=jnp.array([1,2,2]);theta=jnp.array([.2,0.,-.2]);history=[]
loss=lambda theta:dpo_loss(jax.nn.log_softmax(theta),reference,chosen,rejected,.3)
step=jax.jit(jax.value_and_grad(loss))
np.testing.assert_allclose(loss(theta),np.log(2),atol=1e-6)
for _ in range(120):
    value,g=step(theta);history.append(float(value));theta=theta-.4*g
assert history[-1]<history[0]*.5
policy=jax.nn.log_softmax(theta)
margin=np.asarray((policy[chosen]-policy[rejected])-(reference[chosen]-reference[rejected]),np.float64)
np.testing.assert_allclose(loss(theta),np.mean(np.logaddexp(0,-.3*margin)),atol=1e-6)
assert np.all(margin>0)
np.testing.assert_allclose(loss(theta+50),loss(theta),atol=1e-6)
print('DPO initial/final:',history[0],history[-1],'; reference-corrected margins:',margin)


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Verify the reference cancellation
initial=dpo_loss(reference,reference,chosen,rejected,.3)
np.testing.assert_allclose(initial,np.log(2),atol=1e-6)
uncorrected=jnp.mean(jax.nn.softplus(-.3*(reference[chosen]-reference[rejected])))
assert not np.isclose(float(initial),float(uncorrected))
print('Corrected/unadjusted initial loss:',float(initial),float(uncorrected))

# Reference solution. Try the exercise before reading this.
reverse=float(dpo_loss(policy,reference,rejected,chosen,.3))
assert reverse>float(dpo_loss(policy,reference,chosen,rejected,.3))
print('Reversed-label loss:',reverse)

# Reference practice: Freeze the reference numerically and in autodiff
reference_grad=jax.grad(dpo_loss,argnums=1)(policy,reference,chosen,rejected,.3)
np.testing.assert_array_equal(reference_grad,np.zeros(reference_grad.shape))
print('Reference gradient is zero.')
print("PASS: posttraining-05")
