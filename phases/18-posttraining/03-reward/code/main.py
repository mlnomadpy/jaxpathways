"""Learn a reward model from pairwise preferences: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
import numpy as np

def preference_loss(w, chosen, rejected):
    margin = (chosen-rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

features=jnp.array([[1.,0.],[0.,1.],[-1.,-1.]])
chosen=features[jnp.array([0,0,1])];rejected=features[jnp.array([1,2,2])]
w=jnp.zeros(2);history=[]
step=jax.jit(jax.value_and_grad(lambda w:preference_loss(w,chosen,rejected)))
for _ in range(100):
    value,g=step(w);history.append(float(value));w=w-.1*g
rewards=features@w
assert rewards[0]>rewards[1]>rewards[2]
np.testing.assert_allclose(history[0],np.log(2),atol=1e-6)
assert history[-1]<.1
margins=np.asarray((chosen-rejected)@w,dtype=np.float64)
np.testing.assert_allclose(preference_loss(w,chosen,rejected),np.mean(np.logaddexp(0,-margins)),atol=1e-6)
# Only differences are identified: a common score offset leaves preference probabilities unchanged.
np.testing.assert_allclose(jax.nn.sigmoid((chosen@w+9)-(rejected@w+9)),jax.nn.sigmoid((chosen-rejected)@w),atol=1e-6)
print('Synthetic preference NLL initial/final:',history[0],history[-1],'; rewards:',np.asarray(rewards))


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Reverse the comparison
forward=float(preference_loss(w,chosen,rejected))
reverse=float(preference_loss(w,rejected,chosen))
assert reverse>forward
print('Forward/reversed preference NLL:',forward,reverse)

# Reference solution. Try the exercise before reading this.
np.testing.assert_allclose(preference_loss(jnp.zeros(2),chosen,rejected),-np.log(.5),atol=1e-6)
assert np.all(np.asarray((chosen-rejected)@w)>0)
print('Tied baseline and preference directions verified.')

# Reference practice: Show that feature scale can change scores without changing ranking
scaled_rewards=features@(2*w)
np.testing.assert_array_equal(np.argsort(rewards),np.argsort(scaled_rewards))
assert float(preference_loss(2*w,chosen,rejected))<float(preference_loss(w,chosen,rejected))
print('Ranking unchanged; confidence and downstream reward scale change.')
print("PASS: posttraining-03")
