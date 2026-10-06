"""RLHF mechanics: a frozen reward and PPO policy updates: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
import numpy as np

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

features=jnp.array([[1.,0.],[0.,1.],[-1.,-1.]])
chosen=features[jnp.array([0,0,1])];rejected=features[jnp.array([1,2,2])]
reward_w=jnp.zeros(2)
reward_step=jax.jit(jax.grad(lambda w:preference_loss(w,chosen,rejected)))
for _ in range(100):reward_w=reward_w-.1*reward_step(reward_w)
rewards=jax.lax.stop_gradient(features@reward_w)
# A fixed reference policy and a learned reward, with one terminal response action.
reference=jax.nn.log_softmax(jnp.array([.2,0.,-.2]));theta=jnp.array([.2,0.,-.2])
reference_copy=np.asarray(reference).copy();key=jax.random.key(11);history=[];kl_history=[]
step=jax.jit(jax.value_and_grad(ppo_loss))
for iteration in range(40):
    old=jax.nn.log_softmax(theta);probs=jnp.exp(old)
    key,draw=jax.random.split(key);actions=jax.random.categorical(draw,old,shape=(128,))
    old_selected=jax.lax.stop_gradient(old[actions])
    advantage=jax.lax.stop_gradient(rewards[actions]-jnp.sum(probs*rewards))
    for _ in range(3):
        _,g=step(theta,old_selected,actions,advantage,reference,.2,.2);theta=theta-.15*g
    logp=jax.nn.log_softmax(theta)
    history.append(float(jnp.sum(jnp.exp(logp)*rewards)))
    kl_history.append(float(jnp.sum(jnp.exp(logp)*(logp-reference))))
np.testing.assert_array_equal(reference,reference_copy)
initial_reward=float(jnp.sum(jnp.exp(reference)*rewards))
assert history[-1]>initial_reward and kl_history[-1]>0
# Independent finite-action expectation: no evaluation sampling error here.
np.testing.assert_allclose(history[-1],sum(float(a)*float(b) for a,b in zip(jnp.exp(logp),rewards)),rtol=1e-6)
print('Reward initial/final and final KL:',initial_reward,history[-1],kl_history[-1])
print('Synthetic preference bandit with actual sampled PPO updates; no human ratings or language-generation claim.')


# Figure data experiment
visual_data={'panels':[{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'expected learned reward','series':[{'label':'exact policy expectation','x':list(range(1,len(history)+1)),'y':history}]},{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'KL to fixed reference (nats)','series':[{'label':'exact categorical KL','x':list(range(1,len(kl_history)+1)),'y':kl_history}]}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Inspect both clipping directions
ratios=np.array([1.5,.5]);advantages=np.array([2.,-2.])
terms=np.minimum(ratios*advantages,np.clip(ratios,.8,1.2)*advantages)
np.testing.assert_allclose(terms,[2.4,-1.6])
print('Clipped surrogate terms:',terms)

# Reference solution. Try the exercise before reading this.
zero_kl=float(jnp.sum(jnp.exp(reference)*(reference-reference)))
assert zero_kl==0. and kl_history[-1]>zero_kl
print('Reference/final KL:',zero_kl,kl_history[-1])

# Reference practice: Verify detached behavior information
old_grad,adv_grad=jax.grad(ppo_loss,argnums=(1,3))(theta,old_selected,actions,advantage,reference,.2,.2)
np.testing.assert_array_equal(old_grad,np.zeros(old_grad.shape))
np.testing.assert_array_equal(adv_grad,np.zeros(adv_grad.shape))
print('Behavior log probabilities and advantages are detached.')
print("PASS: posttraining-04")
