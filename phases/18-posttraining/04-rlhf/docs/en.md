# RLHF mechanics: a frozen reward and PPO policy updates

Phase 18: Post-training: SFT, LoRA, reward models and RLHF · about 105 minutes · CPU

## What you will be able to do

- Keep three identities separate
- Collect actions before optimizing them
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

After fitting a reward model, how do its scores change a policy? We will collect actual sampled actions, freeze the behavior probabilities for each rollout batch and inspect improvement together with drift from a fixed reference.

## The idea

This lesson isolates RLHF mechanics in a small categorical bandit. A trainable policy is compared with rollout-time probabilities and a fixed reference, while a previously fitted reward model supplies frozen scores. Each role has a different update lifetime.

## Keep the four policy-and-reward roles distinct

The old policy probabilities stay fixed during the inner PPO updates for one rollout batch. They refresh when a new rollout is collected. The reference policy remains fixed across those rollouts and defines the KL comparison. The reward model is frozen during policy optimization.

Advantages and old probabilities should not quietly become new optimization targets during the inner update. This fixture uses an exact baseline and categorical KL; it does not contain a learned language-model critic or human-feedback collection pipeline.

Read reward, KL and action-probability panels together. Higher reward accompanied by a larger reference deviation is a tradeoff, not an unqualified success. Keep each plotted quantity tied to the actual policy state at measurement.

### Different objects have different update lifetimes

**Predict:** Which frozen object refreshes between rollout batches: old policy probabilities or the reference policy?

![Different objects have different update lifetimes](../outputs/mechanism.svg)

*Conceptual / analytic teaching diagram; not a recorded benchmark.*

Current parameters change within PPO updates. Old probabilities stay fixed for that rollout batch and refresh next rollout. The reference and reward remain fixed during policy training. This categorical fixture has an exact baseline, so no learned critic is shown.

### Pause and reason

Which frozen object refreshes between rollout batches: old policy probabilities or the reference policy?

<details><summary>Compare your reasoning</summary>

Old rollout-time probabilities refresh. The reference remains fixed under this lesson's contract. Confusing them changes the optimization objective.

</details>

## Before coding: identify four different objects

Keep a notebook table with current policy, old behavior policy, reference policy and reward model. The current policy changes on optimizer steps. The old policy is a snapshot taken before collecting the batch; its sampled-action log probabilities stay fixed while reusing that batch. The reference and reward model stay fixed for the whole experiment. Confusing old and reference makes even a numerically stable loop implement the wrong objective.

This lab is a bandit: one prompt, three terminal actions and no sequence credit assignment. It lets us calculate expectations exactly. Language PPO additionally needs token masks, rollout termination, value/advantage estimation and a generation pipeline; none of those can be inferred from this loop.

## Keep three identities separate

The reward model scores outputs. The reference policy anchors the desired deviation penalty and stays fixed. The old behavior policy generated the current batch and remains fixed only while that batch is reused. At the next rollout, old behavior updates to the new policy; the reference does not.

## Collect actions before optimizing them

The code samples terminal response actions from the current categorical policy. It stores their old log probabilities and computes an advantage using the frozen reward minus that policy’s exact expected reward. This exact baseline is available only because the action space is tiny; a general language system needs a suitable baseline or value estimator.

## Work a signed clipping example

For advantage $A=2$, ratio $\rho=1.5$, and clip interval $[0.8,1.2]$, the products are $3$ and $2.4$, so the minimum is $2.4$. Further increasing this ratio no longer improves that positive-advantage surrogate term. For $A=-2$ and $\rho=0.5$, the products are $-1$ and $-1.6$, so the minimum is $-1.6$.

Clipping limits the reward for moving in the advantageous direction too far on these samples. It does not freeze all probability changes or prove a global trust region. The explicit KL term can still contribute gradients when a surrogate term is clipped. Read the negative sign in the minimized loss carefully: we minimize negative surrogate plus a penalty.

## Why a baseline can help without changing the expected gradient

For sampled action $a$, the advantage here is reward minus the old policy’s exact expected reward. The baseline is independent of which action was sampled and is frozen for the update. Its expected contribution to the unclipped score-function gradient is zero because policy probabilities sum to one. Subtracting it can reduce sampling variance without changing that expected gradient.

Our three-action environment lets you enumerate this identity instead of trusting a noisy Monte Carlo comparison. The identity alone does not prove that arbitrary baselines, clipped multi-epoch surrogates or learned value estimators are unbiased. Preserve which estimator is being checked.

$$
\sum_a\pi(a)\nabla_\theta\log\pi(a)\,[r(a)-c]=\nabla_\theta\sum_a\pi(a)r(a)
$$

## Clip the surrogate with the advantage sign intact

Form a probability ratio from current and old log probabilities. Take the minimum of unclipped and clipped advantage products. Negative advantages reverse which bound matters; clipping only the ratio and maximizing it indiscriminately changes the algorithm. Advantages and behavior probabilities are detached.

$$
L=-\mathbb E\left[\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)\right]+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
$$

## Measure reward and reference drift together

For this categorical bandit, compute the KL and expected learned reward exactly by summing over every action. The trainer includes this exact KL directly in the objective; it is not an implementation of every token-level reward-shaping variant. A higher reward with greater KL may be expected, and clipping does not guarantee a global trust region.

## Do not confuse reward optimization with alignment

The model can exploit a weak reward model. Keep independent evaluations, inspect unfavorable cases and fix reward/reference versions during a comparison. Actual human-feedback systems need human data provenance and held-out human judgments. This exercise demonstrates sampled policy optimization against a learned synthetic preference reward only.

## Use the exact solution as a bounded optimization reference

For a finite action space with positive reference probabilities, maximizing expected fixed reward minus $\beta$ times forward KL has optimum proportional to reference probability times $\exp(r/\beta)$. This is a reference for the regularized bandit problem, not a statement that forty sampled PPO iterations must reach it. Compute it with log-softmax for numerical stability.

Evaluate the regularized objective of your trained policy and this optimum. A gap can reflect finite optimization, clipping and sampled rollouts. A higher learned reward alone can come from excessive drift. Keep an independent task score and adversarial cases before carrying this pattern to a real reward model.

$$
\pi^*(a)=\frac{\pi_{\mathrm{ref}}(a)e^{r(a)/\beta}}{\sum_b\pi_{\mathrm{ref}}(b)e^{r(b)/\beta}}
$$

## 1. Define the reward likelihood and signed PPO loss

Create main.py in your activated course environment. Paste this block, then run python main.py; function definitions alone print nothing.

```python
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
```

Keep reward fitting and policy loss separate. Inside policy loss, old probabilities and advantages are constants for differentiation.

## 2. Fit and freeze reward, then identify reference state

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
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
```

The reference is stored before policy updates. This step also creates the sampling key and independent reward model.

## 3. Collect rollouts and reuse each batch correctly

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
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
```

Each outer iteration collects a fresh batch; each inner step reuses its saved behavior probabilities. Plot points occur after a rollout batch, unlike pre-update loss curves in other lessons.

## Run the example

```python
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

```

Expected: Sampled PPO updates increase exact expected learned reward while the reference stays unchanged.

## RLHF mechanics: a frozen reward and PPO policy updates — recorded experiment

**Predict:** Which action gains probability as reward rises, and why does that also change reference KL?

![RLHF mechanics: a frozen reward and PPO policy updates — recorded experiment](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The top panel shows exact expected learned reward after each rollout batch, rising toward the highest-scored action. The bottom panel shows exact KL in nats to the fixed reference. The final reward is about $1.92$, with KL about $0.78$. The axes are different quantities and must not be added as if they shared units.

The third panel compares reference and final action probabilities; each series sums to one. Probability moves toward action zero, the response favored by the learned reward. The other actions lose probability, explaining the rising expected reward and KL in the first two panels. Reward is measured in the learned score scale, KL in nats and these bars in probability: do not combine their heights. This is a finite synthetic policy distribution, not a measure of human approval.

### Connect it to the computation

The policy concentrates on the action favored by the learned reward, which also moves it away from the reference. These deterministic expectation measurements evaluate a policy trained from random rollouts; they are not sampled human preference scores or a safety assessment.

```python
visual_data={'panels':[{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'expected learned reward','series':[{'label':'exact policy expectation','x':list(range(1,len(history)+1)),'y':history}]},{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'KL to fixed reference (nats)','series':[{'label':'exact categorical KL','x':list(range(1,len(kl_history)+1)),'y':kl_history}]}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

extra_panel={'kind':'bar','x':[0,1,2],'labels':['action 0','action 1','action 2'],'series':[{'label':'fixed reference','y':np.asarray(jnp.exp(reference)).tolist()},{'label':'trained policy','y':np.asarray(jnp.exp(logp)).tolist()}],'xlabel':'terminal response action','ylabel':'probability','title':'How increased reward redistributes action probability'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

```

## Recorded reference execution

CPU run: 2026-10-06T22:04:55.833597+00:00. JAX 0.9.2.

```text
Reward initial/final and final KL: 0.28191882371902466 1.9185447692871094 0.7824773192405701
Synthetic preference bandit with actual sampled PPO updates; no human ratings or language-generation claim.
Reward initial/final and final KL: 0.28191882371902466 1.9185447692871094 0.7824773192405701
Synthetic preference bandit with actual sampled PPO updates; no human ratings or language-generation claim.
Clipped surrogate terms: [ 2.4 -1.6]
Enumerated baseline score gradient matches exact expected-reward gradient.
Reference/final KL: 0.0 0.7824773192405701
Behavior log probabilities and advantages are detached.
Regularized objective reference/trained/optimum: 0.28191882371902466 1.7620493173599243 1.7987911701202393
PASS: posttraining-04

```

## Inspect both clipping directions

**Predict before running:** For a positive advantage with ratio above the upper bound, and a negative advantage below the lower bound, which product is retained?

```python
ratios=np.array([1.5,.5]);advantages=np.array([2.,-2.])
terms=np.minimum(ratios*advantages,np.clip(ratios,.8,1.2)*advantages)
np.testing.assert_allclose(terms,[2.4,-1.6])
print('Clipped surrogate terms:',terms)
```

**Expected:** The terms are [2.4, -1.6].

The negative-advantage case selects the more negative product. This is why clipping must be applied inside the signed surrogate.

## Enumerate the baseline cancellation

**Predict before running:** Does subtracting a fixed baseline change the exact policy gradient before clipping?

```python
probe=jnp.array([.3,-.2,.1]);probe_probs=jax.nn.softmax(probe)
constant=float(jnp.sum(probe_probs*rewards))
score_jacobian=jax.jacrev(jax.nn.log_softmax)(probe)
enumerated=jnp.sum(probe_probs[:,None]*(rewards-constant)[:,None]*score_jacobian,axis=0)
exact=jax.grad(lambda logits:jnp.sum(jax.nn.softmax(logits)*rewards))(probe)
np.testing.assert_allclose(enumerated,exact,atol=1e-6)
print('Enumerated baseline score gradient matches exact expected-reward gradient.')
```

**Expected:** The two gradients agree up to float32 rounding.

Enumeration removes sampling error. This tests the unclipped estimator identity; it does not assert that every clipped update equals exact gradient ascent.

## Make it yours

Check the KL at the fixed reference and compare it with the final policy’s exact KL.

<details><summary>Reference solution</summary>

```python
zero_kl=float(jnp.sum(jnp.exp(reference)*(reference-reference)))
assert zero_kl==0. and kl_history[-1]>zero_kl
print('Reference/final KL:',zero_kl,kl_history[-1])
```

</details>

## Verify detached behavior information

**Transfer**

Differentiate the objective with respect to supplied old log probabilities and advantages; both must be fixed within this update.

<details><summary>Hint</summary>

Use argnums to inspect only those two inputs.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
old_grad,adv_grad=jax.grad(ppo_loss,argnums=(1,3))(theta,old_selected,actions,advantage,reference,.2,.2)
np.testing.assert_array_equal(old_grad,np.zeros(old_grad.shape))
np.testing.assert_array_equal(adv_grad,np.zeros(adv_grad.shape))
print('Behavior log probabilities and advantages are detached.')
```

Detaching is not enough if you recompute old probabilities after every optimizer step. Store their values at collection time and verify identity across all epochs on that batch.

</details>

## Compare with the finite-action regularized optimum

**Challenge**

Compute the analytic optimum for beta 0.2, then compare its regularized objective with the trained policy and the reference.

<details><summary>Hint</summary>

Evaluate expected reward minus beta times KL for all three policies, and compute the optimum in log space.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
beta=.2
optimal_logp=jax.nn.log_softmax(reference+rewards/beta)
def regularized(logp):return jnp.sum(jnp.exp(logp)*rewards)-beta*jnp.sum(jnp.exp(logp)*(logp-reference))
optimum=float(regularized(optimal_logp));trained=float(regularized(logp));baseline=float(regularized(reference))
assert optimum>=trained-1e-5 and optimum>=baseline-1e-5
print('Regularized objective reference/trained/optimum:',baseline,trained,optimum)
```

An independently derived optimum bounds this finite-action objective. It does not qualify language generation or the reward model’s judgment.

</details>

## Check your understanding

Which policy is refreshed for the next rollout batch?

1. The behavior policy follows the current policy; the reference policy remains frozen.
2. A lower training loss by itself proves the full application is ready.
3. Matching shapes alone establishes the required behavior.

<details><summary>Answer and explanation</summary>

The behavior policy follows the current policy; the reference policy remains frozen.

Behavior identity belongs to the collected data. Reference identity belongs to the regularized objective.

</details>

## Diagnose the result

If ratios always equal one during repeated updates, inspect whether old log probabilities are being recomputed. If reward grows while external quality falls, investigate reward exploitation instead of declaring alignment.

## Carry forward

- For advantage $A=2$, ratio $\rho=1.5$, and clip interval $[0.8,1.2]$, the products are $3$ and $2.4$, so the minimum is $2.4$. Further increasing this ratio no longer improves that positive-advantage surrogate term. For $A=-2$ and $\rho=0.5$, the products are $-1$ and $-1.6$, so the minimum is $-1.6$.
- For a finite action space with positive reference probabilities, maximizing expected fixed reward minus $\beta$ times forward KL has optimum proportional to reference probability times $\exp(r/\beta)$. This is a reference for the regularized bandit problem, not a statement that forty sampled PPO iterations must reach it. Compute it with log-softmax for numerical stability.

## Keep your evidence

Keep reward/reference identity, collected old probabilities, signed clipping checks, exact reward/KL history and detached behavior information.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [InstructGPT and RLHF](https://arxiv.org/abs/2203.02155)
- [PPO](https://arxiv.org/abs/1707.06347)

