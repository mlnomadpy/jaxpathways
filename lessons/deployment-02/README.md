# Adapt a pretrained model and choose a post-training objective

Phase 15: Deployment, interoperability & edge AI · about 125 minutes · CPU

## What you will be able to do

- Pretrain and reload an artifact with recorded data and checkpoint hashes
- Compare supervised adaptation and teacher distillation from the same starting weights
- Verify both objective gradients independently
- Report target-task gains and source-task retention without leaking held-out data

## The problem

A model already learned one task. You now have a small adaptation set and a teacher that supplies probabilities. Should you optimize the observed labels or imitate the teacher? We will actually pretrain, save and reload a tiny model, adapt it both ways, and compare the resulting predictions on an untouched evaluation split. The model is deliberately small so every objective and derivative can be inspected.

## The idea

Post-training changes a learned model using a new objective or dataset. Supervised fine-tuning uses observed targets. Distillation uses teacher predictions as targets; it inherits the teacher’s strengths and errors. Preference learning instead compares alternatives, while reinforcement learning requires a reward and interaction or rollout assumptions. These are different data contracts, not interchangeable labels for an optimizer.

## Define the tasks before choosing an objective

The input has two measured features plus a constant intercept feature. The source task uses one synthetic probability rule; the target task uses another. Source training sees $256$ rows. Adaptation sees only $48$ different rows. The final target report uses $512$ separately generated rows. These are disjoint generated splits with separate keys, not a real benchmark. Binary adaptation labels are sampled from the target probabilities; the teacher supplies the probabilities themselves. That privileged teacher is declared explicitly rather than presented as a generally superior model.

## Understand hard labels and soft targets

For logit $z=x^\top w$, cross-entropy is $\log(1+e^z)-tz$, where $t$ is either a binary observed label or a teacher probability between zero and one. Differentiating gives $(\operatorname{sigmoid}(z)-t)x$. Averaging over rows gives the gradient in our host reference. A soft target contains uncertainty that a single sampled binary label does not. If the teacher is wrong, faithfully matching that uncertainty can also preserve its mistakes.

$$
L(w)=\frac1N\sum_i[\operatorname{softplus}(x_i^\top w)-t_i x_i^\top w],\qquad \nabla L=\frac1N X^\top(\operatorname{sigmoid}(Xw)-t)
$$

## Make pretrained mean previously trained

We start from zero, optimize the source objective for $300$ updates and check that the source loss falls. We save those resulting weights, hash the checkpoint bytes and record the source-array hash, objective, dtype, version and training recipe. Adaptation uses only the reloaded artifact. This establishes a small reproducibly pretrained fixture; it is not a foundation model, a downloaded checkpoint or a random frozen feature masquerading as pretraining. The artifact omits optimizer state, so no exact-resume claim is made.

## Choose a fair comparison

Both adaptation branches receive identical initial weights, adaptation inputs, update count and rate. Their training losses are not directly a competition because their targets differ. We evaluate both on one common target cross-entropy and probability squared error, using held-out generating probabilities unavailable to training. We also evaluate source loss to expose forgetting. Do not select a learning rate using this final report; add a separate validation split for model selection. One seeded synthetic result does not establish which objective will win on another task.

## Locate preference and reinforcement objectives

A pairwise preference objective asks which of two responses is preferred; its supervision consists of prompt, chosen response and rejected response, with a defined sampling and reference policy. A teacher cross-entropy uses distributions over labels or tokens and does not require preference pairs. An RL objective uses a reward and a policy whose generated actions affect the training samples. This lab executes supervised adaptation and distillation only. Renaming teacher probabilities “rewards” would not create an RL experiment.

## Carry the contract to a real pretrained model

Choose an actual model checkpoint under its license, record repository and exact revision, tokenizer or feature preprocessing, architecture configuration, file hashes and base precision. First verify a golden inference example before changing weights. Build disjoint training, validation and final evaluation sets, with no overlap from teacher-generation or calibration data. Start with a small controlled adapter or declared trainable parameter subset; record what remains frozen and save adapter/base identities together. Define the task and source-retention criteria before optimization. Recheck export, precision and target-runtime parity after adaptation. This extension is an execution plan, not a claim that a large external model ran in this CPU lesson.

## 1. Pretrain a small model and retain its source data identity

Create main.py in the CPU environment. These are explicit synthetic classification tasks; the source model is genuinely optimized before it is saved.

```python
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

def design(key,n):
    features=jax.random.normal(key,(n,2))
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
source_X=design(jax.random.key(10),256)
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
adapt_X=design(jax.random.key(11),48)
held_X=design(jax.random.key(12),512)
teacher=jnp.array([.6,1.4,-.3])
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
held_prob=jax.nn.sigmoid(held_X@teacher)
def objective(w,X,targets):
    logits=X@w
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
def train(initial,X,targets,steps=300,rate=.15):
    def step(w,_):
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        return w-rate*grad,loss
    return jax.lax.scan(step,initial,None,length=steps)
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
assert pretrain_history[-1]<pretrain_history[0]-.1
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()
```

The same stable cross-entropy accepts binary labels or teacher probabilities. Those targets express different supervision, even though the algebra is shared.

## 2. Save and reload before adapting

Append this artifact boundary. The manifest names the model, synthetic source split, objective and training recipe; adaptation starts from reloaded weights.

```python
with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'pretrained.npz'
    np.savez(checkpoint,weights=np.asarray(base))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    manifest_path=Path(directory)/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    recorded=json.loads(manifest_path.read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    with np.load(checkpoint,allow_pickle=False) as saved:
        restored=jnp.asarray(saved['weights'])
np.testing.assert_array_equal(restored,base)
frozen=np.array(restored,copy=True)
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
np.testing.assert_array_equal(restored,frozen)
```

Both branches begin with identical pretrained parameters and a fresh stateless SGD update rule. This is adaptation, not exact optimizer-state resumption.

## 3. Compare on a held-out criterion shared by both objectives

Append the independent derivative and final evaluation. Neither optimization branch receives held_X or held_prob.

```python
w_check=jnp.array([.2,-.3,.1])
host_X=np.asarray(adapt_X,dtype=np.float64)
host_w=np.asarray(w_check,dtype=np.float64)
host_p=1/(1+np.exp(-(host_X@host_w)))
for targets in (hard_targets,soft_targets):
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
metrics={}
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
print(json.dumps(metrics,indent=2))
print('checkpoint source hash:',source_hash)
```

The teacher is the known generating rule in this fixture, so matching it is an especially favorable distillation setting. Report source retention as well as target-task improvement.

## Run the example

```python
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

def design(key,n):
    features=jax.random.normal(key,(n,2))
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
source_X=design(jax.random.key(10),256)
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
adapt_X=design(jax.random.key(11),48)
held_X=design(jax.random.key(12),512)
teacher=jnp.array([.6,1.4,-.3])
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
held_prob=jax.nn.sigmoid(held_X@teacher)
def objective(w,X,targets):
    logits=X@w
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
def train(initial,X,targets,steps=300,rate=.15):
    def step(w,_):
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        return w-rate*grad,loss
    return jax.lax.scan(step,initial,None,length=steps)
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
assert pretrain_history[-1]<pretrain_history[0]-.1
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()

with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'pretrained.npz'
    np.savez(checkpoint,weights=np.asarray(base))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    manifest_path=Path(directory)/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    recorded=json.loads(manifest_path.read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    with np.load(checkpoint,allow_pickle=False) as saved:
        restored=jnp.asarray(saved['weights'])
np.testing.assert_array_equal(restored,base)
frozen=np.array(restored,copy=True)
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
np.testing.assert_array_equal(restored,frozen)

w_check=jnp.array([.2,-.3,.1])
host_X=np.asarray(adapt_X,dtype=np.float64)
host_w=np.asarray(w_check,dtype=np.float64)
host_p=1/(1+np.exp(-(host_X@host_w)))
for targets in (hard_targets,soft_targets):
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
metrics={}
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
print(json.dumps(metrics,indent=2))
print('checkpoint source hash:',source_hash)
```

Expected: Both adapted branches reduce the common held-out target cross-entropy relative to the reloaded source model. The printed table also reports probability error and source-task retention; inspect all three quantities.

## Two adaptation objectives evaluated on the same held-out target

**Predict:** Will improving the new task necessarily preserve performance on the source task?

![Two adaptation objectives evaluated on the same held-out target](../../phases/15-deployment/02-adapt-a-pretrained-model-and-choose-a-post-training-objective/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal categories are the reloaded pretrained model, supervised adaptation and teacher distillation. The vertical axis is mean cross-entropy in nats. One series evaluates the same held-out target probabilities for every model; the other evaluates source-task probabilities. Lower is better within each series. The two adapted models improve target-task loss while their source-task loss increases. These bars are evaluations of the resulting weights, not the branches’ incomparable training objectives. In this seeded run the held-out target bars are approximately $0.978,0.555,0.524$, while the source bars are $0.545,0.790,1.005$, in the same model order. Teacher adaptation gives the lowest target loss here but also the largest source-retention loss.

### Connect it to the computation

The teacher exactly matches the declared synthetic target generator, making this an especially favorable distillation case. The held-out comparison demonstrates an objective tradeoff on this fixture. It neither establishes that a real teacher is correct nor that one adaptation recipe is generally best. The failure experiment makes the teacher wrong and rechecks the target criterion.

```python
names=["pretrained","supervised","teacher"]
visual_data={"kind":"bar","x":[0,1,2],"labels":names,"xlabel":"model after training stage","ylabel":"mean cross-entropy (nats)","series":[{"label":"held-out target","y":[metrics[n]["held_cross_entropy"] for n in names]},{"label":"source retention","y":[metrics[n]["source_cross_entropy"] for n in names]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:26:10.922278+00:00. JAX 0.9.2.

```text
{
  "pretrained": {
    "held_cross_entropy": 0.9780182838439941,
    "held_brier": 0.17108705639839172,
    "source_cross_entropy": 0.5451400876045227
  },
  "supervised": {
    "held_cross_entropy": 0.5554870367050171,
    "held_brier": 0.012093935161828995,
    "source_cross_entropy": 0.7904530763626099
  },
  "teacher": {
    "held_cross_entropy": 0.5240296125411987,
    "held_brier": 0.00022982103109825402,
    "source_cross_entropy": 1.0046310424804688
  }
}
checkpoint source hash: f5fb50c38195b46766bee2edaa81a7ad28386523baa850c5294eb71b47f1f84c
{
  "pretrained": {
    "held_cross_entropy": 0.9780182838439941,
    "held_brier": 0.17108705639839172,
    "source_cross_entropy": 0.5451400876045227
  },
  "supervised": {
    "held_cross_entropy": 0.5554870367050171,
    "held_brier": 0.012093935161828995,
    "source_cross_entropy": 0.7904530763626099
  },
  "teacher": {
    "held_cross_entropy": 0.5240296125411987,
    "held_brier": 0.00022982103109825402,
    "source_cross_entropy": 1.0046310424804688
  }
}
checkpoint source hash: f5fb50c38195b46766bee2edaa81a7ad28386523baa850c5294eb71b47f1f84c
wrong-teacher held loss: 1.3151594400405884
source retention losses: [0.5451400876045227, 0.7904530763626099, 1.0046310424804688]
PASS: deployment-02

```

## Check a stable extreme logit

**Predict before running:** Can a logit of $1000$ be evaluated without exponentiating it directly?

```python
extreme=objective(jnp.array([1000.]),jnp.ones((1,1)),jnp.zeros(1))
assert jnp.isfinite(extreme) and jnp.allclose(extreme,1000.)
```

**Expected:** The wrong confident prediction has finite loss $1000$.

logaddexp evaluates softplus stably; a naive log(1+exp(z)) can overflow.

## Reveal teacher error

**Predict before running:** If a teacher flips every target probability, does matching it still solve the original task?

```python
wrong,_=train(restored,adapt_X,1-soft_targets,steps=200)
assert objective(wrong,held_X,held_prob)>objective(distilled,held_X,held_prob)
print("wrong-teacher held loss:",float(objective(wrong,held_X,held_prob)))
```

**Expected:** The deliberately wrong teacher produces worse target evaluation than the correct teacher.

Distillation optimizes fidelity to its teacher, which is not automatically fidelity to the desired task.

## Make it yours

Adapt for zero steps and verify that predictions exactly match the reloaded checkpoint. Then run a short adaptation and show that the saved base is unchanged while the adapted parameters differ.

<details><summary>Reference solution</summary>

```python
unchanged,_=train(restored,adapt_X,hard_targets,steps=0)
np.testing.assert_array_equal(unchanged,restored)
short,_=train(restored,adapt_X,hard_targets,steps=10)
assert not jnp.array_equal(short,restored)
np.testing.assert_array_equal(restored,frozen)
```

</details>

## Keep source-task evidence

**Transfer / diagnosis**

Measure source cross-entropy before and after both adaptation objectives. Explain any forgetting using the different source and target probability rules.

<details><summary>Hint</summary>

Report the actual values; target improvement is not a source-retention guarantee.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
retention=[float(objective(w,source_X,source_targets)) for w in (restored,sft,distilled)]
assert retention[1]>retention[0] and retention[2]>retention[0]
print("source retention losses:",retention)
```

The two tasks prefer different parameters; the measurement exposes that tradeoff.

</details>

## Check the loss without autodiff

**Transfer / diagnosis**

Compare the soft-target loss with a host float64 logaddexp expression on new weights.

<details><summary>Hint</summary>

Use the same logits, reduction and targets; this check does not call the JAX loss as its oracle.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
probe=np.array([-.4,.2,.3])
z=host_X@probe
expected=np.mean(np.logaddexp(0,z)-np.asarray(soft_targets)*z)
np.testing.assert_allclose(objective(jnp.asarray(probe),adapt_X,soft_targets),expected,rtol=1e-6)
```

Independent algebra checks normalization and reduction as well as stability.

</details>

## Check your understanding

Why should the two adaptation branches be compared using a common held-out criterion rather than their own training losses?

1. Teacher targets always make training loss zero
2. Their supervision differs, so their training losses answer different questions
3. Using the evaluation split during every update guarantees fairness

<details><summary>Answer and explanation</summary>

Their supervision differs, so their training losses answer different questions

The common held-out task defines the comparison. Different training targets can yield different loss values even when prediction quality is similar; evaluating the final split during model selection would also leak information.

</details>

## Diagnose the result

If target performance improves but source loss rises, inspect task conflict before blaming checkpoint corruption. If the initial adapted output differs before any update, compare checkpoint hash, preprocessing and precision. If a distillation result looks implausibly strong, verify that teacher labels and evaluation labels were not generated from the same held-out examples.

## Carry forward

- A post-training comparison begins with an identifiable trained artifact.
- Choose an objective according to the supervision actually available.
- Compare target quality, source retention and provenance before exporting a new version.

## Keep your evidence

Source/checkpoint hashes, independent loss and gradient, disjoint evaluation results and teacher-failure diagnosis. Keep the environment, observed outputs and your explanation of the figure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Knowledge distillation: original paper](https://arxiv.org/abs/1503.02531)
- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)

