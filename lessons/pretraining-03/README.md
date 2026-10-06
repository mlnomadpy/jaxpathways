# Contrastive learning: views, positives and negatives

Phase 17: Self-supervised pretraining: masked and contrastive learning · about 75 minutes · CPU

## What you will be able to do

- Define the positive pair before augmenting
- Normalize embeddings and introduce temperature
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

Two transformed views belong to the same source example. Can an encoder place them close together without mapping every input to the same point? We will inspect exactly which pairs the objective treats as positives.

## The idea

The lab trains a shared linear encoder with a symmetric cross-view contrastive loss. Each row’s paired view is its positive and other rows are negatives. This resembles the paired objective used in cross-modal retrieval; it is not the full SimCLR denominator, which includes additional same-view negatives.

## Define the positive pair before augmenting

Keep stable source IDs through both view pipelines. The positive for row i must still be the matching source in the other view. Augmentations encode an invariance assumption: a transformation that changes the label can create a false positive. Here small fixed perturbations preserve four explicitly defined directions.

## Normalize embeddings and introduce temperature

Unit-normalize embeddings, compute dot products, and divide by a positive temperature $\tau$. Lower temperature sharpens the categorical comparison. It does not change which pair is correct, and it is not automatically better. A norm floor defines numerical behavior near zero, but zero embeddings are still a collapsed representation.

$$
L=-\frac{1}{2N}\sum_i\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]
$$

## Check both directions independently

A row asks which right-hand view belongs to a left-hand example; a column asks the reverse question. The code calculates both with a stable host log-sum-exp reference. Check that jointly permuting both views preserves the objective. Permuting only one view without remapping targets should change the task.

## Understand collapse and false negatives

If every embedding is the same, every candidate has equal probability and the loss is $\log N$. That is a useful baseline, not a proof that every gradient path can escape collapse. Distinct source examples may share semantics; treating duplicates as negatives can penalize useful similarity. The cross-modal harness adds a multi-positive objective for that case.

## Separate pretraining evidence from downstream quality

The line records the four-pair training objective, and retrieval is checked on those same pairs. Neither is a linear-probe evaluation. A full representation study freezes the encoder, fits a small supervised probe using training labels and evaluates a held-out split; alternatively compare fine-tuning with random initialization under the same budget. Keep projection-head and encoder outputs distinct.

## Run the example

```python
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

```

Expected: Training loss decreases and all four paired views are nearest neighbors.

## Contrastive learning: views, positives and negatives — recorded experiment

**Predict:** Predict what should change during training and what this curve cannot establish.

![Contrastive learning: views, positives and negatives — recorded experiment](../../phases/17-pretraining/03-contrastive/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The line is symmetric cross-view training loss in nats before each update. It falls from about $0.564$ to $0.014$. Lower values mean that the diagonal pair receives more probability among these four candidates, not that semantic accuracy is that number.

### Connect it to the computation

The shared encoder starts with poorly separated directions and learns an embedding where each view retrieves its paired source. The duplicate-pair exercise changes the denominator and reveals why a loss cannot be compared blindly across batch construction policies.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

```

## Recorded reference execution

CPU run: 2026-10-06T01:27:41.214768+00:00. JAX 0.9.2.

```text
Paired contrastive initial/final: 0.564157247543335 0.013556718826293945 ; all four nearest pairs correct
Collapsed baseline: 1.3862943649291992
Incorrect pair mapping loss: 7.493563652038574
Duplicated diagonal-only penalty: 0.6931472215801477
PASS: pretraining-03

```

## Measure the collapsed baseline

**Predict before running:** What loss should identical embeddings achieve for four candidate pairs?

```python
collapsed=jnp.ones((4,2))
collapse_loss=float(paired_contrastive(collapsed,collapsed))
np.testing.assert_allclose(collapse_loss,np.log(4),atol=1e-6)
print('Collapsed baseline:',collapse_loss)
```

**Expected:** The collapsed loss is log(4), approximately 1.386.

Equal similarity gives each candidate probability one quarter. This baseline catches objectives that reward only attraction and forget competing candidates.

## Make it yours

Permute only the right-hand embeddings after training and demonstrate the effect of broken pair identity.

<details><summary>Reference solution</summary>

```python
wrong_pair_loss=float(paired_contrastive(left@w,(right@w)[permutation]))
assert wrong_pair_loss>float(paired_contrastive(left@w,right@w))
print('Incorrect pair mapping loss:',wrong_pair_loss)
```

</details>

## Duplicate a source and inspect the objective

**Transfer**

Duplicate all four pairs while keeping diagonal-only positives. Calculate the change in loss and explain the false-negative effect.

<details><summary>Hint</summary>

Each original candidate appears twice, but only one copy is labeled positive.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
duplicated=float(paired_contrastive(jnp.tile(left@w,(2,1)),jnp.tile(right@w,(2,1))))
original=float(paired_contrastive(left@w,right@w))
np.testing.assert_allclose(duplicated-original,np.log(2),atol=1e-5)
print('Duplicated diagonal-only penalty:',duplicated-original)
```

The extra log(2) comes from duplicating each denominator candidate without adding the second copy to the positive set. Use source/semantic identity to define multi-positive supervision when appropriate.

</details>

## Check your understanding

What does successful retrieval on the training pairs establish?

1. The encoder solves those pairs under that pairing and augmentation contract.
2. A lower training loss by itself proves the full application is ready.
3. Matching shapes alone establishes the required behavior.

<details><summary>Answer and explanation</summary>

The encoder solves those pairs under that pairing and augmentation contract.

It does not establish downstream task quality or robustness to a new data distribution.

</details>

## Diagnose the result

If training stalls near the collapsed baseline, inspect norms and pairing before reducing temperature. Unexpected loss jumps after batching changes can reflect false negatives or a different candidate set.

## Keep your evidence

Keep source pairing, temperature, stable two-direction loss checks, collapse and duplicate baselines, training outputs and broken-pair diagnosis.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [SimCLR: views and representations](https://arxiv.org/abs/2002.05709)
- [CLIP: symmetric paired contrastive training](https://arxiv.org/abs/2103.00020)

