# Masked language modeling: predict hidden tokens

Phase 17: Self-supervised pretraining: masked and contrastive learning · about 75 minutes · CPU

## What you will be able to do

- Separate the clean targets from the corrupted input
- Normalize over selected target positions
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

A text model can see a sentence but one word is hidden. What prevents it from copying the answer, and which positions should contribute to its loss? We will train a small bidirectional attention model and audit those two boundaries.

## The idea

Masked language modeling predicts selected original tokens from a corrupted input. It differs from causal next-token prediction: allowed context can come from both sides. Our repeated-token fixture makes the correct answer inspectable; this is a small objective and attention experiment, not a BERT reproduction or a language benchmark.

## Separate the clean targets from the corrupted input

Keep an immutable copy of the original token IDs. The selection mask identifies positions to predict; corruption changes the input tokens, not the targets. Here the extra token ID $3$ means hidden, while output classes are $0,1,2$. Passing clean targets into the encoder would leak the answer. Check the corrupted arrays before the first update.

## Normalize over selected target positions

Compute log-softmax over vocabulary, gather the log probability of each original target, then average only selected positions. For uniform predictions over three classes, the initial loss is $\log 3$, regardless of how many valid positions are selected. Reject an empty mask before entering the jitted step; silently dividing by zero creates an undefined objective.

$$
L=-\frac{\sum_{b,t}m_{bt}\log p_\theta(x_{bt}\mid\widetilde x_b)}{\sum_{b,t}m_{bt}}
$$

## Use both sides without using the hidden value

The encoder forms attention scores across every input position. There is no causal triangle: a later visible token is useful evidence here. This tiny model has one attention head, no positional encoding and a linear vocabulary head. Repeated token sequences deliberately avoid word-order ambiguity; a realistic encoder also needs positions, padding handling, residual blocks and a tokenizer.

## Distinguish selection from replacement policy

This lab always replaces selected tokens with the mask ID so information flow is easy to inspect. BERT-style training can mix mask replacement, random replacement and unchanged selected tokens; selection still determines the supervised positions. Keep special/padding tokens ineligible. Track a masking key and regenerate masks under a stated schedule rather than accidentally fixing one easy corruption forever.

## Test a different corruption before claiming progress

The training curve uses one selected position. The code evaluates a second position without additional updates and checks that changing target labels changes the loss. This supports the bounded repeated-token task, not language understanding. A real corpus needs document-separated splits, a frozen tokenizer, mask provenance and downstream evaluation against random initialization.

## Run the example

```python
import jax
import jax.numpy as jnp
import numpy as np

def masked_ce(logits, targets, selected):
    # Caller validates a nonempty mask before a transformed training step.
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(selected, nll, 0.)) / jnp.sum(selected)

def validate_mask(selected, shape):
    a = np.asarray(selected)
    if a.shape != shape or a.dtype != np.bool_ or not a.any():
        raise ValueError('a nonempty Boolean mask of the target shape is required')

def corrupt_tokens(tokens, selected, mask_id):
    validate_mask(selected, tokens.shape)
    return jnp.where(selected, mask_id, tokens)

def mlm_logits(p, corrupted):
    h = p['embedding'][corrupted]
    # Bidirectional single-head attention, deliberately no causal mask.
    scores = h @ jnp.swapaxes(h, -1, -2) / jnp.sqrt(h.shape[-1])
    context = jax.nn.softmax(scores, axis=-1) @ h
    return context @ p['head']

tokens = jnp.array([[0,0,0,0],[1,1,1,1],[2,2,2,2]],jnp.int32)
selected = jnp.array([[False,True,False,False]]*3)
corrupted = corrupt_tokens(tokens,selected,3)
key = jax.random.key(7)
p = {'embedding':jax.random.normal(key,(4,6))*.2,'head':jnp.zeros((6,3))}
loss = lambda p: masked_ce(mlm_logits(p,corrupted),tokens,selected)
step = jax.jit(jax.value_and_grad(loss)); history=[]
for _ in range(100):
    value,g = step(p); history.append(float(value)); p=jax.tree.map(lambda a,b:a-.4*b,p,g)
assert history[-1] < history[0]*.15
np.testing.assert_allclose(history[0],np.log(3),atol=1e-6)
# Changing the clean target after constructing corrupted input cannot change the forward pass.
changed_targets=tokens.at[:,1].set((tokens[:,1]+1)%3)
assert not np.isclose(float(masked_ce(mlm_logits(p,corrupted),changed_targets,selected)),history[-1])
held_mask=jnp.array([[False,False,True,False]]*3)
held_loss=float(masked_ce(mlm_logits(p,corrupt_tokens(tokens,held_mask,3)),tokens,held_mask))
assert held_loss < .2
print('MLM initial/final/changed-mask:',history[0],history[-1],held_loss)

```

Expected: Loss starts near 1.099 and ends below 0.02; an unseen mask position remains below 0.2.

## Masked language modeling: predict hidden tokens — recorded experiment

**Predict:** Predict what should change during training and what this curve cannot establish.

![Masked language modeling: predict hidden tokens — recorded experiment](../../phases/17-pretraining/01-mlm/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is the number of updates already applied; the vertical axis is selected-token cross-entropy in nats. The first value is about $1.099=\log 3$. It falls to about $0.011$ on the fixed synthetic sequences. This is training loss, not masked-token accuracy on natural text.

### Connect it to the computation

The zero vocabulary head makes initial predictions uniform. Falling loss shows the trained attention/embedding/head system can use the remaining repeated tokens. A held mask position is checked separately; the plot does not show a held-out corpus or establish transfer quality.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

```

## Recorded reference execution

CPU run: 2026-10-06T01:27:36.954907+00:00. JAX 0.9.2.

```text
MLM initial/final/changed-mask: 1.0986123085021973 0.010633035562932491 0.010457751341164112
Unselected output logits have zero direct loss gradient.
Changed position loss: 0.010457751341164112
Empty mask rejected
PASS: pretraining-01

```

## Show that unsupervised positions have no direct loss gradient

**Predict before running:** If we change logits only where the selection mask is false, does this masked objective change?

```python
logits=mlm_logits(p,corrupted)
changed_logits=jnp.where(selected[...,None],logits,logits+100.)
np.testing.assert_allclose(masked_ce(changed_logits,tokens,selected),masked_ce(logits,tokens,selected),atol=1e-6)
grad_logits=jax.grad(masked_ce)(logits,tokens,selected)
np.testing.assert_array_equal(np.asarray(grad_logits)[~np.asarray(selected)],0.)
print('Unselected output logits have zero direct loss gradient.')
```

**Expected:** Unselected logits do not affect the masked loss and have zero direct gradient.

Visible input tokens can still influence selected predictions through attention. Zero direct gradient at an unselected output is not a claim that visible-token embeddings never learn.

## Make it yours

Select the first position instead of the second and evaluate the trained model without updating it.

<details><summary>Reference solution</summary>

```python
first_mask=jnp.array([[True,False,False,False]]*3)
first_loss=float(masked_ce(mlm_logits(p,corrupt_tokens(tokens,first_mask,3)),tokens,first_mask))
assert first_loss<.2
print('Changed position loss:',first_loss)
```

</details>

## Reject a missing learning signal

**Transfer**

Pass an empty selection mask through the public corruption boundary and require a clear failure.

<details><summary>Hint</summary>

Validate outside the transformed objective before dividing by a count.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
try:corrupt_tokens(tokens,jnp.zeros_like(tokens,dtype=bool),3)
except ValueError:print('Empty mask rejected')
else:raise AssertionError('empty mask accepted')
```

A skipped batch requires an explicit training policy. It must not silently become a zero loss or a NaN update.

</details>

## Check your understanding

Why are visible input tokens useful even when their output positions are excluded from the loss?

1. They supply context to predictions at selected positions through the encoder.
2. A lower training loss by itself proves the full application is ready.
3. Matching shapes alone establishes the required behavior.

<details><summary>Answer and explanation</summary>

They supply context to predictions at selected positions through the encoder.

The mask controls the supervised outputs, while attention controls how input context influences them.

</details>

## Diagnose the result

If loss is suspiciously perfect at initialization, inspect target leakage and the selection count. If loss becomes NaN, reject empty masks and inspect vocabulary indices before changing the optimizer.

## Keep your evidence

Keep corrupted inputs, selected-target counts, the log-vocabulary baseline, independent masked-loss checks, trained parameters and changed-mask results.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [BERT: masked language pretraining](https://arxiv.org/abs/1810.04805)

