# Supervised fine-tuning with response-only token loss

Phase 18: Post-training: SFT, LoRA, reward models and RLHF · about 75 minutes · CPU

## What you will be able to do

- Align predictions with their target tokens
- Separate the attention mask from the loss mask
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

A conversation contains instructions and an answer. If we want to teach answer behavior, which tokens should receive loss? We will shift targets explicitly and exclude prompt tokens without hiding the prompt from the model.

## The idea

Supervised fine-tuning learns from demonstrations. The example uses a tiny next-token table so the alignment is visible: prompt IDs determine answer IDs, followed by an end token. In a pretrained Transformer, the same response-mask bookkeeping surrounds a much richer causal forward pass.

## Align predictions with their target tokens

Logits at position $t$ predict token $t+1$. Therefore shift the role mask with the target IDs. A prompt position can produce a supervised logit when its next token begins the answer. Masking logits according to the input role instead would drop that first answer token.

## Separate the attention mask from the loss mask

The attention mask controls what the model sees; the response mask controls which predictions are scored. Prompt tokens must remain available as causal context even when predicting them is not the objective. Keep padding and packed-document boundaries out of valid targets. The final input position in this fixture has no next token and contributes no loss.

## Choose a reduction and keep its denominator

This lab sums negative log probabilities of response and end tokens, then divides by the total supervised token count. That is a token-weighted loss: longer answers have greater total influence. Equal weighting per sequence is a different objective. Decide deliberately and retain both sequence and token counts.

$$
L_{\mathrm{SFT}}=-\frac{\sum_{b,t}m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})}{\sum_{b,t}m_{b,t+1}}
$$

## Fine-tune an identified base model

In a real run record the base checkpoint, tokenizer, chat template, data revision and special-token configuration. Demonstrations may contain personal or sensitive text; use data with appropriate permission and review. Split by source or conversation identity before formatting. This executable table starts from uniform logits to isolate the objective; the LoRA lesson separately trains and freezes a base model before adaptation.

## Check behavior and forgetting separately

A falling demonstration loss is not an instruction-following benchmark. Freeze a validation prompt set, compare answer quality under fixed decoding, and retain a base-task evaluation to detect forgetting. Do not tune prompt formatting or training duration against final test prompts.

## Run the example

```python
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

```

Expected: Response-only loss decreases from log(5) to below 0.15.

## Supervised fine-tuning with response-only token loss — recorded experiment

**Predict:** Predict what should change during training and what this curve cannot establish.

![Supervised fine-tuning with response-only token loss — recorded experiment](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The curve shows response-token mean negative log likelihood before each update. Uniform predictions start at $\log 5\approx1.609$, then fall to about $0.080$. It is the training loss of a five-token table, not a natural-language quality score.

### Connect it to the computation

The table learns prompt-to-answer and answer-to-end transitions. The independent gradient check confirms the intended target shift, and the mask exercise changes which target positions count without removing prompt context.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

```

## Recorded reference execution

CPU run: 2026-10-06T01:27:43.307456+00:00. JAX 0.9.2.

```text
Response-only token NLL initial/final: 1.6094379425048828 0.07985548675060272
Prompt rows learn to predict first answers; the final end-token row has no target.
Two first-response targets: 0.07894014567136765
No-target SFT batch rejected
PASS: posttraining-01

```

## Prove the target-mask shift

**Predict before running:** Which table rows should get gradients when only answer and end tokens are supervised?

```python
g=jax.grad(sft_objective)(jnp.zeros((5,5)))
assert np.linalg.norm(np.asarray(g)[0])>0 and np.linalg.norm(np.asarray(g)[1])>0
np.testing.assert_array_equal(np.asarray(g)[4],0.)
print('Prompt rows learn to predict first answers; the final end-token row has no target.')
```

**Expected:** Rows for prompt IDs have nonzero gradients; the unused final-token row has zero gradient.

The prediction made at the prompt position is supervised because its target is an answer. Input-role masking would confuse context with target eligibility.

## Make it yours

Score only the first response token in each sequence and compare the resulting count and objective.

<details><summary>Reference solution</summary>

```python
first_only=roles.at[:,2].set(False)
value=-jnp.sum(response_logps(p[tokens],tokens,first_only))/jnp.sum(first_only[:,1:])
assert int(jnp.sum(first_only[:,1:]))==2
assert np.isfinite(float(value))
print('Two first-response targets:',float(value))
```

</details>

## Reject a batch with no answer targets

**Transfer**

Validate an all-prompt batch before computing the token average.

<details><summary>Hint</summary>

Use the shifted target mask, not the unshifted input mask.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
try:validate_mask(jnp.zeros_like(roles[:,1:]),tokens[:,1:].shape)
except ValueError:print('No-target SFT batch rejected')
else:raise AssertionError('empty supervision accepted')
```

An empty answer mask should trigger a clear skip/rejection policy before the update; a fabricated zero objective disguises lost training data.

</details>

## Check your understanding

Should prompt inputs disappear when using response-only loss?

1. No. They remain context; only their target eligibility changes.
2. A lower training loss by itself proves the full application is ready.
3. Matching shapes alone establishes the required behavior.

<details><summary>Answer and explanation</summary>

No. They remain context; only their target eligibility changes.

Attention visibility and supervised output positions are separate contracts.

</details>

## Diagnose the result

When the first answer token fails to learn, inspect the shifted mask. When answer length changes the metric, compare token versus sequence weighting and retain valid target counts.

## Keep your evidence

Keep shifted target/role masks, valid response-token counts, gradient support and the changed target-selection result.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [InstructGPT: supervised demonstrations and preference training](https://arxiv.org/abs/2203.02155)

