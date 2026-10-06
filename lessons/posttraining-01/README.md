# Supervised fine-tuning with response-only token loss

Phase 18: Post-training: SFT, LoRA, reward models and RLHF · about 105 minutes · CPU

## What you will be able to do

- Align predictions with their target tokens
- Separate the attention mask from the loss mask
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

A conversation contains instructions and an answer. If we want to teach answer behavior, which tokens should receive loss? We will shift targets explicitly and exclude prompt tokens without hiding the prompt from the model.

## The idea

Supervised fine-tuning teaches desired responses under a causal prediction objective. Response-only training scores selected target tokens while retaining prompt context. Correct token/role alignment determines which behavior the loss actually rewards.

## Shift the supervision with the target

Input position $t$ predicts target $t+1$. Therefore the response flag used for that loss must describe the target token. At a prompt/response boundary, using the input's flag instead can omit the first response target or score the wrong position.

A prompt position with no direct loss can still influence scored responses through causal attention. Its representations can participate in gradients. “Unscored” is not the same as “absent from computation.”

Aggregate by valid target tokens rather than averaging sequence or batch means blindly. The training plot demonstrates the fixture's masked objective; it does not establish instruction-following across unseen natural-language tasks.

### Pause and reason

Why can prompt-related parameters change during response-only training?

<details><summary>Compare your reasoning</summary>

The prompt influences scored response predictions through the computation. Removing direct loss at prompt positions does not remove all gradient paths through prompt context.

</details>

## Before coding: write the target table

For input tokens $[0,2,4]$, the model at the first position predicts $2$, and at the second predicts $4$. The last position has no target in this sequence. Mark the two targets as answer and end-of-answer, even though the first supervised prediction is made from a prompt token. Write input, next target and target-role mask as three aligned rows before touching the loss.

The model here is a transition table: each current token selects a row of vocabulary logits. It has no long-range attention and cannot distinguish two histories ending in the same token. This makes the shift easy to inspect while keeping the boundary between objective practice and instruction-following explicit.

## Align predictions with their target tokens

Logits at position $t$ predict token $t+1$. Therefore shift the role mask with the target IDs. A prompt position can produce a supervised logit when its next token begins the answer. Masking logits according to the input role instead would drop that first answer token.

## Separate the attention mask from the loss mask

The attention mask controls what the model sees; the response mask controls which predictions are scored. Prompt tokens must remain available as causal context even when predicting them is not the objective. Keep padding and packed-document boundaries out of valid targets. The final input position in this fixture has no next token and contributes no loss.

## Work through sums, counts and unequal answer lengths

Uniform probabilities over five tokens give negative log probability $\log 5$ at each supervised target. One supervised target contributes that amount; three targets contribute $3\log 5$. Dividing total negative log likelihood by total valid targets returns $\log5$ for either batch. The sum returned by response_logps is intentionally not already a mean.

Once probabilities differ, sequence weighting matters. If one answer has token loss $2$ and another has three losses of $1$, a token mean is $1.25$ while a sequence mean is $1.5$. During gradient accumulation, carry numerator and count. Averaging unequal microbatch means changes the training objective.

## Follow the first gradient to the prompt row

For a uniform five-class prediction with correct class $2$, softmax cross-entropy has derivative $-0.8$ for class $2$ and $0.2$ for each other class, before averaging across targets. The negative gradient increases the correct logit. The lookup row for prompt ID $0$ receives this signal because it predicts the first answer token. Excluding all prompt input positions would incorrectly discard that learning signal.

Do not confuse this with permission to inspect future answer tokens. A real causal Transformer must restrict attention independently of which targets count toward the loss. Changing the loss mask cannot repair a noncausal attention mask.

## Choose a reduction and keep its denominator

This lab sums negative log probabilities of response and end tokens, then divides by the total supervised token count. That is a token-weighted loss: longer answers have greater total influence. Equal weighting per sequence is a different objective. Decide deliberately and retain both sequence and token counts.

$$
L_{\mathrm{SFT}}=-\frac{\sum_{b,t}m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})}{\sum_{b,t}m_{b,t+1}}
$$

## Fine-tune an identified base model

In a real run record the base checkpoint, tokenizer, chat template, data revision and special-token configuration. Demonstrations may contain personal or sensitive text; use data with appropriate permission and review. Split by source or conversation identity before formatting. This executable table starts from uniform logits to isolate the objective; the LoRA lesson separately trains and freezes a base model before adaptation.

## Check behavior and forgetting separately

A falling demonstration loss is not an instruction-following benchmark. Freeze a validation prompt set, compare answer quality under fixed decoding, and retain a base-task evaluation to detect forgetting. Do not tune prompt formatting or training duration against final test prompts.

## Why the tiny table is not enough for instruction following

Two sequences ending in the same prompt token select exactly the same table row, so they must receive the same next-token distribution. If their intended answers differ because of earlier context, this model cannot represent the task. Demonstrate that collision before replacing the table with a Transformer.

For a real SFT experiment, retain a base checkpoint, tokenizer, chat template and source-group split. Test the exact formatted strings and role spans, compare fixed decoding on held-out prompts, and measure a retained base task for forgetting. Packed sequences require document attention boundaries as well as masks; padding must never become a target merely because it shares an end-token ID.

## 1. Define sequence log-probability sums

Create main.py in your activated course environment. Paste this block, then run python main.py; function definitions alone print nothing.

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
```

The slice removes the final logit and first token. The role mask is shifted with targets, preserving supervision of the first answer token.

## 2. Align prompt, response and target masks

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
# Token 0/1 is a prompt; 2/3 is its response; 4 marks the end.
tokens=jnp.array([[0,2,4],[1,3,4]],jnp.int32)
roles=jnp.array([[False,True,True]]*2)
validate_mask(roles[:,1:],tokens[:,1:].shape)
p=jnp.zeros((5,5));history=[]
def sft_objective(p):
    return -jnp.sum(response_logps(p[tokens],tokens,roles))/jnp.sum(roles[:,1:])
step=jax.jit(jax.value_and_grad(sft_objective))
```

Inspect tokens and roles together. Every row has two valid next-token targets, so the loss denominator is four.

## 3. Fit the transition table and check the final position

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
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

The table learns only local transitions. Its falling loss verifies these examples; the context-collision exercise exposes its architectural limit.

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

**Predict:** Predict the zero-gradient column before looking at the heatmap. Does a prompt input position necessarily have zero gradient?

![Supervised fine-tuning with response-only token loss — recorded experiment](../../phases/18-posttraining/01-sft/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The curve shows response-token mean negative log likelihood before each update. Uniform predictions start at $\log 5\approx1.609$, then fall to about $0.080$. It is the training loss of a five-token table, not a natural-language quality score.

The second panel is an initial-gradient audit, not another training curve. Rows are the two sequences and columns are positions producing logits. The first two columns have equal nonzero gradient norm because they predict answer and end tokens from uniform probabilities. The last column is zero because it has no next-token target. The first column belongs to a prompt input: its nonzero gradient is the evidence that the response mask was shifted with targets. This shows output-logit gradients, not all internal parameter gradients.

### Connect it to the computation

The table learns prompt-to-answer and answer-to-end transitions. The independent gradient check confirms the intended target shift, and the mask exercise changes which target positions count without removing prompt context.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'response-token NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

initial_output_grads=jax.grad(lambda values:-jnp.sum(response_logps(values,tokens,roles))/jnp.sum(roles[:,1:]))(jnp.zeros((2,3,5)))
extra_panel={'kind':'heatmap','values':np.asarray(jnp.linalg.norm(initial_output_grads,axis=-1)).tolist(),'rows':['prompt 0 / answer 2','prompt 1 / answer 3'],'columns':['input position 0','input position 1','input position 2'],'unit':'output-logit gradient L2 norm','xlabel':'position producing the prediction','ylabel':'sequence','title':'Initial supervised gradient by output position'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

```

## Recorded reference execution

CPU run: 2026-10-06T22:04:41.131338+00:00. JAX 0.9.2.

```text
Response-only token NLL initial/final: 1.6094379425048828 0.07985548675060272
Response-only token NLL initial/final: 1.6094379425048828 0.07985548675060272
Prompt rows learn to predict first answers; the final end-token row has no target.
Supervised targets per sequence: [2 3]
Two first-response targets: 0.07894014567136765
No-target SFT batch rejected
Identical final tokens force identical table predictions despite different contexts.
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

## Check a ragged batch against a scalar loop

**Predict before running:** Will independent token sums recover the response objective when answers have unequal lengths?

```python
ragged_tokens=jnp.array([[0,2,4,4],[1,3,2,4]])
ragged_roles=jnp.array([[False,True,True,False],[False,True,True,True]])
ragged_logits=jnp.arange(2*4*5,dtype=jnp.float32).reshape(2,4,5)/19
observed=response_logps(ragged_logits,ragged_tokens,ragged_roles)
arr=np.asarray(ragged_logits,dtype=np.float64);expected=[]
for row in range(2):
 total=0.
 for position in range(3):
  if ragged_roles[row,position+1]:
   values=arr[row,position];log_normalizer=values.max()+np.log(np.exp(values-values.max()).sum())
   total+=values[int(ragged_tokens[row,position+1])]-log_normalizer
 expected.append(total)
np.testing.assert_allclose(observed,expected,atol=1e-6)
print('Supervised targets per sequence:',np.asarray(ragged_roles[:,1:].sum(1)))
```

**Expected:** The target counts are two and three; scalar sequence sums match the vectorized implementation.

Padding and role masks affect the target after shifting. The host loop makes every included prediction explicit.

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

## Create a failure this model cannot fix

**Challenge**

Build two contexts ending in token zero but requiring different next tokens. Prove that the transition table gives identical distributions. Name the model capability needed to resolve the conflict.

<details><summary>Hint</summary>

Keep the final input token equal and change an earlier token.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
contexts=jnp.array([[1,0],[3,0]])
last_logits=p[contexts[:,-1]]
np.testing.assert_array_equal(last_logits[0],last_logits[1])
conflicting_targets=jnp.array([2,3])
assert conflicting_targets[0]!=conflicting_targets[1]
print('Identical final tokens force identical table predictions despite different contexts.')
```

A model that conditions on the earlier context can represent this distinction. More iterations on the same transition-table architecture cannot.

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

## Carry forward

- Uniform probabilities over five tokens give negative log probability $\log 5$ at each supervised target. One supervised target contributes that amount; three targets contribute $3\log 5$. Dividing total negative log likelihood by total valid targets returns $\log5$ for either batch. The sum returned by response_logps is intentionally not already a mean.
- Two sequences ending in the same prompt token select exactly the same table row, so they must receive the same next-token distribution. If their intended answers differ because of earlier context, this model cannot represent the task. Demonstrate that collision before replacing the table with a Transformer.

## Keep your evidence

Keep shifted target/role masks, valid response-token counts, gradient support and the changed target-selection result.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [InstructGPT: supervised demonstrations and preference training](https://arxiv.org/abs/2203.02155)

