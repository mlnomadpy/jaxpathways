# Learn a reward model from pairwise preferences

Phase 18: Post-training: SFT, LoRA, reward models and RLHF · about 105 minutes · CPU

## What you will be able to do

- Represent a preference as a comparison
- Fit score differences with a stable likelihood
- Verify the changed-condition exercise and explain the limits of this fixture.

## The problem

You know which of two responses is preferred, but you do not have an absolute score for either. How can a model learn from that comparison, and what does its score mean?

## The idea

A pairwise reward model learns to give a preferred item a higher score than its alternative. The loss uses a score difference. That relative objective does not automatically calibrate absolute quality or make preference labels infallible.

## A preference compares scores rather than calibrating utility

If chosen and rejected scores are $3$ and $1$, their margin is $2$. Adding $10$ to both scores leaves the margin unchanged and therefore leaves a difference-based preference loss unchanged. The absolute offset is not identified by those comparisons.

Conflicting labels can make an intermediate preference probability more appropriate than an extreme one. Inspect disagreement and source quality rather than assuming the model should confidently fit every pair.

The margin plot shows separation on the fixture. To support downstream policy optimization, evaluate generalization and possible exploitation separately. A reward that ranks a few training pairs correctly can still be unreliable on policy-generated outputs.

### Pause and reason

Why is a raw reward score of 100 not automatically better calibrated than 1?

<details><summary>Compare your reasoning</summary>

A relative training objective can permit score offsets and depends on the model's conventions. Compare under a shared evaluation contract; raw magnitudes alone are not universal utility units.

</details>

## Before coding: a preference is a comparison

A training example says that one response was preferred to another for the same prompt and rubric. It does not assign a universally meaningful numeric score to either response. Begin by subtracting the rejected score from the chosen score. A positive margin supports the label; a negative margin contradicts it.

The fixture uses three synthetic response feature vectors and three ordered pairs. It teaches comparison likelihood and score identifiability, not human judgment. Keep the prompt/group identity when replacing these vectors with real response representations; comparing unrelated prompts can teach a meaningless ranking.

## Represent a preference as a comparison

Keep the prompt identity, chosen/rejected response identities, label source and split. Pairwise ratings are conditional on a shared prompt and rubric; comparing unrelated prompts can teach a spurious score offset. The fixture has three response feature vectors with an explicit ordering.

## Fit score differences with a stable likelihood

The probability of preferring a chosen response is the sigmoid of its reward difference. The negative log likelihood is softplus of the negative margin. A tied pair starts at $\log 2$. A common offset to all rewards cancels, so comparisons alone do not identify an absolute reward origin.

$$
L_{\mathrm{RM}}=\mathbb E\left[\operatorname{softplus}\!\left(-(r_w(x,y^+)-r_w(x,y^-))\right)\right]
$$

## Calculate likelihood and its derivative

Let $m=r_\theta(x,y^+)-r_\theta(x,y^-)$. The probability of the observed preference is $\sigma(m)$, and its negative log likelihood is $\log(1+e^{-m})$. At margin zero, probability is one half and loss is $\log2$. At margin $\log3$, probability is three quarters and loss is $-\log(3/4)$. The margin derivative is $\sigma(m)-1$: confident correct pairs receive a smaller update than uncertain or wrong pairs.

For a linear score, multiply this derivative by the chosen-minus-rejected feature vector and average pairs. That supplies an independent gradient oracle without differentiating the library softplus implementation.

$$
\nabla_w L=\frac1N\sum_i\left(\sigma(m_i)-1\right)(f_i^+-f_i^-)
$$

## Disagreement changes the optimum

If the same two responses appear once with each preference direction, the average loss is minimized at a tied score, not at infinite confidence in either label. With unequal vote frequencies, the optimum reflects those frequencies under this likelihood model. Collapsing disputed examples into one apparently certain label discards information.

Group train and evaluation splits by prompt/source. Report agreement, likelihood and calibration under the labeling protocol, and inspect disagreements. A scalar metric can hide systematic preferences that conflict between raters or domains. The next policy optimizer will amplify the score signal, so keep a separate evaluator for qualities the reward model may have missed.

## Separate ranking from calibration

Check ordering, held-out pair likelihood and agreement by prompt/source slice. A large positive margin means the model is confident under its likelihood, not that a response is universally good. Separable preferences can drive growing margins; regularization, validation and stopping rules matter even when every training comparison is correct.

## Preserve disagreement and data provenance

Real raters may disagree for legitimate reasons. Retain the rating rubric, task, annotator process and uncertainty instead of converting every disagreement into a perfect label. Split prompts and near-duplicate responses by source so a held-out score does not reward memorization. This lab does not collect personal data or human ratings.

## Keep the reward model fixed during policy evaluation

The next lesson freezes the learned reward while optimizing a policy. If reward changes mid-comparison, observed improvement may reflect a moving evaluator. Retain a reference policy, independent task metrics and adversarial examples to detect reward exploitation.

## Audit both what scores identify and what they do not

Adding one common offset leaves every preference margin unchanged, so pair comparisons cannot identify an absolute zero of reward. Positive scaling preserves ranking but changes probabilities and confidence. The scale also changes the reward-versus-KL tradeoff in policy optimization. Save normalization and model identity when handing scores to the next phase.

A perfectly fitted training ranking can coexist with overconfidence or poor held-out ranking. On separable data the model can keep increasing margins without learning a new ordering. Track weight norms, held-out likelihood and error slices; choose regularization and stopping on development data rather than demanding the smallest possible training loss.

## 1. Define chosen-minus-rejected likelihood

Create main.py in your activated course environment. Paste this block, then run python main.py; function definitions alone print nothing.

```python
# Step 1 — 1. Define chosen-minus-rejected likelihood: The loss sees feature differences.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `preference_loss(w, chosen, rejected)` implementing this stage's computation:
def preference_loss(w, chosen, rejected):
    # Perform matrix / vector contraction (`@`) to compute `margin`.
    margin = (chosen - rejected) @ w
    # Return `jnp.mean(jax.nn.softplus(-margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-margin))
```

The loss sees feature differences. Predict why adding a common score offset later cannot change it.

## 2. Construct and inspect the three comparisons

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
# Step 2 — 2. Construct and inspect the three comparisons: Write the desired order before fitting: response zero above one,...
# Initialize array `features` with explicit values and shape.
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
# Initialize array `chosen` with explicit values and shape.
chosen = features[jnp.array([0, 0, 1])]
# Initialize array `rejected` with explicit values and shape.
rejected = features[jnp.array([1, 2, 2])]
# Initialize array `w` with explicit values and shape.
w = jnp.zeros(2)
# Evaluate `history` from the current inputs and state.
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: preference_loss(w, chosen, rejected))
)
```

Write the desired order before fitting: response zero above one, and one above two. These labels are synthetic.

## 3. Fit rewards and check their invariances

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
# Step 3 — 3. Fit rewards and check their invariances: The host logaddexp oracle checks stable likelihood values.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `w` from the current inputs and state.
    w = w - 0.1 * g

# Perform matrix / vector contraction (`@`) to compute `rewards`.
rewards = features @ w
# Verify contract: `rewards[0] > rewards[1] > rewards[2]`.
assert rewards[0] > rewards[1] > rewards[2]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(history[0], np.log(2), atol=1e-6)
# Verify contract: `history[-1] < 0.1`.
assert history[-1] < 0.1
# Convert `margins` to a host NumPy array for inspection or verification.
margins = np.asarray((chosen - rejected) @ w, dtype=np.float64)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(
    preference_loss(w, chosen, rejected),
    np.mean(np.logaddexp(0, -margins)),
    atol=1e-6,
)
# Only differences are identified: a common score offset leaves preference probabilities unchanged.
np.testing.assert_allclose(
    jax.nn.sigmoid((chosen @ w + 9) - (rejected @ w + 9)),
    jax.nn.sigmoid((chosen - rejected) @ w),
    atol=1e-6,
)
# Print the observed values to compare against the expected result.
print(
    'Synthetic preference NLL initial/final:',
    history[0],
    history[-1],
    '; rewards:',
    np.asarray(rewards),
)
```

The host logaddexp oracle checks stable likelihood values. The score-offset test establishes non-identifiability, not calibration.

## Run the example

```python
# Step 1 — 1. Define chosen-minus-rejected likelihood: The loss sees feature differences.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Function `preference_loss(w, chosen, rejected)` implementing this stage's computation:
def preference_loss(w, chosen, rejected):
    # Perform matrix / vector contraction (`@`) to compute `margin`.
    margin = (chosen - rejected) @ w
    # Return `jnp.mean(jax.nn.softplus(-margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-margin))

# Step 2 — 2. Construct and inspect the three comparisons: Write the desired order before fitting: response zero above one,...
# Initialize array `features` with explicit values and shape.
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
# Initialize array `chosen` with explicit values and shape.
chosen = features[jnp.array([0, 0, 1])]
# Initialize array `rejected` with explicit values and shape.
rejected = features[jnp.array([1, 2, 2])]
# Initialize array `w` with explicit values and shape.
w = jnp.zeros(2)
# Evaluate `history` from the current inputs and state.
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: preference_loss(w, chosen, rejected))
)

# Step 3 — 3. Fit rewards and check their invariances: The host logaddexp oracle checks stable likelihood values.
for _ in range(100):
    # Run `step` to compute `(value, g)`.
    value, g = step(w)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `w` from the current inputs and state.
    w = w - 0.1 * g

# Perform matrix / vector contraction (`@`) to compute `rewards`.
rewards = features @ w
# Verify contract: `rewards[0] > rewards[1] > rewards[2]`.
assert rewards[0] > rewards[1] > rewards[2]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(history[0], np.log(2), atol=1e-6)
# Verify contract: `history[-1] < 0.1`.
assert history[-1] < 0.1
# Convert `margins` to a host NumPy array for inspection or verification.
margins = np.asarray((chosen - rejected) @ w, dtype=np.float64)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(
    preference_loss(w, chosen, rejected),
    np.mean(np.logaddexp(0, -margins)),
    atol=1e-6,
)
# Only differences are identified: a common score offset leaves preference probabilities unchanged.
np.testing.assert_allclose(
    jax.nn.sigmoid((chosen @ w + 9) - (rejected @ w + 9)),
    jax.nn.sigmoid((chosen - rejected) @ w),
    atol=1e-6,
)
# Print the observed values to compare against the expected result.
print(
    'Synthetic preference NLL initial/final:',
    history[0],
    history[-1],
    '; rewards:',
    np.asarray(rewards),
)
```

Expected: The model learns all three declared preferences; common reward offsets preserve probabilities.

## Learn a reward model from pairwise preferences — recorded experiment

**Predict:** Which final comparison margin should be the largest given the ordering zero above one above two?

![Learn a reward model from pairwise preferences — recorded experiment](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The vertical axis is pairwise negative log likelihood in nats. It starts near $0.693=\log2$, then falls to about $0.087$. Each point is before an update on the same three synthetic comparisons; it is not a human agreement rate.

The second panel reports final chosen-minus-rejected score margins for the three training comparisons. All are positive, so each ordering agrees with its synthetic label. The zero-versus-two comparison spans the other two and has the largest margin. A common reward offset would leave every bar unchanged; multiplying weights by two would double them without changing ordering. Neither operation is visible in a ranking-only accuracy number.

### Connect it to the computation

The initial zero weights tie all response scores. Training increases the chosen-minus-rejected margins. The offset and scaling exercises explain why reward values must be interpreted relative to their model and downstream objective.

```python
# Compute figure data for: Learn a reward model from pairwise preferences — recorded experiment
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'pairwise preference NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Evaluate `panel['x']` from the current inputs and state.
    panel['x']=panel['series'][0]['x']

# Convert `extra_panel` to a host NumPy array for inspection or verification.
extra_panel={'kind':'bar','x':[0,1,2],'labels':['0 preferred to 1','0 preferred to 2','1 preferred to 2'],'series':[{'label':'final score difference','y':np.asarray((chosen-rejected)@w).tolist()}],'xlabel':'synthetic comparison','ylabel':'chosen minus rejected reward','title':'Which comparisons the reward model fitted'}
# Evaluate `visual_data` from the current inputs and state.
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:08:05.329051+00:00. JAX 0.9.2.

```text
Synthetic preference NLL initial/final: 0.6931471824645996 0.08669456839561462 ; rewards: [ 1.9811294  0.327278  -2.3084073]
Synthetic preference NLL initial/final: 0.6931471824645996 0.08669456839561462 ; rewards: [ 1.9811294  0.327278  -2.3084073]
Forward/reversed preference NLL: 0.08596369624137878 2.94565486907959
Balanced conflicting labels prefer a zero margin.
Tied baseline and preference directions verified.
Ranking unchanged; confidence and downstream reward scale change.
Independent reward gradients agree in both label directions.
PASS: posttraining-03

```

## Reverse the comparison

**Predict before running:** How should a positive margin change when chosen and rejected responses swap?

```python
# Experiment — Reverse the comparison: This checks label direction.
forward = float(preference_loss(w, chosen, rejected))
# Evaluate `preference_loss(w, rejected, chosen)` and convert the result into Python scalar/collection `reverse`.
reverse = float(preference_loss(w, rejected, chosen))
# Verify contract: `reverse > forward`.
assert reverse > forward
# Print the observed values to compare against the expected result.
print('Forward/reversed preference NLL:', forward, reverse)
```

**Expected:** The reversed-label loss is larger.

This checks label direction. A decreasing incorrectly signed loss can train a consistent but reversed preference model.

## Fit conflicting labels conceptually

**Predict before running:** If a pair is labeled both ways equally often, should the best margin be positive, negative or zero?

```python
# Experiment — Fit conflicting labels conceptually: Disagreement is not fixed by flipping labels until training...
conflict = lambda margin: 0.5 * (
    jax.nn.softplus(-margin) + jax.nn.softplus(margin)
)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(conflict(0.0), np.log(2), atol=1e-6)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(conflict)(0.0), 0.0, atol=1e-7)
# Verify contract: `float(conflict(2.0)) > float(conflict(0.0))`.
assert float(conflict(2.0)) > float(conflict(0.0))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(conflict(2.0), conflict(-2.0))
# Print the observed values to compare against the expected result.
print('Balanced conflicting labels prefer a zero margin.')
```

**Expected:** Zero margin minimizes the balanced contradictory-pair objective.

Disagreement is not fixed by flipping labels until training accuracy is high. The likelihood must represent the information actually present.

## Make it yours

Calculate the initial loss for tied reward scores independently and verify the trained ordering.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.zeros / jnp.ones(shape, dtype=...)` — Allocates a tensor of the given `shape` initialized with constants.
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.

**Step-by-step implementation plan:**
1. Verify contract: `np.all(np.asarray((chosen - rejected) @ w) > 0)`.
2. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Calculate the initial loss for tied reward scores independently and...
np.testing.assert_allclose(
    preference_loss(jnp.zeros(2), chosen, rejected), -np.log(0.5), atol=1e-6
)
# Verify contract: `np.all(np.asarray((chosen - rejected) @ w) > 0)`.
assert np.all(np.asarray((chosen - rejected) @ w)  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Tied baseline and preference directions verified.')
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Calculate the initial loss for tied reward scores independently and...
np.testing.assert_allclose(
    preference_loss(jnp.zeros(2), chosen, rejected), -np.log(0.5), atol=1e-6
)
# Verify contract: `np.all(np.asarray((chosen - rejected) @ w) > 0)`.
assert np.all(np.asarray((chosen - rejected) @ w) > 0)
# Print the observed values to compare against the expected result.
print('Tied baseline and preference directions verified.')
```

</details>

## Show that feature scale can change scores without changing ranking

**Transfer**

Multiply the trained reward weights by two and compare ordering and preference probabilities.

<details><summary>Hint</summary>

Positive scaling preserves ordering but sharpens probabilities.

</details>

### How to write: Show that feature scale can change scores without changing ranking — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `ranking(...)` — Call `ranking` with your updated parameters or inputs from this lesson's workspace.
- `contraction(...)` — Call `contraction` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Perform matrix / vector contraction (`@`) to compute `scaled_rewards`.
2. Verify that computed values match the expected reference within numerical tolerance.
3. Verify contract: `float(preference_loss(2 * w, chosen, rejected)) < float(preference_l...`.
4. Print the observed values to compare against the expected result.

**Starter code scaffold (fill in the TODOs):**

```python
# Show that feature scale can change scores without changing ranking (Transfer): A policy’s reward-versus-KL tradeoff depends on reward scale.
# Perform matrix / vector contraction (`@`) to compute `scaled_rewards`.
scaled_rewards = ...  # TODO: compute scaled_rewards
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(np.argsort(rewards), np.argsort(scaled_rewards))
# Verify contract: `float(preference_loss(2 * w, chosen, rejected)) < float(preference_l...`.
assert float(preference_loss(2 * w, chosen, rejected))  # TODO: complete assertion check
    preference_loss(w, chosen, rejected)
)
# Print the observed values to compare against the expected result.
print('Ranking unchanged; confidence and downstream reward scale change.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Show that feature scale can change scores without changing ranking (Transfer): A policy’s reward-versus-KL tradeoff depends on reward scale.
# Perform matrix / vector contraction (`@`) to compute `scaled_rewards`.
scaled_rewards = features @ (2 * w)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(np.argsort(rewards), np.argsort(scaled_rewards))
# Verify contract: `float(preference_loss(2 * w, chosen, rejected)) < float(preference_l...`.
assert float(preference_loss(2 * w, chosen, rejected)) < float(
    preference_loss(w, chosen, rejected)
)
# Print the observed values to compare against the expected result.
print('Ranking unchanged; confidence and downstream reward scale change.')
```

A policy’s reward-versus-KL tradeoff depends on reward scale. Rank agreement alone cannot establish an equivalent RL objective.

</details>

## Derive the linear reward gradient

**Challenge**

At a fresh weight vector, calculate the mean feature difference weighted by probability-minus-one and compare with autodiff. Repeat after reversing labels.

<details><summary>Hint</summary>

Use NumPy sigmoid values for the independent calculation; reversing both feature order and probabilities is necessary.

</details>

### How to write: Derive the linear reward gradient — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `jnp.array(values, dtype=...)` — Constructs an immutable device-backed JAX array from Python/NumPy values.
- `x.sum(axis=...) / x.mean(axis=..., keepdims=...)` — Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).
- `jnp.allclose(actual, expected, rtol=..., atol=...)` — Checks that two arrays match elementwise within floating-point tolerance.
- `jax.grad(loss_fn)(params, ...)` — Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.

**Step-by-step implementation plan:**
1. Initialize array `probe_w` with explicit values and shape.
2. Iterate over `(positive, negative)` to step through the computation:
3. Convert `difference` to a host NumPy array for inspection or verification.
4. Perform matrix / vector contraction (`@`) to compute `margin_host`.
5. Reduce along axis=0 to compute `analytic`.

**Starter code scaffold (fill in the TODOs):**

```python
# Derive the linear reward gradient (Challenge): Checking both directions catches a sign error that a single...
# Initialize array `probe_w` with explicit values and shape.
probe_w = jnp.array(...)  # TODO: compute probe_w
# Iterate over `(positive, negative)` to step through the computation:
for positive, negative in [(chosen, rejected), (rejected, chosen)]:
    # Convert `difference` to a host NumPy array for inspection or verification.
    difference = np.asarray(...)  # TODO: compute difference
    # Perform matrix / vector contraction (`@`) to compute `margin_host`.
    margin_host = ...  # TODO: compute margin_host
    # Reduce along axis=0 to compute `analytic`.
    analytic = np.mean(...)  # TODO: compute analytic
        (1 / (1 + np.exp(-margin_host)) - 1)[:, None] * difference, axis=0
    )
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(
        jax.grad(preference_loss)(probe_w, positive, negative),
        analytic,
        atol = ...  # TODO: compute atol
    )
# Print the observed values to compare against the expected result.
print('Independent reward gradients agree in both label directions.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Derive the linear reward gradient (Challenge): Checking both directions catches a sign error that a single...
# Initialize array `probe_w` with explicit values and shape.
probe_w = jnp.array([0.2, -0.4])
# Iterate over `(positive, negative)` to step through the computation:
for positive, negative in [(chosen, rejected), (rejected, chosen)]:
    # Convert `difference` to a host NumPy array for inspection or verification.
    difference = np.asarray(positive - negative, dtype=np.float64)
    # Perform matrix / vector contraction (`@`) to compute `margin_host`.
    margin_host = difference @ np.asarray(probe_w)
    # Reduce along axis=0 to compute `analytic`.
    analytic = np.mean(
        (1 / (1 + np.exp(-margin_host)) - 1)[:, None] * difference, axis=0
    )
    # Differentiate the objective to obtain gradients ``.
    np.testing.assert_allclose(
        jax.grad(preference_loss)(probe_w, positive, negative),
        analytic,
        atol=1e-6,
    )
# Print the observed values to compare against the expected result.
print('Independent reward gradients agree in both label directions.')
```

Checking both directions catches a sign error that a single decreasing training curve might conceal.

</details>

## Check your understanding

What do pairwise preferences identify directly?

1. Reward differences under the chosen comparison model, not an absolute universal utility.
2. A unique zero point for individual response rewards across unrelated prompts.
3. A scale-invariant objective whose positive rescaling leaves downstream KL regularization unchanged.

<details><summary>Answer and explanation</summary>

Reward differences under the chosen comparison model, not an absolute universal utility.

A shared additive offset cancels from every pairwise probability.

</details>

## Diagnose the result

If margins have the wrong sign, inspect chosen/rejected order. If training ranking is perfect but held-out agreement fails, inspect prompt leakage, rubric consistency and reward-model overfitting.

## Carry forward

- Pairwise preference loss depends only on score margins $r_\theta(x,y^+)-r_\theta(x,y^-)$, starting at $\log 2$ for tied scores and weighting feature differences by $\sigma(m)-1$.
- Common additive reward offsets cancel in pairwise comparisons while positive weight rescaling sharpens confidence and alters the downstream reward-versus-KL balance.

## Keep your evidence

Keep synthetic label provenance, stable preference-loss and gradient references, offset/scale tests and the learned ordering.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [InstructGPT: reward modeling from preferences](https://arxiv.org/abs/2203.02155)

