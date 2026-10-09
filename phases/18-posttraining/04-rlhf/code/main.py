"""RLHF mechanics: a frozen reward and PPO policy updates: worked experiments and reference solutions. CPU checks."""

# 1. Define the reward likelihood and signed PPO loss
# Step 1 — 1. Define the reward likelihood and signed PPO loss: Keep reward fitting and policy loss separate.
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

# Define `ppo_loss(logits, old_logps, actions, advantages...)` to evaluate the objective and its automatic derivatives:
def ppo_loss(
    logits, old_logps, actions, advantages, reference_logps, beta=0.1, clip=0.2
):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp = jax.nn.log_softmax(logits)
    # Run `jnp.exp` to compute `ratios`.
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    # Run `jax.lax.stop_gradient` to compute `advantages`.
    advantages = jax.lax.stop_gradient(advantages)
    # Combine or mask array elements to form `clipped`.
    clipped = jnp.clip(ratios, 1.0 - clip, 1.0 + clip)
    # Reduce across the target axis to summarize `surrogate`.
    surrogate = jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(
        jnp.exp(logp) * (logp - jax.lax.stop_gradient(reference_logps))
    )
    # Return `-surrogate + beta * kl` to the caller.
    return -surrogate + beta * kl

# 2. Fit and freeze reward, then identify reference state
# Step 2 — 2. Fit and freeze reward, then identify reference state: The reference is stored before policy updates.
# Construct `features` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])`
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
# Construct `chosen` via `features[jnp.array([0, 0, 1])]`
chosen = features[jnp.array([0, 0, 1])]
# Construct `rejected` via `features[jnp.array([1, 2, 2])]`
rejected = features[jnp.array([1, 2, 2])]
# Construct `reward_w` via `jnp.zeros(2)`
reward_w = jnp.zeros(2)
# Differentiate the objective to obtain `reward_step` via automatic differentiation.
reward_step = jax.jit(jax.grad(lambda w: preference_loss(w, chosen, rejected)))
# Repeat the update loop over `range(100)` steps:
for _ in range(100):
    # Compute `reward_w` from `reward_w - 0.1 * reward_step(reward_w)`
    reward_w = reward_w - 0.1 * reward_step(reward_w)
# Perform matrix contraction / projection to compute `rewards`.
rewards = jax.lax.stop_gradient(features @ reward_w)
# A fixed reference policy and a learned reward, with one terminal response action.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
# Construct `theta` via `jnp.array([0.2, 0.0, -0.2])`
theta = jnp.array([0.2, 0.0, -0.2])
# Convert `reference_copy` to a host NumPy array for inspection or verification.
reference_copy = np.asarray(reference).copy()
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(11)
# Compute `history` from `[]`
history = []
# Compute `kl_history` from `[]`
kl_history = []
# Evaluate both scalar loss and parameter gradients in one pass (`step`).
step = jax.jit(jax.value_and_grad(ppo_loss))

# 3. Collect rollouts and reuse each batch correctly
# Step 3 — 3. Collect rollouts and reuse each batch correctly: Each outer iteration collects a fresh batch; each inner step...
# Iterate over `iteration` to step through the computation:
for iteration in range(40):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`old`).
    old = jax.nn.log_softmax(theta)
    # Run `jnp.exp` to compute `probs`.
    probs = jnp.exp(old)
    # Create or split explicit PRNG key(s) (`(key, draw)`) for reproducible randomness.
    key, draw = jax.random.split(key)
    # Draw pseudorandom samples for `actions` using the explicit RNG state.
    actions = jax.random.categorical(draw, old, shape=(128,))
    # Run `jax.lax.stop_gradient` to compute `old_selected`.
    old_selected = jax.lax.stop_gradient(old[actions])
    # Reduce across the target axis to summarize `advantage`.
    advantage = jax.lax.stop_gradient(
        rewards[actions] - jnp.sum(probs * rewards)
    )
    # Repeat the update loop over `range(3)` steps:
    for _ in range(3):
        # Run `step` to compute `(_, g)`.
        _, g = step(
            theta, old_selected, actions, advantage, reference, 0.2, 0.2
        )
        # Compute `theta` from `theta - 0.15 * g`
        theta = theta - 0.15 * g
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp = jax.nn.log_softmax(theta)
    # Reduce across the target axis to summarize ``.
    history.append(float(jnp.sum(jnp.exp(logp) * rewards)))
    # Reduce across the target axis to summarize ``.
    kl_history.append(float(jnp.sum(jnp.exp(logp) * (logp - reference))))

# Execute `np.testing.assert_array_equal(reference, reference_copy)`
np.testing.assert_array_equal(reference, reference_copy)
# Aggregate array values to compute `initial_reward`.
initial_reward = float(jnp.sum(jnp.exp(reference) * rewards))
# Assert invariant `history[-1] > initial_reward and kl_history[-1] > 0` holds
assert history[-1] > initial_reward and kl_history[-1] > 0
# Independent finite-action expectation: no evaluation sampling error here.
np.testing.assert_allclose(
    history[-1],
    sum(float(a) * float(b) for a, b in zip(jnp.exp(logp), rewards)),
    rtol=1e-6,
)
# Print the observed values to compare against the expected result.
print(
    'Reward initial/final and final KL:',
    initial_reward,
    history[-1],
    kl_history[-1],
)
# Print diagnostic summary of the computed outputs.
print(
    'Synthetic preference bandit with actual sampled PPO updates; synthetic preference target or language-generation claim.'
)

# Step 1 — 1. Define the reward likelihood and signed PPO loss: Keep reward fitting and policy loss separate.
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

# Define `ppo_loss(logits, old_logps, actions, advantages...)` to evaluate the objective and its automatic derivatives:
def ppo_loss(
    logits, old_logps, actions, advantages, reference_logps, beta=0.1, clip=0.2
):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp = jax.nn.log_softmax(logits)
    # Run `jnp.exp` to compute `ratios`.
    ratios = jnp.exp(logp[actions] - jax.lax.stop_gradient(old_logps))
    # Run `jax.lax.stop_gradient` to compute `advantages`.
    advantages = jax.lax.stop_gradient(advantages)
    # Combine or mask array elements to form `clipped`.
    clipped = jnp.clip(ratios, 1.0 - clip, 1.0 + clip)
    # Reduce across the target axis to summarize `surrogate`.
    surrogate = jnp.mean(jnp.minimum(ratios * advantages, clipped * advantages))
    # Exact categorical KL in this one-prompt, one-action teaching environment.
    kl = jnp.sum(
        jnp.exp(logp) * (logp - jax.lax.stop_gradient(reference_logps))
    )
    # Return `-surrogate + beta * kl` to the caller.
    return -surrogate + beta * kl

# Step 2 — 2. Fit and freeze reward, then identify reference state: The reference is stored before policy updates.
# Construct `features` via `jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])`
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
# Construct `chosen` via `features[jnp.array([0, 0, 1])]`
chosen = features[jnp.array([0, 0, 1])]
# Construct `rejected` via `features[jnp.array([1, 2, 2])]`
rejected = features[jnp.array([1, 2, 2])]
# Construct `reward_w` via `jnp.zeros(2)`
reward_w = jnp.zeros(2)
# Differentiate the objective to obtain `reward_step` via automatic differentiation.
reward_step = jax.jit(jax.grad(lambda w: preference_loss(w, chosen, rejected)))
# Repeat the update loop over `range(100)` steps:
for _ in range(100):
    # Compute `reward_w` from `reward_w - 0.1 * reward_step(reward_w)`
    reward_w = reward_w - 0.1 * reward_step(reward_w)
# Perform matrix contraction / projection to compute `rewards`.
rewards = jax.lax.stop_gradient(features @ reward_w)
# A fixed reference policy and a learned reward, with one terminal response action.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
# Construct `theta` via `jnp.array([0.2, 0.0, -0.2])`
theta = jnp.array([0.2, 0.0, -0.2])
# Convert `reference_copy` to a host NumPy array for inspection or verification.
reference_copy = np.asarray(reference).copy()
# Create or split explicit PRNG key(s) (`key`) for reproducible randomness.
key = jax.random.key(11)
# Compute `history` from `[]`
history = []
# Compute `kl_history` from `[]`
kl_history = []
# Evaluate both scalar loss and parameter gradients in one pass (`step`).
step = jax.jit(jax.value_and_grad(ppo_loss))

# Step 3 — 3. Collect rollouts and reuse each batch correctly: Each outer iteration collects a fresh batch; each inner step...
# Iterate over `iteration` to step through the computation:
for iteration in range(40):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`old`).
    old = jax.nn.log_softmax(theta)
    # Run `jnp.exp` to compute `probs`.
    probs = jnp.exp(old)
    # Create or split explicit PRNG key(s) (`(key, draw)`) for reproducible randomness.
    key, draw = jax.random.split(key)
    # Draw pseudorandom samples for `actions` using the explicit RNG state.
    actions = jax.random.categorical(draw, old, shape=(128,))
    # Run `jax.lax.stop_gradient` to compute `old_selected`.
    old_selected = jax.lax.stop_gradient(old[actions])
    # Reduce across the target axis to summarize `advantage`.
    advantage = jax.lax.stop_gradient(
        rewards[actions] - jnp.sum(probs * rewards)
    )
    # Repeat the update loop over `range(3)` steps:
    for _ in range(3):
        # Run `step` to compute `(_, g)`.
        _, g = step(
            theta, old_selected, actions, advantage, reference, 0.2, 0.2
        )
        # Compute `theta` from `theta - 0.15 * g`
        theta = theta - 0.15 * g
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp = jax.nn.log_softmax(theta)
    # Reduce across the target axis to summarize ``.
    history.append(float(jnp.sum(jnp.exp(logp) * rewards)))
    # Reduce across the target axis to summarize ``.
    kl_history.append(float(jnp.sum(jnp.exp(logp) * (logp - reference))))

# Execute `np.testing.assert_array_equal(reference, reference_copy)`
np.testing.assert_array_equal(reference, reference_copy)
# Aggregate array values to compute `initial_reward`.
initial_reward = float(jnp.sum(jnp.exp(reference) * rewards))
# Assert invariant `history[-1] > initial_reward and kl_history[-1] > 0` holds
assert history[-1] > initial_reward and kl_history[-1] > 0
# Independent finite-action expectation: no evaluation sampling error here.
np.testing.assert_allclose(
    history[-1],
    sum(float(a) * float(b) for a, b in zip(jnp.exp(logp), rewards)),
    rtol=1e-6,
)
# Print the observed values to compare against the expected result.
print(
    'Reward initial/final and final KL:',
    initial_reward,
    history[-1],
    kl_history[-1],
)
# Print diagnostic summary of the computed outputs.
print(
    'Synthetic preference bandit with actual sampled PPO updates; synthetic preference target or language-generation claim.'
)

# Figure data experiment
# Compute figure data for: RLHF mechanics: a frozen reward and PPO policy updates — recorded experiment
# Compute `visual_data` from `{'panels':[{'kind':'line','xlabel':'completed PPO ro...`
visual_data={'panels':[{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'expected learned reward','series':[{'label':'exact policy expectation','x':list(range(1,len(history)+1)),'y':history}]},{'kind':'line','xlabel':'completed PPO rollout batches','ylabel':'KL to fixed reference (nats)','series':[{'label':'exact categorical KL','x':list(range(1,len(kl_history)+1)),'y':kl_history}]}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Compute `panel['x']` from `panel['series'][0]['x']`
    panel['x']=panel['series'][0]['x']

# Convert `extra_panel` to a host NumPy array for inspection or verification.
extra_panel={'kind':'bar','x':[0,1,2],'labels':['action 0','action 1','action 2'],'series':[{'label':'fixed reference','y':np.asarray(jnp.exp(reference)).tolist()},{'label':'trained policy','y':np.asarray(jnp.exp(logp)).tolist()}],'xlabel':'terminal response action','ylabel':'probability','title':'How increased reward redistributes action probability'}
# Compute `visual_data` from `{"panels":[*visual_data.get("panels",[visual_data]),...`
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Inspect both clipping directions
# Experiment — Inspect both clipping directions: The negative-advantage case selects the more negative product.
# Compute `ratios` from `np.array([1.5, 0.5])`
ratios = np.array([1.5, 0.5])
# Compute `advantages` from `np.array([2.0, -2.0])`
advantages = np.array([2.0, -2.0])
# Reduce across the target axis to summarize `terms`.
terms = np.minimum(
    ratios * advantages, np.clip(ratios, 0.8, 1.2) * advantages
)
# Execute `np.testing.assert_allclose(terms, [2.4, -1.6])`
np.testing.assert_allclose(terms, [2.4, -1.6])
# Print the observed values to compare against the expected result.
print('Clipped surrogate terms:', terms)

# Experiment: Enumerate the baseline cancellation
# Experiment — Enumerate the baseline cancellation: Enumeration removes sampling error.
# Construct `probe` via `jnp.array([0.3, -0.2, 0.1])`
probe = jnp.array([0.3, -0.2, 0.1])
# Apply nonlinear activation or probability normalization to compute `probe_probs`.
probe_probs = jax.nn.softmax(probe)
# Aggregate array values to compute `constant`.
constant = float(jnp.sum(probe_probs * rewards))
# Compute exact directional derivative / Jacobian / Hessian (`score_jacobian`).
score_jacobian = jax.jacrev(jax.nn.log_softmax)(probe)
# Reduce along axis=0 to compute `enumerated`.
enumerated = jnp.sum(
    probe_probs[:, None] * (rewards - constant)[:, None] * score_jacobian,
    axis=0,
)
# Differentiate the objective to obtain `exact` via automatic differentiation.
exact = jax.grad(lambda logits: jnp.sum(jax.nn.softmax(logits) * rewards))(
    probe
)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(enumerated, exact, atol=1e-6)`
np.testing.assert_allclose(enumerated, exact, atol=1e-6)
# Print the observed values to compare against the expected result.
print('Enumerated baseline score gradient matches exact expected-reward gradient.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Check the KL at the fixed reference and compare it with the final...
# Aggregate array values to compute `zero_kl`.
zero_kl = float(jnp.sum(jnp.exp(reference) * (reference - reference)))
# Assert invariant `zero_kl == 0.0 and kl_history[-1] > zero_kl` holds
assert zero_kl == 0.0 and kl_history[-1] > zero_kl
# Print the observed values to compare against the expected result.
print('Reference/final KL:', zero_kl, kl_history[-1])

# Reference practice: Verify detached behavior information
# Verify detached behavior information (Transfer): Detaching is not enough if you recompute old probabilities...
# Differentiate the objective to obtain `(old_grad, adv_grad)` via automatic differentiation.
old_grad, adv_grad = jax.grad(ppo_loss, argnums=(1, 3))(
    theta, old_selected, actions, advantage, reference, 0.2, 0.2
)
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(old_grad, np.zeros(old_grad.shape))
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(adv_grad, np.zeros(adv_grad.shape))
# Print the observed values to compare against the expected result.
print('Behavior log probabilities and advantages are detached.')

# Reference practice: Compare with the finite-action regularized optimum
# Compare with the finite-action regularized optimum (Challenge): An independently derived optimum bounds this finite-action...
beta = 0.2
# Evaluate numerically stable log-space cross-entropy/likelihood (`optimal_logp`).
optimal_logp = jax.nn.log_softmax(reference + rewards / beta)

# Function `regularized(logp)` implementing this stage's computation:
def regularized(logp):
    # Return `jnp.sum(jnp.exp(logp) * rewards) - beta * jnp.sum(jnp.exp(logp) * (logp - reference))` to the caller.
    return jnp.sum(jnp.exp(logp) * rewards) - beta * jnp.sum(
        jnp.exp(logp) * (logp - reference)
    )

# Evaluate `regularized(optimal_logp)` and convert the result into Python scalar/collection `optimum`.
optimum = float(regularized(optimal_logp))
# Evaluate `regularized(logp)` and convert the result into Python scalar/collection `trained`.
trained = float(regularized(logp))
# Evaluate `regularized(reference)` and convert the result into Python scalar/collection `baseline`.
baseline = float(regularized(reference))
# Assert invariant `optimum >= trained - 1e-5 and optimum >= baseline - 1e-5` holds
assert optimum >= trained - 1e-5 and optimum >= baseline - 1e-5
# Print the observed values to compare against the expected result.
print(
    'Regularized objective reference/trained/optimum:',
    baseline,
    trained,
    optimum,
)
print("PASS: posttraining-04")
