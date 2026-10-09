"""Direct preference optimization and reference-corrected margins: worked experiments and reference solutions. CPU checks."""

# 1. Define the reference-corrected pair objective
# Step 1 — 1. Define the reference-corrected pair objective: The reference correction is detached, while current policy log...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Define `dpo_loss(policy_logps, reference_logps, chosen, rejected...)` to evaluate the objective and its automatic derivatives:
def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=0.2):
    # Evaluate `margin` from the current inputs and state.
    margin = (policy_logps[chosen] - policy_logps[rejected]) - jax.lax.stop_gradient(
        reference_logps[chosen] - reference_logps[rejected]
    )
    # Return `jnp.mean(jax.nn.softplus(-beta * margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-beta * margin))

# 2. Freeze reference probabilities and comparison IDs
# Step 2 — 2. Freeze reference probabilities and comparison IDs: The initial policy equals a nonuniform reference.
# Initialize array `reference` with explicit values and shape.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
# Initialize array `chosen` with explicit values and shape.
chosen = jnp.array([0, 0, 1])
# Initialize array `rejected` with explicit values and shape.
rejected = jnp.array([1, 2, 2])
# Initialize array `theta` with explicit values and shape.
theta = jnp.array([0.2, 0.0, -0.2])
# Evaluate `history` from the current inputs and state.
history = []
# Evaluate numerically stable log-space cross-entropy/likelihood (`loss`).
loss = lambda theta: dpo_loss(
    jax.nn.log_softmax(theta), reference, chosen, rejected, 0.3
)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(theta), np.log(2), atol=1e-6)

# 3. Fit the policy and test cancellation
# Step 3 — 3. Fit the policy and test cancellation: The host stable-softplus calculation checks final margins.
for _ in range(120):
    # Run `step` to compute `(value, g)`.
    value, g = step(theta)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `theta` from the current inputs and state.
    theta = theta - 0.4 * g

# Verify contract: `history[-1] < history[0] * 0.5`.
assert history[-1] < history[0] * 0.5
# Evaluate numerically stable log-space cross-entropy/likelihood (`policy`).
policy = jax.nn.log_softmax(theta)
# Convert `margin` to a host NumPy array for inspection or verification.
margin = np.asarray(
    (policy[chosen] - policy[rejected]) - (reference[chosen] - reference[rejected]),
    np.float64,
)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(
    loss(theta), np.mean(np.logaddexp(0, -0.3 * margin)), atol=1e-6
)
# Verify contract: `np.all(margin > 0)`.
assert np.all(margin > 0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(theta + 50), loss(theta), atol=1e-6)
# Print the observed values to compare against the expected result.
print(
    'DPO initial/final:',
    history[0],
    history[-1],
    '; reference-corrected margins:',
    margin,
)

# Step 1 — 1. Define the reference-corrected pair objective: The reference correction is detached, while current policy log...
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np

# Define `dpo_loss(policy_logps, reference_logps, chosen, rejected...)` to evaluate the objective and its automatic derivatives:
def dpo_loss(policy_logps, reference_logps, chosen, rejected, beta=0.2):
    # Evaluate `margin` from the current inputs and state.
    margin = (policy_logps[chosen] - policy_logps[rejected]) - jax.lax.stop_gradient(
        reference_logps[chosen] - reference_logps[rejected]
    )
    # Return `jnp.mean(jax.nn.softplus(-beta * margin))` to the caller.
    return jnp.mean(jax.nn.softplus(-beta * margin))

# Step 2 — 2. Freeze reference probabilities and comparison IDs: The initial policy equals a nonuniform reference.
# Initialize array `reference` with explicit values and shape.
reference = jax.nn.log_softmax(jnp.array([0.2, 0.0, -0.2]))
# Initialize array `chosen` with explicit values and shape.
chosen = jnp.array([0, 0, 1])
# Initialize array `rejected` with explicit values and shape.
rejected = jnp.array([1, 2, 2])
# Initialize array `theta` with explicit values and shape.
theta = jnp.array([0.2, 0.0, -0.2])
# Evaluate `history` from the current inputs and state.
history = []
# Evaluate numerically stable log-space cross-entropy/likelihood (`loss`).
loss = lambda theta: dpo_loss(
    jax.nn.log_softmax(theta), reference, chosen, rejected, 0.3
)
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(jax.value_and_grad(loss))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(theta), np.log(2), atol=1e-6)

# Step 3 — 3. Fit the policy and test cancellation: The host stable-softplus calculation checks final margins.
for _ in range(120):
    # Run `step` to compute `(value, g)`.
    value, g = step(theta)
    # Append the current step result to `history`.
    history.append(float(value))
    # Evaluate `theta` from the current inputs and state.
    theta = theta - 0.4 * g

# Verify contract: `history[-1] < history[0] * 0.5`.
assert history[-1] < history[0] * 0.5
# Evaluate numerically stable log-space cross-entropy/likelihood (`policy`).
policy = jax.nn.log_softmax(theta)
# Convert `margin` to a host NumPy array for inspection or verification.
margin = np.asarray(
    (policy[chosen] - policy[rejected]) - (reference[chosen] - reference[rejected]),
    np.float64,
)
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(
    loss(theta), np.mean(np.logaddexp(0, -0.3 * margin)), atol=1e-6
)
# Verify contract: `np.all(margin > 0)`.
assert np.all(margin > 0)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(loss(theta + 50), loss(theta), atol=1e-6)
# Print the observed values to compare against the expected result.
print(
    'DPO initial/final:',
    history[0],
    history[-1],
    '; reference-corrected margins:',
    margin,
)

# Figure data experiment
# Compute figure data for: Direct preference optimization and reference-corrected margins — recorded experiment
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'DPO preference loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
# Loop over `panel` in `visual_data.get('panels', [visual_data])`:
for panel in visual_data.get('panels',[visual_data]):
    # Evaluate `panel['x']` from the current inputs and state.
    panel['x']=panel['series'][0]['x']

# Evaluate `extra_panel` from the current inputs and state.
extra_panel={'kind':'bar','x':[0,1,2],'labels':['0 preferred to 1','0 preferred to 2','1 preferred to 2'],'series':[{'label':'final corrected margin','y':margin.tolist()}],'xlabel':'preference pair','ylabel':'reference-corrected log ratio','title':'Relative preference changes behind the loss'}
# Evaluate `visual_data` from the current inputs and state.
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Verify the reference cancellation
# Experiment — Verify the reference cancellation: The initial policy is not uniform.
initial = dpo_loss(reference, reference, chosen, rejected, 0.3)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(initial, np.log(2), atol=1e-6)
# Aggregate array values to compute `uncorrected`.
uncorrected = jnp.mean(
    jax.nn.softplus(-0.3 * (reference[chosen] - reference[rejected]))
)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(float(initial), float(uncorrected))
# Print the observed values to compare against the expected result.
print('Corrected/unadjusted initial loss:', float(initial), float(uncorrected))

# Experiment: Improve the ratio while lowering the chosen probability
# Experiment — Improve the ratio while lowering the chosen probability: The rejected probability falls further.
# Initialize array `example_ref` with explicit values and shape.
example_ref = jnp.log(jnp.array([0.4, 0.4, 0.2]))
# Initialize array `example_policy` with explicit values and shape.
example_policy = jnp.log(jnp.array([0.3, 0.1, 0.6]))
# Initialize array `choice` with explicit values and shape.
choice = jnp.array([0])
# Initialize array `reject` with explicit values and shape.
reject = jnp.array([1])
# Evaluate `dpo_loss(example_ref, example_ref, choice, reject, 0.3)` and convert the result into Python scalar/collection `before`.
before = float(dpo_loss(example_ref, example_ref, choice, reject, 0.3))
# Evaluate `dpo_loss(example_policy, example_ref, choice, reject, 0.3)` and convert the result into Python scalar/collection `after`.
after = float(dpo_loss(example_policy, example_ref, choice, reject, 0.3))
# Verify contract: `after < before and float(jnp.exp(example_policy[0])) < float(jnp.exp...`.
assert after < before and float(jnp.exp(example_policy[0])) < float(
    jnp.exp(example_ref[0])
)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(after, np.logaddexp(0.0, -0.3 * np.log(3)), atol=1e-6)
# Print the observed values to compare against the expected result.
print('Loss before/after:', before, after, '; chosen probability: 0.4 -> 0.3')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Swap chosen and rejected IDs after training and verify that the loss...
reverse = float(dpo_loss(policy, reference, rejected, chosen, 0.3))
# Verify contract: `reverse > float(dpo_loss(policy, reference, chosen, rejected, 0.3))`.
assert reverse > float(dpo_loss(policy, reference, chosen, rejected, 0.3))
# Print the observed values to compare against the expected result.
print('Reversed-label loss:', reverse)

# Reference practice: Freeze the reference numerically and in autodiff
# Freeze the reference numerically and in autodiff (Transfer): Also retain the actual reference checkpoint and tokenizer...
# Differentiate the objective to obtain `reference_grad` via automatic differentiation.
reference_grad = jax.grad(dpo_loss, argnums=1)(
    policy, reference, chosen, rejected, 0.3
)
# Allocate initialized array `` with the specified shape and dtype.
np.testing.assert_array_equal(
    reference_grad, np.zeros(reference_grad.shape)
)
# Print the observed values to compare against the expected result.
print('Reference gradient is zero.')

# Reference practice: Derive a one-pair categorical gradient
# Derive a one-pair categorical gradient (Challenge): The unpaired logit is absent from this categorical...
# Initialize array `probe` with explicit values and shape.
probe = jnp.array([0.6, -0.3, 0.1])
# Initialize array `choice` with explicit values and shape.
choice = jnp.array([0])
# Initialize array `reject` with explicit values and shape.
reject = jnp.array([1])
# Evaluate `beta` from the current inputs and state.
beta = 0.3
# Evaluate numerically stable log-space cross-entropy/likelihood (`one_pair`).
one_pair = lambda logits: dpo_loss(
    jax.nn.log_softmax(logits), reference, choice, reject, beta
)
# Evaluate `probe[0] - probe[1] - (reference[0] - reference[1])` and convert the result into Python scalar/collection `margin_value`.
margin_value = float((probe[0] - probe[1]) - (reference[0] - reference[1]))
# Evaluate `factor` from the current inputs and state.
factor = -beta / (1 + np.exp(beta * margin_value))
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(
    jax.grad(one_pair)(probe), [factor, -factor, 0.0], atol=1e-6
)
# Print the observed values to compare against the expected result.
print('One-pair logit gradient agrees with independent margin derivative.')
print("PASS: posttraining-05")
