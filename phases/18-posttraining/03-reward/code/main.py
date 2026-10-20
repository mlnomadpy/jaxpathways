"""Learn a reward model from pairwise preferences: worked experiments and reference solutions. CPU checks."""

# 1. Define chosen-minus-rejected likelihood
# Step 1 — 1. Define chosen-minus-rejected likelihood: The loss sees feature differences.
import jax
import jax.numpy as jnp
import numpy as np

def preference_loss(w, chosen, rejected):
    # Perform matrix / vector contraction (`@`) to compute `margin`.
    margin = (chosen - rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

# 2. Construct and inspect the three comparisons
# Step 2 — 2. Construct and inspect the three comparisons: Write the desired order before fitting: response zero above one,...
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
chosen = features[jnp.array([0, 0, 1])]
rejected = features[jnp.array([1, 2, 2])]
w = jnp.zeros(2)
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: preference_loss(w, chosen, rejected))
)

# 3. Fit rewards and check their invariances
# Step 3 — 3. Fit rewards and check their invariances: The host logaddexp oracle checks stable likelihood values.
for _ in range(100):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.1 * g

# Perform matrix / vector contraction (`@`) to compute `rewards`.
rewards = features @ w
assert rewards[0] > rewards[1] > rewards[2]
np.testing.assert_allclose(history[0], np.log(2), atol=1e-6)
assert history[-1] < 0.1
# Convert `margins` to a host NumPy array for inspection or verification.
margins = np.asarray((chosen - rejected) @ w, dtype=np.float64)
# Execute `np.testing.assert_allclose(`.
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
print(
    'Synthetic preference NLL initial/final:',
    history[0],
    history[-1],
    '; rewards:',
    np.asarray(rewards),
)

# Step 1 — 1. Define chosen-minus-rejected likelihood: The loss sees feature differences.
import jax
import jax.numpy as jnp
import numpy as np

def preference_loss(w, chosen, rejected):
    # Perform matrix / vector contraction (`@`) to compute `margin`.
    margin = (chosen - rejected) @ w
    return jnp.mean(jax.nn.softplus(-margin))

# Step 2 — 2. Construct and inspect the three comparisons: Write the desired order before fitting: response zero above one,...
features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
chosen = features[jnp.array([0, 0, 1])]
rejected = features[jnp.array([1, 2, 2])]
w = jnp.zeros(2)
history = []
# Differentiate the objective to obtain `step` via automatic differentiation.
step = jax.jit(
    jax.value_and_grad(lambda w: preference_loss(w, chosen, rejected))
)

# Step 3 — 3. Fit rewards and check their invariances: The host logaddexp oracle checks stable likelihood values.
for _ in range(100):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.1 * g

# Perform matrix / vector contraction (`@`) to compute `rewards`.
rewards = features @ w
assert rewards[0] > rewards[1] > rewards[2]
np.testing.assert_allclose(history[0], np.log(2), atol=1e-6)
assert history[-1] < 0.1
# Convert `margins` to a host NumPy array for inspection or verification.
margins = np.asarray((chosen - rejected) @ w, dtype=np.float64)
# Execute `np.testing.assert_allclose(`.
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
print(
    'Synthetic preference NLL initial/final:',
    history[0],
    history[-1],
    '; rewards:',
    np.asarray(rewards),
)

# Figure data experiment
# Compute figure data for: Learn a reward model from pairwise preferences — recorded experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'pairwise preference NLL (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

# Convert `extra_panel` to a host NumPy array for inspection or verification.
extra_panel={'kind':'bar','x':[0,1,2],'labels':['0 preferred to 1','0 preferred to 2','1 preferred to 2'],'series':[{'label':'final score difference','y':np.asarray((chosen-rejected)@w).tolist()}],'xlabel':'synthetic comparison','ylabel':'chosen minus rejected reward','title':'Which comparisons the reward model fitted'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Reverse the comparison
# Experiment — Reverse the comparison: This checks label direction.
forward = float(preference_loss(w, chosen, rejected))
reverse = float(preference_loss(w, rejected, chosen))
assert reverse > forward
print('Forward/reversed preference NLL:', forward, reverse)

# Experiment: Fit conflicting labels conceptually
# Experiment — Fit conflicting labels conceptually: Disagreement is not fixed by flipping labels until training...
conflict = lambda margin: 0.5 * (
    jax.nn.softplus(-margin) + jax.nn.softplus(margin)
)
np.testing.assert_allclose(conflict(0.0), np.log(2), atol=1e-6)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(conflict)(0.0), 0.0, atol=1e-7)
assert float(conflict(2.0)) > float(conflict(0.0))
# Execute `np.testing.assert_allclose(conflict(2.0), conflict(-2.0))`
np.testing.assert_allclose(conflict(2.0), conflict(-2.0))
print('Balanced conflicting labels prefer a zero margin.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Calculate the initial loss for tied reward scores independently and...
np.testing.assert_allclose(
    preference_loss(jnp.zeros(2), chosen, rejected), -np.log(0.5), atol=1e-6
)
assert np.all(np.asarray((chosen - rejected) @ w) > 0)
print('Tied baseline and preference directions verified.')

# Reference practice: Show that feature scale can change scores without changing ranking
# Show that feature scale can change scores without changing ranking (Transfer): A policy’s reward-versus-KL tradeoff depends on reward scale.
# Perform matrix / vector contraction (`@`) to compute `scaled_rewards`.
scaled_rewards = features @ (2 * w)
# Execute `np.testing.assert_array_equal(np.argsort(rewards), np.argsor`
np.testing.assert_array_equal(np.argsort(rewards), np.argsort(scaled_rewards))
assert float(preference_loss(2 * w, chosen, rejected)) < float(
    preference_loss(w, chosen, rejected)
)
print('Ranking unchanged; confidence and downstream reward scale change.')

# Reference practice: Derive the linear reward gradient
# Derive the linear reward gradient (Challenge): Checking both directions catches a sign error that a single...
probe_w = jnp.array([0.2, -0.4])
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
print('Independent reward gradients agree in both label directions.')
print("PASS: posttraining-03")
