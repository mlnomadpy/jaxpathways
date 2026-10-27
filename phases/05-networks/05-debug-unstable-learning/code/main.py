"""Debug unstable learning: worked experiments and reference solutions. CPU checks."""

# 1. Localize and repair three isolated MLP faults
# MLP failure-localization lab : each case changes one condition and uses the same held-out rows.
import jax
import jax.numpy as jnp
import numpy as np
rng = np.random.default_rng(8)
train_np = rng.normal(size=(128, 2)).astype(np.float32)
test_np = rng.normal(size=(96, 2)).astype(np.float32)
train_labels_np = (train_np[:, 0] - 0.6 * train_np[:, 1] > 0.1).astype(np.float32)
test_labels_np = (test_np[:, 0] - 0.6 * test_np[:, 1] > 0.1).astype(np.float32)
train_x, train_y = jnp.asarray(train_np), jnp.asarray(train_labels_np)
test_x, test_y = jnp.asarray(test_np), jnp.asarray(test_labels_np)
k1, k2 = jax.random.split(jax.random.key(4))
initial = {'w1': jax.random.normal(k1, (2, 8)) * 0.5, 'b1': jnp.zeros((8,)),
           'w2': jax.random.normal(k2, (8, 1)) * 0.5, 'b2': jnp.zeros((1,))}
def mlp_forward(params, x, input_scale=1.0):
    preactivation = (input_scale * x) @ params['w1'] + params['b1']
    hidden = jnp.tanh(preactivation)
    logits = (hidden @ params['w2'] + params['b2']).squeeze(-1)
    return logits, (preactivation, hidden)
def mlp_loss(params, x, y, input_scale=1.0):
    logits, _ = mlp_forward(params, x, input_scale)
    return jnp.mean(jax.nn.softplus(logits) - y * logits)
def tree_l2(tree):
    return jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree.leaves(tree)))
def train_mlp(params, rate, steps, x=train_x, y=train_y):
    losses = []
    for _ in range(steps):
        value, grads = jax.value_and_grad(mlp_loss)(params, x, y)
        assert np.isfinite(float(value)) and np.isfinite(float(tree_l2(grads)))
        params = jax.tree.map(lambda p, g: p - rate * g, params, grads)
        losses.append(float(value))
    return params, np.asarray(losses)
# Sound reference: one MLP run improves a distinct held-out set; NumPy computes BCE independently.
reference_model, reference_curve = train_mlp(initial, 0.03, 20)
reference_logits, _ = mlp_forward(reference_model, test_x)
reference_test = float(mlp_loss(reference_model, test_x, test_y))
np.testing.assert_allclose(reference_test,
    np.mean(np.logaddexp(0.0, np.asarray(reference_logits)) - test_labels_np * np.asarray(reference_logits)),
    rtol=1e-5, atol=1e-6)
assert reference_curve[-1] < reference_curve[0] and reference_test < float(mlp_loss(initial, test_x, test_y))
# Fault A: one corrupted feature. Predict the first failing boundary before running a forward pass.
initial_heldout = float(mlp_loss(initial, test_x, test_y))
corrupt_x = train_x.at[7, 1].set(jnp.nan)
first_bad_input = np.argwhere(~np.isfinite(np.asarray(corrupt_x)))
assert first_bad_input.tolist() == [[7, 1]]
bad_data_loss = mlp_loss(initial, corrupt_x, train_y)
assert not np.isfinite(float(bad_data_loss))
# Repair only the source row by restoring its known clean feature value.
repaired_x = corrupt_x.at[7, 1].set(train_x[7, 1])
assert np.isfinite(float(mlp_loss(initial, repaired_x, train_y)))
data_repaired_heldout = float(mlp_loss(initial, test_x, test_y))
np.testing.assert_allclose(data_repaired_heldout, initial_heldout, rtol=1e-6)
np.testing.assert_array_equal(repaired_x, train_x)
# Fault B: change only the input scale entering the first layer. Locate tanh saturation and its gradients.
saturation_params = {**initial, 'b1': jnp.full((8,), 20.0)}
_, (preact_bad, hidden_bad) = mlp_forward(saturation_params, train_x)
saturated = jnp.mean(jnp.abs(hidden_bad) > 0.99)
grad_bad = jax.grad(mlp_loss)(saturation_params, train_x, train_y)
grad_by_layer = {'hidden (w1,b1)': tree_l2({'w1': grad_bad['w1'], 'b1': grad_bad['b1']}),
                 'output (w2,b2)': tree_l2({'w2': grad_bad['w2'], 'b2': grad_bad['b2']})}
assert float(saturated) > 0.99
assert np.all(np.isfinite([float(value) for value in grad_by_layer.values()]))
saturated_test = float(mlp_loss(saturation_params, test_x, test_y))
base_test = float(mlp_loss(initial, test_x, test_y))
saturation_repaired = {**saturation_params, 'b1': initial['b1']}
np.testing.assert_allclose(mlp_loss(saturation_repaired, test_x, test_y), base_test, rtol=1e-6)
assert saturated_test > base_test
# Fault C: leave data, initialization, and gradient calculation fixed; use a grossly oversized learning rate.
large_rate_model, large_rate_curve = train_mlp(initial, 100.0, 3)
small_rate_model, small_rate_curve = train_mlp(initial, 0.03, 3)
large_test = float(mlp_loss(large_rate_model, test_x, test_y))
small_test = float(mlp_loss(small_rate_model, test_x, test_y))
assert large_rate_curve[1] > large_rate_curve[0] and large_rate_curve[2] > large_rate_curve[1]
assert large_test > float(mlp_loss(initial, test_x, test_y))
assert small_rate_curve[-1] < small_rate_curve[0] and small_test < float(mlp_loss(initial, test_x, test_y))
# Plot values are computed from each actual update; x=0 is the initialized model.
def loss_path(rate, steps=3):
    params = initial
    train_values = [float(mlp_loss(params, train_x, train_y))]
    heldout_values = [float(mlp_loss(params, test_x, test_y))]
    for _ in range(steps):
        _, grads = jax.value_and_grad(mlp_loss)(params, train_x, train_y)
        params = jax.tree.map(lambda value, grad: value - rate * grad, params, grads)
        train_values.append(float(mlp_loss(params, train_x, train_y)))
        heldout_values.append(float(mlp_loss(params, test_x, test_y)))
    return train_values, heldout_values
safe_train_path, safe_test_path = loss_path(0.03)
fast_train_path, fast_test_path = loss_path(100.0)
visual_data = {'kind': 'line', 'x': [0, 1, 2, 3], 'xlabel': 'completed full-batch updates (0 = initialized model)',
 'ylabel': 'binary cross-entropy (natural log units)', 'series': [
  {'label': 'training, lr=0.03', 'y': safe_train_path},
  {'label': 'held-out, lr=0.03', 'y': safe_test_path},
  {'label': 'training, lr=100', 'y': fast_train_path},
  {'label': 'held-out, lr=100', 'y': fast_test_path}]}
assert fast_train_path[1] > fast_train_path[0] and fast_test_path[-1] > fast_test_path[0]
assert safe_train_path[-1] < safe_train_path[0] and safe_test_path[-1] < safe_test_path[0]
print('Reference MLP: train loss %.4f -> %.4f; held-out loss %.4f -> %.4f' % (reference_curve[0], reference_curve[-1], float(mlp_loss(initial, test_x, test_y)), reference_test))
print('Data fault: earliest invalid input at row 7, feature 1; repair restored source row and held-out BCE %.4f.' % data_repaired_heldout)
print('Saturation fault: %.1f%% |tanh(h)| > .99; hidden gradient %.5f, output gradient %.5f; held-out loss %.4f versus %.4f after restoring the hidden bias.' % (100 * float(saturated), float(grad_by_layer['hidden (w1,b1)']), float(grad_by_layer['output (w2,b2)']), saturated_test, base_test))
print('Update fault: lr=100 loss %.4f -> %.4f -> %.4f, held-out %.4f; repair lr=.03 ends train %.4f, held-out %.4f.' % (large_rate_curve[0], large_rate_curve[1], large_rate_curve[2], large_test, small_rate_curve[-1], small_test))

# Analytic reference 1. Define the controlled objective
# Step 1 : 1. Define the controlled objective: The initial loss is 56/3 and the initial gradient is -56/3.
import jax
import jax.numpy as jnp
import numpy as np
x=jnp.array([1.,2.,3.])
y=2*x
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
def analytic_gradient(w):return (28/3)*(w-2)
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

# Analytic reference 2. Instrument the update loop
# Step 2 : 2. Instrument the update loop: The rate changes while initial state, data, and objective stay fixed.
def run(rate,steps,features=x,targets=y,clip=None):
    w=jnp.array(0.)
    history=[]
    objective=lambda value:loss(value,features,targets)
    for _ in range(steps):
        value,grad=jax.value_and_grad(objective)(w)
        history.append([float(w),float(value),float(jnp.abs(grad))])
        assert np.all(np.isfinite(history[-1]))
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        w=w-rate*update
    return float(w),np.asarray(history)
stable,good=run(.1,6)
unstable,bad=run(.3,6)
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

# Analytic reference 3. Match the recorded failure to the exact recurrence
# Step 3 : 3. Match the recorded failure to the exact recurrence
for rate,history in [(.1,good),(.3,bad)]:
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    expected_loss=(14/3)*(expected_w-2)**2
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

# Step 1 : 1. Define the controlled objective: The initial loss is 56/3 and the initial gradient is -56/3.
import jax
import jax.numpy as jnp
import numpy as np
x=jnp.array([1.,2.,3.])
y=2*x
def loss(w,features=x,targets=y):return jnp.mean((w*features-targets)**2)
def analytic_gradient(w):return (28/3)*(w-2)
np.testing.assert_allclose(loss(0.),56/3,rtol=1e-6)
np.testing.assert_allclose(jax.grad(loss)(0.),-56/3,rtol=1e-6)

# Step 2 : 2. Instrument the update loop: The rate changes while initial state, data, and objective stay fixed.
def run(rate,steps,features=x,targets=y,clip=None):
    w=jnp.array(0.)
    history=[]
    objective=lambda value:loss(value,features,targets)
    for _ in range(steps):
        value,grad=jax.value_and_grad(objective)(w)
        history.append([float(w),float(value),float(jnp.abs(grad))])
        assert np.all(np.isfinite(history[-1]))
        update=jnp.clip(grad,-clip,clip) if clip is not None else grad
        w=w-rate*update
    return float(w),np.asarray(history)
stable,good=run(.1,6)
unstable,bad=run(.3,6)
assert abs(stable-2)<1e-5 and bad[-1,1]>bad[0,1]*100

# Step 3 : 3. Match the recorded failure to the exact recurrence
for rate,history in [(.1,good),(.3,bad)]:
    expected_w=2-2*(1-rate*28/3)**np.arange(len(history))
    expected_loss=(14/3)*(expected_w-2)**2
    np.testing.assert_allclose(history[:,0],expected_w,rtol=2e-5,atol=1e-5)
    np.testing.assert_allclose(history[:,1],expected_loss,rtol=2e-5,atol=1e-8)
np.testing.assert_allclose(bad[1:,1]/bad[:-1,1],3.24,rtol=1e-5)
print('Stable final weight:',stable,'unstable pre-update [weight,loss,|gradient|]:\n',bad)

# MLP failure-localization lab : each case changes one condition and uses the same held-out rows.
import jax
import jax.numpy as jnp
import numpy as np
rng = np.random.default_rng(8)
train_np = rng.normal(size=(128, 2)).astype(np.float32)
test_np = rng.normal(size=(96, 2)).astype(np.float32)
train_labels_np = (train_np[:, 0] - 0.6 * train_np[:, 1] > 0.1).astype(np.float32)
test_labels_np = (test_np[:, 0] - 0.6 * test_np[:, 1] > 0.1).astype(np.float32)
train_x, train_y = jnp.asarray(train_np), jnp.asarray(train_labels_np)
test_x, test_y = jnp.asarray(test_np), jnp.asarray(test_labels_np)
k1, k2 = jax.random.split(jax.random.key(4))
initial = {'w1': jax.random.normal(k1, (2, 8)) * 0.5, 'b1': jnp.zeros((8,)),
           'w2': jax.random.normal(k2, (8, 1)) * 0.5, 'b2': jnp.zeros((1,))}
def mlp_forward(params, x, input_scale=1.0):
    preactivation = (input_scale * x) @ params['w1'] + params['b1']
    hidden = jnp.tanh(preactivation)
    logits = (hidden @ params['w2'] + params['b2']).squeeze(-1)
    return logits, (preactivation, hidden)
def mlp_loss(params, x, y, input_scale=1.0):
    logits, _ = mlp_forward(params, x, input_scale)
    return jnp.mean(jax.nn.softplus(logits) - y * logits)
def tree_l2(tree):
    return jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree.leaves(tree)))
def train_mlp(params, rate, steps, x=train_x, y=train_y):
    losses = []
    for _ in range(steps):
        value, grads = jax.value_and_grad(mlp_loss)(params, x, y)
        assert np.isfinite(float(value)) and np.isfinite(float(tree_l2(grads)))
        params = jax.tree.map(lambda p, g: p - rate * g, params, grads)
        losses.append(float(value))
    return params, np.asarray(losses)
# Sound reference: one MLP run improves a distinct held-out set; NumPy computes BCE independently.
reference_model, reference_curve = train_mlp(initial, 0.03, 20)
reference_logits, _ = mlp_forward(reference_model, test_x)
reference_test = float(mlp_loss(reference_model, test_x, test_y))
np.testing.assert_allclose(reference_test,
    np.mean(np.logaddexp(0.0, np.asarray(reference_logits)) - test_labels_np * np.asarray(reference_logits)),
    rtol=1e-5, atol=1e-6)
assert reference_curve[-1] < reference_curve[0] and reference_test < float(mlp_loss(initial, test_x, test_y))
# Fault A: one corrupted feature. Predict the first failing boundary before running a forward pass.
initial_heldout = float(mlp_loss(initial, test_x, test_y))
corrupt_x = train_x.at[7, 1].set(jnp.nan)
first_bad_input = np.argwhere(~np.isfinite(np.asarray(corrupt_x)))
assert first_bad_input.tolist() == [[7, 1]]
bad_data_loss = mlp_loss(initial, corrupt_x, train_y)
assert not np.isfinite(float(bad_data_loss))
# Repair only the source row by restoring its known clean feature value.
repaired_x = corrupt_x.at[7, 1].set(train_x[7, 1])
assert np.isfinite(float(mlp_loss(initial, repaired_x, train_y)))
data_repaired_heldout = float(mlp_loss(initial, test_x, test_y))
np.testing.assert_allclose(data_repaired_heldout, initial_heldout, rtol=1e-6)
np.testing.assert_array_equal(repaired_x, train_x)
# Fault B: change only the input scale entering the first layer. Locate tanh saturation and its gradients.
saturation_params = {**initial, 'b1': jnp.full((8,), 20.0)}
_, (preact_bad, hidden_bad) = mlp_forward(saturation_params, train_x)
saturated = jnp.mean(jnp.abs(hidden_bad) > 0.99)
grad_bad = jax.grad(mlp_loss)(saturation_params, train_x, train_y)
grad_by_layer = {'hidden (w1,b1)': tree_l2({'w1': grad_bad['w1'], 'b1': grad_bad['b1']}),
                 'output (w2,b2)': tree_l2({'w2': grad_bad['w2'], 'b2': grad_bad['b2']})}
assert float(saturated) > 0.99
assert np.all(np.isfinite([float(value) for value in grad_by_layer.values()]))
saturated_test = float(mlp_loss(saturation_params, test_x, test_y))
base_test = float(mlp_loss(initial, test_x, test_y))
saturation_repaired = {**saturation_params, 'b1': initial['b1']}
np.testing.assert_allclose(mlp_loss(saturation_repaired, test_x, test_y), base_test, rtol=1e-6)
assert saturated_test > base_test
# Fault C: leave data, initialization, and gradient calculation fixed; use a grossly oversized learning rate.
large_rate_model, large_rate_curve = train_mlp(initial, 100.0, 3)
small_rate_model, small_rate_curve = train_mlp(initial, 0.03, 3)
large_test = float(mlp_loss(large_rate_model, test_x, test_y))
small_test = float(mlp_loss(small_rate_model, test_x, test_y))
assert large_rate_curve[1] > large_rate_curve[0] and large_rate_curve[2] > large_rate_curve[1]
assert large_test > float(mlp_loss(initial, test_x, test_y))
assert small_rate_curve[-1] < small_rate_curve[0] and small_test < float(mlp_loss(initial, test_x, test_y))
# Plot values are computed from each actual update; x=0 is the initialized model.
def loss_path(rate, steps=3):
    params = initial
    train_values = [float(mlp_loss(params, train_x, train_y))]
    heldout_values = [float(mlp_loss(params, test_x, test_y))]
    for _ in range(steps):
        _, grads = jax.value_and_grad(mlp_loss)(params, train_x, train_y)
        params = jax.tree.map(lambda value, grad: value - rate * grad, params, grads)
        train_values.append(float(mlp_loss(params, train_x, train_y)))
        heldout_values.append(float(mlp_loss(params, test_x, test_y)))
    return train_values, heldout_values
safe_train_path, safe_test_path = loss_path(0.03)
fast_train_path, fast_test_path = loss_path(100.0)
visual_data = {'kind': 'line', 'x': [0, 1, 2, 3], 'xlabel': 'completed full-batch updates (0 = initialized model)',
 'ylabel': 'binary cross-entropy (natural log units)', 'series': [
  {'label': 'training, lr=0.03', 'y': safe_train_path},
  {'label': 'held-out, lr=0.03', 'y': safe_test_path},
  {'label': 'training, lr=100', 'y': fast_train_path},
  {'label': 'held-out, lr=100', 'y': fast_test_path}]}
assert fast_train_path[1] > fast_train_path[0] and fast_test_path[-1] > fast_test_path[0]
assert safe_train_path[-1] < safe_train_path[0] and safe_test_path[-1] < safe_test_path[0]
print('Reference MLP: train loss %.4f -> %.4f; held-out loss %.4f -> %.4f' % (reference_curve[0], reference_curve[-1], float(mlp_loss(initial, test_x, test_y)), reference_test))
print('Data fault: earliest invalid input at row 7, feature 1; repair restored source row and held-out BCE %.4f.' % data_repaired_heldout)
print('Saturation fault: %.1f%% |tanh(h)| > .99; hidden gradient %.5f, output gradient %.5f; held-out loss %.4f versus %.4f after restoring the hidden bias.' % (100 * float(saturated), float(grad_by_layer['hidden (w1,b1)']), float(grad_by_layer['output (w2,b2)']), saturated_test, base_test))
print('Update fault: lr=100 loss %.4f -> %.4f -> %.4f, held-out %.4f; repair lr=.03 ends train %.4f, held-out %.4f.' % (large_rate_curve[0], large_rate_curve[1], large_rate_curve[2], large_test, small_rate_curve[-1], small_test))

# Figure data experiment
# Plot values are computed from each actual update; x=0 is the initialized model.
def loss_path(rate, steps=3):
    params = initial
    train_values = [float(mlp_loss(params, train_x, train_y))]
    heldout_values = [float(mlp_loss(params, test_x, test_y))]
    for _ in range(steps):
        _, grads = jax.value_and_grad(mlp_loss)(params, train_x, train_y)
        params = jax.tree.map(lambda value, grad: value - rate * grad, params, grads)
        train_values.append(float(mlp_loss(params, train_x, train_y)))
        heldout_values.append(float(mlp_loss(params, test_x, test_y)))
    return train_values, heldout_values
safe_train_path, safe_test_path = loss_path(0.03)
fast_train_path, fast_test_path = loss_path(100.0)
visual_data = {'kind': 'line', 'x': [0, 1, 2, 3], 'xlabel': 'completed full-batch updates (0 = initialized model)',
 'ylabel': 'binary cross-entropy (nats; log axis)', 'yscale': 'log', 'series': [
  {'label': 'training, lr=0.03', 'y': safe_train_path},
  {'label': 'held-out, lr=0.03', 'y': safe_test_path},
  {'label': 'training, lr=100', 'y': fast_train_path},
  {'label': 'held-out, lr=100', 'y': fast_test_path}]}
assert fast_train_path[1] > fast_train_path[0] and fast_test_path[-1] > fast_test_path[0]
assert safe_train_path[-1] < safe_train_path[0] and safe_test_path[-1] < safe_test_path[0]

# Experiment: Bounded clipped updates
# Experiment : Bounded clipped updates: Clipping changes the effective update.
clipped,clipped_history=run(.3,6,clip=1.)
assert np.all(np.abs(np.diff(clipped_history[:,0]))<=.300001)
assert clipped_history[-1,1]<clipped_history[0,1]
assert clipped_history[0,2]>1.

# Experiment: Feature scaling changes curvature
# Experiment : Feature scaling changes curvature
scaled_final,scaled=run(.1,3,x*10,y*10)
assert scaled[-1,1]>scaled[0,1]
adjusted_final,adjusted=run(.001,6,x*10,y*10)
np.testing.assert_allclose(adjusted[:,0],good[:,0],rtol=1e-5,atol=1e-5)
np.testing.assert_allclose(adjusted[:,1],good[:,1]*100,rtol=1e-4,atol=1e-8)

# Reference solution. Try the exercise before reading this.
# Exercise solution: use the exact fixture and diagnostics defined in first MLP build step.
assert first_bad_input.tolist() == [[7, 1]]
assert np.isfinite(float(mlp_loss(initial, repaired_x, train_y)))
assert float(saturated) > 0.99
assert large_test > float(mlp_loss(initial, test_x, test_y))
assert small_test < float(mlp_loss(initial, test_x, test_y))
np.testing.assert_array_equal(repaired_x, train_x)
print('Each isolated MLP failure was localized and its one-factor repair passed the held-out check.')

# Reference practice: Rescale the objective and compensate the update
# Rescale the objective and compensate the update (Transfer): Scaling the objective scales the gradient.
w0=jnp.array(.25)
base_gradient=jax.grad(loss)(w0)
scaled_gradient=jax.grad(lambda w:5*loss(w))(w0)
np.testing.assert_allclose(scaled_gradient,5*base_gradient,rtol=1e-6)
reference=w0-.1*base_gradient
compensated=w0-(.1/5)*scaled_gradient
uncompensated=w0-.1*scaled_gradient
np.testing.assert_allclose(compensated,reference,rtol=1e-6)
assert not np.isclose(uncompensated,reference)
print('Reference / compensated / unchanged-rate updates:',reference,compensated,uncompensated)

# Reference practice: Diagnose a nonfinite input before an update
# Diagnose a nonfinite input before an update (Intermediate): Nonfinite data is a different cause than excessive step size.
corrupted=x.at[1].set(jnp.nan)
bad_value,bad_grad=jax.value_and_grad(lambda w:loss(w,corrupted,y))(0.)
assert not np.isfinite(float(bad_value)) and not np.isfinite(float(bad_grad))
repaired=jnp.array([1.,2.,3.])
repaired_value,repaired_grad=jax.value_and_grad(lambda w:loss(w,repaired,y))(0.)
np.testing.assert_allclose(repaired_grad,-56/3,rtol=1e-6)
assert np.isfinite(float(repaired_value))
print("PASS: networks-05")
