# A compiled train and evaluation step

Phase 05: Neural network training · about 95 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

**Before this lesson:** [Model and state with Flax NNX](https://www.tahabouhsine.com/jaxpathways/lesson-networks-02.html).

## What you will be able to do

- Choose a transparent train and held-out split
- Read the state transition in execution order
- Compute useful metrics without changing the model
- Make replay a controlled experiment

## The problem

Training changes a model; evaluation should measure it without changing it. We’ll write those two steps separately and compile them with Flax NNX. Then we’ll check the parameters and optimizer counter before and after evaluation, so this important promise is something we test rather than assume.

## The idea

Training computes gradients and changes the model. Evaluation measures a fixed model under a declared input and scoring contract. Keeping those roles separate prevents held-out data from quietly participating in fitting.

## Evaluation observes a frozen model

Suppose validation loss improves after an accidental optimizer step on validation examples. The number is real, but it no longer evaluates the unchanged training result. The evaluation has influenced fitting.

Capture parameter and relevant state values before evaluation and compare afterward. If the model has training-only randomness or running statistics, verify its evaluation convention too. Depict only the mechanisms the actual model uses.

Aggregate evaluation contributions with their counts. Unequal batch sizes make an average of batch means different from an example mean. Finally, inspect actual errors in the held-out prediction field: a visually attractive boundary cannot rule out leakage or a denominator mistake.

This classifier produces one binary logit $z_i$ per example, not a vector of class probabilities. The implementation evaluates the softplus term stably with Optax; the NumPy reference uses `logaddexp`. A zero logit gives loss $\log 2$ for either label. A positive logit lowers loss for label $1$ and raises it for label $0$. Both logits and labels must have shape $(B,)$, otherwise broadcasting can create an unintended pairwise loss.

$$
L=\frac1B\sum_{i=1}^B\left[\log(1+e^{z_i})-y_i z_i\right],\qquad z_i\in\mathbb R,\quad y_i\in\{0,1\}
$$

### Pause and reason

Why compare state before and after evaluation instead of checking only the score?

<details><summary>Compare your reasoning</summary>

A score can look plausible even if evaluation updates parameters or statistics. State comparison tests whether the measurement changed the object it was meant to evaluate.

</details>

## Choose a transparent train and held-out split

Generate separate arrays from fixed independent seeds. Labels follow the known line x0 + $0.5$ x1 > $0$, so there is no label ambiguity and no external dataset download. Forty-eight training rows and forty-eight held-out rows are intentionally tiny. The held-out distribution matches the training distribution; this measures interpolation on a controlled problem, not robustness under distribution shift. Keep held-out rows out of optimizer calls. Real projects also need validation for model selection and a final test set; this example fixes hyperparameters in advance and does not supply a production evaluation protocol.

## Read the state transition in execution order

`nnx.value_and_grad` computes loss and gradients for the current parameters. `optimizer.update(model, grads)` then changes the model and optimizer state. The returned loss is pre-update loss, not the loss at the new weights. `nnx.Optimizer` is configured with `wrt=nnx.Param`, and `optax.adam` maintains first and second moments plus an update counter. A model snapshot alone cannot exactly resume Adam training. Conversely, a metrics call should not increment optimizer.step. Counting $80$ requested updates checks loop behavior, while comparing full snapshots checks evaluation isolation.

```text
(model parameters, optimizer state, training batch) → loss/gradients → updated model and optimizer
(fixed model, held-out batch) → loss and accuracy only
```

## Compute useful metrics without changing the model

Binary cross-entropy uses scores directly. Accuracy thresholds scores at zero; sigmoid is unnecessary for this decision. Return both because confidence and hard decisions measure different behavior. A wrong but extremely confident prediction can dominate loss while changing accuracy by only one observation. We calculate held-out loss independently with NumPy logaddexp and held-out accuracy with NumPy comparisons. Capture every parameter array before evaluation and compare bytes afterward. This model has no dropout or batch normalization; later stateful models require explicit evaluation mode and nonparameter-state checks too.

## Make replay a controlled experiment

Repeat the full initialization and training routine from the same seed and batches. Compare final parameter arrays within tolerance and compare held-out metrics. This is stronger than checking that one seeded initialization looks similar. It verifies the sequence of updates, but only in the tested CPU environment. Floating-point reductions, package versions, and backend changes may affect results. Store model seed, data seeds, batch policy, learning rate, iteration count, dependency versions, and metrics. Do not use a different initialization as a supposed resume of an existing optimizer state.

## Extend the frozen check to BatchNorm and Dropout

The classifier above has no mutable forward state, so its exact state snapshot is a clean first check. This bounded extension adds `nnx.BatchNorm` running mean and variance plus `nnx.Dropout`. `training=True` uses batch statistics and updates `nnx.BatchStat`; it also requests a dropout mask and advances the dropout RNG count/key. `training=False` tells BatchNorm to use its stored running statistics and tells Dropout to return deterministic activations without consuming a key. The optimizer updates only `nnx.Param`. Compare all three categories around repeated held-out calls: parameters, BatchStat, and RngState. Then use a disposable clone as a negative control with `training=True`; its batch statistics and dropout key should change, proving the checks can detect the wrong mode. This synthetic batch is a mechanics test, not an accuracy benchmark.

### Pause and reason

What must a frozen evaluation check compare when the model has BatchNorm and Dropout?

<details><summary>Compare your reasoning</summary>

Compare parameter arrays, BatchStat running values, and RngState key/count before and after deterministic evaluation. A score alone cannot show that those states stayed fixed.

</details>

## 1. Construct data, model, and optimizer

Create main.py in the training dependency environment. The generator returns separate train and held-out arrays.

```python
# Step 1 : 1. Construct data, model, and optimizer
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def dataset(seed,n=48):
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    return jnp.array(a),jnp.array(b)
train_x,train_y=dataset(10)
test_x,test_y=dataset(11)
# Define `Classifier` module / container with explicit state and forward pass:
class Classifier(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        self.out=nnx.Linear(8,1,rngs=rngs)
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
def initialize():
    m=Classifier(0)
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
model,optimizer=initialize()
```

The held-out seed is $11$; every optimizer call below uses seed-10 data only.

## 2. Define the two compiled functions

Append train and evaluation functions. Notice that only train_step receives an optimizer.

```python
# Step 2 : 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Apply the parameter gradients to the mutable Flax model and optimizer state.
    o.update(m,grads)
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80
```

The 80-update counter is an optimizer invariant. Pre-update losses returned during the loop are not the final held-out metric.

## 3. Verify held-out metrics and state isolation

Append snapshot and reference comparisons. Print both training and held-out results instead of a single success number.

```python
# Step 3 : 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
# Transfer the device result to the host for an independent NumPy comparison.
# Transfer the device result to the host for an independent NumPy comparison.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))
```

The synthetic fixture normally exceeds $0.85$ held-out accuracy. Parameter bytes and optimizer count remain unchanged during evaluation.

## Run the example

```python
# Step 1 : 1. Construct data, model, and optimizer
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
def dataset(seed,n=48):
    a=np.random.default_rng(seed).normal(size=(n,2)).astype(np.float32)
    b=(a[:,0]+.5*a[:,1]>0).astype(np.float32)
    return jnp.array(a),jnp.array(b)
train_x,train_y=dataset(10)
test_x,test_y=dataset(11)
# Define `Classifier` module / container with explicit state and forward pass:
class Classifier(nnx.Module):
    def __init__(self,seed=0):
        rngs=nnx.Rngs(seed)
        self.hidden=nnx.Linear(2,8,rngs=rngs)
        self.out=nnx.Linear(8,1,rngs=rngs)
    def __call__(self,x):return self.out(nnx.tanh(self.hidden(x))).squeeze(-1)
def objective(m,x,y):return jnp.mean(optax.sigmoid_binary_cross_entropy(m(x),y))
def initialize():
    m=Classifier(0)
    return m,nnx.Optimizer(m,optax.adam(.03),wrt=nnx.Param)
model,optimizer=initialize()

# Step 2 : 2. Define the two compiled functions: The 80-update counter is an optimizer invariant.
# Define and JIT-compile `train_step(m, o, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def train_step(m,o,x,y):
    value,grads=nnx.value_and_grad(objective)(m,x,y)
    # Apply the parameter gradients to the mutable Flax model and optimizer state.
    o.update(m,grads)
    return value
# Define and JIT-compile `evaluate(m, x, y)` so XLA traces and fuses the operations:
@nnx.jit
def evaluate(m,x,y):
    scores=m(x)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(scores,y)),jnp.mean((scores>0)==y)
def snapshot(m):return [np.array(a,copy=True) for a in jax.tree.leaves(nnx.state(m))]
before_loss=float(objective(model,train_x,train_y))
for _ in range(80):train_step(model,optimizer,train_x,train_y)
assert int(optimizer.step[...])==80

# Step 3 : 3. Verify held-out metrics and state isolation: The synthetic fixture normally exceeds 0.85 held-out accuracy.
frozen=snapshot(model)
step_count=int(optimizer.step[...])
test_loss,test_accuracy=evaluate(model,test_x,test_y)
for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
assert int(optimizer.step[...])==step_count
# Transfer the device result to the host for an independent NumPy comparison.
# Transfer the device result to the host for an independent NumPy comparison.
scores=np.asarray(model(test_x))
labels=np.asarray(test_y)
np.testing.assert_allclose(test_loss,np.mean(np.logaddexp(0.,scores)-labels*scores),rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(test_accuracy,np.mean((scores>0)==labels),atol=1e-6)
assert float(test_accuracy)>.85 and float(objective(model,train_x,train_y))<before_loss*.3
print('Updates:',step_count,'held-out loss:',float(test_loss),'accuracy:',float(test_accuracy))
```

Expected: $80$ optimizer updates; held-out accuracy above $0.85$; evaluation snapshots unchanged. Exact scalar metrics are printed by your run.

## Inspect predictions on held-out inputs

**Predict:** Where would a mistake appear relative to the decision boundary?

![Inspect predictions on held-out inputs](../../phases/05-networks/03-a-compiled-train-and-evaluation-step/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal and vertical axes are input features. The background is the trained model’s probability of label $1$, from pale near $0$ to dark purple near $1$. Circles mark held-out examples with label $0$, and triangles mark held-out examples with label $1$.

A slanted transition separates a mostly pale left side from a dark right side. The circles lie on the low-probability side and the triangles on the high-probability side. The example’s direct evaluation reports accuracy $1$ on this fixed held-out set; the figure shows where those correct decisions occur.

### Connect it to the computation

The data rule assigns label $1$ when $x_0+0.5x_1>0$, whose boundary is $x_1=-2x_0$. That explains the overall downward slant. The network approximates this separation; its transition need not be perfectly straight everywhere.

To inspect a mistake in a changed run, look for a triangle where the probability is below $0.5$, or a circle where it is above $0.5$, and check the model at that exact point. Grid colors are sampled approximations. Perfect accuracy on this small synthetic set is separate from accuracy outside it, and extreme probabilities do not by themselves establish calibration.

```python
axis = jnp.linspace(-3.0, 3.0, 61)
gx, gy = jnp.meshgrid(axis, axis)
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
# Rearrange tensor axes to match the required layout for `prob`.
prob = jax.nn.sigmoid(model(grid)).reshape(gx.shape)
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-3.0, 3.0, -3.0, 3.0], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': test_x.tolist(), 'labels': test_y.tolist()}
```

## Recorded reference execution

CPU run: 2026-10-27T12:29:42.913814+00:00. JAX 0.9.2.

```text
Updates: 80 held-out loss: 0.018426384776830673 accuracy: 1.0
Updates: 80 held-out loss: 0.018426384776830673 accuracy: 1.0
Deterministic evaluation repeated exactly; Param, BatchStat, and RngState unchanged. Negative control changed BatchStat and RngState; parameters stayed separate.
Count-weighted held-out metrics: [0.01842638 0.99999995] constructed accuracy: 0.5416666666666666
PASS: networks-03

```

## Replay the complete update sequence

**Predict before running:** Should equal seed, batches, and update count reproduce the final model?

```python
# Experiment : Replay the complete update sequence: Replay reconstructs optimizer state from the start.
replay,replay_optimizer=initialize()
for _ in range(80):train_step(replay,replay_optimizer,train_x,train_y)
for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(evaluate(replay,test_x,test_y),evaluate(model,test_x,test_y),rtol=1e-6)
```

**Expected:** Final parameters and held-out metrics agree within CPU tolerances.

Replay reconstructs optimizer state from the start. It is not a checkpoint-resume implementation.

## Duplicating evaluation data

**Predict before running:** If the held-out rows are repeated twice, should mean loss or accuracy change?

```python
# Experiment : Duplicating evaluation data: Means normalize the observation count.
doubled=evaluate(model,jnp.concatenate([test_x,test_x]),jnp.concatenate([test_y,test_y]))
np.testing.assert_allclose(doubled,evaluate(model,test_x,test_y),rtol=1e-5,atol=1e-6)
```

**Expected:** Both metrics stay the same.

Means normalize the observation count. Duplicating data does not create independent evaluation evidence.

## Stateful evaluation: running statistics and dropout keys

**Predict before running:** After training forwards, will two `training=False` calls return equal outputs and preserve parameters, running statistics, and dropout RNG state? Which category changes in the negative control?

```python
# Experiment : Stateful evaluation: BatchNorm statistics and dropout RNG are state, distinct from parameters.
from flax import nnx as state_nnx
import jax as state_jax
import jax.numpy as state_jnp
import numpy as state_np
import optax as state_optax
class StatefulProbe(state_nnx.Module):
    def __init__(self, seed=4):
        rngs = state_nnx.Rngs(params=seed, dropout=seed+1)
        self.hidden = state_nnx.Linear(2, 4, rngs=rngs)
        self.norm = state_nnx.BatchNorm(num_features=4, rngs=rngs)
        self.dropout = state_nnx.Dropout(rate=0.5, rngs=rngs)
        self.out = state_nnx.Linear(4, 1, rngs=rngs)
    def __call__(self, x, *, training):
        x = self.norm(self.hidden(x), use_running_average=not training)
        x = self.dropout(state_nnx.relu(x), deterministic=not training)
        return self.out(x).squeeze(-1)
def stateful_loss(model, x, y):
    return state_jnp.mean(state_optax.sigmoid_binary_cross_entropy(model(x, training=True), y))
@state_nnx.jit
def stateful_train_step(model, optimizer, x, y):
    loss, grads = state_nnx.value_and_grad(stateful_loss)(model, x, y)
    optimizer.update(model, grads)
    return loss
def copy_state(state):
    return state_jax.tree.map(lambda value: state_jnp.array(value, copy=True), state)
def same_state(left, right):
    def comparable(value):
        if state_jax.dtypes.issubdtype(value.dtype, state_jax.dtypes.prng_key):
            return state_jax.random.key_data(value)
        return value
    return all(bool(state_np.array_equal(comparable(a), comparable(b))) for a, b in zip(state_jax.tree.leaves(left), state_jax.tree.leaves(right)))
state_x = state_jnp.asarray([[0., 1.], [1., 2.], [-1., 1.], [2., -1.], [1., -2.], [-2., 0.], [0., -1.], [2., 2.]])
state_y = state_jnp.asarray([1., 1., 0., 1., 0., 0., 0., 1.])
state_model = StatefulProbe()
state_optimizer = state_nnx.Optimizer(state_model, state_optax.sgd(0.02), wrt=state_nnx.Param)
for _ in range(3):
    stateful_train_step(state_model, state_optimizer, state_x, state_y)
params_before = copy_state(state_nnx.state(state_model, state_nnx.Param))
stats_before = copy_state(state_nnx.state(state_model, state_nnx.BatchStat))
rng_before = copy_state(state_nnx.state(state_model, state_nnx.RngState))
eval_a = state_model(state_x, training=False)
eval_b = state_model(state_x, training=False)
state_np.testing.assert_array_equal(eval_a, eval_b)
assert same_state(params_before, state_nnx.state(state_model, state_nnx.Param))
assert same_state(stats_before, state_nnx.state(state_model, state_nnx.BatchStat))
assert same_state(rng_before, state_nnx.state(state_model, state_nnx.RngState))
# Negative control: training mode on an independent clone must change BatchStat and consume a dropout key.
graphdef, state_snapshot = state_nnx.split(state_model)
bad_mode_model = state_nnx.merge(graphdef, copy_state(state_snapshot))
bad_stats = copy_state(state_nnx.state(bad_mode_model, state_nnx.BatchStat))
bad_rng = copy_state(state_nnx.state(bad_mode_model, state_nnx.RngState))
_ = bad_mode_model(state_x, training=True)
assert not same_state(bad_stats, state_nnx.state(bad_mode_model, state_nnx.BatchStat))
assert not same_state(bad_rng, state_nnx.state(bad_mode_model, state_nnx.RngState))
assert same_state(params_before, state_nnx.state(state_model, state_nnx.Param))
print('Deterministic evaluation repeated exactly; Param, BatchStat, and RngState unchanged. Negative control changed BatchStat and RngState; parameters stayed separate.')
```

**Expected:** Repeated evaluation outputs match exactly. Param, BatchStat, and RngState snapshots are unchanged. The negative-control training-mode call on a copied model changes BatchStat and RngState while leaving Param unchanged.

NNX categories answer different questions: `Param` is optimizer-targeted, `BatchStat` holds running estimates, and `RngState` contains key/counter values. Deterministic evaluation neither updates running statistics nor requests dropout randomness. The negative control validates that the snapshots can detect a mode mistake.

## Make it yours

Compute held-out metrics in three equal chunks and reconstruct the overall means. Then confirm no state changed.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- A model and held-out arrays split into chunks with a final partial batch.

**Outputs:**
- Count-weighted aggregate loss/accuracy and an unchanged model snapshot.

**Invariants to check:**
- Metrics are weighted by example counts; evaluation does not update model or optimizer state.

**Hints:**
- Accumulate metric sums and counts, then divide once.
- Use a separate clone when deliberately testing a training call on held-out data.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def evaluate_in_chunks(model, features, labels, chunk_size):
    """Count-weighted aggregate loss/accuracy and an unchanged model snapshot."""
    # TODO: Aggregate held-out metric sums by example count and assert evaluation leaves model state unchanged.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
old=snapshot(model)
pieces=[evaluate(model,test_x[i:i+16],test_y[i:i+16]) for i in range(0,48,16)]
# Reduce along axis=0 to compute `aggregate`.
aggregate=np.mean(np.asarray(pieces),axis=0)
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(aggregate,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
for a,b in zip(old,snapshot(model)):np.testing.assert_array_equal(a,b)
```

</details>

## Aggregate unequal batches

**Transfer**

Evaluate the held-out set in batches of seven and forty-one observations. Reconstruct both metrics from sums and counts. Use a separate constructed count example to demonstrate why averaging batch accuracies can be wrong.

<details><summary>Hint</summary>

Weight each batch mean by its number of observations.

</details>

### How to write: Aggregate unequal batches — Step-by-step recipe & starter scaffold

**Inputs:**
- Evaluate the held-out set in batches of seven and forty-one observations. Reconstruct both metrics from sums and counts. Use a separate constructed count example to demonstrate why averaging batch accuracies can be wrong.

**Outputs:**
- Count-weighted aggregate metrics and a constructed unequal-count accuracy example.

**Invariants to check:**
- Batch metrics are weighted by the number of examples, not averaged equally.

**Hints:**
- Weight each batch mean by its number of observations.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp

def practice_aggregate_unequal_batches(model, features, labels, batch_sizes):
    """Implement the aggregate unequal batches check."""
    # TODO: Accumulate metric sums and example counts for batch sizes seven and forty-one, then compare with concatenated evaluation.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
sizes=np.array([7,41])
# Transfer the device result to the host for an independent NumPy comparison.
metrics=np.asarray([evaluate(model,test_x[:7],test_y[:7]),evaluate(model,test_x[7:],test_y[7:])])
# Reduce along axis=0 to compute `weighted`.
weighted=(metrics*sizes[:,None]).sum(axis=0)/sizes.sum()
# Transfer the device result to the host for an independent NumPy comparison.
np.testing.assert_allclose(weighted,np.asarray(evaluate(model,test_x,test_y)),rtol=1e-5,atol=1e-6)
correct=np.array([6,20])
counts=np.array([7,41])
expected=26/48
np.testing.assert_allclose(correct.sum()/counts.sum(),expected)
assert not np.isclose(np.mean(correct/counts),expected)
print('Count-weighted held-out metrics:',weighted,'constructed accuracy:',expected)
```

A batch is a packaging choice, not a unit of evidence. The constructed counts expose the bias even if a particular model happens to obtain equal accuracy on both actual batches.

</details>

## Catch an evaluation function that trains

**Intermediate**

On an independent clone, deliberately call train_step with held-out data and show the isolation checks fail. Keep the real model untouched.

<details><summary>Hint</summary>

Compare both optimizer.step and model parameters around the bad call.

</details>

### How to write: Catch an evaluation function that trains — Step-by-step recipe & starter scaffold

**Inputs:**
- On an independent clone, deliberately call train_step with held-out data and show the isolation checks fail. Keep the real model untouched.

**Outputs:**
- A detected clone mutation and an unchanged real model snapshot.

**Invariants to check:**
- The model used for held-out evaluation and the model used for the deliberate failure are distinct.

**Hints:**
- Compare both optimizer.step and model parameters around the bad call.

**Starter code scaffold (fill in the TODOs):**

```python
from flax import nnx
import jax.numpy as jnp

def practice_catch_an_evaluation_function_that_trains(model, optimizer, heldout_features, heldout_labels):
    """Implement the catch an evaluation function that trains check."""
    # TODO: Run the mutating train_step only on a disposable clone and compare its state snapshot before and after evaluation.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Catch an evaluation function that trains (Intermediate): A plausible returned loss is separate from evaluation.
graph,state=nnx.split(model)
bad_model=nnx.merge(graph,jax.tree.map(lambda a:jnp.array(a,copy=True),state))
bad_optimizer=nnx.Optimizer(bad_model,optax.adam(.03),wrt=nnx.Param)
prior=snapshot(bad_model)
train_step(bad_model,bad_optimizer,test_x,test_y)
assert int(bad_optimizer.step[...])==1
assert any(not np.array_equal(a,b) for a,b in zip(prior,snapshot(bad_model)))
assert int(optimizer.step[...])==80
```

A plausible returned loss is separate from evaluation. Passing held-out data through the update function leaks it into learned state.

</details>

## Check your understanding

The train step returns a loss before optimizer.update. Which parameters does it describe?

1. The parameters entering that update
2. The final parameters after all updates
3. A separate held-out model

<details><summary>Answer and explanation</summary>

**1. The parameters entering that update (correct)**

The order of statements matters: value_and_grad runs before applying the update. Use a fresh evaluation call to measure final parameters.

**2. The final parameters after all updates**

The computed loss belongs to the parameters used before that update. Recompute after updating if you want a post-update loss.

**3. A separate held-out model**

The data used in the loss determines whether it is held out. Calling optimizer.update does not create a validation split.

</details>

## Diagnose the result

If metrics look suspiciously good, trace which arrays reach train_step. If evaluation modifies the model, compare snapshots and optimizer counts. If replay fails, compare data seeds, iteration counts, optimizer initialization, and versions before guessing that compilation is nondeterministic. For stateful modules, compare `nnx.Param`, `nnx.BatchStat`, and `nnx.RngState` separately around evaluation. A deterministic score repeated twice is useful but does not replace the state snapshots; the training-mode negative control checks that the snapshots detect mutation.

## Carry forward

- Choose a transparent train and held-out split
- Read the state transition in execution order
- Compute useful metrics without changing the model
- Make replay a controlled experiment

## Keep your evidence

Keep pre/post parameter snapshots around evaluation, optimizer step count 80, independent held-out loss and accuracy, complete replay comparison, and a deliberately bad held-out update on an isolated clone.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [NNX optimizer API](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/optimizer.html)
- [NNX transforms](https://flax.readthedocs.io/en/latest/guides/transforms.html)

