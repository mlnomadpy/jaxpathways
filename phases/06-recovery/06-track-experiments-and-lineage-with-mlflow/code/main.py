"""Track experiments and lineage with MLflow: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import os
# Configure environment variable before initializing the runtime.
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
# Configure environment variable before initializing the runtime.
os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
# Import required JAX, NumPy, and standard-library modules.
import hashlib, json, tempfile
from pathlib import Path
import numpy as np

# Step 2: Apply the core JAX transformation
import jax
import jax.numpy as jnp
import mlflow
from mlflow import MlflowClient
# Construct `x` via `jnp.array([-2., -1., 0., 1., 2.])`
x = jnp.array([-2., -1., 0., 1., 2.])
y = 2 * x + 1

# Step 3: Verify shapes and numerical invariants
vx = jnp.array([-1.5, -.5, .5, 1.5])
vy = 2 * vx + 1
# Aggregate array values to compute `objective`.
objective = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
# Differentiate the objective to obtain `update` via automatic differentiation.
update = jax.jit(lambda p, rate: p - rate * jax.grad(objective)(p))
# Convert `data_hash` to a host NumPy array for inspection or verification.
data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
# Compute `curves, validation, run_ids` from `[], [], []`
curves, validation, run_ids = [], [], []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    # Read or serialize artifact data on disk (`root`).
    root = Path(folder)
    # Run `mlflow.set_tracking_uri` to perform the next check or state transition.
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    # Run `MlflowClient` to compute `client`.
    client = MlflowClient()
    # Run `client.create_experiment` to compute `experiment`.
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    # Loop over `rate` in `[0.02, 0.15]`:
    for rate in [.02, .15]:
        # Allocate initialized array `p` with the specified shape and dtype.
        # Construct `p` via `jnp.zeros(2)`
        p = jnp.zeros(2)
        history = []
        # Enter managed runtime/context scope for this block:
        with mlflow.start_run(experiment_id=experiment) as run:
            # Run `mlflow.log_params` to perform the next check or state transition.
            mlflow.log_params({'rate': rate, 'updates': 30})
            # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for ``.
            mlflow.set_tags({'data_sha256': data_hash, 'jax': jax.__version__, 'backend': jax.default_backend(), 'metric_boundary': 'after update'})
            # Loop over `step` in `range(1, 31)`:
            for step in range(1, 31):
                # Run `update` to compute `p`.
                p = update(p, rate)
                # Evaluate `objective(p)` and convert the result into Python scalar/collection `metric`.
                # Execute the next step of the computation.
                metric = float(objective(p))
                history.append(metric)
                # Run `mlflow.log_metric` to perform the next check or state transition.
                mlflow.log_metric('train_mse', metric, step=step)
            # Reduce across the target axis to summarize `held`.
            held = float(jnp.mean((p[0] * vx + p[1] - vy) ** 2))
            # Independent host arithmetic checks what the reported validation metric means.
            oracle = sum((float(p[0]) * t + float(p[1]) - (2 * t + 1)) ** 2 for t in [-1.5, -.5, .5, 1.5]) / 4
            # Check numerical equivalence within tolerance: `abs(held - oracle) < 1e-6`
            assert abs(held - oracle) < 1e-6
            # Run `mlflow.log_metric` to perform the next check or state transition.
            mlflow.log_metric('validation_mse', held, step=30)
            # Run `mlflow.log_dict` to perform the next check or state transition.
            mlflow.log_dict({'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}, 'model.json')
            # Append the current step result to `run_ids`.
            run_ids.append(run.info.run_id)
        # Append the current step result to `curves`.
        # Append the current step result to `curves`.
        curves.append(history)
        validation.append(held)
        # Run `client.get_metric_history` to compute `saved`.
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        # Assert invariant `[m.step for m in saved] == list(range(1, 31))` holds
        assert [m.step for m in saved] == list(range(1, 31))
        # Execute `np.testing.assert_allclose([m.value for m in saved], history`
        np.testing.assert_allclose([m.value for m in saved], history)
        # Assert invariant `client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash` holds
        assert client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash
    # Run `client.search_runs` to compute `selected`.
    selected = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])[0]
    # Assert invariant `selected.info.run_id == run_ids[int(np.argmin(validation))]` holds
    assert selected.info.run_id == run_ids[int(np.argmin(validation))]
    # Run `client.download_artifacts` to compute `artifact`.
    artifact = client.download_artifacts(selected.info.run_id, 'model.json')
    # Read or serialize artifact data on disk (`restored`).
    restored = json.loads(Path(artifact).read_text())
    # Check numerical equivalence within tolerance: `abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) ...`
    assert abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) < .001

# Track experiments and lineage with MLflow: A tracker is useful when a plotted point leads back to the data,...
# Import os for this computation.
import os
# Configure environment variable before initializing the runtime.
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
# Configure environment variable before initializing the runtime.
os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
# Import required JAX, NumPy, and standard-library modules.
import hashlib, json, tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import mlflow
from mlflow import MlflowClient
# Construct `x` via `jnp.array([-2., -1., 0., 1., 2.])`
x = jnp.array([-2., -1., 0., 1., 2.])
y = 2 * x + 1
# Construct `vx` via `jnp.array([-1.5, -.5, .5, 1.5])`
vx = jnp.array([-1.5, -.5, .5, 1.5])
vy = 2 * vx + 1
# Aggregate array values to compute `objective`.
objective = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
# Differentiate the objective to obtain `update` via automatic differentiation.
update = jax.jit(lambda p, rate: p - rate * jax.grad(objective)(p))
# Convert `data_hash` to a host NumPy array for inspection or verification.
data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
# Compute `curves, validation, run_ids` from `[], [], []`
curves, validation, run_ids = [], [], []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    # Read or serialize artifact data on disk (`root`).
    root = Path(folder)
    # Run `mlflow.set_tracking_uri` to perform the next check or state transition.
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    # Run `MlflowClient` to compute `client`.
    client = MlflowClient()
    # Run `client.create_experiment` to compute `experiment`.
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    # Loop over `rate` in `[0.02, 0.15]`:
    for rate in [.02, .15]:
        # Allocate initialized array `p` with the specified shape and dtype.
        # Construct `p` via `jnp.zeros(2)`
        p = jnp.zeros(2)
        history = []
        # Enter managed runtime/context scope for this block:
        with mlflow.start_run(experiment_id=experiment) as run:
            # Run `mlflow.log_params` to perform the next check or state transition.
            mlflow.log_params({'rate': rate, 'updates': 30})
            # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for ``.
            mlflow.set_tags({'data_sha256': data_hash, 'jax': jax.__version__, 'backend': jax.default_backend(), 'metric_boundary': 'after update'})
            # Loop over `step` in `range(1, 31)`:
            for step in range(1, 31):
                # Run `update` to compute `p`.
                p = update(p, rate)
                # Evaluate `objective(p)` and convert the result into Python scalar/collection `metric`.
                # Execute the next step of the computation.
                metric = float(objective(p))
                history.append(metric)
                # Run `mlflow.log_metric` to perform the next check or state transition.
                mlflow.log_metric('train_mse', metric, step=step)
            # Reduce across the target axis to summarize `held`.
            held = float(jnp.mean((p[0] * vx + p[1] - vy) ** 2))
            # Independent host arithmetic checks what the reported validation metric means.
            oracle = sum((float(p[0]) * t + float(p[1]) - (2 * t + 1)) ** 2 for t in [-1.5, -.5, .5, 1.5]) / 4
            # Check numerical equivalence within tolerance: `abs(held - oracle) < 1e-6`
            assert abs(held - oracle) < 1e-6
            # Run `mlflow.log_metric` to perform the next check or state transition.
            mlflow.log_metric('validation_mse', held, step=30)
            # Run `mlflow.log_dict` to perform the next check or state transition.
            mlflow.log_dict({'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}, 'model.json')
            # Append the current step result to `run_ids`.
            run_ids.append(run.info.run_id)
        # Append the current step result to `curves`.
        # Append the current step result to `curves`.
        curves.append(history)
        validation.append(held)
        # Run `client.get_metric_history` to compute `saved`.
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        # Assert invariant `[m.step for m in saved] == list(range(1, 31))` holds
        assert [m.step for m in saved] == list(range(1, 31))
        # Execute `np.testing.assert_allclose([m.value for m in saved], history`
        np.testing.assert_allclose([m.value for m in saved], history)
        # Assert invariant `client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash` holds
        assert client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash
    # Run `client.search_runs` to compute `selected`.
    selected = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])[0]
    # Assert invariant `selected.info.run_id == run_ids[int(np.argmin(validation))]` holds
    assert selected.info.run_id == run_ids[int(np.argmin(validation))]
    # Run `client.download_artifacts` to compute `artifact`.
    artifact = client.download_artifacts(selected.info.run_id, 'model.json')
    # Read or serialize artifact data on disk (`restored`).
    restored = json.loads(Path(artifact).read_text())
    # Check numerical equivalence within tolerance: `abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) ...`
    assert abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) < .001
# Print the observed values to compare against the expected result.
print('Validation MSE by learning rate:', dict(zip([.02, .15], validation)))
# Print diagnostic summary of the computed outputs.
print('Verified 30 logged steps per run; selected and reloaded the better validation candidate.')

# Figure data experiment
# Compute figure data for: Compare the runs that MLflow actually recorded
# Compute `visual_data` from `{'kind':'line','x':list(range(1,31)),'xlabel':'compl...`
visual_data={'kind':'line','x':list(range(1,31)),'xlabel':'completed update','ylabel':'post-update training MSE (log scale)','yscale':'log','series':[{'label':'rate 0.02','y':curves[0]},{'label':'rate 0.15','y':curves[1]}]}

# Experiment: Check the stored comparison
# Experiment — Check the stored comparison: Both runs share initialization, data and update count.
# Assert invariant `validation[1] < validation[0]` holds
assert validation[1] < validation[0]
# Print the observed values to compare against the expected result.
print('Validation difference:',validation[0]-validation[1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a third, deliberately wrong model artifact and show why the...
bad_validation = sum((0. * t + 0. - (2*t+1))**2 for t in [-1.5,-.5,.5,1.5])/4
# Assert invariant `bad_validation > min(validation)` holds
assert bad_validation > min(validation)
# Print the observed values to compare against the expected result.
print('Wrong zero model validation MSE:',bad_validation)

# Reference practice: Change a data value
# Change a data value (transfer): Identical filenames or shapes do not imply identical data.
changed_y=np.asarray(y).copy()
changed_y[0]+=.1
# Convert `changed_hash` to a host NumPy array for inspection or verification.
changed_hash=hashlib.sha256(np.asarray(x).tobytes()+changed_y.tobytes()).hexdigest()
# Assert invariant `changed_hash != data_hash` holds
assert changed_hash != data_hash
# Print the observed values to compare against the expected result.
print('Changed target changes dataset identity.')
print("PASS: recovery-06")
