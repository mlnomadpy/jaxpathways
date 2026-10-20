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
curves, validation, run_ids = [], [], []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    root = Path(folder)
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    client = MlflowClient()
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    for rate in [.02, .15]:
        # Allocate initialized array `p` with the specified shape and dtype.
        p = jnp.zeros(2)
        history = []
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.log_params({'rate': rate, 'updates': 30})
            # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for ``.
            mlflow.set_tags({'data_sha256': data_hash, 'jax': jax.__version__, 'backend': jax.default_backend(), 'metric_boundary': 'after update'})
            for step in range(1, 31):
                p = update(p, rate)
                metric = float(objective(p))
                history.append(metric)
                mlflow.log_metric('train_mse', metric, step=step)
            held = float(jnp.mean((p[0] * vx + p[1] - vy) ** 2))
            # Independent host arithmetic checks what the reported validation metric means.
            oracle = sum((float(p[0]) * t + float(p[1]) - (2 * t + 1)) ** 2 for t in [-1.5, -.5, .5, 1.5]) / 4
            assert abs(held - oracle) < 1e-6
            mlflow.log_metric('validation_mse', held, step=30)
            mlflow.log_dict({'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}, 'model.json')
            run_ids.append(run.info.run_id)
        curves.append(history)
        validation.append(held)
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        assert [m.step for m in saved] == list(range(1, 31))
        # Execute `np.testing.assert_allclose([m.value for m in saved], history`
        np.testing.assert_allclose([m.value for m in saved], history)
        assert client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash
    selected = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])[0]
    assert selected.info.run_id == run_ids[int(np.argmin(validation))]
    artifact = client.download_artifacts(selected.info.run_id, 'model.json')
    restored = json.loads(Path(artifact).read_text())
    assert abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) < .001

# Track experiments and lineage with MLflow: A tracker is useful when a plotted point leads back to the data,...
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
x = jnp.array([-2., -1., 0., 1., 2.])
y = 2 * x + 1
vx = jnp.array([-1.5, -.5, .5, 1.5])
vy = 2 * vx + 1
# Aggregate array values to compute `objective`.
objective = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
# Differentiate the objective to obtain `update` via automatic differentiation.
update = jax.jit(lambda p, rate: p - rate * jax.grad(objective)(p))
# Convert `data_hash` to a host NumPy array for inspection or verification.
data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
curves, validation, run_ids = [], [], []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    root = Path(folder)
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    client = MlflowClient()
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    for rate in [.02, .15]:
        # Allocate initialized array `p` with the specified shape and dtype.
        p = jnp.zeros(2)
        history = []
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.log_params({'rate': rate, 'updates': 30})
            # Check which hardware backend (`cpu`, `gpu`, or `tpu`) JAX selected for ``.
            mlflow.set_tags({'data_sha256': data_hash, 'jax': jax.__version__, 'backend': jax.default_backend(), 'metric_boundary': 'after update'})
            for step in range(1, 31):
                p = update(p, rate)
                metric = float(objective(p))
                history.append(metric)
                mlflow.log_metric('train_mse', metric, step=step)
            held = float(jnp.mean((p[0] * vx + p[1] - vy) ** 2))
            # Independent host arithmetic checks what the reported validation metric means.
            oracle = sum((float(p[0]) * t + float(p[1]) - (2 * t + 1)) ** 2 for t in [-1.5, -.5, .5, 1.5]) / 4
            assert abs(held - oracle) < 1e-6
            mlflow.log_metric('validation_mse', held, step=30)
            mlflow.log_dict({'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}, 'model.json')
            run_ids.append(run.info.run_id)
        curves.append(history)
        validation.append(held)
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        assert [m.step for m in saved] == list(range(1, 31))
        # Execute `np.testing.assert_allclose([m.value for m in saved], history`
        np.testing.assert_allclose([m.value for m in saved], history)
        assert client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash
    selected = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])[0]
    assert selected.info.run_id == run_ids[int(np.argmin(validation))]
    artifact = client.download_artifacts(selected.info.run_id, 'model.json')
    restored = json.loads(Path(artifact).read_text())
    assert abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) < .001
print('Validation MSE by learning rate:', dict(zip([.02, .15], validation)))
print('Verified 30 logged steps per run; selected and reloaded the better validation candidate.')

# Figure data experiment
# Compute figure data for: Compare the runs that MLflow actually recorded
visual_data={'kind':'line','x':list(range(1,31)),'xlabel':'completed update','ylabel':'post-update training MSE (log scale)','yscale':'log','series':[{'label':'rate 0.02','y':curves[0]},{'label':'rate 0.15','y':curves[1]}]}

# Experiment: Check the stored comparison
# Experiment — Check the stored comparison: Both runs share initialization, data and update count.
assert validation[1] < validation[0]
print('Validation difference:',validation[0]-validation[1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a third, deliberately wrong model artifact and show why the...
bad_validation = sum((0. * t + 0. - (2*t+1))**2 for t in [-1.5,-.5,.5,1.5])/4
assert bad_validation > min(validation)
print('Wrong zero model validation MSE:',bad_validation)

# Reference practice: Change a data value
# Change a data value (transfer): Identical filenames or shapes do not imply identical data.
changed_y=np.asarray(y).copy()
changed_y[0]+=.1
# Convert `changed_hash` to a host NumPy array for inspection or verification.
changed_hash=hashlib.sha256(np.asarray(x).tobytes()+changed_y.tobytes()).hexdigest()
assert changed_hash != data_hash
print('Changed target changes dataset identity.')
print("PASS: recovery-06")
