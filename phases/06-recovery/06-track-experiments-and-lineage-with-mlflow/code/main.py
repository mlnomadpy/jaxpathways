"""Track experiments and lineage with MLflow: worked experiments and reference solutions. CPU checks."""

# Set MLflow options and load hashing and serialization tools
import os
# Set MLFLOW_DISABLE_AGENT_HINT before importing its runtime.
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
# Set MLFLOW_ENABLE_TELEMETRY before importing its runtime.
os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
import hashlib, json, tempfile
from pathlib import Path
import numpy as np

# Define the synthetic regression data and JAX optimizer
import jax
import jax.numpy as jnp
import mlflow
from mlflow import MlflowClient
x = jnp.array([-2., -1., 0., 1., 2.])
y = 2 * x + 1

# Log, read back, and independently validate two learning-rate runs
vx = jnp.array([-1.5, -.5, .5, 1.5])
vy = 2 * vx + 1
objective = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
update = jax.jit(lambda p, rate: p - rate * jax.grad(objective)(p))
# Transfer the device result to the host for an independent NumPy comparison.
data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
curves, validation, run_ids = [], [], []
candidate_artifacts = []
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    root = Path(folder)
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    client = MlflowClient()
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    for rate in [.02, .15]:
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
            model_record = {'kind': 'linear', 'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}
            candidate_artifacts.append(model_record)
            mlflow.log_dict(model_record, 'model.json')
            run_ids.append(run.info.run_id)
        curves.append(history)
        validation.append(held)
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        assert [m.step for m in saved] == list(range(1, 31))
        np.testing.assert_allclose([m.value for m in saved], history)
        assert client.get_run(run_ids[-1]).data.tags['data_sha256'] == data_hash
    selected = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])[0]
    assert selected.info.run_id == run_ids[int(np.argmin(validation))]
    artifact = client.download_artifacts(selected.info.run_id, 'model.json')
    restored = json.loads(Path(artifact).read_text())
    assert abs(restored['weight'] - 2) < .001 and abs(restored['bias'] - 1) < .001

# Track experiments and lineage with MLflow
import os
# Set MLFLOW_DISABLE_AGENT_HINT before importing its runtime.
os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
# Set MLFLOW_ENABLE_TELEMETRY before importing its runtime.
os.environ['MLFLOW_ENABLE_TELEMETRY'] = 'false'
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
objective = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
update = jax.jit(lambda p, rate: p - rate * jax.grad(objective)(p))
# Transfer the device result to the host for an independent NumPy comparison.
data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
curves, validation, run_ids = [], [], []
candidate_artifacts = []
# Keep this run’s database and artifacts in a scratch directory removed on exit.
with tempfile.TemporaryDirectory(prefix='tracking-lesson-') as folder:
    root = Path(folder)
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'tracking.db'))
    client = MlflowClient()
    experiment = client.create_experiment('two-rates', artifact_location=(root / 'artifacts').as_uri())
    for rate in [.02, .15]:
        p = jnp.zeros(2)
        history = []
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.log_params({'rate': rate, 'updates': 30})
            # Record the selected backend with the experiment metadata.
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
            model_record = {'kind': 'linear', 'weight': float(p[0]), 'bias': float(p[1]), 'preprocessing': 'one unscaled scalar feature'}
            candidate_artifacts.append(model_record)
            mlflow.log_dict(model_record, 'model.json')
            run_ids.append(run.info.run_id)
        curves.append(history)
        validation.append(held)
        saved = client.get_metric_history(run_ids[-1], 'train_mse')
        assert [m.step for m in saved] == list(range(1, 31))
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
visual_data={'kind':'line','x':list(range(1,31)),'xlabel':'completed update','ylabel':'post-update training MSE (log scale)','yscale':'log','series':[{'label':'rate 0.02','y':curves[0]},{'label':'rate 0.15','y':curves[1]}]}

# Experiment: Check the stored comparison
# Experiment — Check the stored comparison: Both runs share initialization, data and update count.
assert validation[1] < validation[0]
print('Validation difference:',validation[0]-validation[1])

# Reference solution. Try the exercise before reading this.
# Exercise solution: a memorizer can win training loss and fail held-out selection.
def predict_artifact(record, values):
    values = np.asarray(values, dtype=np.float64)
    if record['kind'] == 'linear':
        return record['weight'] * values + record['bias']
    if record['kind'] == 'training-lookup':
        mapping = dict(zip(record['training_x'], record['training_y']))
        return np.array([mapping.get(float(value), record['unseen_default']) for value in values])
    raise ValueError('Unknown model artifact kind')

lookup = {'kind': 'training-lookup', 'training_x': np.asarray(x).tolist(),
          'training_y': np.asarray(y).tolist(), 'unseen_default': 0.0}
models = candidate_artifacts + [lookup]
names = ['linear-rate-0.02', 'linear-rate-0.15', 'training-lookup']
assert len(models) == 3
train_errors, held_errors, replay_ids = [], [], []
with tempfile.TemporaryDirectory(prefix='selection-exercise-') as folder:
    selection_root = Path(folder)
    mlflow.set_tracking_uri('sqlite:///' + str(selection_root / 'tracking.db'))
    selection_client = MlflowClient()
    selection_experiment = selection_client.create_experiment(
        'three-candidate-selection', artifact_location=(selection_root / 'artifacts').as_uri())
    for name, record in zip(names, models):
        train_error = float(np.mean((predict_artifact(record, x) - np.asarray(y)) ** 2))
        held_error = float(np.mean((predict_artifact(record, vx) - np.asarray(vy)) ** 2))
        with mlflow.start_run(experiment_id=selection_experiment) as run:
            mlflow.log_params({'candidate': name, 'model_kind': record['kind']})
            mlflow.set_tags({'data_sha256': data_hash,
                             'scope': 'same synthetic data; validation never used to fit lookup or linear candidates'})
            mlflow.log_metric('train_mse', train_error)
            mlflow.log_metric('validation_mse', held_error)
            mlflow.log_dict(record, 'model.json')
            replay_ids.append(run.info.run_id)
        loaded_path = selection_client.download_artifacts(replay_ids[-1], 'model.json')
        loaded_record = json.loads(Path(loaded_path).read_text())
        np.testing.assert_array_equal(predict_artifact(loaded_record, vx), predict_artifact(record, vx))
        train_errors.append(train_error)
        held_errors.append(held_error)
    assert len(selection_client.search_runs([selection_experiment])) == 3
    training_pick = selection_client.search_runs([selection_experiment], order_by=['metrics.train_mse ASC'])[0]
    validation_pick = selection_client.search_runs([selection_experiment], order_by=['metrics.validation_mse ASC'])[0]
    assert training_pick.info.run_id == replay_ids[2]
    assert validation_pick.info.run_id == replay_ids[int(np.argmin(held_errors))]
    assert validation_pick.info.run_id != training_pick.info.run_id
    assert train_errors[2] == 0.0
    np.testing.assert_allclose(held_errors[2], 6.0)
    assert held_errors[2] > min(held_errors)
print('Actual training MSE:', dict(zip(names, train_errors)))
print('Actual held-out MSE:', dict(zip(names, held_errors)))
print('Training-only selects the memorizer; held-out selection rejects it.')


# Reference practice: Change a data value
# Change a data value (transfer): Identical filenames or shapes do not imply identical data.
changed_y=np.asarray(y).copy()
changed_y[0]+=.1
# Transfer the device result to the host for an independent NumPy comparison.
changed_hash=hashlib.sha256(np.asarray(x).tobytes()+changed_y.tobytes()).hexdigest()
assert changed_hash != data_hash
print('Changed target changes dataset identity.')
print("PASS: recovery-06")
