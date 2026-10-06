"""Run real MLflow tracking, pyfunc registry aliases, prompt versions and nested traces locally."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MLFLOW_ENABLE_ASYNC_LOGGING', 'false')
os.environ.setdefault('MLFLOW_ENABLE_ASYNC_TRACE_LOGGING', 'false')
os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY', 'false')
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature


class LinearModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = json.loads(Path(context.artifacts['weights']).read_text())
    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input)
        if x.ndim != 2 or x.shape[1] != 1 or not np.isfinite(x).all():
            raise ValueError('expected finite (batch,1) inputs')
        return self.model['weight'] * x[:, 0] + self.model['bias']


def run(folder):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    uri = 'sqlite:///' + str(folder / 'mlflow.db')
    mlflow.set_tracking_uri(uri); mlflow.set_registry_uri(uri)
    client = MlflowClient()
    experiment = client.create_experiment('engineering-release', artifact_location=(folder / 'artifacts').as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    x = jnp.array([-2., -1., 0., 1., 2.]); y = 2 * x + 1
    vx = jnp.array([-1.5, -.5, .5, 1.5]); vy = 2 * vx + 1
    loss = lambda p: jnp.mean((p[0] * x + p[1] - y) ** 2)
    update = jax.jit(lambda p, rate: p - rate * jax.grad(loss)(p))
    data_hash = hashlib.sha256(np.asarray(x).tobytes() + np.asarray(y).tobytes()).hexdigest()
    candidates = []
    for rate in [.02, .15]:
        with mlflow.start_run(experiment_id=experiment, run_name=f'rate-{rate}') as active:
            mlflow.log_params({'learning_rate': rate, 'steps': 30, 'seed': 0})
            mlflow.set_tags({'data_sha256': data_hash, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'backend': jax.default_backend(), 'validation_ids': 'v0,v1,v2,v3'})
            p = jnp.zeros(2)
            for step in range(30):
                p = update(p, rate)
                mlflow.log_metric('train_mse', float(loss(p)), step=step + 1)
            validation = float(jnp.mean((p[0] * vx + p[1] - vy) ** 2))
            mlflow.log_metric('validation_mse', validation, step=30)
            model = {'schema': 1, 'weight': float(p[0]), 'bias': float(p[1])}
            raw = json.dumps(model, sort_keys=True).encode(); artifact = folder / f'model-{rate}.json'; artifact.write_bytes(raw)
            mlflow.log_dict({'sha256': hashlib.sha256(raw).hexdigest(), 'preprocessing': 'one unscaled scalar feature', 'jax': jax.__version__, 'mlflow': mlflow.__version__}, 'manifest.json')
            info = mlflow.pyfunc.log_model(name='linear-model', python_model=LinearModel(), artifacts={'weights': str(artifact)}, signature=infer_signature(np.array([[0.], [1.]]), np.array([model['bias'], model['weight'] + model['bias']])), input_example=np.array([[0.], [1.]]), pip_requirements=[f'mlflow=={mlflow.__version__}', f'numpy=={np.__version__}'])
            candidates.append({'run_id': active.info.run_id, 'model_uri': info.model_uri, 'mse': validation, 'model': model, 'path': str(artifact)})
    best = min(candidates, key=lambda c: c['mse'])
    searched = client.search_runs([experiment], order_by=['metrics.validation_mse ASC'])
    assert searched[0].info.run_id == best['run_id']
    registered = []
    for candidate in candidates:
        version = mlflow.register_model(candidate['model_uri'], 'course-linear')
        registered.append(str(version.version))
    champion = registered[candidates.index(best)]
    client.set_registered_model_alias('course-linear', 'champion', champion)
    resolved = client.get_model_version_by_alias('course-linear', 'champion')
    assert str(resolved.version) == champion
    loaded = mlflow.pyfunc.load_model(f'models:/course-linear/{champion}')
    probe = np.array([[-.37], [0.], [2.2]])
    np.testing.assert_allclose(loaded.predict(probe), best['model']['weight'] * probe[:, 0] + best['model']['bias'], rtol=1e-6, atol=1e-6)
    # A failed local gate never moves the alias. The registry does not implement this policy for us.
    before = str(resolved.version)
    if candidates[0]['mse'] <= .001: client.set_registered_model_alias('course-linear', 'champion', registered[0])
    assert str(client.get_model_version_by_alias('course-linear', 'champion').version) == before
    # Reversible alias demonstration, then re-resolve the accepted immutable version.
    client.set_registered_model_alias('course-linear', 'candidate', registered[0])
    client.delete_registered_model_alias('course-linear', 'candidate')
    prompt = mlflow.genai.register_prompt(name='course-grounded-answer', template='Answer using {{context}}. Question: {{question}}. Abstain if unsupported.', commit_message='Initial controlled retrieval contract')
    mlflow.genai.set_prompt_alias('course-grounded-answer', 'candidate', prompt.version)
    loaded_prompt = mlflow.genai.load_prompt(f'prompts:/course-grounded-answer/{prompt.version}')
    assert 'Abstain' in loaded_prompt.format(context='Course uses CPU examples.', question='Which backend?')
    with mlflow.start_span(name='retrieval-evaluation', span_type='CHAIN') as root:
        root.set_inputs({'case_id': 'public-fixture-1', 'prompt_version': str(prompt.version)})
        with mlflow.start_span(name='retrieve', span_type='RETRIEVER') as retrieval:
            retrieval.set_outputs({'document_ids': ['course-cpu-v1']})
        with mlflow.start_span(name='replay-answer', span_type='CHAIN') as answer:
            answer.set_outputs({'abstain': False, 'citation_ids': ['course-cpu-v1']})
        root.set_outputs({'passed': True, 'scope': 'deterministic answer replay; no model or paid API call'})
    traces = mlflow.search_traces(experiment_ids=[experiment], return_type='list')
    assert len(traces) == 1 and len(traces[0].data.spans) == 3
    chosen_raw = Path(best['path']).read_bytes(); (folder / 'model.json').write_bytes(chosen_raw)
    report = {'mlflow': mlflow.__version__, 'jax': jax.__version__, 'experiment_id': experiment, 'tracking_uri': uri, 'validation_mse': [c['mse'] for c in candidates], 'rates': [.02, .15], 'run_ids': [c['run_id'] for c in candidates], 'champion_version': champion, 'model_sha256': hashlib.sha256(chosen_raw).hexdigest(), 'prompt_version': str(prompt.version), 'trace_count': len(traces), 'span_count': len(traces[0].data.spans), 'scope': 'Real local SQLite tracking, model reload and registry; synthetic training and deterministic answer replay.'}
    (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path); args = parser.parse_args()
    if args.output: print(json.dumps(run(args.output), indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix='course-mlflow-') as folder: print(json.dumps(run(folder), indent=2))
