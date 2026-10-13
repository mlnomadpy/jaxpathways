"""Cumulative public contracts and an independent end-to-end CPU experiment check."""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import digest, load_module


def reject(fn):
    try: fn()
    except (ValueError, TypeError): return
    raise AssertionError('expected the invalid evidence to be rejected')


def execute(*args, success=True):
    result = subprocess.run([sys.executable, str(ROOT/'experiment.py'), *map(str, args)],
                            env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, capture_output=True, text=True, timeout=180)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', default='solution')
    parser.add_argument('--stage', choices=['1', '2', '3', '4', 'all'], default='all')
    args = parser.parse_args()
    path = ROOT/'solution/workflow.py' if args.implementation == 'solution' else Path(args.implementation).resolve()
    learner = load_module(path)
    stage = 4 if args.stage == 'all' else int(args.stage)
    cfg = {'name': 'baseline', 'seed': 5, 'data_seed': 27, 'learning_rate': .03, 'steps': 3,
           'log_every': 2, 'width': 3, 'backend': 'cpu', 'precision': 'float32', 'loggers': ['csv']}
    source = json.loads(json.dumps(cfg))
    assert learner.resolve_config(cfg) == source and cfg == source
    for key, value in [('steps', True), ('seed', -1), ('learning_rate', float('nan')), ('learning_rate', 0),
                       ('backend', 'auto'), ('precision', 'float16'), ('loggers', ['csv', 'csv']), ('loggers', ['wandb']), ('log_every', 4)]:
        reject(lambda key=key, value=value: learner.resolve_config({**cfg, key: value}))
    reject(lambda: learner.resolve_config({**cfg, 'unknown': 1}))
    print('PASS stage 1: resolved config, nonmutation and invalid boundaries')
    if stage < 2: return
    rows = [{'step': s, 'train_mse': .5, 'validation_mse': .6, 'gradient_norm': None if s == 0 else .2,
             'update_ms': None if s == 0 else 1.} for s in (0, 2, 3)]
    assert learner.validate_history(rows, cfg) == rows
    reject(lambda: learner.validate_history(rows[:-1], cfg))
    reject(lambda: learner.validate_history([rows[0], rows[1], rows[1]], cfg))
    reject(lambda: learner.validate_history([{**rows[0], 'gradient_norm': 0}, *rows[1:]], cfg))
    reject(lambda: learner.validate_history([rows[0], {**rows[1], 'validation_mse': float('inf')}, rows[2]], cfg))
    print('PASS stage 2: complete post-update metric history')
    if stage < 3: return
    records = [{'name': name, 'seed': seed, 'validation_mse': value} for name, values in
               [('baseline', [1., 2., 3.]), ('candidate', [.75, 1.5, 2.25])] for seed, value in zip([5, 9, 17], values)]
    summary = learner.paired_summary(list(reversed(records)), [5, 9, 17])
    assert summary['differences'] == [-.25, -.5, -.75] and summary['mean_difference'] == -.5
    assert summary['sample_sd_difference'] == .25 and summary['count'] == 3
    second = learner.paired_summary([{'name': name, 'seed': seed, 'validation_mse': v} for name, vals in
                                    [('baseline', [3., 1.]), ('candidate', [1., 2.])] for seed, v in zip([2, 31], vals)], [2, 31])
    assert second['mean_difference'] == -.5 and math.isclose(second['sample_sd_difference'], math.sqrt(4.5))
    reject(lambda: learner.paired_summary(records[:-1], [5, 9, 17]))
    reject(lambda: learner.paired_summary(records + [records[0]], [5, 9, 17]))
    print('PASS stage 3: independent paired arithmetic, changed seeds and missing-pair rejection')
    if stage < 4: return
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpy as np
    import jax
    import jax.numpy as jnp
    from model import Regressor, arrays, optimizer, train_step
    # Independent NumPy backprop for a changed shape and seed.
    model = Regressor(3, 5)
    before = arrays(model)
    x = np.array([[1., -.3], [.1, 2.], [-1.2, .7]], dtype=np.float32)
    y = np.array([.6, -.4, 1.1], dtype=np.float32)
    h = np.tanh(x @ before['hidden_kernel'] + before['hidden_bias'])
    residual = (h @ before['out_kernel'] + before['out_bias']).ravel() - y
    dy = 2 * residual[:, None] / len(y)
    dh = (dy @ before['out_kernel'].T) * (1-h*h)
    gradients = {'hidden_kernel': x.T @ dh, 'hidden_bias': dh.sum(0), 'out_kernel': h.T @ dy, 'out_bias': dy.sum(0)}
    train_step(model, optimizer(model, .03), jnp.asarray(x), jnp.asarray(y)).block_until_ready()
    for key, value in arrays(model).items(): np.testing.assert_allclose(value, before[key] - .03*gradients[key], rtol=3e-5, atol=2e-6)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        plan = json.loads((ROOT/'configs/workshop.json').read_text())
        plan['seeds'] = [5, 9]
        plan['common'].update(steps=3, log_every=2, width=3)
        config = tmp/'plan.json'
        config.write_text(json.dumps(plan))
        suite = tmp/'suite'
        execute('plan', '--config', config, '--suite', suite, '--implementation', path)
        execute('analyze', '--suite', suite, '--output', tmp/'missing', success=False)
        incomplete = json.loads((tmp/'missing/comparison.json').read_text())
        assert not incomplete['complete'] and len(incomplete['inventory']) == 4 and 'paired' not in incomplete
        execute('run', '--suite', suite, '--workers', 2, '--retries', 1, '--fail-first', 'baseline-s5')
        job = json.loads((suite/'runs/baseline-s5/job.json').read_text())
        assert [a['status'] for a in job['attempts']] == ['failed', 'completed']
        assert (suite/'runs/baseline-s5/attempt-01/failure.json').exists()
        execute('analyze', '--suite', suite, '--output', tmp/'report')
        report = json.loads((tmp/'report/comparison.json').read_text())
        assert report['complete'] and report['paired']['count'] == 2
        from model import restore
        with np.load(suite/'data.npz') as data:
            for run in report['runs']:
                folder = Path(run['evidence'])
                weights = np.load(folder/'checkpoint.npz')
                prediction = np.tanh(data['validation_x'] @ weights['hidden_kernel'] + weights['hidden_bias']) @ weights['out_kernel'] + weights['out_bias']
                expected = float(np.mean((prediction.ravel().astype(np.float64) - data['validation_y'])**2))
                np.testing.assert_allclose(expected, run['history'][-1]['validation_mse'], rtol=3e-5, atol=2e-6)
                np.testing.assert_allclose(np.asarray(restore(folder/'checkpoint.npz', 3)(data['validation_x'])), prediction.ravel(), rtol=3e-5, atol=2e-6)
        # Same frozen inputs, new process, no injection: complete trajectories replay.
        replay = tmp/'replay'
        shutil.copytree(suite, replay, ignore=shutil.ignore_patterns('runs'))
        (replay/'runs').mkdir()
        script = subprocess.run(['sh', str(replay/'jobs.sh')], cwd=tmp, env={**os.environ, 'PYTHON': sys.executable}, capture_output=True, text=True, timeout=180)
        assert script.returncode == 0, script.stdout + script.stderr
        with (replay/'runs/baseline-s5/attempt-01/metrics.csv').open() as a, (suite/'runs/baseline-s5/attempt-02/metrics.csv').open() as b:
            for left, right in zip(csv.DictReader(a), csv.DictReader(b)):
                for key in ('step', 'train_mse', 'validation_mse', 'gradient_norm'): assert left[key] == right[key]
        execute('run', '--suite', suite, success=False)
        execute('job', '--suite', suite, '--run-id', 'baseline-s5', success=False)
        execute('analyze', '--suite', suite, '--output', tmp/'report', success=False)
        metrics = suite/'runs/baseline-s5/attempt-02/metrics.csv'
        metrics.write_text(metrics.read_text().replace('0,', '1,', 1))
        execute('analyze', '--suite', suite, '--output', tmp/'corrupt', success=False)
        corrupt = json.loads((tmp/'corrupt/comparison.json').read_text())
        assert not corrupt['complete'] and any(r['status'] == 'invalid' for r in corrupt['inventory'])
        deadline = tmp/'deadline'
        shutil.copytree(replay, deadline, ignore=shutil.ignore_patterns('runs'))
        (deadline/'runs').mkdir()
        execute('job', '--suite', deadline, '--run-id', 'candidate-s5', '--timeout', '.001', success=False)
        assert json.loads((deadline/'runs/candidate-s5/job.json').read_text())['status'] == 'timed_out'
        execute('job', '--suite', deadline, '--run-id', 'candidate-s9', '--timeout', 'nan', success=False)
        assert not (deadline/'runs/candidate-s9').exists()
    print('PASS stage 4: NumPy gradients, frozen-source training, paired report, retry/replay, reload, timeout, overwrite and corrupt evidence')


if __name__ == '__main__':
    main()
