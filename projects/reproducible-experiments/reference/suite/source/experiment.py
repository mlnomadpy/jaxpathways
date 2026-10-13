"""Reproducible experiment CLI. Help/argument parsing do not import JAX."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(path):
    spec = importlib.util.spec_from_file_location('learner_workflow', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_suite(suite):
    manifest = json.loads((suite/'manifest.json').read_text())
    if manifest['schema'] != 1 or digest(suite/'data.npz') != manifest['data_sha256']:
        raise ValueError('dataset integrity mismatch')
    for name, expected in manifest['source_files'].items():
        if Path(name).name != name or digest(suite/'source'/name) != expected:
            raise ValueError('source integrity mismatch')
    for run_id, expected in manifest['config_files'].items():
        if Path(run_id).name != run_id or digest(suite/'configs'/(run_id+'.json')) != expected:
            raise ValueError('configuration integrity mismatch')
    return manifest


def prepare(args):
    import numpy as np
    plan = json.loads(args.config.read_text())
    if set(plan) != {'hypothesis', 'decision_rule', 'seeds', 'common', 'learning_rates'}:
        raise ValueError('plan schema mismatch')
    if any(not isinstance(plan[k], str) or not plan[k].strip() for k in ('hypothesis', 'decision_rule')):
        raise ValueError('write a hypothesis and decision rule before running')
    seeds = plan['seeds']
    if not isinstance(seeds, list) or not 2 <= len(seeds) <= 8 or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('declare two to eight distinct training seeds')
    if set(plan['learning_rates']) != {'baseline', 'candidate'}:
        raise ValueError('declare baseline and candidate rates')
    implementation = ROOT/'solution/workflow.py' if args.implementation == 'solution' else Path(args.implementation).resolve()
    workflow = load_module(implementation)
    configs = {}
    for name, rate in plan['learning_rates'].items():
        for seed in seeds:
            cfg = {**plan['common'], 'name': name, 'seed': seed, 'learning_rate': rate}
            if args.backend: cfg['backend'] = args.backend
            if args.loggers: cfg['loggers'] = args.loggers.split(',')
            configs[f'{name}-s{seed}'] = workflow.resolve_config(cfg)
    # The configuration experiment varies one rate, not data or model width.
    args.suite.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'configs', 'runs']: (args.suite/name).mkdir()
    for name in ['experiment.py', 'model.py', 'loggers.py', 'analysis.py']:
        (args.suite/'source'/name).write_bytes((ROOT/name).read_bytes())
    (args.suite/'source/workflow.py').write_bytes(implementation.read_bytes())
    for run_id, cfg in configs.items(): write_json(args.suite/'configs'/(run_id+'.json'), cfg)
    rng = np.random.default_rng(next(iter(configs.values()))['data_seed'])
    x = rng.uniform(-2, 2, (96, 2)).astype(np.float32)
    y = (np.sin(x[:, 0]) + .4*x[:, 1] + rng.normal(0, .05, len(x))).astype(np.float32)
    np.savez(args.suite/'data.npz', train_x=x[:64], train_y=y[:64], validation_x=x[64:], validation_y=y[64:])
    constant_mse = float(np.mean((np.float64(y[:64].mean()) - y[64:].astype(np.float64))**2))
    write_json(args.suite/'manifest.json', {'schema': 1, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hypothesis': plan['hypothesis'], 'decision_rule': plan['decision_rule'], 'seeds': seeds,
        'metric_contract': 'post-update full-split MSE/v1; fixed train 64 and validation 32 rows',
        'data_sha256': digest(args.suite/'data.npz'), 'constant_validation_mse': constant_mse,
        'source_files': {p.name: digest(p) for p in sorted((args.suite/'source').glob('*.py'))},
        'config_files': {run_id: digest(args.suite/'configs'/(run_id+'.json')) for run_id in configs}})
    lines = ['#!/usr/bin/env sh', 'set -eu', 'PYTHON=${PYTHON:-python3}',
             'WORKSHOP_SUITE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
             '# Independent sequential jobs; review before using an approved accelerator host.']
    for run_id in configs:
        lines.append('"$PYTHON" "$WORKSHOP_SUITE/source/experiment.py" job --suite "$WORKSHOP_SUITE" --run-id ' + shlex.quote(run_id) + ' --timeout 120')
    (args.suite/'jobs.sh').write_text('\n'.join(lines)+'\n')
    print(f'Prepared {len(configs)} runs in {args.suite}; no training has run.')


def worker(args):
    suite = args.suite.resolve()
    manifest = verify_suite(suite)
    config = json.loads((suite/'configs'/(args.run_id+'.json')).read_text())
    workflow = load_module(suite/'source/workflow.py')
    config = workflow.resolve_config(config)
    directory = args.attempt_dir.resolve()
    directory.mkdir(exist_ok=False)
    write_json(directory/'config.json', config)
    os.environ['JAX_PLATFORMS'] = config['backend']
    logger = None
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import flax
        import optax
        from flax import nnx
        from model import Regressor, objective, train_step, optimizer, arrays
        from loggers import Loggers
        if jax.default_backend() != config['backend'] or jax.process_count() != 1:
            raise RuntimeError('requires the requested backend and one controller')
        device = jax.local_devices()[0]
        write_json(directory/'environment.json', {'python': platform.python_version(), 'jax': jax.__version__,
            'jaxlib': __import__('jaxlib').__version__, 'flax': flax.__version__, 'optax': optax.__version__,
            'numpy': np.__version__, 'logger_versions': {name: importlib.metadata.version(name) for name in config['loggers'] if name != 'csv'},
            'backend': jax.default_backend(), 'device_kind': device.device_kind,
            'local_devices': [str(d) for d in jax.local_devices()], 'process_count': jax.process_count(),
            'scope': 'single process, first local device, float32; no distributed scaling measurement'})
        with jax.default_device(device), np.load(suite/'data.npz', allow_pickle=False) as data:
            x, y = jnp.asarray(data['train_x']), jnp.asarray(data['train_y'])
            vx, vy = jnp.asarray(data['validation_x']), jnp.asarray(data['validation_y'])
            model = Regressor(config['width'], config['seed'])
            initial = arrays(model)
            opt = optimizer(model, config['learning_rate'])
            warm = nnx.clone(model)
            warm_opt = nnx.clone(opt)
            assert warm_opt.tx is opt.tx  # preserve the exact static optimizer transformation
            train_step(warm, warm_opt, x, y).block_until_ready()
            assert int(opt.step[...]) == 0
            for key, value in arrays(model).items(): np.testing.assert_array_equal(value, initial[key])
            identity = {'plan_sha256': digest(suite/'manifest.json'), 'run_id': args.run_id, 'data_sha256': manifest['data_sha256'],
                        'config_sha256': manifest['config_files'][args.run_id],
                        'metric_contract': manifest['metric_contract']}
            logger = Loggers(directory, config, identity)
            history = []
            def observe(step, train_mse, norm, duration):
                row = {'step': step, 'train_mse': float(train_mse), 'validation_mse': float(objective(model, vx, vy)),
                       'gradient_norm': norm, 'update_ms': duration}
                history.append(row)
                logger.log(row)
            observe(0, objective(model, x, y), None, None)
            start_job = time.perf_counter()
            for step in range(1, config['steps']+1):
                start = time.perf_counter()
                values = train_step(model, opt, x, y).block_until_ready()
                duration = (time.perf_counter()-start)*1000
                if step % config['log_every'] == 0 or step == config['steps']:
                    observed = np.asarray(values)
                    observe(step, observed[0], float(observed[1]), duration)
                if args.fail_after == step: raise RuntimeError('intentional first-attempt failure for recovery practice')
            wall = time.perf_counter()-start_job
            workflow.validate_history(history, config)
            prediction = np.asarray(model(vx))
            oracle = float(np.mean((prediction.astype(np.float64)-np.asarray(vy, dtype=np.float64))**2))
            np.testing.assert_allclose(history[-1]['validation_mse'], oracle, rtol=3e-5, atol=2e-6)
            np.savez(directory/'checkpoint.npz', **arrays(model), validation_prediction=prediction)
            logger.close()
            write_json(directory/'summary.json', {'status': 'completed', **identity, 'final_validation_mse': history[-1]['validation_mse'],
                'checkpoint_sha256': digest(directory/'checkpoint.npz'), 'metrics_sha256': digest(directory/'metrics.csv'),
                'environment_sha256': digest(directory/'environment.json'), 'loggers': logger.receipts,
                'train_wall_seconds': wall, 'scope': 'includes validation/logging; excludes initialization and warmup'})
            print('Completed', args.run_id, 'validation MSE', history[-1]['validation_mse'])
    except Exception as error:
        if logger:
            try: logger.close(success=False)
            except Exception: pass
        write_json(directory/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


def run_job(suite, run_id, timeout, retries, fail_first=None):
    manifest = verify_suite(suite)
    if run_id not in manifest['config_files']: raise ValueError('unknown run id')
    directory = suite/'runs'/run_id
    directory.mkdir(exist_ok=False)
    attempts = []
    for index in range(1, retries+2):
        target = directory/f'attempt-{index:02d}'
        command = [sys.executable, str(suite/'source/experiment.py'), 'worker', '--suite', str(suite), '--run-id', run_id, '--attempt-dir', str(target)]
        if index == 1 and fail_first == run_id: command += ['--fail-after', '1']
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env=env, start_new_session=os.name == 'posix')
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'posix': os.killpg(process.pid, signal.SIGTERM)
            else: process.terminate()
            try: stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'posix': os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
                stdout, stderr = process.communicate()
        target.mkdir(exist_ok=True)  # retain diagnostics even if Python failed before mkdir
        (target/'stdout.txt').write_text(stdout)
        (target/'stderr.txt').write_text(stderr)
        status = 'timed_out' if timed_out else 'completed' if process.returncode == 0 and (target/'summary.json').exists() else 'failed'
        attempts.append({'attempt': index, 'status': status, 'returncode': process.returncode, 'command': command,
                         'intentional_failure': index == 1 and fail_first == run_id,
                         'summary_sha256': digest(target/'summary.json') if status == 'completed' else None})
        write_json(directory/'job.json', {'run_id': run_id, 'status': status, 'attempts': attempts})
        if status == 'completed': return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('plan', help='freeze source, data, run configs and research intent')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--suite', type=Path, required=True)
    a.add_argument('--implementation', default='solution')
    a.add_argument('--backend', choices=['cpu', 'gpu', 'tpu'])
    a.add_argument('--loggers', help='csv plus optional tensorboard,wandb,mlflow')
    for name in ('run', 'job'):
        a = sub.add_parser(name)
        a.add_argument('--suite', type=Path, required=True)
        a.add_argument('--timeout', type=float, default=120)
        a.add_argument('--retries', type=int, choices=[0, 1, 2], default=0)
        a.add_argument('--fail-first', help='run ID to fail once intentionally; retained in attempt evidence')
        if name == 'job': a.add_argument('--run-id', required=True)
        else: a.add_argument('--workers', type=int, choices=[1, 2], default=1)
    a = sub.add_parser('analyze')
    a.add_argument('--suite', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('worker', help=argparse.SUPPRESS)
    a.add_argument('--suite', type=Path, required=True)
    a.add_argument('--run-id', required=True)
    a.add_argument('--attempt-dir', type=Path, required=True)
    a.add_argument('--fail-after', type=int)
    args = p.parse_args()
    if args.action == 'plan': return prepare(args)
    if args.action == 'worker': return worker(args)
    if args.action == 'analyze':
        frozen = args.suite.resolve()/'source/experiment.py'
        if ROOT.resolve() != frozen.parent:
            verify_suite(args.suite.resolve())
            return subprocess.run([sys.executable, str(frozen), 'analyze', '--suite', str(args.suite.resolve()), '--output', str(args.output.resolve())]).returncode
        from analysis import analyze
        return analyze(args.suite.resolve(), args.output.resolve())
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800: p.error('timeout must be finite in (0, 1800] seconds')
    suite = args.suite.resolve()
    manifest = verify_suite(suite)
    if args.fail_first and args.fail_first not in manifest['config_files']: p.error('unknown intentional failure run')
    if args.action == 'job': return 0 if run_job(suite, args.run_id, args.timeout, args.retries, args.fail_first) else 1
    if any((suite/'runs'/run_id).exists() for run_id in manifest['config_files']):
        p.error('suite has job evidence already; use a new plan or launch only a missing job')
    if args.workers > 1 and any(json.loads((suite/'configs'/(run_id+'.json')).read_text())['backend'] != 'cpu' for run_id in manifest['config_files']):
        p.error('parallel runner is CPU only; use one process per approved accelerator allocation')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, suite, run_id, args.timeout, args.retries, args.fail_first) for run_id in manifest['config_files']]
        outcomes = [future.result() for future in futures]
    print(f'{sum(outcomes)}/{len(outcomes)} jobs completed; failures remain in runs/.')
    return 0 if all(outcomes) else 1


if __name__ == '__main__':
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
