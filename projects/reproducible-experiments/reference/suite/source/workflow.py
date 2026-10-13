"""Small learner-owned contracts, independent of JAX and tracker accounts."""
import math
import statistics


def resolve_config(value):
    required = {'name', 'seed', 'data_seed', 'learning_rate', 'steps', 'log_every', 'width', 'backend', 'precision', 'loggers'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('configuration must have exactly the documented fields')
    cfg = dict(value)
    if cfg['name'] not in ('baseline', 'candidate'):
        raise ValueError('name must be baseline or candidate')
    for key, low, high in [('seed', 0, 2**31-1), ('data_seed', 0, 2**31-1), ('steps', 1, 500), ('log_every', 1, 500), ('width', 1, 64)]:
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError('invalid ' + key)
    rate = cfg['learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError('learning_rate must be finite in (0, 1]')
    if cfg['log_every'] > cfg['steps'] or cfg['backend'] not in ('cpu', 'gpu', 'tpu') or cfg['precision'] != 'float32':
        raise ValueError('invalid cadence, backend or precision; this lab qualifies float32')
    logs = cfg['loggers']
    if not isinstance(logs, list) or any(type(x) is not str for x in logs) or len(set(logs)) != len(logs):
        raise ValueError('loggers must be a unique list')
    if 'csv' not in logs or not set(logs) <= {'csv', 'tensorboard', 'wandb', 'mlflow'}:
        raise ValueError('CSV is mandatory; unknown logger')
    cfg['loggers'] = sorted(logs)
    return cfg


def validate_history(rows, config):
    """Require every declared observation, including initial and final state."""
    expected = sorted({0, config['steps'], *range(config['log_every'], config['steps']+1, config['log_every'])})
    fields = {'step', 'train_mse', 'validation_mse', 'gradient_norm', 'update_ms'}
    if len(rows) != len(expected):
        raise ValueError('missing or duplicate observations')
    for row, step in zip(rows, expected):
        if set(row) != fields or type(row['step']) is not int or row['step'] != step:
            raise ValueError('metric schema or completed-step boundary changed')
        for key in fields - {'step'}:
            v = row[key]
            if step == 0 and key in ('gradient_norm', 'update_ms') and v is None:
                continue
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError('invalid metric: ' + key)
        if step == 0 and (row['gradient_norm'] is not None or row['update_ms'] is not None):
            raise ValueError('initial row has no preceding update')
    return rows


def paired_summary(records, expected_seeds):
    """Negative candidate-minus-baseline MSE favors the candidate. SD is descriptive."""
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('need at least two distinct paired seeds')
    pairs = {}
    for r in records:
        if set(r) != {'name', 'seed', 'validation_mse'} or r['name'] not in ('baseline', 'candidate') or r['seed'] not in expected_seeds:
            raise ValueError('unexpected comparison row')
        key = (r['name'], r['seed'])
        if key in pairs or type(r['validation_mse']) not in (int, float) or not math.isfinite(r['validation_mse']) or r['validation_mse'] < 0:
            raise ValueError('duplicate or invalid result')
        pairs[key] = r['validation_mse']
    if set(pairs) != {(name, seed) for name in ('baseline', 'candidate') for seed in expected_seeds}:
        raise ValueError('incomplete pairing; never silently drop failed seeds')
    differences = [pairs['candidate', seed] - pairs['baseline', seed] for seed in expected_seeds]
    return {'seeds': list(expected_seeds), 'differences': differences, 'count': len(differences),
            'mean_difference': statistics.mean(differences), 'sample_sd_difference': statistics.stdev(differences),
            'baseline_mean': statistics.mean(pairs['baseline', s] for s in expected_seeds),
            'candidate_mean': statistics.mean(pairs['candidate', s] for s in expected_seeds)}
