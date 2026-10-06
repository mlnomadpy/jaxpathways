"""MLOps: data contracts, CI gates and monitoring: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
import hashlib, json, math

# Run and inspect the controlled experiment
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    if not rows:
        raise ValueError('empty dataset')
    ids = set()
    groups = {}
    for row in rows:
        if set(row) != {'id', 'group', 'split', 'x', 'y'}:
            raise ValueError('schema mismatch')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('duplicate or empty ID')
        if not isinstance(row['group'], str) or not row['group']:
            raise ValueError('missing source group')
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('unknown split')
        for name in ['x', 'y']:
            if isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                raise ValueError('nonfinite or nonnumeric observation')
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('source group leaks between splits')
        ids.add(row['id']); groups[row['group']] = row['split']
    return digest(rows)

def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    if len(errors) != len(slices) or not errors or not math.isfinite(limit) or limit < 0:
        raise ValueError('invalid metric contract')
    if any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError('invalid squared error')
    if not set(slices) <= {'low', 'high'}:
        raise ValueError('unknown slice')
    result = {name: {'count': slices.count(name), 'mse': None} for name in ['low', 'high']}
    for name, item in result.items():
        selected = [e for e, s in zip(errors, slices) if s == name]
        if selected: item['mse'] = math.fsum(selected) / len(selected)
    result['overall'] = {'count': len(errors), 'mse': math.fsum(errors) / len(errors)}
    result['passed'] = all(v['count'] > 0 and v['mse'] <= limit for v in result.values())
    return result

rows = [dict(id=f'r{i}', group=f'g{i}', split='train' if i<4 else 'validation', x=float(i), y=2.*i+1.) for i in range(6)]
fingerprint = validate_rows(rows)
changed = [dict(r) for r in rows]; changed[-1]['x'] += 1
assert validate_rows(changed) != fingerprint
leaked = [dict(r) for r in rows]; leaked[-1]['group'] = rows[0]['group']
try: validate_rows(leaked)
except ValueError: pass
else: raise AssertionError('cross-split source leakage accepted')
errors = [.01] * 9 + [4.]
slices = ['low'] * 9 + ['high']
metrics = gate_metrics(errors, slices, .5)
assert math.isclose(metrics['overall']['mse'], .409)
assert metrics['overall']['mse'] < .5 and not metrics['passed']
assert metrics['high']['count'] == 1 and metrics['high']['mse'] == 4.
assert not gate_metrics([.01]*9, ['low']*9, .5)['passed']
# Total variation on two declared bins: changed input mix is a diagnostic, not proof of degraded accuracy.
reference_mix = [.9, .1]; current_mix = [.5, .5]
shift = .5 * sum(abs(a-b) for a,b in zip(reference_mix,current_mix))
assert math.isclose(shift, .4)
print('Overall MSE:',metrics['overall']['mse'],'high-slice MSE:',metrics['high']['mse'],'release:',metrics['passed'])
print('Two-bin input total variation:',shift,'(requires investigation; not an automatic retraining order)')

import hashlib, json, math

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    if not rows:
        raise ValueError('empty dataset')
    ids = set()
    groups = {}
    for row in rows:
        if set(row) != {'id', 'group', 'split', 'x', 'y'}:
            raise ValueError('schema mismatch')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('duplicate or empty ID')
        if not isinstance(row['group'], str) or not row['group']:
            raise ValueError('missing source group')
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('unknown split')
        for name in ['x', 'y']:
            if isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                raise ValueError('nonfinite or nonnumeric observation')
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('source group leaks between splits')
        ids.add(row['id']); groups[row['group']] = row['split']
    return digest(rows)

def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    if len(errors) != len(slices) or not errors or not math.isfinite(limit) or limit < 0:
        raise ValueError('invalid metric contract')
    if any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError('invalid squared error')
    if not set(slices) <= {'low', 'high'}:
        raise ValueError('unknown slice')
    result = {name: {'count': slices.count(name), 'mse': None} for name in ['low', 'high']}
    for name, item in result.items():
        selected = [e for e, s in zip(errors, slices) if s == name]
        if selected: item['mse'] = math.fsum(selected) / len(selected)
    result['overall'] = {'count': len(errors), 'mse': math.fsum(errors) / len(errors)}
    result['passed'] = all(v['count'] > 0 and v['mse'] <= limit for v in result.values())
    return result

rows = [dict(id=f'r{i}', group=f'g{i}', split='train' if i<4 else 'validation', x=float(i), y=2.*i+1.) for i in range(6)]
fingerprint = validate_rows(rows)
changed = [dict(r) for r in rows]; changed[-1]['x'] += 1
assert validate_rows(changed) != fingerprint
leaked = [dict(r) for r in rows]; leaked[-1]['group'] = rows[0]['group']
try: validate_rows(leaked)
except ValueError: pass
else: raise AssertionError('cross-split source leakage accepted')
errors = [.01] * 9 + [4.]
slices = ['low'] * 9 + ['high']
metrics = gate_metrics(errors, slices, .5)
assert math.isclose(metrics['overall']['mse'], .409)
assert metrics['overall']['mse'] < .5 and not metrics['passed']
assert metrics['high']['count'] == 1 and metrics['high']['mse'] == 4.
assert not gate_metrics([.01]*9, ['low']*9, .5)['passed']
# Total variation on two declared bins: changed input mix is a diagnostic, not proof of degraded accuracy.
reference_mix = [.9, .1]; current_mix = [.5, .5]
shift = .5 * sum(abs(a-b) for a,b in zip(reference_mix,current_mix))
assert math.isclose(shift, .4)
print('Overall MSE:',metrics['overall']['mse'],'high-slice MSE:',metrics['high']['mse'],'release:',metrics['passed'])
print('Two-bin input total variation:',shift,'(requires investigation; not an automatic retraining order)')


# Figure data experiment
visual_data={'kind':'bar','labels':['overall (10)','low (9)','high (1)'],'xlabel':'evaluation group (count)','ylabel':'mean squared error','series':[{'label':'observed MSE','y':[metrics[k]['mse'] for k in ['overall','low','high']]},{'label':'illustrative limit','y':[.5,.5,.5]}]}

# Experiment: A good aggregate hides a bad slice
assert metrics['overall']['mse'] < .5
assert metrics['high']['mse'] > .5 and metrics['passed'] is False
print('Aggregate-only acceptance would miss the high-slice failure.')

# Reference solution. Try the exercise before reading this.
wrong=(metrics['low']['mse']+metrics['high']['mse'])/2
assert math.isclose(wrong,2.005) and not math.isclose(wrong,metrics['overall']['mse'])
print('Unweighted slice average:',wrong)

# Reference practice: Change the population mix
shifted=gate_metrics([.01]+[4.]*9,['low']+['high']*9,.5)
assert math.isclose(shifted['overall']['mse'],3.601)
assert shifted['high']['mse']==metrics['high']['mse']
print('New population MSE:',shifted['overall']['mse'])
print("PASS: operations-06")
