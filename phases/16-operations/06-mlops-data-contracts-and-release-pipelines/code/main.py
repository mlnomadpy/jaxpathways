"""MLOps: data contracts, CI gates and monitoring: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
# Step 1 — Prepare the explicit contract: The fixture gives every observation an ID, a source group and a split.
# Import hashlib, json, math for this computation.
import hashlib, json, math

# Run and inspect the controlled experiment
# Step 2 — Run and inspect the controlled experiment: The input mix changes from [0.9,0.1] to [0.5,0.5] across two...
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Function `validate_rows(rows)` implementing this stage's computation:
def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    # Guard input contract (`not rows`) and fail fast if violated.
    if not rows:
        raise ValueError('empty dataset')
    # Run `set` to compute `ids`.
    ids = set()
    # Evaluate `groups` from the current inputs and state.
    groups = {}
    # Iterate over `row` to step through the computation:
    for row in rows:
        # Guard input contract (`set(row) != {'id', 'group', 'split', 'x', 'y'}`) and fail fast if violated.
        if set(row) != {'id', 'group', 'split', 'x', 'y'}:
            raise ValueError('schema mismatch')
        # Guard input contract (`not isinstance(row['id'], str) or not row['id'] or row['id'] in ids`) and fail fast if violated.
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('duplicate or empty ID')
        # Guard input contract (`not isinstance(row['group'], str) or not row['group']`) and fail fast if violated.
        if not isinstance(row['group'], str) or not row['group']:
            raise ValueError('missing source group')
        # Guard input contract (`row['split'] not in {'train', 'validation', 'test'}`) and fail fast if violated.
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('unknown split')
        # Iterate over `name` to step through the computation:
        for name in ['x', 'y']:
            # Guard input contract (`isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or (not math.isfinite(row[name]))`) and fail fast if violated.
            if isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                raise ValueError('nonfinite or nonnumeric observation')
        # Guard input contract (`row['group'] in groups and groups[row['group']] != row['split']`) and fail fast if violated.
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('source group leaks between splits')
        # Run `ids.add` to perform the next check or state transition.
        # Evaluate `groups[row['group']]` from the current inputs and state.
        ids.add(row['id']); groups[row['group']] = row['split']
    # Return `digest(rows)` to the caller.
    return digest(rows)

# Function `gate_metrics(errors, slices, limit)` implementing this stage's computation:
def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    # Guard input contract (`len(errors) != len(slices) or not errors or (not math.isfinite(limit)) or (limit < 0)`) and fail fast if violated.
    if len(errors) != len(slices) or not errors or not math.isfinite(limit) or limit < 0:
        raise ValueError('invalid metric contract')
    # Guard input contract (`any((not math.isfinite(e) or e < 0 for e in errors))`) and fail fast if violated.
    if any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError('invalid squared error')
    # Guard input contract (`not set(slices) <= {'low', 'high'}`) and fail fast if violated.
    if not set(slices) <= {'low', 'high'}:
        raise ValueError('unknown slice')
    # Evaluate `result` from the current inputs and state.
    result = {name: {'count': slices.count(name), 'mse': None} for name in ['low', 'high']}
    # Iterate over `(name, item)` to step through the computation:
    for name, item in result.items():
        # Evaluate `selected` from the current inputs and state.
        selected = [e for e, s in zip(errors, slices) if s == name]
        # Branch on condition `selected`:
        if selected: item['mse'] = math.fsum(selected) / len(selected)
    # Evaluate `result['overall']` from the current inputs and state.
    result['overall'] = {'count': len(errors), 'mse': math.fsum(errors) / len(errors)}
    # Run `all` to compute `result['passed']`.
    result['passed'] = all(v['count'] > 0 and v['mse'] <= limit for v in result.values())
    # Return `result` to the caller.
    return result

# Evaluate `rows` from the current inputs and state.
rows = [dict(id=f'r{i}', group=f'g{i}', split='train' if i<4 else 'validation', x=float(i), y=2.*i+1.) for i in range(6)]
# Run `validate_rows` to compute `fingerprint`.
fingerprint = validate_rows(rows)
# Evaluate `changed` from the current inputs and state.
# Accumulate the next contribution into `changed[-1]['x']`.
changed = [dict(r) for r in rows]; changed[-1]['x'] += 1
# Verify contract: `validate_rows(changed) != fingerprint`.
assert validate_rows(changed) != fingerprint
# Evaluate `leaked` from the current inputs and state.
# Evaluate `leaked[-1]['group']` from the current inputs and state.
leaked = [dict(r) for r in rows]; leaked[-1]['group'] = rows[0]['group']
# Run the boundary check and catch the expected exception:
try: validate_rows(leaked)
except ValueError: pass
else: raise AssertionError('cross-split source leakage accepted')
# Evaluate `errors` from the current inputs and state.
errors = [.01] * 9 + [4.]
# Evaluate `slices` from the current inputs and state.
slices = ['low'] * 9 + ['high']
# Run `gate_metrics` to compute `metrics`.
metrics = gate_metrics(errors, slices, .5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert math.isclose(metrics['overall']['mse'], .409)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert metrics['overall']['mse'] < .5 and not metrics['passed']
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert metrics['high']['count'] == 1 and metrics['high']['mse'] == 4.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not gate_metrics([.01]*9, ['low']*9, .5)['passed']
# Total variation on two declared bins: changed input mix is a diagnostic, not proof of degraded accuracy.
reference_mix = [.9, .1]; current_mix = [.5, .5]
# Evaluate `shift` from the current inputs and state.
shift = .5 * sum(abs(a-b) for a,b in zip(reference_mix,current_mix))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert math.isclose(shift, .4)
# Print diagnostic summary of the computed outputs.
print('Overall MSE:',metrics['overall']['mse'],'high-slice MSE:',metrics['high']['mse'],'release:',metrics['passed'])
# Print diagnostic summary of the computed outputs.
print('Two-bin input total variation:',shift,'(requires investigation; not an automatic retraining order)')

# Step 1 — Prepare the explicit contract: The fixture gives every observation an ID, a source group and a split.
# Import hashlib, json, math for this computation.
import hashlib, json, math

# Step 2 — Run and inspect the controlled experiment: The input mix changes from [0.9,0.1] to [0.5,0.5] across two...
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Function `validate_rows(rows)` implementing this stage's computation:
def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    # Guard input contract (`not rows`) and fail fast if violated.
    if not rows:
        raise ValueError('empty dataset')
    # Run `set` to compute `ids`.
    ids = set()
    # Evaluate `groups` from the current inputs and state.
    groups = {}
    # Iterate over `row` to step through the computation:
    for row in rows:
        # Guard input contract (`set(row) != {'id', 'group', 'split', 'x', 'y'}`) and fail fast if violated.
        if set(row) != {'id', 'group', 'split', 'x', 'y'}:
            raise ValueError('schema mismatch')
        # Guard input contract (`not isinstance(row['id'], str) or not row['id'] or row['id'] in ids`) and fail fast if violated.
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('duplicate or empty ID')
        # Guard input contract (`not isinstance(row['group'], str) or not row['group']`) and fail fast if violated.
        if not isinstance(row['group'], str) or not row['group']:
            raise ValueError('missing source group')
        # Guard input contract (`row['split'] not in {'train', 'validation', 'test'}`) and fail fast if violated.
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('unknown split')
        # Iterate over `name` to step through the computation:
        for name in ['x', 'y']:
            # Guard input contract (`isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or (not math.isfinite(row[name]))`) and fail fast if violated.
            if isinstance(row[name], bool) or not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                raise ValueError('nonfinite or nonnumeric observation')
        # Guard input contract (`row['group'] in groups and groups[row['group']] != row['split']`) and fail fast if violated.
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('source group leaks between splits')
        # Run `ids.add` to perform the next check or state transition.
        # Evaluate `groups[row['group']]` from the current inputs and state.
        ids.add(row['id']); groups[row['group']] = row['split']
    # Return `digest(rows)` to the caller.
    return digest(rows)

# Function `gate_metrics(errors, slices, limit)` implementing this stage's computation:
def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    # Guard input contract (`len(errors) != len(slices) or not errors or (not math.isfinite(limit)) or (limit < 0)`) and fail fast if violated.
    if len(errors) != len(slices) or not errors or not math.isfinite(limit) or limit < 0:
        raise ValueError('invalid metric contract')
    # Guard input contract (`any((not math.isfinite(e) or e < 0 for e in errors))`) and fail fast if violated.
    if any(not math.isfinite(e) or e < 0 for e in errors):
        raise ValueError('invalid squared error')
    # Guard input contract (`not set(slices) <= {'low', 'high'}`) and fail fast if violated.
    if not set(slices) <= {'low', 'high'}:
        raise ValueError('unknown slice')
    # Evaluate `result` from the current inputs and state.
    result = {name: {'count': slices.count(name), 'mse': None} for name in ['low', 'high']}
    # Iterate over `(name, item)` to step through the computation:
    for name, item in result.items():
        # Evaluate `selected` from the current inputs and state.
        selected = [e for e, s in zip(errors, slices) if s == name]
        # Branch on condition `selected`:
        if selected: item['mse'] = math.fsum(selected) / len(selected)
    # Evaluate `result['overall']` from the current inputs and state.
    result['overall'] = {'count': len(errors), 'mse': math.fsum(errors) / len(errors)}
    # Run `all` to compute `result['passed']`.
    result['passed'] = all(v['count'] > 0 and v['mse'] <= limit for v in result.values())
    # Return `result` to the caller.
    return result

# Evaluate `rows` from the current inputs and state.
rows = [dict(id=f'r{i}', group=f'g{i}', split='train' if i<4 else 'validation', x=float(i), y=2.*i+1.) for i in range(6)]
# Run `validate_rows` to compute `fingerprint`.
fingerprint = validate_rows(rows)
# Evaluate `changed` from the current inputs and state.
# Accumulate the next contribution into `changed[-1]['x']`.
changed = [dict(r) for r in rows]; changed[-1]['x'] += 1
# Verify contract: `validate_rows(changed) != fingerprint`.
assert validate_rows(changed) != fingerprint
# Evaluate `leaked` from the current inputs and state.
# Evaluate `leaked[-1]['group']` from the current inputs and state.
leaked = [dict(r) for r in rows]; leaked[-1]['group'] = rows[0]['group']
# Run the boundary check and catch the expected exception:
try: validate_rows(leaked)
except ValueError: pass
else: raise AssertionError('cross-split source leakage accepted')
# Evaluate `errors` from the current inputs and state.
errors = [.01] * 9 + [4.]
# Evaluate `slices` from the current inputs and state.
slices = ['low'] * 9 + ['high']
# Run `gate_metrics` to compute `metrics`.
metrics = gate_metrics(errors, slices, .5)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert math.isclose(metrics['overall']['mse'], .409)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert metrics['overall']['mse'] < .5 and not metrics['passed']
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert metrics['high']['count'] == 1 and metrics['high']['mse'] == 4.
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert not gate_metrics([.01]*9, ['low']*9, .5)['passed']
# Total variation on two declared bins: changed input mix is a diagnostic, not proof of degraded accuracy.
reference_mix = [.9, .1]; current_mix = [.5, .5]
# Evaluate `shift` from the current inputs and state.
shift = .5 * sum(abs(a-b) for a,b in zip(reference_mix,current_mix))
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert math.isclose(shift, .4)
# Print diagnostic summary of the computed outputs.
print('Overall MSE:',metrics['overall']['mse'],'high-slice MSE:',metrics['high']['mse'],'release:',metrics['passed'])
# Print diagnostic summary of the computed outputs.
print('Two-bin input total variation:',shift,'(requires investigation; not an automatic retraining order)')

# Figure data experiment
# Compute figure data for: An average can conceal the failing slice
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'bar','labels':['overall (10)','low (9)','high (1)'],'xlabel':'evaluation group (count)','ylabel':'mean squared error','series':[{'label':'observed MSE','y':[metrics[k]['mse'] for k in ['overall','low','high']]},{'label':'illustrative limit','y':[.5,.5,.5]}]}

# Experiment: A good aggregate hides a bad slice
# Experiment — A good aggregate hides a bad slice: A policy must define both aggregation and required slice evidence.
# Verify contract: `metrics['overall']['mse'] < 0.5`.
assert metrics['overall']['mse'] < .5
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert metrics['high']['mse'] > .5 and metrics['passed'] is False
# Print the observed values to compare against the expected result.
print('Aggregate-only acceptance would miss the high-slice failure.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Show that averaging slice means without counts changes the overall error.
wrong=(metrics['low']['mse']+metrics['high']['mse'])/2
# Verify that the numerical values match the expected reference within tolerance.
assert math.isclose(wrong,2.005) and not math.isclose(wrong,metrics['overall']['mse'])
# Print the observed values to compare against the expected result.
print('Unweighted slice average:',wrong)

# Reference practice: Change the population mix
# Change the population mix (transfer): Aggregate performance changes with population mix even when...
shifted=gate_metrics([.01]+[4.]*9,['low']+['high']*9,.5)
# Verify that the numerical values match the expected reference within tolerance.
assert math.isclose(shifted['overall']['mse'],3.601)
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert shifted['high']['mse']==metrics['high']['mse']
# Print the observed values to compare against the expected result.
print('New population MSE:',shifted['overall']['mse'])
print("PASS: operations-06")
