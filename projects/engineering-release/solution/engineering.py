"""Local lifecycle contracts. The caller supplies trusted reviewer identity in this exercise."""
# Import hashlib for this computation.
import hashlib
import json
import math
from pathlib import Path


# Function `digest(value)` implementing this stage's computation:
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
    # Construct dictionary `groups` with the structured fields for this stage.
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
        # Execute `ids.add(row['id'])`.
        ids.add(row['id'])
        groups[row['group']] = row['split']
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
    # Construct dictionary `result` with the structured fields for this stage.
    result = {name: {'count': slices.count(name), 'mse': None} for name in ['low', 'high']}
    # Iterate over `(name, item)` to step through the computation:
    for name, item in result.items():
        # Initialize list `selected` for the stage values.
        selected = [e for e, s in zip(errors, slices) if s == name]
        # Branch on condition `selected`:
        if selected: item['mse'] = math.fsum(selected) / len(selected)
    # Construct dictionary `result['overall']` with the structured fields for this stage.
    result['overall'] = {'count': len(errors), 'mse': math.fsum(errors) / len(errors)}
    # Run `all` to compute `result['passed']`.
    result['passed'] = all(v['count'] > 0 and v['mse'] <= limit for v in result.values())
    # Return `result` to the caller.
    return result


# Function `llm_case(case, answer, allowed_citations)` implementing this stage's computation:
def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    # Branch on condition `set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str)`:
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    # Branch on condition `not isinstance(answer['citations'], list) or any((not isinstance(c, str) for c in answer['citations']))`:
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    # Branch on condition `type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations)`:
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    # Branch on condition `case['unanswerable']`:
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    # Return `not answer['abstain'] and answer['answer'] == case['expected'] and (case['document'] in answer['citations'])` to the caller.
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']


# Function `approval_for(bundle, actor, now, expires)` implementing this stage's computation:
def approval_for(bundle, actor, now, expires):
    # Guard input contract (`not actor or expires <= now`) and fail fast if violated.
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    # Return `{'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}` to the caller.
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}


# Function `verify_release(bundle, approval, now, target)` implementing this stage's computation:
def verify_release(bundle, approval, now, target):
    # Construct dictionary `required` with the structured fields for this stage.
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    # Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    # Iterate over `key` to step through the computation:
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        # Compute `value` as `bundle[key]`.
        value = bundle[key]
        # Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    # Compute `image` as `bundle['image_digest']`.
    image = bundle['image_digest']
    # Guard input contract (`not isinstance(image, str) or not image.startswith('sha256:') or len(image) != 71 or any((c not in '0123456789abcdef' for c in image[7:]))`) and fail fast if violated.
    if not isinstance(image, str) or not image.startswith('sha256:') or len(image) != 71 or any(c not in '0123456789abcdef' for c in image[7:]):
        raise ValueError('resolve a real container digest before approval')
    # Guard input contract (`approval.get('bundle_hash') != digest(bundle) or not approval.get('reviewer')`) and fail fast if violated.
    if approval.get('bundle_hash') != digest(bundle) or not approval.get('reviewer'):
        raise ValueError('approval belongs to another release')
    # Guard input contract (`not approval['approved_at'] <= now < approval['expires_at']`) and fail fast if violated.
    if not approval['approved_at'] <= now < approval['expires_at']:
        raise ValueError('approval is not currently valid')
    # Return `True` to the caller.
    return True


# Function `activate(store, version, bundle, approval, ...)` implementing this stage's computation:
def activate(store, version, bundle, approval, now, target):
    """A single-writer local fixture; a deployed service needs concurrency control and auth."""
    # Run `verify_release` to perform the next check or state transition.
    verify_release(bundle, approval, now, target)
    # Compute `store` as `Path(store)`.
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    # Compute `pointer` as `store / 'active.json'`.
    pointer = store / 'active.json'
    # Read or serialize artifact data on disk (`previous`).
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    # Construct dictionary `selected` with the structured fields for this stage.
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    # Compute `temporary` as `store / 'active.tmp'`.
    temporary = store / 'active.tmp'
    # Run `temporary.write_text` to perform the next check or state transition.
    temporary.write_text(json.dumps(selected))
    temporary.replace(pointer)
    # Return `selected` to the caller.
    return selected
