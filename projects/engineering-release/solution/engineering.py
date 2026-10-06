"""Local lifecycle contracts. The caller supplies trusted reviewer identity in this exercise."""
import hashlib
import json
import math
from pathlib import Path


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


def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']


def approval_for(bundle, actor, now, expires):
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}


def verify_release(bundle, approval, now, target):
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        value = bundle[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    image = bundle['image_digest']
    if not isinstance(image, str) or not image.startswith('sha256:') or len(image) != 71 or any(c not in '0123456789abcdef' for c in image[7:]):
        raise ValueError('resolve a real container digest before approval')
    if approval.get('bundle_hash') != digest(bundle) or not approval.get('reviewer'):
        raise ValueError('approval belongs to another release')
    if not approval['approved_at'] <= now < approval['expires_at']:
        raise ValueError('approval is not currently valid')
    return True


def activate(store, version, bundle, approval, now, target):
    """A single-writer local fixture; a deployed service needs concurrency control and auth."""
    verify_release(bundle, approval, now, target)
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    pointer = store / 'active.json'
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    temporary = store / 'active.tmp'
    temporary.write_text(json.dumps(selected)); temporary.replace(pointer)
    return selected
