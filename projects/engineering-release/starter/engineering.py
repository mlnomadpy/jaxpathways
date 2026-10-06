"""Local lifecycle contracts. The caller supplies trusted reviewer identity in this exercise."""
import hashlib
import json
import math
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    raise NotImplementedError('Implement validate_rows')

def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    raise NotImplementedError('Implement gate_metrics')

def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    raise NotImplementedError('Implement llm_case')

def approval_for(bundle, actor, now, expires):
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}

def verify_release(bundle, approval, now, target):
    """Implement the stage contract."""
    raise NotImplementedError('Implement verify_release')

def activate(store, version, bundle, approval, now, target):
    """A single-writer local fixture; a deployed service needs concurrency control and auth."""
    verify_release(bundle, approval, now, target)
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    pointer = store / 'active.json'
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    temporary = store / 'active.tmp'
    temporary.write_text(json.dumps(selected))
    temporary.replace(pointer)
    return selected
