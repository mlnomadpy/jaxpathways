"""Local lifecycle contracts. The caller supplies trusted reviewer identity in this exercise."""
import hashlib
import json
import math
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def validate_rows(rows):
    """An observation has a stable identity, split, group, input and target."""
    # Key APIs to use: `contract`, `or`, `math.isfinite`, `ids.add`, `digest`
    # Step 1: Guard input contract (`not rows`) and fail fast if violated.
    # Step 2: Run `set` to compute `ids`.
    # Step 3: Evaluate `groups` from the current inputs and state.
    # Step 4: Loop over `row` in `rows`:
    # Step 5: Inside block: Guard input contract (`set(row) != {'id', 'group', 'split', 'x', 'y'}`) and fail fast if violated.
    # Step 6: Inside block: Guard input contract (`not isinstance(row['id'], str) or not row['id'] or row['id'] in ids`) and fail fast if violated.
    raise NotImplementedError('Implement validate_rows')

def gate_metrics(errors, slices, limit):
    """Count-weighted MSE plus every required slice; no empty-slice pass."""
    # Key APIs to use: `contract`, `or`, `math.isfinite`, `any`, `slices.count`
    # Step 1: Guard input contract (`len(errors) != len(slices) or not errors or (not math.isfinite(limit)) or (limit < 0)`) and fail fast if violated.
    # Step 2: Guard input contract (`any((not math.isfinite(e) or e < 0 for e in errors))`) and fail fast if violated.
    # Step 3: Guard input contract (`not set(slices) <= {'low', 'high'}`) and fail fast if violated.
    # Step 4: Evaluate `result` from the current inputs and state.
    # Step 5: Loop over `(name, item)` in `result.items()`:
    # Step 6: Inside block: Evaluate `selected` from the current inputs and state.
    raise NotImplementedError('Implement gate_metrics')

def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    # Key APIs to use: `any`, `type`, `and`
    # Step 1: Branch on condition `set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str)`:
    # Step 2: Branch on condition `not isinstance(answer['citations'], list) or any((not isinstance(c, str) for c in answer['citations']))`:
    # Step 3: Branch on condition `type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations)`:
    # Step 4: Branch on condition `case['unanswerable']`:
    # Step 5: Return `not answer['abstain'] and answer['answer'] == case['expected'] and (case['document'] in answer['citations'])` to the caller.
    raise NotImplementedError('Implement llm_case')

def approval_for(bundle, actor, now, expires):
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}

def verify_release(bundle, approval, now, target):
    """Implement the stage contract."""
    # Key APIs to use: `contract`, `or`, `any`, `image.startswith`, `approval.get`
    # Step 1: Evaluate `required` from the current inputs and state.
    # Step 2: Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    # Step 3: Loop over `key` in `['model_hash', 'data_hash', 'evaluation_hash']`:
    # Step 4: Inside block: Evaluate `value` from the current inputs and state.
    # Step 5: Inside block: Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
    # Step 6: Evaluate `image` from the current inputs and state.
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
