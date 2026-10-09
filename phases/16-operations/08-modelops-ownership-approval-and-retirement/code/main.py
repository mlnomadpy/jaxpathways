"""ModelOps: own, approve, roll back and retire a model: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
# Step 1 — Prepare the explicit contract: Record the model artifact, data/evaluation identity, image digest,...
# Import hashlib, json, math, tempfile for this computation.
import hashlib, json, math, tempfile
from pathlib import Path

# Run and inspect the controlled experiment
# Step 2 — Run and inspect the controlled experiment: A registry alias chooses a version.
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Function `approval_for(bundle, actor, now, expires)` implementing this stage's computation:
def approval_for(bundle, actor, now, expires):
    # Guard input contract (`not actor or expires <= now`) and fail fast if violated.
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    # Return `{'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}` to the caller.
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}

# Function `verify_release(bundle, approval, now, target)` implementing this stage's computation:
def verify_release(bundle, approval, now, target):
    # Compute `required` from `{'model_hash', 'data_hash', 'evaluation_hash', 'imag...`
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    # Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    # Iterate over `key` to step through the computation:
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        # Compute `value` from `bundle[key]`
        value = bundle[key]
        # Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    # Compute `image` from `bundle['image_digest']`
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
    # Compute `pointer` from `store / 'active.json'`
    pointer = store / 'active.json'
    # Read or serialize artifact data on disk (`previous`).
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    # Compute `selected` from `{'current': version, 'previous': previous, 'bundle_h...`
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    # Compute `temporary` from `store / 'active.tmp'`
    temporary = store / 'active.tmp'
    # Run `temporary.write_text` to perform the next check or state transition.
    temporary.write_text(json.dumps(selected))
    temporary.replace(pointer)
    # Return `selected` to the caller.
    return selected

# Compute deterministic cryptographic digest `bundle` for provenance verification.
bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
# Compute `checks` from `[True]`
checks = [True]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    # Run `activate` to compute `active`.
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    # Run `pointer.read_bytes` to compute `before`.
    pointer = Path(folder)/'active.json'
    before = pointer.read_bytes()
    # Iterate over `(candidate, at, target)` to step through the computation:
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        # Run the boundary check and catch the expected exception:
        try:
            activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError:
            checks.append(False)
        else:
            raise AssertionError('invalid release activated')
        # Assert invariant `pointer.read_bytes() == before` holds
        assert pointer.read_bytes() == before
# Assert invariant `checks == [True,False,False,False]` holds
assert checks == [True,False,False,False]
# Print the observed values to compare against the expected result.
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
# Print diagnostic summary of the computed outputs.
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')

# Step 3: Verify invariants on the completed state
# Verify that only the reviewed release passed and all rejected transitions preserved active.json:
assert checks == [True, False, False, False]
print("Accepted original / changed artifact / expired approval / wrong target:", checks)

# Step 1 — Prepare the explicit contract: Record the model artifact, data/evaluation identity, image digest,...
# Import hashlib, json, math, tempfile for this computation.
import hashlib, json, math, tempfile
from pathlib import Path

# Step 2 — Run and inspect the controlled experiment: A registry alias chooses a version.
def digest(value):
    # Return `hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

# Function `approval_for(bundle, actor, now, expires)` implementing this stage's computation:
def approval_for(bundle, actor, now, expires):
    # Guard input contract (`not actor or expires <= now`) and fail fast if violated.
    if not actor or expires <= now:
        raise ValueError('invalid reviewer or expiry')
    # Return `{'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}` to the caller.
    return {'bundle_hash': digest(bundle), 'reviewer': actor, 'approved_at': now, 'expires_at': expires}

# Function `verify_release(bundle, approval, now, target)` implementing this stage's computation:
def verify_release(bundle, approval, now, target):
    # Compute `required` from `{'model_hash', 'data_hash', 'evaluation_hash', 'imag...`
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    # Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    # Iterate over `key` to step through the computation:
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        # Compute `value` from `bundle[key]`
        value = bundle[key]
        # Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    # Compute `image` from `bundle['image_digest']`
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
    # Compute `pointer` from `store / 'active.json'`
    pointer = store / 'active.json'
    # Read or serialize artifact data on disk (`previous`).
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    # Compute `selected` from `{'current': version, 'previous': previous, 'bundle_h...`
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    # Compute `temporary` from `store / 'active.tmp'`
    temporary = store / 'active.tmp'
    # Run `temporary.write_text` to perform the next check or state transition.
    temporary.write_text(json.dumps(selected))
    temporary.replace(pointer)
    # Return `selected` to the caller.
    return selected

# Compute deterministic cryptographic digest `bundle` for provenance verification.
bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
# Compute `checks` from `[True]`
checks = [True]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    # Run `activate` to compute `active`.
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    # Run `pointer.read_bytes` to compute `before`.
    pointer = Path(folder)/'active.json'
    before = pointer.read_bytes()
    # Iterate over `(candidate, at, target)` to step through the computation:
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        # Run the boundary check and catch the expected exception:
        try:
            activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError:
            checks.append(False)
        else:
            raise AssertionError('invalid release activated')
        # Assert invariant `pointer.read_bytes() == before` holds
        assert pointer.read_bytes() == before
# Assert invariant `checks == [True,False,False,False]` holds
assert checks == [True,False,False,False]
# Print the observed values to compare against the expected result.
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
# Print diagnostic summary of the computed outputs.
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')

# Figure data experiment
# Compute figure data for: Only the reviewed release passes
# Compute `visual_data` from `{'kind':'bar','labels':['reviewed','changed model','...`
visual_data={'kind':'bar','labels':['reviewed','changed model','expired','wrong target'],'xlabel':'attempted transition','ylabel':'accepted (1) or rejected (0)','series':[{'label':'observed policy result','y':[int(v) for v in checks]}]}

# Experiment: Test the exact expiry boundary
# Experiment — Test the exact expiry boundary: Boundary conventions must be explicit.
# Assert invariant `verify_release(bundle` holds
assert verify_release(bundle,approval,199,'cpu-demo')
# Run the boundary check and catch the expected exception:
try:
    verify_release(bundle,approval,200,'cpu-demo')
except ValueError:
    print('Expiry is exclusive: 199 accepted, 200 rejected.')
else:
    raise AssertionError('expired approval accepted')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the evaluation hash while keeping the model unchanged, and...
rechecked=dict(bundle,evaluation_hash=digest({'high_slice_mse':9.}))
# Run the boundary check and catch the expected exception:
try:
    verify_release(rechecked,approval,150,'cpu-demo')
except ValueError:
    print('Changed evaluation invalidates the old approval.')
else:
    raise AssertionError('stale approval accepted')

# Reference practice: Review a changed owner
# Review a changed owner (transfer): Ownership is an operational responsibility, so this policy...
changed_owner=dict(bundle,owner='different-team')
# Assert invariant `digest(changed_owner)!=approval['bundle_hash']` holds
assert digest(changed_owner)!=approval['bundle_hash']
# Run the boundary check and catch the expected exception:
try:
    verify_release(changed_owner,approval,150,'cpu-demo')
except ValueError:
    print('Ownership change requires renewed review.')
else:
    raise AssertionError('changed owner accepted')
print("PASS: operations-08")
