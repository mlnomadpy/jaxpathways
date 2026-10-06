"""ModelOps: own, approve, roll back and retire a model: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
import hashlib, json, math, tempfile
from pathlib import Path

# Run and inspect the controlled experiment
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

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

bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
checks = [True]
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    pointer = Path(folder)/'active.json'; before = pointer.read_bytes()
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        try: activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError: checks.append(False)
        else: raise AssertionError('invalid release activated')
        assert pointer.read_bytes() == before
assert checks == [True,False,False,False]
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')

import hashlib, json, math, tempfile
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

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

bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
checks = [True]
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    pointer = Path(folder)/'active.json'; before = pointer.read_bytes()
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        try: activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError: checks.append(False)
        else: raise AssertionError('invalid release activated')
        assert pointer.read_bytes() == before
assert checks == [True,False,False,False]
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')


# Figure data experiment
visual_data={'kind':'bar','labels':['reviewed','changed model','expired','wrong target'],'xlabel':'attempted transition','ylabel':'accepted (1) or rejected (0)','series':[{'label':'observed policy result','y':[int(v) for v in checks]}]}

# Experiment: Test the exact expiry boundary
assert verify_release(bundle,approval,199,'cpu-demo')
try: verify_release(bundle,approval,200,'cpu-demo')
except ValueError: print('Expiry is exclusive: 199 accepted, 200 rejected.')
else: raise AssertionError('expired approval accepted')

# Reference solution. Try the exercise before reading this.
rechecked=dict(bundle,evaluation_hash=digest({'high_slice_mse':9.}))
try: verify_release(rechecked,approval,150,'cpu-demo')
except ValueError: print('Changed evaluation invalidates the old approval.')
else: raise AssertionError('stale approval accepted')

# Reference practice: Review a changed owner
changed_owner=dict(bundle,owner='different-team')
assert digest(changed_owner)!=approval['bundle_hash']
try: verify_release(changed_owner,approval,150,'cpu-demo')
except ValueError: print('Ownership change requires renewed review.')
else: raise AssertionError('changed owner accepted')
print("PASS: operations-08")
