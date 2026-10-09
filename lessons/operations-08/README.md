# ModelOps: own, approve, roll back and retire a model

Phase 16: Workload operations · about 65 minutes · CPU

## What you will be able to do

- Inventory the complete release
- Bind review to immutable evidence
- Make failed transitions leave no side effects
- Separate selection, rollout and monitoring

## The problem

A model passed evaluation last week. Someone changed its preprocessing or image today and reused the old approval. Let’s bind a review to the exact release bundle, reject stale decisions and preserve the current version when a gate fails.

## The idea

Model governance connects ownership, evaluation, approval and lifecycle transitions. An approval should identify exactly what was reviewed, where it may be used and when it remains valid. A label saying approved is not enough.

## Approval applies to a specific release

Suppose a model passes review, then its tokenizer or weights change. Reusing the old approval would apply a decision to an object that was never evaluated under that identity. Bind approvals to the relevant model, processor, configuration and evaluation records.

Specify transitions for activation, expiration, rejection and retirement. Invalid transitions should leave the active state unchanged rather than partially applying a release. A retired artifact may remain archived for reproducibility while no longer being eligible for new use.

The acceptance bars are summaries of discrete policy decisions. Read the underlying state and reason for each result. The fixture demonstrates decision logic, not completion of an organization's actual review process.

### Pause and reason

Does changing a display name require the same response as changing weights?

<details><summary>Compare your reasoning</summary>

Not necessarily. Define which identities and semantics the policy binds. A cosmetic label and a changed numerical artifact are different events, and the policy should distinguish them explicitly.

</details>

## Inventory the complete release

Record the model artifact, data/evaluation identity, image digest, preprocessing/signature, precision, intended target and responsible owner. The fixture compresses that into a small bundle with model, data and evaluation hashes, image digest, owner, target and passing result. Real release records should also identify training/code versions, documented limitations, dependency/license obligations where applicable, and how to reach the owner. Registry tags help discovery but are not access control.

## Bind review to immutable evidence

Hash the whole release bundle, then record who reviewed that hash and when the approval expires. If the model, evaluation, image, target or owner changes, the hash changes and the old approval must not authorize the new bundle. The local function receives a reviewer string as a fixture: it does not authenticate that person. A production workflow needs trusted identity, authorization, protected evidence and an audit log. A checksum proves equality of bytes, not authority or model quality.

## Make failed transitions leave no side effects

The example validates the bundle and approval before writing the active pointer. It uses an atomic rename for a single local writer. A failed validation must leave the previous selection byte-for-byte unchanged. With multiple deployment controllers, use a transactional compare-and-swap or a deployment system with concurrency control; atomic file replacement alone does not prevent competing writers from losing updates.

## Separate selection, rollout and monitoring

A registry alias chooses a version. A rollout changes serving processes and traffic. Monitor the loaded model/image identity as well as quality and service signals to detect disagreement. Canary and shadow releases answer different questions: a canary serves a limited live population, while shadow traffic runs a candidate without using its answer. Define stop conditions, traffic assignment and comparable metrics before observing a favorable curve.

## Rollback needs its own valid target

Keeping a previous pointer is useful but does not guarantee the old artifact still loads, remains approved or matches the current input contract. Revalidate compatibility, artifact availability and approval before rolling back. Practice a rejected rollout and a successful rollback using a known previous bundle. Retire a model by stopping new traffic, removing inappropriate aliases/access and applying the organization’s retention rules to evidence; do not silently delete the only audit trail.

## Work through the four decisions

The fixture accepts the original bundle at time $150$, then rejects changed model bytes, an approval expiring at $200$, and a different target. The valid interval is $[100,200)$: expiration is exclusive. The image digest in this companion is explicitly a format-valid test value. Replace it with the observed container image identity in the connected project before treating the record as deployment evidence.

**Run changed-bundle and expiry checks**

```bash
# Run run changed-bundle and expiry checks using the course Python environment
python projects/engineering-release/tests/check.py --implementation solution --stage 3
```

**Expected:** The gate accepts only the matching, current bundle and every failure preserves the active pointer.

## Prepare the explicit contract

Create main.py and add this setup block. Continue with the next block in the same file.

```python
# Step 1 — Prepare the explicit contract: Record the model artifact, data/evaluation identity, image digest,...
# Import hashlib, json, math, tempfile for this computation.
import hashlib, json, math, tempfile
from pathlib import Path
```

Record the model artifact, data/evaluation identity, image digest, preprocessing/signature, precision, intended target and responsible owner. The fixture compresses that into a small bundle with model, data and evaluation hashes, image digest, owner, target and passing result. Real release records should also identify training/code versions, documented limitations, dependency/license obligations where applicable, and how to reach the owner. Registry tags help discovery but are not access control.

## Run and inspect the controlled experiment

Append this block, run main.py in the course environment, and retain the actual output.

```python
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
    # Evaluate `required` from the current inputs and state.
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    # Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    # Iterate over `key` to step through the computation:
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        # Evaluate `value` from the current inputs and state.
        value = bundle[key]
        # Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    # Evaluate `image` from the current inputs and state.
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
    # Read or serialize artifact data on disk (`store`).
    # Execute the next step of the computation.
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    # Evaluate `pointer` from the current inputs and state.
    pointer = store / 'active.json'
    # Read or serialize artifact data on disk (`previous`).
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    # Evaluate `selected` from the current inputs and state.
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    # Evaluate `temporary` from the current inputs and state.
    temporary = store / 'active.tmp'
    # Read or serialize artifact data on disk (``).
    # Run `temporary.write_text` to perform the next check or state transition.
    temporary.write_text(json.dumps(selected)); temporary.replace(pointer)
    # Return `selected` to the caller.
    return selected

# Compute deterministic cryptographic digest `bundle` for provenance verification.
bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
# Evaluate `checks` from the current inputs and state.
checks = [True]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    # Run `activate` to compute `active`.
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    # Read or serialize artifact data on disk (`pointer`).
    # Run `pointer.read_bytes` to compute `before`.
    pointer = Path(folder)/'active.json'; before = pointer.read_bytes()
    # Iterate over `(candidate, at, target)` to step through the computation:
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        # Run the boundary check and catch the expected exception:
        try: activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError: checks.append(False)
        else: raise AssertionError('invalid release activated')
        # Verify contract: `pointer.read_bytes() == before`.
        assert pointer.read_bytes() == before
# Verify contract: `checks == [True, False, False, False]`.
assert checks == [True,False,False,False]
# Print the observed values to compare against the expected result.
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
# Print diagnostic summary of the computed outputs.
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')
```

A registry alias chooses a version. A rollout changes serving processes and traffic. Monitor the loaded model/image identity as well as quality and service signals to detect disagreement. Canary and shadow releases answer different questions: a canary serves a limited live population, while shadow traffic runs a candidate without using its answer. Define stop conditions, traffic assignment and comparable metrics before observing a favorable curve.

## Run the example

```python
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
    # Evaluate `required` from the current inputs and state.
    required = {'model_hash', 'data_hash', 'evaluation_hash', 'image_digest', 'owner', 'target', 'passed'}
    # Guard input contract (`set(bundle) != required or bundle['passed'] is not True or (not bundle['owner']) or (bundle['target'] != target)`) and fail fast if violated.
    if set(bundle) != required or bundle['passed'] is not True or not bundle['owner'] or bundle['target'] != target:
        raise ValueError('release contract rejected')
    # Iterate over `key` to step through the computation:
    for key in ['model_hash', 'data_hash', 'evaluation_hash']:
        # Evaluate `value` from the current inputs and state.
        value = bundle[key]
        # Guard input contract (`not isinstance(value, str) or len(value) != 64 or any((c not in '0123456789abcdef' for c in value))`) and fail fast if violated.
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid content digest')
    # Evaluate `image` from the current inputs and state.
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
    # Read or serialize artifact data on disk (`store`).
    # Execute the next step of the computation.
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    # Evaluate `pointer` from the current inputs and state.
    pointer = store / 'active.json'
    # Read or serialize artifact data on disk (`previous`).
    previous = json.loads(pointer.read_text())['current'] if pointer.exists() else None
    # Evaluate `selected` from the current inputs and state.
    selected = {'current': version, 'previous': previous, 'bundle_hash': digest(bundle)}
    # Evaluate `temporary` from the current inputs and state.
    temporary = store / 'active.tmp'
    # Read or serialize artifact data on disk (``).
    # Run `temporary.write_text` to perform the next check or state transition.
    temporary.write_text(json.dumps(selected)); temporary.replace(pointer)
    # Return `selected` to the caller.
    return selected

# Compute deterministic cryptographic digest `bundle` for provenance verification.
bundle = dict(model_hash=digest({'weight':2,'bias':1}), data_hash=digest(['fixture-v1']), evaluation_hash=digest({'high_slice_mse':.01}), image_digest='sha256:'+'1'*64, owner='course-team', target='cpu-demo', passed=True)
# This image digest is a format-valid fixture. Real releases must use the observed build digest.
approval = approval_for(bundle,'reviewer-fixture',now=100,expires=200)
# Evaluate `checks` from the current inputs and state.
checks = [True]
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='modelops-') as folder:
    # Run `activate` to compute `active`.
    active = activate(folder,'model-v1',bundle,approval,150,'cpu-demo')
    # Read or serialize artifact data on disk (`pointer`).
    # Run `pointer.read_bytes` to compute `before`.
    pointer = Path(folder)/'active.json'; before = pointer.read_bytes()
    # Iterate over `(candidate, at, target)` to step through the computation:
    for candidate,at,target in [(dict(bundle,model_hash=digest({'weight':3})),150,'cpu-demo'), (bundle,200,'cpu-demo'), (bundle,150,'edge-device')]:
        # Run the boundary check and catch the expected exception:
        try: activate(folder,'rejected-v2',candidate,approval,at,target)
        except ValueError: checks.append(False)
        else: raise AssertionError('invalid release activated')
        # Verify contract: `pointer.read_bytes() == before`.
        assert pointer.read_bytes() == before
# Verify contract: `checks == [True, False, False, False]`.
assert checks == [True,False,False,False]
# Print the observed values to compare against the expected result.
print('Accepted original / changed artifact / expired approval / wrong target:',checks)
# Print diagnostic summary of the computed outputs.
print('All failed gates preserved the selected version. This fixture does not authenticate reviewers.')
```

Expected: Accepted original / changed artifact / expired approval / wrong target: [True, False, False, False]
All failed gates preserved the selected version. This fixture does not authenticate reviewers.

## Only the reviewed release passes

**Predict:** Should changing only the target preserve the approval?

![Only the reviewed release passes](../../phases/16-operations/08-modelops-ownership-approval-and-retirement/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal categories are four attempted transitions. Height $1$ means the local contract accepted the transition; height $0$ means it rejected it. Only the original reviewed bundle passes. Changed model content, expiration and target mismatch fail, and the program verifies that none changed the active pointer. Zero bars are observed rejections, not missing measurements.

### Connect it to the computation

This is a deterministic policy test, not a model-performance comparison or proof of authenticated approval. Use the real model/image identities and a trusted review process when transferring it to deployment.

```python
# Compute figure data for: Only the reviewed release passes
# Evaluate `visual_data` from the current inputs and state.
visual_data={'kind':'bar','labels':['reviewed','changed model','expired','wrong target'],'xlabel':'attempted transition','ylabel':'accepted (1) or rejected (0)','series':[{'label':'observed policy result','y':[int(v) for v in checks]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:07:41.936954+00:00. JAX 0.9.2.

```text
Accepted original / changed artifact / expired approval / wrong target: [True, False, False, False]
All failed gates preserved the selected version. This fixture does not authenticate reviewers.
Accepted original / changed artifact / expired approval / wrong target: [True, False, False, False]
All failed gates preserved the selected version. This fixture does not authenticate reviewers.
Expiry is exclusive: 199 accepted, 200 rejected.
Changed evaluation invalidates the old approval.
Ownership change requires renewed review.
PASS: operations-08

```

## Test the exact expiry boundary

**Predict before running:** Is an approval still valid at its expiration timestamp?

```python
# Experiment — Test the exact expiry boundary: Boundary conventions must be explicit.
# Verify contract: `verify_release(bundle, approval, 199, 'cpu-demo')`.
assert verify_release(bundle,approval,199,'cpu-demo')
# Run the boundary check and catch the expected exception:
try: verify_release(bundle,approval,200,'cpu-demo')
except ValueError: print('Expiry is exclusive: 199 accepted, 200 rejected.')
else: raise AssertionError('expired approval accepted')
```

**Expected:** Time 199 is accepted; 200 is rejected.

Boundary conventions must be explicit. Production clocks and identity systems need their own operational guarantees.

## Make it yours

Change the evaluation hash while keeping the model unchanged, and prove the old approval no longer applies.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `digest(...)` — Call `digest` with your updated parameters or inputs from this lesson's workspace.
- `verify_release(...)` — Call `verify_release` with your updated parameters or inputs from this lesson's workspace.

**Step-by-step implementation plan:**
1. Run the boundary check and catch the expected exception:

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Change the evaluation hash while keeping the model unchanged, and...
rechecked = dict(...)  # TODO: compute rechecked
# Run the boundary check and catch the expected exception:
try: verify_release(rechecked,approval,150,'cpu-demo')
except ValueError: print('Changed evaluation invalidates the old approval.')
else: raise AssertionError('stale approval accepted')
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Change the evaluation hash while keeping the model unchanged, and...
rechecked=dict(bundle,evaluation_hash=digest({'high_slice_mse':9.}))
# Run the boundary check and catch the expected exception:
try: verify_release(rechecked,approval,150,'cpu-demo')
except ValueError: print('Changed evaluation invalidates the old approval.')
else: raise AssertionError('stale approval accepted')
```

</details>

## Review a changed owner

**transfer**

Change the responsible owner and check whether the old bundle approval can be reused.

<details><summary>Hint</summary>

The owner is part of the reviewed release identity.

</details>

### How to write: Review a changed owner — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `owner(...)` — Call `owner` with your updated parameters or inputs from this lesson's workspace.
- `digest(...)` — Call `digest` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Review a changed owner (transfer): Ownership is an operational responsibility, so this policy...
2. Verify contract: `digest(changed_owner) != approval['bundle_hash']`.
3. Run the boundary check and catch the expected exception:

**Starter code scaffold (fill in the TODOs):**

```python
# Review a changed owner (transfer): Ownership is an operational responsibility, so this policy...
changed_owner = dict(...)  # TODO: compute changed_owner
# Verify contract: `digest(changed_owner) != approval['bundle_hash']`.
assert digest(changed_owner)  # TODO: complete assertion check
# Run the boundary check and catch the expected exception:
try: verify_release(changed_owner,approval,150,'cpu-demo')
except ValueError: print('Ownership change requires renewed review.')
else: raise AssertionError('changed owner accepted')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Review a changed owner (transfer): Ownership is an operational responsibility, so this policy...
changed_owner=dict(bundle,owner='different-team')
# Verify contract: `digest(changed_owner) != approval['bundle_hash']`.
assert digest(changed_owner)!=approval['bundle_hash']
# Run the boundary check and catch the expected exception:
try: verify_release(changed_owner,approval,150,'cpu-demo')
except ValueError: print('Ownership change requires renewed review.')
else: raise AssertionError('changed owner accepted')
```

Ownership is an operational responsibility, so this policy requires review after it changes. Organizations may choose different policies, but the rule must be explicit.

</details>

## Check your understanding

An approval record contains a reviewer name and a matching hash. Does that authenticate the reviewer?

1. Yes, a hash verifies the reviewer identity.
2. No. The fixture checks binding and expiry; trusted identity and authorization are separate requirements.
3. Only if the model has low validation loss.

<details><summary>Answer and explanation</summary>

No. The fixture checks binding and expiry; trusted identity and authorization are separate requirements.

Here ModelOps means the operational ownership of models across their lifecycle: inventory, intended use, review, deployment status, monitoring, incident response and retirement. Teams use overlapping MLOps and ModelOps terms; the useful distinction is the responsibility and evidence each process owns.

</details>

## Diagnose the result

Compare the whole bundle hash and target first, then time boundaries and reviewer authority. Never bypass a rejected rollout by editing a passed flag or pointing the alias manually. Keep the rejected transition in the incident record.

## Carry forward

- Inventory the complete release
- Separate selection, rollout and monitoring
- Work through the four decisions

## Keep your evidence

Keep the reviewed bundle hash, owner and target, expiry boundary, failed-transition pointer equality and changed-owner rejection. Explain why the local reviewer field is not authenticated approval.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [MLflow experiment tracking](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
- [MLflow model registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [Kubernetes deployment rollout and rollback](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

