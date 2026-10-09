# Asynchronous checkpoints and failure boundaries

Phase 06: Data & checkpoint recovery · about 90 minutes · CPU

## What you will be able to do

- Distinguish requested, completed, verified and published checkpoint state.
- Continue a functional update while an asynchronous writer is alive.
- Recover from the last accepted checkpoint after an interrupted publication.
- Surface an asynchronous failure before announcing a checkpoint.

## The problem

Your training loop says “saved,” then the process disappears. Which step can you safely resume? We will separate an asynchronous save request, completed storage work and the application’s announcement of a recoverable checkpoint. You will interrupt publication deliberately and show which state recovery selects.

## The idea

Asynchronous saving overlaps writing with training. A snapshot moves through requested, writing, completed and accepted states. Recovery must select a checkpoint whose completion and validity satisfy the actual storage contract.

## Distinguish a save request from a recovery point

Imagine live training reaches step 42 while a snapshot of step 40 is still being written. The live progress log does not make step 42 recoverable. Failure before the pending snapshot is accepted may leave an earlier checkpoint as the valid choice.

Keep completion signals separate from the record used to select accepted snapshots. Validate required files and metadata before making a new snapshot eligible. Also ensure live training cannot mutate the snapshot's intended contents during writing.

The saved/live/accepted bars represent different boundaries. Their gaps describe work at risk of replay or loss, not necessarily a defect. Interpret them against the failure location and the checkpoint API's stated guarantees.

$$
T_{\text{commit}} = \max_{h \in \{1,\dots,H\}} T_{\text{write}}^{(h)} < T_{\text{rename}}
$$

### Pause and reason

Can a “save requested” message justify deleting the previous accepted checkpoint?

<details><summary>Compare your reasoning</summary>

No. The replacement may never complete or validate. Retention should preserve a recoverable accepted checkpoint until the replacement meets the acceptance contract.

</details>

## Define a completed training boundary

Let the completed training step be $s$. Our state contains a weight vector and that step counter. A snapshot at $s=1$ must restore the vector belonging to that step, even if the live computation advances to $s=2$. The example adds the same known increment to every weight, so an independent array comparison can identify the snapshot without relying on a decreasing training loss.

A real training snapshot also includes optimizer memory, random state and input position from the preceding lessons. Asynchronous writing changes when storage finishes; it does not reduce the state that must be captured.

## Separate save latency from acceptance

The timer before and after save measures the blocking part of the call. A second interval includes our continuing update and the remaining completion wait. These intervals are adjacent in one run, so their sum describes this application’s request-to-completion boundary. Neither isolates disk throughput.

Keep the checkpointer alive until its work finishes. A quickly returned call can already be finished for a small snapshot; do not require a race to reproduce. The invariant we control is that the application does not update its accepted-checkpoint manifest before completion and restoration checks.

## Publish a checkpoint you can actually restore

After waiting, restore against an explicit target and compare the step and every saved value with the snapshot. Write a new manifest to a temporary sibling and replace the old manifest only after the checks pass. On the local filesystem used here this avoids a reader seeing a half-written JSON file.

A rename alone is not a universal durability or distributed-consistency protocol. Object stores, power failures, concurrent writers and multiple hosts need storage-specific guarantees. Our local manifest is an instructional boundary around a single writer, not a replacement for a production checkpoint manager.

## Practice interruption without relying on lucky timing

The second checkpoint is fully written, but an injected exception prevents application publication. Recovery still selects step one through the unchanged manifest. Step two exists on disk yet is not selected by this application. This distinction explains why “directory exists” is a weak recovery test.

The extra failure experiment injects an exception in a finalization callback. The exception is observed when waiting; no manifest is published for that attempt. It exercises error propagation after finalization, not a claim to reproduce every disk failure. Preserve the old accepted checkpoint until the new one is accepted.

## Translate the boundary into an operational rule

Log the requested step, completed step and accepted step separately. If a job exits with an outstanding save, wait during orderly shutdown; if it dies abruptly, resume from the last accepted artifact and expect to repeat work after that step. Record the replay policy rather than silently claiming no work was lost.

When a restore differs, compare sample IDs and state before tuning tolerances. A different next batch, reset optimizer or mismatched package/data contract is a different trajectory, even if the restored weights look reasonable.

## Define functional state and publication

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# Step 1 — Define functional state and publication: The manifest has its own acceptance boundary.
# Import json for this computation.
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
# Read or serialize artifact data on disk (`root`).
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
# Function `transition(state)` implementing this stage's computation:
def transition(state):
    # Return `{'weight': state['weight'] + 0.25, 'step': state['step'] + 1}` to the caller.
    return {'weight':state['weight']+.25,'step':state['step']+1}
# Function `publish(path, step)` implementing this stage's computation:
def publish(path, step):
    # Compute `pending` from `root/'LATEST.pending'`
    pending=root/'LATEST.pending'
    # Write the serialized artifact payload to disk.
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    # Run `pending.replace` to perform the next check or state transition.
    pending.replace(root/'LATEST.json')
# Function `accepted()` implementing this stage's computation:
def accepted():
    # Return `json.loads((root / 'LATEST.json').read_text())` to the caller.
    return json.loads((root/'LATEST.json').read_text())
```

The manifest has its own acceptance boundary. State updates return new arrays so later work cannot redefine the snapshot.

## Save, continue and verify

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# Step 2 — Save, continue and verify: Wait on the real background writer, restore against the snapshot...
# Construct `state` via `{'weight':jnp.arange(1024,dtype=jnp.float32),'step':...`
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
# Run `transition` to compute `snapshot`.
snapshot=transition(state)
# Compute `path` from `root/'step_1'`
path=root/'step_1'
# Record execution timing or profiler trace in `t0`.
t0=time.perf_counter()
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    # Record execution timing or profiler trace in `returned`.
    returned=time.perf_counter()
    # Assert invariant `not (root/'LATEST.json').exists()` holds
    assert not (root/'LATEST.json').exists()
    # Run `transition` to compute `live`.
    live=transition(snapshot)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Record execution timing or profiler trace in `committed`.
    committed=time.perf_counter()
    # Run `cp.restore` to compute `restored`.
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    # Execute `np.testing.assert_array_equal(restored['weight'],snapshot['w`
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    # Assert invariant `int(restored['step'])==1 and int(live['step'])==2` holds
    assert int(restored['step'])==1 and int(live['step'])==2
    # Run `publish` to perform the next check or state transition.
    publish(path,1)
# Compute `report` from `{'save_call_seconds':returned-t0,'remaining_work_and...`
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
# Assert invariant `report['accepted_step']==1` holds
assert report['accepted_step']==1
```

Wait on the real background writer, restore against the snapshot contract and publish only after the comparison passes.

## Interrupt publication deliberately

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run the boundary check and catch the expected exception:
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `previous`.
    previous=cp.restore(accepted()['path'],target=snapshot)
# Execute `np.testing.assert_array_equal(previous['weight'],snapshot['w`
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
# Assert invariant `int(previous['step'])==1` holds
assert int(previous['step'])==1
# Print the observed values to compare against the expected result.
print('Checkpoint workspace:',root)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report,indent=2))
# Print diagnostic summary of the computed outputs.
print('Unpublished step 2 exists; application recovery still selects verified step 1.')
```

The injected application interruption leaves the old accepted checkpoint intact even though a newer directory exists.

## Run the example

```python
# Step 1 — Define functional state and publication: The manifest has its own acceptance boundary.
# Import json for this computation.
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
# Read or serialize artifact data on disk (`root`).
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
# Function `transition(state)` implementing this stage's computation:
def transition(state):
    # Return `{'weight': state['weight'] + 0.25, 'step': state['step'] + 1}` to the caller.
    return {'weight':state['weight']+.25,'step':state['step']+1}
# Function `publish(path, step)` implementing this stage's computation:
def publish(path, step):
    # Compute `pending` from `root/'LATEST.pending'`
    pending=root/'LATEST.pending'
    # Write the serialized artifact payload to disk.
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    # Run `pending.replace` to perform the next check or state transition.
    pending.replace(root/'LATEST.json')
# Function `accepted()` implementing this stage's computation:
def accepted():
    # Return `json.loads((root / 'LATEST.json').read_text())` to the caller.
    return json.loads((root/'LATEST.json').read_text())

# Step 2 — Save, continue and verify: Wait on the real background writer, restore against the snapshot...
# Construct `state` via `{'weight':jnp.arange(1024,dtype=jnp.float32),'step':...`
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
# Run `transition` to compute `snapshot`.
snapshot=transition(state)
# Compute `path` from `root/'step_1'`
path=root/'step_1'
# Record execution timing or profiler trace in `t0`.
t0=time.perf_counter()
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    # Record execution timing or profiler trace in `returned`.
    returned=time.perf_counter()
    # Assert invariant `not (root/'LATEST.json').exists()` holds
    assert not (root/'LATEST.json').exists()
    # Run `transition` to compute `live`.
    live=transition(snapshot)
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Record execution timing or profiler trace in `committed`.
    committed=time.perf_counter()
    # Run `cp.restore` to compute `restored`.
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    # Execute `np.testing.assert_array_equal(restored['weight'],snapshot['w`
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    # Assert invariant `int(restored['step'])==1 and int(live['step'])==2` holds
    assert int(restored['step'])==1 and int(live['step'])==2
    # Run `publish` to perform the next check or state transition.
    publish(path,1)
# Compute `report` from `{'save_call_seconds':returned-t0,'remaining_work_and...`
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
# Assert invariant `report['accepted_step']==1` holds
assert report['accepted_step']==1

# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
# Enter `ocp.AsyncCheckpointer(ocp.StandardCheckpointH` context block:
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    # Run `cp.save` to perform the next check or state transition.
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    # Run `cp.wait_until_finished` to perform the next check or state transition.
    cp.wait_until_finished()
    # Run the boundary check and catch the expected exception:
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `previous`.
    previous=cp.restore(accepted()['path'],target=snapshot)
# Execute `np.testing.assert_array_equal(previous['weight'],snapshot['w`
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
# Assert invariant `int(previous['step'])==1` holds
assert int(previous['step'])==1
# Print the observed values to compare against the expected result.
print('Checkpoint workspace:',root)
# Print diagnostic summary of the computed outputs.
print(json.dumps(report,indent=2))
# Print diagnostic summary of the computed outputs.
print('Unpublished step 2 exists; application recovery still selects verified step 1.')
```

Expected: The accepted step remains $1$ while the live state reaches $2$. A real local checkpoint restores the saved vector. Both timing intervals are observed and vary by machine.

## A live step can be ahead of the accepted checkpoint

**Predict:** After the publication interruption, which step will application recovery select?

![A live step can be ahead of the accepted checkpoint](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal labels name three different state positions; the vertical axis is a completed-step count, not elapsed time. The saved snapshot and accepted checkpoint both have height $1$, while the live computation reaches $2$. Equal bars here mean equal step counters.

The gap of one step is intentional. It shows work that the live process has completed but the application has not accepted as its recovery point. The failed publication does not move the accepted bar to two merely because a newer directory is present.

### Connect it to the computation

The code verifies the saved vector and then injects an interruption before publishing the next checkpoint. Recovery consults the earlier manifest and restores the earlier vector. The full state comparison, rather than bar height alone, establishes which values were restored.

This plot is a local controlled failure experiment. It does not measure checkpoint bandwidth, simulate power loss or establish multi-host atomicity. Inspect the separately recorded timing intervals and exception output for those different questions.

```python
# Compute figure data for: A live step can be ahead of the accepted checkpoint
# Compute `visual_data` from `{'kind':'bar','labels':['saved snapshot','live state...`
visual_data={'kind':'bar','labels':['saved snapshot','live state','accepted checkpoint'],'ylabel':'completed step','series':[{'label':'local publication experiment','y':[int(snapshot["step"]),int(live["step"]),report["accepted_step"]]}]}
```

## Recorded reference execution

CPU run: 2026-10-08T14:03:18.426772+00:00. JAX 0.9.2.

```text
Checkpoint workspace: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-async-lesson-uem4o66m
{
  "save_call_seconds": 0.004411875270307064,
  "remaining_work_and_wait_seconds": 0.014456083066761494,
  "accepted_step": 1,
  "live_step": 2
}
Unpublished step 2 exists; application recovery still selects verified step 1.
Checkpoint workspace: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-async-lesson-351xanch
{
  "save_call_seconds": 0.0018216250464320183,
  "remaining_work_and_wait_seconds": 0.009038458112627268,
  "accepted_step": 1,
  "live_step": 2
}
Unpublished step 2 exists; application recovery still selects verified step 1.
Expected background failure surfaced: injected finalization callback failure
Step two verified and accepted; next transition agrees.
Missing target rejected; accepted checkpoint unchanged.
PASS: recovery-05

```

## Trigger an asynchronous error

**Predict before running:** If finalization reports an error, should directory existence justify announcing success?

```python
# Experiment — Trigger an asynchronous error: The controlled failure happens in the finalization callback.
def injected_failure():
    raise RuntimeError('injected finalization callback failure')
# Compute `failed_path` from `root/'callback_failure'`
failed_path=root/'callback_failure'
# Run `ocp.AsyncCheckpointer` to compute `cp`.
cp=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler(),async_options=ocp.options.AsyncOptions(post_finalization_callback=injected_failure))
# Run the boundary check and catch the expected exception:
try:
    cp.save(failed_path,args=ocp.args.StandardSave(live))
    # Run the boundary check and catch the expected exception:
    try:
        cp.wait_until_finished()
    except RuntimeError as error:
        assert 'injected' in str(error)
        print('Expected background failure surfaced:',error)
    else:
        raise AssertionError('background error was hidden')
finally:
    cp.close()
# Assert invariant `accepted()['step']==1` holds
assert accepted()['step']==1
```

**Expected:** An intentional error log and caught RuntimeError; the accepted step is unchanged.

The controlled failure happens in the finalization callback. Some storage may already exist; the application still requires successful completion and verification before accepting it.

## Make it yours

Verify and publish the already written second checkpoint. Restore it and compare the next functional transition against the live state. Keep the original snapshot unchanged.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `ocp.StandardCheckpointer(...)` — Call `ocp.StandardCheckpointer` with your updated parameters or inputs from this lesson's workspace.
- `cp.restore(...)` — Call `cp.restore` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Enter `ocp.StandardCheckpointer()` context block:
2. Run `cp.restore` to compute `verified`.
3. Execute `np.testing.assert_array_equal(verified['weight'],live['weigh`
4. Assert invariant `int(verified['step'])==2` holds
5. Run `publish` to perform the next check or state transition.

**Starter code scaffold (fill in the TODOs):**

```python
# Exercise solution: Verify and publish the already written second checkpoint.
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `verified`.
    verified = cp.restore(...)  # TODO: compute verified
# Execute `np.testing.assert_array_equal(verified['weight'],live['weigh`
np.testing.assert_array_equal(verified['weight'],live['weight'])
# Assert invariant `int(verified['step'])==2` holds
assert int(verified['step'])  # TODO: complete assertion check
# Run `publish` to perform the next check or state transition.
publish(unpublished,2)
# Execute `np.testing.assert_array_equal(transition(verified)['weight']`
np.testing.assert_array_equal(transition(verified)['weight'],transition(live)['weight'])
# Assert invariant `accepted()['step']==2 and int(snapshot['step'])==1` holds
assert accepted()['step']  # TODO: complete assertion check
# Print the observed values to compare against the expected result.
print('Step two verified and accepted; next transition agrees.')
```

<details><summary>Reference solution</summary>

```python
# Exercise solution: Verify and publish the already written second checkpoint.
# Enter `ocp.StandardCheckpointer()` context block:
with ocp.StandardCheckpointer() as cp:
    # Run `cp.restore` to compute `verified`.
    verified=cp.restore(unpublished,target=live)
# Execute `np.testing.assert_array_equal(verified['weight'],live['weigh`
np.testing.assert_array_equal(verified['weight'],live['weight'])
# Assert invariant `int(verified['step'])==2` holds
assert int(verified['step'])==2
# Run `publish` to perform the next check or state transition.
publish(unpublished,2)
# Execute `np.testing.assert_array_equal(transition(verified)['weight']`
np.testing.assert_array_equal(transition(verified)['weight'],transition(live)['weight'])
# Assert invariant `accepted()['step']==2 and int(snapshot['step'])==1` holds
assert accepted()['step']==2 and int(snapshot['step'])==1
# Print the observed values to compare against the expected result.
print('Step two verified and accepted; next transition agrees.')
```

</details>

## Keep only a genuinely accepted target

**Transfer**

Create a manifest pointing to a nonexistent checkpoint in a disposable variable. Write a preflight that rejects it without replacing the real accepted manifest. Explain why a preflight alone cannot prove numerical correctness.

<details><summary>Hint</summary>

Check path existence and the declared step before restoration; still compare restored contents afterward.

</details>

### How to write: Keep only a genuinely accepted target — Step-by-step recipe & starter scaffold

**Key functions & syntax to use:**
- `target(...)` — Call `target` with your updated parameters or inputs from this lesson's workspace.
- `preflight(...)` — Call `preflight` with your updated parameters or inputs from this lesson's workspace.
- `assert condition` — Verify that the observed output shape, status, or numerical value satisfies the contract.

**Step-by-step implementation plan:**
1. Guard input contract (`not Path(record['path']).is_dir() or int(record['step']) < 0`) and fail fast if violated.
2. Return `record` to the caller.
3. Run `accepted` to compute `before`.
4. Run the boundary check and catch the expected exception:
5. Assert invariant `accepted()==before` holds

**Starter code scaffold (fill in the TODOs):**

```python
# Keep only a genuinely accepted target (Transfer): File presence is a necessary local preflight, not proof that...
def preflight(record):
    # Guard input contract (`not Path(record['path']).is_dir() or int(record['step']) < 0`) and fail fast if violated.
    if not Path(record['path']).is_dir() or int(record['step'])<0:
        raise ValueError('checkpoint target unavailable')
    # Return `record` to the caller.
    return ...  # TODO: return computed result
# Run `accepted` to compute `before`.
before = accepted(...)  # TODO: compute before
# Run the boundary check and catch the expected exception:
try:preflight({'path':str(root/'never_saved'),'step':99})
except ValueError:pass
else:raise AssertionError('missing target accepted')
# Assert invariant `accepted()==before` holds
assert accepted()  # TODO: complete assertion check
# Run `preflight` to perform the next check or state transition.
preflight(before)
# Print the observed values to compare against the expected result.
print('Missing target rejected; accepted checkpoint unchanged.')
```

<details><summary>Reference solution and reasoning</summary>

```python
# Keep only a genuinely accepted target (Transfer): File presence is a necessary local preflight, not proof that...
def preflight(record):
    # Guard input contract (`not Path(record['path']).is_dir() or int(record['step']) < 0`) and fail fast if violated.
    if not Path(record['path']).is_dir() or int(record['step'])<0:
        raise ValueError('checkpoint target unavailable')
    # Return `record` to the caller.
    return record
# Run `accepted` to compute `before`.
before=accepted()
# Run the boundary check and catch the expected exception:
try:preflight({'path':str(root/'never_saved'),'step':99})
except ValueError:pass
else:raise AssertionError('missing target accepted')
# Assert invariant `accepted()==before` holds
assert accepted()==before
# Run `preflight` to perform the next check or state transition.
preflight(before)
# Print the observed values to compare against the expected result.
print('Missing target rejected; accepted checkpoint unchanged.')
```

File presence is a necessary local preflight, not proof that metadata, data or training state is correct. The restore comparison supplies the stronger check.

</details>

## Check your understanding

A new checkpoint directory exists but the completion wait raises an exception. Which recovery decision follows this lesson’s protocol?

1. Accept the newest directory immediately.
2. Delete the previous checkpoint because save returned.
3. Retain the previous accepted checkpoint and investigate the failed attempt.

<details><summary>Answer and explanation</summary>

Retain the previous accepted checkpoint and investigate the failed attempt.

A directory is not the application’s acceptance signal. Completion, restoration checks and publication must succeed before replacing the previous recovery point.

</details>

## Diagnose the result

If the accepted step is ahead of the restored state, inspect publication order and the saved step counter. If wait raises, keep that error and the last accepted manifest; do not suppress the error and publish anyway.

## Carry forward

- Record save request, completion and acceptance separately.
- Keep the asynchronous writer alive until completion or reported failure.
- Verify a restored next transition before replacing the accepted recovery point.

## Keep your evidence

Keep the requested/completed/accepted step log, the restored snapshot comparison, interrupted-publication result and surfaced finalization error. Explain why directory presence alone is separate from acceptance.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [Orbax asynchronous checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/async_checkpointing.html)

