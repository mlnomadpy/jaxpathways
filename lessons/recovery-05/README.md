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
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
def transition(state):
    return {'weight':state['weight']+.25,'step':state['step']+1}
def publish(path, step):
    pending=root/'LATEST.pending'
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    pending.replace(root/'LATEST.json')
def accepted():
    return json.loads((root/'LATEST.json').read_text())
```

The manifest has its own acceptance boundary. State updates return new arrays so later work cannot redefine the snapshot.

## Save, continue and verify

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
snapshot=transition(state)
path=root/'step_1'
t0=time.perf_counter()
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    returned=time.perf_counter()
    assert not (root/'LATEST.json').exists()
    live=transition(snapshot)
    cp.wait_until_finished()
    committed=time.perf_counter()
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    assert int(restored['step'])==1 and int(live['step'])==2
    publish(path,1)
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
assert report['accepted_step']==1
```

Wait on the real background writer, restore against the snapshot contract and publish only after the comparison passes.

## Interrupt publication deliberately

Create a Python file in the course environment. Add this block after the preceding block, then run the assembled file.

```python
# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    cp.wait_until_finished()
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
assert accepted()['step']==1
with ocp.StandardCheckpointer() as cp:
    previous=cp.restore(accepted()['path'],target=snapshot)
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
assert int(previous['step'])==1
print('Checkpoint workspace:',root)
print(json.dumps(report,indent=2))
print('Unpublished step 2 exists; application recovery still selects verified step 1.')

```

The injected application interruption leaves the old accepted checkpoint intact even though a newer directory exists.

## Run the example

```python
import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
root=Path(tempfile.mkdtemp(prefix='jax-async-lesson-'))
def transition(state):
    return {'weight':state['weight']+.25,'step':state['step']+1}
def publish(path, step):
    pending=root/'LATEST.pending'
    pending.write_text(json.dumps({'path':str(path),'step':step}))
    pending.replace(root/'LATEST.json')
def accepted():
    return json.loads((root/'LATEST.json').read_text())

state={'weight':jnp.arange(1024,dtype=jnp.float32),'step':jnp.array(0,jnp.int32)}
snapshot=transition(state)
path=root/'step_1'
t0=time.perf_counter()
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(path,args=ocp.args.StandardSave(snapshot))
    returned=time.perf_counter()
    assert not (root/'LATEST.json').exists()
    live=transition(snapshot)
    cp.wait_until_finished()
    committed=time.perf_counter()
    restored=cp.restore(path,args=ocp.args.StandardRestore(snapshot))
    np.testing.assert_array_equal(restored['weight'],snapshot['weight'])
    assert int(restored['step'])==1 and int(live['step'])==2
    publish(path,1)
report={'save_call_seconds':returned-t0,'remaining_work_and_wait_seconds':committed-returned,
        'accepted_step':accepted()['step'],'live_step':int(live['step'])}
assert report['accepted_step']==1

# A deliberately interrupted application publication leaves the previous commit selected.
unpublished=root/'step_2'
with ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()) as cp:
    cp.save(unpublished,args=ocp.args.StandardSave(live))
    cp.wait_until_finished()
    try:
        raise RuntimeError('injected interruption before updating LATEST')
    except RuntimeError:
        pass
assert accepted()['step']==1
with ocp.StandardCheckpointer() as cp:
    previous=cp.restore(accepted()['path'],target=snapshot)
np.testing.assert_array_equal(previous['weight'],snapshot['weight'])
assert int(previous['step'])==1
print('Checkpoint workspace:',root)
print(json.dumps(report,indent=2))
print('Unpublished step 2 exists; application recovery still selects verified step 1.')

```

Expected: The accepted step remains $1$ while the live state reaches $2$. A real local checkpoint restores the saved vector. Both timing intervals are observed and vary by machine.

## A live step can be ahead of the accepted checkpoint

**Predict:** After the publication interruption, which step will application recovery select?

![A live step can be ahead of the accepted checkpoint](../../phases/06-recovery/05-asynchronous-checkpoints-and-failure-boundaries/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal labels name three different state positions; the vertical axis is a completed-step count, not elapsed time. The saved snapshot and accepted checkpoint both have height $1$, while the live computation reaches $2$. Equal bars here mean equal step counters.

The gap of one step is intentional. It shows work that the live process has completed but the application has not accepted as its recovery point. The failed publication does not move the accepted bar to two merely because a newer directory is present.

### Connect it to the computation

The code verifies the saved vector and then injects an interruption before publishing the next checkpoint. Recovery consults the earlier manifest and restores the earlier vector. The full state comparison, rather than bar height alone, establishes which values were restored.

This plot is a local controlled failure experiment. It does not measure checkpoint bandwidth, simulate power loss or establish multi-host atomicity. Inspect the separately recorded timing intervals and exception output for those different questions.

```python
visual_data={'kind':'bar','labels':['saved snapshot','live state','accepted checkpoint'],'ylabel':'completed step','series':[{'label':'local publication experiment','y':[int(snapshot["step"]),int(live["step"]),report["accepted_step"]]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T23:00:31.473819+00:00. JAX 0.9.2.

```text
Checkpoint workspace: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-async-lesson-q4zu44sx
{
  "save_call_seconds": 0.0035762079060077667,
  "remaining_work_and_wait_seconds": 0.016465583816170692,
  "accepted_step": 1,
  "live_step": 2
}
Unpublished step 2 exists; application recovery still selects verified step 1.
Checkpoint workspace: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jax-async-lesson-78sk0k95
{
  "save_call_seconds": 0.002004500012844801,
  "remaining_work_and_wait_seconds": 0.008641333784908056,
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
def injected_failure():
    raise RuntimeError('injected finalization callback failure')
failed_path=root/'callback_failure'
cp=ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler(),async_options=ocp.options.AsyncOptions(post_finalization_callback=injected_failure))
try:
    cp.save(failed_path,args=ocp.args.StandardSave(live))
    try:
        cp.wait_until_finished()
    except RuntimeError as error:
        assert 'injected' in str(error)
        print('Expected background failure surfaced:',error)
    else:
        raise AssertionError('background error was hidden')
finally:
    cp.close()
assert accepted()['step']==1

```

**Expected:** An intentional error log and caught RuntimeError; the accepted step is unchanged.

The controlled failure happens in the finalization callback. Some storage may already exist; the application still requires successful completion and verification before accepting it.

## Make it yours

Verify and publish the already written second checkpoint. Restore it and compare the next functional transition against the live state. Keep the original snapshot unchanged.

<details><summary>Reference solution</summary>

```python
with ocp.StandardCheckpointer() as cp:
    verified=cp.restore(unpublished,target=live)
np.testing.assert_array_equal(verified['weight'],live['weight'])
assert int(verified['step'])==2
publish(unpublished,2)
np.testing.assert_array_equal(transition(verified)['weight'],transition(live)['weight'])
assert accepted()['step']==2 and int(snapshot['step'])==1
print('Step two verified and accepted; next transition agrees.')
```

</details>

## Keep only a genuinely accepted target

**Transfer**

Create a manifest pointing to a nonexistent checkpoint in a disposable variable. Write a preflight that rejects it without replacing the real accepted manifest. Explain why a preflight alone cannot prove numerical correctness.

<details><summary>Hint</summary>

Check path existence and the declared step before restoration; still compare restored contents afterward.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def preflight(record):
    if not Path(record['path']).is_dir() or int(record['step'])<0:
        raise ValueError('checkpoint target unavailable')
    return record
before=accepted()
try:preflight({'path':str(root/'never_saved'),'step':99})
except ValueError:pass
else:raise AssertionError('missing target accepted')
assert accepted()==before
preflight(before)
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

Keep the requested/completed/accepted step log, the restored snapshot comparison, interrupted-publication result and surfaced finalization error. Explain why directory presence alone does not establish acceptance.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Orbax asynchronous checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/async_checkpointing.html)

