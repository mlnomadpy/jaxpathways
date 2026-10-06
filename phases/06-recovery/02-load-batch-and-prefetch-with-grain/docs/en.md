# Load, batch, and checkpoint order with Grain

Phase 06: Data & checkpoint recovery · about 75 minutes · CPU

## What you will be able to do

- Build a Grain source/sampler/batch pipeline and verify its record order.
- Restore the next-read boundary rather than restarting from a seed.
- Check each epoch coverage independently.
- Distinguish tested serial iteration from unmeasured worker/prefetch performance.

## The problem

Now that you can check example order yourself, let’s use Grain to manage the input pipeline. We’ll shuffle two small epochs, batch the records, save the iterator position, and replay the next batch. Using zero worker processes keeps the sequence easy to inspect; this exercise does not measure prefetch speed.

## The idea

A resumable iterator preserves a position in a particular data sequence. Restoring it should reproduce the next batch under the same dataset and configuration. Prefetching makes it especially important to separate requested data from consumed data.

## Restore the next batch by identity

Suppose batches are A/B, C/D and E/F. After training completes the C/D update, the next restored batch should be E/F. Returning another two-element array passes a shape check but may repeat or skip data.

Capture the iterator state at the model's completed-update boundary. Do not substitute a guessed offset unless the pipeline contract makes it sufficient. Shuffling, filtering and prefetching can make requested batch counts differ from completed training work.

Compare exact IDs and values before and after restore. Then change the dataset identity deliberately and require the restore process to detect the incompatible sequence. The same numeric cursor can point to different examples in another dataset.

### Pause and reason

Why can a correct cursor still produce an incorrect resume?

<details><summary>Compare your reasoning</summary>

Position only has meaning relative to its data sequence and configuration. Recovery needs those identities as well as the cursor.

</details>

## Map a record key to a testable source

TutorialSource implements len and indexing. A request for key i returns the same $\mathrm{id}=i$, $x=i/10$ and $y=2x-1$. Its repr includes a version and count because the iterator state records its source configuration. No file I/O, decoder or download is hidden in this example.

In a real corpus, a source key must remain associated with the same record revision. A human-readable repr is only a basic compatibility check; preserve a dataset revision or digest separately. Changing the underlying values without changing the source identity can defeat a cursor restore even if record counts match.

## Follow the sampler-to-batch boundary

IndexSampler chooses record keys over two epochs from a fixed seed. DataLoader reads them and applies `Batch(4)`. The ID vector is the evidence of actual record order after these operations; feature shapes alone would not tell you whether the order changed.

We have twelve records, so one epoch contains exactly three full batches. operations order matters: per-example transforms normally precede batching, while a batch transform receives grouped values. Keep an independent relation check on labels to verify that transforms did not break alignment.

```text
IndexSampler → record keys → TutorialSource[key]
             → Batch(4) → iterator → {id[4],x[4],y[4]}
```

## Snapshot the iterator after consumption

After `next(iterator)` returns the first batch, get_state captures the next-read boundary. We read the second batch as our expected future. A fresh iterator receives `set_state(snapshot)`, then next returns that same second batch. Saving before the first batch would repeat it; saving after the second would skip it on a restore intended for the first boundary.

The snapshot is opaque bytes. Let Grain interpret it rather than editing its internal fields. The two iterators must use compatible sampler, source and worker configuration. The practical check compares all three fields, not only IDs, so deterministic order cannot hide changed content.

## Worker and buffer settings are workload choices

`worker_count=0` runs in the current process and is convenient for debugging or notebook execution. With workers enabled, decoding/transformation can run in child processes and buffer outputs; top-level source and transform classes must be serializable, and a main guard is important in standalone multiprocessing scripts. The code here deliberately does not start workers.

Prefetching can overlap preparation and consumption, but its benefit depends on storage, transform cost, device demand and memory use. Changing workers or buffers is a new configuration to validate and measure. Do not infer that this small in-memory source needs prefetching, or claim a pipeline speedup without a representative synchronized workload.

## Prepare the data and state

Create main.py in your course workspace. Use the tested course environment and add this block first.

```python
import numpy as np
import grain.python as grain
class TutorialSource:
    def __init__(self,count=12):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self,index):
        value = np.float32(index/10)
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    def __repr__(self):
        return f"TutorialSource(v1,n={self.count})"
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)
```

This setup makes the dataset and software assumptions explicit. No external dataset or accelerator is required.

## Build the pipeline or state transition

Append this block to the same file; follow the named state objects through each function.

```python
loader = make_loader()
iterator = iter(loader)
first_batch = next(iterator)
snapshot = iterator.get_state()
second_batch = next(iterator)
restored_iterator = iter(make_loader())
restored_iterator.set_state(snapshot)
replayed_batch = next(restored_iterator)
```

The function boundaries expose which inputs determine the next output and which state must be preserved.

## Run the comparison

Append the checks, save main.py and run python main.py. Predict what should agree before running.

```python
np.testing.assert_array_equal(second_batch["id"],replayed_batch["id"])
for key in ("x","y"):
    np.testing.assert_array_equal(second_batch[key],replayed_batch[key])
for batch in (first_batch,second_batch):
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
one_epoch = list(make_loader())[:3]
np.testing.assert_array_equal(np.sort(np.concatenate([b["id"] for b in one_epoch])),np.arange(12))
print("First IDs:",first_batch["id"].tolist())
print("Next IDs:",second_batch["id"].tolist())
print("Iterator state bytes:",len(snapshot))
print("Restored next batch matches")
```

The comparison checks the next behavior, not merely whether a save call succeeded.

## Run the example

```python
import numpy as np
import grain.python as grain
class TutorialSource:
    def __init__(self,count=12):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self,index):
        value = np.float32(index/10)
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    def __repr__(self):
        return f"TutorialSource(v1,n={self.count})"
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

loader = make_loader()
iterator = iter(loader)
first_batch = next(iterator)
snapshot = iterator.get_state()
second_batch = next(iterator)
restored_iterator = iter(make_loader())
restored_iterator.set_state(snapshot)
replayed_batch = next(restored_iterator)

np.testing.assert_array_equal(second_batch["id"],replayed_batch["id"])
for key in ("x","y"):
    np.testing.assert_array_equal(second_batch[key],replayed_batch[key])
for batch in (first_batch,second_batch):
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
one_epoch = list(make_loader())[:3]
np.testing.assert_array_equal(np.sort(np.concatenate([b["id"] for b in one_epoch])),np.arange(12))
print("First IDs:",first_batch["id"].tolist())
print("Next IDs:",second_batch["id"].tolist())
print("Iterator state bytes:",len(snapshot))
print("Restored next batch matches")
```

Expected: The restored next IDs, features and labels match. One epoch contains twelve unique IDs. Exact shuffled order is determined by tested Grain version/configuration.

## Restoring the iterator repeats the next batch

**Predict:** Does the restored sequence start at the first batch again?

![Restoring the iterator repeats the next batch](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

Each group is a slot within a batch, and bar height gives the example ID in that slot. Compare the “next batch” and “restored next” bars beside one another: their heights match at every slot. They are adjacent bars, not overlapping curves.

The first batch contains $(8,6,7,9)$. The uninterrupted next batch contains $(0,5,1,2)$, and the restored iterator produces that same sequence. The zero-height bars in the first slot represent example ID $0$; they are not missing data.

### Connect it to the computation

The checkpoint captures the iterator after the first batch. Restoring it therefore resumes at the next batch, rather than replaying the first one. Equal IDs in the same slots establish that both membership and ordering agree at this boundary.

IDs are labels, so a taller bar does not indicate a larger input value or a better batch. This comparison verifies the next batch only. To establish a complete training restart, also restore the matching model, optimizer, random state, and dataset configuration, then compare the next update.

```python
visual_data = {'kind': 'bar', 'labels': ['slot 0', 'slot 1', 'slot 2', 'slot 3'], 'ylabel': 'example ID', 'series': [{'label': 'first batch', 'y': first_batch['id'].tolist()}, {'label': 'next batch', 'y': second_batch['id'].tolist()}, {'label': 'restored next', 'y': replayed_batch['id'].tolist()}]}
```

## Recorded reference execution

CPU run: 2026-10-06T21:58:17.292717+00:00. JAX 0.9.2.

```text
First IDs: [8, 6, 7, 9]
Next IDs: [0, 5, 1, 2]
Iterator state bytes: 329
Restored next batch matches
First IDs: [8, 6, 7, 9]
Next IDs: [0, 5, 1, 2]
Iterator state bytes: 329
Restored next batch matches
Same seed restarts; it does not resume
Each epoch covers all twelve IDs
Next-epoch IDs replayed: [[2, 0, 11, 1], [5, 7, 6, 8]]
Expected sampler compatibility rejection
PASS: recovery-02

```

## A fresh iterator restarts

**Predict before running:** If you use the same seed but do not restore state, which batch arrives next?

```python
fresh_first = next(iter(make_loader()))
np.testing.assert_array_equal(fresh_first["id"],first_batch["id"])
assert not np.array_equal(fresh_first["id"],second_batch["id"])
print("Same seed restarts; it does not resume")
```

**Expected:** The first batch repeats, rather than replaying the saved next batch.

The seed fixes ordering. Position is separate state.

## Audit both epochs

**Predict before running:** Predict total record count and uniqueness per epoch.

```python
two_epochs = list(make_loader())
assert len(two_epochs)==6
for start in (0,3):
    values = np.concatenate([b["id"] for b in two_epochs[start:start+3]])
    np.testing.assert_array_equal(np.sort(values),np.arange(12))
print("Each epoch covers all twelve IDs")
```

**Expected:** Six batches; each set of three covers twelve records exactly once.

Across epochs, repeated IDs are intentional. Check coverage at the correct epoch boundary instead of treating every repeat as a bug.

## Make it yours

Consume two batches, snapshot, and replay the third batch with a new compatible iterator.

<details><summary>Reference solution</summary>

```python
later = iter(make_loader())
next(later);next(later)
later_state = later.get_state()
expected_third = next(later)
new_iterator = iter(make_loader());new_iterator.set_state(later_state)
np.testing.assert_array_equal(next(new_iterator)["id"],expected_third["id"])
```

</details>

## Replay through an epoch boundary

**Transfer**

Save the iterator after the third batch and compare the next two batches with a fresh restored iterator. Check IDs and values. Explain why reproducing only the last batch of one epoch is not sufficient.

<details><summary>Hint</summary>

The loader has two epochs; the fourth and fifth batches belong to the next epoch.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
boundary=iter(make_loader())
for _ in range(3):next(boundary)
boundary_state=boundary.get_state()
expected=[next(boundary) for _ in range(2)]
replay=iter(make_loader());replay.set_state(boundary_state)
for reference in expected:
    actual=next(replay)
    for name in ('id','x','y'):np.testing.assert_array_equal(actual[name],reference[name])
print('Next-epoch IDs replayed:',[b['id'].tolist() for b in expected])
```

An epoch boundary changes the sampling context. Testing across it checks more of the iterator contract than replaying one batch inside an epoch.

</details>

## Reject a changed sampler

**Challenge**

Try restoring the original snapshot into a loader with a different seed. Capture the compatibility error, then rebuild the matching loader.

<details><summary>Hint</summary>

Read the restore diagnostic; changing the seed changes sampler identity.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
incompatible = iter(make_loader(seed=99))
try:
    incompatible.set_state(snapshot)
except ValueError:
    print("Expected sampler compatibility rejection")
else:
    raise AssertionError("Expected changed sampler rejection")
compatible = iter(make_loader(seed=42));compatible.set_state(snapshot)
np.testing.assert_array_equal(next(compatible)["id"],second_batch["id"])
```

A snapshot cannot be assumed to apply to a different ordering configuration. Recreate the same pipeline configuration before restoring; explicit version/data identity checks remain necessary.

</details>

## Check your understanding

What makes a fresh loader resume at the saved next batch?

1. Only using the original seed.
2. Restoring compatible iterator state after rebuilding its source/sampler configuration.
3. Calling next once regardless of checkpoint position.

<details><summary>Answer and explanation</summary>

Restoring compatible iterator state after rebuilding its source/sampler configuration.

Seed and source determine order, but iterator state determines where the resumed reader begins.

</details>

## Diagnose the result

A snapshot cannot be assumed to apply to a different ordering configuration. Recreate the same pipeline configuration before restoring; explicit version/data identity checks remain necessary.

## Carry forward

- Map a record key to a testable source
- Follow the sampler-to-batch boundary
- Snapshot the iterator after consumption
- Worker and buffer settings are workload choices

## Keep your evidence

Keep Grain sampler configuration and exact example IDs for full and partial batches; save iterator bytes, recreate the loader, and compare the next batch before/after restore. Record the incompatible configuration failure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [NumPy Generator permutation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html)
- [Grain DataLoader guide](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html)

