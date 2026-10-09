"""Load, batch, and checkpoint order with Grain: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import numpy for this computation.
import numpy as np
import grain.python as grain
# Define `TutorialSource` module / container with explicit state and forward pass:
class TutorialSource:
    # Function `__init__(self, count)` implementing this stage's computation:
    def __init__(self,count=12):
        # Compute `self.count` from `count`
        self.count = count
    # Function `__len__(self)` implementing this stage's computation:
    def __len__(self):
        # Return `self.count` to the caller.
        return self.count
    # Function `__getitem__(self, index)` implementing this stage's computation:
    def __getitem__(self,index):
        # Cast or evaluate `value` in explicit floating-point precision.
        value = np.float32(index/10)
        # Return `{'id': np.int64(index), 'x': value, 'y': np.float32(2 * value - 1)}` to the caller.
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    # Function `__repr__(self)` implementing this stage's computation:
    def __repr__(self):
        # Return `f'TutorialSource(v1,n={self.count})'` to the caller.
        return f"TutorialSource(v1,n={self.count})"
# Function `make_loader(seed, batch_size, count, drop_remainder)` implementing this stage's computation:
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    # Return `grain.DataLoader(data_source=TutorialSource(count), sampler=grain.IndexSampler(num_records=count, num_epochs=2, shuffle=True, seed=seed), operations=[grain.Batch(batch_size, drop_remainder=drop_remainder)], worker_count=0)` to the caller.
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

# Build the pipeline or state transition
# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
loader = make_loader()
# Run `iter` to compute `iterator`.
iterator = iter(loader)
# Run `next` to compute `first_batch`.
first_batch = next(iterator)
# Run `iterator.get_state` to compute `snapshot`.
snapshot = iterator.get_state()
# Run `next` to compute `second_batch`.
second_batch = next(iterator)
# Run `iter` to compute `restored_iterator`.
restored_iterator = iter(make_loader())
# Run `restored_iterator.set_state` to perform the next check or state transition.
restored_iterator.set_state(snapshot)
# Run `next` to compute `replayed_batch`.
replayed_batch = next(restored_iterator)

# Run the comparison
# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
np.testing.assert_array_equal(second_batch["id"],replayed_batch["id"])
# Iterate over `key` to step through the computation:
for key in ("x","y"):
    # Execute `np.testing.assert_array_equal(second_batch[key],replayed_bat`
    np.testing.assert_array_equal(second_batch[key],replayed_batch[key])
# Iterate over `batch` to step through the computation:
for batch in (first_batch,second_batch):
    # Compute `np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol` as `1e-7)`.
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
# Evaluate `make_loader())[:3` and convert the result into Python scalar/collection `one_epoch`.
one_epoch = list(make_loader())[:3]
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(np.sort(np.concatenate([b["id"] for b in one_epoch])),np.arange(12))
# Print the observed values to compare against the expected result.
print("First IDs:",first_batch["id"].tolist())
# Print diagnostic summary of the computed outputs.
print("Next IDs:",second_batch["id"].tolist())
# Print diagnostic summary of the computed outputs.
print("Iterator state bytes:",len(snapshot))
# Print diagnostic summary of the computed outputs.
print("Restored next batch matches")

# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import numpy for this computation.
import numpy as np
import grain.python as grain
# Define `TutorialSource` module / container with explicit state and forward pass:
class TutorialSource:
    # Function `__init__(self, count)` implementing this stage's computation:
    def __init__(self,count=12):
        # Compute `self.count` from `count`
        self.count = count
    # Function `__len__(self)` implementing this stage's computation:
    def __len__(self):
        # Return `self.count` to the caller.
        return self.count
    # Function `__getitem__(self, index)` implementing this stage's computation:
    def __getitem__(self,index):
        # Cast or evaluate `value` in explicit floating-point precision.
        value = np.float32(index/10)
        # Return `{'id': np.int64(index), 'x': value, 'y': np.float32(2 * value - 1)}` to the caller.
        return {"id":np.int64(index),"x":value,"y":np.float32(2*value-1)}
    # Function `__repr__(self)` implementing this stage's computation:
    def __repr__(self):
        # Return `f'TutorialSource(v1,n={self.count})'` to the caller.
        return f"TutorialSource(v1,n={self.count})"
# Function `make_loader(seed, batch_size, count, drop_remainder)` implementing this stage's computation:
def make_loader(seed=42,batch_size=4,count=12,drop_remainder=False):
    # Return `grain.DataLoader(data_source=TutorialSource(count), sampler=grain.IndexSampler(num_records=count, num_epochs=2, shuffle=True, seed=seed), operations=[grain.Batch(batch_size, drop_remainder=drop_remainder)], worker_count=0)` to the caller.
    return grain.DataLoader(
        data_source=TutorialSource(count),
        sampler=grain.IndexSampler(num_records=count,num_epochs=2,shuffle=True,seed=seed),
        operations=[grain.Batch(batch_size,drop_remainder=drop_remainder)],
        worker_count=0)

# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
loader = make_loader()
# Run `iter` to compute `iterator`.
iterator = iter(loader)
# Run `next` to compute `first_batch`.
first_batch = next(iterator)
# Run `iterator.get_state` to compute `snapshot`.
snapshot = iterator.get_state()
# Run `next` to compute `second_batch`.
second_batch = next(iterator)
# Run `iter` to compute `restored_iterator`.
restored_iterator = iter(make_loader())
# Run `restored_iterator.set_state` to perform the next check or state transition.
restored_iterator.set_state(snapshot)
# Run `next` to compute `replayed_batch`.
replayed_batch = next(restored_iterator)

# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
np.testing.assert_array_equal(second_batch["id"],replayed_batch["id"])
# Iterate over `key` to step through the computation:
for key in ("x","y"):
    # Execute `np.testing.assert_array_equal(second_batch[key],replayed_bat`
    np.testing.assert_array_equal(second_batch[key],replayed_batch[key])
# Iterate over `batch` to step through the computation:
for batch in (first_batch,second_batch):
    # Compute `np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol` as `1e-7)`.
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
# Evaluate `make_loader())[:3` and convert the result into Python scalar/collection `one_epoch`.
one_epoch = list(make_loader())[:3]
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(np.sort(np.concatenate([b["id"] for b in one_epoch])),np.arange(12))
# Print the observed values to compare against the expected result.
print("First IDs:",first_batch["id"].tolist())
# Print diagnostic summary of the computed outputs.
print("Next IDs:",second_batch["id"].tolist())
# Print diagnostic summary of the computed outputs.
print("Iterator state bytes:",len(snapshot))
# Print diagnostic summary of the computed outputs.
print("Restored next batch matches")

# Figure data experiment
# Compute figure data for: Restoring the iterator repeats the next batch
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['slot 0', 'slot 1', 'slot...`
visual_data = {'kind': 'bar', 'labels': ['slot 0', 'slot 1', 'slot 2', 'slot 3'], 'ylabel': 'example ID', 'series': [{'label': 'first batch', 'y': first_batch['id'].tolist()}, {'label': 'next batch', 'y': second_batch['id'].tolist()}, {'label': 'restored next', 'y': replayed_batch['id'].tolist()}]}

# Experiment: A fresh iterator restarts
# Experiment — A fresh iterator restarts: The seed fixes ordering.
fresh_first = next(iter(make_loader()))
# Execute `np.testing.assert_array_equal(fresh_first["id"],first_batch[`
np.testing.assert_array_equal(fresh_first["id"],first_batch["id"])
# Assert invariant `not np.array_equal(fresh_first["id"]` holds
assert not np.array_equal(fresh_first["id"],second_batch["id"])
# Print the observed values to compare against the expected result.
print("Same seed restarts; it does not resume")

# Experiment: Audit both epochs
# Experiment — Audit both epochs: Across epochs, repeated IDs are intentional.
two_epochs = list(make_loader())
# Assert invariant `len(two_epochs)==6` holds
assert len(two_epochs)==6
# Iterate over `start` to step through the computation:
for start in (0,3):
    # Combine or mask array elements to form `values`.
    values = np.concatenate([b["id"] for b in two_epochs[start:start+3]])
    # Create evenly spaced index values in ``.
    np.testing.assert_array_equal(np.sort(values),np.arange(12))
# Print the observed values to compare against the expected result.
print("Each epoch covers all twelve IDs")

# Reference solution. Try the exercise before reading this.
# Exercise solution: Consume two batches, snapshot, and replay the third batch with a new...
later = iter(make_loader())
# Run `next` to perform the next check or state transition.
# Run `next` to perform the next check or state transition.
next(later)
next(later)
# Run `later.get_state` to compute `later_state`.
later_state = later.get_state()
# Run `next` to compute `expected_third`.
expected_third = next(later)
# Run `iter` to compute `new_iterator`.
# Compute `new_iterator` as `iter(make_loader())`.
new_iterator = iter(make_loader())
new_iterator.set_state(later_state)
# Execute `np.testing.assert_array_equal(next(new_iterator)["id"],expec`
np.testing.assert_array_equal(next(new_iterator)["id"],expected_third["id"])

# Reference practice: Replay through an epoch boundary
# Replay through an epoch boundary (Transfer): An epoch boundary changes the sampling context.
boundary=iter(make_loader())
# Repeat the update loop over `range(3)` steps:
# Execute `for _ in range(3):next(boundary)`.
for _ in range(3):next(boundary)
# Run `boundary.get_state` to compute `boundary_state`.
boundary_state=boundary.get_state()
# Compute `expected` from `[next(boundary) for _ in range(2)]`
expected=[next(boundary) for _ in range(2)]
# Run `iter` to compute `replay`.
# Compute `replay` as `iter(make_loader())`.
replay=iter(make_loader())
replay.set_state(boundary_state)
# Iterate over `reference` to step through the computation:
for reference in expected:
    # Run `next` to compute `actual`.
    actual=next(replay)
    # Iterate over `name` to step through the computation:
    for name in ('id','x','y'):np.testing.assert_array_equal(actual[name],reference[name])
# Print the observed values to compare against the expected result.
print('Next-epoch IDs replayed:',[b['id'].tolist() for b in expected])

# Reference practice: Reject a changed sampler
# Reject a changed sampler (Challenge): A snapshot cannot be assumed to apply to a different...
incompatible = iter(make_loader(seed=99))
# Run the boundary check and catch the expected exception:
try:
    incompatible.set_state(snapshot)
except ValueError:
    print("Expected sampler compatibility rejection")
else:
    raise AssertionError("Expected changed sampler rejection")
# Run `iter` to compute `compatible`.
# Compute `compatible` as `iter(make_loader(seed=42))`.
compatible = iter(make_loader(seed=42))
compatible.set_state(snapshot)
# Execute `np.testing.assert_array_equal(next(compatible)["id"],second_`
np.testing.assert_array_equal(next(compatible)["id"],second_batch["id"])
print("PASS: recovery-02")
