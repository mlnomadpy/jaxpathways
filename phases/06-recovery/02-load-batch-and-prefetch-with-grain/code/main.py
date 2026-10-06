"""Load, batch, and checkpoint order with Grain: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
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

# Build the pipeline or state transition
loader = make_loader()
iterator = iter(loader)
first_batch = next(iterator)
snapshot = iterator.get_state()
second_batch = next(iterator)
restored_iterator = iter(make_loader())
restored_iterator.set_state(snapshot)
replayed_batch = next(restored_iterator)

# Run the comparison
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

# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['slot 0', 'slot 1', 'slot 2', 'slot 3'], 'ylabel': 'example ID', 'series': [{'label': 'first batch', 'y': first_batch['id'].tolist()}, {'label': 'next batch', 'y': second_batch['id'].tolist()}, {'label': 'restored next', 'y': replayed_batch['id'].tolist()}]}

# Experiment: A fresh iterator restarts
fresh_first = next(iter(make_loader()))
np.testing.assert_array_equal(fresh_first["id"],first_batch["id"])
assert not np.array_equal(fresh_first["id"],second_batch["id"])
print("Same seed restarts; it does not resume")

# Experiment: Audit both epochs
two_epochs = list(make_loader())
assert len(two_epochs)==6
for start in (0,3):
    values = np.concatenate([b["id"] for b in two_epochs[start:start+3]])
    np.testing.assert_array_equal(np.sort(values),np.arange(12))
print("Each epoch covers all twelve IDs")

# Reference solution. Try the exercise before reading this.
later = iter(make_loader())
next(later);next(later)
later_state = later.get_state()
expected_third = next(later)
new_iterator = iter(make_loader());new_iterator.set_state(later_state)
np.testing.assert_array_equal(next(new_iterator)["id"],expected_third["id"])

# Reference practice: Replay through an epoch boundary
boundary=iter(make_loader())
for _ in range(3):next(boundary)
boundary_state=boundary.get_state()
expected=[next(boundary) for _ in range(2)]
replay=iter(make_loader());replay.set_state(boundary_state)
for reference in expected:
    actual=next(replay)
    for name in ('id','x','y'):np.testing.assert_array_equal(actual[name],reference[name])
print('Next-epoch IDs replayed:',[b['id'].tolist() for b in expected])

# Reference practice: Reject a changed sampler
incompatible = iter(make_loader(seed=99))
try:
    incompatible.set_state(snapshot)
except ValueError:
    print("Expected sampler compatibility rejection")
else:
    raise AssertionError("Expected changed sampler rejection")
compatible = iter(make_loader(seed=42));compatible.set_state(snapshot)
np.testing.assert_array_equal(next(compatible)["id"],second_batch["id"])
print("PASS: recovery-02")
