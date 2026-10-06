"""Design an input pipeline: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
import numpy as np
import hashlib
import time
ids = np.arange(11,dtype=np.int64)
features = (ids/10).astype(np.float32)
labels = (2*features-1).astype(np.float32)
def dataset_digest(x,y):
    digest = hashlib.sha256()
    for array in (x,y):
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
fingerprint = dataset_digest(features,labels)
def epoch_order(seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(ids))

# Build the pipeline or state transition
def make_batches(seed,epoch,batch_size=4,drop_last=False):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = epoch_order(seed,epoch)
    for start in range(0,len(order),batch_size):
        chosen = order[start:start+batch_size]
        if drop_last and len(chosen) < batch_size:
            break
        yield {"id":ids[chosen],"x":features[chosen],"y":labels[chosen]}
start_time = time.perf_counter()
batches = list(make_batches(seed=17,epoch=0))
elapsed = time.perf_counter()-start_time
seen = np.concatenate([b["id"] for b in batches])

# Run the comparison
assert [len(b["id"]) for b in batches] == [4,4,3]
np.testing.assert_array_equal(np.sort(seen),ids)
assert len(np.unique(seen)) == len(ids)
for batch in batches:
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
repeated = np.concatenate([b["id"] for b in make_batches(17,0)])
np.testing.assert_array_equal(seen,repeated)
print("Batch sizes:",[len(b["id"]) for b in batches])
print("Example order:",seen.tolist())
print("Preparation seconds (local observation only):",elapsed)
print("Dataset SHA256:",fingerprint)

import numpy as np
import hashlib
import time
ids = np.arange(11,dtype=np.int64)
features = (ids/10).astype(np.float32)
labels = (2*features-1).astype(np.float32)
def dataset_digest(x,y):
    digest = hashlib.sha256()
    for array in (x,y):
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
fingerprint = dataset_digest(features,labels)
def epoch_order(seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(ids))

def make_batches(seed,epoch,batch_size=4,drop_last=False):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = epoch_order(seed,epoch)
    for start in range(0,len(order),batch_size):
        chosen = order[start:start+batch_size]
        if drop_last and len(chosen) < batch_size:
            break
        yield {"id":ids[chosen],"x":features[chosen],"y":labels[chosen]}
start_time = time.perf_counter()
batches = list(make_batches(seed=17,epoch=0))
elapsed = time.perf_counter()-start_time
seen = np.concatenate([b["id"] for b in batches])

assert [len(b["id"]) for b in batches] == [4,4,3]
np.testing.assert_array_equal(np.sort(seen),ids)
assert len(np.unique(seen)) == len(ids)
for batch in batches:
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
repeated = np.concatenate([b["id"] for b in make_batches(17,0)])
np.testing.assert_array_equal(seen,repeated)
print("Batch sizes:",[len(b["id"]) for b in batches])
print("Example order:",seen.tolist())
print("Preparation seconds (local observation only):",elapsed)
print("Dataset SHA256:",fingerprint)

# Figure data experiment
visual_data = {'kind': 'line', 'x': list(range(len(seen))), 'xlabel': 'position in epoch', 'ylabel': 'example ID', 'series': [{'label': 'shuffled order', 'y': seen.tolist()}], 'boundaries': [3.5, 7.5]}

# Experiment: Compare epochs without a hard-coded order
epoch_one = np.concatenate([b["id"] for b in make_batches(17,1)])
np.testing.assert_array_equal(np.sort(epoch_one),ids)
assert not np.array_equal(epoch_one,seen)
print("Epoch one order:",epoch_one.tolist())

# Experiment: Expose dropped records
dropped = list(make_batches(17,0,drop_last=True))
dropped_ids = np.concatenate([b["id"] for b in dropped])
assert len(dropped_ids)==8
omitted = np.setdiff1d(ids,dropped_ids)
assert len(omitted)==3
print("Omitted IDs:",omitted.tolist())

# Reference solution. Try the exercise before reading this.
last = batches[-1]["y"]
padded_labels = np.pad(last,(0,4-len(last)))
mask = np.array([1]*len(last)+[0]*(4-len(last)),dtype=np.float32)
masked = (padded_labels*mask).sum()/mask.sum()
np.testing.assert_allclose(masked,last.mean(),atol=1e-7)
assert not np.isclose(padded_labels.mean(),last.mean())

# Reference practice: Fingerprint meaning, not just a row count
same=dataset_digest(features.copy(),labels.copy())
reversed_digest=dataset_digest(features[::-1],labels[::-1])
changed_labels=labels.copy();changed_labels[0]+=0.1
changed_digest=dataset_digest(features,changed_labels)
assert same==fingerprint
assert reversed_digest!=fingerprint and changed_digest!=fingerprint
print('Copy preserves fingerprint; reordered source and edited labels invalidate it.')

# Reference practice: Catch a feature/label shuffle mismatch
batch = batches[0]
wrong_labels = batch["y"][::-1]
assert not np.allclose(wrong_labels,2*batch["x"]-1)
fixed = labels[batch["id"]]
np.testing.assert_allclose(fixed,2*batch["x"]-1,atol=1e-7)
changed_labels = labels.copy(); changed_labels[0]+=1
assert dataset_digest(features,changed_labels)!=fingerprint
print("PASS: recovery-01")
