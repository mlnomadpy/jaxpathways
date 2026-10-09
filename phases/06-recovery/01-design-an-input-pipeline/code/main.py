"""Design an input pipeline: worked experiments and reference solutions. CPU checks."""

# Prepare the data and state
# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import numpy for this computation.
import numpy as np
import hashlib
import time
# Initialize array `ids` with explicit values and shape.
ids = np.arange(11,dtype=np.int64)
# Cast or evaluate `features` in explicit floating-point precision.
features = (ids/10).astype(np.float32)
# Cast or evaluate `labels` in explicit floating-point precision.
labels = (2*features-1).astype(np.float32)
# Function `dataset_digest(x, y)` implementing this stage's computation:
def dataset_digest(x,y):
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest = hashlib.sha256()
    # Iterate over `array` to step through the computation:
    for array in (x,y):
        # Update state in place with the new values.
        digest.update(str(array.dtype).encode())
        # Update state in place with the new values.
        digest.update(str(array.shape).encode())
        # Update state in place with the new values.
        digest.update(array.tobytes())
    # Return `digest.hexdigest()` to the caller.
    return digest.hexdigest()
# Run `dataset_digest` to compute `fingerprint`.
fingerprint = dataset_digest(features,labels)
# Function `epoch_order(seed, epoch)` implementing this stage's computation:
def epoch_order(seed,epoch):
    # Return `np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(ids))` to the caller.
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(ids))

# Build the pipeline or state transition
# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
def make_batches(seed,epoch,batch_size=4,drop_last=False):
    # Guard input contract (`batch_size <= 0`) and fail fast if violated.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Run `epoch_order` to compute `order`.
    order = epoch_order(seed,epoch)
    # Iterate over `start` to step through the computation:
    for start in range(0,len(order),batch_size):
        # Evaluate `chosen` from the current inputs and state.
        chosen = order[start:start+batch_size]
        # Branch on condition `drop_last and len(chosen) < batch_size`:
        if drop_last and len(chosen) < batch_size:
            break
        # Execute the next step of the computation.
        yield {"id":ids[chosen],"x":features[chosen],"y":labels[chosen]}
# Record execution timing or profiler trace in `start_time`.
start_time = time.perf_counter()
# Evaluate `make_batches(seed=17, epoch=0)` and convert the result into Python scalar/collection `batches`.
batches = list(make_batches(seed=17,epoch=0))
# Record execution timing or profiler trace in `elapsed`.
elapsed = time.perf_counter()-start_time
# Combine or mask array elements to form `seen`.
seen = np.concatenate([b["id"] for b in batches])

# Run the comparison
# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
# Verify contract: `[len(b['id']) for b in batches] == [4, 4, 3]`.
assert [len(b["id"]) for b in batches] == [4,4,3]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(np.sort(seen),ids)
# Verify contract: `len(np.unique(seen)) == len(ids)`.
assert len(np.unique(seen)) == len(ids)
# Iterate over `batch` to step through the computation:
for batch in batches:
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
# Combine or mask array elements to form `repeated`.
repeated = np.concatenate([b["id"] for b in make_batches(17,0)])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(seen,repeated)
# Print the observed values to compare against the expected result.
print("Batch sizes:",[len(b["id"]) for b in batches])
# Print diagnostic summary of the computed outputs.
print("Example order:",seen.tolist())
# Print diagnostic summary of the computed outputs.
print("Preparation seconds (local observation only):",elapsed)
# Print diagnostic summary of the computed outputs.
print("Dataset SHA256:",fingerprint)

# Step 1 — Prepare the data and state: This setup makes the dataset and software assumptions explicit.
# Import numpy for this computation.
import numpy as np
import hashlib
import time
# Initialize array `ids` with explicit values and shape.
ids = np.arange(11,dtype=np.int64)
# Cast or evaluate `features` in explicit floating-point precision.
features = (ids/10).astype(np.float32)
# Cast or evaluate `labels` in explicit floating-point precision.
labels = (2*features-1).astype(np.float32)
# Function `dataset_digest(x, y)` implementing this stage's computation:
def dataset_digest(x,y):
    # Compute deterministic cryptographic digest `digest` for provenance verification.
    digest = hashlib.sha256()
    # Iterate over `array` to step through the computation:
    for array in (x,y):
        # Update state in place with the new values.
        digest.update(str(array.dtype).encode())
        # Update state in place with the new values.
        digest.update(str(array.shape).encode())
        # Update state in place with the new values.
        digest.update(array.tobytes())
    # Return `digest.hexdigest()` to the caller.
    return digest.hexdigest()
# Run `dataset_digest` to compute `fingerprint`.
fingerprint = dataset_digest(features,labels)
# Function `epoch_order(seed, epoch)` implementing this stage's computation:
def epoch_order(seed,epoch):
    # Return `np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(ids))` to the caller.
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(ids))

# Step 2 — Build the pipeline or state transition: The function boundaries expose which inputs determine the next...
def make_batches(seed,epoch,batch_size=4,drop_last=False):
    # Guard input contract (`batch_size <= 0`) and fail fast if violated.
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Run `epoch_order` to compute `order`.
    order = epoch_order(seed,epoch)
    # Iterate over `start` to step through the computation:
    for start in range(0,len(order),batch_size):
        # Evaluate `chosen` from the current inputs and state.
        chosen = order[start:start+batch_size]
        # Branch on condition `drop_last and len(chosen) < batch_size`:
        if drop_last and len(chosen) < batch_size:
            break
        # Execute the next step of the computation.
        yield {"id":ids[chosen],"x":features[chosen],"y":labels[chosen]}
# Record execution timing or profiler trace in `start_time`.
start_time = time.perf_counter()
# Evaluate `make_batches(seed=17, epoch=0)` and convert the result into Python scalar/collection `batches`.
batches = list(make_batches(seed=17,epoch=0))
# Record execution timing or profiler trace in `elapsed`.
elapsed = time.perf_counter()-start_time
# Combine or mask array elements to form `seen`.
seen = np.concatenate([b["id"] for b in batches])

# Step 3 — Run the comparison: The comparison checks the next behavior, not merely whether a save...
# Verify contract: `[len(b['id']) for b in batches] == [4, 4, 3]`.
assert [len(b["id"]) for b in batches] == [4,4,3]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(np.sort(seen),ids)
# Verify contract: `len(np.unique(seen)) == len(ids)`.
assert len(np.unique(seen)) == len(ids)
# Iterate over `batch` to step through the computation:
for batch in batches:
    # Verify that computed values match the expected reference within numerical tolerance.
    np.testing.assert_allclose(batch["y"],2*batch["x"]-1,atol=1e-7)
# Combine or mask array elements to form `repeated`.
repeated = np.concatenate([b["id"] for b in make_batches(17,0)])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(seen,repeated)
# Print the observed values to compare against the expected result.
print("Batch sizes:",[len(b["id"]) for b in batches])
# Print diagnostic summary of the computed outputs.
print("Example order:",seen.tolist())
# Print diagnostic summary of the computed outputs.
print("Preparation seconds (local observation only):",elapsed)
# Print diagnostic summary of the computed outputs.
print("Dataset SHA256:",fingerprint)

# Figure data experiment
# Compute figure data for: Shuffling changes order, not dataset membership
# Evaluate `visual_data` from the current inputs and state.
visual_data = {'kind': 'line', 'x': list(range(len(seen))), 'xlabel': 'position in epoch', 'ylabel': 'example ID', 'series': [{'label': 'shuffled order', 'y': seen.tolist()}], 'boundaries': [3.5, 7.5]}

# Experiment: Compare epochs without a hard-coded order
# Experiment — Compare epochs without a hard-coded order: Coverage is an invariant; ordering is a deterministic...
epoch_one = np.concatenate([b["id"] for b in make_batches(17,1)])
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_array_equal(np.sort(epoch_one),ids)
# Verify contract: `not np.array_equal(epoch_one, seen)`.
assert not np.array_equal(epoch_one,seen)
# Print the observed values to compare against the expected result.
print("Epoch one order:",epoch_one.tolist())

# Experiment: Expose dropped records
# Experiment — Expose dropped records: Logging just the number of batches hides exclusions.
dropped = list(make_batches(17,0,drop_last=True))
# Combine or mask array elements to form `dropped_ids`.
dropped_ids = np.concatenate([b["id"] for b in dropped])
# Verify contract: `len(dropped_ids) == 8`.
assert len(dropped_ids)==8
# Run `np.setdiff1d` to compute `omitted`.
omitted = np.setdiff1d(ids,dropped_ids)
# Verify contract: `len(omitted) == 3`.
assert len(omitted)==3
# Print the observed values to compare against the expected result.
print("Omitted IDs:",omitted.tolist())

# Reference solution. Try the exercise before reading this.
# Exercise solution: Pad the final three-label batch to four elements with zero.
last = batches[-1]["y"]
# Combine or mask array elements to form `padded_labels`.
padded_labels = np.pad(last,(0,4-len(last)))
# Initialize array `mask` with explicit values and shape.
mask = np.array([1]*len(last)+[0]*(4-len(last)),dtype=np.float32)
# Aggregate array values to compute `masked`.
masked = (padded_labels*mask).sum()/mask.sum()
# Reduce across the target axis to summarize ``.
np.testing.assert_allclose(masked,last.mean(),atol=1e-7)
# Verify that the numerical values match the expected reference within tolerance.
assert not np.isclose(padded_labels.mean(),last.mean())

# Reference practice: Fingerprint meaning, not just a row count
# Fingerprint meaning, not just a row count (Transfer): A sampler permutation can be reproduced against a fixed source.
same=dataset_digest(features.copy(),labels.copy())
# Run `dataset_digest` to compute `reversed_digest`.
reversed_digest=dataset_digest(features[::-1],labels[::-1])
# Run `labels.copy` to compute `changed_labels`.
# Accumulate the next contribution into `changed_labels[0]`.
changed_labels=labels.copy();changed_labels[0]+=0.1
# Run `dataset_digest` to compute `changed_digest`.
changed_digest=dataset_digest(features,changed_labels)
# Verify contract: `same == fingerprint`.
assert same==fingerprint
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert reversed_digest!=fingerprint and changed_digest!=fingerprint
# Print the observed values to compare against the expected result.
print('Copy preserves fingerprint; reordered source and edited labels invalidate it.')

# Reference practice: Catch a feature/label shuffle mismatch
# Catch a feature/label shuffle mismatch (Challenge): Wrong alignment can preserve shapes and numeric ranges.
batch = batches[0]
# Evaluate `wrong_labels` from the current inputs and state.
wrong_labels = batch["y"][::-1]
# Verify that the numerical values match the expected reference within tolerance.
assert not np.allclose(wrong_labels,2*batch["x"]-1)
# Evaluate `fixed` from the current inputs and state.
fixed = labels[batch["id"]]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(fixed,2*batch["x"]-1,atol=1e-7)
# Run `labels.copy` to compute `changed_labels`.
# Accumulate the next contribution into `changed_labels[0]`.
changed_labels = labels.copy(); changed_labels[0]+=1
# Verify contract: `dataset_digest(features, changed_labels) != fingerprint`.
assert dataset_digest(features,changed_labels)!=fingerprint
print("PASS: recovery-01")
