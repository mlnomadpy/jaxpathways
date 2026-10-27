# Contrastive learning: views, positives and negatives

Phase 17: Self-supervised pretraining: masked and contrastive learning · about 105 minutes · CPU

This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt. Use the execution matrix to review device selection and data ownership before adapting a separate target program.

**Before this lesson:** [Masked image modeling: reconstruct missing patches](https://www.tahabouhsine.com/jaxpathways/lesson-pretraining-02.html).

## What you will be able to do

- Define the positive pair before augmenting
- Normalize embeddings and introduce temperature
- Verify the changed-condition exercise and explain the limits of this fixture.
- Evaluate a frozen contrastive encoder on source-group-held-out retrieval with multiple positives, explicit false-negative exclusion, and an independent ranking oracle.

## The problem

Two transformed views belong to the same source example. Can an encoder place them close together without mapping every input to the same point? We will inspect exactly which pairs the objective treats as positives.

## The idea

Contrastive learning compares representations using declared positive and negative relationships. Encoders and normalization produce comparable vectors, but the pairing rule decides which similarities the objective rewards.

## Define positive pairs before building the score matrix

Contrastive learning turns paired views into a retrieval task: a row asks which right-hand view matches a left-hand view, and the reverse direction is a column. The diagram’s positive mask should follow source identity, not assume that an array diagonal is always correct. If two candidates share a valid meaning, marking one as a negative creates a false negative.

Temperature divides cosine similarities before softmax. It changes how sharply probability favors the top candidates, not the original cosine values. Read the loss and retrieval ranking together: a small loss on a tiny fixture is not broad semantic evidence.

### Pause and reason

Can a high off-diagonal similarity always be called an error?

<details><summary>Compare your reasoning</summary>

No. It may be a false negative or another valid positive. Interpret scores using the declared semantic pairing contract, not index position alone.

</details>

## Before coding: make the retrieval problem explicit

Write four source IDs on cards and make two views of each source. For the first left-hand card, its positive is the right-hand card with the same source ID. Repeat for every row, then reverse the lookup for every column. If the right-hand cards are reordered, the correct positives move with their IDs; a diagonal target left in place now teaches the wrong pairing.

The fixture uses short numeric vectors passed through one shared linear encoder. CLIP uses image and text towers, but the pairing and retrieval bookkeeping applies to both. This exercise is about that bookkeeping, not a cross-modal model.

## Define the positive pair before augmenting

Keep each source ID attached through view generation. An augmentation is safe only if it preserves the relation you want the embedding to represent. Here small fixed perturbations preserve four specified directions. If an augmentation changes the source’s meaning, the resulting pair is a false positive and the loss will train the wrong invariance.

## Normalize embeddings and introduce temperature

First normalize each embedding to unit length. Their dot product is then cosine similarity. Divide each score by positive temperature $\tau$ before the categorical comparison. Lower temperature makes the probability distribution sharper, increasing the penalty for a confident wrong match; it does not make an incorrect pair correct. A norm floor prevents division instability, but identical zero vectors still describe a collapsed representation.

$$
L=-\frac{1}{2N}\sum_i\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]
$$

## Calculate a two-pair objective by hand

For two orthogonal unit vectors paired with themselves, the similarity matrix has diagonal values 1 and off-diagonal values 0. The correct candidate probability is $e^{1/\tau}/(e^{1/\tau}+1)$. Both directions give the same loss in this symmetric example. Lowering temperature lowers this particular correct-pair loss while preserving the ranking.

Now swap the right-hand views but keep diagonal labels. The true positives are off diagonal, so optimization is instructed to prefer the wrong cards. The recorded incorrect-mapping loss is $7.49$, much higher than the correctly trained fixture’s final loss of $0.0136$. This is why source IDs and target mapping must travel together.

$$
L=\log\left(1+e^{-1/\tau}\right)
$$

## Follow one row of the derivative

For one row, cross-entropy pushes probability toward its positive and away from competing candidates. The column loss contributes the reverse-direction signal, and both pass through unit normalization into the encoder. The recorded finite-difference estimates (about $-1.3803$) closely match autodiff (about $-1.38032$), checking that gradient path on a model weight.

When every embedding collapses to the same vector, all four choices are equally likely and loss is $\log 4\approx1.3863$. That is the recorded collapsed baseline, and it helps interpret the trained loss; it does not show all collapsed states can escape under every optimizer.

## Check both directions independently

Compute row and column objectives independently. Jointly permuting both view lists should preserve the task; permuting only one list should change the pair mapping unless you also update the positive relation. The code checks both directions against a stable host calculation, so a single-direction bug cannot hide behind a plausible total.

## Understand collapse and false negatives

Diagonal-only supervision treats every off-diagonal candidate as negative. If duplicate sources share a valid label, that creates a false negative: the model is penalized for placing a relevant candidate nearby. In the recorded duplicate example, diagonal-only supervision still incurs a penalty of $0.6931$. A multi-positive objective can count every valid match in the positive set.

## Separate pretraining evidence from downstream quality

On the four training pairs, loss falls from $0.5642$ to $0.0136$, and all four nearest pairs are correct. This confirms that the model learned the fixture’s pairings. The separate held-out synthetic handoff reports multi-positive loss $0.0162$, Recall@1 of 1.0 and mAP of 1.0. These perfect rankings come from a tiny synthetic class-group task with an ID-set oracle; they are not a real-world semantic benchmark or a linear-probe result.

## Design an evaluation that is not just training-pair lookup

For a meaningful next evaluation, freeze source-group splits before augmentation, fit any probe only on training groups, and evaluate retrieval on unseen groups with a fixed candidate pool. Define multiple-positive judgments and tie handling in advance. Compare random, frozen pretrained and adapted encoders under the same protocol, and inspect both a correct retrieval and a plausible false negative.

The cited embedding pilot’s phase-three floors failed. Keep that project result separate from the synthetic retrieval mechanics here; a perfect score on this fixture does not reverse the pilot evidence.

## Route the bounded experiment into a frozen transfer check

The bounded CPU handoff saves the projection matrix, view contract, temperature, positive-relation policy, group split and parameter identity, then reloads them for held-out groups g4–g9. It computes multi-positive loss and ranking metrics, and checks those metrics against a source-ID oracle and a relation-matrix path. The run reports loss $0.0162$, Recall@1 $1.0$ and mAP $1.0$ on synthetic groups. Use it to audit identity and metric plumbing, not to claim semantic transfer.

The pilot’s source/query split and ranking evidence are in the cited project files, and its phase-three floors failed. This exercise preserves those measured limits.

## 1. Define both retrieval directions

Create main.py in your activated course environment. Paste this block, then run python main.py; function definitions alone print nothing.

```python
# Step 1 — 1. Define both retrieval directions
import jax
import jax.numpy as jnp
import numpy as np

def paired_contrastive(left, right, temperature=0.2):
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-6)
    scores = left @ right.T / temperature
    return -0.5 * (
        jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1)))
        + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0)))
    )
```

Row and column log-softmax share scores but normalize different candidate axes. Draw which source each diagonal entry represents.

## 2. Construct paired views and an encoder

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
# Step 2 — 2. Construct paired views and an encoder: The fixed perturbations preserve source identity.
left = jnp.array(
    [[1.0, 0.0, 0.2], [0.0, 1.0, -0.2], [-1.0, 0.0, 0.1], [0.0, -1.0, -0.1]]
)
right = left + jnp.array(
    [[0.02, -0.01, 0.0], [-0.01, 0.02, 0.0], [0.01, 0.01, 0.0], [-0.02, -0.01, 0.0]]
)
w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
history = []
step = jax.jit(
    jax.value_and_grad(lambda w: paired_contrastive(left @ w, right @ w, 0.2))
)
```

The fixed perturbations preserve source identity. Before training, list the four target column indices and predict the consequence of a one-sided permutation.

## 3. Train, then inspect identity and symmetry

Append this block to main.py and run python main.py again. Keep the earlier blocks above it.

```python
# Step 3 — 3. Train, then inspect identity and symmetry: The host calculation checks both denominators.
for _ in range(60):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.03 * g

assert history[-1] < history[0]
zi = left @ w
zt = right @ w
zi = zi / jnp.linalg.norm(zi, axis=1, keepdims=True)
zt = zt / jnp.linalg.norm(zt, axis=1, keepdims=True)
scores = zi @ zt.T / 0.2
# Transfer the device result to the host for an independent NumPy comparison.
host = np.asarray(scores, dtype=np.float64)

def host_ce(a):
    return np.mean(
        np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a)
    )

np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w, 0.2),
    0.5 * (host_ce(host) + host_ce(host.T)),
    atol=1e-6,
)
assert np.array_equal(np.argmax(host, axis=1), np.arange(4))
permutation = jnp.array([2, 0, 3, 1])
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w),
    paired_contrastive((left @ w)[permutation], (right @ w)[permutation]),
    atol=1e-6,
)
print(
    'Paired contrastive initial/final:',
    history[0],
    history[-1],
    '; all four nearest pairs correct',
)
```

The host calculation checks both denominators. Joint permutation preserves the problem; the later one-sided permutation deliberately changes it.

## 4. Save the encoder, change one condition, and probe it

Run the preceding build cells first, then continue with this saved-encoder handoff. Run this block in a fresh Python process. It starts by recreating this lesson’s canonical synthetic fixture, then saves/reloads it, changes one declared condition, and runs the transfer check. The duplicated setup is intentional because generated scripts place staged blocks before the main example.

```python
# Held-out multi-positive retrieval + saved identity + independent ranking oracle
import hashlib, json, tempfile
from pathlib import Path

def contrastive_digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
# Held-out fixture: two independent source groups for each of three semantic classes.
# Query/document rows share class relevance, not row identity; there are multiple positives.
held_left = jnp.array([[1.,0.,0.1],[.9,.1,0.],[0.,1.,.1],[.1,.9,0.],[-1.,0.,.1],[-.9,.1,0.]])
held_right = jnp.array([[.95,.05,.1],[.85,.15,0.],[.05,.95,.1],[.15,.85,0.],[-.95,.05,.1],[-.85,.15,0.]])
held_labels = np.array([0,0,1,1,2,2])
relation = jnp.asarray(held_labels[:,None] == held_labels[None,:])
# Same-class candidates are positives; they cannot also be denominator negatives.
def multi_positive_loss(query, doc, positive, temperature=0.2):
    q = query / jnp.maximum(jnp.linalg.norm(query,axis=-1,keepdims=True),1e-6)
    d = doc / jnp.maximum(jnp.linalg.norm(doc,axis=-1,keepdims=True),1e-6)
    scores = q @ d.T / temperature
    logp = jax.nn.log_softmax(scores,axis=1)
    return -jnp.mean(jax.scipy.special.logsumexp(jnp.where(positive,logp,-jnp.inf),axis=1))
held_q, held_d = held_left @ w, held_right @ w
held_loss = float(multi_positive_loss(held_q,held_d,relation))

train_group_ids, heldout_group_ids = ['g0','g1','g2','g3'], ['g4','g5','g6','g7','g8','g9']
assert set(train_group_ids).isdisjoint(heldout_group_ids)
processor_id = 'lesson-three-float-view-v1'
source_split_id = 'synthetic-class-groups-v1:g0-g3/g4-g9'
weights = np.asarray(w)
manifest = {'model_id':'contrastive-linear-encoder-v1','processor_id':processor_id,
            'source_split_id':source_split_id,'objective':'multi-positive-InfoNCE',
            'temperature':0.2,'positive_policy':'same-class rows positive; never negative',
            'fit_group_ids':train_group_ids,'heldout_group_ids':['g4','g5','g6','g7','g8','g9'],'initialization_id':'lesson-fixed-projection',
            'parameter_sha256':contrastive_digest(weights)}
output_dir = Path(tempfile.mkdtemp(prefix='jaxpathways-contrastive-transfer-'))
artifact = output_dir / 'encoder.npz'

np.savez(artifact, projection=weights, manifest=np.array(json.dumps(manifest,sort_keys=True)))
with np.load(artifact,allow_pickle=False) as saved:
    loaded = saved['projection'].copy(); loaded_manifest = json.loads(str(saved['manifest']))
assert loaded_manifest == manifest and contrastive_digest(loaded) == manifest['parameter_sha256']
checkpoint_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
identity_path=output_dir/'identity.json'
identity={**manifest,'checkpoint_sha256':checkpoint_hash}
identity_path.write_text(json.dumps(identity,sort_keys=True))
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==identity['checkpoint_sha256']
q = np.asarray(held_left) @ loaded; doc = np.asarray(held_right) @ loaded
q /= np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-6)
doc /= np.maximum(np.linalg.norm(doc,axis=1,keepdims=True),1e-6)
scores = q @ doc.T
# Independent oracle: explicit stable sort, relevant-ID set, Recall@1 and AP.
ranking = np.argsort(-scores,axis=1,kind='stable')
oracle_recall1=[]; oracle_ap=[]
for i, order in enumerate(ranking):
    relevant=set(np.flatnonzero(held_labels == held_labels[i]).tolist())
    hits=[int(j in relevant) for j in order]
    oracle_recall1.append(float(any(hits[:1])))
    precision=[sum(hits[:k+1])/(k+1) for k,h in enumerate(hits) if h]
    oracle_ap.append(float(np.mean(precision)))
# Second metric path: relation-matrix ranks and vectorized AP, independent of the ID-set loop above.
ranked_positive = np.take_along_axis(np.asarray(relation), ranking, axis=1)
metric_recall1 = float(np.mean(ranked_positive[:,0]))
hit_count = np.cumsum(ranked_positive,axis=1)
precision_at_rank = hit_count / np.arange(1,ranked_positive.shape[1]+1)[None,:]
metric_ap = float(np.mean(np.sum(precision_at_rank*ranked_positive,axis=1) / ranked_positive.sum(axis=1)))
oracle_recall, oracle_map = float(np.mean(oracle_recall1)), float(np.mean(oracle_ap))
assert all(len(set(np.flatnonzero(held_labels == y))) == 2 for y in held_labels)
np.testing.assert_allclose(metric_recall1,oracle_recall,atol=0)
np.testing.assert_allclose(metric_ap,oracle_map,atol=1e-12)
print('Held-out multi-positive loss / Recall@1 / mAP; matrix path = ID-set oracle:', held_loss, metric_recall1, metric_ap, oracle_recall, oracle_map)
print('Artifact path:', artifact)
print('Identity sidecar:', identity_path)
print('Saved identity:', manifest['model_id'], manifest['processor_id'], manifest['source_split_id'])
```

The relation matrix has multiple positives per query. Same-class false negatives are removed from the denominator and retained in the positive numerator. Retrieval is ranked independently from the loss code so a loss implementation bug cannot certify its own metric.

## Run the example

```python
# Step 1 — 1. Define both retrieval directions
import jax
import jax.numpy as jnp
import numpy as np

def paired_contrastive(left, right, temperature=0.2):
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-6)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-6)
    scores = left @ right.T / temperature
    return -0.5 * (
        jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=1)))
        + jnp.mean(jnp.diag(jax.nn.log_softmax(scores, axis=0)))
    )

# Step 2 — 2. Construct paired views and an encoder: The fixed perturbations preserve source identity.
left = jnp.array(
    [[1.0, 0.0, 0.2], [0.0, 1.0, -0.2], [-1.0, 0.0, 0.1], [0.0, -1.0, -0.1]]
)
right = left + jnp.array(
    [[0.02, -0.01, 0.0], [-0.01, 0.02, 0.0], [0.01, 0.01, 0.0], [-0.02, -0.01, 0.0]]
)
w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
history = []
step = jax.jit(
    jax.value_and_grad(lambda w: paired_contrastive(left @ w, right @ w, 0.2))
)

# Step 3 — 3. Train, then inspect identity and symmetry: The host calculation checks both denominators.
for _ in range(60):
    value, g = step(w)
    history.append(float(value))
    w = w - 0.03 * g

assert history[-1] < history[0]
zi = left @ w
zt = right @ w
zi = zi / jnp.linalg.norm(zi, axis=1, keepdims=True)
zt = zt / jnp.linalg.norm(zt, axis=1, keepdims=True)
scores = zi @ zt.T / 0.2
# Transfer the device result to the host for an independent NumPy comparison.
host = np.asarray(scores, dtype=np.float64)

def host_ce(a):
    return np.mean(
        np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - np.diag(a)
    )

np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w, 0.2),
    0.5 * (host_ce(host) + host_ce(host.T)),
    atol=1e-6,
)
assert np.array_equal(np.argmax(host, axis=1), np.arange(4))
permutation = jnp.array([2, 0, 3, 1])
np.testing.assert_allclose(
    paired_contrastive(left @ w, right @ w),
    paired_contrastive((left @ w)[permutation], (right @ w)[permutation]),
    atol=1e-6,
)
print(
    'Paired contrastive initial/final:',
    history[0],
    history[-1],
    '; all four nearest pairs correct',
)
```

Expected: Training loss decreases and all four paired views are nearest neighbors.

## Contrastive learning: views, positives and negatives — recorded experiment

**Predict:** Predict where the largest similarity should appear in each row. What would a one-sided permutation do?

![Contrastive learning: views, positives and negatives — recorded experiment](../../phases/17-pretraining/03-contrastive/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The line is symmetric cross-view training loss in nats before each update. It falls from about $0.564$ to $0.014$. Lower values mean that the diagonal pair receives more probability among these four candidates, not that semantic accuracy is that number.

The second panel shows cosine similarity before division by temperature. A row is one left-view source and a column one right-view source. The row’s largest value lies at its matching ID on the diagonal; opposite directions can have negative similarity. Color encodes similarity, not a probability. Read both rows and columns because the loss trains both retrieval directions. A joint reorder moves the visible pattern without changing pair quality; an untracked one-sided reorder changes the labels.

### Connect it to the computation

The shared encoder starts with poorly separated directions and learns an embedding where each view retrieves its paired source. The duplicate-pair exercise changes the denominator and reveals why a loss cannot be compared blindly across batch construction policies.

```python
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'symmetric contrastive loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

# Transfer the device result to the host for an independent NumPy comparison.
extra_panel={'kind':'heatmap','values':np.asarray(zi@zt.T).tolist(),'rows':['left 0','left 1','left 2','left 3'],'columns':['right 0','right 1','right 2','right 3'],'unit':'cosine similarity','diverging':True,'xlabel':'right-view source ID','ylabel':'left-view source ID','title':'Final cross-view similarities before temperature scaling'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}
```

## Recorded reference execution

CPU run: 2026-10-27T16:11:43.633161+00:00. JAX 0.9.2.

```text
Paired contrastive initial/final: 0.564157247543335 0.013556718826293945 ; all four nearest pairs correct
Held-out multi-positive loss / Recall@1 / mAP; matrix path = ID-set oracle: 0.016225090250372887 1.0 1.0 1.0 1.0
Artifact path: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jaxpathways-contrastive-transfer-h185trru/encoder.npz
Identity sidecar: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jaxpathways-contrastive-transfer-h185trru/identity.json
Saved identity: contrastive-linear-encoder-v1 lesson-three-float-view-v1 synthetic-class-groups-v1:g0-g3/g4-g9
Paired contrastive initial/final: 0.564157247543335 0.013556718826293945 ; all four nearest pairs correct
Collapsed baseline: 1.3862943649291992
Temperature/loss: 0.2 0.006715348921716213
Temperature/loss: 1.0 0.31326162815093994
Temperature/loss: 2.0 0.4740769863128662
Held-out multi-positive loss / Recall@1 / mAP; matrix path = ID-set oracle: 0.016225090250372887 1.0 1.0 1.0 1.0
Artifact path: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jaxpathways-contrastive-transfer-kqtkb2kd/encoder.npz
Identity sidecar: /var/folders/mp/_mq7srqx2y10p5nhjz9m7hj40000gn/T/jaxpathways-contrastive-transfer-kqtkb2kd/identity.json
Saved identity: contrastive-linear-encoder-v1 lesson-three-float-view-v1 synthetic-class-groups-v1:g0-g3/g4-g9
Incorrect pair mapping loss: 7.493563652038574
Duplicated diagonal-only penalty: 0.6931472215801477
Step / finite difference / autodiff: 0.001 -1.3803242444992065 -1.3803226947784424
Step / finite difference / autodiff: 0.0005 -1.3802646398544312 -1.3803226947784424
PASS: pretraining-03

```

## Measure the collapsed baseline

**Predict before running:** What loss should identical embeddings achieve for four candidate pairs?

```python
# Experiment — Measure the collapsed baseline: Equal similarity gives each candidate probability one quarter.
collapsed = jnp.ones((4, 2))
collapse_loss = float(paired_contrastive(collapsed, collapsed))
np.testing.assert_allclose(collapse_loss, np.log(4), atol=1e-6)
print('Collapsed baseline:', collapse_loss)
```

**Expected:** The collapsed loss is log(4), approximately 1.386.

Equal similarity gives each candidate probability one quarter. This baseline catches objectives that reward only attraction and forget competing candidates.

## Separate temperature from ranking

**Predict before running:** For two orthogonal matched pairs, does lowering temperature change top-1 retrieval or only confidence?

```python
# Experiment — Separate temperature from ranking
orthogonal = jnp.eye(2)
for tau in [0.2, 1.0, 2.0]:
    observed = float(paired_contrastive(orthogonal, orthogonal, tau))
    np.testing.assert_allclose(observed, np.logaddexp(0.0, -1.0 / tau), atol=1e-6)
    # Transfer the device result to the host for an independent NumPy comparison.
    np.testing.assert_array_equal(
        np.argmax(np.asarray(orthogonal @ orthogonal.T) / tau, axis=1), [0, 1]
    )
    print('Temperature/loss:', tau, observed)
```

**Expected:** All three temperatures keep the same ranking while their losses differ.

The independent binary-softmax expression checks scale handling and demonstrates why loss comparisons require a fixed temperature and pool.

## Reload the frozen encoder and audit held-out multi-positive rankings

**Predict before running:** Do same-class held-out candidates count as negatives, and does an independent stable ranking produce the reported Recall@1 and mAP?

```python
# Held-out multi-positive retrieval + saved identity + independent ranking oracle
import hashlib, json, tempfile
from pathlib import Path

def contrastive_digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
# Held-out fixture: two independent source groups for each of three semantic classes.
# Query/document rows share class relevance, not row identity; there are multiple positives.
held_left = jnp.array([[1.,0.,0.1],[.9,.1,0.],[0.,1.,.1],[.1,.9,0.],[-1.,0.,.1],[-.9,.1,0.]])
held_right = jnp.array([[.95,.05,.1],[.85,.15,0.],[.05,.95,.1],[.15,.85,0.],[-.95,.05,.1],[-.85,.15,0.]])
held_labels = np.array([0,0,1,1,2,2])
relation = jnp.asarray(held_labels[:,None] == held_labels[None,:])
# Same-class candidates are positives; they cannot also be denominator negatives.
def multi_positive_loss(query, doc, positive, temperature=0.2):
    q = query / jnp.maximum(jnp.linalg.norm(query,axis=-1,keepdims=True),1e-6)
    d = doc / jnp.maximum(jnp.linalg.norm(doc,axis=-1,keepdims=True),1e-6)
    scores = q @ d.T / temperature
    logp = jax.nn.log_softmax(scores,axis=1)
    return -jnp.mean(jax.scipy.special.logsumexp(jnp.where(positive,logp,-jnp.inf),axis=1))
held_q, held_d = held_left @ w, held_right @ w
held_loss = float(multi_positive_loss(held_q,held_d,relation))

train_group_ids, heldout_group_ids = ['g0','g1','g2','g3'], ['g4','g5','g6','g7','g8','g9']
assert set(train_group_ids).isdisjoint(heldout_group_ids)
processor_id = 'lesson-three-float-view-v1'
source_split_id = 'synthetic-class-groups-v1:g0-g3/g4-g9'
weights = np.asarray(w)
manifest = {'model_id':'contrastive-linear-encoder-v1','processor_id':processor_id,
            'source_split_id':source_split_id,'objective':'multi-positive-InfoNCE',
            'temperature':0.2,'positive_policy':'same-class rows positive; never negative',
            'fit_group_ids':train_group_ids,'heldout_group_ids':['g4','g5','g6','g7','g8','g9'],'initialization_id':'lesson-fixed-projection',
            'parameter_sha256':contrastive_digest(weights)}
output_dir = Path(tempfile.mkdtemp(prefix='jaxpathways-contrastive-transfer-'))
artifact = output_dir / 'encoder.npz'

np.savez(artifact, projection=weights, manifest=np.array(json.dumps(manifest,sort_keys=True)))
with np.load(artifact,allow_pickle=False) as saved:
    loaded = saved['projection'].copy(); loaded_manifest = json.loads(str(saved['manifest']))
assert loaded_manifest == manifest and contrastive_digest(loaded) == manifest['parameter_sha256']
checkpoint_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
identity_path=output_dir/'identity.json'
identity={**manifest,'checkpoint_sha256':checkpoint_hash}
identity_path.write_text(json.dumps(identity,sort_keys=True))
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==identity['checkpoint_sha256']
q = np.asarray(held_left) @ loaded; doc = np.asarray(held_right) @ loaded
q /= np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-6)
doc /= np.maximum(np.linalg.norm(doc,axis=1,keepdims=True),1e-6)
scores = q @ doc.T
# Independent oracle: explicit stable sort, relevant-ID set, Recall@1 and AP.
ranking = np.argsort(-scores,axis=1,kind='stable')
oracle_recall1=[]; oracle_ap=[]
for i, order in enumerate(ranking):
    relevant=set(np.flatnonzero(held_labels == held_labels[i]).tolist())
    hits=[int(j in relevant) for j in order]
    oracle_recall1.append(float(any(hits[:1])))
    precision=[sum(hits[:k+1])/(k+1) for k,h in enumerate(hits) if h]
    oracle_ap.append(float(np.mean(precision)))
# Second metric path: relation-matrix ranks and vectorized AP, independent of the ID-set loop above.
ranked_positive = np.take_along_axis(np.asarray(relation), ranking, axis=1)
metric_recall1 = float(np.mean(ranked_positive[:,0]))
hit_count = np.cumsum(ranked_positive,axis=1)
precision_at_rank = hit_count / np.arange(1,ranked_positive.shape[1]+1)[None,:]
metric_ap = float(np.mean(np.sum(precision_at_rank*ranked_positive,axis=1) / ranked_positive.sum(axis=1)))
oracle_recall, oracle_map = float(np.mean(oracle_recall1)), float(np.mean(oracle_ap))
assert all(len(set(np.flatnonzero(held_labels == y))) == 2 for y in held_labels)
np.testing.assert_allclose(metric_recall1,oracle_recall,atol=0)
np.testing.assert_allclose(metric_ap,oracle_map,atol=1e-12)
print('Held-out multi-positive loss / Recall@1 / mAP; matrix path = ID-set oracle:', held_loss, metric_recall1, metric_ap, oracle_recall, oracle_map)
print('Artifact path:', artifact)
print('Identity sidecar:', identity_path)
print('Saved identity:', manifest['model_id'], manifest['processor_id'], manifest['source_split_id'])
```

**Expected:** The saved identity round-trips; held-out queries have multiple positives; same-class documents form the positive numerator; stable NumPy ranking returns Recall@1 and mAP.

The oracle checks metric computation, not retrieval quality: six synthetic rows cannot support benchmark claims. A production route needs source-component-disjoint held-out groups, all relevant IDs, qrels, complete candidate rankings and a declared baseline. The pilot report preserves its actual failed phase-3 floors.

## Make it yours

Permute only the right-hand embeddings after training and demonstrate the effect of broken pair identity.

### How to write this exercise — Step-by-step recipe & starter scaffold

**Inputs:**
- Left and right embeddings, a trained projection, and pair IDs.

**Outputs:**
- The original and right-permuted contrastive losses with a pair-identity diagnosis.

**Invariants to check:**
- Only the intended paired entries are positives; a permutation changes identity without changing vector values.

**Hints:**
- Apply the same projection before comparing both arrangements.
- Keep the right-side permutation explicit.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def compare_pairing_losses(left, right, projection, permutation):
    # TODO: measure the objective before and after permuting right-side pair identity.
    raise NotImplementedError
```

<details><summary>Reference solution</summary>

```python
# Exercise solution
wrong_pair_loss = float(paired_contrastive(left @ w, (right @ w)[permutation]))
assert wrong_pair_loss > float(paired_contrastive(left @ w, right @ w))
print('Incorrect pair mapping loss:', wrong_pair_loss)
```

</details>

## Duplicate a source and inspect the objective

**Transfer**

Duplicate all four pairs while keeping diagonal-only positives. Calculate the change in loss and explain the false-negative effect.

<details><summary>Hint</summary>

Each original candidate appears twice, but only one copy is labeled positive.

</details>

### How to write: Duplicate a source and inspect the objective — Step-by-step recipe & starter scaffold

**Inputs:**
- Four paired embeddings and a duplicated copy of each pair.

**Outputs:**
- Original and duplicated diagonal-positive losses and their difference.

**Invariants to check:**
- Duplicating candidates while retaining one diagonal positive changes the negative set.

**Hints:**
- Duplicate both sides in the same order.
- Do not relabel the extra copies as positives.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def inspect_duplicate_pair_effect(left, right, projection, temperature):
    # TODO: compare diagonal-positive losses before and after duplicating the batch.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Duplicate a source and inspect the objective (Transfer)
duplicated = float(
    paired_contrastive(jnp.tile(left @ w, (2, 1)), jnp.tile(right @ w, (2, 1)))
)
original = float(paired_contrastive(left @ w, right @ w))
np.testing.assert_allclose(duplicated - original, np.log(2), atol=1e-5)
print('Duplicated diagonal-only penalty:', duplicated - original)
```

The extra log(2) comes from duplicating each denominator candidate without adding the second copy to the positive set. Use source/semantic identity to define multi-positive supervision when appropriate.

</details>

## Check the encoder gradient through normalization

**Challenge**

Perturb one encoder weight at the initial, noncollapsed matrix and compare centered differences with autodiff. Repeat with a second finite-difference step.

<details><summary>Hint</summary>

Use a fresh matrix; a gradient near a trained optimum may be too small to diagnose accurately.

</details>

### How to write: Check the encoder gradient through normalization — Step-by-step recipe & starter scaffold

**Inputs:**
- A noncollapsed encoder weight matrix, fixed paired inputs, and two finite-difference step sizes.

**Outputs:**
- Autodiff and centered-difference estimates for one encoder weight.

**Invariants to check:**
- The tested parameter is away from a collapsed representation and both perturbations are symmetric.

**Hints:**
- Differentiate the full normalized-pair objective.
- Report errors for each step size; finite differences have truncation and roundoff error.

**Starter code scaffold (fill in the TODOs):**

```python
import jax.numpy as jnp
import numpy as np

def check_encoder_gradient(weights, left, right, steps):
    # TODO: compare one autodiff gradient with centered differences at each requested step.
    raise NotImplementedError
```

<details><summary>Reference solution and reasoning</summary>

```python
# Check the encoder gradient through normalization (Challenge): This checks the complete input-to-normalized-similarity chain.
audit_w = jnp.array([[0.2, 0.1], [0.1, 0.1], [0.02, -0.01]])
objective = lambda weights: paired_contrastive(left @ weights, right @ weights, 0.2)
auto = float(jax.grad(objective)(audit_w)[0, 0])
for epsilon in [1e-3, 5e-4]:
    delta = jnp.zeros_like(audit_w).at[0, 0].set(epsilon)
    estimate = float(
        (objective(audit_w + delta) - objective(audit_w - delta)) / (2 * epsilon)
    )
    np.testing.assert_allclose(estimate, auto, rtol=3e-3, atol=1e-3)
    print('Step / finite difference / autodiff:', epsilon, estimate, auto)
```

This checks the complete input-to-normalized-similarity chain. Trying two step sizes helps distinguish truncation error from a broken derivative.

</details>

## Check your understanding

What does successful retrieval on the training pairs establish?

1. The encoder solves those pairs under that pairing and augmentation contract.
2. Lowering the temperature parameter $\tau$ improves the nearest-neighbor ranking order of the cosine matrix.
3. Diagonal-only supervision automatically merges duplicate semantic examples into a single positive target.

<details><summary>Answer and explanation</summary>

**1. The encoder solves those pairs under that pairing and augmentation contract. (correct)**

Training-pair retrieval shows that this objective and implementation can fit the observed pairs. It does not establish held-out semantic retrieval, robustness, or the absence of duplicate-positive problems. Keep the evaluation candidate pool and split identities separate.

**2. Lowering the temperature parameter $\tau$ improves the nearest-neighbor ranking order of the cosine matrix.**

Temperature rescales logits but does not change a fixed embedding’s cosine ranking. Training effects require a separate comparison.

**3. Diagonal-only supervision automatically merges duplicate semantic examples into a single positive target.**

A diagonal label still penalizes duplicate positives in other columns. Use a multi-positive objective or data policy when the task requires it.

</details>

## Diagnose the result

If training stalls near the collapsed baseline, inspect norms and pairing before reducing temperature. Unexpected loss jumps after batching changes can reflect false negatives or a different candidate set.

## Carry forward

- Track source identity explicitly across both view pipelines: symmetric contrastive loss normalizes row and column retrieval probabilities over the candidate pool, and a collapsed representation sits at $\log N$.
- Keep temperature $\tau$ and candidate-pool size fixed when comparing contrastive losses, and evaluate frozen representations on held-out source groups rather than training-pair lookup.
- For contrastive transfer, define source groups and relevance before calculating loss. Multi-positive training and held-out retrieval need separate metrics; a low training loss or correct paired batch does not qualify retrieval.

## Keep your evidence

Keep pair/temperature/false-negative checks, held-out source-group membership, saved encoder identity, multiple-positive Recall@k and mAP calculations against an independent ranking oracle.

Save your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.

## Primary references

- [SimCLR: views and representations](https://arxiv.org/abs/2002.05709)
- [CLIP: symmetric paired contrastive training](https://arxiv.org/abs/2103.00020)
- [Saved embedding pilot: source-group recipe](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/embedding-model-300m/data/frozen_recipe.json)
- [Saved embedding pilot evidence and limits](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/embedding-model-300m/reference/README.md)
- [Saved embedding pilot: complete held-out rankings report](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/embedding-model-300m/reference/evidence/final8/pilot-report.json)

