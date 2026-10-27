"""Contrastive learning: views, positives and negatives: worked experiments and reference solutions. CPU checks."""

# 1. Define both retrieval directions
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

# 2. Construct paired views and an encoder
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

# 3. Train, then inspect identity and symmetry
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

# 4. Save the encoder, change one condition, and probe it
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

# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'symmetric contrastive loss (nats)','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']

# Transfer the device result to the host for an independent NumPy comparison.
extra_panel={'kind':'heatmap','values':np.asarray(zi@zt.T).tolist(),'rows':['left 0','left 1','left 2','left 3'],'columns':['right 0','right 1','right 2','right 3'],'unit':'cosine similarity','diverging':True,'xlabel':'right-view source ID','ylabel':'left-view source ID','title':'Final cross-view similarities before temperature scaling'}
visual_data={"panels":[*visual_data.get("panels",[visual_data]),extra_panel]}

# Experiment: Measure the collapsed baseline
# Experiment — Measure the collapsed baseline: Equal similarity gives each candidate probability one quarter.
collapsed = jnp.ones((4, 2))
collapse_loss = float(paired_contrastive(collapsed, collapsed))
np.testing.assert_allclose(collapse_loss, np.log(4), atol=1e-6)
print('Collapsed baseline:', collapse_loss)

# Experiment: Separate temperature from ranking
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

# Experiment: Reload the frozen encoder and audit held-out multi-positive rankings
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

# Reference solution. Try the exercise before reading this.
# Exercise solution
wrong_pair_loss = float(paired_contrastive(left @ w, (right @ w)[permutation]))
assert wrong_pair_loss > float(paired_contrastive(left @ w, right @ w))
print('Incorrect pair mapping loss:', wrong_pair_loss)

# Reference practice: Duplicate a source and inspect the objective
# Duplicate a source and inspect the objective (Transfer)
duplicated = float(
    paired_contrastive(jnp.tile(left @ w, (2, 1)), jnp.tile(right @ w, (2, 1)))
)
original = float(paired_contrastive(left @ w, right @ w))
np.testing.assert_allclose(duplicated - original, np.log(2), atol=1e-5)
print('Duplicated diagonal-only penalty:', duplicated - original)

# Reference practice: Check the encoder gradient through normalization
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
print("PASS: pretraining-03")
