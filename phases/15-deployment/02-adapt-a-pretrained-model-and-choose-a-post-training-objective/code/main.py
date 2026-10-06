"""Adapt a pretrained model and choose a post-training objective: worked experiments and reference solutions. CPU checks."""

# 1. Pretrain a small model and retain its source data identity
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

def design(key,n):
    features=jax.random.normal(key,(n,2))
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
source_X=design(jax.random.key(10),256)
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
adapt_X=design(jax.random.key(11),48)
held_X=design(jax.random.key(12),512)
teacher=jnp.array([.6,1.4,-.3])
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
held_prob=jax.nn.sigmoid(held_X@teacher)
def objective(w,X,targets):
    logits=X@w
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
def train(initial,X,targets,steps=300,rate=.15):
    def step(w,_):
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        return w-rate*grad,loss
    return jax.lax.scan(step,initial,None,length=steps)
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
assert pretrain_history[-1]<pretrain_history[0]-.1
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()

# 2. Save and reload before adapting
with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'pretrained.npz'
    np.savez(checkpoint,weights=np.asarray(base))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    manifest_path=Path(directory)/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    recorded=json.loads(manifest_path.read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    with np.load(checkpoint,allow_pickle=False) as saved:
        restored=jnp.asarray(saved['weights'])
np.testing.assert_array_equal(restored,base)
frozen=np.array(restored,copy=True)
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
np.testing.assert_array_equal(restored,frozen)

# 3. Compare on a held-out criterion shared by both objectives
w_check=jnp.array([.2,-.3,.1])
host_X=np.asarray(adapt_X,dtype=np.float64)
host_w=np.asarray(w_check,dtype=np.float64)
host_p=1/(1+np.exp(-(host_X@host_w)))
for targets in (hard_targets,soft_targets):
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
metrics={}
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
print(json.dumps(metrics,indent=2))
print('checkpoint source hash:',source_hash)

import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

def design(key,n):
    features=jax.random.normal(key,(n,2))
    return jnp.concatenate([features,jnp.ones((n,1))],axis=1)
source_X=design(jax.random.key(10),256)
source_targets=jax.nn.sigmoid(source_X@jnp.array([1.2,-.8,.2]))
adapt_X=design(jax.random.key(11),48)
held_X=design(jax.random.key(12),512)
teacher=jnp.array([.6,1.4,-.3])
soft_targets=jax.nn.sigmoid(adapt_X@teacher)
hard_targets=jax.random.bernoulli(jax.random.key(13),soft_targets).astype(jnp.float32)
held_prob=jax.nn.sigmoid(held_X@teacher)
def objective(w,X,targets):
    logits=X@w
    return jnp.mean(jnp.logaddexp(0.,logits)-targets*logits)
def train(initial,X,targets,steps=300,rate=.15):
    def step(w,_):
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        return w-rate*grad,loss
    return jax.lax.scan(step,initial,None,length=steps)
base,pretrain_history=train(jnp.zeros(3),source_X,source_targets)
assert pretrain_history[-1]<pretrain_history[0]-.1
source_hash=hashlib.sha256(np.asarray(source_X).tobytes()+np.asarray(source_targets).tobytes()).hexdigest()

with tempfile.TemporaryDirectory() as directory:
    checkpoint=Path(directory)/'pretrained.npz'
    np.savez(checkpoint,weights=np.asarray(base))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest={'model':'three-parameter logistic synthetic fixture','source_data_sha256':source_hash,
              'weights_sha256':digest,'source_seed':10,'pretrain_steps':300,'learning_rate':.15,
              'objective':'binary cross entropy with synthetic source probabilities',
              'jax':jax.__version__,'dtype':'float32','optimizer_state_included':False}
    manifest_path=Path(directory)/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    recorded=json.loads(manifest_path.read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==recorded['weights_sha256']
    with np.load(checkpoint,allow_pickle=False) as saved:
        restored=jnp.asarray(saved['weights'])
np.testing.assert_array_equal(restored,base)
frozen=np.array(restored,copy=True)
sft,sft_history=train(restored,adapt_X,hard_targets,steps=200)
distilled,distill_history=train(restored,adapt_X,soft_targets,steps=200)
np.testing.assert_array_equal(restored,frozen)

w_check=jnp.array([.2,-.3,.1])
host_X=np.asarray(adapt_X,dtype=np.float64)
host_w=np.asarray(w_check,dtype=np.float64)
host_p=1/(1+np.exp(-(host_X@host_w)))
for targets in (hard_targets,soft_targets):
    expected=host_X.T@(host_p-np.asarray(targets))/len(host_X)
    np.testing.assert_allclose(jax.grad(objective)(w_check,adapt_X,targets),expected,rtol=1e-5,atol=1e-6)
metrics={}
for name,weights in [('pretrained',restored),('supervised',sft),('teacher',distilled)]:
    metrics[name]={'held_cross_entropy':float(objective(weights,held_X,held_prob)),
                   'held_brier':float(jnp.mean((jax.nn.sigmoid(held_X@weights)-held_prob)**2)),
                   'source_cross_entropy':float(objective(weights,source_X,source_targets))}
assert metrics['supervised']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
assert metrics['teacher']['held_cross_entropy']<metrics['pretrained']['held_cross_entropy']
print(json.dumps(metrics,indent=2))
print('checkpoint source hash:',source_hash)

# Figure data experiment
names=["pretrained","supervised","teacher"]
visual_data={"kind":"bar","x":[0,1,2],"labels":names,"xlabel":"model after training stage","ylabel":"mean cross-entropy (nats)","series":[{"label":"held-out target","y":[metrics[n]["held_cross_entropy"] for n in names]},{"label":"source retention","y":[metrics[n]["source_cross_entropy"] for n in names]}]}

# Experiment: Check a stable extreme logit
extreme=objective(jnp.array([1000.]),jnp.ones((1,1)),jnp.zeros(1))
assert jnp.isfinite(extreme) and jnp.allclose(extreme,1000.)

# Experiment: Reveal teacher error
wrong,_=train(restored,adapt_X,1-soft_targets,steps=200)
assert objective(wrong,held_X,held_prob)>objective(distilled,held_X,held_prob)
print("wrong-teacher held loss:",float(objective(wrong,held_X,held_prob)))

# Reference solution. Try the exercise before reading this.
unchanged,_=train(restored,adapt_X,hard_targets,steps=0)
np.testing.assert_array_equal(unchanged,restored)
short,_=train(restored,adapt_X,hard_targets,steps=10)
assert not jnp.array_equal(short,restored)
np.testing.assert_array_equal(restored,frozen)

# Reference practice: Keep source-task evidence
retention=[float(objective(w,source_X,source_targets)) for w in (restored,sft,distilled)]
assert retention[1]>retention[0] and retention[2]>retention[0]
print("source retention losses:",retention)

# Reference practice: Check the loss without autodiff
probe=np.array([-.4,.2,.3])
z=host_X@probe
expected=np.mean(np.logaddexp(0,z)-np.asarray(soft_targets)*z)
np.testing.assert_allclose(objective(jnp.asarray(probe),adapt_X,soft_targets),expected,rtol=1e-6)
print("PASS: deployment-02")
