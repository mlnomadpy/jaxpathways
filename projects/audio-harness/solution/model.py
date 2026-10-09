"""Connected CPU sound-event teaching harness. No speech recognition claim."""
# Import hashlib for this computation.
import hashlib
import json
from pathlib import Path
import time
import wave
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

# Combine or mask array elements to form `PREPROCESS`.
PREPROCESS = {"version":"audio-stft-v1","sample_rate":8000,"channels":1,"samples":1024,
              "clip_samples":1280,"window":128,"hop":64,"window_function":"symmetric Hann",
              "spectrum":"rfft magnitude squared / sum(window squared)",
              "feature":"mean over frames of log1p(power); 65 frequency bins",
              "normalization":"training mean and std; std floor 0.05"}
# Evaluate `CLASSES` from the current inputs and state.
CLASSES = ["background","low-tone","high-tone"]
# Evaluate `BATCHES` from the current inputs and state.
BATCHES = (1,4)


# Function `digest_arrays()` implementing this stage's computation:
def digest_arrays(*arrays):
    # Compute deterministic cryptographic digest `h` for provenance verification.
    h=hashlib.sha256()
    # Iterate over `item` to step through the computation:
    for item in arrays:
        # Run `np.ascontiguousarray` to compute `a`.
        a=np.ascontiguousarray(item)
        # Update state in place with the new values.
        # Update state in place with the new values.
        # Update state in place with the new values.
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    # Return `h.hexdigest()` to the caller.
    return h.hexdigest()


# Function `dataset_hash(data)` implementing this stage's computation:
def dataset_hash(data):
    # Return `digest_arrays(data['waveforms'], data['labels'], np.asarray(data['ids']), np.asarray(data['groups']), np.asarray([data['sample_rate']]))` to the caller.
    return digest_arrays(data["waveforms"],data["labels"],np.asarray(data["ids"]),
                         np.asarray(data["groups"]),np.asarray([data["sample_rate"]]))


# Function `make_dataset(seed, count, split, shift)` implementing this stage's computation:
def make_dataset(seed=41,count=96,split="train",shift=False):
    """Synthetic recording IDs and groups are disjoint by split and seed."""
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    rng=np.random.default_rng(seed)
    # Initialize array `labels` with explicit values and shape.
    labels=np.arange(count,dtype=np.int32)%3
    # Initialize array `time_axis` with explicit values and shape.
    time_axis=np.arange(PREPROCESS["clip_samples"])/PREPROCESS["sample_rate"]
    # Evaluate `waves` from the current inputs and state.
    waves=[]
    # Loop over `label` in `labels`:
    for label in labels:
        # Draw pseudorandom samples for `gain` using the explicit RNG state.
        gain=rng.uniform(.2,.7)
        # Draw pseudorandom samples for `noise` using the explicit RNG state.
        noise=rng.normal(0,.008 if not shift else .025,len(time_axis))
        # Branch on condition `label == 0`:
        if label==0:
            signal=noise
        else:
            center=(500.,1500.)[label-1]
            freq=center+rng.uniform(-35,35) if not shift else center+rng.uniform(-90,90)
            signal=gain*np.sin(2*np.pi*freq*time_axis+rng.uniform(0,2*np.pi))+noise
        # Append the current step result to `waves`.
        waves.append(signal)
    # Return `{'waveforms': np.asarray(waves, dtype=np.float32), 'labels': labels, 'ids': [f'{split}-recording-{seed}-{i:04d}' for i in range(count)], 'groups': [f'{split}-source-{seed}-{i:04d}' for i in range(count)], 'sample_rate': 8000, 'provenance': 'generated tone/background fixture; no external recordings'}` to the caller.
    return {"waveforms":np.asarray(waves,dtype=np.float32),"labels":labels,
            "ids":[f"{split}-recording-{seed}-{i:04d}" for i in range(count)],
            "groups":[f"{split}-source-{seed}-{i:04d}" for i in range(count)],
            "sample_rate":8000,"provenance":"generated tone/background fixture; no external recordings"}


# Function `validate_waveforms(waveforms, sample_rate, channels, version)` implementing this stage's computation:
def validate_waveforms(waveforms,sample_rate=8000,channels=1,version="audio-stft-v1"):
    # Convert `a` to a host NumPy array for inspection or verification.
    a=np.asarray(waveforms)
    # Guard input contract (`sample_rate != 8000 or channels != 1 or version != PREPROCESS['version']`) and fail fast if violated.
    if sample_rate!=8000 or channels!=1 or version!=PREPROCESS["version"]:
        raise ValueError("sample rate, channel count or preprocessing version mismatch")
    # Guard input contract (`a.ndim != 2 or a.shape[1] != 1024 or a.dtype.kind not in 'fiu'`) and fail fast if violated.
    if a.ndim!=2 or a.shape[1]!=1024 or a.dtype.kind not in "fiu":
        raise ValueError("waveforms must have shape (batch,1024) with numeric amplitudes")
    # Guard input contract (`a.shape[0] == 0 or not np.isfinite(a).all() or np.max(np.abs(a)) > 1.0`) and fail fast if violated.
    if a.shape[0]==0 or not np.isfinite(a).all() or np.max(np.abs(a))>1.:
        raise ValueError("require finite nonempty PCM amplitudes in [-1,1]")
    # Return `np.asarray(a, dtype=np.float32)` to the caller.
    return np.asarray(a,dtype=np.float32)


# Function `spectrum_core(waveforms)` implementing this stage's computation:
def spectrum_core(waveforms):
    # Initialize array `offsets` with explicit values and shape.
    offsets=jnp.arange(15)[:,None]*64+jnp.arange(128)[None,:]
    # Evaluate `frames` from the current inputs and state.
    frames=waveforms[:,offsets]
    # Cast or evaluate `window` in explicit floating-point precision.
    window=jnp.hanning(128).astype(jnp.float32)
    # Run `jnp.fft.rfft` to compute `transformed`.
    transformed=jnp.fft.rfft(frames*window,axis=-1)
    # Return `jnp.abs(transformed) ** 2 / jnp.sum(window ** 2)` to the caller.
    return (jnp.abs(transformed)**2)/jnp.sum(window**2)


# Define and JIT-compile `features_core(waveforms)` so XLA traces and fuses the operations:
@jax.jit
# Function `features_core(waveforms)` implementing this stage's computation:
def features_core(waveforms):
    # Return `jnp.mean(jnp.log1p(spectrum_core(waveforms)), axis=1)` to the caller.
    return jnp.mean(jnp.log1p(spectrum_core(waveforms)),axis=1)


# Function `features(waveforms, sample_rate, channels, version)` implementing this stage's computation:
def features(waveforms,sample_rate=8000,channels=1,version="audio-stft-v1"):
    # Return `features_core(jnp.asarray(validate_waveforms(waveforms, sample_rate, channels, version)))` to the caller.
    return features_core(jnp.asarray(validate_waveforms(waveforms,sample_rate,channels,version)))


# Function `fixed_windows(data)` implementing this stage's computation:
def fixed_windows(data):
    # Deterministic center crop; training augmentation has a separate path.
    return np.asarray(data["waveforms"][:,128:1152],dtype=np.float32)


# Function `objective(W, b, normalized, labels)` implementing this stage's computation:
def objective(W,b,normalized,labels):
    # Guard input contract (`normalized.ndim != 2 or labels.shape != (normalized.shape[0],) or (not jnp.issubdtype(labels.dtype, jnp.integer))`) and fail fast if violated.
    if normalized.ndim!=2 or labels.shape!=(normalized.shape[0],) or not jnp.issubdtype(labels.dtype,jnp.integer):
        raise ValueError("one integer class label per feature row required")
    # Perform matrix contraction / projection to compute `scores`.
    scores=normalized@W+b
    # Return `-jnp.mean(jax.nn.log_softmax(scores)[jnp.arange(len(labels)), labels])` to the caller.
    return -jnp.mean(jax.nn.log_softmax(scores)[jnp.arange(len(labels)),labels])


# Function `create_state(data, seed, batch_size, rate, ...)` implementing this stage's computation:
def create_state(data,seed=0,batch_size=12,rate=.04,momentum=.85):
    # Guard input contract (`data['sample_rate'] != 8000 or len(data['ids']) != len(data['labels'])`) and fail fast if violated.
    if data["sample_rate"]!=8000 or len(data["ids"])!=len(data["labels"]):
        raise ValueError("dataset contract mismatch")
    # Guard input contract (`len(data['groups']) != len(data['ids']) or len(set(data['ids'])) != len(data['ids']) or (not np.isfinite(data['waveforms']).all()) or (np.max(np.abs(data['waveforms'])) > 1) or (np.asarray(data['labels']).dtype.kind not in 'iu') or np.any((data['labels'] < 0) | (data['labels'] > 2))`) and fail fast if violated.
    if (len(data["groups"])!=len(data["ids"]) or len(set(data["ids"]))!=len(data["ids"])
        or not np.isfinite(data["waveforms"]).all() or np.max(np.abs(data["waveforms"]))>1
        or np.asarray(data["labels"]).dtype.kind not in "iu" or np.any((data["labels"]<0)|(data["labels"]>2))):
        raise ValueError("invalid recording identities, PCM values or class labels")
    # Guard input contract (`np.asarray(data['waveforms']).shape != (len(data['labels']), 1280)`) and fail fast if violated.
    if np.asarray(data["waveforms"]).shape!=(len(data["labels"]),1280):
        raise ValueError("training recording shape must be (recordings,1280)")
    # Guard input contract (`batch_size < 1 or rate <= 0 or (not 0 <= momentum < 1)`) and fail fast if violated.
    if batch_size<1 or rate<=0 or not 0<=momentum<1:
        raise ValueError("invalid optimizer or batch configuration")
    # Run `features` to compute `train_features`.
    train_features=features(fixed_windows(data))
    # Reduce across the target axis to summarize `mean`.
    mean=train_features.mean(axis=0)
    # Reduce across the target axis to summarize `std`.
    std=jnp.maximum(train_features.std(axis=0),.05)
    # Initialize explicit deterministic PRNG key `(key, init_key, order_key)`.
    key,init_key,order_key=jax.random.split(jax.random.PRNGKey(seed),3)
    # Draw pseudorandom samples for `W` using the explicit RNG state.
    W=.01*jax.random.normal(init_key,(65,3))
    # Return `{'W': W, 'b': jnp.zeros(3), 'vW': jnp.zeros_like(W), 'vb': jnp.zeros(3), 'mean': mean, 'std': std, 'key': key, 'order': jax.random.permutation(order_key, len(data['labels'])), 'step': 0, 'cursor': 0, 'epoch': 0, 'config': {'seed': seed, 'batch_size': batch_size, 'rate': rate, 'momentum': momentum}, 'dataset_sha256': dataset_hash(data)}` to the caller.
    return {"W":W,"b":jnp.zeros(3),"vW":jnp.zeros_like(W),"vb":jnp.zeros(3),
            "mean":mean,"std":std,"key":key,"order":jax.random.permutation(order_key,len(data["labels"])),
            "step":0,"cursor":0,"epoch":0,
            "config":{"seed":seed,"batch_size":batch_size,"rate":rate,"momentum":momentum},
            "dataset_sha256":dataset_hash(data)}


@jax.jit
# Function `update_arrays(W, b, vW, vb, ...)` implementing this stage's computation:
def update_arrays(W,b,vW,vb,normalized,labels,rate,momentum):
    # Evaluate both scalar loss and parameter gradients in one pass (`(loss, (gW, gb))`).
    loss,(gW,gb)=jax.value_and_grad(objective,argnums=(0,1))(W,b,normalized,labels)
    # Evaluate `vW` from the current inputs and state.
    # Evaluate `vb` from the current inputs and state.
    vW=momentum*vW+gW;vb=momentum*vb+gb
    # Return `(W - rate * vW, b - rate * vb, vW, vb, loss)` to the caller.
    return W-rate*vW,b-rate*vb,vW,vb,loss


# Function `step(state, data)` implementing this stage's computation:
def step(state,data):
    # Guard input contract (`state['dataset_sha256'] != dataset_hash(data)`) and fail fast if violated.
    if state["dataset_sha256"]!=dataset_hash(data):
        raise ValueError("training dataset identity changed")
    # Evaluate `state` and convert the result into Python scalar/collection `state`.
    state=dict(state)
    # Branch on condition `state['cursor'] == len(data['labels'])`:
    if state["cursor"]==len(data["labels"]):
        state["key"],order_key=jax.random.split(state["key"])
        state["order"]=jax.random.permutation(order_key,len(data["labels"]))
        state["cursor"]=0;state["epoch"]+=1
    # Evaluate `cursor` from the current inputs and state.
    cursor=state["cursor"]
    # Convert `indices` to a host NumPy array for inspection or verification.
    indices=np.asarray(state["order"][cursor:cursor+state["config"]["batch_size"]])
    # Split the PRNG key deterministically into independent subkeys (`(state['key'], crop_key, gain_key, noise_key)`).
    state["key"],crop_key,gain_key,noise_key=jax.random.split(state["key"],4)
    # Draw pseudorandom samples for `offsets` using the explicit RNG state.
    offsets=jax.random.randint(crop_key,(len(indices),),0,257)
    # Draw pseudorandom samples for `gains` using the explicit RNG state.
    gains=jax.random.uniform(gain_key,(len(indices),1),minval=.8,maxval=1.15)
    # Create device-backed JAX array `clips`.
    clips=jnp.asarray(data["waveforms"][indices])
    # Create evenly spaced index values in `windows`.
    windows=jnp.take_along_axis(clips,offsets[:,None]+jnp.arange(1024)[None,:],axis=1)
    # Draw pseudorandom samples for `augmented` using the explicit RNG state.
    augmented=windows*gains+.003*jax.random.normal(noise_key,windows.shape)
    # Run `features_core` to compute `feat`.
    feat=features_core(augmented)
    # Evaluate `normalized` from the current inputs and state.
    normalized=(feat-state["mean"])/state["std"]
    # Create device-backed JAX array `(state['W'], state['b'], state['vW'], state['vb'], loss)`.
    state["W"],state["b"],state["vW"],state["vb"],loss=update_arrays(
        state["W"],state["b"],state["vW"],state["vb"],normalized,
        jnp.asarray(data["labels"][indices]),state["config"]["rate"],state["config"]["momentum"])
    # Accumulate the next contribution into `state['cursor']`.
    # Accumulate the next contribution into `state['step']`.
    state["cursor"]+=len(indices);state["step"]+=1
    # Convert `trace` to a host NumPy array for inspection or verification.
    trace={"ids":[data["ids"][int(i)] for i in indices],"crop_offsets":np.asarray(offsets).tolist(),
           "gains":np.asarray(gains).ravel().tolist(),"feature_sha256":digest_arrays(np.asarray(feat)),
           "loss":float(loss),"completed_step":state["step"]}
    # Return `(state, trace)` to the caller.
    return state,trace


# Evaluate `ARRAY_FIELDS` from the current inputs and state.
ARRAY_FIELDS=("W","b","vW","vb","mean","std","key","order")


# Function `save_checkpoint(path, state)` implementing this stage's computation:
def save_checkpoint(path,state):
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    # Convert `` to a host NumPy array for inspection or verification.
    np.savez(path/"state.npz",**{k:np.asarray(state[k]) for k in ARRAY_FIELDS})
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={k:state[k] for k in ("step","cursor","epoch","config","dataset_sha256")}
    # Compute deterministic cryptographic digest `` for provenance verification.
    manifest.update(preprocessing=PREPROCESS,jax=jax.__version__,
                    state_sha256=hashlib.sha256((path/"state.npz").read_bytes()).hexdigest())
    # Read or serialize artifact data on disk (``).
    (path/"checkpoint.json").write_text(json.dumps(manifest,indent=2))


# Function `load_checkpoint(path, data, config)` implementing this stage's computation:
def load_checkpoint(path,data,config):
    # Read or serialize artifact data on disk (`path`).
    # Read or serialize artifact data on disk (`manifest`).
    path=Path(path);manifest=json.loads((path/"checkpoint.json").read_text())
    # Guard input contract (`manifest['dataset_sha256'] != dataset_hash(data) or manifest['config'] != config`) and fail fast if violated.
    if manifest["dataset_sha256"]!=dataset_hash(data) or manifest["config"]!=config:
        raise ValueError("dataset or training configuration mismatch")
    # Guard input contract (`manifest['preprocessing'] != PREPROCESS or manifest['jax'] != jax.__version__`) and fail fast if violated.
    if manifest["preprocessing"]!=PREPROCESS or manifest["jax"]!=jax.__version__:
        raise ValueError("preprocessing or runtime mismatch")
    # Guard input contract (`hashlib.sha256((path / 'state.npz').read_bytes()).hexdigest() != manifest['state_sha256']`) and fail fast if violated.
    if hashlib.sha256((path/"state.npz").read_bytes()).hexdigest()!=manifest["state_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    # Enter managed runtime/context scope for this block:
    with np.load(path/"state.npz",allow_pickle=False) as data_arrays:
        # Create device-backed JAX array `state`.
        state={k:jnp.asarray(data_arrays[k]) for k in ARRAY_FIELDS}
    # Compute deterministic cryptographic digest `` for provenance verification.
    state.update({k:manifest[k] for k in ("step","cursor","epoch","config","dataset_sha256")})
    # Return `state` to the caller.
    return state


# Function `scores(state, waveforms)` implementing this stage's computation:
def scores(state,waveforms):
    # Run `features` to compute `x`.
    x=features(waveforms)
    # Return `(x - state['mean']) / state['std'] @ state['W'] + state['b']` to the caller.
    return ((x-state["mean"])/state["std"])@state["W"]+state["b"]


# Function `evaluate(state, data, batch_size)` implementing this stage's computation:
def evaluate(state,data,batch_size=13):
    # Evaluate `count` from the current inputs and state.
    # Evaluate `loss_sum` from the current inputs and state.
    # Evaluate `correct` from the current inputs and state.
    # Allocate initialized array `confusion` with the specified shape and dtype.
    count=0;loss_sum=0.;correct=0;confusion=np.zeros((3,3),dtype=np.int64)
    # Run `fixed_windows` to compute `windows`.
    windows=fixed_windows(data)
    # Loop over `start` in `range(0, len(windows), batch_size)`:
    for start in range(0,len(windows),batch_size):
        # Evaluate `y` from the current inputs and state.
        y=data["labels"][start:start+batch_size]
        # Convert `logits` to a host NumPy array for inspection or verification.
        logits=np.asarray(scores(state,windows[start:start+batch_size]))
        # Reduce across the target axis to summarize `shifted`.
        shifted=logits-logits.max(axis=1,keepdims=True)
        # Reduce across the target axis to summarize `logp`.
        logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        # Run `logits.argmax` to compute `predicted`.
        predicted=logits.argmax(axis=1)
        # Accumulate the next contribution into `loss_sum`.
        # Accumulate the next contribution into `count`.
        loss_sum+=float(-logp[np.arange(len(y)),y].sum());count+=len(y)
        # Accumulate the next contribution into `correct`.
        correct+=int((predicted==y).sum())
        # Run `np.add.at` to perform the next check or state transition.
        np.add.at(confusion,(y,predicted),1)
    # Return `{'count': count, 'loss_sum': loss_sum, 'loss': loss_sum / count, 'correct': correct, 'accuracy': correct / count, 'confusion': confusion.tolist()}` to the caller.
    return {"count":count,"loss_sum":loss_sum,"loss":loss_sum/count,
            "correct":correct,"accuracy":correct/count,"confusion":confusion.tolist()}


# Function `calibrate(state, training_data)` implementing this stage's computation:
def calibrate(state,training_data):
    # Guard input contract (`dataset_hash(training_data) != state['dataset_sha256']`) and fail fast if violated.
    if dataset_hash(training_data)!=state["dataset_sha256"]:
        raise ValueError("calibration must use declared training data, not evaluation data")
    # Run `features` to compute `feature`.
    feature=features(fixed_windows(training_data))
    # Convert `normalized` to a host NumPy array for inspection or verification.
    normalized=np.asarray((feature-state["mean"])/state["std"])
    # Reduce across the target axis to summarize `activation_scale`.
    activation_scale=max(float(np.max(np.abs(normalized)))/127,1e-8)
    # Convert `W` to a host NumPy array for inspection or verification.
    # Reduce across the target axis to summarize `maximum`.
    W=np.asarray(state["W"]);maximum=np.max(np.abs(W),axis=0,keepdims=True)
    # Cast or evaluate `weight_scales` in explicit floating-point precision.
    weight_scales=np.where(maximum==0,1.,maximum/127).astype(np.float32)
    # Combine or mask array elements to form `q`.
    q=np.clip(np.rint(W/weight_scales),-127,127).astype(np.int8)
    # Return `{'qW': q, 'weight_scales': weight_scales, 'activation_scale': activation_scale, 'calibration_sha256': dataset_hash(training_data)}` to the caller.
    return {"qW":q,"weight_scales":weight_scales,"activation_scale":activation_scale,
            "calibration_sha256":dataset_hash(training_data)}


# Function `policy_scores(state, waveforms, calibration, policy)` implementing this stage's computation:
def policy_scores(state,waveforms,calibration,policy="fp32"):
    # Evaluate `x` from the current inputs and state.
    x=(features_core(waveforms)-state["mean"])/state["std"]
    # Branch on condition `policy == 'fp32'`:
    if policy=="fp32":return x@state["W"]+state["b"]
    # Create device-backed JAX array `q`.
    # Create device-backed JAX array `s`.
    q=jnp.asarray(calibration["qW"]);s=jnp.asarray(calibration["weight_scales"])
    # Branch on condition `policy == 'w8a32'`:
    if policy=="w8a32":return x@(q.astype(jnp.float32)*s)+state["b"]
    # Guard input contract (`policy != 'w8a8'`) and fail fast if violated.
    if policy!="w8a8":raise ValueError("unknown precision policy")
    # Evaluate `a` from the current inputs and state.
    a=calibration["activation_scale"]
    # Combine or mask array elements to form `qa`.
    qa=jnp.clip(jnp.rint(x/a),-127,127).astype(jnp.int8)
    # Perform matrix contraction / projection to compute `accum`.
    accum=qa.astype(jnp.int32)@q.astype(jnp.int32)
    # Return `accum.astype(jnp.float32) * (a * s) + state['b']` to the caller.
    return accum.astype(jnp.float32)*(a*s)+state["b"]


# Function `export_release(path, state, calibration)` implementing this stage's computation:
def export_release(path,state,calibration):
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    # Guard input contract (`calibration['calibration_sha256'] != state['dataset_sha256']`) and fail fast if violated.
    if calibration["calibration_sha256"]!=state["dataset_sha256"]:
        raise ValueError("calibration provenance mismatch")
    # Evaluate `artifacts` from the current inputs and state.
    artifacts={}
    # Loop over `policy` in `('fp32', 'w8a32', 'w8a8')`:
    for policy in ("fp32","w8a32","w8a8"):
        # Loop over `batch` in `BATCHES`:
        for batch in BATCHES:
            # Compile and trace the function with XLA (`function`).
            function=jax.jit(lambda x,policy=policy:policy_scores(state,x,calibration,policy))
            # Cast or evaluate `data` in explicit floating-point precision.
            data=export.export(function)(jax.ShapeDtypeStruct((batch,1024),jnp.float32)).serialize()
            # Evaluate `name` from the current inputs and state.
            # Execute the next step of the computation.
            name=f"{policy}-b{batch}.jaxexport";(path/name).write_bytes(data)
            # Compute deterministic cryptographic digest `artifacts[f'{policy}:{batch}']` for provenance verification.
            artifacts[f"{policy}:{batch}"]={"file":name,"sha256":hashlib.sha256(data).hexdigest()}
    # Convert `manifest` to a host NumPy array for inspection or verification.
    manifest={"preprocessing":PREPROCESS,"classes":CLASSES,"batches":list(BATCHES),
              "training_data_sha256":state["dataset_sha256"],"completed_steps":state["step"],
              "parameter_sha256":digest_arrays(np.asarray(state["W"]),np.asarray(state["b"])),
              "normalizer_sha256":digest_arrays(np.asarray(state["mean"]),np.asarray(state["std"])),
              "calibration_sha256":calibration["calibration_sha256"],"jax":jax.__version__,
              "activation_scale":calibration["activation_scale"],"weight_scales":calibration["weight_scales"].tolist(),
              "integer_contract":{"weight_axis":"output class","zero_point":0,"range":[-127,127],"bias":"float32 after rescaling","codes_dtype":"int8"},
              "policies":{"fp32":"FP32 STFT/features/weights/accumulation/output",
                          "w8a32":"FP32 STFT/features; INT8 weight storage dequantized to FP32; FP32 accumulation/bias/output",
                          "w8a8":"FP32 STFT/features; INT8 normalized features and weights; INT32 dot accumulation; FP32 bias/output"},
              "native_integer_acceleration_claimed":False,"device":"CPU","artifacts":artifacts}
    # Read or serialize artifact data on disk (``).
    (path/"release.json").write_text(json.dumps(manifest,indent=2))
    # Return `manifest` to the caller.
    return manifest


# Function `load_release(path)` implementing this stage's computation:
def load_release(path):
    # Read or serialize artifact data on disk (`path`).
    # Read or serialize artifact data on disk (`manifest`).
    path=Path(path);manifest=json.loads((path/"release.json").read_text())
    # Guard input contract (`manifest['preprocessing'] != PREPROCESS or manifest['jax'] != jax.__version__`) and fail fast if violated.
    if manifest["preprocessing"]!=PREPROCESS or manifest["jax"]!=jax.__version__:
        raise ValueError("release preprocessing/runtime mismatch")
    # Evaluate `artifacts` from the current inputs and state.
    artifacts={}
    # Loop over `(key, item)` in `manifest['artifacts'].items()`:
    for key,item in manifest["artifacts"].items():
        # Evaluate `data` from the current inputs and state.
        data=(path/item["file"]).read_bytes()
        # Guard input contract (`hashlib.sha256(data).hexdigest() != item['sha256']`) and fail fast if violated.
        if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("artifact checksum mismatch")
        # Run `export.deserialize` to compute `artifacts[key]`.
        artifacts[key]=export.deserialize(data)
    # Return `(manifest, artifacts)` to the caller.
    return manifest,artifacts


# Function `infer_release(manifest, artifacts, waveforms, sample_rate, ...)` implementing this stage's computation:
def infer_release(manifest,artifacts,waveforms,sample_rate=8000,channels=1,
                  policy="fp32",version="audio-stft-v1"):
    # Run `validate_waveforms` to compute `a`.
    a=validate_waveforms(waveforms,sample_rate,channels,version)
    # Guard input contract (`len(a) not in manifest['batches'] or policy not in manifest['policies']`) and fail fast if violated.
    if len(a) not in manifest["batches"] or policy not in manifest["policies"]:
        raise ValueError("unsupported batch or precision policy")
    # Synchronize host execution until asynchronous device computation completes.
    logits=artifacts[f"{policy}:{len(a)}"].call(jnp.asarray(a)).block_until_ready()
    # Return `{'logits': np.asarray(logits), 'class_ids': np.asarray(logits).argmax(axis=1)}` to the caller.
    return {"logits":np.asarray(logits),"class_ids":np.asarray(logits).argmax(axis=1)}


# Function `benchmark(manifest, artifacts, waveforms, policy, ...)` implementing this stage's computation:
def benchmark(manifest,artifacts,waveforms,policy="fp32",repeats=30):
    # Record execution timing or profiler trace in `began`.
    # Execute the next step of the computation.
    began=time.perf_counter();infer_release(manifest,artifacts,waveforms,policy=policy)
    # Record execution timing or profiler trace in `first_ms`.
    first_ms=(time.perf_counter()-began)*1000
    # Evaluate `samples` from the current inputs and state.
    samples=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `began`.
        # Execute the next step of the computation.
        began=time.perf_counter();infer_release(manifest,artifacts,waveforms,policy=policy)
        # Record execution timing or profiler trace in ``.
        samples.append((time.perf_counter()-began)*1000)
    # Return `{'first_request_ms': first_ms, 'samples_ms': samples, 'p50_ms': float(np.percentile(samples, 50)), 'p95_ms': float(np.percentile(samples, 95)), 'boundary': 'validated decoded PCM + placement + exported STFT/normalization/classifier + wait + host outputs', 'excluded': 'file decoding, audio capture, window buffering, networking', 'device': 'CPU', 'audio_duration_ms': 128.0, 'examples_per_second': 1000 * len(waveforms) / float(np.percentile(samples, 50))}` to the caller.
    return {"first_request_ms":first_ms,"samples_ms":samples,
            "p50_ms":float(np.percentile(samples,50)),"p95_ms":float(np.percentile(samples,95)),
            "boundary":"validated decoded PCM + placement + exported STFT/normalization/classifier + wait + host outputs",
            "excluded":"file decoding, audio capture, window buffering, networking","device":"CPU",
            "audio_duration_ms":128.,"examples_per_second":1000*len(waveforms)/float(np.percentile(samples,50))}


# Function `load_wav_manifest(manifest_path, split)` implementing this stage's computation:
def load_wav_manifest(manifest_path,split):
    """Ingest user-provided licensed PCM16/mono/8kHz files; no implicit resampling."""
    # Read or serialize artifact data on disk (`manifest_path`).
    # Read or serialize artifact data on disk (`rows`).
    manifest_path=Path(manifest_path);rows=json.loads(manifest_path.read_text())
    # Evaluate `groups` from the current inputs and state.
    # Run `set` to compute `ids`.
    groups={};ids=set()
    # Loop over `row` in `rows`:
    for row in rows:
        # Loop over `field` in `('id', 'file', 'label', 'split', 'group', 'license', 'source', 'sha256')`:
        for field in ("id","file","label","split","group","license","source","sha256"):
            # Guard input contract (`not row.get(field) and row.get(field) != 0`) and fail fast if violated.
            if not row.get(field) and row.get(field)!=0:raise ValueError("missing provenance field: "+field)
        # Guard input contract (`row['id'] in ids`) and fail fast if violated.
        if row["id"] in ids:raise ValueError("duplicate recording identity")
        # Run `ids.add` to perform the next check or state transition.
        # Run `ids.add` to perform the next check or state transition.
        ids.add(row["id"]);groups.setdefault(row["group"],set()).add(row["split"])
    # Guard input contract (`any((len(splits) > 1 for splits in groups.values()))`) and fail fast if violated.
    if any(len(splits)>1 for splits in groups.values()):raise ValueError("recording group leaks across splits")
    # Evaluate `output` from the current inputs and state.
    # Evaluate `labels` from the current inputs and state.
    # Evaluate `selected` from the current inputs and state.
    output=[];labels=[];selected=[]
    # Loop over `row` in `rows`:
    for row in rows:
        # Branch on condition `row['split'] != split`:
        if row["split"]!=split:continue
        # Evaluate `path` from the current inputs and state.
        path=manifest_path.parent/row["file"]
        # Guard input contract (`hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']`) and fail fast if violated.
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("WAV checksum mismatch")
        # Enter managed runtime/context scope for this block:
        with wave.open(str(path),"rb") as wav:
            # Guard input contract (`wav.getframerate() != 8000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2`) and fail fast if violated.
            if wav.getframerate()!=8000 or wav.getnchannels()!=1 or wav.getsampwidth()!=2:
                raise ValueError("require PCM16 mono 8kHz; convert explicitly and preserve source provenance")
            # Cast or evaluate `samples` in explicit floating-point precision.
            samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2").astype(np.float32)/32768.
        # Evaluate `row.get('offset_samples', 0)` and convert the result into Python scalar/collection `offset`.
        offset=int(row.get("offset_samples",0))
        # Guard input contract (`offset < 0 or len(samples) < offset + 1280 or row['label'] not in (0, 1, 2)`) and fail fast if violated.
        if offset<0 or len(samples)<offset+1280 or row["label"] not in (0,1,2):raise ValueError("invalid segment or label")
        # Append the current step result to `output`.
        # Append the current step result to `output`.
        # Append the current step result to `output`.
        output.append(samples[offset:offset+1280]);labels.append(row["label"]);selected.append(row)
    # Guard input contract (`not output`) and fail fast if violated.
    if not output:raise ValueError("requested split is empty")
    # Return `{'waveforms': np.asarray(output, dtype=np.float32), 'labels': np.asarray(labels, dtype=np.int32), 'ids': [r['id'] for r in selected], 'groups': [r['group'] for r in selected], 'sample_rate': 8000, 'provenance': 'user-provided WAV manifest ' + str(manifest_path)}` to the caller.
    return {"waveforms":np.asarray(output,dtype=np.float32),"labels":np.asarray(labels,dtype=np.int32),
            "ids":[r["id"] for r in selected],"groups":[r["group"] for r in selected],
            "sample_rate":8000,"provenance":"user-provided WAV manifest "+str(manifest_path)}
