"""Local CPU release audit. Integer arithmetic is explicit; no device speedup claim."""
# Import hashlib for this computation.
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export


# Function `objective(weights, X, targets)` implementing this stage's computation:
def objective(weights, X, targets):
    # Perform matrix / vector contraction (`@`) to compute `scores`.
    scores=X@weights
    # Return `jnp.mean(jnp.logaddexp(0.0, scores) - targets * scores)` to the caller.
    return jnp.mean(jnp.logaddexp(0.,scores)-targets*scores)


# Define `train(initial, X, targets, steps...)` to evaluate the objective and its automatic derivatives:
def train(initial, X, targets, steps=250, rate=.15):
    # Create device-backed JAX array `(X, targets, initial)`.
    X,targets,initial=map(jnp.asarray,(X,targets,initial))
    # Guard input contract (`X.ndim != 2 or targets.shape != (X.shape[0],) or initial.shape != (X.shape[1],)`) and fail fast if violated.
    if X.ndim!=2 or targets.shape!=(X.shape[0],) or initial.shape!=(X.shape[1],):
        raise ValueError("expected X (rows,features), targets (rows,), weights (features,)")
    # Guard input contract (`len(X) == 0 or not np.isfinite(np.asarray(X)).all() or (not np.isfinite(np.asarray(targets)).all()) or (not np.isfinite(np.asarray(initial)).all())`) and fail fast if violated.
    if len(X)==0 or not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(targets)).all() or not np.isfinite(np.asarray(initial)).all():
        raise ValueError("training data must be finite and nonempty")
    # Guard input contract (`np.any((np.asarray(targets) < 0) | (np.asarray(targets) > 1))`) and fail fast if violated.
    if np.any((np.asarray(targets)<0)|(np.asarray(targets)>1)):
        raise ValueError("binary/soft targets must lie between zero and one")
    # Guard input contract (`steps < 0 or not np.isfinite(rate) or rate <= 0`) and fail fast if violated.
    if steps<0 or not np.isfinite(rate) or rate<=0:raise ValueError("invalid optimizer configuration")
    # Define `update(w, _)` to evaluate the objective and its automatic derivatives:
    def update(w,_):
        # Differentiate the objective to obtain `(loss, grad)` via automatic differentiation.
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        # Return `(w - rate * grad, loss)` to the caller.
        return w-rate*grad,loss
    # Return `jax.lax.scan(update, initial, None, length=steps)` to the caller.
    return jax.lax.scan(update,initial,None,length=steps)


# Function `quantize(weights, bits)` implementing this stage's computation:
def quantize(weights,bits=8):
    """Per-output symmetric codes for a (features, outputs) matrix."""
    # Convert `weights` to a host NumPy array for inspection or verification.
    weights=np.asarray(weights,dtype=np.float32)
    # Guard input contract (`weights.ndim != 2 or not np.isfinite(weights).all() or bits not in (4, 8)`) and fail fast if violated.
    if weights.ndim!=2 or not np.isfinite(weights).all() or bits not in (4,8):
        raise ValueError("finite matrix and signed 4/8-bit policy required")
    # Compute `limit` as `2**(bits-1)-1`.
    limit=2**(bits-1)-1
    # Reduce along axis=0 to compute `maximum`.
    maximum=np.max(np.abs(weights),axis=0,keepdims=True)
    # Cast or evaluate `scales` in explicit floating-point precision.
    scales=np.where(maximum==0,1.,maximum/limit).astype(np.float32)
    # Combine or mask array elements to form `codes`.
    codes=np.clip(np.rint(weights/scales),-limit,limit).astype(np.int8)
    # Return `(codes, scales)` to the caller.
    return codes,scales


# Function `prepare(directory, weights, calibration, provenance)` implementing this stage's computation:
def prepare(directory,weights,calibration,provenance):
    """Write trained weights and six actual serialized inference computations."""
    # Compute `directory` as `Path(directory)`.
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    # Convert `weights` to a host NumPy array for inspection or verification.
    # Convert `calibration` to a host NumPy array for inspection or verification.
    weights=np.asarray(weights,dtype=np.float32)
    calibration=np.asarray(calibration,dtype=np.float32)
    # Guard input contract (`weights.ndim != 1 or calibration.ndim != 2 or calibration.shape[1] != weights.size or (len(calibration) == 0)`) and fail fast if violated.
    if weights.ndim!=1 or calibration.ndim!=2 or calibration.shape[1]!=weights.size or len(calibration)==0:
        raise ValueError("calibration and weights must have matching feature axes")
    # Guard input contract (`not np.isfinite(calibration).all() or not np.isfinite(weights).all()`) and fail fast if violated.
    if not np.isfinite(calibration).all() or not np.isfinite(weights).all():
        raise ValueError("finite calibration/weights required")
    # Compute deterministic cryptographic digest `required` for provenance verification.
    required={"source_data_sha256","adaptation_data_sha256","training_steps","objective","base_checkpoint_sha256"}
    # Guard input contract (`not required <= provenance.keys()`) and fail fast if violated.
    if not required<=provenance.keys():raise ValueError("training and base-checkpoint provenance required")
    # Run `quantize` to compute `(q, s)`.
    q,s=quantize(weights[:,None],8)
    # Cast or evaluate `activation_scale` in explicit floating-point precision.
    activation_scale=np.float32(max(float(np.max(np.abs(calibration)))/127,1e-8))
    # Run `np.save` to perform the next check or state transition.
    np.save(directory/"weights.npy",weights,allow_pickle=False)
    # Construct dictionary `policies` with the structured fields for this stage.
    policies={
        "fp32":dict(weight="float32",activation="float32",accumulator="float32",output="float32"),
        "w8a32":dict(weight="int8",activation="float32",accumulator="float32",output="float32"),
        "w8a8":dict(weight="int8",activation="int8",accumulator="int32",output="float32"),
    }
    # Function `make_infer(policy)` implementing this stage's computation:
    def make_infer(policy):
        # Branch on condition `policy == 'fp32'`:
        if policy=="fp32":
            return jax.jit(lambda x:x@jnp.asarray(weights))
        # Branch on condition `policy == 'w8a32'`:
        if policy=="w8a32":
            return jax.jit(lambda x:(x@(jnp.asarray(q,dtype=jnp.float32)*jnp.asarray(s))).reshape(-1))
        # Function `integer(x)` implementing this stage's computation:
        def integer(x):
            # Combine or mask array elements to form `qa`.
            qa=jnp.clip(jnp.rint(x/activation_scale),-127,127).astype(jnp.int8)
            # Create device-backed JAX array `accum`.
            accum=qa.astype(jnp.int32)@jnp.asarray(q,dtype=jnp.int32)
            # Return `(accum.astype(jnp.float32) * (activation_scale * jnp.asarray(s))).reshape(-1)` to the caller.
            return (accum.astype(jnp.float32)*(activation_scale*jnp.asarray(s))).reshape(-1)
        # Return `jax.jit(integer)` to the caller.
        return jax.jit(integer)
    # Construct dictionary `artifacts` with the structured fields for this stage.
    artifacts={}
    # Loop over `policy` in `policies`:
    for policy in policies:
        # Loop over `batch` in `(1, 8)`:
        for batch in (1,8):
            # Cast or evaluate `artifact` in explicit floating-point precision.
            artifact=export.export(make_infer(policy))(jax.ShapeDtypeStruct((batch,len(weights)),jnp.float32))
            # Compute `name` as `f"{policy}-b{batch}.jaxexport"`.
            name=f"{policy}-b{batch}.jaxexport"
            # Run `artifact.serialize` to compute `data`.
            # Compute `data` as `artifact.serialize()`.
            data=artifact.serialize()
            (directory/name).write_bytes(data)
            # Compute deterministic cryptographic digest `artifacts[f'{policy}:{batch}']` for provenance verification.
            artifacts[f"{policy}:{batch}"]={"file":name,"sha256":hashlib.sha256(data).hexdigest()}
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={"schema_version":1,"features":len(weights),"batch_sizes":[1,8],"jax":jax.__version__,
              "platform":"cpu","preprocessing":"identity float32 features; intercept is caller-supplied",
              "weights_sha256":hashlib.sha256((directory/"weights.npy").read_bytes()).hexdigest(),
              "calibration_sha256":hashlib.sha256(calibration.tobytes()).hexdigest(),
              "activation_scale":float(activation_scale),"weight_scales":s.tolist(),
              "policies":policies,"artifacts":artifacts,"provenance":provenance,
              "boundary":"serialized JAX CPU inference; integer kernel acceleration unclaimed"}
    # Compute `(directory/"manifest.json").write_text(json.dumps(manifest,indent` as `2))`.
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # Return `manifest` to the caller.
    return manifest


# Function `load(directory)` implementing this stage's computation:
def load(directory):
    # Read or serialize artifact data on disk (`directory`).
    directory=Path(directory)
    # Read or serialize artifact data on disk (`manifest`).
    manifest=json.loads((directory/"manifest.json").read_text())
    # Guard input contract (`hashlib.sha256((directory / 'weights.npy').read_bytes()).hexdigest() != manifest['weights_sha256']`) and fail fast if violated.
    if hashlib.sha256((directory/"weights.npy").read_bytes()).hexdigest()!=manifest["weights_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    # Construct dictionary `restored` with the structured fields for this stage.
    restored={}
    # Iterate over `(key, item)` to step through the computation:
    for key,item in manifest["artifacts"].items():
        # Evaluate the compound expression for `data`.
        data=(directory/item["file"]).read_bytes()
        # Guard input contract (`hashlib.sha256(data).hexdigest() != item['sha256']`) and fail fast if violated.
        if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("artifact hash mismatch")
        # Run `export.deserialize` to compute `restored[key]`.
        restored[key]=export.deserialize(data)
    # Return `(manifest, restored)` to the caller.
    return manifest,restored


# Function `request(payload, manifest, restored, policy)` implementing this stage's computation:
def request(payload,manifest,restored,policy="fp32"):
    """In-process JSON service boundary: validation, transfer, inference, completion, encoding."""
    # Guard input contract (`not isinstance(payload, str) or len(payload.encode('utf8')) > 100000`) and fail fast if violated.
    if not isinstance(payload,str) or len(payload.encode("utf8"))>100_000:
        raise ValueError("payload must be a bounded JSON string")
    # Run the boundary check and catch the expected exception:
    try:
        item=json.loads(payload)
        raw=np.asarray(item["features"])
        if raw.dtype.kind not in "iuf":raise ValueError("features must be numeric")
        x=raw.astype(np.float32)
    except (KeyError,TypeError,OverflowError,json.JSONDecodeError) as err:
        raise ValueError("malformed feature payload") from err
    # Guard input contract (`x.ndim != 2 or x.shape[1] != manifest['features'] or x.shape[0] not in manifest['batch_sizes']`) and fail fast if violated.
    if x.ndim!=2 or x.shape[1]!=manifest["features"] or x.shape[0] not in manifest["batch_sizes"]:
        raise ValueError("unsupported request shape")
    # Guard input contract (`not np.isfinite(x).all() or policy not in manifest['policies']`) and fail fast if violated.
    if not np.isfinite(x).all() or policy not in manifest["policies"]:
        raise ValueError("nonfinite input or unsupported precision policy")
    # Synchronize host execution until asynchronous device computation completes.
    result=restored[f"{policy}:{len(x)}"].call(jnp.asarray(x)).block_until_ready()
    # Aggregate array values to compute `clipped`.
    clipped=int(np.sum(np.abs(x)>127*manifest["activation_scale"])) if policy=="w8a8" else 0
    # Return `json.dumps({'scores': np.asarray(result).tolist(), 'policy': policy, 'clipped_activations': clipped})` to the caller.
    return json.dumps({"scores":np.asarray(result).tolist(),"policy":policy,"clipped_activations":clipped})


# Function `benchmark(payload, manifest, restored, policy, ...)` implementing this stage's computation:
def benchmark(payload,manifest,restored,policy="fp32",repeats=30):
    # Guard input contract (`repeats < 2`) and fail fast if violated.
    if repeats<2:raise ValueError("need multiple observations")
    # Record execution timing or profiler trace in `began`.
    # Compute `began` as `time.perf_counter()`.
    began=time.perf_counter()
    request(payload,manifest,restored,policy)
    # Record execution timing or profiler trace in `first_ms`.
    first_ms=(time.perf_counter()-began)*1000
    # Initialize list `values` for the stage values.
    values=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `began`.
        # Compute `began` as `time.perf_counter()`.
        began=time.perf_counter()
        request(payload,manifest,restored,policy)
        # Record execution timing or profiler trace in ``.
        values.append((time.perf_counter()-began)*1000)
    # Read or serialize artifact data on disk (`batch`).
    batch=len(json.loads(payload)["features"])
    # Return `{'first_request_ms': first_ms, 'samples_ms': values, 'batch': batch, 'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)), 'examples_per_second': 1000 * batch / float(np.percentile(values, 50)), 'boundary': 'JSON decode + validate + transfer + restored call + wait + JSON encode', 'network_included': False, 'device': 'CPU'}` to the caller.
    return {"first_request_ms":first_ms,"samples_ms":values,"batch":batch,
            "p50_ms":float(np.percentile(values,50)),"p95_ms":float(np.percentile(values,95)),
            "examples_per_second":1000*batch/float(np.percentile(values,50)),
            "boundary":"JSON decode + validate + transfer + restored call + wait + JSON encode",
            "network_included":False,"device":"CPU"}


# Function `simulate(arrival_ms, service_ms)` implementing this stage's computation:
def simulate(arrival_ms,service_ms):
    # Convert `arrivals` to a host NumPy array for inspection or verification.
    arrivals=np.asarray(arrival_ms,dtype=float)
    # Guard input contract (`arrivals.ndim != 1 or not np.isfinite(arrivals).all() or np.any(arrivals < 0) or np.any(np.diff(arrivals) < 0)`) and fail fast if violated.
    if arrivals.ndim!=1 or not np.isfinite(arrivals).all() or np.any(arrivals<0) or np.any(np.diff(arrivals)<0):
        raise ValueError("ordered nonnegative finite arrival times required")
    # Guard input contract (`not np.isfinite(service_ms) or service_ms <= 0`) and fail fast if violated.
    if not np.isfinite(service_ms) or service_ms<=0:raise ValueError("positive service milliseconds required")
    # Compute `ready` as `0.`.
    ready=0.
    finish=[]
    # Iterate over `arrival` to step through the computation:
    for arrival in arrivals:
        # Run `max` to compute `ready`.
        # Compute `ready` as `max(ready,float(arrival))+service_ms`.
        ready=max(ready,float(arrival))+service_ms
        finish.append(ready)
    # Return `np.asarray(finish) - arrivals` to the caller.
    return np.asarray(finish)-arrivals
