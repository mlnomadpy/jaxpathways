"""Local CPU release audit. Integer arithmetic is explicit; no device speedup claim."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export


def objective(weights, X, targets):
    scores=X@weights
    return jnp.mean(jnp.logaddexp(0.,scores)-targets*scores)


def train(initial, X, targets, steps=250, rate=.15):
    X,targets,initial=map(jnp.asarray,(X,targets,initial))
    if X.ndim!=2 or targets.shape!=(X.shape[0],) or initial.shape!=(X.shape[1],):
        raise ValueError("expected X (rows,features), targets (rows,), weights (features,)")
    if len(X)==0 or not np.isfinite(np.asarray(X)).all() or not np.isfinite(np.asarray(targets)).all() or not np.isfinite(np.asarray(initial)).all():
        raise ValueError("training data must be finite and nonempty")
    if np.any((np.asarray(targets)<0)|(np.asarray(targets)>1)):
        raise ValueError("binary/soft targets must lie between zero and one")
    if steps<0 or not np.isfinite(rate) or rate<=0:raise ValueError("invalid optimizer configuration")
    def update(w,_):
        loss,grad=jax.value_and_grad(objective)(w,X,targets)
        return w-rate*grad,loss
    return jax.lax.scan(update,initial,None,length=steps)


def quantize(weights,bits=8):
    """Per-output symmetric codes for a (features, outputs) matrix."""
    weights=np.asarray(weights,dtype=np.float32)
    if weights.ndim!=2 or not np.isfinite(weights).all() or bits not in (4,8):
        raise ValueError("finite matrix and signed 4/8-bit policy required")
    limit=2**(bits-1)-1
    maximum=np.max(np.abs(weights),axis=0,keepdims=True)
    scales=np.where(maximum==0,1.,maximum/limit).astype(np.float32)
    codes=np.clip(np.rint(weights/scales),-limit,limit).astype(np.int8)
    return codes,scales


def prepare(directory,weights,calibration,provenance):
    """Write trained weights and six actual serialized inference computations."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    weights=np.asarray(weights,dtype=np.float32);calibration=np.asarray(calibration,dtype=np.float32)
    if weights.ndim!=1 or calibration.ndim!=2 or calibration.shape[1]!=weights.size or len(calibration)==0:
        raise ValueError("calibration and weights must have matching feature axes")
    if not np.isfinite(calibration).all() or not np.isfinite(weights).all():
        raise ValueError("finite calibration/weights required")
    required={"source_data_sha256","adaptation_data_sha256","training_steps","objective","base_checkpoint_sha256"}
    if not required<=provenance.keys():raise ValueError("training and base-checkpoint provenance required")
    q,s=quantize(weights[:,None],8)
    activation_scale=np.float32(max(float(np.max(np.abs(calibration)))/127,1e-8))
    np.save(directory/"weights.npy",weights,allow_pickle=False)
    policies={
        "fp32":dict(weight="float32",activation="float32",accumulator="float32",output="float32"),
        "w8a32":dict(weight="int8",activation="float32",accumulator="float32",output="float32"),
        "w8a8":dict(weight="int8",activation="int8",accumulator="int32",output="float32"),
    }
    def make_infer(policy):
        if policy=="fp32":
            return jax.jit(lambda x:x@jnp.asarray(weights))
        if policy=="w8a32":
            return jax.jit(lambda x:(x@(jnp.asarray(q,dtype=jnp.float32)*jnp.asarray(s))).reshape(-1))
        def integer(x):
            qa=jnp.clip(jnp.rint(x/activation_scale),-127,127).astype(jnp.int8)
            accum=qa.astype(jnp.int32)@jnp.asarray(q,dtype=jnp.int32)
            return (accum.astype(jnp.float32)*(activation_scale*jnp.asarray(s))).reshape(-1)
        return jax.jit(integer)
    artifacts={}
    for policy in policies:
        for batch in (1,8):
            artifact=export.export(make_infer(policy))(jax.ShapeDtypeStruct((batch,len(weights)),jnp.float32))
            name=f"{policy}-b{batch}.jaxexport"
            data=artifact.serialize();(directory/name).write_bytes(data)
            artifacts[f"{policy}:{batch}"]={"file":name,"sha256":hashlib.sha256(data).hexdigest()}
    manifest={"schema_version":1,"features":len(weights),"batch_sizes":[1,8],"jax":jax.__version__,
              "platform":"cpu","preprocessing":"identity float32 features; intercept is caller-supplied",
              "weights_sha256":hashlib.sha256((directory/"weights.npy").read_bytes()).hexdigest(),
              "calibration_sha256":hashlib.sha256(calibration.tobytes()).hexdigest(),
              "activation_scale":float(activation_scale),"weight_scales":s.tolist(),
              "policies":policies,"artifacts":artifacts,"provenance":provenance,
              "boundary":"serialized JAX CPU inference; integer kernel acceleration unclaimed"}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def load(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    if hashlib.sha256((directory/"weights.npy").read_bytes()).hexdigest()!=manifest["weights_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    restored={}
    for key,item in manifest["artifacts"].items():
        data=(directory/item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("artifact hash mismatch")
        restored[key]=export.deserialize(data)
    return manifest,restored


def request(payload,manifest,restored,policy="fp32"):
    """In-process JSON service boundary: validation, transfer, inference, completion, encoding."""
    if not isinstance(payload,str) or len(payload.encode("utf8"))>100_000:
        raise ValueError("payload must be a bounded JSON string")
    try:
        item=json.loads(payload)
        raw=np.asarray(item["features"])
        if raw.dtype.kind not in "iuf":raise ValueError("features must be numeric")
        x=raw.astype(np.float32)
    except (KeyError,TypeError,OverflowError,json.JSONDecodeError) as err:
        raise ValueError("malformed feature payload") from err
    if x.ndim!=2 or x.shape[1]!=manifest["features"] or x.shape[0] not in manifest["batch_sizes"]:
        raise ValueError("unsupported request shape")
    if not np.isfinite(x).all() or policy not in manifest["policies"]:
        raise ValueError("nonfinite input or unsupported precision policy")
    result=restored[f"{policy}:{len(x)}"].call(jnp.asarray(x)).block_until_ready()
    clipped=int(np.sum(np.abs(x)>127*manifest["activation_scale"])) if policy=="w8a8" else 0
    return json.dumps({"scores":np.asarray(result).tolist(),"policy":policy,"clipped_activations":clipped})


def benchmark(payload,manifest,restored,policy="fp32",repeats=30):
    if repeats<2:raise ValueError("need multiple observations")
    began=time.perf_counter();request(payload,manifest,restored,policy)
    first_ms=(time.perf_counter()-began)*1000
    values=[]
    for _ in range(repeats):
        began=time.perf_counter();request(payload,manifest,restored,policy)
        values.append((time.perf_counter()-began)*1000)
    batch=len(json.loads(payload)["features"])
    return {"first_request_ms":first_ms,"samples_ms":values,"batch":batch,
            "p50_ms":float(np.percentile(values,50)),"p95_ms":float(np.percentile(values,95)),
            "examples_per_second":1000*batch/float(np.percentile(values,50)),
            "boundary":"JSON decode + validate + transfer + restored call + wait + JSON encode",
            "network_included":False,"device":"CPU"}


def simulate(arrival_ms,service_ms):
    arrivals=np.asarray(arrival_ms,dtype=float)
    if arrivals.ndim!=1 or not np.isfinite(arrivals).all() or np.any(arrivals<0) or np.any(np.diff(arrivals)<0):
        raise ValueError("ordered nonnegative finite arrival times required")
    if not np.isfinite(service_ms) or service_ms<=0:raise ValueError("positive service milliseconds required")
    ready=0.;finish=[]
    for arrival in arrivals:
        ready=max(ready,float(arrival))+service_ms;finish.append(ready)
    return np.asarray(finish)-arrivals
