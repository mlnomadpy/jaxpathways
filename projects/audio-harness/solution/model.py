"""Connected CPU sound-event teaching harness. No speech recognition claim."""
import hashlib
import json
from pathlib import Path
import time
import wave
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

PREPROCESS = {"version":"audio-stft-v1","sample_rate":8000,"channels":1,"samples":1024,
              "clip_samples":1280,"window":128,"hop":64,"window_function":"symmetric Hann",
              "spectrum":"rfft magnitude squared / sum(window squared)",
              "feature":"mean over frames of log1p(power); 65 frequency bins",
              "normalization":"training mean and std; std floor 0.05"}
CLASSES = ["background","low-tone","high-tone"]
BATCHES = (1,4)


def digest_arrays(*arrays):
    h=hashlib.sha256()
    for item in arrays:
        a=np.ascontiguousarray(item)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def dataset_hash(data):
    return digest_arrays(data["waveforms"],data["labels"],np.asarray(data["ids"]),
                         np.asarray(data["groups"]),np.asarray([data["sample_rate"]]))


def make_dataset(seed=41,count=96,split="train",shift=False):
    """Synthetic recording IDs and groups are disjoint by split and seed."""
    rng=np.random.default_rng(seed)
    labels=np.arange(count,dtype=np.int32)%3
    time_axis=np.arange(PREPROCESS["clip_samples"])/PREPROCESS["sample_rate"]
    waves=[]
    for label in labels:
        gain=rng.uniform(.2,.7)
        noise=rng.normal(0,.008 if not shift else .025,len(time_axis))
        if label==0:
            signal=noise
        else:
            center=(500.,1500.)[label-1]
            freq=center+rng.uniform(-35,35) if not shift else center+rng.uniform(-90,90)
            signal=gain*np.sin(2*np.pi*freq*time_axis+rng.uniform(0,2*np.pi))+noise
        waves.append(signal)
    return {"waveforms":np.asarray(waves,dtype=np.float32),"labels":labels,
            "ids":[f"{split}-recording-{seed}-{i:04d}" for i in range(count)],
            "groups":[f"{split}-source-{seed}-{i:04d}" for i in range(count)],
            "sample_rate":8000,"provenance":"generated tone/background fixture; no external recordings"}


def validate_waveforms(waveforms,sample_rate=8000,channels=1,version="audio-stft-v1"):
    a=np.asarray(waveforms)
    if sample_rate!=8000 or channels!=1 or version!=PREPROCESS["version"]:
        raise ValueError("sample rate, channel count or preprocessing version mismatch")
    if a.ndim!=2 or a.shape[1]!=1024 or a.dtype.kind not in "fiu":
        raise ValueError("waveforms must have shape (batch,1024) with numeric amplitudes")
    if a.shape[0]==0 or not np.isfinite(a).all() or np.max(np.abs(a))>1.:
        raise ValueError("require finite nonempty PCM amplitudes in [-1,1]")
    return np.asarray(a,dtype=np.float32)


def spectrum_core(waveforms):
    offsets=jnp.arange(15)[:,None]*64+jnp.arange(128)[None,:]
    frames=waveforms[:,offsets]
    window=jnp.hanning(128).astype(jnp.float32)
    transformed=jnp.fft.rfft(frames*window,axis=-1)
    return (jnp.abs(transformed)**2)/jnp.sum(window**2)


@jax.jit
def features_core(waveforms):
    return jnp.mean(jnp.log1p(spectrum_core(waveforms)),axis=1)


def features(waveforms,sample_rate=8000,channels=1,version="audio-stft-v1"):
    return features_core(jnp.asarray(validate_waveforms(waveforms,sample_rate,channels,version)))


def fixed_windows(data):
    # Deterministic center crop; training augmentation has a separate path.
    return np.asarray(data["waveforms"][:,128:1152],dtype=np.float32)


def objective(W,b,normalized,labels):
    if normalized.ndim!=2 or labels.shape!=(normalized.shape[0],) or not jnp.issubdtype(labels.dtype,jnp.integer):
        raise ValueError("one integer class label per feature row required")
    scores=normalized@W+b
    return -jnp.mean(jax.nn.log_softmax(scores)[jnp.arange(len(labels)),labels])


def create_state(data,seed=0,batch_size=12,rate=.04,momentum=.85):
    if data["sample_rate"]!=8000 or len(data["ids"])!=len(data["labels"]):
        raise ValueError("dataset contract mismatch")
    if (len(data["groups"])!=len(data["ids"]) or len(set(data["ids"]))!=len(data["ids"])
        or not np.isfinite(data["waveforms"]).all() or np.max(np.abs(data["waveforms"]))>1
        or np.asarray(data["labels"]).dtype.kind not in "iu" or np.any((data["labels"]<0)|(data["labels"]>2))):
        raise ValueError("invalid recording identities, PCM values or class labels")
    if np.asarray(data["waveforms"]).shape!=(len(data["labels"]),1280):
        raise ValueError("training recording shape must be (recordings,1280)")
    if batch_size<1 or rate<=0 or not 0<=momentum<1:
        raise ValueError("invalid optimizer or batch configuration")
    train_features=features(fixed_windows(data))
    mean=train_features.mean(axis=0)
    std=jnp.maximum(train_features.std(axis=0),.05)
    key,init_key,order_key=jax.random.split(jax.random.PRNGKey(seed),3)
    W=.01*jax.random.normal(init_key,(65,3))
    return {"W":W,"b":jnp.zeros(3),"vW":jnp.zeros_like(W),"vb":jnp.zeros(3),
            "mean":mean,"std":std,"key":key,"order":jax.random.permutation(order_key,len(data["labels"])),
            "step":0,"cursor":0,"epoch":0,
            "config":{"seed":seed,"batch_size":batch_size,"rate":rate,"momentum":momentum},
            "dataset_sha256":dataset_hash(data)}


@jax.jit
def update_arrays(W,b,vW,vb,normalized,labels,rate,momentum):
    loss,(gW,gb)=jax.value_and_grad(objective,argnums=(0,1))(W,b,normalized,labels)
    vW=momentum*vW+gW;vb=momentum*vb+gb
    return W-rate*vW,b-rate*vb,vW,vb,loss


def step(state,data):
    if state["dataset_sha256"]!=dataset_hash(data):
        raise ValueError("training dataset identity changed")
    state=dict(state)
    if state["cursor"]==len(data["labels"]):
        state["key"],order_key=jax.random.split(state["key"])
        state["order"]=jax.random.permutation(order_key,len(data["labels"]))
        state["cursor"]=0;state["epoch"]+=1
    cursor=state["cursor"]
    indices=np.asarray(state["order"][cursor:cursor+state["config"]["batch_size"]])
    state["key"],crop_key,gain_key,noise_key=jax.random.split(state["key"],4)
    offsets=jax.random.randint(crop_key,(len(indices),),0,257)
    gains=jax.random.uniform(gain_key,(len(indices),1),minval=.8,maxval=1.15)
    clips=jnp.asarray(data["waveforms"][indices])
    windows=jnp.take_along_axis(clips,offsets[:,None]+jnp.arange(1024)[None,:],axis=1)
    augmented=windows*gains+.003*jax.random.normal(noise_key,windows.shape)
    feat=features_core(augmented)
    normalized=(feat-state["mean"])/state["std"]
    state["W"],state["b"],state["vW"],state["vb"],loss=update_arrays(
        state["W"],state["b"],state["vW"],state["vb"],normalized,
        jnp.asarray(data["labels"][indices]),state["config"]["rate"],state["config"]["momentum"])
    state["cursor"]+=len(indices);state["step"]+=1
    trace={"ids":[data["ids"][int(i)] for i in indices],"crop_offsets":np.asarray(offsets).tolist(),
           "gains":np.asarray(gains).ravel().tolist(),"feature_sha256":digest_arrays(np.asarray(feat)),
           "loss":float(loss),"completed_step":state["step"]}
    return state,trace


ARRAY_FIELDS=("W","b","vW","vb","mean","std","key","order")


def save_checkpoint(path,state):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    np.savez(path/"state.npz",**{k:np.asarray(state[k]) for k in ARRAY_FIELDS})
    manifest={k:state[k] for k in ("step","cursor","epoch","config","dataset_sha256")}
    manifest.update(preprocessing=PREPROCESS,jax=jax.__version__,
                    state_sha256=hashlib.sha256((path/"state.npz").read_bytes()).hexdigest())
    (path/"checkpoint.json").write_text(json.dumps(manifest,indent=2))


def load_checkpoint(path,data,config):
    path=Path(path);manifest=json.loads((path/"checkpoint.json").read_text())
    if manifest["dataset_sha256"]!=dataset_hash(data) or manifest["config"]!=config:
        raise ValueError("dataset or training configuration mismatch")
    if manifest["preprocessing"]!=PREPROCESS or manifest["jax"]!=jax.__version__:
        raise ValueError("preprocessing or runtime mismatch")
    if hashlib.sha256((path/"state.npz").read_bytes()).hexdigest()!=manifest["state_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    with np.load(path/"state.npz",allow_pickle=False) as data_arrays:
        state={k:jnp.asarray(data_arrays[k]) for k in ARRAY_FIELDS}
    state.update({k:manifest[k] for k in ("step","cursor","epoch","config","dataset_sha256")})
    return state


def scores(state,waveforms):
    x=features(waveforms)
    return ((x-state["mean"])/state["std"])@state["W"]+state["b"]


def evaluate(state,data,batch_size=13):
    count=0;loss_sum=0.;correct=0;confusion=np.zeros((3,3),dtype=np.int64)
    windows=fixed_windows(data)
    for start in range(0,len(windows),batch_size):
        y=data["labels"][start:start+batch_size]
        logits=np.asarray(scores(state,windows[start:start+batch_size]))
        shifted=logits-logits.max(axis=1,keepdims=True)
        logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        predicted=logits.argmax(axis=1)
        loss_sum+=float(-logp[np.arange(len(y)),y].sum());count+=len(y)
        correct+=int((predicted==y).sum())
        np.add.at(confusion,(y,predicted),1)
    return {"count":count,"loss_sum":loss_sum,"loss":loss_sum/count,
            "correct":correct,"accuracy":correct/count,"confusion":confusion.tolist()}


def calibrate(state,training_data):
    if dataset_hash(training_data)!=state["dataset_sha256"]:
        raise ValueError("calibration must use declared training data, not evaluation data")
    feature=features(fixed_windows(training_data))
    normalized=np.asarray((feature-state["mean"])/state["std"])
    activation_scale=max(float(np.max(np.abs(normalized)))/127,1e-8)
    W=np.asarray(state["W"]);maximum=np.max(np.abs(W),axis=0,keepdims=True)
    weight_scales=np.where(maximum==0,1.,maximum/127).astype(np.float32)
    q=np.clip(np.rint(W/weight_scales),-127,127).astype(np.int8)
    return {"qW":q,"weight_scales":weight_scales,"activation_scale":activation_scale,
            "calibration_sha256":dataset_hash(training_data)}


def policy_scores(state,waveforms,calibration,policy="fp32"):
    x=(features_core(waveforms)-state["mean"])/state["std"]
    if policy=="fp32":return x@state["W"]+state["b"]
    q=jnp.asarray(calibration["qW"]);s=jnp.asarray(calibration["weight_scales"])
    if policy=="w8a32":return x@(q.astype(jnp.float32)*s)+state["b"]
    if policy!="w8a8":raise ValueError("unknown precision policy")
    a=calibration["activation_scale"]
    qa=jnp.clip(jnp.rint(x/a),-127,127).astype(jnp.int8)
    accum=qa.astype(jnp.int32)@q.astype(jnp.int32)
    return accum.astype(jnp.float32)*(a*s)+state["b"]


def export_release(path,state,calibration):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if calibration["calibration_sha256"]!=state["dataset_sha256"]:
        raise ValueError("calibration provenance mismatch")
    artifacts={}
    for policy in ("fp32","w8a32","w8a8"):
        for batch in BATCHES:
            function=jax.jit(lambda x,policy=policy:policy_scores(state,x,calibration,policy))
            data=export.export(function)(jax.ShapeDtypeStruct((batch,1024),jnp.float32)).serialize()
            name=f"{policy}-b{batch}.jaxexport";(path/name).write_bytes(data)
            artifacts[f"{policy}:{batch}"]={"file":name,"sha256":hashlib.sha256(data).hexdigest()}
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
    (path/"release.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def load_release(path):
    path=Path(path);manifest=json.loads((path/"release.json").read_text())
    if manifest["preprocessing"]!=PREPROCESS or manifest["jax"]!=jax.__version__:
        raise ValueError("release preprocessing/runtime mismatch")
    artifacts={}
    for key,item in manifest["artifacts"].items():
        data=(path/item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("artifact checksum mismatch")
        artifacts[key]=export.deserialize(data)
    return manifest,artifacts


def infer_release(manifest,artifacts,waveforms,sample_rate=8000,channels=1,
                  policy="fp32",version="audio-stft-v1"):
    a=validate_waveforms(waveforms,sample_rate,channels,version)
    if len(a) not in manifest["batches"] or policy not in manifest["policies"]:
        raise ValueError("unsupported batch or precision policy")
    logits=artifacts[f"{policy}:{len(a)}"].call(jnp.asarray(a)).block_until_ready()
    return {"logits":np.asarray(logits),"class_ids":np.asarray(logits).argmax(axis=1)}


def benchmark(manifest,artifacts,waveforms,policy="fp32",repeats=30):
    began=time.perf_counter();infer_release(manifest,artifacts,waveforms,policy=policy)
    first_ms=(time.perf_counter()-began)*1000
    samples=[]
    for _ in range(repeats):
        began=time.perf_counter();infer_release(manifest,artifacts,waveforms,policy=policy)
        samples.append((time.perf_counter()-began)*1000)
    return {"first_request_ms":first_ms,"samples_ms":samples,
            "p50_ms":float(np.percentile(samples,50)),"p95_ms":float(np.percentile(samples,95)),
            "boundary":"validated decoded PCM + placement + exported STFT/normalization/classifier + wait + host outputs",
            "excluded":"file decoding, audio capture, window buffering, networking","device":"CPU",
            "audio_duration_ms":128.,"examples_per_second":1000*len(waveforms)/float(np.percentile(samples,50))}


def load_wav_manifest(manifest_path,split):
    """Ingest user-provided licensed PCM16/mono/8kHz files; no implicit resampling."""
    manifest_path=Path(manifest_path);rows=json.loads(manifest_path.read_text())
    groups={};ids=set()
    for row in rows:
        for field in ("id","file","label","split","group","license","source","sha256"):
            if not row.get(field) and row.get(field)!=0:raise ValueError("missing provenance field: "+field)
        if row["id"] in ids:raise ValueError("duplicate recording identity")
        ids.add(row["id"]);groups.setdefault(row["group"],set()).add(row["split"])
    if any(len(splits)>1 for splits in groups.values()):raise ValueError("recording group leaks across splits")
    output=[];labels=[];selected=[]
    for row in rows:
        if row["split"]!=split:continue
        path=manifest_path.parent/row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("WAV checksum mismatch")
        with wave.open(str(path),"rb") as wav:
            if wav.getframerate()!=8000 or wav.getnchannels()!=1 or wav.getsampwidth()!=2:
                raise ValueError("require PCM16 mono 8kHz; convert explicitly and preserve source provenance")
            samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2").astype(np.float32)/32768.
        offset=int(row.get("offset_samples",0))
        if offset<0 or len(samples)<offset+1280 or row["label"] not in (0,1,2):raise ValueError("invalid segment or label")
        output.append(samples[offset:offset+1280]);labels.append(row["label"]);selected.append(row)
    if not output:raise ValueError("requested split is empty")
    return {"waveforms":np.asarray(output,dtype=np.float32),"labels":np.asarray(labels,dtype=np.int32),
            "ids":[r["id"] for r in selected],"groups":[r["group"] for r in selected],
            "sample_rate":8000,"provenance":"user-provided WAV manifest "+str(manifest_path)}
