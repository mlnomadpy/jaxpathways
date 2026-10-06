"""Cumulative independent CPU checks: Transformer, state, cache and serialized runtime."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--implementation",default="starter")
parser.add_argument("--stage",choices=["1","2","3","4","all"],default="all")
parser.add_argument("--resume-worker");parser.add_argument("--worker-output")
args=parser.parse_args()
path=(ROOT/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)).resolve()
spec=importlib.util.spec_from_file_location("learner_text",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
stage=4 if args.stage=="all" else int(args.stage)

def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError("Expected a clear input/contract ValueError")

def same_state(a,b):
    for group in ("params","m","v"):
        for name in a[group]:np.testing.assert_array_equal(a[group][name],b[group][name])
    for key in ("key","order"):np.testing.assert_array_equal(a[key],b[key])
    for key in ("step","cursor","epoch"):assert a[key]==b[key]

train=m.make_corpus(31,96,"train")
if args.resume_worker:
    config=json.loads((Path(args.resume_worker)/"checkpoint.json").read_text())["config"]
    state=m.load_checkpoint(args.resume_worker,train,config);trace=[]
    for _ in range(4):
        state,item=m.step(state,train);trace.append(item)
    m.save_checkpoint(args.worker_output,state)
    (Path(args.worker_output)/"trace.json").write_text(json.dumps(trace))
    raise SystemExit(0)

held=m.make_corpus(89,32,"held",exclude=train["texts"])
assert not set(train["texts"])&set(held["texts"])
assert not set(train["ids"])&set(held["ids"])
encoded=m.encode("éa");assert encoded==[1,198,172,100] and m.decode(encoded)=="éa"
rejects(lambda:m.encode("x"*12))
rejects(lambda:m.validate_prompt([1,0,100]))
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);(root/"a.txt").write_text("ab|ba")
    row={"id":"external-1","file":"a.txt","split":"train","group":"source-1","license":"owned fixture",
         "source":"generated local test","sha256":hashlib.sha256((root/"a.txt").read_bytes()).hexdigest()}
    manifest=root/"documents.json";manifest.write_text(json.dumps([row]))
    actual=m.load_text_manifest(manifest,"train")
    assert actual["texts"]==["ab|ba"]
    manifest.write_text(json.dumps([row,dict(row,id="other",split="held")]))
    rejects(lambda:m.load_text_manifest(manifest,"train"))
# Independent layer-norm, attention row and causal/padding invariants.
initial=m.initialize(train,seed=3)
tokens=jnp.asarray(train["tokens"][:2,:-1])
logits,attention,_=m.forward(initial["params"],tokens)
changed=tokens.at[:,5:].set(110)
np.testing.assert_allclose(m.forward(initial["params"],changed)[0][:,:5],logits[:,:5],atol=1e-6)
assert np.max(np.triu(np.asarray(attention),k=1))==0
params=initial["params"];host=np.asarray(params["embed"])[np.asarray(tokens)]+np.asarray(params["position"])[None,:,:]
norm=(host-host.mean(-1,keepdims=True))/np.sqrt(np.mean((host-host.mean(-1,keepdims=True))**2,-1,keepdims=True)+1e-5)
Q=(norm@np.asarray(params["q"])).reshape(2,12,2,12).transpose(0,2,1,3)
K=(norm@np.asarray(params["k"])).reshape(2,12,2,12).transpose(0,2,1,3)
row=Q[0,0,3]@K[0,0,:4].T/np.sqrt(12.)
p=np.exp(row-row.max());p/=p.sum()
np.testing.assert_allclose(np.asarray(attention)[0,0,3,:4],p,atol=2e-6,rtol=2e-5)
targets=np.asarray(train["tokens"][:2,1:]);raw=np.asarray(logits);shift=raw-raw.max(-1,keepdims=True)
lp=shift-np.log(np.exp(shift).sum(-1,keepdims=True));mask=targets!=0
expected=-np.take_along_axis(lp,targets[...,None],axis=-1)[...,0][mask].mean()
np.testing.assert_allclose(m.token_loss(params,jnp.asarray(train["tokens"][:2])),expected,rtol=1e-6)
print("PASS stage 1: UTF-8 bytes, split/content provenance, independent attention and masked token loss, causal invariance")

if stage>=2:
    documents=jnp.asarray(train["tokens"][:3])
    grad=jax.grad(m.token_loss)(params,documents)["head"][0,100]
    epsilon=.002
    plus=dict(params,head=params["head"].at[0,100].add(epsilon))
    minus=dict(params,head=params["head"].at[0,100].add(-epsilon))
    finite=(m.token_loss(plus,documents)-m.token_loss(minus,documents))/(2*epsilon)
    np.testing.assert_allclose(grad,finite,atol=3e-4,rtol=.03)
    state=initial
    for _ in range(5):state,_=m.step(state,train)
    with tempfile.TemporaryDirectory() as folder:
        saved=Path(folder)/"saved";fresh=Path(folder)/"fresh"
        m.save_checkpoint(saved,state)
        expected=state;trace=[]
        for _ in range(4):
            expected,item=m.step(expected,train);trace.append(item)
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--implementation",str(path),
                               "--resume-worker",str(saved),"--worker-output",str(fresh)],
                              capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
        restored=m.load_checkpoint(fresh,train,state["config"]);same_state(expected,restored)
        assert trace==json.loads((fresh/"trace.json").read_text())
        rejects(lambda:m.load_checkpoint(saved,held,state["config"]))
        rejects(lambda:m.load_checkpoint(saved,train,dict(state["config"],rate=.01)))
    for _ in range(235):state,_=m.step(state,train)
    report=m.evaluate(state["params"],held)
    assert report["nll"]<np.log(259)*.55
    whole=m.evaluate(state["params"],held,batch_size=32)
    np.testing.assert_allclose(report["nll"],whole["nll"],rtol=1e-5,atol=1e-6)
    assert report["valid_tokens"]==whole["valid_tokens"]
    before={k:np.array(v,copy=True) for k,v in state["params"].items()}
    m.evaluate(state["params"],held)
    for key,value in before.items():np.testing.assert_array_equal(value,state["params"][key])
    changed_seed=m.initialize(train,seed=7)
    for _ in range(120):changed_seed,_=m.step(changed_seed,train)
    changed_report=m.evaluate(changed_seed["params"],held)
    assert changed_report["nll"]<np.log(259)*.6
    print("PASS stage 2: derivative check, full Adam/dropout/iterator fresh-process recovery, actual training and fixed token evaluation")
    print("Measured held-out token metrics:",report,"changed-seed NLL:",changed_report["nll"])

if stage>=3:
    params=state["params"];prompt=m.encode("abc|")
    for cache_policy in ("fp32","bf16","int8"):
        current=m.prefill(params,prompt,cache_policy=cache_policy)
        expected_dtype={"fp32":jnp.float32,"bf16":jnp.bfloat16,"int8":jnp.int8}[cache_policy]
        assert current[1].dtype==expected_dtype and current[2].dtype==expected_dtype
        prefix=list(prompt)
        for char in "cba":
            token=ord(char)+3;current=m.decode_step(params,token,current,cache_policy=cache_policy);prefix.append(token)
            padded=np.zeros((1,12),np.int32);padded[0,:len(prefix)]=prefix
            full=m.forward(params,jnp.asarray(padded),cache_policy=cache_policy)[0][:,len(prefix)-1,:]
            np.testing.assert_allclose(current[0],full,rtol=5e-5,atol=5e-5)
        # A populated prefix cache must stay unchanged when a new token is appended.
        first=m.prefill(params,prompt,cache_policy=cache_policy)
        previous=np.asarray(first[1]).copy()
        next_state=m.decode_step(params,ord("c")+3,first,cache_policy=cache_policy)
        np.testing.assert_array_equal(next_state[1][:,:,:len(prompt)],previous[:,:,:len(prompt)])
    full=m.prefill(params,[1]+[100]*11)
    rejects(lambda:m.decode_step(params,100,full))
    # Cache INT8 reconstruction uses a different scale per token and head.
    values=jnp.asarray(np.linspace(-2,3,48).reshape(1,2,2,12),jnp.float32)
    q,scales=m.pack_cache(values,"int8")
    assert q.dtype==jnp.int8 and scales.shape==(1,2,2,1)
    np.testing.assert_array_less(np.abs(np.asarray(q)*np.asarray(scales)-np.asarray(values)),
                                 np.broadcast_to(np.asarray(scales)/2+1e-6,values.shape))
    # Independent dynamic integer dense arithmetic.
    X=np.array([[.2,-.8,.4],[1.,.3,-.5]],np.float32)
    W=np.array([[.3,-.4],[.2,.8],[-.7,.5]],np.float32)
    sx=np.maximum(np.abs(X).max(-1,keepdims=True)/127,1e-8)
    sw=np.maximum(np.abs(W).max(0,keepdims=True)/127,1e-8)
    oracle=(np.rint(X/sx).astype(np.int64)@np.rint(W/sw).astype(np.int64)).astype(np.float32)*sx*sw
    np.testing.assert_allclose(m.linear(jnp.asarray(X),jnp.asarray(W),"w8a8"),oracle,rtol=1e-6,atol=1e-6)
    print("PASS stage 3: actual FP32/BF16/INT8 KV storage, sequential cached/full parity, immutable prefix, overflow and independent integer arithmetic")

if stage>=4:
    with tempfile.TemporaryDirectory() as folder:
        release=Path(folder)/"release";manifest=m.export_release(release,state);loaded,artifacts=m.load_release(release)
        assert len(artifacts)==6
        for name,(compute,cache_policy) in m.RELEASE_POLICIES.items():
            first=m.exported_prefill(loaded,artifacts,prompt,name)
            reference=m.prefill(params,prompt,compute,cache_policy)
            np.testing.assert_allclose(first[0],reference[0],atol=2e-4,rtol=2e-4)
            nxt=m.exported_decode(loaded,artifacts,102,first,name)
            refnext=m.decode_step(params,102,reference,compute,cache_policy)
            np.testing.assert_allclose(nxt[0],refnext[0],atol=2e-4,rtol=2e-4)
        output=Path(folder)/"fresh.npy"
        code="import importlib.util,numpy as np,sys;s=importlib.util.spec_from_file_location('m',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);a,b=m.load_release(sys.argv[2]);c=m.exported_prefill(a,b,m.encode('abc|'));d=m.exported_decode(a,b,102,c);np.save(sys.argv[3],np.asarray(d[0]))"
        result=subprocess.run([sys.executable,"-c",code,str(path),str(release),str(output)],capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
        fp=m.exported_prefill(loaded,artifacts,prompt)
        np.testing.assert_allclose(np.load(output),m.exported_decode(loaded,artifacts,102,fp)[0],rtol=1e-6,atol=1e-6)
        item=next(iter(manifest["artifacts"].values()));artifact=release/item["file"];original=artifact.read_bytes()
        artifact.write_bytes(original+b"corrupt");rejects(lambda:m.load_release(release));artifact.write_bytes(original)
        rejects(lambda:m.generate(loaded,artifacts,"abc|",max_new_tokens=12))
        with jax.profiler.trace(str(Path(folder)/"profile"),create_perfetto_link=False,create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation("text-prefill-check"):
                profiled=m.exported_prefill(loaded,artifacts,prompt)
            with jax.profiler.TraceAnnotation("text-decode-check"):
                m.exported_decode(loaded,artifacts,102,profiled)
        assert list((Path(folder)/"profile").rglob("*.xplane.pb")),"actual profiler trace required"
        timed=m.benchmark(loaded,artifacts,"abc|",repeats=12)
        assert len(timed["prefill_samples_ms"])==12 and min(timed["decode_samples_ms"])>0
        np.testing.assert_allclose(timed["decode_p50_ms"],np.percentile(timed["decode_samples_ms"],50))
        generated=m.generate(loaded,artifacts,"abc|",max_new_tokens=4)
        print("PASS stage 4: six serialized endpoints, fresh-process inference, corruption rejection, bounded generation and actual prefill/decode timing")
        print("Actual generated output:",generated)
        print("Measured CPU prefill/decode p50 milliseconds:",timed["prefill_p50_ms"],timed["decode_p50_ms"])
print("All requested text stages passed. Symbolic byte-sequence fixture and CPU evidence only.")
