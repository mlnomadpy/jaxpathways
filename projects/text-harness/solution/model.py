"""Tiny causal Transformer with explicit training, cache and export contracts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

VOCAB=259
CONTEXT=12
WIDTH=24
HEADS=2
HEAD_DIM=12
TOKENIZER={"version":"utf8-byte-v1","pad_id":0,"bos_id":1,"eos_id":2,"byte_offset":3,
           "vocabulary_size":259,"context_tokens":12,"normalization":"none",
           "unknown_policy":"all UTF-8 bytes represented; overlong input rejected"}
PARAM_SHAPES={"embed":(VOCAB,WIDTH),"position":(CONTEXT,WIDTH),"q":(WIDTH,WIDTH),
              "k":(WIDTH,WIDTH),"v":(WIDTH,WIDTH),"o":(WIDTH,WIDTH),
              "ff1":(WIDTH,48),"ff2":(48,WIDTH),"head":(WIDTH,VOCAB)}


def encode(text,eos=False):
    if not isinstance(text,str):raise ValueError("text must be a string")
    ids=[1]+[int(b)+3 for b in text.encode("utf8")]
    if eos:ids.append(2)
    if len(ids)>(CONTEXT+1 if eos else CONTEXT):raise ValueError("text exceeds context; no silent truncation")
    return ids


def decode(ids):
    output=[]
    for token in ids:
        token=int(token)
        if token==2:break
        if token in (0,1):continue
        if token<3 or token>=VOCAB:raise ValueError("token outside vocabulary")
        output.append(token-3)
    return bytes(output).decode("utf8",errors="replace")


def corpus_hash(data):
    payload={"ids":data["ids"],"groups":data["groups"],"texts":data["texts"],"tokenizer":TOKENIZER}
    a=np.ascontiguousarray(data["tokens"])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def make_corpus(seed=31,count=96,split="train",exclude=()):
    rng=np.random.default_rng(seed);seen=set(exclude);texts=[]
    while len(texts)<count:
        length=int(rng.integers(2,5))
        prefix="".join(rng.choice(list("abcdef"),size=length))
        text=prefix+"|"+prefix[::-1]
        if text in seen:continue
        seen.add(text);texts.append(text)
    tokens=np.zeros((count,CONTEXT+1),dtype=np.int32)
    for i,text in enumerate(texts):
        ids=encode(text,eos=True);tokens[i,:len(ids)]=ids
    return {"tokens":tokens,"texts":texts,"ids":[f"{split}-document-{seed}-{i:04d}" for i in range(count)],
            "groups":[f"{split}-source-{seed}-{i:04d}" for i in range(count)],
            "provenance":"generated symbolic reversal strings; not a natural-language benchmark"}


def load_text_manifest(path,split):
    path=Path(path);rows=json.loads(path.read_text());seen_ids=set();seen_text={};groups={}
    texts=[];ids=[];selected_groups=[]
    for row in rows:
        for key in ("id","file","split","group","license","source","sha256"):
            if not row.get(key):raise ValueError("missing provenance field "+key)
        if row["id"] in seen_ids:raise ValueError("duplicate document identity")
        seen_ids.add(row["id"]);groups.setdefault(row["group"],set()).add(row["split"])
        raw=(path.parent/row["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row["sha256"]:raise ValueError("text checksum mismatch")
        text=raw.decode("utf8");encode(text,eos=True)
        if text in seen_text and seen_text[text]!=row["split"]:raise ValueError("duplicate content crosses splits")
        seen_text[text]=row["split"]
        if row["split"]==split:texts.append(text);ids.append(row["id"]);selected_groups.append(row["group"])
    if any(len(values)>1 for values in groups.values()):raise ValueError("source group crosses splits")
    if not texts:raise ValueError("empty requested split")
    tokens=np.zeros((len(texts),CONTEXT+1),dtype=np.int32)
    for i,text in enumerate(texts):
        encoded=encode(text,eos=True);tokens[i,:len(encoded)]=encoded
    return {"tokens":tokens,"texts":texts,"ids":ids,"groups":selected_groups,
            "provenance":"user-permitted UTF-8 documents from "+str(path)}


def layer_norm(x):
    mean=x.mean(-1,keepdims=True)
    variance=jnp.mean((x-mean)**2,-1,keepdims=True)
    return (x-mean)*jax.lax.rsqrt(variance+1e-5)


def linear(x,W,policy="fp32"):
    if policy=="fp32":return x@W
    if policy=="bf16":
        return x.astype(jnp.bfloat16).astype(jnp.float32)@W.astype(jnp.bfloat16).astype(jnp.float32)
    if policy!="w8a8":raise ValueError("unknown computation policy")
    sx=jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True)/127,1e-8)
    sw=jnp.maximum(jnp.max(jnp.abs(W),axis=0,keepdims=True)/127,1e-8)
    qx=jnp.clip(jnp.rint(x/sx),-127,127).astype(jnp.int8)
    qw=jnp.clip(jnp.rint(W/sw),-127,127).astype(jnp.int8)
    dot=qx.astype(jnp.int32)@qw.astype(jnp.int32)
    return dot.astype(jnp.float32)*sx*sw


def pack_cache(x,policy):
    if policy=="fp32":return x.astype(jnp.float32),jnp.ones(x.shape[:-1]+(1,),jnp.float32)
    if policy=="bf16":return x.astype(jnp.bfloat16),jnp.ones(x.shape[:-1]+(1,),jnp.float32)
    if policy!="int8":raise ValueError("unknown KV-cache policy")
    scale=jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True)/127,1e-8)
    return jnp.clip(jnp.rint(x/scale),-127,127).astype(jnp.int8),scale


def unpack_cache(x,scale):
    return x.astype(jnp.float32)*scale


def split_heads(x):
    return x.reshape(x.shape[0],x.shape[1],HEADS,HEAD_DIM).transpose(0,2,1,3)


def join_heads(x):
    return x.transpose(0,2,1,3).reshape(x.shape[0],x.shape[2],WIDTH)


def dropout(x,key):
    return x*jax.random.bernoulli(key,.9,x.shape).astype(x.dtype)/.9


def forward(params,tokens,key=None,compute_policy="fp32",cache_policy="fp32"):
    positions=jnp.arange(tokens.shape[1])
    x=params["embed"][tokens]+params["position"][positions][None,:,:]
    normalized=layer_norm(x)
    q=split_heads(linear(normalized,params["q"],compute_policy))
    k=split_heads(linear(normalized,params["k"],compute_policy))
    v=split_heads(linear(normalized,params["v"],compute_policy))
    stored_k,ks=pack_cache(k,cache_policy);stored_v,vs=pack_cache(v,cache_policy)
    k=unpack_cache(stored_k,ks);v=unpack_cache(stored_v,vs)
    score=jnp.einsum("bhtd,bhsd->bhts",q,k)/jnp.sqrt(float(HEAD_DIM))
    allowed=(positions[:,None]>=positions[None,:])[None,None,:,:] & (tokens!=0)[:,None,None,:]
    attention=jax.nn.softmax(jnp.where(allowed,score,-1e9),axis=-1)
    attended=linear(join_heads(jnp.einsum("bhts,bhsd->bhtd",attention,v)),params["o"],compute_policy)
    if key is not None:
        first,second=jax.random.split(key);attended=dropout(attended,first)
    x=x+attended
    hidden=jax.nn.gelu(linear(layer_norm(x),params["ff1"],compute_policy))
    residual=linear(hidden,params["ff2"],compute_policy)
    if key is not None:residual=dropout(residual,second)
    x=x+residual
    logits=linear(layer_norm(x),params["head"],compute_policy)
    return logits,attention,(stored_k,stored_v,ks,vs)


def token_loss(params,documents,key=None):
    inputs=documents[:,:-1];targets=documents[:,1:]
    logits=forward(params,inputs,key)[0]
    logp=jax.nn.log_softmax(logits,axis=-1)
    values=-jnp.take_along_axis(logp,targets[...,None],axis=-1)[...,0]
    mask=targets!=0
    return jnp.sum(jnp.where(mask,values,0.))/jnp.maximum(mask.sum(),1)


def initialize(data,seed=3,batch_size=16,rate=.006):
    if data["tokens"].shape!=(len(data["ids"]),CONTEXT+1) or len(set(data["ids"]))!=len(data["ids"]):
        raise ValueError("document shape or identity contract")
    if np.asarray(data["tokens"]).dtype.kind not in "iu" or np.any((data["tokens"]<0)|(data["tokens"]>=VOCAB)):
        raise ValueError("invalid byte token")
    if batch_size<1 or rate<=0 or not np.isfinite(rate):raise ValueError("invalid training configuration")
    for row in np.asarray(data["tokens"]):
        ends=np.flatnonzero(row==2)
        if row[0]!=1 or len(ends)!=1 or np.any(row[1:ends[0]]<3) or np.any(row[ends[0]+1:]!=0):
            raise ValueError("documents require BOS, byte content, one EOS and right padding")
    key=jax.random.PRNGKey(seed);params={}
    for name,shape in PARAM_SHAPES.items():
        key,subkey=jax.random.split(key)
        scale=.06 if name in ("embed","position") else 1/np.sqrt(shape[0])
        params[name]=jax.random.normal(subkey,shape)*scale
    key,order_key=jax.random.split(key)
    zeros=jax.tree.map(jnp.zeros_like,params)
    return {"params":params,"m":zeros,"v":jax.tree.map(jnp.zeros_like,params),
            "key":key,"order":jax.random.permutation(order_key,len(data["ids"])),
            "step":0,"cursor":0,"epoch":0,"data_sha256":corpus_hash(data),
            "config":{"seed":seed,"batch_size":batch_size,"rate":rate,
                      "beta1":.9,"beta2":.999,"epsilon":1e-8,"dropout_keep":.9}}


@jax.jit
def update(params,m,v,documents,key,count,rate):
    loss,grad=jax.value_and_grad(token_loss)(params,documents,key)
    m=jax.tree.map(lambda old,g:.9*old+.1*g,m,grad)
    v=jax.tree.map(lambda old,g:.999*old+.001*g*g,v,grad)
    params=jax.tree.map(lambda p,a,b:p-rate*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
    return params,m,v,loss


def step(state,data):
    if corpus_hash(data)!=state["data_sha256"]:raise ValueError("corpus identity changed")
    result=dict(state)
    if result["cursor"]==len(data["ids"]):
        result["key"],order_key=jax.random.split(result["key"])
        result["order"]=jax.random.permutation(order_key,len(data["ids"]))
        result["cursor"]=0;result["epoch"]+=1
    indices=np.asarray(result["order"][result["cursor"]:result["cursor"]+result["config"]["batch_size"]])
    result["key"],dropout_key=jax.random.split(result["key"])
    result["params"],result["m"],result["v"],loss=update(result["params"],result["m"],result["v"],
        jnp.asarray(data["tokens"][indices]),dropout_key,jnp.asarray(result["step"]+1),result["config"]["rate"])
    result["step"]+=1;result["cursor"]+=len(indices)
    return result,{"ids":[data["ids"][int(i)] for i in indices],"dropout_key":np.asarray(dropout_key).tolist(),
                   "pre_update_loss":float(loss),"valid_tokens":int(np.sum(data["tokens"][indices,1:]!=0)),"completed_step":result["step"]}


def save_checkpoint(path,state):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    arrays={f"{group}__{name}":np.asarray(value) for group in ("params","m","v") for name,value in state[group].items()}
    arrays.update(key=np.asarray(state["key"]),order=np.asarray(state["order"]))
    np.savez(path/"state.npz",**arrays)
    manifest={k:state[k] for k in ("step","cursor","epoch","data_sha256","config")}
    manifest.update(tokenizer=TOKENIZER,jax=jax.__version__,architecture={"width":WIDTH,"heads":HEADS,"blocks":1},
                    state_sha256=hashlib.sha256((path/"state.npz").read_bytes()).hexdigest())
    (path/"checkpoint.json").write_text(json.dumps(manifest,indent=2))


def load_checkpoint(path,data,config):
    path=Path(path);manifest=json.loads((path/"checkpoint.json").read_text())
    if manifest["data_sha256"]!=corpus_hash(data) or manifest["config"]!=config:raise ValueError("corpus/configuration mismatch")
    if manifest["tokenizer"]!=TOKENIZER or manifest["jax"]!=jax.__version__:raise ValueError("tokenizer/runtime mismatch")
    if manifest["architecture"]!={"width":WIDTH,"heads":HEADS,"blocks":1}:raise ValueError("architecture mismatch")
    if hashlib.sha256((path/"state.npz").read_bytes()).hexdigest()!=manifest["state_sha256"]:raise ValueError("checkpoint corrupt")
    with np.load(path/"state.npz",allow_pickle=False) as arrays:
        state={group:{name:jnp.asarray(arrays[f"{group}__{name}"]) for name in PARAM_SHAPES} for group in ("params","m","v")}
        state.update(key=jnp.asarray(arrays["key"]),order=jnp.asarray(arrays["order"]))
    state.update({k:manifest[k] for k in ("step","cursor","epoch","data_sha256","config")})
    return state


def evaluate(params,data,batch_size=11,compute_policy="fp32",cache_policy="fp32"):
    total=0.;count=0;correct=0
    for start in range(0,len(data["ids"]),batch_size):
        docs=data["tokens"][start:start+batch_size]
        logits=np.asarray(forward(params,jnp.asarray(docs[:,:-1]),compute_policy=compute_policy,cache_policy=cache_policy)[0])
        shifted=logits-logits.max(-1,keepdims=True)
        logp=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        targets=docs[:,1:];mask=targets!=0
        total+=float(-np.take_along_axis(logp,targets[...,None],axis=-1)[...,0][mask].sum())
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==targets)&mask).sum())
    return {"nll_sum":total,"valid_tokens":count,"nll":total/count,"token_accuracy":correct/count,
            "perplexity":float(np.exp(total/count))}


def validate_prompt(tokens):
    a=np.asarray(tokens)
    if a.ndim!=1 or len(a)<1 or len(a)>CONTEXT or a.dtype.kind not in "iu":
        raise ValueError("prompt must be a bounded one-dimensional integer token sequence")
    if a[0]!=1 or np.any(a[1:]<3) or np.any(a>=VOCAB):
        raise ValueError("prompt must begin with BOS and contain only byte tokens thereafter")
    return np.asarray(a,dtype=np.int32)


def prefill_core(params,padded,length,compute_policy="fp32",cache_policy="fp32"):
    logits,_,cache=forward(params,padded,compute_policy=compute_policy,cache_policy=cache_policy)
    return (logits[:,length-1,:],*cache,length)


def decode_core(params,token,K,V,Ks,Vs,length,compute_policy="fp32",cache_policy="fp32"):
    x=params["embed"][token][:,None,:]+params["position"][length][None,None,:]
    norm=layer_norm(x)
    q=split_heads(linear(norm,params["q"],compute_policy))
    new_k,scale_k=pack_cache(split_heads(linear(norm,params["k"],compute_policy)),cache_policy)
    new_v,scale_v=pack_cache(split_heads(linear(norm,params["v"],compute_policy)),cache_policy)
    K=jax.lax.dynamic_update_slice(K,new_k,(0,0,length,0))
    V=jax.lax.dynamic_update_slice(V,new_v,(0,0,length,0))
    Ks=jax.lax.dynamic_update_slice(Ks,scale_k,(0,0,length,0));Vs=jax.lax.dynamic_update_slice(Vs,scale_v,(0,0,length,0))
    score=jnp.einsum("bhtd,bhsd->bhts",q,unpack_cache(K,Ks))/jnp.sqrt(float(HEAD_DIM))
    allowed=jnp.arange(CONTEXT)<=length
    attention=jax.nn.softmax(jnp.where(allowed[None,None,None,:],score,-1e9),axis=-1)
    x=x+linear(join_heads(jnp.einsum("bhts,bhsd->bhtd",attention,unpack_cache(V,Vs))),params["o"],compute_policy)
    x=x+linear(jax.nn.gelu(linear(layer_norm(x),params["ff1"],compute_policy)),params["ff2"],compute_policy)
    logits=linear(layer_norm(x),params["head"],compute_policy)[:,0,:]
    return logits,K,V,Ks,Vs,length+1


def prefill(params,prompt,compute_policy="fp32",cache_policy="fp32"):
    prompt=validate_prompt(prompt);padded=np.zeros((1,CONTEXT),np.int32);padded[0,:len(prompt)]=prompt
    return prefill_core(params,jnp.asarray(padded),jnp.asarray(len(prompt),jnp.int32),compute_policy,cache_policy)


def decode_step(params,token,state,compute_policy="fp32",cache_policy="fp32"):
    if not isinstance(token,(int,np.integer)) or token<2 or token>=VOCAB:raise ValueError("invalid next token")
    if not 1<=int(state[-1])<CONTEXT:raise ValueError("KV context capacity exhausted or invalid; no wraparound")
    return decode_core(params,jnp.asarray([token],jnp.int32),*state[1:],compute_policy,cache_policy)


RELEASE_POLICIES={"fp32":("fp32","fp32"),"bf16":("bf16","bf16"),"w8a8-int8kv":("w8a8","int8")}


def export_release(path,state):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);artifacts={}
    for name,(compute,cache) in RELEASE_POLICIES.items():
        pre=jax.jit(lambda tokens,length:prefill_core(state["params"],tokens,length,compute,cache))
        signature=(jax.ShapeDtypeStruct((1,CONTEXT),jnp.int32),jax.ShapeDtypeStruct((),jnp.int32))
        pre_artifact=export.export(pre)(*signature)
        dtype={"fp32":jnp.float32,"bf16":jnp.bfloat16,"int8":jnp.int8}[cache]
        shapes=[jax.ShapeDtypeStruct((1,),jnp.int32),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,HEAD_DIM),dtype),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,HEAD_DIM),dtype),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,1),jnp.float32),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,1),jnp.float32),jax.ShapeDtypeStruct((),jnp.int32)]
        dec=jax.jit(lambda token,K,V,Ks,Vs,length:decode_core(state["params"],token,K,V,Ks,Vs,length,compute,cache))
        dec_artifact=export.export(dec)(*shapes)
        for endpoint,artifact in (("prefill",pre_artifact),("decode",dec_artifact)):
            data=artifact.serialize();filename=f"{name}-{endpoint}.jaxexport";(path/filename).write_bytes(data)
            artifacts[f"{name}:{endpoint}"]={"file":filename,"sha256":hashlib.sha256(data).hexdigest()}
    parameter_hash=hashlib.sha256(b"".join(np.asarray(state["params"][k]).tobytes() for k in PARAM_SHAPES)).hexdigest()
    manifest={"tokenizer":TOKENIZER,"architecture":{"blocks":1,"heads":HEADS,"width":WIDTH,"head_dim":HEAD_DIM},
              "training_data_sha256":state["data_sha256"],"completed_steps":state["step"],"parameter_sha256":parameter_hash,
              "jax":jax.__version__,"platform":"CPU","artifacts":artifacts,
              "precision":{"fp32":"FP32 weights/activations/dot/attention/KV/output",
              "bf16":"BF16-rounded dense operands, FP32 dot/attention/norm/softmax/output, BF16 K/V",
              "w8a8-int8kv":"dynamic per-row INT8 dense activations, per-output INT8 dense weights, INT32 dense accumulation; FP32 embedding/norm/attention/output; INT8 per-token/head K/V with FP32 scales"},
              "native_low_precision_acceleration_claimed":False}
    (path/"release.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def load_release(path):
    path=Path(path);manifest=json.loads((path/"release.json").read_text())
    if manifest["tokenizer"]!=TOKENIZER or manifest["jax"]!=jax.__version__:raise ValueError("tokenizer/runtime mismatch")
    artifacts={}
    for key,item in manifest["artifacts"].items():
        data=(path/item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item["sha256"]:raise ValueError("artifact checksum mismatch")
        artifacts[key]=export.deserialize(data)
    return manifest,artifacts


def exported_prefill(manifest,artifacts,prompt,policy="fp32"):
    if policy not in manifest["precision"]:raise ValueError("unsupported precision policy")
    prompt=validate_prompt(prompt);tokens=np.zeros((1,CONTEXT),np.int32);tokens[0,:len(prompt)]=prompt
    result=artifacts[f"{policy}:prefill"].call(jnp.asarray(tokens),jnp.asarray(len(prompt),jnp.int32))
    return jax.tree.map(lambda x:x.block_until_ready(),result)


def exported_decode(manifest,artifacts,token,state,policy="fp32"):
    if policy not in manifest["precision"] or not 1<=int(state[-1])<CONTEXT:raise ValueError("policy/context limit")
    if not isinstance(token,(int,np.integer)) or token<2 or token>=VOCAB:raise ValueError("invalid next token")
    result=artifacts[f"{policy}:decode"].call(jnp.asarray([token],jnp.int32),*state[1:])
    return jax.tree.map(lambda x:x.block_until_ready(),result)


def generate(manifest,artifacts,text,max_new_tokens=4,policy="fp32"):
    prompt=encode(text)
    if max_new_tokens<0 or len(prompt)+max_new_tokens>CONTEXT:raise ValueError("generation exceeds context budget")
    state=exported_prefill(manifest,artifacts,prompt,policy);generated=[]
    for _ in range(max_new_tokens):
        logits=np.asarray(state[0])[0].copy();logits[:2]=-np.inf
        token=int(logits.argmax());generated.append(token)
        if token==2:break
        state=exported_decode(manifest,artifacts,token,state,policy)
    return {"prompt":text,"generated_token_ids":generated,"generated_text":decode(generated),
            "finished_eos":bool(generated and generated[-1]==2),"policy":policy}


def benchmark(manifest,artifacts,prompt,policy="fp32",repeats=30):
    tokens=encode(prompt)
    if len(tokens)>=CONTEXT:raise ValueError("one free cache slot needed for decode measurement")
    started=time.perf_counter();cache=exported_prefill(manifest,artifacts,tokens,policy)
    first_prefill=(time.perf_counter()-started)*1000
    token=ord("a")+3
    started=time.perf_counter();exported_decode(manifest,artifacts,token,cache,policy)
    first_decode=(time.perf_counter()-started)*1000
    pre=[];dec=[]
    for _ in range(repeats):
        started=time.perf_counter();exported_prefill(manifest,artifacts,tokens,policy)
        pre.append((time.perf_counter()-started)*1000)
        started=time.perf_counter();exported_decode(manifest,artifacts,token,cache,policy)
        dec.append((time.perf_counter()-started)*1000)
    return {"prompt_tokens":len(tokens),"first_prefill_ms":first_prefill,"first_decode_ms":first_decode,
            "prefill_samples_ms":pre,"decode_samples_ms":dec,
            "prefill_p50_ms":float(np.percentile(pre,50)),"decode_p50_ms":float(np.percentile(dec,50)),
            "decode_p95_ms":float(np.percentile(dec,95)),"single_step_tokens_per_second":1000/float(np.percentile(dec,50)),
            "cache_bytes_including_scales":sum(np.asarray(x).nbytes for x in cache[1:5]),
            "boundary":"validated token inputs + exported call + complete returned arrays; no networking",
            "decode_workload":"one token from the same fixed prefix cache per timed repetition","device":"CPU"}
