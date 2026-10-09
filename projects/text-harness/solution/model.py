"""Tiny causal Transformer with explicit training, cache and export contracts."""
# Import hashlib for this computation.
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

# Evaluate `VOCAB` from the current inputs and state.
VOCAB=259
# Evaluate `CONTEXT` from the current inputs and state.
CONTEXT=12
# Evaluate `WIDTH` from the current inputs and state.
WIDTH=24
# Evaluate `HEADS` from the current inputs and state.
HEADS=2
# Evaluate `HEAD_DIM` from the current inputs and state.
HEAD_DIM=12
# Combine or mask array elements to form `TOKENIZER`.
TOKENIZER={"version":"utf8-byte-v1","pad_id":0,"bos_id":1,"eos_id":2,"byte_offset":3,
           "vocabulary_size":259,"context_tokens":12,"normalization":"none",
           "unknown_policy":"all UTF-8 bytes represented; overlong input rejected"}
# Evaluate `PARAM_SHAPES` from the current inputs and state.
PARAM_SHAPES={"embed":(VOCAB,WIDTH),"position":(CONTEXT,WIDTH),"q":(WIDTH,WIDTH),
              "k":(WIDTH,WIDTH),"v":(WIDTH,WIDTH),"o":(WIDTH,WIDTH),
              "ff1":(WIDTH,48),"ff2":(48,WIDTH),"head":(WIDTH,VOCAB)}


# Function `encode(text, eos)` implementing this stage's computation:
def encode(text,eos=False):
    # Guard input contract (`not isinstance(text, str)`) and fail fast if violated.
    if not isinstance(text,str):raise ValueError("text must be a string")
    # Evaluate `ids` from the current inputs and state.
    ids=[1]+[int(b)+3 for b in text.encode("utf8")]
    # Branch on condition `eos`:
    if eos:ids.append(2)
    # Guard input contract (`len(ids) > (CONTEXT + 1 if eos else CONTEXT)`) and fail fast if violated.
    if len(ids)>(CONTEXT+1 if eos else CONTEXT):raise ValueError("text exceeds context; no silent truncation")
    # Return `ids` to the caller.
    return ids


# Function `decode(ids)` implementing this stage's computation:
def decode(ids):
    # Evaluate `output` from the current inputs and state.
    output=[]
    # Iterate over `token` to step through the computation:
    for token in ids:
        # Evaluate `token` and convert the result into Python scalar/collection `token`.
        token=int(token)
        # Branch on condition `token == 2`:
        if token==2:break
        # Branch on condition `token in (0, 1)`:
        if token in (0,1):continue
        # Guard input contract (`token < 3 or token >= VOCAB`) and fail fast if violated.
        if token<3 or token>=VOCAB:raise ValueError("token outside vocabulary")
        # Append the current step result to `output`.
        output.append(token-3)
    # Return `bytes(output).decode('utf8', errors='replace')` to the caller.
    return bytes(output).decode("utf8",errors="replace")


# Function `corpus_hash(data)` implementing this stage's computation:
def corpus_hash(data):
    # Evaluate `payload` from the current inputs and state.
    payload={"ids":data["ids"],"groups":data["groups"],"texts":data["texts"],"tokenizer":TOKENIZER}
    # Run `np.ascontiguousarray` to compute `a`.
    a=np.ascontiguousarray(data["tokens"])
    # Return `hashlib.sha256(json.dumps(payload, sort_keys=True).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()` to the caller.
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


# Function `make_corpus(seed, count, split, exclude)` implementing this stage's computation:
def make_corpus(seed=31,count=96,split="train",exclude=()):
    # Draw pseudorandom samples for `rng` using the explicit RNG state.
    # Run `set` to compute `seen`.
    # Evaluate `texts` from the current inputs and state.
    rng=np.random.default_rng(seed);seen=set(exclude);texts=[]
    while len(texts)<count:
        length=int(rng.integers(2,5))
        prefix="".join(rng.choice(list("abcdef"),size=length))
        text=prefix+"|"+prefix[::-1]
        if text in seen:continue
        seen.add(text);texts.append(text)
    # Allocate initialized array `tokens` with the specified shape and dtype.
    tokens=np.zeros((count,CONTEXT+1),dtype=np.int32)
    # Iterate over `(i, text)` to step through the computation:
    for i,text in enumerate(texts):
        # Run `encode` to compute `ids`.
        # Evaluate `tokens[i, :len(ids)]` from the current inputs and state.
        ids=encode(text,eos=True);tokens[i,:len(ids)]=ids
    # Return `{'tokens': tokens, 'texts': texts, 'ids': [f'{split}-document-{seed}-{i:04d}' for i in range(count)], 'groups': [f'{split}-source-{seed}-{i:04d}' for i in range(count)], 'provenance': 'generated symbolic reversal strings; not a natural-language benchmark'}` to the caller.
    return {"tokens":tokens,"texts":texts,"ids":[f"{split}-document-{seed}-{i:04d}" for i in range(count)],
            "groups":[f"{split}-source-{seed}-{i:04d}" for i in range(count)],
            "provenance":"generated symbolic reversal strings; not a natural-language benchmark"}


# Function `load_text_manifest(path, split)` implementing this stage's computation:
def load_text_manifest(path,split):
    # Read or serialize artifact data on disk (`path`).
    # Read or serialize artifact data on disk (`rows`).
    # Run `set` to compute `seen_ids`.
    # Evaluate `seen_text` from the current inputs and state.
    # Evaluate `groups` from the current inputs and state.
    path=Path(path);rows=json.loads(path.read_text());seen_ids=set();seen_text={};groups={}
    # Evaluate `texts` from the current inputs and state.
    # Evaluate `ids` from the current inputs and state.
    # Evaluate `selected_groups` from the current inputs and state.
    texts=[];ids=[];selected_groups=[]
    # Iterate over `row` to step through the computation:
    for row in rows:
        # Iterate over `key` to step through the computation:
        for key in ("id","file","split","group","license","source","sha256"):
            # Guard input contract (`not row.get(key)`) and fail fast if violated.
            if not row.get(key):raise ValueError("missing provenance field "+key)
        # Guard input contract (`row['id'] in seen_ids`) and fail fast if violated.
        if row["id"] in seen_ids:raise ValueError("duplicate document identity")
        # Run `seen_ids.add` to perform the next check or state transition.
        # Run `seen_ids.add` to perform the next check or state transition.
        seen_ids.add(row["id"]);groups.setdefault(row["group"],set()).add(row["split"])
        # Evaluate `raw` from the current inputs and state.
        raw=(path.parent/row["file"]).read_bytes()
        # Guard input contract (`hashlib.sha256(raw).hexdigest() != row['sha256']`) and fail fast if violated.
        if hashlib.sha256(raw).hexdigest()!=row["sha256"]:raise ValueError("text checksum mismatch")
        # Run `raw.decode` to compute `text`.
        # Execute the next step of the computation.
        text=raw.decode("utf8");encode(text,eos=True)
        # Guard input contract (`text in seen_text and seen_text[text] != row['split']`) and fail fast if violated.
        if text in seen_text and seen_text[text]!=row["split"]:raise ValueError("duplicate content crosses splits")
        # Evaluate `seen_text[text]` from the current inputs and state.
        seen_text[text]=row["split"]
        # Branch on condition `row['split'] == split`:
        if row["split"]==split:texts.append(text);ids.append(row["id"]);selected_groups.append(row["group"])
    # Guard input contract (`any((len(values) > 1 for values in groups.values()))`) and fail fast if violated.
    if any(len(values)>1 for values in groups.values()):raise ValueError("source group crosses splits")
    # Guard input contract (`not texts`) and fail fast if violated.
    if not texts:raise ValueError("empty requested split")
    # Allocate initialized array `tokens` with the specified shape and dtype.
    tokens=np.zeros((len(texts),CONTEXT+1),dtype=np.int32)
    # Loop over `(i, text)` in `enumerate(texts)`:
    for i,text in enumerate(texts):
        # Run `encode` to compute `encoded`.
        # Evaluate `tokens[i, :len(encoded)]` from the current inputs and state.
        encoded=encode(text,eos=True);tokens[i,:len(encoded)]=encoded
    # Return `{'tokens': tokens, 'texts': texts, 'ids': ids, 'groups': selected_groups, 'provenance': 'user-permitted UTF-8 documents from ' + str(path)}` to the caller.
    return {"tokens":tokens,"texts":texts,"ids":ids,"groups":selected_groups,
            "provenance":"user-permitted UTF-8 documents from "+str(path)}


# Function `layer_norm(x)` implementing this stage's computation:
def layer_norm(x):
    # Aggregate array values to compute `mean`.
    mean=x.mean(-1,keepdims=True)
    # Aggregate array values to compute `variance`.
    variance=jnp.mean((x-mean)**2,-1,keepdims=True)
    # Return `(x - mean) * jax.lax.rsqrt(variance + 1e-05)` to the caller.
    return (x-mean)*jax.lax.rsqrt(variance+1e-5)


# Function `linear(x, W, policy)` implementing this stage's computation:
def linear(x,W,policy="fp32"):
    # Branch on condition `policy == 'fp32'`:
    if policy=="fp32":return x@W
    # Branch on condition `policy == 'bf16'`:
    if policy=="bf16":
        return x.astype(jnp.bfloat16).astype(jnp.float32)@W.astype(jnp.bfloat16).astype(jnp.float32)
    # Guard input contract (`policy != 'w8a8'`) and fail fast if violated.
    if policy!="w8a8":raise ValueError("unknown computation policy")
    # Reduce along axis=-1 to compute `sx`.
    sx=jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True)/127,1e-8)
    # Reduce along axis=0 to compute `sw`.
    sw=jnp.maximum(jnp.max(jnp.abs(W),axis=0,keepdims=True)/127,1e-8)
    # Combine or mask array elements to form `qx`.
    qx=jnp.clip(jnp.rint(x/sx),-127,127).astype(jnp.int8)
    # Combine or mask array elements to form `qw`.
    qw=jnp.clip(jnp.rint(W/sw),-127,127).astype(jnp.int8)
    # Perform matrix contraction / projection to compute `dot`.
    dot=qx.astype(jnp.int32)@qw.astype(jnp.int32)
    # Return `dot.astype(jnp.float32) * sx * sw` to the caller.
    return dot.astype(jnp.float32)*sx*sw


# Function `pack_cache(x, policy)` implementing this stage's computation:
def pack_cache(x,policy):
    # Branch on condition `policy == 'fp32'`:
    if policy=="fp32":return x.astype(jnp.float32),jnp.ones(x.shape[:-1]+(1,),jnp.float32)
    # Branch on condition `policy == 'bf16'`:
    if policy=="bf16":return x.astype(jnp.bfloat16),jnp.ones(x.shape[:-1]+(1,),jnp.float32)
    # Guard input contract (`policy != 'int8'`) and fail fast if violated.
    if policy!="int8":raise ValueError("unknown KV-cache policy")
    # Reduce across the target axis to summarize `scale`.
    scale=jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True)/127,1e-8)
    # Return `(jnp.clip(jnp.rint(x / scale), -127, 127).astype(jnp.int8), scale)` to the caller.
    return jnp.clip(jnp.rint(x/scale),-127,127).astype(jnp.int8),scale


# Function `unpack_cache(x, scale)` implementing this stage's computation:
def unpack_cache(x,scale):
    # Return `x.astype(jnp.float32) * scale` to the caller.
    return x.astype(jnp.float32)*scale


# Function `split_heads(x)` implementing this stage's computation:
def split_heads(x):
    # Return `x.reshape(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(0, 2, 1, 3)` to the caller.
    return x.reshape(x.shape[0],x.shape[1],HEADS,HEAD_DIM).transpose(0,2,1,3)


# Function `join_heads(x)` implementing this stage's computation:
def join_heads(x):
    # Return `x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], WIDTH)` to the caller.
    return x.transpose(0,2,1,3).reshape(x.shape[0],x.shape[2],WIDTH)


# Function `dropout(x, key)` implementing this stage's computation:
def dropout(x,key):
    # Return `x * jax.random.bernoulli(key, 0.9, x.shape).astype(x.dtype) / 0.9` to the caller.
    return x*jax.random.bernoulli(key,.9,x.shape).astype(x.dtype)/.9


# Function `forward(params, tokens, key, compute_policy, ...)` implementing this stage's computation:
def forward(params,tokens,key=None,compute_policy="fp32",cache_policy="fp32"):
    # Create evenly spaced index values in `positions`.
    positions=jnp.arange(tokens.shape[1])
    # Evaluate `x` from the current inputs and state.
    x=params["embed"][tokens]+params["position"][positions][None,:,:]
    # Run `layer_norm` to compute `normalized`.
    normalized=layer_norm(x)
    # Run `split_heads` to compute `q`.
    q=split_heads(linear(normalized,params["q"],compute_policy))
    # Run `split_heads` to compute `k`.
    k=split_heads(linear(normalized,params["k"],compute_policy))
    # Run `split_heads` to compute `v`.
    v=split_heads(linear(normalized,params["v"],compute_policy))
    # Run `pack_cache` to compute `(stored_k, ks)`.
    # Run `pack_cache` to compute `(stored_v, vs)`.
    stored_k,ks=pack_cache(k,cache_policy);stored_v,vs=pack_cache(v,cache_policy)
    # Run `unpack_cache` to compute `k`.
    # Run `unpack_cache` to compute `v`.
    k=unpack_cache(stored_k,ks);v=unpack_cache(stored_v,vs)
    # Perform matrix contraction / projection to compute `score`.
    score=jnp.einsum("bhtd,bhsd->bhts",q,k)/jnp.sqrt(float(HEAD_DIM))
    # Evaluate `allowed` from the current inputs and state.
    allowed=(positions[:,None]>=positions[None,:])[None,None,:,:] & (tokens!=0)[:,None,None,:]
    # Apply nonlinear activation or probability normalization to compute `attention`.
    attention=jax.nn.softmax(jnp.where(allowed,score,-1e9),axis=-1)
    # Perform matrix contraction / projection to compute `attended`.
    attended=linear(join_heads(jnp.einsum("bhts,bhsd->bhtd",attention,v)),params["o"],compute_policy)
    # Branch on condition `key is not None`:
    if key is not None:
        first,second=jax.random.split(key);attended=dropout(attended,first)
    # Evaluate `x` from the current inputs and state.
    x=x+attended
    # Apply nonlinear activation or probability normalization to compute `hidden`.
    hidden=jax.nn.gelu(linear(layer_norm(x),params["ff1"],compute_policy))
    # Run `linear` to compute `residual`.
    residual=linear(hidden,params["ff2"],compute_policy)
    # Branch on condition `key is not None`:
    if key is not None:residual=dropout(residual,second)
    # Evaluate `x` from the current inputs and state.
    x=x+residual
    # Run `linear` to compute `logits`.
    logits=linear(layer_norm(x),params["head"],compute_policy)
    # Return `(logits, attention, (stored_k, stored_v, ks, vs))` to the caller.
    return logits,attention,(stored_k,stored_v,ks,vs)


# Function `token_loss(params, documents, key)` implementing this stage's computation:
def token_loss(params,documents,key=None):
    # Evaluate `inputs` from the current inputs and state.
    # Evaluate `targets` from the current inputs and state.
    inputs=documents[:,:-1];targets=documents[:,1:]
    # Run `forward` to compute `logits`.
    logits=forward(params,inputs,key)[0]
    # Evaluate numerically stable log-space cross-entropy/likelihood (`logp`).
    logp=jax.nn.log_softmax(logits,axis=-1)
    # Evaluate `values` from the current inputs and state.
    values=-jnp.take_along_axis(logp,targets[...,None],axis=-1)[...,0]
    # Evaluate `mask` from the current inputs and state.
    mask=targets!=0
    # Return `jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(mask.sum(), 1)` to the caller.
    return jnp.sum(jnp.where(mask,values,0.))/jnp.maximum(mask.sum(),1)


# Function `initialize(data, seed, batch_size, rate)` implementing this stage's computation:
def initialize(data,seed=3,batch_size=16,rate=.006):
    # Guard input contract (`data['tokens'].shape != (len(data['ids']), CONTEXT + 1) or len(set(data['ids'])) != len(data['ids'])`) and fail fast if violated.
    if data["tokens"].shape!=(len(data["ids"]),CONTEXT+1) or len(set(data["ids"]))!=len(data["ids"]):
        raise ValueError("document shape or identity contract")
    # Guard input contract (`np.asarray(data['tokens']).dtype.kind not in 'iu' or np.any((data['tokens'] < 0) | (data['tokens'] >= VOCAB))`) and fail fast if violated.
    if np.asarray(data["tokens"]).dtype.kind not in "iu" or np.any((data["tokens"]<0)|(data["tokens"]>=VOCAB)):
        raise ValueError("invalid byte token")
    # Guard input contract (`batch_size < 1 or rate <= 0 or (not np.isfinite(rate))`) and fail fast if violated.
    if batch_size<1 or rate<=0 or not np.isfinite(rate):raise ValueError("invalid training configuration")
    # Loop over `row` in `np.asarray(data['tokens'])`:
    for row in np.asarray(data["tokens"]):
        # Run `np.flatnonzero` to compute `ends`.
        ends=np.flatnonzero(row==2)
        # Guard input contract (`row[0] != 1 or len(ends) != 1 or np.any(row[1:ends[0]] < 3) or np.any(row[ends[0] + 1:] != 0)`) and fail fast if violated.
        if row[0]!=1 or len(ends)!=1 or np.any(row[1:ends[0]]<3) or np.any(row[ends[0]+1:]!=0):
            raise ValueError("documents require BOS, byte content, one EOS and right padding")
    # Initialize explicit deterministic PRNG key `key`.
    # Evaluate `params` from the current inputs and state.
    key=jax.random.PRNGKey(seed);params={}
    # Loop over `(name, shape)` in `PARAM_SHAPES.items()`:
    for name,shape in PARAM_SHAPES.items():
        # Split the PRNG key deterministically into independent subkeys (`(key, subkey)`).
        key,subkey=jax.random.split(key)
        # Evaluate `scale` from the current inputs and state.
        scale=.06 if name in ("embed","position") else 1/np.sqrt(shape[0])
        # Draw pseudorandom samples for `params[name]` using the explicit RNG state.
        params[name]=jax.random.normal(subkey,shape)*scale
    # Split the PRNG key deterministically into independent subkeys (`(key, order_key)`).
    key,order_key=jax.random.split(key)
    # Allocate initialized array `zeros` with the specified shape and dtype.
    zeros=jax.tree.map(jnp.zeros_like,params)
    # Return `{'params': params, 'm': zeros, 'v': jax.tree.map(jnp.zeros_like, params), 'key': key, 'order': jax.random.permutation(order_key, len(data['ids'])), 'step': 0, 'cursor': 0, 'epoch': 0, 'data_sha256': corpus_hash(data), 'config': {'seed': seed, 'batch_size': batch_size, 'rate': rate, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08, 'dropout_keep': 0.9}}` to the caller.
    return {"params":params,"m":zeros,"v":jax.tree.map(jnp.zeros_like,params),
            "key":key,"order":jax.random.permutation(order_key,len(data["ids"])),
            "step":0,"cursor":0,"epoch":0,"data_sha256":corpus_hash(data),
            "config":{"seed":seed,"batch_size":batch_size,"rate":rate,
                      "beta1":.9,"beta2":.999,"epsilon":1e-8,"dropout_keep":.9}}


@jax.jit
# Function `update(params, m, v, documents, ...)` implementing this stage's computation:
def update(params,m,v,documents,key,count,rate):
    # Evaluate both scalar loss and parameter gradients in one pass (`(loss, grad)`).
    loss,grad=jax.value_and_grad(token_loss)(params,documents,key)
    # Apply leaf-wise transformation across the PyTree to produce `m`.
    m=jax.tree.map(lambda old,g:.9*old+.1*g,m,grad)
    # Apply leaf-wise transformation across the PyTree to produce `v`.
    v=jax.tree.map(lambda old,g:.999*old+.001*g*g,v,grad)
    # Apply leaf-wise transformation across the PyTree to produce `params`.
    params=jax.tree.map(lambda p,a,b:p-rate*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
    # Return `(params, m, v, loss)` to the caller.
    return params,m,v,loss


# Function `step(state, data)` implementing this stage's computation:
def step(state,data):
    # Guard input contract (`corpus_hash(data) != state['data_sha256']`) and fail fast if violated.
    if corpus_hash(data)!=state["data_sha256"]:raise ValueError("corpus identity changed")
    # Evaluate `state` and convert the result into Python scalar/collection `result`.
    result=dict(state)
    # Branch on condition `result['cursor'] == len(data['ids'])`:
    if result["cursor"]==len(data["ids"]):
        result["key"],order_key=jax.random.split(result["key"])
        result["order"]=jax.random.permutation(order_key,len(data["ids"]))
        result["cursor"]=0;result["epoch"]+=1
    # Convert `indices` to a host NumPy array for inspection or verification.
    indices=np.asarray(result["order"][result["cursor"]:result["cursor"]+result["config"]["batch_size"]])
    # Split the PRNG key deterministically into independent subkeys (`(result['key'], dropout_key)`).
    result["key"],dropout_key=jax.random.split(result["key"])
    # Create device-backed JAX array `(result['params'], result['m'], result['v'], loss)`.
    result["params"],result["m"],result["v"],loss=update(result["params"],result["m"],result["v"],
        jnp.asarray(data["tokens"][indices]),dropout_key,jnp.asarray(result["step"]+1),result["config"]["rate"])
    # Accumulate the next contribution into `result['step']`.
    # Accumulate the next contribution into `result['cursor']`.
    result["step"]+=1;result["cursor"]+=len(indices)
    # Return `(result, {'ids': [data['ids'][int(i)] for i in indices], 'dropout_key': np.asarray(dropout_key).tolist(), 'pre_update_loss': float(loss), 'valid_tokens': int(np.sum(data['tokens'][indices, 1:] != 0)), 'completed_step': result['step']})` to the caller.
    return result,{"ids":[data["ids"][int(i)] for i in indices],"dropout_key":np.asarray(dropout_key).tolist(),
                   "pre_update_loss":float(loss),"valid_tokens":int(np.sum(data["tokens"][indices,1:]!=0)),"completed_step":result["step"]}


# Function `save_checkpoint(path, state)` implementing this stage's computation:
def save_checkpoint(path,state):
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    # Convert `arrays` to a host NumPy array for inspection or verification.
    arrays={f"{group}__{name}":np.asarray(value) for group in ("params","m","v") for name,value in state[group].items()}
    # Convert `` to a host NumPy array for inspection or verification.
    arrays.update(key=np.asarray(state["key"]),order=np.asarray(state["order"]))
    # Run `np.savez` to perform the next check or state transition.
    np.savez(path/"state.npz",**arrays)
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={k:state[k] for k in ("step","cursor","epoch","data_sha256","config")}
    # Compute deterministic cryptographic digest `` for provenance verification.
    manifest.update(tokenizer=TOKENIZER,jax=jax.__version__,architecture={"width":WIDTH,"heads":HEADS,"blocks":1},
                    state_sha256=hashlib.sha256((path/"state.npz").read_bytes()).hexdigest())
    # Read or serialize artifact data on disk (``).
    (path/"checkpoint.json").write_text(json.dumps(manifest,indent=2))


# Function `load_checkpoint(path, data, config)` implementing this stage's computation:
def load_checkpoint(path,data,config):
    # Read or serialize artifact data on disk (`path`).
    # Read or serialize artifact data on disk (`manifest`).
    path=Path(path);manifest=json.loads((path/"checkpoint.json").read_text())
    # Guard input contract (`manifest['data_sha256'] != corpus_hash(data) or manifest['config'] != config`) and fail fast if violated.
    if manifest["data_sha256"]!=corpus_hash(data) or manifest["config"]!=config:raise ValueError("corpus/configuration mismatch")
    # Guard input contract (`manifest['tokenizer'] != TOKENIZER or manifest['jax'] != jax.__version__`) and fail fast if violated.
    if manifest["tokenizer"]!=TOKENIZER or manifest["jax"]!=jax.__version__:raise ValueError("tokenizer/runtime mismatch")
    # Guard input contract (`manifest['architecture'] != {'width': WIDTH, 'heads': HEADS, 'blocks': 1}`) and fail fast if violated.
    if manifest["architecture"]!={"width":WIDTH,"heads":HEADS,"blocks":1}:raise ValueError("architecture mismatch")
    # Guard input contract (`hashlib.sha256((path / 'state.npz').read_bytes()).hexdigest() != manifest['state_sha256']`) and fail fast if violated.
    if hashlib.sha256((path/"state.npz").read_bytes()).hexdigest()!=manifest["state_sha256"]:raise ValueError("checkpoint corrupt")
    # Enter managed runtime/context scope for this block:
    with np.load(path/"state.npz",allow_pickle=False) as arrays:
        # Create device-backed JAX array `state`.
        state={group:{name:jnp.asarray(arrays[f"{group}__{name}"]) for name in PARAM_SHAPES} for group in ("params","m","v")}
        # Create device-backed JAX array ``.
        state.update(key=jnp.asarray(arrays["key"]),order=jnp.asarray(arrays["order"]))
    # Compute deterministic cryptographic digest `` for provenance verification.
    state.update({k:manifest[k] for k in ("step","cursor","epoch","data_sha256","config")})
    # Return `state` to the caller.
    return state


# Function `evaluate(params, data, batch_size, compute_policy, ...)` implementing this stage's computation:
def evaluate(params,data,batch_size=11,compute_policy="fp32",cache_policy="fp32"):
    # Evaluate `total` from the current inputs and state.
    # Evaluate `count` from the current inputs and state.
    # Evaluate `correct` from the current inputs and state.
    total=0.;count=0;correct=0
    # Loop over `start` in `range(0, len(data['ids']), batch_size)`:
    for start in range(0,len(data["ids"]),batch_size):
        # Evaluate `docs` from the current inputs and state.
        docs=data["tokens"][start:start+batch_size]
        # Create device-backed JAX array `logits`.
        logits=np.asarray(forward(params,jnp.asarray(docs[:,:-1]),compute_policy=compute_policy,cache_policy=cache_policy)[0])
        # Reduce across the target axis to summarize `shifted`.
        shifted=logits-logits.max(-1,keepdims=True)
        # Reduce across the target axis to summarize `logp`.
        logp=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        # Evaluate `targets` from the current inputs and state.
        # Evaluate `mask` from the current inputs and state.
        targets=docs[:,1:];mask=targets!=0
        # Accumulate the next contribution into `total`.
        total+=float(-np.take_along_axis(logp,targets[...,None],axis=-1)[...,0][mask].sum())
        # Accumulate the next contribution into `count`.
        # Accumulate the next contribution into `correct`.
        count+=int(mask.sum());correct+=int(((logits.argmax(-1)==targets)&mask).sum())
    # Return `{'nll_sum': total, 'valid_tokens': count, 'nll': total / count, 'token_accuracy': correct / count, 'perplexity': float(np.exp(total / count))}` to the caller.
    return {"nll_sum":total,"valid_tokens":count,"nll":total/count,"token_accuracy":correct/count,
            "perplexity":float(np.exp(total/count))}


# Function `validate_prompt(tokens)` implementing this stage's computation:
def validate_prompt(tokens):
    # Convert `a` to a host NumPy array for inspection or verification.
    a=np.asarray(tokens)
    # Guard input contract (`a.ndim != 1 or len(a) < 1 or len(a) > CONTEXT or (a.dtype.kind not in 'iu')`) and fail fast if violated.
    if a.ndim!=1 or len(a)<1 or len(a)>CONTEXT or a.dtype.kind not in "iu":
        raise ValueError("prompt must be a bounded one-dimensional integer token sequence")
    # Guard input contract (`a[0] != 1 or np.any(a[1:] < 3) or np.any(a >= VOCAB)`) and fail fast if violated.
    if a[0]!=1 or np.any(a[1:]<3) or np.any(a>=VOCAB):
        raise ValueError("prompt must begin with BOS and contain only byte tokens thereafter")
    # Return `np.asarray(a, dtype=np.int32)` to the caller.
    return np.asarray(a,dtype=np.int32)


# Function `prefill_core(params, padded, length, compute_policy, ...)` implementing this stage's computation:
def prefill_core(params,padded,length,compute_policy="fp32",cache_policy="fp32"):
    # Combine or mask array elements to form `(logits, _, cache)`.
    logits,_,cache=forward(params,padded,compute_policy=compute_policy,cache_policy=cache_policy)
    # Return `(logits[:, length - 1, :], *cache, length)` to the caller.
    return (logits[:,length-1,:],*cache,length)


# Function `decode_core(params, token, K, V, ...)` implementing this stage's computation:
def decode_core(params,token,K,V,Ks,Vs,length,compute_policy="fp32",cache_policy="fp32"):
    # Evaluate `x` from the current inputs and state.
    x=params["embed"][token][:,None,:]+params["position"][length][None,None,:]
    # Run `layer_norm` to compute `norm`.
    norm=layer_norm(x)
    # Run `split_heads` to compute `q`.
    q=split_heads(linear(norm,params["q"],compute_policy))
    # Run `pack_cache` to compute `(new_k, scale_k)`.
    new_k,scale_k=pack_cache(split_heads(linear(norm,params["k"],compute_policy)),cache_policy)
    # Run `pack_cache` to compute `(new_v, scale_v)`.
    new_v,scale_v=pack_cache(split_heads(linear(norm,params["v"],compute_policy)),cache_policy)
    # Run `jax.lax.dynamic_update_slice` to compute `K`.
    K=jax.lax.dynamic_update_slice(K,new_k,(0,0,length,0))
    # Run `jax.lax.dynamic_update_slice` to compute `V`.
    V=jax.lax.dynamic_update_slice(V,new_v,(0,0,length,0))
    # Run `jax.lax.dynamic_update_slice` to compute `Ks`.
    # Run `jax.lax.dynamic_update_slice` to compute `Vs`.
    Ks=jax.lax.dynamic_update_slice(Ks,scale_k,(0,0,length,0));Vs=jax.lax.dynamic_update_slice(Vs,scale_v,(0,0,length,0))
    # Perform matrix contraction / projection to compute `score`.
    score=jnp.einsum("bhtd,bhsd->bhts",q,unpack_cache(K,Ks))/jnp.sqrt(float(HEAD_DIM))
    # Create evenly spaced index values in `allowed`.
    allowed=jnp.arange(CONTEXT)<=length
    # Apply nonlinear activation or probability normalization to compute `attention`.
    attention=jax.nn.softmax(jnp.where(allowed[None,None,None,:],score,-1e9),axis=-1)
    # Perform matrix contraction / projection to compute `x`.
    x=x+linear(join_heads(jnp.einsum("bhts,bhsd->bhtd",attention,unpack_cache(V,Vs))),params["o"],compute_policy)
    # Apply nonlinear activation or probability normalization to compute `x`.
    x=x+linear(jax.nn.gelu(linear(layer_norm(x),params["ff1"],compute_policy)),params["ff2"],compute_policy)
    # Run `linear` to compute `logits`.
    logits=linear(layer_norm(x),params["head"],compute_policy)[:,0,:]
    # Return `(logits, K, V, Ks, Vs, length + 1)` to the caller.
    return logits,K,V,Ks,Vs,length+1


# Function `prefill(params, prompt, compute_policy, cache_policy)` implementing this stage's computation:
def prefill(params,prompt,compute_policy="fp32",cache_policy="fp32"):
    # Run `validate_prompt` to compute `prompt`.
    # Allocate initialized array `padded` with the specified shape and dtype.
    # Evaluate `padded[0, :len(prompt)]` from the current inputs and state.
    prompt=validate_prompt(prompt);padded=np.zeros((1,CONTEXT),np.int32);padded[0,:len(prompt)]=prompt
    # Return `prefill_core(params, jnp.asarray(padded), jnp.asarray(len(prompt), jnp.int32), compute_policy, cache_policy)` to the caller.
    return prefill_core(params,jnp.asarray(padded),jnp.asarray(len(prompt),jnp.int32),compute_policy,cache_policy)


# Function `decode_step(params, token, state, compute_policy, ...)` implementing this stage's computation:
def decode_step(params,token,state,compute_policy="fp32",cache_policy="fp32"):
    # Guard input contract (`not isinstance(token, (int, np.integer)) or token < 2 or token >= VOCAB`) and fail fast if violated.
    if not isinstance(token,(int,np.integer)) or token<2 or token>=VOCAB:raise ValueError("invalid next token")
    # Guard input contract (`not 1 <= int(state[-1]) < CONTEXT`) and fail fast if violated.
    if not 1<=int(state[-1])<CONTEXT:raise ValueError("KV context capacity exhausted or invalid; no wraparound")
    # Return `decode_core(params, jnp.asarray([token], jnp.int32), *state[1:], compute_policy, cache_policy)` to the caller.
    return decode_core(params,jnp.asarray([token],jnp.int32),*state[1:],compute_policy,cache_policy)


# Evaluate `RELEASE_POLICIES` from the current inputs and state.
RELEASE_POLICIES={"fp32":("fp32","fp32"),"bf16":("bf16","bf16"),"w8a8-int8kv":("w8a8","int8")}


# Function `export_release(path, state)` implementing this stage's computation:
def export_release(path,state):
    # Read or serialize artifact data on disk (`path`).
    # Execute the next step of the computation.
    # Evaluate `artifacts` from the current inputs and state.
    path=Path(path);path.mkdir(parents=True,exist_ok=True);artifacts={}
    # Loop over `(name, (compute, cache))` in `RELEASE_POLICIES.items()`:
    for name,(compute,cache) in RELEASE_POLICIES.items():
        # Compile and trace the function with XLA (`pre`).
        pre=jax.jit(lambda tokens,length:prefill_core(state["params"],tokens,length,compute,cache))
        # Evaluate `signature` from the current inputs and state.
        signature=(jax.ShapeDtypeStruct((1,CONTEXT),jnp.int32),jax.ShapeDtypeStruct((),jnp.int32))
        # Run `export.export` to compute `pre_artifact`.
        pre_artifact=export.export(pre)(*signature)
        # Cast or evaluate `dtype` in explicit floating-point precision.
        dtype={"fp32":jnp.float32,"bf16":jnp.bfloat16,"int8":jnp.int8}[cache]
        # Cast or evaluate `shapes` in explicit floating-point precision.
        shapes=[jax.ShapeDtypeStruct((1,),jnp.int32),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,HEAD_DIM),dtype),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,HEAD_DIM),dtype),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,1),jnp.float32),
                jax.ShapeDtypeStruct((1,HEADS,CONTEXT,1),jnp.float32),jax.ShapeDtypeStruct((),jnp.int32)]
        # Compile and trace the function with XLA (`dec`).
        dec=jax.jit(lambda token,K,V,Ks,Vs,length:decode_core(state["params"],token,K,V,Ks,Vs,length,compute,cache))
        # Run `export.export` to compute `dec_artifact`.
        dec_artifact=export.export(dec)(*shapes)
        # Loop over `(endpoint, artifact)` in `(('prefill', pre_artifact), ('decode', dec_artifact))`:
        for endpoint,artifact in (("prefill",pre_artifact),("decode",dec_artifact)):
            # Run `artifact.serialize` to compute `data`.
            # Evaluate `filename` from the current inputs and state.
            # Execute the next step of the computation.
            data=artifact.serialize();filename=f"{name}-{endpoint}.jaxexport";(path/filename).write_bytes(data)
            # Compute deterministic cryptographic digest `artifacts[f'{name}:{endpoint}']` for provenance verification.
            artifacts[f"{name}:{endpoint}"]={"file":filename,"sha256":hashlib.sha256(data).hexdigest()}
    # Convert `parameter_hash` to a host NumPy array for inspection or verification.
    parameter_hash=hashlib.sha256(b"".join(np.asarray(state["params"][k]).tobytes() for k in PARAM_SHAPES)).hexdigest()
    # Compute deterministic cryptographic digest `manifest` for provenance verification.
    manifest={"tokenizer":TOKENIZER,"architecture":{"blocks":1,"heads":HEADS,"width":WIDTH,"head_dim":HEAD_DIM},
              "training_data_sha256":state["data_sha256"],"completed_steps":state["step"],"parameter_sha256":parameter_hash,
              "jax":jax.__version__,"platform":"CPU","artifacts":artifacts,
              "precision":{"fp32":"FP32 weights/activations/dot/attention/KV/output",
              "bf16":"BF16-rounded dense operands, FP32 dot/attention/norm/softmax/output, BF16 K/V",
              "w8a8-int8kv":"dynamic per-row INT8 dense activations, per-output INT8 dense weights, INT32 dense accumulation; FP32 embedding/norm/attention/output; INT8 per-token/head K/V with FP32 scales"},
              "native_low_precision_acceleration_claimed":False}
    # Read or serialize artifact data on disk (``).
    (path/"release.json").write_text(json.dumps(manifest,indent=2))
    # Return `manifest` to the caller.
    return manifest


# Function `load_release(path)` implementing this stage's computation:
def load_release(path):
    # Read or serialize artifact data on disk (`path`).
    # Read or serialize artifact data on disk (`manifest`).
    path=Path(path);manifest=json.loads((path/"release.json").read_text())
    # Guard input contract (`manifest['tokenizer'] != TOKENIZER or manifest['jax'] != jax.__version__`) and fail fast if violated.
    if manifest["tokenizer"]!=TOKENIZER or manifest["jax"]!=jax.__version__:raise ValueError("tokenizer/runtime mismatch")
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


# Function `exported_prefill(manifest, artifacts, prompt, policy)` implementing this stage's computation:
def exported_prefill(manifest,artifacts,prompt,policy="fp32"):
    # Guard input contract (`policy not in manifest['precision']`) and fail fast if violated.
    if policy not in manifest["precision"]:raise ValueError("unsupported precision policy")
    # Run `validate_prompt` to compute `prompt`.
    # Allocate initialized array `tokens` with the specified shape and dtype.
    # Evaluate `tokens[0, :len(prompt)]` from the current inputs and state.
    prompt=validate_prompt(prompt);tokens=np.zeros((1,CONTEXT),np.int32);tokens[0,:len(prompt)]=prompt
    # Create device-backed JAX array `result`.
    result=artifacts[f"{policy}:prefill"].call(jnp.asarray(tokens),jnp.asarray(len(prompt),jnp.int32))
    # Return `jax.tree.map(lambda x: x.block_until_ready(), result)` to the caller.
    return jax.tree.map(lambda x:x.block_until_ready(),result)


# Function `exported_decode(manifest, artifacts, token, state, ...)` implementing this stage's computation:
def exported_decode(manifest,artifacts,token,state,policy="fp32"):
    # Guard input contract (`policy not in manifest['precision'] or not 1 <= int(state[-1]) < CONTEXT`) and fail fast if violated.
    if policy not in manifest["precision"] or not 1<=int(state[-1])<CONTEXT:raise ValueError("policy/context limit")
    # Guard input contract (`not isinstance(token, (int, np.integer)) or token < 2 or token >= VOCAB`) and fail fast if violated.
    if not isinstance(token,(int,np.integer)) or token<2 or token>=VOCAB:raise ValueError("invalid next token")
    # Create device-backed JAX array `result`.
    result=artifacts[f"{policy}:decode"].call(jnp.asarray([token],jnp.int32),*state[1:])
    # Return `jax.tree.map(lambda x: x.block_until_ready(), result)` to the caller.
    return jax.tree.map(lambda x:x.block_until_ready(),result)


# Function `generate(manifest, artifacts, text, max_new_tokens, ...)` implementing this stage's computation:
def generate(manifest,artifacts,text,max_new_tokens=4,policy="fp32"):
    # Run `encode` to compute `prompt`.
    prompt=encode(text)
    # Guard input contract (`max_new_tokens < 0 or len(prompt) + max_new_tokens > CONTEXT`) and fail fast if violated.
    if max_new_tokens<0 or len(prompt)+max_new_tokens>CONTEXT:raise ValueError("generation exceeds context budget")
    # Run `exported_prefill` to compute `state`.
    # Evaluate `generated` from the current inputs and state.
    state=exported_prefill(manifest,artifacts,prompt,policy);generated=[]
    # Repeat the update loop over `range(max_new_tokens)` steps:
    for _ in range(max_new_tokens):
        # Convert `logits` to a host NumPy array for inspection or verification.
        # Evaluate `logits[:2]` from the current inputs and state.
        logits=np.asarray(state[0])[0].copy();logits[:2]=-np.inf
        # Evaluate `logits.argmax()` and convert the result into Python scalar/collection `token`.
        # Execute the next step of the computation.
        token=int(logits.argmax());generated.append(token)
        # Branch on condition `token == 2`:
        if token==2:break
        # Run `exported_decode` to compute `state`.
        state=exported_decode(manifest,artifacts,token,state,policy)
    # Return `{'prompt': text, 'generated_token_ids': generated, 'generated_text': decode(generated), 'finished_eos': bool(generated and generated[-1] == 2), 'policy': policy}` to the caller.
    return {"prompt":text,"generated_token_ids":generated,"generated_text":decode(generated),
            "finished_eos":bool(generated and generated[-1]==2),"policy":policy}


# Function `benchmark(manifest, artifacts, prompt, policy, ...)` implementing this stage's computation:
def benchmark(manifest,artifacts,prompt,policy="fp32",repeats=30):
    # Run `encode` to compute `tokens`.
    tokens=encode(prompt)
    # Guard input contract (`len(tokens) >= CONTEXT`) and fail fast if violated.
    if len(tokens)>=CONTEXT:raise ValueError("one free cache slot needed for decode measurement")
    # Record execution timing or profiler trace in `started`.
    # Run `exported_prefill` to compute `cache`.
    started=time.perf_counter();cache=exported_prefill(manifest,artifacts,tokens,policy)
    # Record execution timing or profiler trace in `first_prefill`.
    first_prefill=(time.perf_counter()-started)*1000
    # Run `ord` to compute `token`.
    token=ord("a")+3
    # Record execution timing or profiler trace in `started`.
    # Execute the next step of the computation.
    started=time.perf_counter();exported_decode(manifest,artifacts,token,cache,policy)
    # Record execution timing or profiler trace in `first_decode`.
    first_decode=(time.perf_counter()-started)*1000
    # Evaluate `pre` from the current inputs and state.
    # Evaluate `dec` from the current inputs and state.
    pre=[];dec=[]
    # Repeat the update loop over `range(repeats)` steps:
    for _ in range(repeats):
        # Record execution timing or profiler trace in `started`.
        # Execute the next step of the computation.
        started=time.perf_counter();exported_prefill(manifest,artifacts,tokens,policy)
        # Record execution timing or profiler trace in ``.
        pre.append((time.perf_counter()-started)*1000)
        # Record execution timing or profiler trace in `started`.
        # Execute the next step of the computation.
        started=time.perf_counter();exported_decode(manifest,artifacts,token,cache,policy)
        # Record execution timing or profiler trace in ``.
        dec.append((time.perf_counter()-started)*1000)
    # Return `{'prompt_tokens': len(tokens), 'first_prefill_ms': first_prefill, 'first_decode_ms': first_decode, 'prefill_samples_ms': pre, 'decode_samples_ms': dec, 'prefill_p50_ms': float(np.percentile(pre, 50)), 'decode_p50_ms': float(np.percentile(dec, 50)), 'decode_p95_ms': float(np.percentile(dec, 95)), 'single_step_tokens_per_second': 1000 / float(np.percentile(dec, 50)), 'cache_bytes_including_scales': sum((np.asarray(x).nbytes for x in cache[1:5])), 'boundary': 'validated token inputs + exported call + complete returned arrays; no networking', 'decode_workload': 'one token from the same fixed prefix cache per timed repetition', 'device': 'CPU'}` to the caller.
    return {"prompt_tokens":len(tokens),"first_prefill_ms":first_prefill,"first_decode_ms":first_decode,
            "prefill_samples_ms":pre,"decode_samples_ms":dec,
            "prefill_p50_ms":float(np.percentile(pre,50)),"decode_p50_ms":float(np.percentile(dec,50)),
            "decode_p95_ms":float(np.percentile(dec,95)),"single_step_tokens_per_second":1000/float(np.percentile(dec,50)),
            "cache_bytes_including_scales":sum(np.asarray(x).nbytes for x in cache[1:5]),
            "boundary":"validated token inputs + exported call + complete returned arrays; no networking",
            "decode_workload":"one token from the same fixed prefix cache per timed repetition","device":"CPU"}
