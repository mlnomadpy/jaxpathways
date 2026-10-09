"""Train, resume, export, profile and visualize the actual tiny Transformer."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"jaxpathways-text-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--implementation",default="solution")
parser.add_argument("--output",default=str(ROOT/"outputs"))
args=parser.parse_args()
path=ROOT/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)
spec=importlib.util.spec_from_file_location("text_model",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out=Path(args.output)
out.mkdir(parents=True,exist_ok=True)
train=m.make_corpus(31,96,"train")
held=m.make_corpus(89,32,"held",exclude=train["texts"])
state=m.initialize(train,seed=3)
history=[]
for _ in range(120):
    state,item=m.step(state,train)
    history.append(item)
m.save_checkpoint(out/"checkpoint",state)
resumed=m.load_checkpoint(out/"checkpoint",train,state["config"])
for _ in range(120):
    state,original=m.step(state,train)
    resumed,replay=m.step(resumed,train)
    assert original==replay
    history.append(replay)
for group in ("params","m","v"):
    for name in state[group]:np.testing.assert_array_equal(state[group][name],resumed[group][name])
state=resumed
metrics=m.evaluate(state["params"],held)
counts=np.bincount(train["tokens"][:,1:].ravel(),minlength=m.VOCAB).astype(float)
counts[0]=0
prob=(counts+1)/(counts.sum()+m.VOCAB)
targets=held["tokens"][:,1:]
valid=targets[targets!=0]
unigram_nll=float(-np.log(prob[valid]).mean())
manifest=m.export_release(out/"release",state)
loaded,artifacts=m.load_release(out/"release")
fixed_logits=np.asarray(m.forward(state["params"],jnp.asarray(held["tokens"][:,:-1]))[0])
precision={}
for name,(compute,cache) in m.RELEASE_POLICIES.items():
    logits=np.asarray(m.forward(state["params"],jnp.asarray(held["tokens"][:,:-1]),compute_policy=compute,cache_policy=cache)[0])
    precision[name]={"metrics":m.evaluate(state["params"],held,compute_policy=compute,cache_policy=cache),
                     "maximum_valid_logit_error":float(np.max(np.abs(logits-fixed_logits)[held["tokens"][:,1:]!=0]))}
cache_only={}
for cache in ("fp32","bf16","int8"):
    logits=np.asarray(m.forward(state["params"],jnp.asarray(held["tokens"][:,:-1]),cache_policy=cache)[0])
    current=m.prefill(state["params"],m.encode("abc|"),cache_policy=cache)
    cache_only[cache]={"maximum_valid_logit_error":float(np.max(np.abs(logits-fixed_logits)[held["tokens"][:,1:]!=0])),
                       "cache_bytes_including_scales":sum(np.asarray(x).nbytes for x in current[1:5])}
generations=[]
for document in held["texts"][:3]:
    prefix=document.split("|")[0]
    item=m.generate(loaded,artifacts,prefix+"|",max_new_tokens=len(prefix)+1)
    item.update(expected_completion=prefix[::-1],held_document=document)
    generations.append(item)
diagnostic=m.generate(loaded,artifacts,"abc|",max_new_tokens=4)
diagnostic.update(expected_completion="cba",held_document=None,scope="fixed diagnostic prompt; not a held-out aggregate")
generations.append(diagnostic)
timings={name:m.benchmark(loaded,artifacts,"abc|",policy=name,repeats=30) for name in m.RELEASE_POLICIES}
# Warm the exact regions before recording a real local profiler trace.
shadow=state
shadow,_=m.step(shadow,train)
prefilled=m.exported_prefill(loaded,artifacts,m.encode("abc|"))
m.exported_decode(loaded,artifacts,102,prefilled)
profile_root=out/"profile"/("run-"+str(time.time_ns()))
with jax.profiler.trace(str(profile_root),create_perfetto_link=False,create_perfetto_trace=True):
    for _ in range(5):
        with jax.profiler.TraceAnnotation("text-training-update"):
            shadow,_=m.step(shadow,train)
        with jax.profiler.TraceAnnotation("text-prefill"):
            prefilled=m.exported_prefill(loaded,artifacts,m.encode("abc|"))
        with jax.profiler.TraceAnnotation("text-cached-decode"):
            m.exported_decode(loaded,artifacts,102,prefilled)
profile_files=[str(p.relative_to(out)) for p in profile_root.rglob("*") if p.is_file()]
assert any(p.endswith(".xplane.pb") for p in profile_files)
trace_summary={}
for relative in profile_files:
    if relative.endswith(".trace.json.gz"):
        trace=json.loads(gzip.decompress((out/relative).read_bytes()))
        events=trace.get("traceEvents",[])
        trace_summary={"event_count":len(events),"annotated_regions":{name:sum(e.get("name")==name for e in events)
            for name in ("text-training-update","text-prefill","text-cached-decode")}}
        break
assert all(trace_summary.get("annotated_regions",{}).get(name,0)>=5 for name in ("text-training-update","text-prefill","text-cached-decode"))
prompt=m.encode("abc|")
padded=np.zeros((1,12),np.int32)
padded[0,:len(prompt)]=prompt
attention=np.asarray(m.forward(state["params"],jnp.asarray(padded))[1])[0,0,:len(prompt),:len(prompt)]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"svg.fonttype":"none"})
fig,axes=plt.subplots(2,1,figsize=(8.5,9),layout="constrained")
processed=np.cumsum([item["valid_tokens"] for item in history])
axes[0].plot(processed,[item["pre_update_loss"] for item in history],color="#6240ad",label="training minibatch (dropout active)")
axes[0].axhline(metrics["nll"],color="#087c83",linestyle="--",label="final-checkpoint held-out NLL")
axes[0].axhline(unigram_nll,color="#b05c32",linestyle=":",label="training-frequency baseline on held-out")
axes[0].set(xlabel="cumulative valid target tokens processed",ylabel="mean negative log likelihood (nats/token)")
axes[0].legend()
im=axes[1].imshow(attention,cmap="Purples",vmin=0,vmax=1)
labels=["BOS","a","b","c","|"]
axes[1].set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel="key token",ylabel="query token",title="Actual head 1 attention on prompt abc|")
for (i,j),value in np.ndenumerate(attention):
    axes[1].text(j,i,f"{value:.2f}",ha="center",va="center",color="white" if value>.55 else "#241c31")
fig.colorbar(im,ax=axes[1],label="attention weight")
fig.savefig(out/"training-attention.png",dpi=150)
fig.savefig(out/"training-attention.svg")
plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(8.5,8),layout="constrained")
names=list(cache_only)
axes[0].bar(names,[cache_only[n]["cache_bytes_including_scales"] for n in names],color=["#6240ad","#087c83","#b05c32"])
axes[0].set(ylabel="bytes (K/V plus stored FP32 scales)",title="Actual allocated cache arrays: one sequence, 12 positions")
bars=axes[1].bar(names,[cache_only[n]["maximum_valid_logit_error"] for n in names],color=["#6240ad","#087c83","#b05c32"])
axes[1].bar_label(bars,fmt="%.4f")
axes[1].set(ylabel="maximum absolute logit difference",title="Cache-only precision change on fixed held-out targets")
fig.savefig(out/"cache-precision.png",dpi=150)
fig.savefig(out/"cache-precision.svg")
plt.close(fig)
report={"schema_version":1,"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment":{"python":sys.version,"jax":jax.__version__,"numpy":np.__version__,"device":str(jax.devices()[0])},
        "task":"synthetic UTF-8 byte reversal grammar; not natural-language quality",
        "tokenizer":m.TOKENIZER,"training_documents":train["texts"],"held_documents":held["texts"],
        "training_hash":m.corpus_hash(train),"held_hash":m.corpus_hash(held),
        "training_loss":[item["pre_update_loss"] for item in history],"processed_valid_tokens":processed.tolist(),
        "held":metrics,"unigram_held_nll":unigram_nll,"precision":precision,"cache_only":cache_only,
        "generation":generations,"attention_head1":attention.tolist(),"timings":timings,
        "recovery":{"interrupted_step":120,"resumed_to":240,"matching_subsequent_trace_and_state":True,
                    "fresh_process_evidence":"tests/check.py stage 2"},
        "profile_files":profile_files,"profile_summary":trace_summary,
        "artifacts":manifest["artifacts"],
        "limitations":["CPU only","fixed 12-token context and one request at a time","symbolic synthetic corpus",
                       "no natural-language or unseen-domain quality claim","no native low-bit acceleration claim",
                       "no network/cold-process/TPU/distributed performance claim"]}
(out/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ("held","unigram_held_nll","cache_only","generation","profile_summary")},indent=2))
print("Wrote",out)
