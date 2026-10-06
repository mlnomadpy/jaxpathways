"""Public cumulative checks; launch fresh so the four-device CPU setting takes effect."""
import argparse
import datetime
import os
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import jax
jax.config.update('jax_platforms','cpu')
jax.config.update('jax_num_cpu_devices',4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--implementation',default='starter');p.add_argument('--stage',choices=['1','2','3','4','all'],default='all')
p.add_argument('--output',type=Path);p.add_argument('--child',type=Path)
p.add_argument('--seed',type=int,default=17);p.add_argument('--batch',type=int,default=7)
args=p.parse_args();stage=4 if args.stage=='all' else int(args.stage)
path=ROOT/args.implementation/'model.py' if args.implementation in ('starter','solution') else Path(args.implementation).resolve()
if not path.is_file():p.error('implementation must be an existing Python file')
spec=importlib.util.spec_from_file_location('learner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mesh=Mesh(np.asarray(jax.devices()),('data',));assert mesh.size==4
step=m.make_step(mesh) if stage>=2 or args.child else None

def fixture(seed,n=23,d=8):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,d)).astype(np.float32)
    y=(x@np.linspace(-.8,.9,d,dtype=np.float32)+.12*rng.normal(size=n)).astype(np.float32)
    return x,y

def reject(fn,error=ValueError):
    try:fn()
    except error:return
    raise AssertionError('missing rejection: '+error.__name__)

def close(a,b):np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
def same_state(a,b):
    assert a.keys()==b.keys()
    for k in a:
        if k in ('weights','momentum'):close(a[k],b[k])
        else:np.testing.assert_array_equal(a[k],b[k])

def continue_run(state,x,y,batch,steps):
    ids=[];loss=[]
    for _ in range(steps):
        state,l,i=m.transition(state,x,y,batch,step,mesh)
        l.block_until_ready();ids.extend(i.tolist());loss.append(float(l))
    return state,ids,loss

if args.child:
    x,y=fixture(args.seed);metadata=m.contract(x,y,args.batch,args.seed)
    state=m.restore(args.child/'checkpoint.npz',metadata,mesh)
    state,ids,loss=continue_run(state,x,y,args.batch,2*((len(x)+args.batch-1)//args.batch)+1)
    np.savez(args.child/'child-state.npz',**{k:np.asarray(v) for k,v in state.items()})
    (args.child/'child-result.json').write_text(json.dumps(dict(ids=ids,loss=loss,devices=jax.device_count(),backend=jax.default_backend())))
    sys.exit(0)

# Stage 1: deterministic batching, no input mutation, exact epoch coverage, partial tails.
for seed,batch in [(17,7),(29,10),(31,1),(37,64)]:
    x,y=fixture(seed);s=m.initial(len(x),x.shape[1],seed,mesh);seen=[]
    frozen={k:np.asarray(v).copy() for k,v in s.items()}
    s2,ids,(xb,yb,valid)=m.next_batch(s,x,y,batch,mesh)
    same_state(s,frozen)
    replay,again,packed=m.next_batch(s,x,y,batch,mesh)
    same_state(s2,replay);np.testing.assert_array_equal(ids,again)
    close(np.asarray(xb)[:len(ids)],x[ids]);close(np.asarray(yb)[:len(ids)],y[ids])
    assert int(np.sum(valid))==len(ids) and xb.shape[0]%4==0
    for leaf in (xb,yb,valid):assert len(leaf.addressable_shards)==4
    for _ in range((len(x)+batch-1)//batch):
        s,ids,_=m.next_batch(s,x,y,batch,mesh);seen.extend(ids.tolist())
    np.testing.assert_array_equal(sorted(seen),np.arange(len(x)))
    epoch2,_,_=m.next_batch(s,x,y,batch,mesh)
    assert not np.array_equal(epoch2['key'],s['key'])
    reject(lambda:m.next_batch(s,x,y,0,mesh))
    reject(lambda:m.next_batch(s,x.astype(np.float64),y,batch,mesh))
print('PASS stage 1: immutable sampler, key progression, complete epochs, changed seeds and padded tails')

metrics={'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],
         'jax':jax.__version__,'numpy':np.__version__,'implementation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'executed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if stage>=2:
    counts_evidence=[]
    for seed,batch in [(17,7),(29,10),(31,1),(37,64)]:
        x,y=fixture(seed);s=m.initial(len(x),x.shape[1],seed,mesh)
        # Several nonzero momentum updates, including reshuffle and remainder batches.
        for _ in range(2*((len(x)+batch-1)//batch)+1):
            _,ids,packed=m.next_batch(s,x,y,batch,mesh)
            hostw=np.asarray(s['weights']);hostv=np.asarray(s['momentum'])
            residual=x[ids]@hostw-y[ids]
            oracle=2*x[ids].T@residual/len(ids)
            velocity=.8*hostv+oracle
            nw,nv,loss,gradient=step(*packed,s['weights'],s['momentum'])
            close(gradient,oracle);close(nw,hostw-.03*velocity);close(nv,velocity)
            close(loss,np.mean(residual**2))
            # Independent per-partition numerator/count reconstruction, with zero-count ranks.
            xparts=np.split(np.asarray(packed[0]),4);yparts=np.split(np.asarray(packed[1]),4);masks=np.split(np.asarray(packed[2]),4)
            counts=[int(mask.sum()) for mask in masks]
            nums=[2*a.T@((a@hostw-b)*mask) for a,b,mask in zip(xparts,yparts,masks)]
            close(sum(nums)/sum(counts),oracle)
            for leaf in (nw,nv,gradient):
                assert leaf.sharding.is_fully_replicated and len(leaf.addressable_shards)==4
                for shard in leaf.addressable_shards:close(shard.data,np.asarray(leaf))
            new,l,newids=m.transition(s,x,y,batch,step,mesh)
            close(new['weights'],nw);close(new['momentum'],nv);close(l,loss);np.testing.assert_array_equal(newids,ids)
            assert int(new['step'])==int(s['step'])+1
            s=new
            if counts not in counts_evidence:counts_evidence.append(counts)
    lowered=step.lower(*packed,s['weights'],s['momentum'])
    hlo=str(lowered.compiler_ir(dialect='stablehlo'))
    assert 'all_reduce' in hlo
    metrics.update(partition_counts=counts_evidence,stablehlo_all_reduce_occurrences=hlo.count('all_reduce'),local_weight_shape=list(s['weights'].addressable_shards[0].data.shape))
    print('PASS stage 2: actual four-device collectives, independent NumPy gradients, unequal/empty partition counts and replicated momentum')

fault_evidence=[]
if stage>=3:
    for seed,batch in [(17,7),(29,10)]:
        x,y=fixture(seed);metadata=m.contract(x,y,batch,seed)
        initial=m.initial(len(x),x.shape[1],seed,mesh)
        s,_,_=continue_run(initial,x,y,batch,(len(x)+batch-1)//batch+1)
        remaining=2*((len(x)+batch-1)//batch)+1
        baseline,ids,loss=continue_run(s,x,y,batch,remaining)
        with tempfile.TemporaryDirectory(prefix='sharded-replay-') as tmp:
            folder=Path(tmp);checkpoint=folder/'checkpoint.npz';m.save(checkpoint,s,metadata)
            restored=m.restore(checkpoint,metadata,mesh);same_state(s,restored)
            reject(lambda:m.save(checkpoint,s,metadata),FileExistsError)
            reject(lambda:m.restore(checkpoint,{**metadata,'seed':seed+1},mesh))
            corrupt={k:np.asarray(v) for k,v in s.items() if k!='momentum'}
            np.savez(folder/'incomplete.npz',**corrupt,metadata=np.asarray(json.dumps(metadata)))
            reject(lambda:m.restore(folder/'incomplete.npz',metadata,mesh))
            broken={k:np.asarray(v) for k,v in s.items()};broken['order']=np.zeros(len(x),np.int32)
            np.savez(folder/'corrupt.npz',**broken,metadata=np.asarray(json.dumps(metadata)))
            reject(lambda:m.restore(folder/'corrupt.npz',metadata,mesh))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--implementation',str(path.resolve()),'--child',str(folder),'--seed',str(seed),'--batch',str(batch)],check=True,capture_output=True,text=True)
            with np.load(folder/'child-state.npz') as archive:child={k:archive[k] for k in archive.files}
            same_state(baseline,child)
            result=json.loads((folder/'child-result.json').read_text())
            assert result['ids']==ids and result['devices']==4 and result['backend']=='cpu';close(result['loss'],loss)
            gaps={}
            for field in ('momentum','cursor','key'):
                broken=dict(restored);broken[field]=initial[field]
                divergent,badids,_=continue_run(broken,x,y,batch,remaining)
                gap=float(np.max(np.abs(np.asarray(divergent['weights'])-np.asarray(baseline['weights']))))
                assert gap>1e-5
                if field in ('cursor','key'):assert badids!=ids
                gaps[field]=gap
            fault_evidence.append(dict(seed=seed,batch_size=batch,saved_step=int(s['step']),saved_cursor=int(s['cursor']),restored_max_weight_error=float(np.max(np.abs(np.asarray(child['weights'])-np.asarray(baseline['weights'])))),fault_max_weight_error=gaps))
    metrics['recovery']=fault_evidence
    print('PASS stage 3: fresh-process complete replay, metadata/corruption rejection and observable momentum/cursor/key faults')

if stage>=4:
    output=args.output or Path(tempfile.mkdtemp(prefix='sharded-training-audit-'))
    output.mkdir(parents=True,exist_ok=True)
    # A larger, separate fixture supplies meaningful work; no training-quality claim follows.
    x,y=fixture(53,n=1023,d=64);s=m.initial(len(x),64,53,mesh)
    _,_,packed=m.next_batch(s,x,y,1023,mesh);inputs=(*packed,s['weights'],s['momentum'])
    compiled=m.make_step(mesh);eager=m.make_step(mesh,compiled=False)
    begin=time.perf_counter();executable=compiled.lower(*inputs).compile();compile_ms=(time.perf_counter()-begin)*1000
    def ready(result):return jax.block_until_ready(result)
    @jax.jit
    def single(a,b,w,v):
        residual=a@w-b;gradient=2*a.T@residual/len(a);velocity=.8*v+gradient
        return w-.03*velocity,velocity,jnp.mean(residual**2),gradient
    one_inputs=tuple(jax.device_put(a,jax.devices()[0]) for a in (x,y,np.asarray(s['weights']),np.asarray(s['momentum'])))
    begin=time.perf_counter();one_executable=single.lower(*one_inputs).compile();one_compile_ms=(time.perf_counter()-begin)*1000
    for _ in range(3):ready(eager(*inputs));ready(executable(*inputs));ready(one_executable(*one_inputs))
    eager_result=ready(eager(*inputs));compiled_result=ready(executable(*inputs))
    for a,b in zip(eager_result,compiled_result):close(a,b)
    for a,b in zip(ready(one_executable(*one_inputs)),compiled_result):close(a,b)
    oracle=2*x.T@(x@np.asarray(s['weights'])-y)/len(x)
    close(compiled_result[3],oracle)
    samples={'eager_shard_map':[],'compiled_shard_map':[],'compiled_single_device':[]}
    # Alternate order to reduce one-sided drift; compilation is already excluded.
    for repeat in range(11):
        choices=[('eager_shard_map',lambda:eager(*inputs)),('compiled_shard_map',lambda:executable(*inputs)),('compiled_single_device',lambda:one_executable(*one_inputs))]
        for name,fn in choices[::(-1 if repeat%2 else 1)]:
            begin=time.perf_counter();ready(fn());samples[name].append((time.perf_counter()-begin)*1000)
    hlo=str(compiled.lower(*inputs).compiler_ir(dialect='stablehlo'));(output/'step.stablehlo.mlir').write_text(hlo)
    trace_dir=output/'trace'
    with jax.profiler.trace(trace_dir,create_perfetto_trace=True):
        for index in range(4):
            with jax.profiler.StepTraceAnnotation('sharded_learner_step',step_num=index):
                with jax.profiler.TraceAnnotation('prepare_and_place'):
                    sampled,_,batch=m.next_batch(s,x,y,1023,mesh)
                    jax.block_until_ready(batch)
                with jax.profiler.TraceAnnotation('compiled_update_and_wait'):
                    w,v,loss,_=ready(executable(*batch,s['weights'],s['momentum']))
                    s={**sampled,'weights':w,'momentum':v,'step':np.asarray(int(s['step'])+1,np.int32)}
    perfetto=sorted(trace_dir.rglob('perfetto_trace.json.gz'),key=lambda f:f.stat().st_mtime_ns)[-1:]
    xplanes=list(perfetto[0].parent.glob('*.xplane.pb')) if perfetto else []
    assert perfetto and xplanes
    events=json.loads(gzip.decompress(perfetto[0].read_bytes()))['traceEvents']
    def durations(name):return [e['dur']/1000 for e in events if e.get('name')==name and 'dur' in e]
    phases={name:durations(name) for name in ('prepare_and_place','compiled_update_and_wait')}
    assert all(len(v)==4 for v in phases.values())
    metrics.update(compile_ms=compile_ms,single_device_compile_ms=one_compile_ms,timing_ms=samples,timing_scope='already placed inputs, output readiness included, host sampling excluded',trace_scope='four annotated preparation/placement and compiled update steps; profiler overhead included',trace_durations_ms=phases,trace_files=[str(f.relative_to(output)) for f in perfetto+xplanes],fixture_shape=list(x.shape),eager_to_compiled_ratio=float(np.median(samples['eager_shard_map'])/np.median(samples['compiled_shard_map'])),single_to_four_logical_device_ratio=float(np.median(samples['compiled_single_device'])/np.median(samples['compiled_shard_map'])))
    (output/'audit.json').write_text(json.dumps(metrics,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='sharded-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    labels=['momentum reset','cursor reset','key reset'];positions=np.arange(3)
    for index,item in enumerate(fault_evidence):
        values=[item['fault_max_weight_error'][k] for k in ('momentum','cursor','key')]
        axes[0].bar(positions+index*.35,values,width=.35,label=f"seed {item['seed']}, batch {item['batch_size']}")
    axes[0].set_xticks(positions+.175,labels,rotation=20);axes[0].set_ylabel('Maximum final weight error');axes[0].set_title('Missing state changes the run');axes[0].legend(fontsize=8)
    axes[1].boxplot(list(samples.values()),tick_labels=['eager 4','compiled 4','compiled 1']);axes[1].set_yscale('log');axes[1].set_ylabel('Synchronized call time (ms, log scale)');axes[1].set_title('Same sharded update, warm calls')
    for index,(name,values) in enumerate(phases.items()):axes[2].plot(range(1,5),values,'o-',label=name.replace('_',' '))
    axes[2].set_xlabel('Traced step');axes[2].set_ylabel('Host annotation duration (ms)');axes[2].set_title('Trace phases include profiler overhead');axes[2].legend(fontsize=7)
    fig.savefig(output/'audit.png',dpi=160);fig.savefig(output/'audit.svg');plt.close(fig)
    print('PASS stage 4: lowered/compiled equivalence, synchronized warm samples, actual XPlane/Perfetto trace and source-bound plotted evidence')
    print('Evidence:',output)
