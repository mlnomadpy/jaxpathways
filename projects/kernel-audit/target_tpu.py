"""Run actual TPU correctness and benchmark evidence; deliberately no CPU fallback."""
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--kernel',choices=['fused','pipeline'],default='fused')
p.add_argument('--implementation',default='solution')
p.add_argument('--shape',type=int,nargs=2,default=[257,513])
p.add_argument('--block',type=int,nargs=2,default=[8,128])
p.add_argument('--buffers',type=int,choices=[2,3],default=2)
p.add_argument('--synchronous',action='store_true')
p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
p.add_argument('--repeats',type=int,default=20)
args=p.parse_args()
# Check before allocations, interpretation, compilation or writing a receipt.
if jax.default_backend()!='tpu':
    p.exit(2,'Real TPU backend required. No CPU or interpret-mode fallback; no target receipt produced.\n')
if min(args.shape)<1 or args.block[0]%8 or args.block[1]%128 or min(args.block)<1 or args.repeats<5:
    p.error('positive shape, block multiples of (8,128), and at least five repeats required')
path=ROOT/args.implementation/'model.py' if args.implementation in ('starter','solution') else Path(args.implementation)
if not path.is_file():p.error('implementation must be an existing Python file')
spec=importlib.util.spec_from_file_location('kernel_implementation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=jax.devices('tpu')[0];dtype=getattr(jnp,args.dtype);rng=np.random.default_rng(71)
x=jax.device_put(jnp.asarray(rng.normal(size=tuple(args.shape)),dtype=dtype),device)
if args.kernel=='fused':
    y=jax.device_put(jnp.asarray(rng.normal(size=args.shape[1]),dtype=dtype),device)
    candidate=lambda a,b:m.fused_bias_relu(a,b,tuple(args.block),mode='tpu')
    baseline=lambda a,b:jnp.maximum(a.astype(jnp.float32)+b.astype(jnp.float32)[None,:],0).astype(a.dtype)
    reference=np.maximum(np.asarray(x,dtype=np.float32)+np.asarray(y,dtype=np.float32)[None,:],0)
else:
    y=jax.device_put(jnp.asarray(rng.normal(size=tuple(args.shape)),dtype=dtype),device)
    candidate=lambda a,b:m.pipelined_axpy(a,b,tuple(args.block),buffers=args.buffers,no_pipelining=args.synchronous,mode='tpu')
    baseline=lambda a,b:(2*a.astype(jnp.float32)+b.astype(jnp.float32)).astype(a.dtype)
    reference=2*np.asarray(x,dtype=np.float32)+np.asarray(y,dtype=np.float32)
assert all(d.platform=='tpu' for a in (x,y) for d in a.devices())
reference=np.asarray(jnp.asarray(reference,dtype=dtype),dtype=np.float32)
actual=candidate(x,y).block_until_ready()
np.testing.assert_allclose(np.asarray(actual,dtype=np.float32),reference,rtol=1e-6,atol=1e-6)
max_error=float(np.max(np.abs(np.asarray(actual,dtype=np.float32)-reference)))
measurements=m.target_benchmark(candidate,baseline,(x,y),repeats=args.repeats)
assert measurements['actual_backend']=='tpu'
record={'schemaVersion':1,'executionMode':'real-tpu','targetValidated':True,
    'implementationSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,
    'devices':[{'platform':d.platform,'kind':d.device_kind,'id':d.id} for d in jax.devices()],
    'configuration':vars(args),'max_abs_error_vs_represented_numpy_oracle':max_error,
    'measurement':measurements,
    'limits':'One target workload/device/configuration; no claim of general superiority or multi-host scaling.'}
print(json.dumps(record,indent=2,allow_nan=False))
