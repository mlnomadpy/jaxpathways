"""Actual Pallas CPU semantic checks; no target performance claims."""
import argparse
import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['1','2','3','all'],default='all')
p.add_argument('--implementation',default='starter')
args=p.parse_args()
stage=3 if args.stage=='all' else int(args.stage)
path=ROOT/args.implementation/'model.py' if args.implementation in ('starter','solution') else Path(args.implementation)
if not path.is_file():p.error('implementation must name an existing Python file')
spec=importlib.util.spec_from_file_location('learner_kernel',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if jax.default_backend()!='cpu':p.error('These tests record CPU interpretation; use target_tpu.py for real target evidence')

def reject(fn,error=ValueError):
    try:fn()
    except error:return
    raise AssertionError('Expected '+error.__name__)

def rounded_reference(x,y,fused=False):
    xh=np.asarray(x,dtype=np.float32)
    yh=np.asarray(y,dtype=np.float32)
    value=np.maximum(xh+yh[None,:],0) if fused else 2*xh+yh
    return np.asarray(jnp.asarray(value,dtype=x.dtype))

rng=np.random.default_rng(73)
for shape in [(1,7),(5,11),(8,12)]:
    for dtype in (jnp.float32,jnp.bfloat16):
        x=jnp.asarray(rng.normal(size=shape),dtype=dtype)
        y=jnp.asarray(rng.normal(size=shape),dtype=dtype)
        for block in [(2,4),(3,5)]:
            out=m.blocked_axpy(x,y,block)
            assert out.shape==shape and out.dtype==dtype
            np.testing.assert_array_equal(np.asarray(out),rounded_reference(x,y))
reject(lambda:m.blocked_axpy(jnp.ones((2,3)),jnp.ones((2,1))))
reject(lambda:m.blocked_axpy(jnp.ones((2,3)),jnp.ones((2,3)),(0,4)))
reject(lambda:m.blocked_axpy(jnp.ones((2,3),jnp.int32),jnp.ones((2,3),jnp.int32)))
print('PASS stage 1: actual grid semantics, full and partial tiles, singleton rows, two dtypes and rejected contracts')

if stage>=2:
    for shape in [(1,1),(8,128),(9,129)]:
        for dtype in (jnp.float32,jnp.bfloat16):
            x=jnp.asarray(rng.normal(size=shape),dtype=dtype)
            bias=jnp.asarray(rng.normal(size=shape[1]),dtype=dtype)
            out=m.fused_bias_relu(x,bias)
            assert out.shape==shape and out.dtype==dtype
            np.testing.assert_array_equal(np.asarray(out),rounded_reference(x,bias,True))
    x=jnp.array([[-1.,0.,1.],[1.,-2.,.5]],dtype=jnp.float32)
    bias=jnp.array([1.,0.,-1.],dtype=jnp.float32)
    np.testing.assert_array_equal(m.fused_bias_relu(x,bias),np.array([[0,0,0],[2,0,0]],np.float32))
    reject(lambda:m.fused_bias_relu(x,jnp.ones((1,3),dtype=jnp.float32)))
    reject(lambda:m.fused_bias_relu(x,bias,mode='automatic'))
    reject(lambda:m.fused_bias_relu(x,bias,mode='tpu'),RuntimeError)
    print('PASS stage 2: fused broadcast contract, ReLU boundaries, tails, two precisions and no CPU target fallback')

if stage>=3:
    for shape in [(1,1),(9,129),(17,257)]:
        for dtype in (jnp.float32,jnp.bfloat16):
            x=jnp.asarray(rng.normal(size=shape),dtype=dtype)
            y=jnp.asarray(rng.normal(size=shape),dtype=dtype)
            for block,buffers,sync in [((8,128),2,False),((16,256),3,False),((8,128),2,True)]:
                out=m.pipelined_axpy(x,y,block,buffers=buffers,no_pipelining=sync)
                assert out.shape==shape and out.dtype==dtype
                np.testing.assert_array_equal(np.asarray(out),rounded_reference(x,y))
    x=jnp.ones((8,128),dtype=jnp.float32)
    y=jnp.ones_like(x)
    reject(lambda:m.pipelined_axpy(x,y,(3,5)))
    reject(lambda:m.pipelined_axpy(x,y,buffers=4))
    reject(lambda:m.pipelined_axpy(x,y,mode='tpu'),RuntimeError)
    reject(lambda:m.target_benchmark(lambda a,b:a+b,lambda a,b:a+b,(x,y)),RuntimeError)
    reject(lambda:m.pipelined_axpy(x,y.astype(jnp.bfloat16)))
    print('PASS stage 3: actual pipeline CPU simulation, synchronous/buffered agreement, input buffering, tails/dtypes and guarded target benchmark')
    print('Actual backend: cpu. No TPU compilation, execution latency or speedup validated.')
