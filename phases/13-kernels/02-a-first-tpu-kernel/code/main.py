"""A first TPU kernel: worked experiments and reference solutions. CPU checks."""

# Define separate interpreted and target-only launch modes
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    bm,bn=block
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    m,n=x.shape;pm=(m+bm-1)//bm*bm;pn=(n+bn-1)//bn*bn
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    def kernel(x_ref,bias_ref,out_ref):
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    return output[:m,:n]

# Execute the kernel semantics on CPU
x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
actual=fused_bias_relu(x,bias)
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
assert actual.shape==(9,129)
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))

# Prove the target path cannot silently claim CPU success
if jax.default_backend()=="cpu":
    rejected=False
    try:fused_bias_relu(x,bias,mode="tpu")
    except RuntimeError as error:
        rejected=True
        print("Expected target refusal:",error)
    assert rejected
else:
    print("Reference lesson is intended for CPU; target execution uses the separate project runner")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    bm,bn=block
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    m,n=x.shape;pm=(m+bm-1)//bm*bm;pn=(n+bn-1)//bn*bn
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    def kernel(x_ref,bias_ref,out_ref):
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    return output[:m,:n]

x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
actual=fused_bias_relu(x,bias)
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
assert actual.shape==(9,129)
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))

if jax.default_backend()=="cpu":
    rejected=False
    try:fused_bias_relu(x,bias,mode="tpu")
    except RuntimeError as error:
        rejected=True
        print("Expected target refusal:",error)
    assert rejected
else:
    print("Reference lesson is intended for CPU; target execution uses the separate project runner")

# Figure data experiment
visual_data={'kind':'line','x':list(range(129)),'xlabel':'logical column','ylabel':'output activation','series':[{'label':'first row, CPU interpretation','y':np.asarray(actual)[0].tolist()},{'label':'last row, CPU interpretation','y':np.asarray(actual)[-1].tolist()}]}

# Experiment: Check the represented bfloat16 contract
bx=x.astype(jnp.bfloat16);bb=bias.astype(jnp.bfloat16)
bactual=fused_bias_relu(bx,bb)
bref=jnp.maximum(bx.astype(jnp.float32)+bb.astype(jnp.float32)[None,:],0).astype(jnp.bfloat16)
np.testing.assert_array_equal(np.asarray(bactual),np.asarray(bref))
assert bactual.dtype==jnp.bfloat16
print("bfloat16 represented-input reference matched exactly")

# Experiment: Place the ReLU boundary deliberately
edge_x=jnp.array([[-1.,0.,1.],[1.,-2.,.5]],dtype=jnp.float32)
edge_bias=jnp.array([1.,0.,-1.],dtype=jnp.float32)
edge=fused_bias_relu(edge_x,edge_bias)
np.testing.assert_array_equal(edge,np.array([[0.,0.,0.],[2.,0.,0.]],dtype=np.float32))
print("Boundary fixture:",np.asarray(edge))

# Reference solution. Try the exercise before reading this.
rng=np.random.default_rng(5)
changed_x=jnp.asarray(rng.normal(size=(17,257)),dtype=jnp.float32)
changed_bias=jnp.asarray(rng.normal(size=257),dtype=jnp.float32)
expected=np.maximum(np.asarray(changed_x)+np.asarray(changed_bias)[None,:],0)
for block in [(8,128),(16,256)]:
    result=fused_bias_relu(changed_x,changed_bias,block)
    np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
print("Changed tails and two TPU-compatible blocks verified in interpretation")

# Reference practice: Reject hidden bias-axis broadcasting
for malformed in (jnp.ones((1,129),dtype=jnp.float32),jnp.ones(9,dtype=jnp.float32)):
    rejected=False
    try:fused_bias_relu(x,malformed)
    except ValueError:rejected=True
    assert rejected
print("Malformed bias axes rejected")

# Reference practice: Check a permutation invariant
order=jnp.arange(128,-1,-1)
permuted=fused_bias_relu(x[:,order],bias[order])
np.testing.assert_allclose(permuted[:,order],actual,rtol=1e-6,atol=1e-6)
print("Column/bias pairing survives a joint permutation")
print("PASS: kernels-02")
