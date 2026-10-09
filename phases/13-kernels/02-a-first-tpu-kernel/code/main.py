"""A first TPU kernel: worked experiments and reference solutions. CPU checks."""

# Define separate interpreted and target-only launch modes
# Step 1 — Define separate interpreted and target-only launch modes: The launch path states its execution mode explicitly.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `fused_bias_relu(x, bias, block, mode)` implementing this stage's computation:
def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    # Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    # Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Guard input contract (`not isinstance(bm, int) or not isinstance(bn, int) or bm < 1 or (bn < 1)`) and fail fast if violated.
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    # Guard input contract (`mode not in ('interpret', 'tpu')`) and fail fast if violated.
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    # Branch on condition `mode == 'tpu'`:
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    # Evaluate `(m, n)` from the current inputs and state.
    # Evaluate `pm` from the current inputs and state.
    # Compute `m,n` from `x.shape`
    m,n=x.shape
    pm=(m+bm-1)//bm*bm
    pn=(n+bn-1)//bn*bn
    # Combine or mask array elements to form `px`.
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    # Combine or mask array elements to form `pb`.
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    # Function `kernel(x_ref, bias_ref, out_ref)` implementing this stage's computation:
    def kernel(x_ref,bias_ref,out_ref):
        # Cast or evaluate `value` in explicit floating-point precision.
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        # Reduce across the target axis to summarize `out_ref[...]`.
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    # Invoke custom Pallas kernel or tile specification (`matrix_spec`).
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    # Invoke custom Pallas kernel or tile specification (`bias_spec`).
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    # Invoke custom Pallas kernel or tile specification (`output`).
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    # Return `output[:m, :n]` to the caller.
    return output[:m,:n]

# Execute the kernel semantics on CPU
# Step 2 — Execute the kernel semantics on CPU: The matrix and column bias use different BlockSpecs.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
# Construct `bias` via `jnp.linspace(-.3,.4,129,dtype=jnp.float32)`
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
# Run `fused_bias_relu` to compute `actual`.
actual=fused_bias_relu(x,bias)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)`
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Check tensor shape invariant: `actual.shape==(9,129)`
assert actual.shape==(9,129)
# Print the observed values to compare against the expected result.
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
# Print diagnostic summary of the computed outputs.
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))

# Prove the target path cannot silently claim CPU success
# Step 3 — Prove the target path cannot silently claim CPU success: The registered lesson executes CPU interpretation only.
if jax.default_backend()=="cpu":
    rejected=False
    try:fused_bias_relu(x,bias,mode="tpu")
    except RuntimeError as error:
        rejected=True
        print("Expected target refusal:",error)
    assert rejected
else:
    print("Reference lesson is intended for CPU; target execution uses the separate project runner")

# Step 1 — Define separate interpreted and target-only launch modes: The launch path states its execution mode explicitly.
# Import jax for this computation.
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Function `fused_bias_relu(x, bias, block, mode)` implementing this stage's computation:
def fused_bias_relu(x,bias,block=(8,128),mode="interpret"):
    # Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    if x.ndim!=2 or bias.ndim!=1 or bias.shape[0]!=x.shape[1] or min(x.shape)<1:
        raise ValueError("x must be (M,N), bias must be (N,), and dimensions nonempty")
    # Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    if x.dtype!=bias.dtype or x.dtype not in (jnp.float32,jnp.bfloat16):
        raise ValueError("matching float32 or bfloat16 required")
    # Compute `bm,bn` from `block`
    bm,bn=block
    # Guard input contract (`not isinstance(bm, int) or not isinstance(bn, int) or bm < 1 or (bn < 1)`) and fail fast if violated.
    if not isinstance(bm,int) or not isinstance(bn,int) or bm<1 or bn<1:
        raise ValueError("positive integer block sizes required")
    # Guard input contract (`mode not in ('interpret', 'tpu')`) and fail fast if violated.
    if mode not in ("interpret","tpu"):
        raise ValueError("mode must explicitly be interpret or tpu")
    # Branch on condition `mode == 'tpu'`:
    if mode=="tpu":
        if jax.default_backend()!="tpu" or any(not isinstance(a,jax.core.Tracer) and any(d.platform!="tpu" for d in a.devices()) for a in (x,bias)):
            raise RuntimeError("Real TPU inputs/backend required; CPU fallback is disabled")
        if bm%8 or bn%128:
            raise ValueError("this TPU wrapper requires block multiples of (8,128)")
    # Evaluate `(m, n)` from the current inputs and state.
    # Evaluate `pm` from the current inputs and state.
    # Compute `m,n` from `x.shape`
    m,n=x.shape
    pm=(m+bm-1)//bm*bm
    pn=(n+bn-1)//bn*bn
    # Combine or mask array elements to form `px`.
    px=jnp.pad(x,((0,pm-m),(0,pn-n)))
    # Combine or mask array elements to form `pb`.
    pb=jnp.pad(bias,(0,pn-n))[None,:]
    # Function `kernel(x_ref, bias_ref, out_ref)` implementing this stage's computation:
    def kernel(x_ref,bias_ref,out_ref):
        # Cast or evaluate `value` in explicit floating-point precision.
        value=x_ref[...].astype(jnp.float32)+bias_ref[...].astype(jnp.float32)
        # Reduce across the target axis to summarize `out_ref[...]`.
        out_ref[...]=jnp.maximum(value,0).astype(out_ref.dtype)
    # Invoke custom Pallas kernel or tile specification (`matrix_spec`).
    matrix_spec=pl.BlockSpec((bm,bn),lambda i,j:(i,j))
    # Invoke custom Pallas kernel or tile specification (`bias_spec`).
    bias_spec=pl.BlockSpec((1,bn),lambda i,j:(0,j))
    # Invoke custom Pallas kernel or tile specification (`output`).
    output=pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((pm,pn),x.dtype),
        grid=(pm//bm,pn//bn),in_specs=(matrix_spec,bias_spec),out_specs=matrix_spec,
        interpret=(mode=="interpret"),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")))(px,pb)
    # Return `output[:m, :n]` to the caller.
    return output[:m,:n]

# Step 2 — Execute the kernel semantics on CPU: The matrix and column bias use different BlockSpecs.
# Construct and reshape `x` into the target tensor dimensions.
x=jnp.linspace(-2,2,9*129,dtype=jnp.float32).reshape(9,129)
# Construct `bias` via `jnp.linspace(-.3,.4,129,dtype=jnp.float32)`
bias=jnp.linspace(-.3,.4,129,dtype=jnp.float32)
# Run `fused_bias_relu` to compute `actual`.
actual=fused_bias_relu(x,bias)
# Convert `reference` to a host NumPy array for inspection or verification.
reference=np.maximum(np.asarray(x)+np.asarray(bias)[None,:],0)
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)`
np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
# Check tensor shape invariant: `actual.shape==(9,129)`
assert actual.shape==(9,129)
# Print the observed values to compare against the expected result.
print("Interpretation max error:",float(np.max(np.abs(np.asarray(actual)-reference))))
# Print diagnostic summary of the computed outputs.
print("Zero fraction and endpoint:",float(np.mean(reference==0)),float(actual[-1,-1]))

# Step 3 — Prove the target path cannot silently claim CPU success: The registered lesson executes CPU interpretation only.
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
# Compute figure data for: Bias and ReLU alter the values, not the tile ownership contract
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'kind':'line','x':list(range(129)),'xlabel':'logical column','ylabel':'output activation','series':[{'label':'first row, CPU interpretation','y':np.asarray(actual)[0].tolist()},{'label':'last row, CPU interpretation','y':np.asarray(actual)[-1].tolist()}]}

# Experiment: Check the represented bfloat16 contract
# Experiment — Check the represented bfloat16 contract: This checks the declared compute/cast policy.
bx=x.astype(jnp.bfloat16)
bb=bias.astype(jnp.bfloat16)
# Run `fused_bias_relu` to compute `bactual`.
bactual=fused_bias_relu(bx,bb)
# Cast or evaluate `bref` in explicit floating-point precision.
bref=jnp.maximum(bx.astype(jnp.float32)+bb.astype(jnp.float32)[None,:],0).astype(jnp.bfloat16)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(np.asarray(bactual),np.asarray(bref))
# Assert invariant `bactual.dtype==jnp.bfloat16` holds
assert bactual.dtype==jnp.bfloat16
# Print the observed values to compare against the expected result.
print("bfloat16 represented-input reference matched exactly")

# Experiment: Place the ReLU boundary deliberately
# Experiment — Place the ReLU boundary deliberately: Testing activation boundaries catches wrong bias axes and...
# Construct `edge_x` via `jnp.array([[-1.,0.,1.],[1.,-2.,.5]],dtype=jnp.float32)`
edge_x=jnp.array([[-1.,0.,1.],[1.,-2.,.5]],dtype=jnp.float32)
# Construct `edge_bias` via `jnp.array([1.,0.,-1.],dtype=jnp.float32)`
edge_bias=jnp.array([1.,0.,-1.],dtype=jnp.float32)
# Run `fused_bias_relu` to compute `edge`.
edge=fused_bias_relu(edge_x,edge_bias)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_array_equal(edge,np.array([[0.,0.,0.],[2.,0.,0.]],dtype=np.float32))
# Print the observed values to compare against the expected result.
print("Boundary fixture:",np.asarray(edge))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Change the activation shape to (17,257), freeze a NumPy random seed...
rng=np.random.default_rng(5)
# Create device-backed JAX array `changed_x`.
changed_x=jnp.asarray(rng.normal(size=(17,257)),dtype=jnp.float32)
# Create device-backed JAX array `changed_bias`.
changed_bias=jnp.asarray(rng.normal(size=257),dtype=jnp.float32)
# Convert `expected` to a host NumPy array for inspection or verification.
expected=np.maximum(np.asarray(changed_x)+np.asarray(changed_bias)[None,:],0)
# Iterate over `block` to step through the computation:
for block in [(8,128),(16,256)]:
    # Run `fused_bias_relu` to compute `result`.
    result=fused_bias_relu(changed_x,changed_bias,block)
    # Check numerical equivalence within tolerance: `np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)`
    np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Changed tails and two TPU-compatible blocks verified in interpretation")

# Reference practice: Reject hidden bias-axis broadcasting
# Reject hidden bias-axis broadcasting (Practice): Explicit validation turns a confusing broadcast result into...
# Iterate over `malformed` to step through the computation:
for malformed in (jnp.ones((1,129),dtype=jnp.float32),jnp.ones(9,dtype=jnp.float32)):
    # Compute `rejected` from `False`
    rejected=False
    # Run the boundary check and catch the expected exception:
    try:fused_bias_relu(x,malformed)
    except ValueError:rejected=True
    # Assert invariant `rejected` holds
    assert rejected
# Print the observed values to compare against the expected result.
print("Malformed bias axes rejected")

# Reference practice: Check a permutation invariant
# Check a permutation invariant (Challenge): The kernel is columnwise when the corresponding bias travels...
# Construct `order` via `jnp.arange(128,-1,-1)`
order=jnp.arange(128,-1,-1)
# Run `fused_bias_relu` to compute `permuted`.
permuted=fused_bias_relu(x[:,order],bias[order])
# Check numerical equivalence within tolerance: `np.testing.assert_allclose(permuted[:,order],actual,rtol=1e-6,ato...`
np.testing.assert_allclose(permuted[:,order],actual,rtol=1e-6,atol=1e-6)
# Print the observed values to compare against the expected result.
print("Column/bias pairing survives a joint permutation")
print("PASS: kernels-02")
