"""Implement explicit kernel contracts. Interpretation is not TPU qualification."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def blocked_axpy(x,y,block=(2,4)):
    """Actual interpreted Pallas Z=2X+Y, matching nonempty matrix shapes/dtypes.

    Support float32/bfloat16 with float32 arithmetic and one output cast.
    Validate positive blocks, pad full tiles explicitly, crop logical output.
    """
    # Key APIs to use: `pad_pair`, `specification`, `pl.BlockSpec`, `pl.pallas_call`, `jax.ShapeDtypeStruct`
    # Step 1: Combine or mask array elements to form `(px, py, padded)`.
    # Step 2: Evaluate `(bm, bn)` from the current inputs and state.
    # Step 3: Invoke custom Pallas kernel or tile specification (`spec`).
    # Step 4: Combine or mask array elements to form `out`.
    # Step 5: Return `out[:x.shape[0], :x.shape[1]]` to the caller.
    raise NotImplementedError('Stage 1: grid, BlockSpecs and explicit tails')


def fused_bias_relu(x,bias,block=(8,128),mode='interpret'):
    """Actual Pallas max(X+bias,0), matrix X and vector bias of matching dtype.

    interpret is CPU semantics
    tpu must refuse non-TPU backend/inputs and use
    interpret=False. Target blocks are multiples of (8,128). Cast arithmetic to
    float32 and output once to the input dtype. Preserve logical output shape.
    """
    # Key APIs to use: `contract`, `or`, `min`, `be`, `in`
    # Step 1: Guard input contract (`x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or (min(x.shape) < 1)`) and fail fast if violated.
    # Step 2: Guard input contract (`x.dtype != bias.dtype or x.dtype not in (jnp.float32, jnp.bfloat16)`) and fail fast if violated.
    # Step 3: Evaluate `(bm, bn)` from the current inputs and state.
    # Step 4: Guard input contract (`not isinstance(bm, int) or not isinstance(bn, int) or bm < 1 or (bn < 1)`) and fail fast if violated.
    # Step 5: Guard input contract (`mode not in ('interpret', 'tpu')`) and fail fast if violated.
    # Step 6: Branch on condition `mode == 'tpu'`:
    raise NotImplementedError('Stage 2: fused target path with no CPU fallback')


def pipelined_axpy(x,y,block=(8,128),buffers=2,no_pipelining=False,mode='simulate'):
    """Actual emit_pipeline for Z=2X+Y with explicit tails and dtype policy.

    Two or three input buffers
    two output buffers. Blocks multiple of (8,128).
    simulate uses TPU InterpretParams and explicitly simulated layout metadata;
    tpu requires real TPU backend/inputs and interpret=False. Never substitute
    simulation for a target request. no_pipelining exposes synchronous debug
    copies. Check representative shapes before discussing performance.
    """
    # Key APIs to use: `pad_pair`, `contract`, `of`, `in`, `and`
    # Step 1: Combine or mask array elements to form `(px, py, padded)`.
    # Step 2: Evaluate `(bm, bn)` from the current inputs and state.
    # Step 3: Guard input contract (`bm % 8 or bn % 128`) and fail fast if violated.
    # Step 4: Guard input contract (`buffers not in (2, 3)`) and fail fast if violated.
    # Step 5: Guard input contract (`mode not in ('simulate', 'tpu')`) and fail fast if violated.
    # Step 6: Guard input contract (`mode == 'tpu' and (jax.default_backend() != 'tpu' or any((not isinstance(a, jax.core.Tracer) and any((d.platform != 'tpu' for d in a.devices())) for a in (x, y))))`) and fail fast if violated.
    raise NotImplementedError('Stage 3: pipeline semantics and evidence boundary')


def target_benchmark(candidate,baseline,args,repeats=20):
    """Refuse non-TPU inputs/backend; compile separately, warm up, synchronize.

    Return actual_backend, device_kind, boundary and candidate/baseline records
    containing compile_seconds, samples_ms, median_ms and p90_ms. Time the full
    logical-input padded/cropped wrapper, excluding input placement/compilation.
    Require at least five samples. CPU interpretation is not timed here.
    """
    # Key APIs to use: `contract`, `jax.default_backend`, `any`, `a.devices`, `in`
    # Step 1: Guard input contract (`jax.default_backend() != 'tpu' or any((d.platform != 'tpu' for a in args for d in a.devices()))`) and fail fast if violated.
    # Step 2: Guard input contract (`repeats < 5`) and fail fast if violated.
    # Step 3: Import required JAX, NumPy, and standard-library modules.
    # Step 4: Evaluate `records` from the current inputs and state.
    # Step 5: Loop over `(label, function)` in `(('candidate', candidate), ('baseline', baseline))`:
    # Step 6: Inside block: Record execution timing or profiler trace in `started`.
    raise NotImplementedError('Stage 3: honest target-only measurement protocol')
