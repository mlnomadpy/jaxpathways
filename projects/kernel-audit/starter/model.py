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
    raise NotImplementedError('Stage 1: grid, BlockSpecs and explicit tails')


def fused_bias_relu(x,bias,block=(8,128),mode='interpret'):
    """Actual Pallas max(X+bias,0), matrix X and vector bias of matching dtype.

    interpret is CPU semantics; tpu must refuse non-TPU backend/inputs and use
    interpret=False. Target blocks are multiples of (8,128). Cast arithmetic to
    float32 and output once to the input dtype. Preserve logical output shape.
    """
    raise NotImplementedError('Stage 2: fused target path with no CPU fallback')


def pipelined_axpy(x,y,block=(8,128),buffers=2,no_pipelining=False,mode='simulate'):
    """Actual emit_pipeline for Z=2X+Y with explicit tails and dtype policy.

    Two or three input buffers; two output buffers. Blocks multiple of (8,128).
    simulate uses TPU InterpretParams and explicitly simulated layout metadata;
    tpu requires real TPU backend/inputs and interpret=False. Never substitute
    simulation for a target request. no_pipelining exposes synchronous debug
    copies. Check representative shapes before discussing performance.
    """
    raise NotImplementedError('Stage 3: pipeline semantics and evidence boundary')


def target_benchmark(candidate,baseline,args,repeats=20):
    """Refuse non-TPU inputs/backend; compile separately, warm up, synchronize.

    Return actual_backend, device_kind, boundary and candidate/baseline records
    containing compile_seconds, samples_ms, median_ms and p90_ms. Time the full
    logical-input padded/cropped wrapper, excluding input placement/compilation.
    Require at least five samples. CPU interpretation is not timed here.
    """
    raise NotImplementedError('Stage 3: honest target-only measurement protocol')
