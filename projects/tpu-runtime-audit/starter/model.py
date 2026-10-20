"""Learner starter: Inspect actual CPU device inventory and compare explicit BF16 storage with FP32 accumulation."""
import jax
import jax.numpy as jnp

def runtime_report() -> dict:
    raise NotImplementedError("implement the inspect actual cpu device inventory and compare explicit bf16 storage with fp32 accumulation contract")

def simulated_collective_sum(values, omitted=()):
    raise NotImplementedError("simulate participant accounting explicitly and label it CPU-only")
