"""Learner starter: Account for parameter-state bytes and derive scaling efficiency from supplied measurements."""
import jax.numpy as jnp

def systems_report(parameter_count, dtype_bytes, master_bytes, moment_count, measured_examples_per_second, worker_count, single_worker_examples_per_second) -> dict:
    raise NotImplementedError("implement the account for parameter-state bytes and derive scaling efficiency from supplied measurements contract")
