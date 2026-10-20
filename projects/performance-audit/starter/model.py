"""Starter: profile a small CPU workload and retain trace evidence."""
import json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

def performance_report(output_dir, repeats=5):
    raise NotImplementedError("measure synchronized calls, count shape traces, and save report plus JAXPR")
