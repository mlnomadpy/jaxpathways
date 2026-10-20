"""CPU environment inspection and a saved arithmetic identity check."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp

def environment_report(output_path=None):
    x=jnp.array([[1.,2.],[3.,4.]],dtype=jnp.float32)
    observed=x@x.T
    report={"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],"input":x.tolist(),"gram":observed.tolist(),"checksum":float(jnp.sum(observed))}
    if output_path is not None: Path(output_path).write_text(json.dumps(report,indent=2)+"\n")
    return report
