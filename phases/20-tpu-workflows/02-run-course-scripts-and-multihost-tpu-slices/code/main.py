"""Run course scripts and multi-host TPU Pod slices: worked experiments and reference solutions. CPU checks."""

# Tabulate single-host and multi-host TPU slice topologies
# Step 1 — Tabulate single-host and multi-host TPU slice topologies
import json
import jax
import numpy as np

SLICE_CATALOG = [
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v6e-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 32},
]
for entry in SLICE_CATALOG:
    entry["global_chips"] = entry["hosts"] * entry["chips_per_host"]
    entry["total_hbm_gib"] = entry["global_chips"] * entry["hbm_per_chip_gib"]
    entry["multi_host"] = entry["hosts"] > 1

# Generate single-host and multi-host gcloud command pipelines
def build_tpu_experiment_commands(tpu_name, zone, lesson_id, hosts=1, execution_class="review-required"):
    if execution_class == "review-required":
        raise ValueError("classify and review the script before generating a TPU launch command")
    if execution_class == "cpu-only":
        if hosts != 1:
            raise ValueError("CPU-only scripts must use the CPU route")
        return {"run": f"python public/exercises/{lesson_id}.py", "multi_host": False}
    if execution_class not in {"single-host-compatible", "multi-host-qualified"}:
        raise ValueError("unknown execution class")
    if execution_class == "single-host-compatible" and hosts != 1:
        raise ValueError("single-host scripts cannot run across a TPU Pod")
    if hosts > 1 and execution_class != "multi-host-qualified":
        raise ValueError("multi-host launch requires a qualified training program")
    if hosts < 1:
        raise ValueError("hosts must be positive")
    worker_flag = " --worker=all" if hosts > 1 else ""
    scp_stage = f"gcloud compute tpus tpu-vm scp public/exercises/{lesson_id}.py {tpu_name}:~/jax-tpu-lab/ --zone={zone}{worker_flag}"
    ssh_run = f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone}{worker_flag} --command='set -o pipefail; JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/{lesson_id}.py | tee ~/jax-tpu-lab/{lesson_id}-tpu.txt'"
    scp_fetch = f"gcloud compute tpus tpu-vm scp {tpu_name}:~/jax-tpu-lab/{lesson_id}-tpu.txt ./ --zone={zone}"
    return {"scp_stage": scp_stage, "ssh_run": ssh_run, "scp_fetch": scp_fetch, "multi_host": hosts > 1}

single_cmds = build_tpu_experiment_commands("jax-tpu-course", "us-west1-c", "distributed-02", hosts=1, execution_class="cpu-only")
assert single_cmds["run"] == "python public/exercises/distributed-02.py"
for kwargs in ({}, {"hosts": 4, "execution_class": "cpu-only"}):
    try:
        build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("unreviewed or CPU-only work must not be routed to a TPU Pod")
print("CPU route:", single_cmds["run"])

# Step 3: Verify invariants on the completed state
assert single_cmds["run"] == "python public/exercises/distributed-02.py"
assert single_cmds["multi_host"] is False
try:
    build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", hosts=4, execution_class="cpu-only")
except ValueError:
    print("CPU-only route remains excluded from TPU Pod execution")
else:
    raise AssertionError("CPU-only work must not be routed to a TPU Pod")

import json
import jax
import numpy as np

SLICE_CATALOG = [
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v6e-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 32},
]
for entry in SLICE_CATALOG:
    entry["global_chips"] = entry["hosts"] * entry["chips_per_host"]
    entry["total_hbm_gib"] = entry["global_chips"] * entry["hbm_per_chip_gib"]
    entry["multi_host"] = entry["hosts"] > 1


def build_tpu_experiment_commands(tpu_name, zone, lesson_id, hosts=1, execution_class="review-required"):
    if execution_class == "review-required":
        raise ValueError("classify and review the script before generating a TPU launch command")
    if execution_class == "cpu-only":
        if hosts != 1:
            raise ValueError("CPU-only scripts must use the CPU route")
        return {"run": f"python public/exercises/{lesson_id}.py", "multi_host": False}
    if execution_class == "single-host-compatible" and hosts != 1:
        raise ValueError("single-host scripts cannot run across a TPU Pod")
    if hosts > 1 and execution_class != "multi-host-qualified":
        raise ValueError("multi-host launch requires a qualified training program")
    worker_flag = " --worker=all" if hosts > 1 else ""
    scp_stage = f"gcloud compute tpus tpu-vm scp public/exercises/{lesson_id}.py {tpu_name}:~/jax-tpu-lab/ --zone={zone}{worker_flag}"
    ssh_run = f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone}{worker_flag} --command='set -o pipefail; JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/{lesson_id}.py | tee ~/jax-tpu-lab/{lesson_id}-tpu.txt'"
    scp_fetch = f"gcloud compute tpus tpu-vm scp {tpu_name}:~/jax-tpu-lab/{lesson_id}-tpu.txt ./ --zone={zone}"
    return {"scp_stage": scp_stage, "ssh_run": ssh_run, "scp_fetch": scp_fetch, "multi_host": hosts > 1}

single_cmds = build_tpu_experiment_commands("jax-tpu-course", "us-west1-c", "distributed-02", hosts=1, execution_class="cpu-only")
pod_cmds = None  # This lesson has no multi-host-qualified training program.
# Assert the CPU route does not request TPU execution.
assert single_cmds["run"] == "python public/exercises/distributed-02.py"
try:
    build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", hosts=4, execution_class="cpu-only")
except ValueError:
    print("Correctly rejected CPU-only lesson for TPU Pod execution")
else:
    raise AssertionError("CPU-only lesson must not be routed to TPU Pod")
print("Slice catalog global chips:", {s["slice"]: s["global_chips"] for s in SLICE_CATALOG})
print("CPU route:", single_cmds["run"])

# Figure data experiment
visual_data = {
    'kind': 'bar',
    'labels': [s['slice'] for s in SLICE_CATALOG],
    'xlabel': 'Cloud TPU slice topology',
    'ylabel': 'host / chip count',
    'series': [
        {'label': 'Linux VM hosts', 'y': [float(s['hosts']) for s in SLICE_CATALOG]},
        {'label': 'local chips per host', 'y': [float(s['chips_per_host']) for s in SLICE_CATALOG]},
        {'label': 'global TPU chips', 'y': [float(s['global_chips']) for s in SLICE_CATALOG]},
    ],
}

# Experiment: Compare local vs global device counts across slices
# Experiment — Compare local vs global device counts across slices
v5e16 = next(s for s in SLICE_CATALOG if s["slice"] == "v5litepod-16")
idle_without_worker_all = v5e16["global_chips"] - v5e16["chips_per_host"]
print("v5litepod-16 global chips:", v5e16["global_chips"], "Idle chips without --worker=all:", idle_without_worker_all)
# Assert invariant `idle_without_worker_all == 12` holds
assert idle_without_worker_all == 12

# Experiment: Verify total HBM scaling from v5litepod-16 to v6e-16
# Experiment — Verify total HBM scaling from v5litepod-16 to v6e-16
hbm_by_slice = {s["slice"]: s["total_hbm_gib"] for s in SLICE_CATALOG}
print("Total HBM GiB by slice:", hbm_by_slice)
assert hbm_by_slice["v6e-16"] == 512 and hbm_by_slice["v5litepod-16"] == 256

# Reference solution. Try the exercise before reading this.
cpu_cmds = build_tpu_experiment_commands("jax-tpu-course", "us-west1-c", "distributed-02", hosts=1, execution_class="cpu-only")
print("CPU route:", cpu_cmds["run"])
assert cpu_cmds["run"] == "python public/exercises/distributed-02.py"
for kwargs in ({}, {"hosts": 4, "execution_class": "cpu-only"}):
    try:
        build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", **kwargs)
    except ValueError:
        print("Rejected unsafe route:", kwargs)
    else:
        raise AssertionError("unreviewed or CPU-only work must not be routed to a TPU Pod")

# Reference practice: Filter multi-host slices from the catalog
# Filter multi-host slices from the catalog (Foundations)
pod_slices = [s["slice"] for s in SLICE_CATALOG if s["multi_host"]]
print("Multi-host Pod slices:", pod_slices)
# Assert invariant `pod_slices == ["v5litepod-16"` holds
assert pod_slices == ["v5litepod-16", "v6e-16"]

# Reference practice: Compute per-host vs global batch size on a 4-host Pod slice
microbatch_per_chip = 8
per_host_batch = microbatch_per_chip * 4
global_batch = microbatch_per_chip * 16
print("Per-host batch:", per_host_batch, "Global batch:", global_batch)
# Assert invariant `per_host_batch == 32 and global_batch == 128` holds
assert per_host_batch == 32 and global_batch == 128
print("PASS: tpu-05")
