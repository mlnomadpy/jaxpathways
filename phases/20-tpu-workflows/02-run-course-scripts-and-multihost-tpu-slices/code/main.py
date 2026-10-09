"""Run course scripts and multi-host TPU Pod slices: worked experiments and reference solutions. CPU checks."""

# Tabulate single-host and multi-host TPU slice topologies
# Step 1 — Tabulate single-host and multi-host TPU slice topologies: Separating hosts from chips_per_host shows immediately when a...
# Import json for this computation.
import json
import jax
import numpy as np

# Evaluate `SLICE_CATALOG` from the current inputs and state.
SLICE_CATALOG = [
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v6e-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 32},
]
# Iterate over `entry` to step through the computation:
for entry in SLICE_CATALOG:
    # Evaluate `entry['global_chips']` from the current inputs and state.
    entry["global_chips"] = entry["hosts"] * entry["chips_per_host"]
    # Evaluate `entry['total_hbm_gib']` from the current inputs and state.
    entry["total_hbm_gib"] = entry["global_chips"] * entry["hbm_per_chip_gib"]
    # Evaluate `entry['multi_host']` from the current inputs and state.
    entry["multi_host"] = entry["hosts"] > 1

# Generate single-host and multi-host gcloud command pipelines
# Step 2 — Generate single-host and multi-host gcloud command pipelines: Generating the stage, run, and fetch commands from the host count...
def build_tpu_experiment_commands(tpu_name, zone, lesson_id, hosts=1):
    # Evaluate `worker_flag` from the current inputs and state.
    worker_flag = " --worker=all" if hosts > 1 else ""
    # Evaluate `scp_stage` from the current inputs and state.
    scp_stage = f"gcloud compute tpus tpu-vm scp public/exercises/{lesson_id}.py {tpu_name}:~/jax-tpu-lab/ --zone={zone}{worker_flag}"
    # Evaluate `ssh_run` from the current inputs and state.
    ssh_run = f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone}{worker_flag} --command='JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/{lesson_id}.py'"
    # Evaluate `scp_fetch` from the current inputs and state.
    scp_fetch = f"gcloud compute tpus tpu-vm scp --recurse {tpu_name}:~/jax-tpu-lab/ ./tpu-runs/{lesson_id}/ --zone={zone}"
    # Return `{'scp_stage': scp_stage, 'ssh_run': ssh_run, 'scp_fetch': scp_fetch, 'multi_host': hosts > 1}` to the caller.
    return {"scp_stage": scp_stage, "ssh_run": ssh_run, "scp_fetch": scp_fetch, "multi_host": hosts > 1}


# Run `build_tpu_experiment_commands` to compute `single_cmds`.
single_cmds = build_tpu_experiment_commands("jax-tpu-course", "us-west1-c", "distributed-02", hosts=1)
# Run `build_tpu_experiment_commands` to compute `pod_cmds`.
pod_cmds = build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", hosts=4)
# Verify contract: `'--worker=all' not in single_cmds['ssh_run']`.
assert "--worker=all" not in single_cmds["ssh_run"]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert "--worker=all" in pod_cmds["scp_stage"] and "--worker=all" in pod_cmds["ssh_run"]
# Print the observed values to compare against the expected result.
print("Slice catalog global chips:", {s["slice"]: s["global_chips"] for s in SLICE_CATALOG})
# Print diagnostic summary of the computed outputs.
print("Pod SSH command:", pod_cmds["ssh_run"])

# Run course scripts and multi-host TPU Pod slices: Every lesson in this course generates a self-contained script in...
# Import json for this computation.
import json
import jax
import numpy as np

# Evaluate `SLICE_CATALOG` from the current inputs and state.
SLICE_CATALOG = [
    {"slice": "v5litepod-4", "hosts": 1, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-8", "hosts": 1, "chips_per_host": 8, "hbm_per_chip_gib": 16},
    {"slice": "v5litepod-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 16},
    {"slice": "v6e-16", "hosts": 4, "chips_per_host": 4, "hbm_per_chip_gib": 32},
]
# Iterate over `entry` to step through the computation:
for entry in SLICE_CATALOG:
    # Evaluate `entry['global_chips']` from the current inputs and state.
    entry["global_chips"] = entry["hosts"] * entry["chips_per_host"]
    # Evaluate `entry['total_hbm_gib']` from the current inputs and state.
    entry["total_hbm_gib"] = entry["global_chips"] * entry["hbm_per_chip_gib"]
    # Evaluate `entry['multi_host']` from the current inputs and state.
    entry["multi_host"] = entry["hosts"] > 1


# Function `build_tpu_experiment_commands(tpu_name, zone, lesson_id, hosts)` implementing this stage's computation:
def build_tpu_experiment_commands(tpu_name, zone, lesson_id, hosts=1):
    # Evaluate `worker_flag` from the current inputs and state.
    worker_flag = " --worker=all" if hosts > 1 else ""
    # Evaluate `dist_init` from the current inputs and state.
    dist_init = "import jax; jax.distributed.initialize(); " if hosts > 1 else ""
    # Evaluate `scp_stage` from the current inputs and state.
    scp_stage = f"gcloud compute tpus tpu-vm scp public/exercises/{lesson_id}.py {tpu_name}:~/jax-tpu-lab/ --zone={zone}{worker_flag}"
    # Read or serialize artifact data on disk (`ssh_run`).
    ssh_run = (
        f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone}{worker_flag} "
        f"--command=\"JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python -c '{dist_init}exec(open(\\\"PosixPath\\\"))'\""
        if False
        else f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone}{worker_flag} --command='JAX_PLATFORMS=tpu ~/jax-tpu-lab/.venv/bin/python ~/jax-tpu-lab/{lesson_id}.py'"
    )
    # Evaluate `scp_fetch` from the current inputs and state.
    scp_fetch = f"gcloud compute tpus tpu-vm scp --recurse {tpu_name}:~/jax-tpu-lab/ ./tpu-runs/{lesson_id}/ --zone={zone}"
    # Return `{'scp_stage': scp_stage, 'ssh_run': ssh_run, 'scp_fetch': scp_fetch, 'multi_host': hosts > 1}` to the caller.
    return {"scp_stage": scp_stage, "ssh_run": ssh_run, "scp_fetch": scp_fetch, "multi_host": hosts > 1}


# Run `build_tpu_experiment_commands` to compute `single_cmds`.
single_cmds = build_tpu_experiment_commands("jax-tpu-course", "us-west1-c", "distributed-02", hosts=1)
# Run `build_tpu_experiment_commands` to compute `pod_cmds`.
pod_cmds = build_tpu_experiment_commands("jax-tpu-pod", "us-west1-c", "distributed-02", hosts=4)
# Verify contract: `'--worker=all' not in single_cmds['ssh_run']`.
assert "--worker=all" not in single_cmds["ssh_run"]
# Verify that the output satisfies the expected shape, finite-value, or numerical contract.
assert "--worker=all" in pod_cmds["scp_stage"] and "--worker=all" in pod_cmds["ssh_run"]
# Print the observed values to compare against the expected result.
print("Slice catalog global chips:", {s["slice"]: s["global_chips"] for s in SLICE_CATALOG})
# Print diagnostic summary of the computed outputs.
print("Pod SSH command:", pod_cmds["ssh_run"])

# Figure data experiment
# Compute figure data for: Host count, per-host local chips, and global TPU chips across single-host and Pod slices
# Evaluate `visual_data` from the current inputs and state.
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
# Experiment — Compare local vs global device counts across slices: Worker 0 only controls its own 4 local chips; the other 3 hosts...
v5e16 = next(s for s in SLICE_CATALOG if s["slice"] == "v5litepod-16")
# Evaluate `idle_without_worker_all` from the current inputs and state.
idle_without_worker_all = v5e16["global_chips"] - v5e16["chips_per_host"]
# Print the observed values to compare against the expected result.
print("v5litepod-16 global chips:", v5e16["global_chips"], "Idle chips without --worker=all:", idle_without_worker_all)
# Verify contract: `idle_without_worker_all == 12`.
assert idle_without_worker_all == 12

# Experiment: Verify total HBM scaling from v5litepod-16 to v6e-16
# Experiment — Verify total HBM scaling from v5litepod-16 to v6e-16: Doubling per-chip HBM on Trillium (v6e) doubles the total...
hbm_by_slice = {s["slice"]: s["total_hbm_gib"] for s in SLICE_CATALOG}
# Print the observed values to compare against the expected result.
print("Total HBM GiB by slice:", hbm_by_slice)
# Verify contract: `hbm_by_slice['v6e-16'] == 512 and hbm_by_slice['v5litepod-16'] == 25...`.
assert hbm_by_slice["v6e-16"] == 512 and hbm_by_slice["v5litepod-16"] == 256

# Reference solution. Try the exercise before reading this.
# Exercise solution: Call build_tpu_experiment_commands('jax-tpu-pod', 'us-east5-a',...
tx_cmds = build_tpu_experiment_commands("jax-tpu-pod", "us-east5-a", "transformers-04", hosts=4)
# Print the observed values to compare against the expected result.
print("Stage:", tx_cmds["scp_stage"])
# Print diagnostic summary of the computed outputs.
print("Run:", tx_cmds["ssh_run"])
# Verify contract: `'--worker=all' in tx_cmds['scp_stage'] and '--worker=all' in tx_cmds...`.
assert "--worker=all" in tx_cmds["scp_stage"] and "--worker=all" in tx_cmds["ssh_run"] and "transformers-04.py" in tx_cmds["ssh_run"]

# Reference practice: Filter multi-host slices from the catalog
# Filter multi-host slices from the catalog (Foundations): Both 16-chip slices span 4 hosts (4 chips per host) and...
pod_slices = [s["slice"] for s in SLICE_CATALOG if s["multi_host"]]
# Print the observed values to compare against the expected result.
print("Multi-host Pod slices:", pod_slices)
# Verify contract: `pod_slices == ['v5litepod-16', 'v6e-16']`.
assert pod_slices == ["v5litepod-16", "v6e-16"]

# Reference practice: Compute per-host vs global batch size on a 4-host Pod slice
# Compute per-host vs global batch size on a 4-host Pod slice (Transfer / diagnosis): Each host feeds its 4 local chips (32 sequences per host),...
microbatch_per_chip = 8
# Evaluate `per_host_batch` from the current inputs and state.
per_host_batch = microbatch_per_chip * 4
# Evaluate `global_batch` from the current inputs and state.
global_batch = microbatch_per_chip * 16
# Print the observed values to compare against the expected result.
print("Per-host batch:", per_host_batch, "Global batch:", global_batch)
# Verify contract: `per_host_batch == 32 and global_batch == 128`.
assert per_host_batch == 32 and global_batch == 128
print("PASS: tpu-05")
