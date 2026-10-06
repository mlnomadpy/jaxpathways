# Phase 00: Setup & first steps

Start here.

Run a saved program and explain its environment, array output and device placement. Start on one CPU; logical devices are a later exercise in placement.

Keep a first-run log that another learner can reproduce.

**Prerequisites:** Basic Python; no previous JAX experience.

**Hardware:** CPU reference and executable TPU bridge; TPU execution unverified.

## Study guide: Can someone else reproduce your first result?

Start with one saved program and a known answer. The terminal, interpreter and working folder are part of the experiment: a notebook that imports a package is not yet proof that the downloaded script uses the same environment.

### Check your starting point

You saved a script in your course folder, but the terminal reports that the file does not exist. Which observation should you check first?

<details><summary>Compare your reasoning</summary>

Check the terminal’s current folder and the exact filename, including its extension. Then run the environment’s Python explicitly. Reinstalling JAX will not repair a wrong path.

</details>

Review: [Set up your learning workspace](01-set-up-your-learning-workspace/docs/en.md).

### Build in stages

1. **Make one reproducible result.** Run the row-sum example as a saved file. Record the Python executable, package versions, backend, input shape and computed sums. Explain why a package version alone cannot identify which interpreter ran the file.

   Lessons: [Set up your learning workspace](01-set-up-your-learning-workspace/docs/en.md) · [Meet your arrays and devices](02-meet-your-arrays-and-devices/docs/en.md).

2. **Separate local practice from accelerator execution.** Start a fresh process for virtual CPU devices. Compare its device receipt with the ordinary process. Treat the TPU lesson as preparation until you have a receipt from an actual TPU.

   Lessons: [Practice with four virtual CPU devices](03-virtual-cpu-devices/docs/en.md) · [Move your experiment to a TPU](03-move-your-experiment-to-a-tpu/docs/en.md).

### Try a changed condition

A colleague gets the right sums but a different device count. Decide whether the arithmetic is wrong and how to compare the runs.

<details><summary>Compare an approach</summary>

Compare values and shapes first, then backend, process startup flags and environment. Device enumeration is a runtime property; equal arithmetic does not imply equal hardware or performance.

</details>

**Symptom:** Import works in a notebook but fails in the terminal.

**Check next:** Compare interpreter paths before changing packages; launch both from the intended environment.

### Decide what is ready

Keep a saved script, its command and actual output, plus an explanation of one repaired setup error. Use foundation-toolkit stage 1. A screenshot without a rerunnable file is incomplete.

### Further work

Fresh-machine Windows/Linux/macOS walkthroughs and novice usability sessions remain unverified; accelerator setup needs a separately observed device run.

## Lesson sequence

### 00.01 Set up your learning workspace

[Read the lesson](01-set-up-your-learning-workspace/docs/en.md) · [Run the code](01-set-up-your-learning-workspace/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Download a starter workspace, select its Python environment, run a saved program, and verify one deliberate change.

**Evidence:** Keep first_experiment.py, my-first-run.txt with package/device output, the environment interpreter path, and a note deriving the change from a sum of 6 to 15. Record a setup error and its repair if you encountered one.

**Checkpoint:** Which evidence lets another learner reproduce your environment?

### 00.02 Meet your arrays and devices

[Read the lesson](02-meet-your-arrays-and-devices/docs/en.md) · [Run the code](02-meet-your-arrays-and-devices/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Create an array, inspect its dtype and device, and explain the result.

**Evidence:** Keep the labeled table, row/column calculations, centered-data check, array dtype/device report and the failed reshape with its element-count diagnosis.

**Checkpoint:** For an array of shape (2, 3), what shape results from sum(axis=0)?

### 00.03 Practice with four virtual CPU devices

[Read the lesson](03-virtual-cpu-devices/docs/en.md) · [Run the code](03-virtual-cpu-devices/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Start four logical CPU devices, inspect row-sharded and replicated arrays, and diagnose an indivisible batch.

**Evidence:** Keep package/device reports, global and local shapes, shard index/value tables, NumPy comparisons, changed-shape predictions, and the five-row failure/repair. State that the run uses one host and does not validate TPU performance.

**Checkpoint:** What does a (12,2) global array sharded with P("data",None) over four devices mean?

### 00.04 Move your experiment to a TPU

[Read the lesson](03-move-your-experiment-to-a-tpu/docs/en.md) · [Run the code](03-move-your-experiment-to-a-tpu/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Verify the device of a completed prediction and its independent numerical reference. Run an explicit TPU target path that fails when TPU is unavailable.

**Evidence:** Keep separate CPU and TPU reports with requested/observed platform, package versions, completed predictions, NumPy error and changed-bias and row-permutation checks. Until the target command succeeds, TPU execution remains unverified.

**Checkpoint:** The values match NumPy, but the output array is on CPU when the requested platform was TPU. What has been demonstrated?

## Phase project

A reproducible environment report.

**Demonstrate:** Explain which device ran your code and reproduce the result in a fresh environment.

Project status: implemented staged practice · [Open source](../../projects/foundation-toolkit/README.md). Use stages 1 for this phase.



[Primary documentation](https://docs.jax.dev/en/latest/installation.html).
