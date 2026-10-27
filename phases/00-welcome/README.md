# Phase 00: Setup & first steps

Start here.

Run and inspect saved programs in a CPU environment, then learn how to prepare and verify explicit, source-reviewed TPU targets. TPU jobs are optional and separate from the course’s CPU practice.

Start with each lesson’s recorded CPU baseline. For first physical TPU practice, use only the bounded launcher and precision targets in the downloadable `jax-tpu-gcp.zip` workspace with its pinned qualification requirements; inspect their receipts for the observed TPU backend. Review each additional program’s source, dependencies, topology, and data ownership before considering a separate target run.

**Prerequisites:** Basic Python; no previous JAX experience.

**Hardware:** CPU; the TPU bridge is optional accelerator practice.

## Study guide: Can someone else reproduce your first result?

Your first goal is to make one saved program produce an answer you can predict. When a notebook and a terminal disagree, the cause may be the interpreter, working folder or input file before JAX itself enters the picture. Record those choices beside the result so you can explain what actually ran.

Follow the local CPU route first. The optional virtual-device branch later makes array ownership visible; the cloud branch adds an explicitly requested target and its runtime receipt. A matching answer establishes the checked arithmetic, while the backend and device fields explain where it happened.

### Check your starting point

You saved a script in your course folder, but the terminal reports that the file does not exist. Which observation should you check first?

<details><summary>Compare your reasoning</summary>

Check the terminal’s current folder and the exact filename, including its extension. Then run the environment’s Python explicitly. Reinstalling JAX will not repair a wrong path. After the first array lesson, a CPU learner can continue to Arrays & pure functions. Virtual-device and cloud setup can wait until the placement and transformation concepts are familiar.

</details>

Review: [Set up your learning workspace](01-set-up-your-learning-workspace/docs/en.md).

### Build in stages

1. **Make one reproducible result.** Run the row-sum example as a saved file. Record the Python executable, package versions, backend, input shape and computed sums. Explain why a package version alone cannot identify which interpreter ran the file.

   Lessons: [Set up your learning workspace](01-set-up-your-learning-workspace/docs/en.md) · [Meet your arrays and devices](02-meet-your-arrays-and-devices/docs/en.md).

2. **Separate local practice from parallel TPU execution.** Optional advanced branch: start a fresh process for virtual CPU devices and compare row-sharded versus replicated placement receipts. Beginners can continue directly from welcome-02 to Arrays & pure functions on CPU, then return after the transformation and sharding lessons. For physical TPU practice use only the named source-reviewed targets, pinned environments and explicit expected backend/topology checks; classify any additional script before staging it.

   Lessons: [Practice with four virtual CPU devices](03-virtual-cpu-devices/docs/en.md) · [Move your experiment to a TPU](03-move-your-experiment-to-a-tpu/docs/en.md).

### Try a changed condition

You have identical predictions on two computers, but one receipt says CPU and the other TPU. Which statements can you keep?

<details><summary>Compare an approach</summary>

Keep the tested numerical agreement and both environment records. Reject any claim of equal device performance: different compilation, synchronization and timing boundaries could explain differences. Reproduce one changed input before expanding the numerical claim.

</details>

**Symptom:** Import works in a notebook but fails in the terminal.

**Check next:** Compare interpreter paths before changing packages; launch both from the intended environment.

### Decide what is ready

Keep a saved script, its command and actual output, plus an explanation of one repaired setup error. Use foundation-toolkit stage 1. A screenshot without a rerunnable file is incomplete.

### Further work

Fresh-machine Windows/Linux/macOS walkthroughs and novice usability sessions remain unverified; accelerator setup needs a separately observed device run. The optional placement branch is not a prerequisite for ordinary CPU lessons. Before taking it, explain which axis is split, what a replica holds and why jit input shapes affect specialization.

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

**Intended outcome:** Use the recorded CPU reference to check arithmetic and placement, then run only a named, source-reviewed TPU target with pinned dependencies and verify its observed backend receipt.

**Evidence:** Keep the lesson’s CPU output separate from physical TPU evidence. For TPU practice, retain the launcher events/checkpoint, precision report/trace, and resolved environment from the downloadable qualification workspace. The lesson’s recorded `main.py` output is CPU-only evidence.

**Checkpoint:** The values match NumPy, but the output array is on CPU when the requested platform was TPU. What has been demonstrated?

## Phase project

Check the CPU environment and a saved matrix arithmetic identity.

**Demonstrate:** Explain which device ran your code and reproduce the result in a fresh environment.

Project status: implemented staged practice · [Open source](../../projects/welcome-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Work in projects/welcome-audit/starter/model.py. Complete stages 1 and 2 for independent calculations and changed-condition diagnosis, then retain the stage 3 evidence with --evidence-dir runs/welcome-audit. Stage 4 integrates the checks. Learner commands explicitly select the starter; --implementation solution is instructor reference execution. This bounded practice does not certify the entire phase or target hardware.

Additional project: [Connect arrays, transformations and explicit state](../../projects/foundation-toolkit/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/installation.html).
