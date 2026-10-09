# Tools, project structure, and experiments

You have a notebook that works. A week later, you change the data and cannot explain which version produced your best result. The next skill is making the experiment inspectable: someone should be able to find its inputs, run it, and understand what changed.

**Before you start:** complete [workspace setup](lesson.html?lesson=welcome-01). Basic Python and a terminal are enough. This guide is shared preparation for every project, then a reference during training, recovery, and deployment.

**Your result:** a small project with two separate run directories, saved input snapshots, a tested command-line interface (CLI), and a written comparison. The included CLI prepares experiments; training and accelerator execution come from the linked course lessons.

## 1. Give each file a clear responsibility

Begin with the [downloadable workspace](downloads/jax-project-workspace.zip). Extract it and open the `jax-project-workspace` folder in your editor. It contains this layout:

```text
jax-project-workspace/
├── README.md                 how to start and verify
├── GUIDE.md                  this guide, for offline use
├── run.py                    small CLI entry point
├── src/jax_lab/
│   ├── __init__.py
│   └── cli.py                argument validation and run preparation
├── configs/baseline.json     choices for one experiment
├── data/tiny.csv             small, versionable teaching fixture
├── tests/test_workspace.py   provenance and overwrite checks
├── .gitignore
└── runs/                    created by the CLI; ignored by Git
    └── baseline-01/
        ├── config.json      resolved choices used for this run
        ├── manifest.json    input hashes, environment, time and command
        ├── source/          snapshot of this exercise's Python source
        ├── data.snapshot    exact tiny input used for this run
        └── notes.md         hypothesis, observations and decision
```

Read the tree from top to bottom: source describes **how**, configuration describes **which choices**, data supplies **the inputs**, and a run preserves **one attempt**. A new attempt gets a new directory. Changing `configs/baseline.json` later must not rewrite history inside `runs/baseline-01/`.

When the project grows, add `model.py` for the model, `data.py` for loading and splitting, `train.py` for training, and `evaluate.py` for evaluation inside `src/jax_lab/`. Keep `notebooks/` for exploration and explanation; import these functions instead of maintaining a second training loop in notebook cells. Add `reports/` for selected figures and conclusions, and a `checkpoints/` subdirectory within each run for recoverable training state. These are growth steps, not empty folders you need on day one.

**What belongs in Git?** Commit source, small fixtures, configs, tests, dependency declarations/lockfiles, and written decisions. Ignore environments, caches, credentials, large data, and generated runs. Ignored files still need a backup plan: preserve selected evidence in durable artifact storage. `.gitignore` neither backs up files nor removes already tracked files.

## 2. Separate a notebook from a reproducible entry point

A notebook can accidentally depend on a cell executed yesterday. Restart its kernel and run every cell in order. If it fails, identify the missing dependency before exporting the code.

Move reusable functions into the package. Let a thin CLI parse choices and call those functions. The training function should receive its data, configuration and random state explicitly. It should not read arguments or start a tracker when the module is imported. In JAX, keep file writes and logging outside transformed functions; Python side effects inside `jit` can happen at tracing rather than at every update.

The workspace uses Python's built-in [argparse](https://docs.python.org/3/library/argparse.html). Open `src/jax_lab/cli.py` and follow `main()` into `prepare()`: parsing supplies a config path and run ID; validation rejects invalid choices; only then is a new run directory created. `run.py` locates the package relative to its own file, so the command does not depend on a notebook's current directory.

A future training CLI should have explicit `train`, `evaluate`, and `resume` actions. Require the selected checkpoint and evaluation split, and explain invalid arguments. This starter implements only `prepare`; do not mistake a prepared manifest for a completed training result.

## 3. Prepare an experiment you can inspect

Open a terminal in the extracted workspace. On macOS/Linux use the commands below; on Windows PowerShell replace `python3` with `py`. Python 3.10 or later is required. This exercise uses only the standard library, so it needs no package installation, account or accelerator.

```sh
# Run run command in terminal using the course Python environment
python3 --version
python3 run.py --help
python3 run.py prepare --config configs/baseline.json --run-id baseline-01
python3 -m unittest discover -s tests -v
```

Expected result: the prepare command prints `Prepared runs/baseline-01; no training has run.` The test command finishes with `OK`. Open the created `config.json`, `manifest.json` and `notes.md` in your editor. Hashes and timestamps vary with your files and machine; compare identities, not an example timestamp.

The config contains `seed`, `learning_rate`, `steps`, and `data`. The CLI checks their types, bounds and file availability, then saves the actual config. It snapshots the tiny dataset and Python source and records their SHA-256 hashes. A hash is a fingerprint for detecting changes; it does not preserve a missing file, which is why this small exercise also keeps snapshots. Git commit and dirty state are recorded when the extracted folder has its own Git repository. A ZIP without Git history legitimately records `null`.

**Predict:** if you run the prepare command twice with `baseline-01`, should it overwrite, append, or fail? It fails. Open the original manifest afterward: it must be unchanged. That refusal protects the first attempt's evidence.

The starter records Python and operating-system information, not a complete ML environment. Before running JAX, also retain your dependency lock or package inventory, JAX/jaxlib versions, accelerator runtime and device information. A seed alone does not promise bitwise agreement across different backends or precision policies.

## 4. Change one choice and keep the comparison honest

Copy `configs/baseline.json` to `configs/lower-rate.json` in your editor. Change only `learning_rate` from `0.01` to `0.005`. Leave the seed, steps and dataset unchanged.

```sh
# Run run command in terminal using the course Python environment
python3 run.py prepare --config configs/lower-rate.json --run-id lower-rate-01
```

Compare the two manifests. The saved config hashes should differ; the source and data hashes should match. Both runs are still `prepared`. There is no loss curve yet and no evidence that the smaller learning rate is better.

Write your hypothesis in each run's notes before connecting the [training loop lesson](lesson.html?lesson=networks-03). For an actual comparison, make training consume the saved configuration and identified data, record metrics with step numbers, and retain the final checkpoint. Do not save one config and quietly train with defaults from another file.

This diagram is a conceptual workflow, not a measured training result:

```text
versioned source + data + resolved configuration
                        |
                 prepare a new run
                        |
          saved inputs + environment + command
                        |
             execute training and save state
                        |
         held-out evaluation + metrics + report
                        |
             decision with stated limitations
```

The first arrow preserves the question you intended to test. Training creates observations. Evaluation makes those observations comparable. A report records why you accepted or rejected the change. Skipping the input snapshot means a good-looking final chart may have no traceable explanation.

For real runs, distinguish metrics from artifacts. A metric is a named value with a step, split and definition; an artifact is a file such as a checkpoint, plot or evaluation table. State whether loss is per token, example or batch; log the learning rate, gradient norm, non-finite counts and precision. For throughput, record the measured interval and synchronize device work. Keep validation for model selection and reserve the test split for the stated final evaluation. Repeat across training seeds when your conclusion concerns training variability.

## 5. Add tools when they solve a problem

Use the table as a decision guide. The starter above does not require these optional tools. Examples below assume each tool has been installed using its official instructions; check its `--help` if your installed version differs.

| Need | Tool and first action | What to retain |
| --- | --- | --- |
| Recreate a Python environment | [uv](https://docs.astral.sh/uv/guides/projects/): manage `pyproject.toml`, generate a lock, then `uv sync --locked` | Dependency declaration, `uv.lock`, Python version and device setup |
| Inspect changes | Git: `git status --short` and `git diff` from your project root | Reviewed changes and a commit; uncommitted changes also affect results |
| Catch code mistakes | [Ruff](https://docs.astral.sh/ruff/tutorial/): `uv run ruff check src tests` in a uv project with Ruff installed | Lint results; formatting is not a numerical correctness check |
| Check behavior | Starter: `python3 -m unittest discover -s tests -v`; later add model/input/recovery tests | Passing checks for changed inputs and expected failures |
| Compare local training histories | [MLflow](https://mlflow.org/docs/latest/ml/tracking/): follow the linked course lesson for local tracking | Run ID, params, step metrics, data identity, artifacts and selection criteria |
| Share dashboards or run sweeps | [W&B](https://docs.wandb.ai/models/ref/cli): `wandb offline` for local recording with instrumented training | Run ID, logged history, artifact versions and sweep configuration |
| Package the runtime | [Docker](https://docs.docker.com/reference/cli/docker/container/run/): inspect `docker run --help`, then follow the container lesson | Dockerfile, dependency lock, image digest and execution command |
| Retrieve or publish model artifacts | [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli): `hf --help` | Model revision, model card, tokenizer and evaluation report |
| Inspect existing TPU resources | [gcloud](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/list): `gcloud compute tpus tpu-vm list --project PROJECT_ID --zone ZONE` | Actual project, zone, resource and runtime identity |

Use the gcloud example only with your configured account and replace both placeholders. Listing resources does not provision a TPU. CPU success does not validate TPU execution.

For a **new, separate ML repository** with uv already installed, this is an optional environment workflow. Do not run it inside the course checkout to replace its tested environment:

```sh
# Run command in terminal
uv init --lib my-jax-project
cd my-jax-project
uv add jax flax optax
uv add --dev ruff pytest
uv sync --locked
uv run python -c "import jax; print(jax.devices())"
```

These commands resolve current packages and create a lock; they do not reproduce the course's pinned package set. Expect a printed device list, not a guarantee of GPU/TPU availability. Keep the lock in version control and validate the environment you resolved. The [uv project guide](https://docs.astral.sh/uv/guides/projects/) explains dependency management and execution; use JAX's [installation guide](https://docs.jax.dev/en/latest/installation.html) for accelerator-specific requirements.

MLflow and W&B require instrumentation in your training program. Installing a CLI cannot infer the scientific meaning of a run. Start with one tracker and map its run ID to the local run directory. `wandb offline` keeps SDK logs local; a later `wandb sync` uploads them. Choose what to publish deliberately, keeping credentials and restricted datasets out of artifacts.

To navigate the course itself, use its CLI from the **full course workspace or checkout**, not from the small project-organizing download above:

```sh
# Run run command in terminal using the course Python environment
python3 scripts/course.py list --available
python3 scripts/course.py show recovery-06
python3 scripts/course.py plan training-engineer
```

These commands list available lessons, display the MLflow lesson, and print a proposed career route. See [terminal help](developer.html#terminal) for running companions. The course reference runner includes solutions; successful execution does not mean you completed the learner exercise.

## 6. Grow toward a training harness

A harness connects data, a model, training, evaluation and state recovery through repeatable interfaces. Use the same separation for text, images, audio and cross-modal work. The shared run manifest stays stable; each modality supplies its own preprocessing contract, data revisions, objective and evaluation protocol.

A config is a record of intent, not a checkpoint. Exact continuation can require model and optimizer state, random keys, scheduler step, data iterator position and precision settings. Save these together and test an interrupted run against uninterrupted work. A weights-only export is appropriate for some inference uses but cannot establish training continuation.

Keep deployment artifacts separate from training checkpoints. An inference bundle needs the expected input/output contract, preprocessing/tokenizer, weights, precision, runtime dependencies and validation evidence. A container packages software; it does not automatically include your data or prove compatible accelerator drivers.

Continue with [MLflow lineage](lesson.html?lesson=recovery-06), the [recovery phase guide](phase-guides/recovery.html), and the [operations phase guide](phase-guides/operations.html). Use the [modality projects](projects.html#modality-tracks) to apply the same structure to different model families.

## 7. Diagnose the failure before adding another tool

| Symptom | Evidence to inspect | Repair |
| --- | --- | --- |
| `run.py` not found | Current folder and extracted directory | Open the terminal in `jax-project-workspace`, or use the full path to `run.py` |
| Run ID already exists | Existing manifest and notes | Keep the old run; choose a new descriptive ID |
| Invalid config or missing data | Error message, key names, value types and relative path | Repair the config before preparing a run |
| Same run label, different results | Source/data hashes, resolved config, runtime, precision and random state | Compare identities; a human-readable label is not proof of equivalence |
| Notebook works, script fails | Restarted notebook, interpreter and import paths | Move hidden state into explicit inputs and use the same environment |
| Tracker shows a curve but no reproducible run | Saved input identities and artifact locations | Log or archive the missing provenance; a dashboard is not a backup |
| Resume changes the next update | Optimizer, random state, step and data position | Restore full training state and compare the next update |

## 8. Verify your understanding and keep evidence

Try these without changing the reference CLI first:

1. Prepare two runs whose only intended difference is the learning rate. Explain which manifest fields should match and which should differ.
2. Change one row of `data/tiny.csv`, then prepare a third run. Confirm that its data hash changes while the earlier data snapshot stays unchanged.
3. Set `steps` to zero in a new config. The command should reject it without creating that run directory. Explain why failing before execution is useful.
4. Rerun the tests. Inspect the test that launches the CLI from a different working folder: why does locating files relative to the workspace matter?

Reference reasoning: learning-rate changes alter the config identity, not source or data; changing data alters its fingerprint; invalid steps cannot describe this exercise's valid run; a package's location is more reliable than an accidental notebook working directory. Timestamps, run IDs and commands naturally differ between attempts.

Keep both original run directories, the changed-data run, test output and your completed notes. Your explanation should identify the controlled variable, the limits of the environment record, and the difference between **prepared**, **executed**, and **evaluated**. These checks establish experiment organization. The linked numerical lessons establish whether the model itself works.
