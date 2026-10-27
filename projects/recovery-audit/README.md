# Resume a complete training stream

A saved parameter vector is not a saved training run. Momentum changes the next update, a random key changes the next perturbation, and a cursor only makes sense when it still points into the same ordered data. This project turns those facts into a small but real protocol: deterministic sample order, nondivisible epoch tails, complete state, fresh-process restoration, interrupted generation writes, and two writers competing for one expected parent.

The fixture is deliberately tiny. It teaches local process and filesystem behavior; it does not qualify power-loss durability, network filesystems, multiple hosts, accelerators, or a production checkpoint service.

## What the state means

The two trainable values are \(w\), and `velocity` is the momentum buffer \(v\). For a batch with targets \(y_i\), this example uses

\[
g = \frac{1}{B}\sum_i (w-y_i) + \sigma\epsilon,\qquad
v' = \mu v + g,\qquad
w' = w - \eta v'.
\]

Here \(B\) is the actual batch size, including a short final batch, \(\mu\) is momentum, \(\eta\) is the learning rate, and \(\epsilon\) is generated from the explicit saved key. `optimizer_step`, `epoch`, `cursor`, `epoch_order`, `source_ids`, `consumed_ids`, and `update_ends` record which data made those updates. The current generation also binds the configuration and caller-expected candidate, dataset, ordered-ID, and training-source identities. The caller config includes `parameter_shape: [2]`, so both parameter and momentum arrays must be exactly two float32 values. A schema copied from malformed bytes cannot authorize a new shape.

With five examples and batch size three, the epoch has update sizes three and two. The reference run selected IDs `[41, 55, 18]`, then the tail `[93, 7]`, then after restoring at cursor five selected the next epoch's first batch `[7, 18, 93]`. A checkpoint after update two therefore preserves a nonempty short tail and still knows that the next update begins a new epoch.

The checkpoint store writes a new immutable generation first. Its arrays and manifest are synced before an atomic `CURRENT.json` replacement exposes that generation. A POSIX `fcntl.flock` protects a compare-and-swap against the expected parent: among two writers that began at the same generation, one publishes and the other reports a conflict. Hashes detect accidental mismatch; they do not authenticate data. These ordinary process tests are not a claim about sudden power loss.

## Before you start

You need Python 3 and NumPy. From the repository root, make a virtual environment if needed:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-beginner.txt
```

On Windows PowerShell, use `py -3 -m venv .venv` and `.\.venv\Scripts\python.exe -m pip install -r requirements-beginner.txt`. Stages 1–3 use platform-compatible Python and NumPy. The immutable-generation writer in stages 5–7 uses POSIX `fcntl.flock`; therefore stage 4, stages 5–7, `all`, and `default` are POSIX-only in this project. Windows locking has not been implemented or tested. All temporary files stay on the selected local filesystem.

## Build and verify

Start with the learner file. Its TODOs are expected to fail until you implement them:

```sh
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 1
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 5
```

Implement the original three APIs in `starter/model.py` first: `train_step`, `save_checkpoint`, and `load_checkpoint`. Stages 1–4 retain their original commands and meaning. Then implement `initial_stream_state`, `stream_step`, `save_generation`, and `restore_generation` for the full training cursor. Stages 5, 6, and 7 check fresh-process replay, crash boundaries, and concurrent writers.

Run each stage and then the full reference suite:

```sh
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 1
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 2
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 3 --evidence-dir /tmp/recovery-audit-evidence
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 5
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 6
python3 projects/recovery-audit/tests/check.py --implementation starter --stage 7
python3 projects/recovery-audit/tests/check.py --implementation starter --stage all --evidence-dir /tmp/recovery-audit-evidence
python3 projects/recovery-audit/tests/check.py --implementation solution --stage all --evidence-dir /tmp/recovery-audit-reference
```

The checker also accepts an external candidate file, for example `--implementation /path/to/model.py`. `--scratch-root /path/to/scratch` optionally chooses a parent for generated temporary workspaces. If omitted, the system temporary directory is used. The full run retains `protocol-evidence.json` with source hashes, process IDs and exit codes, stage results, and explicit protocol limits.

## Read the checks

Stage 1 compares one update against a separate calculation and checks two directions of the squared-error gradient at two finite-difference step sizes. Stage 2 rebuilds the sample order and update independently, so a matching checkpoint replay cannot hide a wrong next batch. Stage 3 rejects changed configuration and changed bytes through the original single-file API. Stage 5 also tests caller-expected source/config identity, parameter and optimizer shapes, scalar clock rank, vector history rank, and rechecksummed epoch/order forgeries.

Stage 5 saves after the two-update tail in one process, starts a fresh Python process, restores, takes the next epoch batch, and compares every state field with an uninterrupted independent trajectory. Stage 6 exits worker processes at a partial temporary write, after payload completion, after manifest completion, and after publication. Before publication, the prior generation must remain readable. After publication, the complete new generation must load. Stage 7 releases two separate processes from a rendezvous with the same expected parent and checks for one published winner and one explicit conflict.

The full command also runs owned mutations for missing optimizer momentum, repeated random state, a changed sample order, and a weight-only rechecksummed checkpoint. A no-op mutation is a checker self-test: it must make the negative-control test fail, while an unchanged reference must pass the positive control. Rechecksummed wrong-shape optimizer state, nonscalar clocks, wrong sample order, and an impossible epoch are also rejected. The grader creates mutations from the reference source; it does not rewrite a learner's implementation. A harmlessly renamed and reformatted equivalent implementation should still pass based on its behavior.

## Changed-condition exercise

Before changing the data, predict what would happen if you keep the file path and configuration but replace one sample ID or reorder `source_ids`. Explain which identity must change and why the old cursor cannot safely name a batch in the new stream. Then change `batch_size` from three to two and continue from an old generation. Inspect the first rejected contract and compare the independent batch boundaries. Save the changed and original identity hashes, command, output, and your explanation in the evidence directory.

## Keep the evidence honest

Record the exact candidate and checker hashes, command, Python/NumPy versions, process IDs, exit codes, published generation IDs, and the uninterrupted-versus-resumed comparison. Stages 5–7 use POSIX `fcntl.flock` and were run on one local host/filesystem. Windows lock behavior is not implemented or tested. These tests do not show behavior under power loss, a remote filesystem, multiple machines, worker loss, a real accelerator job, or a production serving system. Phase 06's Orbax lessons provide a separate framework-oriented recovery path.
