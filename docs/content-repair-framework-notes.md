# Framework and deployment exercise repair notes

Scope: P2 #83 framework parity extension and deployment-07 changed-artifact exercise. Lesson source files updated: deployment-01, deployment-08, deployment-07. Added the optional bounded PyTorch-to-NNX fixture at `phases/15-deployment/08-pytorch-to-flax-numerical-parity/labs/transformer_bridge.py`.

## Framework fixture

The optional fixture creates a local PyTorch `state_dict`, saves and reloads it, then maps it into an Flax NNX single-block decoder matching the course pre-normalized attention + residual + pre-normalized GELU MLP + residual pattern. It binds a toy vocabulary processor identity, explicit nonsequential position IDs, valid-key padding mask semantics, tied token/output embeddings, and the source-to-target tensor layout. Width 8, two heads, MLP width 16, one block, seven token IDs. No production checkpoint or model download is required. The pin set is `jax==0.9.2`, `flax==0.12.6`, `torch==2.11.0`, `numpy==2.4.4`; framework installation stays optional.

Exact CPU command used:

```sh
JAX_PLATFORMS=cpu /private/tmp/jaxpathways-0170-clean-learner-env/bin/python phases/15-deployment/08-pytorch-to-flax-numerical-parity/labs/transformer_bridge.py
```

Output evidence: all ordered boundaries passed; maximum observed activation absolute error was `1.9073486328125e-06` at logits with relative-L2 `9.659265462336476e-08`; nonuniform VJP max absolute error was `4.76837158203125e-07`. The deliberate same-shape head-layout and mask-semantics controls first diverged at `attention` (absolute errors `1.2339559942483902` and `1.439870148897171`). A separate Python child loaded the saved converted artifact and reproduced logits exactly. The declared fixture tolerance is elementwise `3e-6 + 3e-5*abs(reference)`, chosen to cover observed float32 CPU operation-order differences while remaining far below the semantic fault errors.

Learner setup and run commands are stored in deployment-08 lesson JSON. They create an isolated `.venv-transformer-parity` from `python3` and install the four exact pins before running the script with `JAX_PLATFORMS=cpu`.

## Changed artifact exercise

The deployment-07 solution retains the old expected digest, writes changed model bytes to `MODEL_PATH`, and submits a valid request to the actual service source in a new Python process. It asserts nonzero exit, the integrity rejection, and empty stdout, so no prediction is activated or returned.

Exact generated-lesson check used: import `lessonScript` from `scripts/lesson-artifacts.mjs`, write the deployment-07 lesson script to `/private/tmp/jaxpathways-deployment07-repaired.py`, then run `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python /private/tmp/jaxpathways-deployment07-repaired.py`. It exited 0, printed `Changed artifact reached fresh-process loader: rejected before prediction; no output.`, and ended with `PASS: deployment-07`.

## Limits

These results establish arithmetic and serialization parity for one synthetic decoder contract. They do not qualify arbitrary architecture families, tokenizers, production checkpoints, dtype policies, accelerators, or serving containers. Optional PyTorch/Flax dependencies are needed only for the new fixture. No paid model or cloud run was used.
