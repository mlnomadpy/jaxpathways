# From a laptop experiment to a TPU training run

Your training script works on your laptop. What has to change before you can run it on a TPU, recover its state, and show that the cloud resources are gone afterward?

This guide connects those steps. You will produce a **run dossier**: configuration, environment, backend evidence, training events, a checkpoint, a recovery comparison, and resource cleanup evidence. Begin with basic Python and the [state and recovery lessons](course.html?phase=recovery). The [project organization guide](project-workflow.html) explains where to keep source, configuration, and run evidence.

The downloadable launcher reuses the course's [workload operations project](project.html?id=workload-operations). Its CPU execution and recovery are tested. The cloud commands are documentation-checked instructions, not a recorded TPU run. This small single-process workload does not shard work across all visible chips or establish production model quality.

After recovery practice, continue with [TPU generations, precision and profiling](tpu-performance.html). That guide compares training and serving workloads, budgets model and cache memory, and measures real numerical error before interpreting a target trace.

## 1. Separate the resource from the job

Before looking at commands, predict what happens to a VM when Python finishes successfully.

```text
LAPTOP                         CLOUD VM                     SAVED EVIDENCE
gcloud creates a resource ---> Python launches a worker ---> events + checkpoint
                               worker exits                 copy to laptop/storage
gcloud deletes the resource -> VM is removed                 evidence remains
```

Read the arrows in order. Creating the VM gives your process somewhere to run. The worker's exit ends the training job, but does not delete the VM. Copying evidence makes it independent of the VM's lifetime. Deletion is a separate operation. This is a conceptual lifecycle diagram, not a timing measurement.

| Layer | Question it answers | Evidence |
| --- | --- | --- |
| Cloud resource | Where can the process run? | Project, zone, VM ID, creation/deletion status |
| Python process | Did training complete within its limit? | Exit status and `run.json` |
| JAX backend | Which backend actually performed the work? | `started` event with backend and visible devices |
| Experiment | Can another process continue the same state? | Configuration, checkpoint and recovery comparison |

There are three different limits: a resource's lifetime, a Python process timeout, and your spending allowance. None substitutes for the other two. Billing budgets send alerts; they do not cap spending. See [Google Cloud budget behavior](https://docs.cloud.google.com/billing/docs/how-to/budgets).

## 2. Rehearse locally and inspect the folder

Download the [practice workspace](downloads/jax-tpu-gcp.zip) and extract it. Open a terminal in the extracted `jax-tpu-gcp` directory. These commands use a Bash-compatible shell on macOS or Linux. Windows learners can use WSL. Follow the [setup lesson](lesson.html?lesson=welcome-01) if Python or your terminal is unfamiliar.

```text
jax-tpu-gcp/
  GUIDE.md
  resources/tpu-gcp/
    launch.py          # explicit backend, timeout, resume, evidence preservation
    check.py           # CPU recovery and failure checks
  projects/workload-operations/solution/
    model.py           # shared numerical worker, not a second implementation
  runs/                # created by you, excluded from source control
```

Create a separate environment on your laptop, then launch eight updates:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'jax==0.9.2' 'numpy==2.4.4'
.venv/bin/python resources/tpu-gcp/launch.py \
  --platform cpu --run-dir runs/cpu-first --steps 8 --timeout 180
```

These are the course's tested CPU package versions. The guide's cloud environment is installed separately using the current TPU installation instructions. Do not copy a laptop virtual environment to the VM.

The worker fits a line to synthetic examples with momentum updates. Its purpose here is to expose state and process behavior with very little numerical work. Open `runs/cpu-first/summary.json`: expect `status` to be `completed`, `updates` to be 8 and `checkpoint_step` to be 8. The exact duration depends on your machine. Run the command again with the same folder: it should refuse to overwrite evidence.

Each run contains `config.json`, `environment.json`, `worker.py`, `events.jsonl`, `run.json`, `summary.json`, and, once saved, `checkpoint.json`. The checkpoint stores parameters, momentum, the random key, step, and source/configuration/data hashes. The launcher collects child-process events when the worker exits. It does not stream live training metrics. A long silent period is not proof of progress; the timeout bounds the wait and retains diagnostics.

## 3. Recover before you rent hardware

Predict whether four updates followed by a restart to step eight will equal an uninterrupted eight-update run. What state would you lose if you saved only the parameters?

```sh
.venv/bin/python resources/tpu-gcp/launch.py \
  --platform cpu --run-dir runs/recovery --steps 4
.venv/bin/python resources/tpu-gcp/launch.py \
  --platform cpu --run-dir runs/recovery --steps 8 --resume
.venv/bin/python resources/tpu-gcp/check.py
```

`--steps 8` means stop at total step eight, so the second command performs four new updates. Earlier evidence is copied into `history/attempt-0001/` before the next attempt. The integration check compares the complete restored state with an uninterrupted run, including the random key and momentum. Equal parameters alone would miss a different next minibatch or optimizer state.

Try `--timeout 0.001` in a fresh folder. Expect a nonzero exit and `timed_out`, not a successful training result. Inspect `run.json` even when no checkpoint exists. A checkpoint cannot recover updates that never reached a save. The timeout only stops this local worker; it makes no cloud API call.

## 4. Choose a cloud project and an access route

A project groups your resources and permissions. A billing account pays for usage. Quota limits what you may request; available capacity determines whether a request can be fulfilled. Credits are a billing arrangement, not proof of permission, quota or available hardware.

Install the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install). On your **laptop**, authenticate and identify the project approved for your experiment:

```sh
gcloud auth login
gcloud auth list
export PROJECT_ID='REPLACE_WITH_YOUR_PROJECT'
gcloud projects describe "$PROJECT_ID"
gcloud billing projects describe "$PROJECT_ID"
gcloud services list --enabled --project "$PROJECT_ID"
gcloud version
```

Save the selected project and CLI version in your dossier. A permission error means you need help from the project administrator, not a different random project ID. Before provisioning, confirm billing, applicable credits, quota, network/SSH access, and an approved VM service account. Your login identity creates resources; the VM service account is the identity available to code inside the VM. Ask for the permissions needed for that task rather than granting broad project ownership. See [project setup and required access](https://docs.cloud.google.com/tpu/docs/setup-gcp-account).

Use **Compute Engine** for the walkthrough below. Google now recommends it or GKE for new TPU resource management. Older project or programme instructions may use the **Cloud TPU API** and queued resources instead. Those use `gcloud compute tpus ...`, different resource names and different cleanup commands. Follow the approved route for your allocation; do not substitute one command family into the other. The distinction is documented in [Cloud TPU API guidance](https://docs.cloud.google.com/tpu/docs/request-using-flex-start).

## 5. Request one host with a bounded lifetime

This step can create billable infrastructure. First agree on a maximum resource lifetime and a separate deadline after which you will cancel an unfulfilled request. Keep this exercise to **one single-host TPU VM**. Multi-host slices need distributed initialization and coordinated launch, which belong in the [distributed phase](course.html?phase=distributed).

On your laptop, fill in values approved for your project and available in the [current TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones). Choose a matching single-host machine type and OS image from the [Compute Engine Flex-start guide](https://docs.cloud.google.com/tpu/docs/create-flex-start-compute). Capacity is not guaranteed.

```sh
export REGION='REPLACE_WITH_APPROVED_REGION'
export ZONE='REPLACE_WITH_APPROVED_ZONE'
export MACHINE_TYPE='REPLACE_WITH_SINGLE_HOST_TPU_MACHINE_TYPE'
export IMAGE_FAMILY='REPLACE_WITH_MATCHING_TPU_IMAGE_FAMILY'
export IMAGE_PROJECT='REPLACE_WITH_IMAGE_PROJECT'
export SERVICE_ACCOUNT='REPLACE_WITH_APPROVED_VM_SERVICE_ACCOUNT'
export SUBNET='REPLACE_WITH_APPROVED_SUBNETWORK'
export TEMPLATE='jax-practice-template-01'
export GROUP='jax-practice-group-01'
export RUN_DURATION='30m'

gcloud services enable compute.googleapis.com --project "$PROJECT_ID"
gcloud compute instance-templates create "$TEMPLATE" \
  --project "$PROJECT_ID" --region "$REGION" \
  --machine-type "$MACHINE_TYPE" \
  --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
  --subnet "$SUBNET" --service-account "$SERVICE_ACCOUNT" \
  --scopes cloud-platform --provisioning-model FLEX_START \
  --instance-termination-action DELETE --max-run-duration "$RUN_DURATION" \
  --maintenance-policy TERMINATE

gcloud compute instance-groups managed create "$GROUP" \
  --project "$PROJECT_ID" --zone "$ZONE" --size 1 \
  --template "projects/$PROJECT_ID/regions/$REGION/instanceTemplates/$TEMPLATE"
```

The template records how to create a VM; the managed instance group requests it. The OAuth scope permits use of APIs, while IAM still controls what the service account can access. The illustrative 30-minute duration includes setup, package installation, training, and copying evidence. It is not an estimate of training time or an assurance that the experiment fits your allowance. Review the [template flag reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instance-templates/create) with your administrator, including network policy.

Inspect the group and record the instance name once it is running:

```sh
gcloud compute instance-groups managed describe "$GROUP" \
  --project "$PROJECT_ID" --zone "$ZONE" --format=json
gcloud compute instance-groups managed list-instances "$GROUP" \
  --project "$PROJECT_ID" --zone "$ZONE"
export VM='REPLACE_WITH_INSTANCE_NAME_FROM_OUTPUT'
gcloud compute instances describe "$VM" \
  --project "$PROJECT_ID" --zone "$ZONE" --format=json
```

An accepted request is not a ready runtime. Inspect status and errors before trying SSH. If your waiting deadline expires, delete the group as shown below even if training never started. The [list-instances reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instance-groups/managed/list-instances) explains the resource inspection command. A managed group can maintain capacity independently of your Python process, so the end of a job is not your cleanup signal from the cloud.

## 6. Launch inside the VM and verify the backend

On your **laptop**, from the folder containing the downloaded ZIP, upload the same workspace you rehearsed:

```sh
gcloud compute scp jax-tpu-gcp.zip "$VM:~/jax-tpu-gcp.zip" \
  --project "$PROJECT_ID" --zone "$ZONE"
gcloud compute ssh "$VM" --project "$PROJECT_ID" --zone "$ZONE"
```

Use your organization's approved SSH or IAP configuration if direct access is unavailable. Do not disable host verification to hide an access error. See the [SSH](https://docs.cloud.google.com/sdk/gcloud/reference/compute/ssh) and [SCP](https://docs.cloud.google.com/sdk/gcloud/reference/compute/scp) references.

The following commands run **inside the VM**, in a Bash-compatible shell:

```sh
python3 -m zipfile -e jax-tpu-gcp.zip .
cd jax-tpu-gcp
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade 'jax[tpu]'
.venv/bin/python -m pip freeze > tpu-environment.txt
set -o pipefail
.venv/bin/python resources/tpu-gcp/launch.py \
  --platform tpu --run-dir runs/tpu-first --steps 8 --timeout 180 \
  2>&1 | tee tpu-launch.log
```

Use a Python version supported by the selected JAX release and the [current JAX TPU installation instructions](https://docs.jax.dev/en/latest/installation.html). If the image lacks `venv`, install the matching Python venv package using the image's administrator-approved setup. Record the resolved environment, then pin it for subsequent comparisons. These cloud install commands have not been qualified on TPU hardware by this course.

The explicit platform sets `JAX_PLATFORMS=tpu` in the child. A missing TPU runtime must fail rather than produce a CPU result labelled as TPU. Check the `started` event in `events.jsonl` for `backend: tpu`, then check `completed` and the checkpoint. Device count reports visibility; this worker does not distribute its arrays across every visible device.

Read `compile_warmup_s` separately from update timing. `job_examples_per_s` includes process startup, compilation and checkpoint overhead. `update_examples_per_s` measures synchronized updates. Their difference is useful here, but neither measures TPU utilization. This tiny line fit is a runtime exercise, not evidence that TPUs accelerate your real model.

Repeat the four-step recovery exercise on the same target backend in a new folder. Compare same-backend uninterrupted and resumed states. Do not demand bitwise equality between CPU and TPU arithmetic. For a cross-backend comparison, first define a numerical error budget and evaluate held-out outputs as taught in [numerical parity](lesson.html?lesson=deployment-08).

## 7. Retrieve evidence, then delete the owned resources

Before the VM deadline, return to your **laptop** with `exit`. Save results outside the VM. Choose a new local destination for each attempt:

```sh
mkdir -p evidence/tpu-first
gcloud compute scp --recurse "$VM:~/jax-tpu-gcp/runs/tpu-first" \
  evidence/tpu-first/ --project "$PROJECT_ID" --zone "$ZONE"
gcloud compute scp "$VM:~/jax-tpu-gcp/tpu-environment.txt" \
  "$VM:~/jax-tpu-gcp/tpu-launch.log" evidence/tpu-first/ \
  --project "$PROJECT_ID" --zone "$ZONE"
```

Open the copied checkpoint and logs locally before deleting anything. Local VM files are not durable storage. Larger experiments need periodic checkpoints in a durable store, such as an authorized Cloud Storage bucket, plus a tested restore procedure. Copying files only at the end cannot protect against interruption. Continue with [checkpoint recovery](course.html?phase=recovery) and [workload operations](project.html?id=workload-operations) before expanding the run.

Delete **the group and template you created for this exercise**, including when a job fails. Do not delete a shared group or template. Deleting only its VM can cause a managed group to replace it. See [managed group deletion](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instance-groups/managed/delete).

```sh
gcloud compute instance-groups managed delete "$GROUP" \
  --project "$PROJECT_ID" --zone "$ZONE"
gcloud compute instance-templates delete "$TEMPLATE" \
  --project "$PROJECT_ID" --region "$REGION"
gcloud compute instance-groups managed list --project "$PROJECT_ID"
gcloud compute instances list --project "$PROJECT_ID"
gcloud compute disks list --project "$PROJECT_ID"
gcloud compute addresses list --project "$PROJECT_ID"
```

Verify that your group and VM are absent. Inspect any retained disks or addresses associated with the experiment; do not delete unrelated resources. Save the successful command outputs and reconcile actual usage in Billing afterward. Stored artifacts may still incur storage costs. Preserve the evidence you deliberately retained.

## 8. Diagnose, explain, and extend

| Observation | First explanation to investigate | Evidence to inspect |
| --- | --- | --- |
| CLI permission denied | Wrong active identity or missing permission | Active account, project, named denied permission |
| Request does not become ready | Quota, capacity or configuration, not training speed | Group status and error details |
| TPU backend fails before training | Environment, runtime or hardware discovery | `run.json` stderr, package versions, instance configuration |
| Loss is absent after a timeout | Worker may not have completed an update | Events and last saved checkpoint |
| SSH closes after training | Session ended; resource may still exist | Cloud resource lists, not the shell prompt |
| Resumed result diverges | Different state, data, source or environment | Checkpoint provenance, random key and optimizer state |

**Try a changed condition:** run to step four, preserve the checkpoint, then resume to step twelve in the same environment. Compare with a fresh twelve-step run. Explain why resuming to step four is rejected and why a timeout before the first save leaves nothing to restore. Keep the commands, outputs and your explanation.

**Checkpoint:** your log says `completed`, and JAX reports eight devices. Have you proved distributed training and stopped resource charges? No. You proved the worker reached its target step. The current code has no sharding or multi-controller launch, and cloud deletion is a separate step. Demonstrate each claim with its own evidence.

Your dossier is complete when another learner can identify the source and environment, replay the CPU exercise, inspect actual target evidence if collected, explain a failure, restore state, and verify cleanup. Leave TPU execution marked pending until you have its real logs.

Continue through the [TPU and Google Cloud learning path](pathways.html?path=tpu): model training, profiling, distributed arrays, recovery, deployment and operations. Replace the synthetic workload only after its operational checks make sense. The [text harness](project.html?id=text-harness) supplies the next model-building project; moving its CPU-qualified implementation onto TPUs is a separate qualification task.

## 9. Prepare a Codelab or credit request

A good first Codelab asks one bounded question, such as whether a resumed training state matches an uninterrupted run. Give learners a CPU route first, then a separately measured TPU experiment. Include the resource lifetime, expected artifacts, failure diagnosis and cleanup in the lab itself.

If seeking support through TPU Builders or another programme, ask the programme team for its current public application link and eligibility rules. This course does not grant credits or guarantee TPU access. For an application, prepare:

- The learner goal and project source revision.
- The requested hardware, region, number of hosts and maximum duration.
- A measured pilot, including setup and idle time, plus the number of learner attempts.
- The checkpoints, logs and benchmarks you will publish, with data and model licensing reviewed.
- Who monitors usage, who performs cleanup, and when the experiment ends.

Confirm approved amount, expiry, project and eligible services before relying on credits. Small tests can use less compute than model training, but estimate them from a pilot. A future 300M-parameter reference-model project needs its own data recipe, evaluation plan and compute estimate; this eight-update exercise does not establish that budget.
