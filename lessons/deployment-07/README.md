# Containerize a model service and verify its boundary

Phase 15: Deployment, interoperability & edge AI · about 75 minutes · CPU

## What you will be able to do

- Separate training from inference
- Understand the build context
- Keep runtime inputs explicit
- Prove the same request survives the boundary

## The problem

Your model works in a notebook, but another machine has a different interpreter, library set and working directory. Let’s export a small prediction artifact, verify it in a fresh process, then run the same boundary in a real Docker container.

## The idea

A container image packages the service runtime and code. The model artifact, runtime configuration and release identity are separate inputs. A reproducible service verifies all of them instead of assuming that an image tag names the same bytes forever.

## Separate training from inference

Our service evaluates an exported scalar affine model using the Python standard library. The connected lab actually trains that model in JAX first. Inference does not need the optimizer, training data or MLflow server. This small portable contract makes it possible to compare Python-process, MLflow pyfunc and container outputs independently. Larger JAX exports need a compatible runtime and their own serialization checks; copying two coefficients does not demonstrate arbitrary graph portability.

## Understand the build context

Docker only sees the build context sent to the engine. The supplied .dockerignore excludes everything except Dockerfile and service.py, so local data, notebooks, environment files and training artifacts do not enter the image accidentally. COPY names only the service source. The base image is pinned by its observed digest, not just a moving tag. Pinning records an identity; you still need a deliberate dependency update process and image scanning.

## Keep runtime inputs explicit

MODEL_PATH selects a mounted artifact and MODEL_SHA256 verifies its bytes before serving. Configuration can be supplied at runtime; credentials belong in a runtime secret mechanism rather than build arguments or image layers. The container runs as a numeric non-root user, with a read-only filesystem and model mount. The default network-free check needs no published port. The optional HTTP mode binds a service port, enforces a small request-body and batch limit, and exposes readiness. It is a teaching server, not a production concurrency stack.

## Prove the same request survives the boundary

The CPU companion writes an artifact and starts a fresh Python process for every request. Singleton and changed-size batches must match independent arithmetic. An empty batch, boolean input and changed artifact digest are rejected. These are process checks; the connected Docker checker additionally builds the actual image, mounts the artifact, checks non-root execution, performs inference and tests readiness inside the running container. Keep those two evidence scopes distinct.

**Train and retain the model first**

```bash
python projects/engineering-release/mlflow_lab.py --output ./engineering-run
```

**Expected:** Creates model.json and report.json using actual tracked training.

**Build the local teaching image**

```bash
docker build -t jaxpathways-engineering:course projects/engineering-release
```

**Expected:** Build succeeds from the pinned base digest and copies only the inference service.

**Run the container qualification lab**

```bash
python projects/engineering-release/container_check.py --model ./engineering-run/model.json --output ./container-report.json
```

**Expected:** Checks actual container predictions, digest rejection, non-root identity and HTTP readiness. Requires a running local Docker engine.

## Readiness, liveness and startup are different

A process can be alive while its model failed to load. Readiness should require a verified model and a completed smoke prediction; liveness asks whether the service is stuck. Docker HEALTHCHECK reports health but does not itself implement a rollout controller or restart policy. A Kubernetes startup probe can protect a slow load, readiness can remove an instance from service traffic, and liveness may restart it. Benchmark cold startup separately from warmed inference, then add concurrency and overload limits.

## Ship the artifact you actually checked

CI should run tests, build once, record the resulting image identity, verify the model and request contract, and attach evaluation evidence. Promote that immutable bundle into staging before selecting production. Rebuilding from a tag at deployment time can change the environment after testing. On accelerators, also verify drivers, runtime libraries, device discovery and kernel support. This CPU container receipt is not GPU or edge qualification.

## Run the example

```python
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
MODEL = {'schema': 1, 'weight': 2., 'bias': 1.}
# The deployment contract uses exported coefficients, not a training environment.
SERVICE = '''import hashlib,json,math,os,sys
raw=open(os.environ['MODEL_PATH'],'rb').read()
if hashlib.sha256(raw).hexdigest()!=os.environ['MODEL_SHA256']: raise ValueError('artifact mismatch')
m=json.loads(raw)
x=json.load(sys.stdin)['inputs']
if not isinstance(x,list) or not 1<=len(x)<=32: raise ValueError('batch limit')
if any(type(v) not in (int,float) or not math.isfinite(v) for v in x): raise ValueError('invalid input')
print(json.dumps({'predictions':[m['weight']*v+m['bias'] for v in x]},allow_nan=False))
'''
accepted, rejected = 0, 0
with tempfile.TemporaryDirectory(prefix='container-contract-') as folder:
    root = Path(folder); artifact = root / 'model.json'; service = root / 'service.py'
    artifact.write_text(json.dumps(MODEL, sort_keys=True)); service.write_text(SERVICE)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    env = dict(os.environ, MODEL_PATH=str(artifact), MODEL_SHA256=sha)
    for values in [[-.5], [-2., 0., 1.5]]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps({'inputs': values}), text=True, capture_output=True, env=env, check=True)
        actual = json.loads(completed.stdout)['predictions']
        assert actual == [2 * value + 1 for value in values]
        accepted += 1
    for payload, changes in [({'inputs': []}, {}), ({'inputs': [True]}, {}), ({'inputs': [0.]}, {'MODEL_SHA256': '0' * 64})]:
        completed = subprocess.run([sys.executable, str(service)], input=json.dumps(payload), text=True, capture_output=True, env=dict(env, **changes))
        assert completed.returncode != 0; rejected += 1
print('Fresh-process valid requests:', accepted, 'rejected boundaries:', rejected)
print('This companion tests the process/artifact contract. Run the Docker lab for container evidence.')

```

Expected: Fresh-process valid requests: 2 rejected boundaries: 3
This companion tests the process/artifact contract. Run the Docker lab for container evidence.

## Observe the process boundary before containerizing it

**Predict:** Should invalid requests produce a prediction or a failed process?

![Observe the process boundary before containerizing it](../../phases/15-deployment/07-containerize-a-model-service/outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal categories separate the two valid request shapes from three deliberately invalid boundaries. Bar height is the number of cases observed by the local subprocess checker: $2$ successful prediction cases and $3$ expected rejections. These counts do not represent throughput, latency or a failure probability. Each success is also checked against independent affine arithmetic.

### Connect it to the computation

A container must preserve this contract. The Docker lab supplies additional engine, user and readiness evidence; this plot comes from ordinary Python processes and cannot prove image behavior by itself.

```python
visual_data={'kind':'bar','labels':['valid predictions','expected rejections'],'xlabel':'local process contract','ylabel':'observed case count','series':[{'label':'executed CPU cases','y':[accepted,rejected]}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:26:22.985853+00:00. JAX 0.9.2.

```text
Fresh-process valid requests: 2 rejected boundaries: 3
This companion tests the process/artifact contract. Run the Docker lab for container evidence.
Independent three-input predictions: 0.5, 2.5, 5.0
Changed artifact requires a new recorded digest.
Nonfinite input rejected before output.
PASS: deployment-07

```

## Change the batch without changing the model

**Predict before running:** Will a three-element request give the same values as three singleton requests?

```python
batch=[-.25,.75,2.]
assert [MODEL['weight']*x+MODEL['bias'] for x in batch]==[.5,2.5,5.]
print('Independent three-input predictions: 0.5, 2.5, 5.0')
```

**Expected:** Independent predictions are 0.5, 2.5 and 5.0.

Batching changes transport shape, not the per-observation prediction contract. A stateful or preprocessing-dependent service needs additional checks.

## Make it yours

Reject an artifact whose bytes changed after its expected digest was recorded.

<details><summary>Reference solution</summary>

```python
original=json.dumps(MODEL,sort_keys=True).encode()
mutated=json.dumps(dict(MODEL,bias=2.),sort_keys=True).encode()
assert hashlib.sha256(original).hexdigest()!=hashlib.sha256(mutated).hexdigest()
print('Changed artifact requires a new recorded digest.')
```

</details>

## Reject a nonfinite request

**transfer**

Send an infinite input to the process contract and verify the rejection.

<details><summary>Hint</summary>

The validation boundary should reject before producing JSON predictions.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'model').write_text(json.dumps(MODEL));(p/'service.py').write_text(SERVICE)
    env=dict(os.environ,MODEL_PATH=str(p/'model'),MODEL_SHA256=hashlib.sha256((p/'model').read_bytes()).hexdigest())
    bad=subprocess.run([sys.executable,str(p/'service.py')],input=json.dumps({'inputs':[float('inf')]}),text=True,capture_output=True,env=env)
    assert bad.returncode != 0
print('Nonfinite input rejected before output.')
```

A request can parse as JSON in Python while still violating the numerical service contract.

</details>

## Check your understanding

The image built successfully. What should happen before serving traffic?

1. Assume the trained model is included and ready.
2. Load the expected artifact, verify its digest and complete a prediction through the target runtime.
3. Ignore preprocessing because dependencies are pinned.

<details><summary>Answer and explanation</summary>

Load the expected artifact, verify its digest and complete a prediction through the target runtime.

A container image packages the service runtime and code. The model artifact, runtime configuration and release identity are separate inputs. A reproducible service verifies all of them instead of assuming that an image tag names the same bytes forever.

</details>

## Diagnose the result

A missing model mount is a runtime configuration error, not a reason to rebuild the model into every image. Permission errors call for mount/user inspection. A successful local process check does not prove Docker is running. Keep engine and target failures in their own receipt.

## Carry forward

- Separate training from inference
- Prove the same request survives the boundary
- Ship the artifact you actually checked

## Keep your evidence

Keep the artifact digest, valid and rejected requests, fresh-process output and, when Docker is available, the actual image identity, runtime user and container qualification receipt.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [Docker build best practices](https://docs.docker.com/build/building/best-practices/)
- [Dockerfile reference and health checks](https://docs.docker.com/reference/dockerfile/)
- [Kubernetes liveness, readiness and startup probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)

