# Choose, measure and profile a TPU workload

You have a model and access to a TPU. Should you change the batch size, use smaller numbers, shard across more chips, or choose another generation? Start by finding what limits the workload. A faster matrix unit cannot fix a stalled input pipeline.

By the end, you will have a **hardware and precision report**: a workload description, memory estimate, explicit precision policy, numerical comparisons, synchronized timings and a short profile. Begin with [TPU and Google Cloud setup](tpu-gcp.html), [honest benchmarking](lesson.html?lesson=performance-01) and [reading an XProf trace](lesson.html?lesson=performance-03). The [practice workspace](downloads/jax-tpu-gcp.zip) contains the executable lab below.

The recorded reference outputs below use JAX 0.9.2 and NumPy 2.4.4 on CPU, with hardware specification tables checked against Google Cloud TPU documentation. Run the same `precision_profile.py` script on your Cloud TPU VM to compare target-device MXU timings against your CPU baseline.

## 1. Choose for the workload, then check the generation

Training runs forward computation, backward computation and optimizer updates. Its persistent state includes more than weights. Inference usually retains weights and, for a causal model, a growing key/value cache. **Prefill** processes the prompt together; **decode** generates another token for each active sequence. Prefill can expose large matrix products, while small-batch decode often spends much of its time moving weights and cache data. Test which regime dominates your workload by profiling both phases.

| Work you are doing | First question | Measure before choosing |
| --- | --- | --- |
| Pretraining or fine-tuning | Does complete training state fit at the required sequence length? | Tokens per second, step time, peak memory, loss and recovery |
| Embedding or image batch inference | Can requests form useful batches without missing deadlines? | Examples per second, batch latency and task quality |
| Causal prefill | What prompt lengths and concurrent requests must be supported? | Time to first token, prefill throughput and memory |
| Causal decode | How much weight/cache traffic does each new token require? | Inter-token latency, output tokens per second and cache growth |
| Audio streaming | Can chunks complete before the next deadline? | Chunk latency, real-time factor and task quality |

There is no fixed rule that one suffix means training only and another means inference only: v5e, v6e, and TPU7x support both training and serving configurations, while v5p's large HBM capacity and 3D torus interconnect target large-scale training slices.

The table below lists **per physical chip** specifications from Google Cloud's documentation (preserving documented GB vs. GiB units):

| Generation | Peak BF16 TFLOPs/chip | HBM/chip, as documented | Topology at scale | What to investigate |
| --- | ---: | --- | --- | --- |
| [v2](https://docs.cloud.google.com/tpu/docs/v2) | Not listed on the linked overview | Not listed on the linked overview | Legacy configurations | Existing allocations and runtime compatibility; compare with newer options |
| [v3](https://docs.cloud.google.com/tpu/docs/v3) | 123 | 32 GiB | 2D torus | Legacy experiments; host and device numbering |
| [v4](https://docs.cloud.google.com/tpu/docs/v4) | 275 | 32 GiB | 3D mesh, with supported torus configurations | Existing v4 access, BF16 training, memory and communication limits |
| [v5e](https://docs.cloud.google.com/tpu/docs/v5e) | 197 | 16 GB | 2D torus | Training and serving; relatively small per-chip memory |
| [v5p](https://docs.cloud.google.com/tpu/docs/v5p) | 459 | 95 GiB | 3D torus at supported slice sizes | Large training states and scaling across a slice |
| [v6e / Trillium](https://docs.cloud.google.com/tpu/docs/v6e) | 918 | 32 GB | 2D torus | Training and serving, larger matrix units, supported INT8 operations |
| [TPU7x / Ironwood](https://docs.cloud.google.com/tpu/docs/tpu7x) | 2,307 | 192 GiB | 3D torus | Large training and serving workloads, FP8 recipes and chiplet placement |

Use the linked pages when selecting a machine, region, software image and resource API. Availability and price are separate from these specifications. Compare actual cost per useful training token or per request meeting your latency target when you have approved access and current pricing.

**Check the counting unit.** A physical chip, TensorCore, JAX device and host VM are different objects. For example, a v5p-8 allocation contains four chips, with eight TensorCores. TPU7x exposes two JAX devices per chip, corresponding to chiplets. Never infer a mesh from the suffix alone. Record `jax.devices()`, `jax.local_devices()`, `jax.process_count()` and the resource topology before sharding. The single-device lab below uses only the first local device, even when more are visible.

TPU7x currently supports JAX and PyTorch, with TensorFlow unsupported on that generation. A Keras/TensorFlow model therefore needs an explicitly supported execution route or a validated conversion; a model file loading successfully is not proof that its operators run on the selected TPU. For architecture context, see [TPU memory, matrix units and interconnect](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm).

**Checkpoint:** a small decode workload is slow on a device with much higher peak compute. Is the hardware broken? A good answer first checks batching, memory movement, launch overhead and the serving runtime. Peak matrix throughput does not measure those limits.

## 2. Make a memory budget before requesting a larger slice

Let \(P\) be the number of parameters and \(b_w\) the bytes stored per weight. Weight storage alone is:

\[
M_{\mathrm{weights}} = P b_w.
\]

For \(P=300{,}000{,}000\), FP32 weights occupy 1.2 decimal GB, BF16 weights 0.6 GB and INT8 weights 0.3 GB **before** scales, metadata and padding. Packed INT4 would occupy 0.15 GB before overhead. Restricting an INT8 array to values between minus seven and seven still stores a byte per element; the four-bit saving requires packing and a kernel that consumes that representation.

Training needs another budget. Consider a policy with a BF16 working copy, FP32 master weights, two FP32 Adam moment arrays and FP32 gradients:

\[
M_{\mathrm{state}} = P(2+4+4+4+4) = 18P\ \mathrm{bytes}.
\]

That is **5.4 decimal GB** for 300M parameters before activations, temporary buffers, inputs and communication. Some implementations avoid a separate working copy, shard moments or use another optimizer. Inspect the actual state tree; do not apply 18 bytes blindly. Activation memory changes with batch size, sequence length, attention implementation and rematerialization. Compiler memory estimates and a measured runtime peak answer different questions.

For a causal model, let \(B\) be active sequences, \(T\) cached tokens per sequence, \(L\) layers, \(H_{kv}\) key/value heads, \(D_h\) head dimension and \(b_{kv}\) bytes per cache element. A dense, unpadded key plus value cache needs:

\[
M_{\mathrm{KV}} = 2 B T L H_{kv} D_h b_{kv}.
\]

With \(B=8\), \(T=4096\), \(L=24\), \(H_{kv}=8\), \(D_h=64\) and BF16 cache entries, this is **1.5 GiB**. Doubling context length doubles this component. Use the key/value head count for grouped-query attention, not the query head count. Paged caches, padding, scales and replication add overhead; sharding changes where these bytes live.

**Work it out:** if the same cache is stored as INT8, its raw entries take 0.75 GiB. Is that a safe serving configuration? Only after accounting for scales and layout, verifying the cache update/read implementation, and measuring held-out generation quality and latency.

## 3. Specify precision separately for each part

“Use low precision” leaves important decisions unresolved. Write down **weight storage, activation storage, dot-product accumulation, reductions, gradients, optimizer state and inference cache**. A model can use several policies at once.

| Representation | What it offers | What you must handle |
| --- | --- | --- |
| FP32 | Useful reference and more fractional precision | More storage; input dtype alone does not establish the TPU's multiply algorithm |
| BF16 | Two-byte storage and exponent range similar to FP32 | Coarser rounding; test sensitive reductions and small updates |
| FP16 | Two-byte storage and finer fraction than BF16 within its range | Narrower exponent range; check overflow/underflow and loss-scaling needs; verify target lowering |
| FP8 | One-byte floating storage with format-specific range | Choose format, scaling granularity and update policy; monitor clipping and quality |
| INT8 | One-byte integer storage, with scale/zero-point policy | Calibration or dynamic scales, integer accumulation, supported kernels and dequantization |
| INT4 | Potentially half a byte when packed | Packing/layout, group scales, unpack overhead and target-specific kernel support |

BF16 retains fewer fraction bits than FP32. On TPU, BF16 matrix products can accumulate into FP32, but accumulation cannot restore detail already rounded out of the operands. TPU BF16 conversion also has specific handling of very small values. See [Google's BF16 guide](https://docs.cloud.google.com/tpu/docs/bfloat16). Keep loss statistics, normalization and optimizer updates in FP32 where the validated recipe needs it. BF16's wider range makes it different from FP16; it is not a promise of unchanged convergence.

Hardware support must be tied to an operation. v5e and v6e document INT8 peak rates of 393 and 1,836 TOPs per chip respectively. TPU7x documents native FP8 acceleration, with peak FP8 throughput of 4,614 TFLOPs per chip. v4 also documents INT8 matrix support and an eight-bit weight-loading mode intended to help small-batch inference. v5p lists FP8 and BF16 at the same peak rate of 459 TFLOPs per chip. A smaller format therefore does not always double arithmetic throughput. These peaks do not imply the same end-to-end speedup; inspect the compiled operation and profile.

A JAX dtype can exist even when the backend casts it, emulates an operation or uses a slower path. FP16 is not automatically the best two-byte choice for TPU. INT4 storage does not demonstrate accelerated INT4 arithmetic. For FP8 on TPU7x, the [official performance guide](https://docs.cloud.google.com/tpu/docs/ironwood-performance) describes scaling choices and MaxText's QWIX route. Formats such as E4M3 and E5M2 trade fraction bits for exponent range. Pin the framework and quantization-library revision, and test the chosen forward/backward recipe.

For integer quantization, distinguish **weight-only**, **weight plus activation**, and **cache** policies. Weight-only quantization can reduce storage while a kernel still performs floating-point multiplication. A true integer dot needs quantized operands, a supported integer operation and an appropriate accumulator. Google's [AQT repository](https://github.com/google/aqt) explains this distinction for JAX. INT8 dots in our lab request INT32 accumulation and are checked against independent NumPy integer arithmetic.

Post-training quantization (PTQ) starts from trained weights. Use representative, disjoint calibration inputs when estimating fixed activation ranges. Quantization-aware training (QAT) exposes a training recipe to simulated or actual quantization effects; integer casts alone do not supply useful gradients. Neither method removes the need for held-out evaluation. For embeddings, inspect retrieval quality and ranking changes; for causal models, loss and generation cases; for image/audio models, the relevant task metric and difficult subsets. Track clipping, scale drift, nonfinite values and distribution shifts along with throughput.

**Diagnose this failure:** storage shrinks but latency worsens. Check whether the graph converts every weight back to FP32, whether activations are quantized on every request, whether packed data is unpacked into large temporaries, and whether the batch is too small to amortize those costs. Inspect the executable and profile before attributing the result to the hardware generation.

## 4. Run the precision lab and read its actual results

Open a terminal in the extracted `jax-tpu-gcp` workspace. Reuse the CPU environment from the setup guide:

```sh
# Run command in terminal
.venv/bin/python resources/tpu-gcp/precision_profile.py \
  --platform cpu --run-dir runs/precision-cpu --trace
```

The script compares FP32 operands, BF16 operands with FP32 output/accumulation requested, and symmetrically quantized INT8 operands with INT32 accumulation followed by FP32 rescaling. Floating dots request JAX's `HIGHEST` precision; target lowering still needs inspection. All results are compared with a NumPy FP64 dot of the **original FP32 input values**. The reference reduces arithmetic error but cannot recover information absent from those original values.

The example computes \(Y=XW\), where \(X\) is \(64\times128\) and \(W\) is \(128\times64\). Inputs are synthetic and reproducible. A second case multiplies feature zero by 30 to introduce an outlier. The program saves six results, source/input hashes, device identity, actual compiler memory estimates, lowered executables and a warmed trace. It refuses to reuse an existing run folder and writes failure diagnostics if the requested backend cannot execute.

![Two plots from the recorded CPU precision lab. Top: relative L2 output error for FP32, BF16 and INT8, with balanced and outlier inputs. Bottom: stored input and weight operand bytes, including INT8 scales.](downloads/tpu-performance/precision.png)

[Open the full-size figure](downloads/tpu-performance/precision.png).

**Read the top plot:** the horizontal axis selects the operand policy. The vertical axis is relative L2 error on a logarithmic scale, so each tick multiplies the error rather than adding a fixed amount. Blue points show the balanced inputs; orange points show the outlier case. FP32 has small but nonzero arithmetic error. BF16 has about 0.23% relative error in these cases. INT8 rises from about 0.91% to 2.45% after the perturbation. The INT8 row scale must now cover a wider range, leaving larger gaps between representable values for ordinary features. These percentages describe this synthetic dot, not model accuracy.

**Read the bottom plot:** the bars count stored operands supplied to the executable. FP32 uses 65,536 bytes; BF16 uses 32,768; INT8 uses 16,896, including 512 bytes of row/column scales. Outputs remain FP32. The bars exclude live original arrays, compilation, temporary buffers, optimizer state and cache, so they are not runtime peak memory. Smaller bars do not predict faster execution.

Inspect the [recorded JSON results](downloads/tpu-performance/report.json), [reproducible plotting script](downloads/tpu-performance/render_precision.py) and [lab source](downloads/tpu-performance/precision_profile.py). The figure is rendered from the JSON, with a source hash checked at build time. The reference includes CPU timings, but the plotted conclusions concern error and stored bytes. CPU timing is not a TPU benchmark.

For row \(i\) and output column \(j\), the symmetric quantizer uses separate scales:

\[
s_i = \frac{\max_k |X_{ik}|}{127},\qquad
 t_j = \frac{\max_k |W_{kj}|}{127}.
\]

All-zero rows/columns use a scale of one to avoid division by zero. Values are rounded to the nearest integer and clipped to \([-127,127]\). The output approximation is:

\[
\widehat{Y}_{ij} = s_i t_j \sum_k q(X_{ik})q(W_{kj}).
\]

For a tiny row \([0,1,-2]\), the scale is \(2/127\), so one maps to 64 after nearest-even rounding and reconstructs as approximately 1.0079. A larger accumulator cannot undo that input rounding. INT32 accumulation also has a finite range: the lab bounds its contraction length, whereas a general kernel must check its own overflow conditions.

Our plotted error is:

\[
e_{\mathrm{rel}} = \frac{\|\widehat{Y}-Y_{\mathrm{ref}}\|_2}{\|Y_{\mathrm{ref}}\|_2}.
\]

The script records maximum absolute error as well. Relative error is undefined for a zero reference norm, so the report uses `null` rather than inventing a denominator. Elementwise relative error near zero can be misleading; inspect absolute errors and task metrics too.

**Transfer exercise:** predict whether batch one changes the INT8 scale overhead or hardware utilization, then run:

```sh
# Run command in terminal
.venv/bin/python resources/tpu-gcp/precision_profile.py \
  --platform cpu --run-dir runs/precision-batch-one \
  --batch 1 --features 127 --outputs 65 --steps 20 --trace
.venv/bin/python resources/tpu-gcp/check_precision.py
```

The reference check covers zero inputs, changed shapes, exact integer accumulation and arithmetic on already-rounded BF16 operands. Batch one has fewer activation rows but still one scale for every output column. Non-tile-aligned dimensions can introduce padding or less efficient target kernels; a different CPU duration cannot establish that this occurred on TPU.

## 5. Capture a short TPU profile, then explain the timeline

Install the supported JAX/TPU runtime on an approved VM as described in the [cloud workflow](tpu-gcp.html). Record its resolved versions. Run the same lab with `--platform tpu` in a fresh folder. If the TPU backend is unavailable, it must fail, not quietly substitute CPU:

```sh
# Run run command in terminal using the course Python environment
python resources/tpu-gcp/precision_profile.py \
  --platform tpu --run-dir runs/precision-tpu --steps 20 --trace
```

Compilation, transfer and input quantization happen outside the timing loop. Each sample waits for the returned array with `block_until_ready()`. The trace is a separate warmed span, so profiler overhead does not contaminate those timing samples. For a real server, also measure the whole request path including quantization, queueing and transfers. For training, include the data loader and update step in a separate end-to-end measurement.

The six trace steps are labelled `precision_case`, with inner names such as `balanced-int8`. The tiny default dot may be dominated by dispatch overhead. Use it to learn annotation and error checks. Larger approved workloads and sufficiently long steady-state windows are needed to explain accelerator utilization.

After copying the trace folder to your laptop, install XProf in a separate viewer environment. It need not contain the training runtime:

```sh
# Run run command in terminal using the course Python environment
python3 -m venv .profile-viewer
.profile-viewer/bin/python -m pip install xprof
.profile-viewer/bin/xprof --port 8791 /absolute/path/to/runs/precision-tpu/trace
```

Open `http://127.0.0.1:8791`. Record the installed viewer version. XProf reads the trace tree, including files under `plugins/profile/`; retain that tree when copying. The saved Perfetto trace is another timeline view. Follow [JAX's current profiling instructions](https://docs.jax.dev/en/latest/profiling.html) if viewer installation differs in your environment.

Read a trace from outside inward. Locate the warmed step annotation, then identify host and device lanes. A host annotation includes dispatch and waiting; it is not an accelerator kernel duration. Overlapping work on different lanes is concurrent. Nested durations are not independent costs that you can add together. A collective's waiting time may be caused by another host arriving late.

| What you observe | Hypothesis to test | Next controlled comparison |
| --- | --- | --- |
| Long gap before device work | Input preparation or dispatch starvation | Use already-prepared batches; measure loader time separately |
| Compilation events during measured steps | Shape, static argument or function changes | Stabilize one factor and check compilation logs |
| Large matrix activity with high compute demand | Compute may limit this region | Try a supported precision recipe while checking quality |
| Heavy data movement, small batch | Memory traffic or launch overhead may dominate | Change batch size within the latency/memory budget |
| Long collectives or mismatched host timelines | Communication or host skew | Inspect all hosts, topology and data arrival; change one mesh axis |
| More recomputation after rematerialization | Memory saving trades additional work | Compare runtime peak and step time at the same shape |

These rows generate experiments, not automatic diagnoses. Profile summaries, memory views and roofline information depend on backend and captured data. Missing device lanes are not evidence that device work took zero time. Check the trace window, runtime/profiler compatibility and actual backend. For multi-host training, coordinate capture across processes and save host identities; this lab is single-process and does not establish distributed scaling. Google's [TPU profiling guide](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) connects these tools to actual TPU workloads.

## 6. Use a roofline estimate to decide what to try

For a dense product with \(X\) shaped \(m\times k\) and \(W\) shaped \(k\times n\), count roughly \(F=2mkn\) floating-point operations. If every operand is read once and the output written once, an ideal byte estimate is:

\[
A_{\min} = mkb_x + knb_w + mnb_y,\qquad I = F/A_{\min}.
\]

Here \(b_x,b_w,b_y\) are bytes per activation, weight and output. This optimistic estimate assumes reuse within the operation and excludes scales, padding, temporary buffers and communication. Actual memory traffic can be larger, or affected by data already resident in a faster memory tier.

With peak compute \(C\) in FLOPs/second and memory bandwidth \(B_w\) in bytes/second, a simplified ideal lower bound is:

\[
t_{\mathrm{ideal}} \geq \max(F/C, A_{\min}/B_w).
\]

The first term represents arithmetic; the second represents moving bytes. It excludes launch and synchronization costs and is not an expected measured latency. Keep the units consistent when using GB/s versus GiB/s. Integer TOPs should not silently be labelled floating-point FLOPs.

**Worked example:** for \(m=1,k=n=1024\), BF16 input/weights and FP32 output give \(F=2{,}097{,}152\) and \(A_{\min}=2{,}103{,}296\) bytes. Arithmetic intensity is approximately one FLOP per byte. Increasing to \(m=64\) raises reuse of the weight matrix: \(F=134{,}217{,}728\), \(A_{\min}=2{,}490{,}368\) and intensity is about 53.9 FLOPs per byte. This explains why batching is worth testing, while queueing latency may stop you from choosing a large batch.

On TPU, HBM capacity, HBM bandwidth, on-chip memory and interconnect each constrain a different part. ICI connects chips within a slice; DCN connects slices. More devices can introduce communication and replicated state. A bigger slice should follow an actual memory or throughput requirement and an inspected sharding layout. Connect this estimate to the [HLO and roofline lesson](lesson.html?lesson=performance-04) and [distributed phase](course.html?phase=distributed).

## 7. Submit a decision report, not a precision slogan

Use these three activities in order: hardware/memory choice, precision experiment, then profiler diagnosis. Keep the original baseline and change one factor per comparison.

Your report should contain:

1. Workload: modality, objective, batch/sequence distributions, training or serving, quality metric and latency target.
2. Hardware: generation, physical chips, host count, JAX devices, topology, actual backend and resolved package versions.
3. Precision policy: weights, activations, accumulators, reductions, gradients, optimizer state and cache; scale granularity and calibration split where applicable.
4. Evidence: configuration/source/input hashes, numerical error, held-out task quality, synchronized timing samples, actual memory observations and the trace folder.
5. Interpretation: what the figure/timeline shows, one tested bottleneck hypothesis, what changed, and whether quality still met a declared target.
6. Scope: single-device versus distributed, microbenchmark versus full model/request, CPU versus actual TPU, and remaining qualification work.

**Reference diagnosis:** INT8 lowers stored bytes in the synthetic dot but introduces more error in the outlier case. That supports investigating finer scaling or a floating policy for sensitive activations. It does not establish that a quantized embedding model retains retrieval quality, or that INT8 is faster on your TPU. Those need the real model, held-out data and target profile.

For training, continue with the [state and recovery phase](course.html?phase=recovery), precision monitoring and the [pretraining methods](course.html?phase=pretraining). For serving, connect the report to [deployment precision checks](course.html?phase=deployment) and the modality harness's inference endpoints. Use MaxText as a reference for a complete training system after these small checks, recording the exact recipe revision. A successful microbenchmark is the start of a model qualification report, not its conclusion.
