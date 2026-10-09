# Shipping synthesis reviewer notes

Use these reference notes to verify adaptation gradients, quantization error bounds, and batching latency calculations.

## Adaptation

The base must have a measured training improvement and an actual saved/reloaded artifact. Both adaptation branches must begin with the same base. Stable binary cross-entropy has derivative \(X^\top(\operatorname{sigmoid}(Xw)-t)/N\). A soft teacher target and a hard observed target can use the same algebra but encode different supervision. Compare their final predictions on a common held-out criterion rather than ranking incomparable training losses.

A known synthetic generating teacher is deliberately favorable; a corrupted-teacher experiment exposes that assumption. Teacher imitation is not preference optimization or RL. Check source retention separately from target-task gains.

## Precision and artifacts

For symmetric signed \(b\)-bit quantization, \(q_{\max}=2^{b-1}-1\), \(s_j=\max_i|W_{ij}|/q_{\max}\), and \(\widehat W_{ij}=s_jq_{ij}\). Zero columns need a finite special-case scale. Without clipping, weight reconstruction error is at most \(s_j/2\), which bounds output error through \(|X|\,|\widehat W-W|\).

For symmetric activation scale \(a\), an integer dot product reconstructs as \(a s_j\sum_i q_{x,i}q_{w,ij}\), plus any separately declared bias convention. INT32 accumulation is a numeric contract, not evidence of a specialized fast device kernel. Shifted activation ranges can introduce clipping error beyond a rounding-only bound.

Check actual deserialized files, not a direct call to the original Python function. Artifact hashes establish byte identity; neither hashes nor parity certify model quality.

## Timing

For batch time \(t_B\), the batch execution latency remains \(t_B\) milliseconds. The amortized compute cost is \(t_B/B\) milliseconds per example. Full-batch compute throughput is \(1000B/t_B\) examples per second. None includes collection or queue delay unless the measured boundary explicitly does.

First-call compilation and warmed calls must be separated. Check that completion is inside the timer. Sample percentiles describe collected observations; thirty observations cannot support strong production p99 or worst-case claims. A request function without networking must not be reported as a full HTTP load test.

## Capacity

For arrivals \((0,1,4)\) and service time \(2\), starts are \((0,2,4)\), finishes \((2,4,6)\), and response times \((2,3,2)\), all milliseconds. A burst of four at zero finishes at \(2,4,6,8\). Growing response times can come from queueing even when execution time is constant.

For batch eight and two-millisecond arrival spacing, the earliest request waits fourteen milliseconds and the average waits seven before execution, under the specified full-batch assumption.

The hypothetical replica estimate is \(\lceil300/(0.7\cdot200)\rceil=3\). This is a planning calculation. Require a separate protocol for measured effective throughput, realistic arrivals, startup/warmup, batching, error rates, tail latency and failure behavior before treating it as a production capacity claim.

## Container delivery extension

Check actual Docker evidence rather than the presence of a Dockerfile. Model/image identities must match the evaluated artifact and mounted bytes; readiness must follow a successful load/prediction. Ask for wrong-digest rejection and explain the service's concurrency/request limits. Base digest pinning and non-root execution are useful controls, not a claim of comprehensive security or production qualification.
