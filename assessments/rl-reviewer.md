# RL synthesis: reviewer notes

[Return to the assessment](rl.md). Use these numerical references to verify your environment transitions, policy gradients, and evaluation rollouts.

## Boundary arithmetic

From position \(1\), the two right actions pay \(-0.01\) and \(1\), giving \(0.99\). The second transition is terminated and not truncated, even though it is the deadline. Additional calls return zero reward with the same finished state and no new ending events.

From position \(0\), the two moves reach position \(2\), paying \(-0.01\) twice; the second transition truncates. The uniform-start always-right expected return is \((0.99-0.02)/2=0.485\).

A continuing-task timeout may bootstrap to \(-0.01+0.9(2)=1.79\). The project’s finite-horizon objective assigns no reward beyond its action budget, giving \(-0.01\) at a non-goal final transition. The distinction depends on the task definition, not the spelling of a flag.

## Expectation and derivatives

For logits \((-0.4,0.6,1.2)\) and horizon \(4\), host recursive enumeration and the JAX backup should agree at approximately \(0.4503909552\). The derivative vector is approximately \( (0.09401754,0.16403031,0.08473177) \). Float32 computations and finite differences require a stated tolerance.

The host reference should recursively average each valid left/right branch, end at the goal or exhausted horizon, and average the two start positions. It must not call the learner’s transition or dynamic-program evaluator to manufacture independence.

The derivatives are positive: increasing any right-action logit improves expected return at this policy. Sampling errors vary by stream. Accept a sound gradient estimator with error evidence; reject claims that any specific batch-size increase must reduce error for every realization.

## Clipping arithmetic

For target \(2\), ordinary terms are \((1.2,2,2.8)\), clipped terms are \((1.6,2,2.4)\), and minima are \((1.2,2,2.4)\).

For target \(-2\), ordinary terms are \((-1.2,-2,-2.8)\), clipped terms are \((-1.6,-2,-2.4)\), and minima are \((-1.6,-2,-2.8)\).

For a positive target, increasing the ratio beyond the upper clip gains no additional sampled objective. For a negative target, decreasing the ratio below the lower clip gains no additional sampled objective. Harmful changes remain penalized. These are statements about the surrogate terms, not guaranteed behavior of a full optimization step or unseen actions/states.

If both new and old probabilities are recomputed from the same candidate, their difference in log space is zero and the ratio is one. Depending on gradient handling, this either destroys the intended derivative or continually resets the comparison point. It is not a legitimate replacement for a frozen behavior policy.

## Changed protocol review

At horizon \(6\), always-left earns \(-0.06\), while always-right remains at \(0.985\), because both possible starts reach the goal within three moves. The uniform-policy baseline is \(0.367578125\) at that horizon; it must be recomputed rather than copied from the horizon-eight lesson. Evaluate sampled baselines against the independent enumerator.

Do not impose the original lesson’s final-return thresholds on the changed training budget. Inspect correct updates, replay, exact and sampled metrics, and honest failures. Distinguish compute-unmatched training procedures, shared-stream pairing, the number of trained agents and the number of episodes. A report that pools episodes into thousands of supposedly independent training runs needs revision.
