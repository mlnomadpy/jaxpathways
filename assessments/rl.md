# RL synthesis: defend an agent trained from interaction

**Scope:** tabular policies, functional finite-horizon environments, randomized rollouts, REINFORCE and clipped policy updates, independent expectation checks, and held-out evaluation.

[Open the project](../project.html?id=policy-evaluation). Complete `rl-01` through `rl-04`, then retain your own implementation, `assessment.py`, and a short report. Keep reference code separate from learner evidence.

## Establish the baseline

Run the cumulative project checks from the workspace root:

```sh
# Run run command in terminal using the course Python environment
python3 projects/policy-evaluation/tests/check.py --implementation projects/policy-evaluation/my_model.py --stage 4
```

Record the source revision, Python/JAX/NumPy versions, backend, dtype, configuration, training seeds and evaluation seed sets. Predict each changed-condition result before execution. The following tasks require new evidence beyond passing the public fixture.

## Task 1: defend an episode boundary

Use a horizon of \(2\), start at position \(1\), and take two right actions. Record the observation before and after each transition, reward, elapsed count, termination, truncation and done state. Call step twice more and show that reward does not repeat.

Then start at position \(0\) with the same action sequence. Explain why the outcome differs. Compute the expected always-right return under the uniform start distribution by hand. Demonstrate that goal-at-deadline priority and absorbing padding match the contract.

Finally, compare the finite-horizon timeout target with a continuing-task target when reward is \(-0.01\), discount is \(0.9\), and next-state value is \(2\). State which task each target represents. A numerical match without the task distinction is incomplete.

## Task 2: transfer the collector

Collect \(19\) environments for horizon \(4\), using logits \((-0.4,0.6,1.2)\) and a fresh declared seed. Draw the axes of every stored array. Reconstruct each trajectory with ordinary Python transition rules and compare observations, rewards and ending events.

Implement a deliberately broken collector that stores its active mask from the state **after** step. Show a concrete successful episode whose reward is lost. Restore the correct mask and demonstrate exactly one ending event per episode, zero reward in padding, and preservation of the final observation.

Keep both traces and explain the repair. Merely changing a boolean until a test passes does not demonstrate the contract.

## Task 3: verify two routes to an expectation

For the same logits and horizon \(4\), calculate exact expected return by recursively enumerating both actions from each possible starting position. Use host floating-point arithmetic and a separate implementation from the JAX dynamic program.

Compare the two expected returns. Estimate all three parameter derivatives with central differences using perturbations \(10^{-3}\) and \(10^{-4}\). Compare with autodiff of the dynamic program, state tolerances and explain why an arbitrarily tiny perturbation may amplify cancellation.

Then estimate the REINFORCE gradient from independently sampled rollouts. Predeclare batch sizes \(256\), \(2048\) and \(16384\), with at least \(4\) independent streams at each size. Report error norms relative to the exact derivative. Do not require every individual larger batch to have lower error: explain the sampling pattern across streams and preserve exceptions.

## Task 4: explain the clipped objective, including its limits

Use probability ratios \((0.6,1.0,1.4)\), clipping parameter \(0.2\), and targets \(2\) and \(-2\). Compute unclipped terms, clipped terms and the pointwise minimum for each target. Explain which direction becomes flat and which remains penalized.

Collect a batch at zero logits, keep its behavior log probabilities, and evaluate a changed candidate policy. Verify that the denominator and return targets receive zero gradient. Show that replacing old log probabilities with the changing candidate makes ratios remain at one and eliminates the intended comparison.

Explain why clipping individual sampled terms is not a hard bound on all policy changes, a bound on KL divergence, or a guarantee that expected return improves. Distinguish this project’s clipped policy-gradient method from a complete neural PPO implementation with critic and advantage estimation.

## Task 5: evaluate a changed training protocol

Before training, write a protocol using \(5\) training seeds absent from the lesson, \(30\) update batches, horizon \(6\), and \(4\) reserved evaluation seeds. Compare REINFORCE with one gradient step per batch and the clipped method with three. Keep the same episode batch size for both, and disclose that the compute budgets differ.

For every fitted agent, retain parameters, complete expected-return history, raw held-out episode returns and per-agent mean. Include uniform, always-left and always-right baselines under the changed horizon. Prove evaluation leaves policy state unchanged. Replay one entire training run and compare the history and final parameters.

Report individual results even when an agent disappoints. Summarize variation across trained agents separately from evaluation noise. State whether evaluation streams are shared, and preserve paired comparisons if they are. Do not select the best seed or change hyperparameters after reading the held-out results while continuing to call them untouched test data.

No fixed score threshold is imposed on this changed protocol. The assessment asks for a faithful, reproducible experiment and a diagnosis supported by evidence, not a conveniently chosen successful run.

## Evidence and review

Use **accept**, **revise**, or **not demonstrated** for each task:

- **Boundary contract:** an independent trace shows deadline priority, final observation, no repeated reward and the distinction between task endings and collector limits.
- **Collector:** all changed-size trajectories match a separate reference; the broken-mask failure and repair are visible.
- **Gradient:** enumeration, finite differences and sampled estimators are compared with justified precision and sampling tolerances.
- **Clipping:** both target signs are correct; behavior data stay frozen; the explanation avoids claiming a guaranteed trust region.
- **Evaluation:** multiple agents, reserved streams, complete histories, baselines, replay, isolation and honest aggregation are all inspectable.
- **Limits:** the report identifies the small synthetic environment, stationary policy class, CPU-only verification, public tests and absence of independent expert review.

Every task needs inspectable learner evidence for overall acceptance. A reviewer may request a different horizon, policy or seed set. The public [reviewer notes](rl-reviewer.md) help check arithmetic after the attempt; they are not a hidden exam key.
