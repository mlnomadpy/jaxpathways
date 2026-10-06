# Train and audit a policy from interaction

Build a complete, small RL experiment: functional environment, stochastic trajectories, policy updates and held-out evaluation. A successful artifact trains policies from fresh interaction rather than optimizing a fabricated surrogate alone.

This is a CPU teaching project with an enumerable corridor, not a control benchmark. It implements a tabular REINFORCE agent and a clipped PPO-style policy-gradient agent without a critic. The environment and exact evaluator make mistakes inspectable before introducing larger neural agents.

## Start

Use Python and the pinned `requirements-cpu.txt` from the course workspace or project download. From the extracted top-level folder:

```sh
python3 -m pip install -r requirements-cpu.txt
cp projects/policy-evaluation/starter/model.py projects/policy-evaluation/my_model.py
python3 projects/policy-evaluation/tests/check.py --stage 1 --implementation projects/policy-evaluation/my_model.py
```

PowerShell preparation:

```powershell
Copy-Item projects/policy-evaluation/starter/model.py projects/policy-evaluation/my_model.py
```

Edit `my_model.py`. Checks are cumulative: choose stages 1 through 4. The reference implementation is separate in `solution/model.py`. Run it with `--implementation solution`; reading it is optional, and copying it does not demonstrate your own understanding.

## Contract

- Positions are 0, 1, 2 and goal 3. Reset starts uniformly at 0 or 1.
- Action 0 moves left; action 1 moves right. The left wall clips movement.
- An active non-goal transition pays -0.01; entering the goal pays 1.
- The default horizon is 8. Reaching the goal at the deadline is termination, not truncation.
- Finished states absorb without extra rewards or ending events. Step never auto-resets.
- State includes position, elapsed time and done. Observation is position; the tabular policy is stationary and does not observe time remaining.
- A policy has three finite logits, one per unfinished position. Its sigmoid gives the probability of moving right.
- A rollout contains one episode per environment. Arrays are time-major, shape `(horizon, batch_size)`. The active mask describes the state **before** the action.
- Reward-to-go has no discount or bootstrap: this objective is total reward within the finite action budget. A continuing task with collector timeouts needs a different bootstrap contract.

## Stage 1 — Test the environment before learning

Implement `reset` and `step`. Enumerate moves at both boundaries, a timeout, success at the deadline and repeated calls after finishing. Check the final observation survives; do not replace it with a reset observation. Save a hand-worked transition trace.

```sh
python3 projects/policy-evaluation/tests/check.py --stage 1 --implementation projects/policy-evaluation/my_model.py
```

## Stage 2 — Collect genuine trajectories

Implement stable `log_probability`, `rollout` and `returns_to_go`. Split keys across reset, time and environments. Use vmap within scan; preserve fixed carry shapes. Record observation, action, reward, active, terminated, truncated and old_logp. The checker reconstructs trajectories with independent Python rules and changes batch sizes, horizons and seeds.

```sh
python3 projects/policy-evaluation/tests/check.py --stage 2 --implementation projects/policy-evaluation/my_model.py
```

Keep an active-count plot and explain why a fixed array can contain fewer live episodes at later times. Show that masking with the next-state done flag incorrectly discards the successful action.

## Stage 3 — Derive and check objectives

Implement `policy_loss` and `exact_return`. REINFORCE weights log probabilities by frozen reward-to-go. The clipped objective uses the minimum of unclipped and clipped ratio-weighted targets, with frozen behavior log probabilities. Normalize the action sum by episode count. Work through both positive and negative targets.

```sh
python3 projects/policy-evaluation/tests/check.py --stage 3 --implementation projects/policy-evaluation/my_model.py
```

The checks compare dynamic programming to a separate recursive enumeration, compare exact gradients to central differences, and compare a large sampled REINFORCE gradient to that exact gradient. They also verify that old log probabilities and return targets receive zero gradient.

## Stage 4 — Train, evaluate and report

Implement `train` and `evaluate`. Every update collects fresh episodes with a new key. REINFORCE takes one gradient step per fresh batch; the clipped method takes three steps while preserving its behavior probabilities. Return the complete exact-return history including the initial policy.

```sh
python3 projects/policy-evaluation/tests/check.py --stage 4 --implementation projects/policy-evaluation/my_model.py
```

The public checks run both methods across three training seeds. Your report should use at least five training seeds and four reserved evaluation streams. Save policy parameters, raw episode returns, source revision, package versions, settings and per-agent means. Compare uniform, always-left and always-right policies. Distinguish variability across trained agents from evaluation sampling noise. Shared evaluation streams create paired comparisons; do not treat every episode as an independent training run.

## Reference evidence and limits

`validation.json` contains source hashes, environment and six executed reference training/evaluation results. In the recorded CPU environment the uniform exact return is 0.466875; the always-right ceiling is 0.985. Reference policies improve, but the public fixture is not a hidden exam or professional credential.

No GPU/TPU performance, general control capability, production PPO equivalence or independent learner review is claimed. PPO clipping is not a hard trust region or a guarantee of monotonic return. The project intentionally omits critic fitting, generalized advantage estimation, neural observation encoders and continuous action distributions. Those are extensions requiring new contracts and checks.

## Synthesis assessment

After the project, use the RL synthesis questions and reviewer guide under `assessments/rl.md` and `assessments/rl-reviewer.md` in the complete course workspace. Explain one failure and its repair; passing public assertions alone is insufficient evidence of understanding.
