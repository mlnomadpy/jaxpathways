# Phase 10: Scientific computing

Specializations.

Build and differentiate a physical simulator, recover a positive rate from observations, and separate numerical accuracy, identifiability and measured execution cost.

Complete gradients, explicit state and scan first. Start with analytic references, check discretization and sensitivity, then fit parameters and defend held-out predictions.

**Prerequisites:** 04: Math & optimization.

**Hardware:** CPU; target accelerator execution requires separate evidence.

## Study guide: Are you fitting the physical model or the solver’s error?

Use an analytic system to separate simulation correctness, numerical discretization and parameter inference. An accurate autodiff derivative can still belong to an inaccurate discretized model.

### Check your starting point

A solver’s output is differentiable. Does that establish that its gradient matches the continuous differential equation?

<details><summary>Compare your reasoning</summary>

No. Autodiff follows the implemented discrete steps. Compare a discrete reference and refine the step size toward an analytic or higher-accuracy continuous reference.

</details>

Review: [Differentiate through a solver](03-differentiate-through-a-solver/docs/en.md).

### Build in stages

1. **Validate dynamics and axes.** Compare one integration step independently, refine the grid and separate independent trajectory axes from dependent time steps.

   Lessons: [A functional physical simulation](01-a-functional-physical-simulation/docs/en.md) · [Vectorize trajectories and scan time](02-vectorize-trajectories-and-scan-time/docs/en.md).

2. **Fit an identifiable inverse problem.** Check objective derivatives, choose observations that constrain the unknown parameter and evaluate predictions at held-out times. Separate noise sensitivity from discretization bias.

   Lessons: [Differentiate through a solver](03-differentiate-through-a-solver/docs/en.md) · [Fit parameters and scale the simulation](04-fit-parameters-and-scale-the-simulation/docs/en.md).

### Try a changed condition

All observations are taken at the initial time of a decay process. Can they identify its decay rate?

<details><summary>Compare an approach</summary>

They identify initial amplitude but contain no elapsed-time decay signal. Collect later observations and inspect parameter sensitivity; more optimizer steps cannot create missing information.

</details>

**Symptom:** The fitted rate changes as the time step is refined.

**Check next:** Compare solver error and parameter estimates across resolutions before attributing the shift to physical variation.

### Decide what is ready

Use scientific-inverse. Keep discrete and continuous references, a refinement study, derivative checks, recovered parameters and a held-out/noise sensitivity report.

### Further work

Adaptive solvers, event handling, stiff systems, PDEs and domain-specific physical validation remain outside the scalar ODE lab.

## Lesson sequence

### 10.01 A functional physical simulation

[Read the lesson](01-a-functional-physical-simulation/docs/en.md) · [Run the code](01-a-functional-physical-simulation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement a pure scalar state transition with units and a fixed time grid. Separate a discrete implementation oracle from continuous approximation error.

**Evidence:** Full geometric/exponential comparisons, fixed-horizon refinement table, unstable-step failure and repair. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** The code matches the geometric-sequence reference exactly but differs from the exponential. What have we established?

### 10.02 Vectorize trajectories and scan time

[Read the lesson](02-vectorize-trajectories-and-scan-time/docs/en.md) · [Run the code](02-vectorize-trajectories-and-scan-time/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Construct and verify a fourth-order fixed-step transition. Distinguish batch-major from time-major trajectory layouts.

**Evidence:** Polynomial/exponential references, batch/time shape diagram, permutation and final-only checks. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** What does a shape of (3, 41) mean in the batch-major solver?

### 10.03 Differentiate through a solver

[Read the lesson](03-differentiate-through-a-solver/docs/en.md) · [Run the code](03-differentiate-through-a-solver/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive an independent discrete solver sensitivity. Check an observation-loss gradient at a nonoptimal parameter.

**Evidence:** Polynomial derivative, finite-difference window, sensitivity refinement and detached-gradient diagnosis. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** An autodiff derivative matches the discrete RK4 formula, but both differ from the exact ODE derivative. What should you test next?

### 10.04 Fit parameters and scale the simulation

[Read the lesson](04-fit-parameters-and-scale-the-simulation/docs/en.md) · [Run the code](04-fit-parameters-and-scale-the-simulation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Fit a constrained physical parameter through an executed simulator. Separate training fit, held-out prediction and solver refinement.

**Evidence:** Full loss/rate histories, held-out and refined-grid metrics, noisy reference search, timing boundary and limitations. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** You fit only observations at time zero and every rate gives zero loss. What is the problem?

## Phase project

A differentiable inverse problem.

**Demonstrate:** Recover a known physical parameter and report error, stability, and gradient checks.

Project status: implemented staged practice · [Open source](../../projects/scientific-inverse/README.md). Copy `projects/scientific-inverse/starter/model.py` to `projects/scientific-inverse/my_model.py` and write your code in `projects/scientific-inverse/my_model.py`. Run `python3 projects/scientific-inverse/tests/check.py --implementation projects/scientific-inverse/my_model.py --stage 1` from the top-level folder to verify all 3 stages.



[Primary documentation](https://docs.kidger.site/diffrax/).
