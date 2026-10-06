# Science synthesis: defend an inferred physical parameter

**Status:** review draft. **Scope:** fixed-step differentiable cooling, numerical error, parameter inference, and CPU measurement. Complete the scientific computing lessons and [scientific inverse project](../project.html?id=scientific-inverse). This assessment is public and has reference notes; completion requires a review of your own evidence, not copying expected output. No accelerator qualification or expert scientific approval is implied.

## Prepare an independent assessment

Retain your project implementation and create a separate assessment script and report. Record Python, JAX, NumPy, backend, device count and dtype. Keep predictions, observed values and conclusions separate. Run the project checker first, then transfer the methods to the fixtures below without changing its checks.

## Task 1: demonstrate conservation in a different physical system

Two compartments exchange a conserved substance. Their state is \(u=(a,b)\), with

\[
\frac{da}{dt}=-k(a-b),\qquad \frac{db}{dt}=k(a-b).
\]

Use \(k=0.4\), initial state \((3,1)\), and a horizon of three seconds. Derive the conserved total \(a+b\) and an independent solution by introducing the difference \(d=a-b\). Implement a vector-state RK4 solver and verify both compartments at every saved time with forty and eighty updates. State tolerances and show the initial point. Compare conservation error and trajectory error separately: a method can conserve a quantity while approximating its trajectory poorly.

**Keep:** derivation, state/axis diagram, independent reference, refinement table and an explanation of why conservation alone is insufficient.

## Task 2: check both useful and uninformative sensitivities

Differentiate the final first-compartment value with respect to \(k\). Compare with an analytic expression and central differences at two perturbations. Then differentiate the final total \(a+b\): explain why its sensitivity is zero for every rate.

Demonstrate that fitting only total-substance measurements cannot identify the rate, even with many observation times. Propose a measurement that can distinguish rates and show its nonzero sensitivity. A zero derivative here is a physical property, not automatically a software defect.

**Keep:** derivative derivation, measured comparisons, an unidentifiable objective, and a repaired observation design.

## Task 3: infer from one compartment and test a new initial condition

Generate independent analytic first-compartment observations at twenty-one evenly spaced times over three seconds. Use true rate \(0.4\), initial state \((3,1)\), and NumPy generator seed `71` with additive Gaussian noise of standard deviation \(0.005\). Freeze this dataset before fitting.

Fit a positive rate from initial guesses \(0.2\) and \(0.9\). Compare each estimate with an independent analytic grid search over \([0.25,0.55]\), with at least three thousand intervals. Report actual fit quality; do not select a new noise seed when a result looks inconvenient. Inspect the objective profile around the minimum rather than inferring uncertainty from one point estimate.

Before fitting, reserve a noiseless test trajectory with initial state \((4,0.5)\) and the same true rate. Evaluate both compartments without further updates. Refit on a finer numerical grid and report parameter movement. Compare fixed-parameter refined predictions separately from refitting; they answer different questions.

**Keep:** immutable data-generation rules, two complete histories, independent search, held-out errors, numerical-refinement evidence, and the limits of a single synthetic dataset.

## Task 4: explain a figure and defend timing boundaries

Create a two-panel figure: observed first-compartment data with fitted trajectories, then residual versus time. Label units, initial conditions and the known synthetic truth separately from estimates. Describe at least two actual visible values or patterns, and explain what the figure cannot establish.

Measure batches of one, thirty-two and two hundred fifty-six initial states on the available CPU. Compile/warm up each shape, use already-placed inputs, synchronize outputs, and collect at least seven samples. Report median latency and time per trajectory with the batch, grid, precision and saved-output policy. Do not assert a speedup unless the measurements show it. A CPU batch curve provides no TPU or multi-host throughput evidence.

**Keep:** plotting code and data, figure interpretation, timing samples and an explicit inclusion/exclusion boundary.

## Reviewer decision

- **Accept numerical model:** An independent two-compartment derivation, correct state/time axes, conserved total and convergent trajectory errors are all present. **Revise:** only loss decreases or conservation is used as the sole accuracy test.
- **Accept sensitivity:** Both informative and uninformative derivatives are explained and checked. **Revise:** all zero gradients are called bugs or the oracle repeats the same autodiff program.
- **Accept inference:** Fixed noisy observations, changed initial guesses, independent search, untouched held-out states and separate refinement/refit comparisons are reported. **Revise:** data or tolerances are selected after observing a desired answer.
- **Accept explanation:** The figures are interpreted using actual data and the timing boundary is reproducible. **Revise:** generic captions replace observations or CPU results are described as hardware scaling proof.

Submit the implementation, executable assessment, report, source data and figure assets. An unresolved discrepancy is evidence to investigate, not a reason to remove the comparison.
