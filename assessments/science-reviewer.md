# Reviewer notes: science synthesis

Use these analytical derivations to check your conservation invariants, discretization error, and inverse-problem gradients.

The conserved sum is \(s=a+b=a_0+b_0\). The difference solves \(d'=-2kd\), so

\[
a(t)=\frac{s+d_0e^{-2kt}}2,\qquad b(t)=\frac{s-d_0e^{-2kt}}2.
\]

For the first fixture, \(s=4\), \(d_0=2\), giving \(a(3)=2+e^{-2.4}\approx2.09071795\) and \(b(3)=2-e^{-2.4}\approx1.90928205\). The first-compartment rate derivative is \(-t d_0e^{-2kt}\), about \(-0.54430772\) at three seconds. The total derivative is zero. The two-compartment linear RK4 update preserves the sum to floating-point error even when its difference-mode trajectory has noticeable discretization error.

Check that the implementation integrates a vector state, not two independent scalar decay equations. Those would incorrectly decay the total. The initial state belongs in the returned trajectory. The batch dimension indexes independent pairs; the final state dimension indexes compartments.

For inference, a noise-free rate is not a compulsory result on noisy observations. Compare against the specified analytic grid search. The two starts should approach the same basin for this bounded smooth one-parameter problem, but a stalled optimizer must be diagnosed from evidence rather than hidden by rewriting the data. No fixed timing or noisy-fit result is prescribed by this draft.

Review grid refinement at fixed physical observation times. Doubling internal steps changes the numerical method; changing measurement times changes the experiment. Interpolate or save exactly at observation times with a documented contract. Distinguish evaluating a fitted rate on a finer grid from optimizing a new rate on that grid.

A single seed and one synthetic family do not establish calibrated uncertainty, real-world model adequacy, experimental identifiability under all conditions, or hardware scaling.
