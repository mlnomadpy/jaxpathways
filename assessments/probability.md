# Probability synthesis: distinguish inference error from model error

**Scope:** exact conjugate inference, Hamiltonian Monte Carlo sampling, variational approximation, and posterior predictive checks building on the four probability lessons and [Bayesian regression project](../project.html?id=bayesian-regression).

Create probability_assessment.py and a report. Write predictions before running experiments, record actual arrays and seeds, and retain failures. Use CPU float32 JAX calculations and independent float64 NumPy references with justified tolerances.

## Task 1: derive a posterior before approximating it

Use design rows \((1,-1),(1,0),(1,1)\), targets \((-1,1,3)\), observation standard deviation \(\sigma=1\) and prior standard deviation \(\tau=2\). State which variables are random and which are conditioned on.

1. Derive the posterior precision, mean and covariance. Check the rational-number result by hand and the general formula with a NumPy solve.
2. Compare the gradient of the normalized log joint with an independent analytic gradient at two weight vectors.
3. Shift all nonconstant input values by one. Recompute the posterior and explain why intercept-slope covariance is now nonzero.
4. Repeat the original rows in the array and show what the model would infer if they were independent measurements. Explain why a copied file does not justify that independence assumption.

**Keep:** model statement, formulas, two gradient checks, covariance table and duplicate-data warning grounded in the likelihood.

## Task 2: defend the sampler evidence

Run four dispersed HMC chains against the exact posterior or an explicitly declared correlated Gaussian test target. Keep momentum and acceptance keys separate.

1. Check leapfrog reversibility with a momentum reversal.
2. Compare retained means and the full covariance matrix with the exact answer for two seeds.
3. Plot separate traces; record fixed warmup discard, step size, trajectory length and acceptance.
4. Compute classical split R-hat and explicitly distinguish it from modern rank-normalized split R-hat.
5. Reproduce unstable integration with a larger step and stuck/disagreeing chains with a synthetic fixture. Explain why pooling chains or removing repeated rejected states conceals a failure.

**Keep:** code, environment, actual energy-error summaries, numerical tolerances, traces and a statement of what these checks cannot establish. Explain why high acceptance alone does not imply effective exploration.

## Task 3: measure variational family error

Let the exact target have mean \((1,-1)\) and covariance

\[
\Sigma=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}.
\]

Approximate it with a diagonal normal distribution by minimizing \(\operatorname{KL}(q\Vert p)\).

1. Derive the optimal mean and diagonal variances from the Gaussian KL formula.
2. Optimize the objective with JAX and compare to the independent result.
3. Compute the variance of the sum of the two coordinates under the target and under the approximation.
4. Explain why a small optimization gradient does not remove the remaining KL.
5. Change the correlation to zero and show how the limitation changes. Describe a variational family that can represent the original target exactly.

**Keep:** exact and optimized parameters, remaining KL, a variance comparison plot and an explanation of approximation error versus optimization error.

## Task 4: examine the prediction, not just the parameters

For the original regression model, compute predictions at inputs zero and two. Report the mean, latent variance and observation variance separately.

Draw posterior weights and independent observation noise. Verify the empirical prediction variance against the analytic result. Then construct a deliberately shifted or curved outcome fixture, without changing the fitted model, and measure how its residual pattern and interval coverage differ.

Explain why model-generated self-coverage checks arithmetic but cannot establish real-data calibration. Propose a real-data protocol with frozen splits, provenance, group sizes, a predeclared discrepancy and a final held-out report. Do not describe that proposed study as executed.

**Keep:** a prediction plot with axes, units and interpreted values; model-consistent and mismatched results; and the proposed external-validation protocol.

## Review

Mark each task **accept**, **revise**, or **not demonstrated**. Accept requires agreement between the implementation, independent reference, observed output and explanation. Figures must explain actual plotted quantities. A reviewer should reject convergence claims based only on near-one R-hat, predictive intervals missing observation noise, or an approximation described as exact because its means match.

After attempting the assessment, read [reviewer notes](probability-reviewer.md).
