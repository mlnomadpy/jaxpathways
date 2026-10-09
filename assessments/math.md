# Math synthesis: explain an optimizer’s behavior

**Scope:** linear algebra, conditioning, spectrum analysis, and optimizer dynamics across the twelve Math & optimization lessons and the [regression audit project](../project.html?id=regression-audit).

Create `math_assessment.py` and a report. Write predictions before execution, then keep actual arrays, plots and assertions. Use independent NumPy calculations for your reference, JAX for automatic derivatives and updates, and a recorded float32 or float64 policy. An oracle that calls the same autodiff function is not independent.

## Task 1: connect least squares, geometry and curvature

Use the three-row design matrix and target vector below. The first column is a constant feature; both entries of the weight vector are parameters.

```python
# Reference snippet
# Evaluate `X` from the current inputs and state.
X = [[1., 0.], [1., 1.], [1., 2.]]
# Evaluate `y` from the current inputs and state.
y = [1., 2., 2.]
# Evaluate `w0` from the current inputs and state.
w0 = [0., 0.]
```

Define \(L(w)=\lVert Xw-y\rVert_2^2/3\). Derive its gradient and Hessian before using autodiff. Solve least squares using a host solver, compute the signed residuals and check that \(X^\top(Xw-y)\) is approximately zero. Explain this orthogonality as a projection statement; it does not require every residual to be zero.

1. Compare your gradient with JAX at the initial point and one nonzero point. Compare a directional derivative along \(v=(1,-2)\) with a central difference. Start with perturbation `0.01` and report dtype and observed error before choosing a tolerance.
2. Calculate the Hessian eigenvalues. Derive the constant-rate interval for contraction of every error component on this positive-definite quadratic: \(0<\eta<2/\lambda_{\max}(H)\).
3. Run rates `0.1` and `0.5` from the same initial point for 20 updates. Plot loss against update number and explain each trajectory through the eigenvalue condition. Label any log scale and do not silently discard exploding values.
4. Replace the second column with a copy of the first. Explain why individual weights are no longer identifiable, even though a least-squares prediction still exists. Demonstrate two different weight vectors that produce identical predictions.

**Keep:** independent solution, residual table, directional check, eigenvalues, two explained curves and the rank-deficient counterexample. Report numerical error and justify tolerances instead of widening them until a check passes.

## Task 2: distinguish regularization from validation

For the original matrix, define \(L_\lambda(w)=\lVert Xw-y\rVert_2^2/3+\lambda\lVert w\rVert_2^2\), with \(\lambda=1\). This task regularizes both parameters, including the constant feature’s coefficient.

Derive the regularized normal equations and compare their solution to gradient-based optimization. Explain which part of the objective shrinks the parameter norm and why the unregularized training residual can increase.

Evaluate both the unregularized and regularized solutions on a fixed new input `[1., 3.]` with target `2.5`. Report prediction and squared error. This single observation illustrates a comparison; it requires separate verification to establish a generally superior regularizer. Describe a train/validation/test protocol in which validation chooses the regularization strength and the final test set is used only after that choice.

Draw predictions from both fitted lines alongside the three training points and the distinct held-out point. Name the point used for selection, if any, and explain what the plot can and requires separate verification to establish.

**Keep:** both objectives, parameter norms, the two prediction errors, annotated plot and the split protocol. Do not report the penalized objective as if it were ordinary prediction error.

## Task 3: measure minibatch gradient noise exactly

At zero weights, calculate the gradient contributed by each of the three observations. The full gradient is their arithmetic mean. Enumerate all nine ordered batches of size two sampled uniformly **with replacement**. Compute the mean gradient of each batch, then the population mean and covariance across the nine batch gradients.

Verify that the batch-gradient mean equals the full gradient and that its covariance is half the single-observation covariance. Explain the assumption that makes this result valid. Next enumerate the three size-two subsets sampled **without replacement** and measure the different covariance; do not apply the independent-sampling formula unchanged.

Plot the individual gradients, batch gradients and full gradient in the same coordinate system. Explain why points spread around the full gradient even when the estimator is unbiased. This plot describes the fixed initial parameters, not the entire training trajectory.

**Keep:** enumeration code, all gradients, both covariance matrices, the explained plot and the sampling assumptions. Use population normalization rather than the sample-estimator correction for the exact enumerated distribution.

## Task 4: explain clipping and Adam’s first update

Start with gradient \(g=(3,4)\), zero Adam moments, learning rate \(0.1\), moment coefficients \(0.9\) and \(0.999\), and epsilon \(10^{-8}\) added outside the square root. Clip the gradient by global norm to a maximum of one **before** updating moments.

1. Calculate the clipped gradient by hand. Contrast global-norm clipping with clipping each coordinate separately.
2. Derive the first moments, bias corrections and first parameter update. Compare to an independently written NumPy implementation and to Optax with the same order and epsilon convention.
3. Compare unclipped and clipped first Adam updates, then compare their plain gradient-descent updates. Explain why a bound on gradient norm is not automatically the same bound on Adam’s parameter-update norm.
4. Repeat two updates using a second gradient `(0.3, 0.4)`. Preserve the first step’s moments and count; show why resetting them changes the second update.
5. Apply a two-step learning-rate schedule of `0.1` then `0.01`. Record the rate actually used at each update and verify that the moment state is not reset by a schedule change.

**Keep:** hand calculation, named optimizer-state leaves, stepwise NumPy/Optax comparisons and a short diagnosis of one deliberately reset-state run.

## Review the reasoning and the evidence

Review each task separately as **accept**, **revise**, or **not demonstrated**. Accept only when the implementation, independent calculation, observed output and explanation agree.

- Geometry: projection residuals and the rank-deficient example are correct; the eigenvalues explain the two curves.
- Regularization: the penalty convention is explicit; training objective and held-out error are distinguished; no generalization claim rests on one point.
- Sampling: replacement policies, ordered outcomes and covariance normalization match the stated distribution.
- Optimizer: clipping order, epsilon placement, bias correction and schedule indexing are explicit; the comparison preserves state.
- Figures: axes, units or scales and plotted quantities are named; the explanation cites actual values and a limitation.

Keep unresolved differences and the next diagnostic question in your report. After your attempt, [read the reviewer notes](math-reviewer.md). These public notes help review the reasoning; they do not supply independent evidence of a learner’s work.
