# Probability synthesis reviewer notes

These notes supply public reference reasoning. They do not prove that a learner executed or independently understood the work.

## Exact posterior

With the stated centered design, \(X^\top X=\operatorname{diag}(3,2)\) and \(X^\top y=(3,4)\). Prior precision is \(I/4\), so posterior mean is \((12/13,16/9)\) and covariance is \(\operatorname{diag}(4/13,4/9)\). The normalized log-joint gradient is \(X^\top(y-Xw)/\sigma^2-w/\tau^2\).

After shifting inputs to zero, one and two, \(X^\top X=\begin{pmatrix}3&3\\3&5\end{pmatrix}\). The posterior covariance has a negative off-diagonal entry: intercept and slope compensate. The exact values can be checked by a host solve. A likelihood over duplicated rows treats them as independent observations; copying identical records is not evidence for that assumption.

## Sampler

Reversibility should return the starting position and negative initial momentum within a justified numerical tolerance. Do not require an exact string representation of samples. Require checks of the full covariance, not only marginal means.

Keep rejected positions: their repetition represents time spent in that state under the Markov transition. Extremely high acceptance can coexist with short, ineffective motion; very low acceptance with large energy errors suggests poor integration settings. Classical split R-hat close to one is not proof of convergence. Modern rank-normalized diagnostics, effective sample sizes, Monte Carlo standard errors and divergence diagnostics complement the teaching implementation.

A zero within-chain variance makes the formula invalid. The right response is to report a stuck-chain diagnostic failure, not force the value to one.

## Variational approximation

For the correlated target, diagonal precision is \(1/0.36\); minimizing this KL direction yields diagonal variances \(0.36,0.36\). The target variance of the sum is \(1+1+2(0.8)=3.6\), while the approximation gives \(0.72\). The minimum KL is \(-\tfrac12\log(0.36)\approx0.510826\). This nonzero remainder is family error. Zero correlation lets the diagonal family match exactly; a full-covariance Gaussian can represent the original target.

Do not generalize this example into a claim that all variational procedures underestimate every uncertainty measure.

## Prediction and model checking

At input zero, the latent variance is \(4/13\); the observation variance is \(17/13\). At input two, the latent variance is \(4/13+16/9=244/117\), and observation variance is \(361/117\). Standard deviations require square roots; adding standard deviations directly is wrong for independent noise.

Simulation under the fitted model can check the arithmetic and interval construction. It does not show that the assumed model fits external observations. A shifted mean or curved residual pattern can expose model mismatch even with a numerically correct posterior. Require the learner to distinguish a small deterministic stress fixture from an estimated population coverage rate and to keep an external-data study labeled as proposed until it is executed.
