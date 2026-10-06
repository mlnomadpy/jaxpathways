# Math synthesis reviewer notes

This is an editorial review draft. Check the learner’s independent implementation and explanations as well as the values below. These references were calculated with host NumPy; they are not a learner execution receipt.

## Geometry

For the mean squared objective, the gradient is twice the transposed design matrix times the residual vector, divided by three. The Hessian is `[[2,2],[2,10/3]]`. Its eigenvalues are approximately `0.55848156` and `4.77485177`, so the strict rate ceiling is approximately `0.41886117`.

The least-squares weights are `[7/6, 1/2]`, signed residuals `[1/6,-1/3,1/6]`, and mean squared loss `1/18`. Rate `0.1` contracts both eigendirections. Rate `0.5` expands the largest-eigenvalue direction. Ask the learner to connect a visible curve segment to this recurrence.

With identical constant columns, only the sum of the two coefficients is identified. For example `[5/3,0]` and `[0,5/3]` give the same optimal constant prediction. A zero eigenvalue prevents strict contraction of every parameter-error direction.

## Regularization

At strength one, the normal equation adds three times the identity to the Gram matrix. The weights are `[22/39,21/39]`. The unregularized prediction at the new row is `8/3`; the regularized prediction is `85/39`. Their squared errors against `5/2` are respectively `1/36` and `625/6084`. This example does not show a universal benefit from regularization.

## Sampling

Per-observation gradients at zero are `[-2,0]`, `[-4,-4]`, `[-4,-8]`; their mean is `[-10/3,-4]`. The population covariance is `[[8/9,8/3],[8/3,32/3]]`. Size-two independent sampling halves this matrix. Uniform size-two sampling without replacement gives one quarter of the original covariance. Ordered pairs with replacement include repeated observations; subsets without replacement do not.

## Adam

The norm is five; norm-clipped gradient is `[0.6,0.8]`. The first moments are `[0.06,0.08]` and `[0.00036,0.00064]`. Bias corrections recover the clipped gradient and its elementwise square. The first update is approximately `[-0.1,-0.1]`, with a small epsilon effect. Its norm is about `0.1414`, whereas the plain gradient-descent update after clipping has norm `0.1`.

Check the second update using the retained first moments and count. A correct first update is not enough to validate the state transition. Optax comparison must use the same chain order and epsilon placement. A schedule scales the update at its stated count; it does not reset the moments.
