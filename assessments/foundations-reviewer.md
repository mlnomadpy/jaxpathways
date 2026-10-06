# Foundations synthesis: reviewer notes

Use these after inspecting the learner’s attempt. Numerical answers alone are insufficient evidence. [Return to the assignment](foundations.md).

Task 1 targets are `[3.25,0.25,-4.25]`, predictions `[-1.5,-0.5,1]` and residuals `[-4.75,-0.75,5.25]`. The loss is `50.6875/3 = 16.8958333333`. Derivatives are `2*sum(residual*x)/3 = 16.8333333333` and `2*sum(residual)/3 = -0.1666666667`. The first update at rate 0.025 gives weight `0.0791666667` and bias `-0.4958333333`. These values should be derived rather than supplied as an answer hint.

For the changed fixture, the mean-MSE Hessian is `2*[[13/3,1/3],[1/3,1]]`. Its largest eigenvalue is approximately `8.733`; rate 0.1 lies within the quadratic contraction interval `0 < rate < 2/lambda_max`. An independent NumPy float32 loop from the specified initial parameters reaches weight `-1.5` and bias `0.249999896` after 200 updates, comfortably within the required `1e-4` tolerance. This verifies that the assignment’s recovery requirement is feasible; it does not prescribe a rate for neural networks or Adam.

For Task 3, mean(x)=0 and the bias target is 1. With `e_t=b_t-1`, `e_(t+1)=(1-2*rate)*e_t`. Rates 0.1, 0.5 and 1.1 give factors 0.8, 0 and -1.2. Rate 0.5 eliminates bias error in one update; weight error need not vanish then. Rate 1.1 produces alternating growth even while early losses remain finite. The contracting interval for this isolated bias component is `(0,1)`.

Task 4 produces a `(3,3)` matrix: its rows index targets and its columns index predictions. An explicit `targets[:,0]` conversion is reasonable only when the caller has validated the intended single target column. `squeeze()` without an axis can destroy a singleton batch dimension.

Do not require exact finite-difference errors or timings across machines. Verify explanations against the actual dtype/values. If copied numeric answers appear without code or reasoning, request a new three-point fixture rather than awarding acceptance.
