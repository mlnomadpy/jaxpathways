# Math & optimization: teaching sequence

This revision expands phase 04 from four lessons to twelve. It starts from Python, array operations, basic derivatives and the earlier JAX transformations/state lessons. It does not assume a prior linear algebra or probability course. New concepts are introduced through small calculations before the JAX API.

## Learning sequence

| Order | Stable lesson ID | Learner question | Evidence |
| --- | --- | --- | --- |
| 1 | optimization-05 | What does a dot product measure? | Hand projection, orthogonal residual, rescaling and zero-direction checks |
| 2 | optimization-01 | How do rows become predictions and one loss? | Shape contract, hand residuals, weight/bias derivatives, mean-versus-sum gradients |
| 3 | optimization-06 | Can we solve a fit directly, and trust its coefficients? | Least-squares reference, numerical rank, target perturbation, conditioning |
| 4 | optimization-07 | How does sensitivity move through a vector function? | Hand Jacobian, JVP/VJP, adjoint identity, detached-gradient diagnosis |
| 5 | optimization-02 | How do we check a derivative independently? | Perturbation sweep, analytic comparison, nonquadratic transfer, kink counterexample |
| 6 | optimization-08 | Why does a learning rate work in one direction and fail in another? | Hessian products, derived stability interval, scaling and saddle examples |
| 7 | optimization-03 | What does an explicit update actually do? | Hand steps, divergent trajectory, stopping-rule counterexample |
| 8 | optimization-09 | Why do classification losses use logarithms? | Bernoulli likelihood, stable logits loss, extreme-score repair, multiclass shift invariance |
| 9 | optimization-10 | When should we accept a worse training fit? | Ridge derivation, separate prediction/objective losses, validation sweep, bias mask |
| 10 | optimization-11 | Is a minibatch gradient wrong or noisy? | Exact expectation/variance, unequal-batch weighting, accumulation and sampling diagnosis |
| 11 | optimization-04 | What state does an optimizer remember? | Manual/Optax equivalence, momentum continuation, replay and zero-gradient coasting |
| 12 | optimization-12 | How do we justify Adam, clipping and a schedule? | Two manual Adam steps, clipping order, schedule boundary, controlled optimizer comparison |

Lesson IDs and existing directories remain stable so saved learner progress and deep links survive. Manifest order, rather than the numeric suffix of a directory, determines teaching order. All routes that reference the phase receive the expanded sequence.

## Authoring and delivery

Each of the eight new lessons has a concrete problem, defined notation, worked calculations, staged implementation, two prediction experiments, a main exercise, two transfer/diagnosis tasks, a checkpoint and primary references. The original four each receive an additional explanation and experiment. Math uses explicit inline TeX and display equations; executable Python remains literal code. Each source generates browser content, Markdown, Python, notebooks and offline books.

The phase is a foundation for the existing regression audit and later model/deployment work. The final lesson explains how to retain the new evidence alongside the project. The existing project's public checks still cover their stated four stages; they do not assess every new topic. No new certification or reviewed-mastery claim is added.

## Boundaries and further work

These are authored teaching drafts with numerical reference checks. They need learner walkthroughs and independent pedagogical review. Tiny synthetic examples demonstrate mechanisms, not optimizer rankings or generalization guarantees. Practice time estimates are provisional. This expansion does not attempt full courses in probability, numerical linear algebra or convex optimization. Longer statistical estimation, constrained optimization, proximal methods and second-order methods remain future curriculum decisions rather than placeholder lessons added here.

Validation is recorded in the source-bound CPU receipt and the whole-content audit. Browser, notebook and export checks must pass for this content revision before reporting it as validated. No accelerator performance is inferred from CPU execution.
