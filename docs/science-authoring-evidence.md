# Scientific computing authoring evidence — 2026-10-05

## Scope and status

Four substantive CPU lesson sources are ready for registry integration and whole-course execution. They use the existing planned paths and IDs `science-01` through `science-04`. No shared curriculum, generator, route or registry was edited by this authoring agent. Root integration and the full generated script/notebook smoke suite remain necessary before the website can expose them with complete execution receipts.

| Lesson     | Minutes | Concrete result                                                                                                                                                                  |
| ---------- | ------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| science-01 |      85 | Pure Euler cooling simulation; independent discrete/continuous references, refinement, sign preservation and stability diagnostics                                               |
| science-02 |     100 | RK4 scan/vmap ensemble; batch/time contracts, polynomial oracle, independent rates and Cartesian-product transfer                                                                |
| science-03 |     105 | Discrete solver and continuous sensitivities; loss finite differences, fourth-order convergence, weighted objective and detached-gradient diagnosis                              |
| science-04 |     115 | Positive-rate inverse fit; independently generated data, held-out/refined grids, non-identifiability, Euler parameter bias, noisy reference search and synchronized CPU batching |

The work includes staged builds, two executed experiments and two distinct transfer practices per lesson, KaTeX prose, independent oracles, and four interpreted figure artifacts. Every figure was regenerated in a fresh CPU process after the final content edit. No receipt hashes were copied or retagged.

## Root integration instructions

For each existing science lesson in `curriculum/course.json`:

- Preserve its ID, title, path and sequential prerequisites.
- Set `status` to `authored`.
- Set `exercise` to the canonical `lesson.json` content exercise.
- Replace placeholder `evidence`, `objective` and `check` using the table below. The generator reads source minutes/content from the existing path.

| ID         | Objective                                                                                                            | Evidence                                                                                                              | Check                                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| science-01 | Build a functional simulation and distinguish implementation error, time-step error and numerical instability.       | Full geometric/exponential comparisons, fixed-horizon refinement table, unstable-step failure and repair.             | Explain why matching the discrete oracle does not establish agreement with the continuous model.        |
| science-02 | Batch independent trajectories and verify time/state axes with a higher-order scan solver.                           | Polynomial/exponential references, batch/time shape diagram, permutation and final-only checks.                       | Distinguish paired independent experiments from a Cartesian parameter grid or a coupled physical state. |
| science-03 | Differentiate a numerical solver and check both discrete and continuous sensitivities independently.                 | Polynomial derivative, finite-difference window, sensitivity refinement and detached-gradient diagnosis.              | Explain why a correct autodiff gradient may retain solver discretization bias.                          |
| science-04 | Recover a positive physical parameter and audit held-out predictions, identifiability and CPU batching measurements. | Full loss/rate histories, held-out and refined-grid metrics, noisy reference search, timing boundary and limitations. | Distinguish optimizer failure, uninformative observations and numerical parameter bias.                 |

Recommended phase metadata:

- `projectId`: `scientific-inverse`
- `description`: `Build and differentiate a physical simulator, recover a positive rate from observations, and separate numerical accuracy, identifiability and measured execution cost.`
- `learningAdvice`: `Complete gradients, explicit state and scan first. Begin with an analytic reference, check discretization and sensitivity, then fit parameters and defend a held-out inference report.`
- `hardware`: `CPU float64 reference; accelerator scaling requires separate execution evidence`

Add `projects/scientific-inverse/project.json` to `curriculum/projects.json`. Its explicit `requiredLessonIds` list contains all four science lessons. The project has three cumulative stages and an implementation-path checker, CPU starter/reference, README, replay and independent noisy-data checks.

Add this assessment registry entry:

```json
{
  "id": "science",
  "projectId": "scientific-inverse",
  "title": "Scientific inverse-problem synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/science.md",
  "url": "assessments/science.html"
}
```

Update `learning-paths/science.json` to remove the now-stale missing-specialization text and connect capstone `projectId: scientific-inverse`, `assessmentId: science`, authored project status with explicit public review-draft assessment limits. Suggested description: `Simulate physical dynamics, check numerical and derivative accuracy, and recover unknown parameters with held-out evidence. Build a reusable CPU inverse-problem audit before extending to adaptive solvers or accelerators.` Suggested first artifact: `A cooling simulator with a convergence table and an independent sensitivity check.` Core libraries should read `JAX · NumPy; optional Diffrax ecosystem extension` rather than imply executed Diffrax/Equinox/Lineax use.

## Executed checks

Environment: Python 3.14; JAX 0.9.2; NumPy 2.4.4; CPU float64 (enabled before array construction). Existing figure renderer/Matplotlib used without changes.

1. Executed each lesson's complete example, both experiments, main reference solution, both practice solutions and figure-data code sequentially in a separate Python process. All four passed. Build steps concatenate exactly to the complete code, so this also exercised their code in order.
2. Loaded `scripts/render-lesson-figures.py` via importlib and called `worker({'id': ..., 'path': ...})` for each lesson in a fresh Python process after final content edits. All four passed; local PNGs were visually inspected for correct legends, scales, readable axes and agreement with prose. Source-bound SVG/PNG/visual JSON artifacts are present.
3. Rendered all canonical non-code fields through `inlineMath` / `renderMath`: 71, 71, 72 and 78 fields respectively. No KaTeX errors.
4. Ran `python3 projects/scientific-inverse/tests/check.py --implementation solution --stage all`: all three stages passed. Changed true rates 0.35 and 1.05 recovered approximately 0.35000000028 and 1.05000006945; held-out RMSE was near float64 roundoff on the same discrete grid. Refined-grid error passed its separate bound. Noisy fit agreed with independent analytic grid search within 0.0002.
5. Ran the starter at stage 1: it failed at the intended `NotImplementedError`, proving the scaffold does not pass as completed work.
6. Independently executed a two-compartment vector RK4 check for the assessment derivation: 40/80-step maximum trajectory errors were 4.176e-8 / 2.546e-9; conserved-sum errors below 1.8e-15. Endpoint approximately (2.09071798, 1.90928202); autodiff rate sensitivity -0.54430756 versus analytic -0.54430772. Seed-71 noisy analytic grid minimum was 0.4004. The assessment does not prescribe machine timings or require a noisy fit to equal the generating parameter exactly.

Representative lesson outputs:

- Euler coarse/fine endpoint: 0.3570125 / 0.42920185; analytic 0.49319393. Last Euler refinement error ratio 2.0287.
- RK4 batch endpoints: 0.12329848, 0.24659697, 0.49319394. Polynomial maximum difference 1.44e-15.
- Endpoint sensitivity: autodiff -0.98638780967, discrete oracle -0.98638780967, continuous oracle -0.98638785577.
- Sensitivity refinement errors: 2.449e-4, 1.319e-5, 7.654e-7, 4.609e-8.
- Inverse rate: 1.4 → 0.7000000090. Initial loss 0.1167663; final near 1.95e-32. Original-grid held-out RMSE 4.57e-16; finer-grid RMSE 7.28e-9. Prose explicitly explains why finer-grid residual can increase after the fitted rate compensated for tiny discretization bias.
- Perfect coarse Euler fit rate for true 0.7: 0.59062382. This exposes why low training loss alone does not justify the physical estimate.

## Boundaries and remaining review

No full repository build/check/smoke was run by this agent to avoid races with parallel authors. Root must integrate, regenerate, run those checks, and inspect the generated content. The assessment is a public review draft; its changed two-compartment core and analytic references were executed, but it has no independent expert or learner review. Browser review was not attempted.

Diffrax, Equinox and Lineax are not installed in the current environment. Official JAX scan/vmap/autodiff and Diffrax getting-started documentation were inspected. These lessons clearly label the installed-JAX fixed-step core and optional ecosystem extension. They do not claim adaptive solver, stiffness, event handling, Diffrax execution, TPU execution or distributed scaling validation.
