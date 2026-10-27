# Networks content repair execution notes

Scope: `networks-02`, `networks-03`, and `networks-05` canonical lesson JSON. No course registry or generated artifact files were changed by this repair.

Environment: `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`, JAX 0.9.2, Flax 0.12.6, Optax 0.2.8, CPU.

## Executed lesson sources

Generated reference scripts with:

```sh
node --input-type=module - <<'JS'
import fs from 'node:fs'; import {lessonScript} from './scripts/lesson-artifacts.mjs';
for(const slug of ['02-model-and-state-with-flax-nnx','03-a-compiled-train-and-evaluation-step','05-debug-unstable-learning']) { const lesson=JSON.parse(fs.readFileSync(`phases/05-networks/${slug}/lesson.json`)); fs.writeFileSync(`/tmp/${lesson.id}-lesson-script.py`,lessonScript(lesson)); }
JS
```

Ran all generated scripts, which execute build steps before the canonical main code, then figure code, experiments, exercise solution, and practice solutions:

```sh
for f in /tmp/networks-02-lesson-script.py /tmp/networks-03-lesson-script.py /tmp/networks-05-lesson-script.py; do /private/tmp/jaxpathways-0170-clean-learner-env/bin/python "$f"; done
```

Results: all three ended with `PASS`. Networks-02 main, both experiments, both practice references, and exercise passed. Networks-03 main, two original experiments, the BatchNorm/Dropout stateful experiment, both practice references, and exercise passed. Networks-05 main, both original experiments, both practice references, the new MLP exercise, and the visual-data calculation passed.

Key measured CPU fixture values from networks-05: reference MLP training BCE changed from 0.6258 to 0.5295 after 20 updates; held-out BCE changed from 0.6397 to 0.5417. The data fault was first detected at `[7, 1]`, then the repaired feature array exactly matched its clean source and held-out BCE stayed 0.6397. Setting all hidden biases to 20 saturated 100.0% of tanh outputs beyond 0.99; hidden-layer gradient norm was 0.00000 and output-layer norm was 0.19231. Held-out BCE changed from 0.6397 to 0.6903 and returned to 0.6397 after restoring the hidden bias. With rate 100, training BCE was 0.6258, 1.6054, then 32.9974 and held-out BCE reached 71.8487 after three updates. Rate 0.03 reduced both training and held-out BCE on the same fixture. These values are deterministic fixture checks, not general optimization claims.

Networks-03 stateful evaluation printed that repeated evaluation outputs matched and `Param`, `BatchStat`, and `RngState` were unchanged. Its negative-control training-mode call on a copied model changed BatchStat and RngState while leaving parameters separate.

## Figure/data schema checks

Rendered only the two repaired figures to a temporary directory, without changing checked-in assets:

```sh
python scripts/render-lesson-figures.py --lesson networks-02 --output-dir /tmp/network-figure-checks
python scripts/render-lesson-figures.py --lesson networks-05 --output-dir /tmp/network-figure-checks
```

Both returned `PASS`. Networks-02 diagram labels actual state paths and shared/copy identity conditions. Its plotted counts are `hidden.kernel=6`, `hidden.bias=3`, `out.kernel=3`, and `out.bias=1`, with the calls counter explicitly excluded. Networks-05 line data has four finite series on completed-update coordinates 0 through 3; assertions check that the high-rate held-out loss rises while the repaired-rate held-out loss falls.
