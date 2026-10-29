# Running focused checks

Use `npm test` during development. It runs the unit suite without generating the course or building the site. Run the broader suites when their inputs change; `npm run check` still runs every Node suite and the runner regressions in CI.

| Command | Coverage | Automatic preparation |
| --- | --- | --- |
| `npm test` / `npm run test:unit` | Isolated logic, storage, UI controls, release tooling, source architecture | None |
| `npm run test:app` | Manifest parsing, views, routes, math, notebook records | Course content |
| `npm run test:content` | Companions, downloads, guides, artifact provenance, Python audit/workspace checks | Course content |
| `npm run test:site` | Built lesson pages, reader interactions, offline book | Course content and Astro build |
| `npm run test:all` | All Node suites above | Course content and Astro build, once |
| `npm run test:tools` | Python runner selection, receipts, hashing and CI scopes | None; Python standard library |
| `npm run test:browser` | Real Chromium desktop/mobile behavior | Existing site build |

The content pipeline refreshes generated files. It does not rerun lesson executions, numerical figures or hardware qualification. A stale source-bound receipt is a real failure: regenerate it using the documented execution tool, or update its status when the evidence no longer applies. Test preparation does not change release snapshots; `npm run check` and `npm run build` retain the release gate.

To refresh a changed lesson, recompute its figure, generate the current notebook, execute both companions, then attach the recorded outputs:

```sh
npm run figures:build -- --lesson transformers-01
npm run content:build
npm run smoke:cpu -- --lesson transformers-01
npm run content:build
```

Authored source changes require a new release version and snapshot; existing published snapshots stay immutable. The release checker hashes the exact bytes of the declared corrupt-tokenizer canary so its intentional invalid JSON remains protected. Every other JSON source must parse successfully and retains canonical hashing. The release regression tests check byte changes, malformed neighboring JSON and snapshot preservation.

## Select by changed inputs

```sh
npm run test:changed -- --list                    # Preview selection for local changes
npm run test:changed                             # Staged, unstaged and untracked inputs
npm run test:changed -- --since origin/main       # Also include committed branch changes
npm run test:changed -- src/lib/storage.js        # Select from specific changed paths
npm run test:changed -- --suite unit              # Restrict selection to unit tests
```

Selection follows local JavaScript imports through their dependencies. Content and page inputs also select their generated-output consumers. Tooling, dependencies and unknown paths select all Node tests conservatively. Real browser tests and Python execution suites have separate commands; a Node selection does not run those suites. CI always runs the complete site checks, while PR path selection skips unrelated specialized execution jobs. Manual and reusable/deployment verification run all jobs by default.

Every Node test must appear in exactly one suite in `scripts/test-suites.mjs`. Adding, deleting or duplicating a test without updating that inventory fails immediately, so focused runs cannot silently forget new coverage.

## Select files or test names

```sh
npm test -- tests/choices.test.mjs
npm test -- tests/http.test.mjs --test-name-pattern 'timeouts'
npm run test:site -- --no-build                   # Reuse a known-current build
npm run test:all -- --no-build --test-concurrency 2
npm run test:browser -- browser-tests/reader-print.spec.ts --project desktop-chromium
```

Use `--no-build` only after generating the artifacts needed by the selected tests. Otherwise the runner builds the required content/site once before starting the tests. `--list` never builds or runs tests. Run `npm test -- --help` for all Node runner options.

## Run selected Python projects and lessons

```sh
npm run test:project -- --list
npm run test:project -- --project regression-audit --receipt /tmp/projects.json
npm run test:project -- --project regression-audit --project mlp-classifier --list
npm run smoke:cpu -- --phase welcome --list
npm run smoke:cpu -- --lesson welcome-01 --receipt /tmp/lessons.json
npm run smoke:cpu -- --phase transformers --receipt /tmp/transformers.json
```

Project IDs and phase IDs come from the course registry. Selectors can be repeated, and unknown or empty selections fail before execution. Project sharding applies after project selection. Focused and sharded project runs require a separate receipt so they cannot replace the complete registry receipt. Lesson runs can write a separate receipt; the existing `--lesson` behavior without `--receipt` merges unchanged previous results and discards stale ones. Running lessons also refreshes the selected lessons' execution artifacts. Listing requires no JAX installation; execution uses the relevant scientific dependencies.

The former all-lesson browser simulation is now a representative reader interaction test plus a lightweight check of every built lesson's equations and controls. Starter syntax checks share one Python process. Registry-based completeness assertions replace historical lesson/task counts, and duplicate lifecycle/sharding assertions have been removed while their stronger domain checks remain.
