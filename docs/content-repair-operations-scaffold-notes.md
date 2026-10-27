# Operations lesson scaffold repair notes

Scope: operations-01 through operations-05 teaching structure, plus the behavioral rollback portion of operations-04. Operations-06 through operations-08 remain owned by the operations content repair.

## What changed

The first guided-reasoning section of each lesson now presents a short excerpt, labels reviewed support versus the learner-authored decision, and walks through a concrete state or event trace. The staged build now contains only three short functions tied to that lesson's concept. Total staged code fell from the previous large repeated listings to 10, 6, 7, 17 and 18 lines for operations-01 through -05, while the complete runnable source remains in each lesson.

The examples focus on process outcome classification, progress versus durable checkpoint interpretation, exact replay and state comparison, selection-dependent model behavior, and capacity arithmetic. Exercises use changed inputs or cadence and ask learners to interpret emitted events or observed outputs.

Operations-04 publishes two valid model payloads with different parameters. Its consumer reads the selected content-addressed artifact on each request. The CPU run predicts 3.0 for A and 3.4 for B at x=1; publication alone leaves A's prediction active, activation selects B, a rejected candidate leaves pointer bytes and behavior unchanged, and rollback restores A and prediction 3.0. The scope is local, single-writer and unauthenticated.

## Source and packaging

No shared support module was introduced. Every lesson keeps its complete worker and support code inline, so each Python script, exercise and downloaded reference remains standalone. The compact staged snippets are executable function definitions also present in the full source. The generated reference script executes the staged definitions before the full source, so they intentionally avoid work until called and are safe before the source imports and setup run.

The companion helper parity check confirmed the canonical full source and staged excerpts appear in generated lesson script text, notebook code cells and book Markdown. I did not run the repository-wide distribution generator because it writes generated outputs for the whole course while other repairs are in progress. A targeted archive regeneration is still needed before publishing the new downloadable files. No helper packaging change is required.

## Verification

Commands run from `/private/tmp/jaxpathways-content-repairs-review`:

- The exact parity command was `node --input-type=module -e 'import {readFileSync,writeFileSync,globSync} from "node:fs"; import {lessonScript,lessonMarkdown,lessonCells} from "./scripts/lesson-artifacts.mjs"; for(let i=1;i<=5;i++){const l=JSON.parse(readFileSync(globSync(`phases/16-operations/0${i}-*/lesson.json`)[0],"utf8")); const c=l.content; const script=lessonScript({title:l.id,content:c}); const book=lessonMarkdown({title:l.id,minutes:l.minutes,content:c},{number:"16",title:"Operations"}); const cells=lessonCells({title:l.id,minutes:l.minutes,content:c}); if(!script.includes(c.code)||!book.includes(c.sections[0].body)||!cells.some(x=>x[0]==="code"&&x[1]===c.code)) throw Error(`parity ${i}`); for(const step of c.buildSteps){if(!c.code.includes(step.code)||!script.includes(step.code)||!book.includes(step.code)||!cells.some(x=>x[0]==="code"&&x[1]===step.code)) throw Error(`excerpt parity ${i}`);} writeFileSync(`/tmp/ops0${i}-lesson-script.py`,script); console.log(`operations-0${i}: script/notebook/book/excerpt parity passed`);}'`. It checks canonical code and staged excerpt identity in generated script text, notebook code cells and book Markdown, then writes fresh generated scripts only under `/tmp`.
- The exact CPU command was `for i in 1 2 3 4 5; do /private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/ops0${i}-lesson-script.py || exit $?; done`. All five generated scripts exited successfully in the clean CPU environment. Operations-01 printed completed, failed and timed-out outcomes; operations-02 printed healthy and step-5/checkpoint-4 incident summaries; operations-03 confirmed restored step 2 and matching resumed state hashes; operations-04 printed consumer predictions `3.0`, `3.4`, `3.0`, unchanged rejected activation, and successful rollback; operations-05 printed measured durations and the modeled capacity report.
- The complete canonical `content.code` source for operations-01 through -05 was also run directly in temporary directories with the same clean CPU interpreter before the final helper refactor. The final generated-script execution above supersedes it and additionally exercises the staged snippets and appended exercise/practice code.

After the full five-script pass, operations-04 had one final checkpoint-checksum refinement. Its final targeted rerun also passed: `node --input-type=module -e 'import {readFileSync,writeFileSync,globSync} from "node:fs"; import {lessonScript} from "./scripts/lesson-artifacts.mjs"; const l=JSON.parse(readFileSync(globSync("phases/16-operations/04-*/lesson.json")[0],"utf8")); writeFileSync("/tmp/ops04-lesson-script.py",lessonScript({title:l.id,content:l.content}));'` followed by `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/ops04-lesson-script.py`.

No paid or cloud quality checks were used.
