import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import {
  buildStarterScaffold,
  renderExerciseCodeTeacher,
  renderLearnerStarterContract,
} from '../src/lib/code-teaching.js';
import {
  learnerNotebookCells,
  learnerScript,
  learnerArtifactPaths,
  learnerStarterInventory,
  readyLearnerArtifactPaths,
  lessonCells,
  lessonMarkdown,
  lessonScript,
} from '../scripts/lesson-artifacts.mjs';
import { learnerDownloadLink, learnerStarterCoverage } from '../src/lib/lesson-template.js';

const contract = {
  inputs: [
    'logits has shape (B, T, V), with B=2, T=4, and V=7 for this held-out check.',
    'valid is a boolean mask of shape (B, T) with both valid and padded positions.',
    'key is an explicit JAX PRNG key; repeat with a different seed as well.',
  ],
  outputs: ['Return one sampled token ID per batch and time position, with shape (B, T).'],
  invariants: [
    'Masked positions must not be sampled.',
    'Every returned token ID is in [0, V).',
    'The same key and inputs reproduce the same result.',
  ],
  hints: ['Broadcast the validity mask over the vocabulary axis before sampling.'],
  code: [
    'import jax',
    'import jax.numpy as jnp',
    '',
    'def sample_valid_tokens(logits, valid, key):',
    '    # TODO: exclude invalid positions from the candidate logits',
    '    masked_logits = ...',
    '    # TODO: sample along the vocabulary axis using the supplied key',
    '    return ...',
  ].join('\n'),
};

function makeLesson(solution = 'answer_value = 91827\n# ANSWER_RATIONALE_SENTINEL') {
  return {
    id: 'held-out-mask-sampling',
    title: 'Held-out masked sampling',
    minutes: 25,
    source: 'https://docs.jax.dev/',
    evidence: 'Save the prediction, modified code, and observed checks.',
    content: {
      idea: 'A mask removes invalid candidates before a random choice.',
      code: 'REFERENCE_EXAMPLE_SENTINEL = 44819',
      output: 'RECORDED_EXPECTED_OUTPUT_SENTINEL',
      exercise: 'Sample one valid token for each batch and time position.',
      solution,
      learnerStarter: structuredClone(contract),
      experiments: [{
        title: 'Worked experiment that must stay out of learner downloads',
        prediction: 'EXPERIMENT_PREDICTION_SENTINEL',
        code: 'EXPERIMENT_CODE_SENTINEL = 41',
        output: 'EXPERIMENT_OUTPUT_SENTINEL',
        explanation: 'EXPERIMENT_RATIONALE_SENTINEL',
      }],
      practice: [{
        title: 'Change the mask and key',
        difficulty: 'Transfer',
        prompt: 'Use a different mask layout and a new seed.',
        hint: 'Keep the mask shape aligned with the first two logits axes.',
        solution: 'PRACTICE_REFERENCE_SENTINEL = 781',
        explanation: 'PRACTICE_RATIONALE_SENTINEL',
        learnerStarter: structuredClone(contract),
      }],
      question: 'CHECKPOINT_QUESTION_SENTINEL',
      options: ['CHECKPOINT_OPTION_SENTINEL'],
      answer: 0,
      explanation: 'CHECKPOINT_RATIONALE_SENTINEL',
      optionFeedback: ['CHECKPOINT_FEEDBACK_SENTINEL'],
      references: [],
    },
  };
}

const phase = { number: '7', title: 'Sampling' };

test('explicit contract renders independent inputs, outputs, invariants, hints, and TODO code', () => {
  const html = renderLearnerStarterContract(contract);
  assert.match(html, /Inputs/);
  assert.match(html, /Outputs/);
  assert.match(html, /Invariants to check/);
  assert.match(html, /Hints/);
  assert.match(html, /TODO: exclude invalid positions/);
  assert.match(html, /B=2, T=4, and V=7/);
  assert.doesNotMatch(html, /91827|ANSWER_RATIONALE_SENTINEL/);
  assert.equal(buildStarterScaffold('x = 91827\n# answer rationale'), '');
});

test('browser writing guide requires the learner contract and never derives a scaffold from the solution', () => {
  const absent = renderExerciseCodeTeacher('x = 91827\n# ANSWER_RATIONALE_SENTINEL');
  assert.match(absent, /Learner starter pending/);
  assert.doesNotMatch(absent, /91827|ANSWER_RATIONALE_SENTINEL/);

  const present = renderExerciseCodeTeacher('x = 91827\n# ANSWER_RATIONALE_SENTINEL', 'Write code', contract);
  assert.match(present, /same key and inputs reproduce the same result/);
  assert.match(present, /sample_valid_tokens/);
  assert.doesNotMatch(present, /91827|ANSWER_RATIONALE_SENTINEL/);
});

test('held-out learner downloads ignore reference mutations and omit worked answers and expected outputs', () => {
  const first = makeLesson();
  const changedReference = makeLesson('new_answer = 123456\n# TOTALLY_DIFFERENT_RATIONALE');
  assert.deepEqual(learnerNotebookCells(first), learnerNotebookCells(changedReference));
  assert.equal(learnerScript(first), learnerScript(changedReference));

  const notebook = JSON.stringify(learnerNotebookCells(first));
  const script = learnerScript(first);
  for (const forbidden of [
    '91827', 'ANSWER_RATIONALE_SENTINEL', '123456', 'TOTALLY_DIFFERENT_RATIONALE',
    'REFERENCE_EXAMPLE_SENTINEL', 'RECORDED_EXPECTED_OUTPUT_SENTINEL',
    'EXPERIMENT_CODE_SENTINEL', 'EXPERIMENT_OUTPUT_SENTINEL',
    'PRACTICE_REFERENCE_SENTINEL', 'PRACTICE_RATIONALE_SENTINEL',
    'CHECKPOINT_QUESTION_SENTINEL', 'CHECKPOINT_OPTION_SENTINEL',
    'CHECKPOINT_RATIONALE_SENTINEL', 'CHECKPOINT_FEEDBACK_SENTINEL',
  ]) {
    assert.equal(notebook.includes(forbidden), false, `learner notebook leaked ${forbidden}`);
    assert.equal(script.includes(forbidden), false, `learner script leaked ${forbidden}`);
  }
  assert.match(notebook, /shape \(B, T, V\).*B=2, T=4, and V=7/);
  assert.match(notebook, /boolean mask of shape \(B, T\)/);
  assert.match(notebook, /different seed/);
  assert.match(script, /TODO: sample along the vocabulary axis/);
});

test('learner notebook code cells contain editable authored TODO code and task-level pending status', () => {
  const lesson = makeLesson();
  const cells = learnerNotebookCells(lesson);
  const code = cells.filter(([type]) => type === 'code').map(([, source]) => source);
  assert.equal(code.length, 2);
  assert.match(code[0], /^import jax\nimport jax\.numpy as jnp/);
  assert.doesNotMatch(code[0], /^# import jax/m);
  assert.match(JSON.stringify(cells), /Authored learner starters: 2\/2 tasks\. Pending: 0/);

  delete lesson.content.practice[0].learnerStarter;
  assert.match(JSON.stringify(learnerNotebookCells(lesson)), /Learner starter pending/);
  assert.match(JSON.stringify(learnerNotebookCells(lesson)), /Authored learner starters: 1\/2 tasks\. Pending: 1/);
});

test('download labels report accurate full and partial learner-contract coverage', () => {
  const lesson = makeLesson();
  assert.deepEqual(learnerStarterCoverage(lesson.content), { ready: 2, total: 2 });
  assert.match(learnerDownloadLink({ learnerNotebook: 'starter.ipynb' }, lesson.content, 'notebook'), /learner-only notebook · 2 task starters/);

  delete lesson.content.practice[0].learnerStarter;
  assert.deepEqual(learnerStarterCoverage(lesson.content), { ready: 1, total: 2 });
  const partial = learnerDownloadLink({ learnerNotebook: 'starter.ipynb' }, lesson.content, 'notebook');
  assert.match(partial, /1 of 2 task starters are authored/);
  assert.doesNotMatch(partial, /href=/);

  delete lesson.content.learnerStarter;
  const unavailable = learnerDownloadLink({ learnerNotebook: 'starter.ipynb' }, lesson.content, 'notebook');
  assert.match(unavailable, /Learner-only notebook pending/);
  assert.doesNotMatch(unavailable, /href=/);
});

test('canonical starter inventory reports each complete and missing exercise or practice contract', () => {
  const lesson = makeLesson();
  lesson.status = 'authored';
  const complete = learnerStarterInventory([lesson]);
  assert.equal(complete.completeLessons, 1);
  assert.equal(complete.incompleteLessons, 0);
  assert.equal(complete.readyContracts, 2);
  assert.deepEqual(complete.lessons[0].missing, []);

  delete lesson.content.learnerStarter;
  const incomplete = learnerStarterInventory([lesson]);
  assert.equal(incomplete.completeLessons, 0);
  assert.equal(incomplete.incompleteLessons, 1);
  assert.equal(incomplete.readyContracts, 1);
  assert.deepEqual(incomplete.lessons[0].missing, [{ scope: 'exercise' }]);
});

test('distributed, science, kernels, and internals have complete authored learner-only contracts', () => {
  const course = JSON.parse(readFileSync(new URL('../curriculum/course.json', import.meta.url), 'utf8'));
  const selected = new Set(['distributed', 'science', 'kernels', 'internals']);
  const lessons = course.phases.filter((phase) => selected.has(phase.id)).flatMap((phase) =>
    phase.lessons.map((lesson) => ({
      ...lesson,
      content: JSON.parse(readFileSync(new URL(`../${lesson.path}/lesson.json`, import.meta.url), 'utf8')).content,
    })),
  );
  const inventory = learnerStarterInventory(lessons);
  assert.equal(lessons.length, 16);
  assert.equal(inventory.totalContracts, 46);
  assert.equal(inventory.readyContracts, 46);
  assert.equal(inventory.completeLessons, 16);
  assert.equal(inventory.incompleteLessons, 0);
  assert.deepEqual(inventory.lessons.flatMap((lesson) => lesson.missing), []);

  for (const lesson of lessons) {
    assert.deepEqual(readyLearnerArtifactPaths(lesson), learnerArtifactPaths(lesson));
    const contracts = [lesson.content.learnerStarter, ...lesson.content.practice.map((practice) => practice.learnerStarter)];
    for (const contract of contracts) {
      for (const field of ['inputs', 'outputs', 'invariants', 'hints']) {
        assert.ok(Array.isArray(contract[field]) && contract[field].length > 0, `${lesson.id} has no ${field}`);
      }
      assert.ok(contract.code.trim(), `${lesson.id} has an empty scaffold`);
      const syntax = spawnSync('python3', ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'], {
        input: contract.code,
        encoding: 'utf8',
      });
      assert.equal(syntax.status, 0, `${lesson.id} scaffold is not valid Python: ${syntax.stderr}`);
      assert.match(contract.code, /NotImplementedError\("[^"]+"\)/, `${lesson.id} scaffold needs an actionable TODO stop`);
    }
  }
});

test('learner download URL and source paths derive from the lesson registry identity', () => {
  const identity = { id: 'held-out-mask-sampling', path: 'phases/07-transformers/02-sampling' };
  assert.deepEqual(learnerArtifactPaths(identity), {
    learnerNotebook: 'exercises/held-out-mask-sampling-learner.ipynb',
    learnerScript: 'exercises/held-out-mask-sampling-learner.py',
    learnerNotebookSource: 'phases/07-transformers/02-sampling/notebooks/learner.ipynb',
    learnerScriptSource: 'phases/07-transformers/02-sampling/code/learner.py',
  });
  const lesson = makeLesson();
  Object.assign(lesson, identity);
  assert.deepEqual(readyLearnerArtifactPaths(lesson), learnerArtifactPaths(lesson));
  delete lesson.content.practice[0].learnerStarter;
  assert.deepEqual(readyLearnerArtifactPaths(lesson), {});
});

test('held-out shape, mask layout, and PRNG seed flow from the contract rather than a fixed reference fixture', () => {
  const lesson = makeLesson();
  const changedInputs = [
    'logits has changed shape (B, T, V) = (3, 2, 5).',
    'valid is [[true, false], [false, true], [true, true]] with shape (3, 2).',
    'Use explicit PRNG seed 991 for this held-out run.',
  ];
  lesson.content.learnerStarter.inputs = changedInputs;
  lesson.content.practice[0].learnerStarter.inputs = changedInputs;
  const notebook = JSON.stringify(learnerNotebookCells(lesson));
  const script = learnerScript(lesson);
  for (const heldOut of ['(3, 2, 5)', '[[true, false], [false, true], [true, true]]', 'seed 991']) {
    assert.ok(notebook.includes(heldOut), `learner notebook omitted changed fixture ${heldOut}`);
    assert.ok(script.includes(heldOut), `learner script omitted changed fixture ${heldOut}`);
  }
  assert.doesNotMatch(notebook, /B=2, T=4, and V=7/);
  assert.doesNotMatch(script, /B=2, T=4, and V=7/);
});

test('reference artifacts keep their deliberate solution reveal and execution content', () => {
  const lesson = makeLesson();
  assert.match(lessonScript(lesson), /answer_value = 91827/);
  assert.match(lessonScript(lesson), /REFERENCE_EXAMPLE_SENTINEL/);
  assert.ok(lessonCells(lesson).some(([, content]) => content.includes('answer_value = 91827')));
  const markdown = lessonMarkdown(lesson, phase);
  assert.match(markdown, /Reference solution/);
  assert.match(markdown, /ANSWER_RATIONALE_SENTINEL/);
  assert.match(markdown, /RECORDED_EXPECTED_OUTPUT_SENTINEL/);
});

test('missing contracts are labeled pending instead of copying reference answers', () => {
  const lesson = makeLesson();
  delete lesson.content.learnerStarter;
  delete lesson.content.practice[0].learnerStarter;
  const learnerOutput = `${JSON.stringify(learnerNotebookCells(lesson))}\n${learnerScript(lesson)}`;
  assert.match(learnerOutput, /Learner starter pending/);
  assert.doesNotMatch(learnerOutput, /91827|ANSWER_RATIONALE_SENTINEL|PRACTICE_REFERENCE_SENTINEL/);
});
