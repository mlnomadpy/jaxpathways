import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Window } from 'happy-dom';
import { renderLesson } from '../src/lib/lesson-template.js';
import { executionScope } from '../src/lib/runtime.js';
import { lessonCells, lessonMarkdown } from '../scripts/lesson-artifacts.mjs';

const course = JSON.parse(readFileSync('public/curriculum.json', 'utf8'));
function view(id, experiments) {
  const phase = course.phases.find((p) => p.lessons.some((l) => l.id === id));
  const lesson = structuredClone(phase.lessons.find((l) => l.id === id));
  if (experiments !== undefined) lesson.content.experiments = experiments;
  const window = new Window({ settings: { enableJavaScriptEvaluation: false } });
  window.document.body.innerHTML = renderLesson({
    phase,
    lesson,
    prerequisiteLinks: '',
    allLessons: [lesson],
    index: 0,
    activePath: 'all',
  });
  return { window, document: window.document, lesson };
}

test('beginner build and practice stay in the extracted workspace', async () => {
  const { window, document } = view('welcome-01');
  try {
    const guides = [...document.querySelectorAll('.stage-code-guide')]
      .map((node) => node.textContent)
      .join('\n');
    assert.match(guides, /first_experiment\.py/);
    assert.doesNotMatch(guides, /phases\/|exercises\/|main\.py/);
    assert.match(
      document.querySelector('.cumulative-build').textContent,
      /existing|already included/,
    );
  } finally {
    await window.happyDOM.abort();
  }
});

test('stage counts and banner numbers match lessons with four and five stages', async () => {
  for (const experiments of [
    [],
    [
      {
        title: 'Boundary',
        prediction: 'Predict',
        code: 'print(1)',
        output: '1',
        explanation: 'Observe',
      },
    ],
  ]) {
    const { window, document } = view('welcome-01', experiments);
    try {
      const stages = [...document.querySelectorAll('[data-lesson-stage]')];
      const pills = [...document.querySelectorAll('.stage-step-num')].map((node) =>
        Number(node.textContent),
      );
      assert.deepEqual(
        pills,
        stages.map((_, i) => i + 1),
      );
      stages.forEach((stage, i) =>
        assert.match(
          stage.querySelector('.stage-kicker').textContent,
          new RegExp(`Stage ${i + 1} of ${stages.length}`),
        ),
      );
    } finally {
      await window.happyDOM.abort();
    }
  }
});

test('runtime classification separates CPU execution from target qualification guidance', () => {
  const lessons = course.phases.flatMap((p) => p.lessons);
  assert.equal(executionScope(lessons.find((l) => l.id === 'welcome-01')).reference, 'cpu-only');
  assert.equal(
    executionScope(lessons.find((l) => l.id === 'distributed-02')).reference,
    'logical-cpu',
  );
  assert.match(executionScope(lessons.find((l) => l.id === 'tpu-05')).target, /execution matrix/);
  assert.doesNotMatch(executionScope(lessons.find((l) => l.id === 'tpu-05')).target, /implemented/);
  for (const id of ['tpu-04', 'tpu-06', 'kernels-02']) {
    const target = executionScope(lessons.find((lesson) => lesson.id === id)).target;
    assert.match(target, /v5e1 launch, resume and precision checks/);
    assert.match(target, /four single-device Pallas cases/);
    assert.match(target, /four-device v5e collective fixture/);
    assert.match(target, /recorded source, environment, cases and hardware/);
    assert.doesNotMatch(target, /qualification is still pending/);
  }
  for (const lesson of lessons)
    assert.doesNotMatch(executionScope(lesson).description, /stages directly|run on physical TPU/);
});

test('practice guidance links bounded hardware receipts without claiming general qualification', async () => {
  const { window, document } = view('tpu-04');
  try {
    const panel = document.querySelector('[data-practice-panel="tpu"]');
    assert.match(panel.textContent, /v5e1 launch, resume, and precision checks/);
    assert.match(panel.textContent, /four single-device Pallas cases/);
    assert.match(panel.textContent, /four-device v5e collective fixture/);
    assert.match(panel.textContent, /recorded source, environment, cases, and hardware/);
    assert.doesNotMatch(panel.textContent, /remain unqualified on physical TPU hardware/);
    assert.equal(
      document.querySelector('#practice-drawer a[href="tpu-performance.html"]')?.textContent,
      'TPU performance and hardware evidence guide',
    );
  } finally {
    await window.happyDOM.abort();
  }
});

test('TPU staging guidance keeps CPU examples and target qualification separate', () => {
  const manifest = JSON.parse(readFileSync('curriculum/course.json', 'utf8'));
  const phase = manifest.phases.find((item) => item.id === 'tpu-workflows');
  const staged = phase.lessons.find((item) => item.id === 'tpu-05');
  const authored = JSON.parse(readFileSync(`${staged.path}/lesson.json`, 'utf8'));
  assert.deepEqual(
    staged.content,
    authored.content,
    'stale embedded content must not contradict the authored execution matrix',
  );
  for (const prose of [
    phase.learningAdvice,
    staged.objective,
    staged.exercise,
    authored.content.problem,
    ...authored.content.takeaways,
  ]) {
    assert.doesNotMatch(prose, /(?:run|stage|upload) any (?:lesson|course|public\/exercises)/i);
  }
  assert.match(authored.content.idea, /distributed-02.*CPU-only/);
  assert.match(
    authored.content.idea,
    /backend selection, dependencies, topology assumptions, and data ownership/,
  );
  assert.match(authored.content.sections[2].body, /discovery check only/);
});

test('onboarding uses bounded, reviewed TPU targets instead of arbitrary lesson uploads', () => {
  const manifest = JSON.parse(readFileSync('curriculum/course.json', 'utf8'));
  const welcome = manifest.phases.find((phase) =>
    phase.lessons.some((lesson) => lesson.id === 'welcome-03'),
  );
  const lesson = welcome.lessons.find((item) => item.id === 'welcome-03');
  const authored = JSON.parse(readFileSync(`${lesson.path}/lesson.json`, 'utf8'));
  assert.deepEqual(lesson.content, authored.content);
  const advice = manifest.phases.map((phase) => phase.learningAdvice).join('\n');
  assert.doesNotMatch(advice, /run (?:subsequent lessons|every course|any accelerator phase)/i);
  const commands = authored.content.sections
    .flatMap((section) => section.commands || [])
    .map((command) => command.code)
    .join('\n');
  assert.match(commands, /queued-resources create/);
  assert.match(commands, /--provisioning-model=FLEX_START --max-run-duration=1h/);
  assert.match(commands, /--valid-until-duration=30m/);
  assert.match(commands, /uv venv --python 3\.12\.13 --managed-python/);
  assert.match(commands, /requirements-qualification\.txt/);
  assert.match(commands, /launch\.py --platform tpu/);
  assert.match(commands, /precision_profile\.py --platform tpu/);
  assert.match(commands, /queued-resources delete/);
  assert.doesNotMatch(commands, /JAX_PLATFORMS=tpu .*exercises\//);
  assert.match(authored.content.sections[4].body, /CPU reference/);
  assert.match(
    authored.content.sections[5].body,
    /source, dependencies, backend assertion, topology, and input ownership/,
  );
});

test('downloaded notebooks and Markdown preserve the same runtime boundaries as the reader', () => {
  for (const phase of course.phases) {
    for (const lesson of phase.lessons.filter((item) => item.status === 'authored')) {
      const scope = executionScope(lesson);
      const intro = lessonCells(lesson)[0][1];
      assert(intro.includes(scope.description), lesson.id);
      assert(intro.includes(scope.target), lesson.id);
      assert.doesNotMatch(intro, /Runnable locally on CPU or on a Cloud TPU VM/);
      const markdown = lessonMarkdown(lesson, phase);
      assert(markdown.includes(scope.description), lesson.id);
      assert(markdown.includes(scope.target), lesson.id);
    }
  }
});

test('the offline book links the archive referenced by its beginner setup walkthrough', async () => {
  const window = new Window({ settings: { enableJavaScriptEvaluation: false } });
  try {
    window.document.write(readFileSync('dist/downloads/jax-foundations.html', 'utf8'));
    const chapter = window.document.querySelector('#chapter-0');
    const link = chapter.querySelector('a[href$="downloads/jax-start-here.zip"]');
    assert(link, 'the setup chapter must not refer to a missing download button');
    assert.match(link.href, /^https:\/\/www\.tahabouhsine\.com\/jaxpathways\//);
    assert.match(chapter.textContent, /first_experiment\.py/);
  } finally {
    await window.happyDOM.abort();
  }
});
