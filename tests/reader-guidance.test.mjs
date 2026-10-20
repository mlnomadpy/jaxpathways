import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Window } from 'happy-dom';
import { renderLesson } from '../src/lib/lesson-template.js';
import { executionScope } from '../src/lib/runtime.js';

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
  for (const lesson of lessons)
    assert.doesNotMatch(executionScope(lesson).description, /stages directly|run on physical TPU/);
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
