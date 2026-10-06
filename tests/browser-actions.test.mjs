import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';
import { bindClipboard, populateSelect } from '../src/scripts/browser-actions.js';
import { readStoredJson } from '../src/lib/storage.js';
import { loadCourse, courseState } from '../src/lib/course-state.js';

test('optional stored JSON handles missing, corrupt and unavailable records', () => {
  assert.deepEqual(readStoredJson('key', [], { getItem: () => null }), []);
  assert.equal(readStoredJson('key', null, { getItem: () => '{' }), null);
  assert.equal(
    readStoredJson('key', null, {
      getItem: () => {
        throw Error('blocked');
      },
    }),
    null,
  );
  assert.deepEqual(readStoredJson('key', null, { getItem: () => '{"version":1}' }), { version: 1 });
});

test('clipboard binding is repeatable, preserves literal text and reports failure', async () => {
  const window = new Window();
  globalThis.document = window.document;
  Object.defineProperty(globalThis, 'navigator', { value: window.navigator, configurable: true });
  document.body.innerHTML =
    '<button>Copy</button><code>&lt;script&gt;$HOME&lt;/script&gt;</code><p role="status"></p>';
  const copied = [];
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText: async (text) => copied.push(text) },
    configurable: true,
  });
  const config = {
    selector: 'button',
    source: () => document.querySelector('code'),
    feedback: () => document.querySelector('p'),
    success: 'Copied',
    failure: 'Copy manually',
  };
  bindClipboard(config);
  bindClipboard(config);
  await document.querySelector('button').onclick();
  assert.deepEqual(copied, ['<script>$HOME</script>']);
  assert.equal(document.querySelector('p').dataset.state, 'success');
  navigator.clipboard.writeText = async () => {
    throw Error('blocked');
  };
  await document.querySelector('button').onclick();
  assert.equal(document.querySelector('p').textContent, 'Copy manually');
  assert.equal(document.querySelector('p').dataset.state, 'error');
  document.querySelector('code').remove();
  await document.querySelector('button').onclick();
  await window.happyDOM.abort();
});

test('select population treats labels as text and replaces previous options', async () => {
  const window = new Window();
  globalThis.document = window.document;
  const select = document.createElement('select');
  populateSelect(select, [{ id: 'a', title: '<img src=x>' }], 'Choose');
  assert.equal(select.options.length, 2);
  assert.equal(select.options[1].textContent, '<img src=x>');
  assert.equal(select.querySelector('img'), null);
  populateSelect(select, [{ id: 'b', title: 'Second' }]);
  assert.equal(select.options.length, 1);
  assert.equal(select.value, 'b');
  await window.happyDOM.abort();
});

test('course loading rejects malformed reading records and clears stale in-memory progress', async () => {
  const window = new Window({ url: 'https://example.org/' });
  globalThis.document = window.document;
  globalThis.localStorage = window.localStorage;
  localStorage.setItem(
    'jaxpathways-reading-v1',
    JSON.stringify({ version: 1, lessons: [], lastLessonId: 'missing' }),
  );
  courseState.lessonProgress = {
    version: 1,
    lessons: { stale: { checkpointPassed: true, evidenceSaved: true } },
  };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      phases: [{ id: 'arrays', lessons: [{ id: 'arrays-01', status: 'authored' }] }],
      pathways: [{ id: 'foundations', phaseIds: ['arrays'] }],
    }),
  });
  await loadCourse();
  assert.deepEqual(courseState.readingState, { version: 1, lessons: {} });
  assert.deepEqual(courseState.lessonProgress, { version: 1, lessons: {} });
  await window.happyDOM.abort();
});
