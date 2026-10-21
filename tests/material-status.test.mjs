import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { Window } from 'happy-dom';
import { materialBadge, materialNotice, materialStatus } from '../src/lib/material-status.js';
import { contributionUrl } from '../src/lib/server/material-directory.js';

const registry = JSON.parse(fs.readFileSync('public/api/v1/content-status.json', 'utf8'));

test('audit rejects stale execution, failed stages and unaccepted human reviews', () => {
  execFileSync('python3', ['tests/test_material_audit.py'], { stdio: 'pipe' });
});

test('learner notices describe scope without awarding progress or review', async () => {
  const window = new Window();
  try {
    window.localStorage.setItem('jaxpathways-lessons-v1', 'saved learner work');
    window.document.body.innerHTML = materialNotice('lessons', 'welcome-01');
    const notice = window.document.querySelector('aside');
    assert.equal(notice.querySelectorAll('input,select,textarea,button').length, 0);
    assert.match(notice.textContent, /draft/i);
    assert.match(notice.textContent, /review pending/i);
    assert.match(notice.textContent, /CPU reference checked/);
    assert.match(notice.querySelector('a').href, /content-status\.html#lessons-welcome-01$/);
    assert.equal(window.localStorage.getItem('jaxpathways-lessons-v1'), 'saved learner work');
    window.document.body.innerHTML = `<details><summary>${materialBadge('phases', 'distributed')}</summary></details>`;
    assert.equal(window.document.querySelector('summary').querySelectorAll('a,button,input').length, 0);
    assert.equal(window.document.querySelector('.material-badge').dataset.maturity, 'in-development');
    assert.throws(() => materialStatus('lessons', 'unregistered-source'), /Missing/);
  } finally {
    await window.happyDOM.abort();
  }
});

test('every contribution link preserves the exact material and suggested work', () => {
  for (const item of registry.items) {
    const url = new URL(contributionUrl(item.kind, item.id));
    assert.equal(url.hostname, 'github.com');
    assert.equal(url.pathname, '/mlnomadpy/jaxpathways/issues/new');
    assert.equal(url.searchParams.get('template'), 'course-feedback.yml');
    assert(url.searchParams.get('context').includes(item.url));
    assert(url.searchParams.get('context').includes(item.contribution));
    assert(url.searchParams.get('context').includes(item.level));
    assert.equal(materialStatus(item.kind, item.id).level, item.level);
  }
});
