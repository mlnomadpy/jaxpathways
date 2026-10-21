import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { assessmentContent } from '../src/lib/server/assessment-content.js';

test('Markdown figures open their resolved full-size asset with accessible escaped names', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'assessment-figure-'));
  const sourcePath = path.join(directory, 'guide.md');
  writeFileSync(sourcePath, '# Figure guide\n\n![Error <curve> & "residuals"](../figure.svg)\n');
  const resolvedHref = '/assets/figure.svg?size="full"&mode=<all>';
  const resolved = [];

  try {
    const content = assessmentContent(sourcePath, (href) => {
      resolved.push(href);
      return resolvedHref;
    });
    const escapedHref = '/assets/figure.svg?size=&quot;full&quot;&amp;mode=&lt;all&gt;';
    const escapedAlt = 'Error &lt;curve&gt; &amp; &quot;residuals&quot;';
    const accessibleName = `Open full-size figure: ${escapedAlt}`;

    assert.deepEqual(resolved, ['../figure.svg']);
    assert.equal(
      content.body,
      `<p><a href="${escapedHref}" aria-label="${accessibleName}" title="${accessibleName}"><img src="${escapedHref}" alt="${escapedAlt}" loading="lazy" /></a></p>\n`,
    );
    assert.match(content.body, /<p><a [\s\S]*><img [\s\S]*<\/a><\/p>/);
    assert(!content.body.includes('onerror='));
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('embedded Markdown figures stay bare and preserve their escaped alt text', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'assessment-embedded-figure-'));
  const sourcePath = path.join(directory, 'guide.md');
  writeFileSync(sourcePath, '# Embedded figure guide\n\n![Embedded <plot> & data](../plot.svg)\n');
  const resolvedHref = 'data:image/svg+xml;base64,PHN2Zz4=';
  const resolved = [];

  try {
    const content = assessmentContent(sourcePath, (href) => {
      resolved.push(href);
      return resolvedHref;
    });

    assert.deepEqual(resolved, ['../plot.svg']);
    assert.equal(
      content.body,
      '<p><img src="data:image/svg+xml;base64,PHN2Zz4=" alt="Embedded &lt;plot&gt; &amp; data" loading="lazy" /></p>\n',
    );
    assert(!content.body.includes('<a '));
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
