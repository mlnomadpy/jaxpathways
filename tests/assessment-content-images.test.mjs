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

test('local PNG and SVG figures reserve their intrinsic aspect ratio before loading', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'assessment-local-figure-'));
  const sourcePath = path.join(directory, 'guide.md');
  const svgPath = path.join(directory, 'figure.svg');
  const ratioSvgPath = path.join(directory, 'ratio.svg');
  const physicalSvgPath = path.join(directory, 'physical.svg');
  const pngPath = path.join(directory, 'figure.png');
  writeFileSync(
    svgPath,
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400"><rect width="800" height="400"/></svg>',
  );
  writeFileSync(
    ratioSvgPath,
    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 100"><rect width="200" height="100"/></svg>',
  );
  writeFileSync(
    physicalSvgPath,
    '<svg xmlns="http://www.w3.org/2000/svg" width="2in" height="1in" viewBox="0 0 200 100"><rect width="200" height="100"/></svg>',
  );
  const pngHeader = Buffer.alloc(24);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(pngHeader, 0);
  pngHeader.writeUInt32BE(13, 8);
  pngHeader.write('IHDR', 12, 'ascii');
  pngHeader.writeUInt32BE(640, 16);
  pngHeader.writeUInt32BE(320, 20);
  writeFileSync(pngPath, pngHeader);
  writeFileSync(
    sourcePath,
    '# Figure guide\n\n![SVG plot](figure.svg#panel)\n\n![Ratio plot](ratio.svg)\n\n![Physical plot](physical.svg)\n\n![PNG image](figure.png?revision=1)\n',
  );

  try {
    const content = assessmentContent(sourcePath, (href) => `/assets/${href}`);
    assert.match(
      content.body,
      /src="\/assets\/figure\.svg#panel" alt="SVG plot" width="800" height="400" loading="lazy"/,
    );
    assert.match(
      content.body,
      /src="\/assets\/ratio\.svg" alt="Ratio plot" width="200" height="200" loading="lazy"/,
    );
    assert.match(
      content.body,
      /src="\/assets\/physical\.svg" alt="Physical plot" width="192" height="96" loading="lazy"/,
    );
    assert.match(
      content.body,
      /src="\/assets\/figure\.png\?revision=1" alt="PNG image" width="640" height="320" loading="lazy"/,
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
