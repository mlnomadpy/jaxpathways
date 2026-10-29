import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

test('EPUB prose renders a plain Markdown link even without adjacent math or code', () => {
  const folder = mkdtempSync(join(tmpdir(), 'book-link-contract-'));
  try {
    mkdirSync(join(folder, 'public'));
    const plain = 'Continue to [Arrays & pure functions](https://example.com/arrays.html).';
    const withMath = 'Read [the gradient](https://example.com/gradient.html) at \\(x=2\\).';
    writeFileSync(join(folder, 'public/curriculum.json'), JSON.stringify({ body: plain, idea: withMath }));
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/render-book-math.mjs', import.meta.url))],
      { cwd: folder, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const rendered = JSON.parse(result.stdout).prose;
    assert.match(rendered[plain], /<a href="https:\/\/example\.com\/arrays\.html">Arrays &amp; pure functions<\/a>/);
    assert(!rendered[plain].includes('[Arrays'));
    assert.match(rendered[withMath], /href="https:\/\/example\.com\/gradient\.html"/);
    assert.match(rendered[withMath], /<math/);
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
});
