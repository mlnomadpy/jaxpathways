import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import postcss from 'postcss';

const styles = path.resolve('src/styles');
function filesIn(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? filesIn(file) : [file];
  });
}

test('shared CSS composition has no dependency on page feature styles', () => {
  const base = postcss.parse(fs.readFileSync(path.join(styles, 'base.css'), 'utf8'));
  base.walkAtRules('import', (rule) => {
    const target = rule.params.replace(/['"]/g, '');
    assert.ok(target.startsWith('./shared/'), `Page feature leaked into shared CSS: ${target}`);
    assert.ok(fs.existsSync(path.resolve(styles, target)), `Missing stylesheet: ${target}`);
  });
});

test('page templates do not import the lesson reader to obtain shared primitives', () => {
  for (const file of filesIn('src/pages').filter((name) => name.endsWith('.astro'))) {
    assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /import\s+['"][^'"]*features\/reader\.css/);
  }
});

test('CSS imports resolve and feature rules cannot restyle the document globally', () => {
  for (const file of filesIn(styles).filter((name) => name.endsWith('.css'))) {
    const tree = postcss.parse(fs.readFileSync(file, 'utf8'), { from: file });
    tree.walkAtRules('import', (rule) => {
      const target = rule.params.replace(/['"]/g, '');
      assert.ok(fs.existsSync(path.resolve(path.dirname(file), target)), `${file}: ${target}`);
    });
    if (!file.includes(`${path.sep}features${path.sep}`)) continue;
    tree.walkRules((rule) => {
      for (const selector of rule.selectors) {
        assert.doesNotMatch(
          selector,
          /^(?:body|header|nav|input|select|textarea|summary|pre)(?:\s|$|:)/,
          `${file}: global selector ${selector}`,
        );
      }
    });
  }
});
