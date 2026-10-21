import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { normalizeEpubMathML } from '../src/lib/mathml-export.js';
import { renderMath } from '../src/lib/math.js';

test('normalizes KaTeX max/min subscripts and preserves msup tokens under the two-child contract', () => {
  const source = '<math><msub><mi>σ</mi><mi>max</mi><mo>⁡</mo></msub><msup><mi>x</mi><mn>2</mn><mi>y</mi></msup></math>';
  const output = normalizeEpubMathML(source);
  assert.match(output, /<msub><mi>σ<\/mi><mi>max<\/mi><\/msub>/);
  assert.match(output, /<msup><mi>x<\/mi><mrow><mn>2<\/mn><mi>y<\/mi><\/mrow><\/msup>/);
  assert.doesNotMatch(output, /<msub>[^]*?<mo>⁡<\/mo>/);
});

test('groups additional script and fraction operands without discarding their token sequence', () => {
  const source = '<math><msub><mi>a</mi><mi>b</mi><mi>c</mi></msub><msup><mi>x</mi><mn>2</mn><mi>y</mi></msup><mfrac><mi>n</mi><mi>d</mi><mi>e</mi></mfrac></math>';
  const output = normalizeEpubMathML(source);
  assert.match(output, /<msub><mi>a<\/mi><mrow><mi>b<\/mi><mi>c<\/mi><\/mrow><\/msub>/);
  assert.match(output, /<msup><mi>x<\/mi><mrow><mn>2<\/mn><mi>y<\/mi><\/mrow><\/msup>/);
  assert.match(output, /<mfrac><mi>n<\/mi><mrow><mi>d<\/mi><mi>e<\/mi><\/mrow><\/mfrac>/);
});

test('flattens a nested operator token while preserving its operator text and attributes', () => {
  const output = normalizeEpubMathML('<mo lspace="0.22em" rspace="0.22em"><mrow><mi mathvariant="normal">m</mi><mi mathvariant="normal">o</mi><mi mathvariant="normal">d</mi></mrow></mo>');
  assert.equal(output, '<mo lspace="0.22em" rspace="0.22em">mod</mo>');
  assert.doesNotMatch(output, /<mo[^>]*>\s*</);
});

test('normalizes rendered KaTeX while retaining its original TeX annotation', () => {
  const rendered = normalizeEpubMathML(renderMath('\\kappa=\\sigma_{\\max}/\\sigma_{\\min}', false, 'mathml'));
  assert.match(rendered, /<msub><mi>σ<\/mi><mi>max<\/mi><\/msub>/);
  assert.match(rendered, /<msub><mi>σ<\/mi><mi>min<\/mi><\/msub>/);
  assert.match(rendered, /<annotation encoding="application\/x-tex">\\kappa=\\sigma_\{\\max\}\/\\sigma_\{\\min\}<\/annotation>/);
});

test('normalization leaves already valid MathML and allowed glyph children unchanged', () => {
  const valid = '<math><mfrac><mn>1</mn><mn>2</mn></mfrac><mo><mglyph alt="x" /></mo></math>';
  assert.equal(normalizeEpubMathML(valid), valid);
});

test('offline book chapters render authored learner contracts before the separate solutions appendix', () => {
  const python = String.raw`
import ast, html, pathlib
source = pathlib.Path('scripts/build-distribution.py').read_text()
tree = ast.parse(source)
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'learner_starter')
namespace = {
  'esc': html.escape,
  'rich': lambda value: '<math-source>' + html.escape(value) + '</math-source>',
  'code_block': lambda value: '<pre><code>' + html.escape(value) + '</code></pre>',
}
exec(compile(ast.Module(body=[helper], type_ignores=[]), 'build-distribution.py', 'exec'), namespace)
contract = {
  'inputs': [r'\(x\) is a two-element vector'],
  'outputs': ['A scalar plus its two-element tangent'],
  'invariants': ['The tangent shape matches the primal shape'],
  'hints': ['Use the explicit tangent in the linear map'],
  'code': 'def learner_attempt(x, dx):\n    raise NotImplementedError("Implement the requested tangent check")',
  'solution': 'SOLUTION_SENTINEL',
  'expectedAnswer': 'ANSWER_SENTINEL',
  'rationale': 'RATIONALE_SENTINEL',
}
rendered = namespace['learner_starter'](contract, 'Exercise')
for expected in ('Exercise · learner-only starter', 'Inputs', 'Outputs', 'Invariants to check', 'Hints', 'Your TODO scaffold', 'NotImplementedError'):
  assert expected in rendered, expected
assert '<math-source>' in rendered, 'contract fields should pass through the book math renderer'
for secret in ('SOLUTION_SENTINEL', 'ANSWER_SENTINEL', 'RATIONALE_SENTINEL'):
  assert secret not in rendered, f'learner section leaked {secret}'
pending = namespace['learner_starter']({'inputs': ['x']}, 'Practice')
assert 'Learner starter pending' in pending

chapter = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'chapter')
calls = [node for node in ast.walk(chapter) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'learner_starter']
calls.sort(key=lambda node: node.lineno)
assert len(calls) == 2, 'main exercise and practice must each render their own authored contract'
solution_appends = [node.lineno for node in ast.walk(chapter) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'solutions' and node.func.attr == 'append']
assert solution_appends and calls[0].lineno < solution_appends[0], 'exercise starter precedes its solution appendix entry'
assert any(call.lineno < min(line for line in solution_appends if line > call.lineno) for call in calls[1:]), 'practice starter precedes its reference solution'
`;
  const result = spawnSync('python3', ['-c', python], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.stdout);
});
