import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';
import { inlineMath, mathProse, renderMath } from '../src/lib/math.js';
import { deepConcepts, codePanel } from '../src/lib/lesson-content.js';

test('inline and display math include accessible MathML without interpreting currency or code', async () => {
  const window = new Window();
  try {
    window.document.body.innerHTML = mathProse(String.raw`Slope \(f'(x)=2x\), cost $5. ` + 'Code `'  + String.raw`\(x\)` + '`' + '\n\n' + String.raw`$$\frac{1}{N}\sum_i r_i^2$$`);
    assert.equal(window.document.querySelectorAll('math').length, 2);
    assert.equal(window.document.querySelectorAll('.math-display').length, 1);
    assert.match(window.document.body.textContent, /cost \$5/);
    assert.equal(window.document.querySelector('code').textContent, String.raw`\(x\)`);
    assert.equal(window.document.querySelector('.math-display').getAttribute('tabindex'), '0');
  } finally { await window.happyDOM.abort(); }
});

test('math prose escapes raw HTML, rejects malformed TeX and does not trust URL commands', () => {
  assert.match(inlineMath('<img src=x onerror=alert(1)>'), /&lt;img/);
  assert.throws(() => renderMath(String.raw`\frac{1}{`));
  assert(!/<a(?:\s|>)/.test(renderMath(String.raw`\href{javascript:alert(1)}{x}`)));
  assert(!inlineMath(String.raw`\(\text{<script>}\)`).includes('<script>'));
});

test('equations and shape sketches stay distinct, and code panels never typeset source', () => {
  const html = deepConcepts({sections:[{title:'Loss',body:'Read one residual at a time.',math:String.raw`L=\frac{1}{N}\sum_i r_i^2`,formula:'(B,D) → (B,)'}]});
  assert.match(html, /class="math-display"/);
  assert.match(html, /<pre class="lesson-equation"/);
  assert(!codePanel(String.raw`pattern = "\(x\)"`).includes('class="katex"'));
});

test('Markdown math conversion preserves Python and inline code exactly', async () => {
  const { markdownText, lessonCells, lessonScript } = await import('../scripts/lesson-artifacts.mjs');
  const source = String.raw`For \(x=2\), keep ` + '`' + String.raw`\(literal\)` + '`';
  assert.equal(markdownText(source), 'For $x=2$, keep `' + String.raw`\(literal\)` + '`');
  const code = String.raw`pattern = r"\(x\)"`;
  const lesson = {id:'math-fixture',title:'Fixture',evidence:'Explain it.',content:{idea:source,code,solution:code,exercise:source,question:source,options:[source],answer:0,explanation:source}};
  const cells = lessonCells(lesson);
  assert(cells.some(([type,text]) => type === 'markdown' && text.includes('$x=2$')));
  assert(cells.filter(([type]) => type === 'code').some(([,text])=>text === code));
  assert(lessonScript(lesson).includes(code));
});

test('regression explanation typesets inline dimensions and loss while preserving the Python expression', async () => {
  const fs = await import('node:fs');
  const lesson = JSON.parse(fs.readFileSync('phases/04-optimization/01-linear-algebra-and-loss-intuition/lesson.json','utf8'));
  const window = new Window();
  try {
    window.document.body.innerHTML = mathProse(lesson.content.idea);
    const math = [...window.document.querySelectorAll('annotation')].map(node=>node.textContent);
    for (const tex of [String.raw`X\in\mathbb{R}^{n\times d}`,String.raw`w\in\mathbb{R}^{d}`,String.raw`\hat y=Xw+b\mathbf{1}`,String.raw`(n,)`,String.raw`(n,n)`]) assert(math.includes(tex),`missing inline notation: ${tex}`);
    assert.equal(window.document.querySelector('code').textContent,'X @ w');
    assert(!window.document.body.textContent.includes(String.raw`\(`));
  } finally {await window.happyDOM.abort();}
});

test('authored lesson prose does not leave mathematical shapes or operators as plain text', async () => {
  const fs = await import('node:fs');
  const course = JSON.parse(fs.readFileSync('curriculum/course.json','utf8'));
  const proseKeys = new Set(['idea','problem','body','instruction','explanation','prediction','output','buildOutput','exercise','prompt','hint','question','options','takeaways','objectives','diagnosis','reading','connection']);
  const excluded = new Set(['code','solution','buildCode','formula','math','commands','references','setupBundle']);
  function inspect(value, key, label) {
    if(excluded.has(key))return;
    if(typeof value==='string'&&proseKeys.has(key)) {
      const plain=value.replace(/`+[^`]*`+|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$/g,' ');
      assert.doesNotMatch(plain, /[²³ᵀ∇∂Σηδα₀₁₂ᵢ]|\([nNdDBTK],\s*(?:[nNdDBTK0-9]|\))|\b\d+[xwh]\b|\bX\s*@\s*w\b/,label);
    } else if(Array.isArray(value))value.forEach((item,index)=>inspect(item,key,`${label}[${index}]`));
    else if(value&&typeof value==='object')for(const [childKey,child] of Object.entries(value))inspect(child,childKey,`${label}.${childKey}`);
  }
  for(const phase of course.phases)for(const lesson of phase.lessons)if(lesson.status==='authored'){
    const source=JSON.parse(fs.readFileSync(`${lesson.path}/lesson.json`,'utf8'));
    inspect(source.content,'content',lesson.id);
  }
});

test('inlineMath and mathProse render Markdown bold, italic, links, lists, and code in headings', async () => {
  const window = new Window();
  try {
    const html = mathProse(
      'Choose a route:\n\n' +
        '- **Route 1 (`gcloud`)**: Run *warmed* steps on [TPU course](course.html#phase-tpu).\n' +
        '- **Route 2**: Verify \\(x^2\\) and keep `2 * N^3` intact.\n\n' +
        '1. First step\n\n' +
        '2. Second step',
    );
    window.document.body.innerHTML = html;
    assert.equal(window.document.querySelectorAll('ul > li').length, 2);
    assert.equal(window.document.querySelectorAll('ol > li').length, 2);
    assert.equal(window.document.querySelector('strong code')?.textContent, 'gcloud');
    assert.equal(window.document.querySelector('em')?.textContent, 'warmed');
    assert.equal(
      window.document.querySelector('a')?.getAttribute('href'),
      'course.html#phase-tpu',
    );
    assert(
      !inlineMath('[bad](javascript:alert(1))').includes('<a'),
      'unsafe URL schemes must remain escaped text',
    );
    const sectionHtml = deepConcepts({
      sections: [{ title: '2. Authenticate `gcloud` and `.venv`', body: 'Ready.' }],
    });
    assert.match(sectionHtml, /<h2>2\. Authenticate <code>gcloud<\/code> and <code>\.venv<\/code><\/h2>/);
  } finally {
    await window.happyDOM.abort();
  }
});
