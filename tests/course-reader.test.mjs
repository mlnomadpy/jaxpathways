import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';
import { inlineMath, mathProse } from '../src/lib/math.js';
import { renderLearnerStarterContract } from '../src/lib/code-teaching.js';

test('learner contract rendering escapes literal HTML once and preserves authored inline math', async () => {
  const window = new Window();
  try {
    window.document.body.innerHTML = renderLearnerStarterContract(
      {
        inputs: ['Literal <script> & x > 0.'],
        outputs: ['A scalar for \\(x < 1\\).'],
        invariants: ['Keep the condition visible as text and math.'],
        hints: ['Check `x` at the boundary.'],
        code: 'def learner_attempt(x):\n    raise NotImplementedError',
      },
      'Escaping regression',
    );
    const section = window.document.querySelector('.learner-starter-contract');
    assert(section);
    assert.equal(section.querySelectorAll('script').length, 0, 'literal tags are never elements');
    assert.equal(section.querySelector('li').textContent, 'Literal <script> & x > 0.');
    assert(section.querySelector('.katex math'), 'delimited inline math renders as MathML');
    assert.equal(section.querySelector('.katex-error'), null);
    assert.match(section.querySelector('li code').textContent, /x/);
    assert.match(section.querySelector('pre code').textContent, /NotImplementedError/);
  } finally {
    await window.happyDOM.abort();
  }
});

test('downloaded reader renders every learner contract and provides a complete print solutions appendix', async () => {
  const html = fs.readFileSync('dist/downloads/jax-foundations.html', 'utf8');
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  const lessons = course.phases.flatMap((p) => p.lessons).filter((l) => l.status === 'authored');
  const window = new Window({
    settings: { enableJavaScriptEvaluation: false, disableCSSFileLoading: true },
  });
  try {
    window.document.write(html);
    const articles = [...window.document.querySelectorAll('main article')];
    assert.equal(articles.length, lessons.length);
    let contractCount = 0;
    let renderedContractCount = 0;
    for (const [index, lesson] of lessons.entries()) {
      const c = lesson.content;
      const tasks = [
        { scope: 'exercise', contract: c.learnerStarter },
        ...(c.practice || []).map((practice, practiceIndex) => ({
          scope: `practice-${practiceIndex + 1}`,
          contract: practice.learnerStarter,
        })),
      ];
      const contracts = [...articles[index].querySelectorAll('.learner-starter-contract')];
      const pending = [...articles[index].querySelectorAll('.learner-starter-pending')];
      const validTasks = tasks.filter(
        ({ contract }) =>
          contract &&
          ['inputs', 'outputs', 'invariants', 'hints'].every(
            (key) => Array.isArray(contract[key]) && contract[key].length > 0,
          ) &&
          typeof contract.code === 'string' &&
          contract.code.trim(),
      );
      contractCount += tasks.length;
      renderedContractCount += contracts.length;
      assert.equal(contracts.length, validTasks.length, `${lesson.id}: learner contracts`);
      assert.equal(
        pending.length,
        tasks.length - validTasks.length,
        `${lesson.id}: pending contracts`,
      );
      const unmatchedContracts = [...contracts];
      for (const { contract } of validTasks) {
        const expectedText = [
          ...contract.inputs,
          ...contract.outputs,
          ...contract.invariants,
          ...contract.hints,
        ];
        const sectionIndex = unmatchedContracts.findIndex(
          (candidate) => candidate.querySelector('pre code')?.textContent === contract.code.trim(),
        );
        assert.notEqual(sectionIndex, -1, `${lesson.id}: exact authored starter code`);
        const [section] = unmatchedContracts.splice(sectionIndex, 1);
        for (const item of expectedText) {
          const expectedNode = window.document.createElement('div');
          expectedNode.innerHTML = inlineMath(item);
          const expectedContent = expectedNode.textContent.replace(/\s+/g, ' ').trim();
          assert(
            section.textContent.replace(/\s+/g, ' ').includes(expectedContent),
            `${lesson.id}: contract item ${item}`,
          );
        }
        const referenceAnswers = [
          c.solution,
          ...(c.practice || []).map((practice) => practice.solution),
          c.explanation,
          ...(c.practice || []).map((practice) => practice.explanation),
        ];
        for (const answer of referenceAnswers)
          assert(
            !section.textContent.includes(answer),
            `${lesson.id}: learner section leaked an answer`,
          );
      }
      assert.equal(
        unmatchedContracts.length,
        0,
        `${lesson.id}: every rendered contract is task-specific`,
      );
      const expected = [
        ...(c.sections || []).flatMap((s) => (s.commands || []).map((command) => command.code)),
        ...(c.buildSteps || []).map((step) => step.code),
        ...(c.buildCode ? [c.buildCode] : []),
        c.code,
        ...(c.experiments || []).map((e) => e.code),
        c.solution,
        ...(c.practice || []).map((p) => p.solution),
        ...(c.buildCode ? [c.buildCode] : []),
        c.code,
        c.solution,
        ...(c.practice || []).map((p) => p.solution),
      ];
      const panels = [...articles[index].querySelectorAll('.code-panel')];
      assert.deepEqual(
        panels.map((p) => p.querySelector('pre code').textContent),
        expected,
        lesson.id,
      );
      for (const panel of panels) assert.equal(panel.querySelector('pre').tabIndex, 0);
      assert.match(
        panels
          .find((p) => p.querySelector('pre code').textContent === c.code)
          .closest('details')
          .querySelector('summary').textContent,
        /assembled example/i,
      );
      assert.match(
        panels
          .find((p) => p.querySelector('pre code').textContent === c.solution)
          .closest('details')
          .querySelector('summary').textContent,
        /reference solution/i,
      );
      const printSolutions = articles[index].querySelector('.print-solutions');
      assert(printSolutions, `${lesson.id}: print solutions appendix exists`);
      assert.equal(printSolutions.id, `chapter-${index}-print-solutions`);
      assert.equal(
        printSolutions.querySelector('.chapter-kicker')?.textContent.trim(),
        `Chapter ${index + 1} · ${lesson.id}`,
        `${lesson.id}: printed chapter identity`,
      );
      assert.equal(
        printSolutions.querySelector('h2')?.textContent.trim(),
        `${lesson.title}: reference solutions and checkpoint reasoning`,
        `${lesson.id}: printed appendix title`,
      );
      const printLinks = [...articles[index].querySelectorAll('.print-solution-link a')];
      assert.equal(printLinks.length, 2 + (c.practice || []).length, `${lesson.id}: print links`);
      assert(
        printLinks.every((link) => link.getAttribute('href') === `#${printSolutions.id}`),
        `${lesson.id}: task links resolve to printed solutions`,
      );
      assert.deepEqual(
        [...printSolutions.querySelectorAll('.code-panel pre code')].map(
          (code) => code.textContent,
        ),
        [
          ...(c.buildCode ? [c.buildCode] : []),
          c.code,
          c.solution,
          ...(c.practice || []).map((p) => p.solution),
        ],
        `${lesson.id}: print appendix preserves numerical/example/exercise/practice code`,
      );
      assert(
        printSolutions.innerHTML.includes(
          inlineMath(`Answer: ${c.options[c.answer]}. ${c.explanation}`),
        ),
        `${lesson.id}: exact checkpoint answer and explanation`,
      );
      const checkpointReasoning = printSolutions.querySelector('.printed-checkpoint-reasoning');
      assert(
        checkpointReasoning,
        `${lesson.id}: checkpoint answer and choice reasoning share a block`,
      );
      assert.match(
        checkpointReasoning.querySelector('h3')?.textContent ?? '',
        /checkpoint answer/i,
      );
      assert.match(checkpointReasoning.querySelector('h4')?.textContent ?? '', /other choices/i);
      assert.equal(checkpointReasoning.querySelectorAll('ul > li').length, c.options.length - 1);
      for (const optionIndex of c.options.keys()) {
        if (optionIndex === c.answer) continue;
        const expectedFeedback = window.document.createElement('div');
        expectedFeedback.innerHTML = mathProse(c.optionFeedback[optionIndex]);
        assert(
          printSolutions.textContent
            .replace(/\s+/g, ' ')
            .includes(expectedFeedback.textContent.replace(/\s+/g, ' ').trim()),
          `${lesson.id}: checkpoint reasoning for option ${optionIndex + 1}`,
        );
      }
    }
    assert(lessons.length > 0, 'the printed reader has authored lessons');
    assert.equal(renderedContractCount, contractCount, 'all authored task contracts are rendered');
    assert.equal(
      window.document.querySelectorAll('main img:not([src^="data:image/"])').length,
      0,
      'book figures must travel inside the HTML file',
    );
    assert(
      window.document.querySelectorAll('main img[src^="data:image/"]').length >= lessons.length,
    );
    assert(window.document.querySelector('#appendix-project-workflow'));
    assert(window.document.querySelector('#appendix-experiments'));
    assert.match(window.document.querySelector('.book-cover').textContent, /v\d+\.\d+\.\d+/);
    assert(window.document.querySelectorAll('.syntax-keyword').length > 100);
    assert(window.document.querySelectorAll('.katex math').length > 100);
    assert.equal(window.document.querySelector('.katex-error'), null);
    const css = [...window.document.querySelectorAll('style')].map((s) => s.textContent).join('\n');
    const fonts = [...css.matchAll(/url\(data:font\/woff2;base64,([^)]+)\)/g)];
    assert.equal(fonts.length, 20);
    for (const [, font] of fonts)
      assert.equal(Buffer.from(font, 'base64').subarray(0, 4).toString(), 'wOF2');
    assert.doesNotMatch(css, /url\(["']?fonts\//);
    assert.equal(window.document.querySelector('link[rel="stylesheet"]'), null);
    assert.match(css, /@media print/);
    assert.match(css, /\.solution-reveal\s*\{\s*display:\s*none/s);
    assert.match(css, /\.print-solutions\s*\{\s*display:\s*block/s);
    assert.match(css, /details:not\(\.solution-reveal\)\s*>\s*summary/);
    assert.match(css, /\.printed-checkpoint-reasoning[\s\S]*break-inside:\s*avoid/);
    assert.match(css, /\.reasoning-check[\s\S]*break-inside:\s*avoid/);
    assert.match(css, /\.code-panel\s*>\s*\.code-panel-heading\s*\{[^}]*color:\s*#111/s);
    assert.match(css, /:is\(\.code-panel,\s*\.starter-scaffold\)\s+pre\s+code\s+:is\(/);
    assert.match(
      [...window.document.querySelectorAll('script')]
        .map((script) => script.textContent)
        .join('\n'),
      /beforeprint[\s\S]*afterprint/,
      'print mode expands and then restores the closed reasoning disclosures',
    );
  } finally {
    await window.happyDOM.abort();
  }
});

test('reader copy preserves code literally and offers selection when clipboard access fails', async () => {
  const html = fs.readFileSync('dist/downloads/jax-foundations.html', 'utf8');
  const script = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)]
    .map((match) => match[1])
    .find((source) => source.includes('readerReady'));
  assert(script);
  const window = new Window();
  try {
    window.document.body.innerHTML =
      '<div class="code-panel"><button class="copy-snippet">Copy</button><pre><code></code></pre><p class="copy-feedback" role="status"></p></div>';
    const code = 'x = "<script>literal</script>"\nprint(x)\n';
    window.document.querySelector('code').textContent = code;
    let copied;
    window.navigator.clipboard.writeText = async (value) => {
      copied = value;
    };
    window.eval(script);
    window.document.querySelector('button').click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(copied, code);
    assert.equal(window.document.querySelector('[role="status"]').textContent, 'Copied code.');
    window.navigator.clipboard.writeText = async () => {
      throw Error('blocked');
    };
    window.document.querySelector('button').click();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.match(window.document.querySelector('[role="status"]').textContent, /Code selected/);
    assert.equal(window.getSelection().toString(), code);
  } finally {
    await window.happyDOM.abort();
  }
});
