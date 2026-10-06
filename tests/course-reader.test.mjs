import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';

test('downloaded reader preserves every highlighted source and carries its own math styles/fonts', async () => {
  const html = fs.readFileSync('dist/downloads/jax-foundations.html', 'utf8');
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  const lessons = course.phases.flatMap(p => p.lessons).filter(l => l.status === 'authored');
  const window = new Window({ settings: { enableJavaScriptEvaluation: false, disableCSSFileLoading: true } });
  try {
    window.document.write(html);
    const articles = [...window.document.querySelectorAll('main > article')];
    assert.equal(articles.length, lessons.length);
    for (const [index, lesson] of lessons.entries()) {
      const c = lesson.content;
      const expected = [
        ...(c.sections || []).flatMap(s => (s.commands || []).map(command => command.code)),
        ...(c.buildSteps || []).map(step => step.code),
        ...(c.buildCode ? [c.buildCode] : []), c.code,
        ...(c.experiments || []).map(e => e.code), c.solution,
        ...(c.practice || []).map(p => p.solution),
      ];
      const panels = [...articles[index].querySelectorAll('.code-panel')];
      assert.deepEqual(panels.map(p => p.querySelector('pre code').textContent), expected, lesson.id);
      for (const panel of panels) assert.equal(panel.querySelector('pre').tabIndex, 0);
    }
    assert(window.document.querySelectorAll('.syntax-keyword').length > 100);
    assert(window.document.querySelectorAll('.katex math').length > 100);
    assert.equal(window.document.querySelector('.katex-error'), null);
    const css = [...window.document.querySelectorAll('style')].map(s => s.textContent).join('\n');
    const fonts = [...css.matchAll(/url\(data:font\/woff2;base64,([^)]+)\)/g)];
    assert.equal(fonts.length, 20);
    for (const [, font] of fonts) assert.equal(Buffer.from(font, 'base64').subarray(0, 4).toString(), 'wOF2');
    assert.doesNotMatch(css, /url\(["']?fonts\//);
    assert.equal(window.document.querySelector('link[rel="stylesheet"]'), null);
    assert.match(css, /@media print/);
  } finally { await window.happyDOM.abort(); }
});

test('reader copy preserves code literally and offers selection when clipboard access fails', async () => {
  const html = fs.readFileSync('dist/downloads/jax-foundations.html', 'utf8');
  const script = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)]
    .map(match => match[1]).find(source => source.includes('readerReady'));
  assert(script);
  const window = new Window();
  try {
    window.document.body.innerHTML = '<div class="code-panel"><button class="copy-snippet">Copy</button><pre><code></code></pre><p class="copy-feedback" role="status"></p></div>';
    const code = 'x = "<script>literal</script>"\nprint(x)\n';
    window.document.querySelector('code').textContent = code;
    let copied;
    window.navigator.clipboard.writeText = async value => { copied = value; };
    window.eval(script);
    window.document.querySelector('button').click();
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(copied, code);
    assert.equal(window.document.querySelector('[role="status"]').textContent, 'Copied code.');
    window.navigator.clipboard.writeText = async () => { throw Error('blocked'); };
    window.document.querySelector('button').click();
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.match(window.document.querySelector('[role="status"]').textContent, /Code selected/);
    assert.equal(window.getSelection().toString(), code);
  } finally { await window.happyDOM.abort(); }
});
