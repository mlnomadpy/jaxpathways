import { mathProse, inlineMath, renderMath } from './math.js';
import { CourseCode } from './code.js';
import { escapeHtml } from './html.js';
import { lessonMechanism } from './lesson-visuals.js';

function highlightCode(code, label) {
  return CourseCode.highlight(
    code,
    /shell|terminal|powershell|command|^sh$/i.test(label) ? 'shell' : 'python',
  );
}

const prose = mathProse;

function codePanel(code, label = 'Python') {
  return `<div class="code-panel"><div class="code-panel-heading"><span>${escapeHtml(label)}</span><button type="button" class="copy-snippet" aria-label="Copy ${escapeHtml(label)} code">Copy code</button></div><pre data-language="${/shell|terminal|powershell|command|^sh$/i.test(label) ? 'shell' : 'python'}"><code>${highlightCode(code, label)}</code></pre><p class="copy-feedback" role="status"></p></div>`;
}

function deepConcepts(c, lesson) {
  return (c.sections || [])
    .map(
      (s) =>
        `<section class="concept-section"><h2>${escapeHtml(s.title)}</h2>${prose(s.body)}${s.id === 'guided-reasoning' && lesson ? lessonMechanism(lesson) : ''}${s.check ? `<aside class="reasoning-check"><h3>Pause and reason</h3>${prose(s.check.prompt)}<details><summary>Compare your reasoning</summary>${prose(s.check.answer)}</details></aside>` : ''}${(s.commands || []).map((cmd) => `<div class="terminal-step"><h3>${escapeHtml(cmd.label)}</h3>${codePanel(cmd.code, cmd.shell || 'Terminal')}<p><strong>Expected:</strong></p>${prose(cmd.expected)}</div>`).join('')}${s.math ? renderMath(s.math, true) : ''}${s.formula ? `<pre class="lesson-equation" data-language="text"><code>${escapeHtml(s.formula)}</code></pre>` : ''}</section>`,
    )
    .join('');
}

function workedExperiments(c) {
  return (c.experiments || [])
    .map(
      (e) =>
        `<section class="worked-experiment"><h2>${escapeHtml(e.title)}</h2><aside class="prediction-prompt"><strong>Predict before running</strong>${prose(e.prediction)}</aside>${codePanel(e.code, e.title + ' · Python experiment')}<p><strong>Expected:</strong> ${inlineMath(e.output)}</p>${prose(e.explanation)}</section>`,
    )
    .join('');
}

function furtherPractice(c) {
  return (c.practice || [])
    .map(
      (p) =>
        `<section class="lesson-practice"><h2>${escapeHtml(p.title)}</h2><p class="edition">${escapeHtml(p.difficulty)}</p>${prose(p.prompt)}<details><summary>Show a hint</summary>${prose(p.hint)}</details><details><summary>Reveal reference solution and reasoning</summary>${codePanel(p.solution, p.title + ' · Python solution')}${prose(p.explanation)}</details></section>`,
    )
    .join('');
}

export { highlightCode, prose, codePanel, deepConcepts, workedExperiments, furtherPractice };
