import figures from '../data/concept-figures.json' with { type: 'json' };
// Original conceptual illustrations. These explain contracts; they do not run JAX.
function courseConceptFigure(id) {
  const f = figures[id];
  if (!f) return '';
  const esc = (s) =>
    s.replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  return `<figure class="concept-mechanism"><figcaption><strong>${esc(f.title)}</strong></figcaption><ol>${f.steps.map(([label, value, note]) => `<li><span>${esc(label)}</span><code>${esc(value)}</code><p>${esc(note)}</p></li>`).join('')}</ol><p>${esc(f.note)}</p><p class="concept-scope">Conceptual illustration. Verify the computation in the lesson’s Python experiments.</p></figure>`;
}

export { courseConceptFigure };
