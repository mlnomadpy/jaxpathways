import figures from '../data/concept-figures.json' with { type: 'json' };
/** @type {Record<string, { title: string; steps: string[][]; note: string }>} */
const figureMap = figures;
/** @type {Record<string, string>} */
const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
// Original conceptual illustrations. These explain contracts; they do not run JAX.
/** @param {string} id */
function courseConceptFigure(id) {
  const f = figureMap[id];
  if (!f) return '';
  /** @param {string} s */
  const esc = (s) => s.replace(/[&<>"']/g, (c) => entities[c] || c);
  return `<figure class="concept-mechanism"><figcaption><strong>${esc(f.title)}</strong></figcaption><ol>${f.steps.map(([label, value, note]) => `<li><span>${esc(label)}</span><code>${esc(value)}</code><p>${esc(note)}</p></li>`).join('')}</ol><p>${esc(f.note)}</p><p class="concept-scope">Conceptual illustration. Verify the computation in the lesson’s Python experiments.</p></figure>`;
}

export { courseConceptFigure };
