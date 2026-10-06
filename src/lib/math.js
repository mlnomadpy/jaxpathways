import katex from 'katex';
import { escapeHtml } from './html.js';

/** Explicit delimiters avoid treating currency and shell dollar signs as math. */
export function renderMath(tex, displayMode = false, output = 'htmlAndMathml') {
  const rendered = katex.renderToString(String(tex), {
    displayMode,
    output,
    throwOnError: true,
    strict: 'error',
    trust: false,
    maxExpand: 1000,
    maxSize: 20,
    macros: {},
  });
  return displayMode
    ? `<span class="math-display" tabindex="0" role="group" aria-label="Equation">${rendered}</span>`
    : rendered;
}

/** Render authored math and inline code; all other source remains escaped text. */
export function inlineMath(text = '', output = 'htmlAndMathml') {
  const source = String(text);
  const tokens = /(`+)([^\n]*?)\1|\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g;
  let html = '',
    cursor = 0;
  for (const match of source.matchAll(tokens)) {
    html += escapeHtml(source.slice(cursor, match.index));
    html += match[1]
      ? `<code>${escapeHtml(match[2])}</code>`
      : renderMath(match[3] ?? match[4] ?? match[5], match[3] === undefined, output);
    cursor = match.index + match[0].length;
  }
  return html + escapeHtml(source.slice(cursor));
}

export function mathProse(text = '', output = 'htmlAndMathml') {
  return String(text)
    .split('\n\n')
    .filter(Boolean)
    .map((paragraph) => `<p>${inlineMath(paragraph, output)}</p>`)
    .join('');
}
