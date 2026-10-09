import katex from 'katex';
import { escapeHtml } from './html.js';

/**
 * Explicit delimiters avoid treating currency and shell dollar signs as math.
 * @param {string} tex
 * @param {boolean} [displayMode]
 * @param {'html' | 'mathml' | 'htmlAndMathml'} [output]
 */
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

const SAFE_HREF =
  /^(?:https?:\/\/|mailto:|#|\/|\.\.?\/|[a-zA-Z0-9._-]+(?:\/[a-zA-Z0-9._-]+)*\.[a-zA-Z0-9]+(?:[?#][^\s)]*)?$)/;

/**
 * Render authored math, inline code, bold, italic, and safe links; all other source remains escaped text.
 * @param {string} [text]
 * @param {'html' | 'mathml' | 'htmlAndMathml'} [output]
 * @returns {string}
 */
export function inlineMath(text = '', output = 'htmlAndMathml') {
  const source = String(text);
  const tokens =
    /(`+)([^\n]*?)\1|\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$|\[((?:`+[^\n]*?`+|\\[\(\[][\s\S]*?\\[\)\]]|[^\n\]])+)\]\(([^)\s]+)\)|\*\*((?:`+[^\n]*?`+|\\[\(\[][\s\S]*?\\[\)\]]|\$\$[\s\S]*?\$\$|[^\n*])+?)\*\*|(?<![\\*\w])\*([^\s*](?:`+[^\n]*?`+|\\[\(\[][\s\S]*?\\[\)\]]|[^\n*])*?[^\s*]|[^\s*])\*(?![*\w])/g;
  let html = '',
    cursor = 0;
  for (const match of source.matchAll(tokens)) {
    const index = match.index ?? 0;
    html += escapeHtml(source.slice(cursor, index));
    if (match[1]) {
      html += `<code>${escapeHtml(match[2])}</code>`;
    } else if (match[3] !== undefined || match[4] !== undefined || match[5] !== undefined) {
      html += renderMath(match[3] ?? match[4] ?? match[5], match[3] === undefined, output);
    } else if (match[6] !== undefined && match[7] !== undefined) {
      const href = match[7].trim();
      if (SAFE_HREF.test(href) && !/^\s*(?:javascript|data|vbscript):/i.test(href)) {
        html += `<a href="${escapeHtml(href)}">${inlineMath(match[6], output)}</a>`;
      } else {
        html += escapeHtml(match[0]);
      }
    } else if (match[8] !== undefined) {
      html += `<strong>${inlineMath(match[8], output)}</strong>`;
    } else if (match[9] !== undefined) {
      html += `<em>${inlineMath(match[9], output)}</em>`;
    }
    cursor = index + match[0].length;
  }
  return html + escapeHtml(source.slice(cursor));
}

/**
 * Render block-level Markdown (paragraphs, unordered/ordered lists, headings, blockquotes, code fences) with inline math & Markdown.
 * @param {string} [text]
 * @param {'html' | 'mathml' | 'htmlAndMathml'} [output]
 * @returns {string}
 */
export function mathProse(text = '', output = 'htmlAndMathml') {
  const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
  /** @type {string[]} */
  const blocks = [];
  /** @type {string[]} */
  let paragraph = [];
  /** @type {string[]} */
  let quote = [];
  /** @type {'ul' | 'ol' | null} */
  let listType = null;
  /** @type {string[]} */
  let listItems = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${inlineMath(paragraph.join('\n'), output)}</p>`);
    paragraph = [];
  };
  const flushQuote = () => {
    if (!quote.length) return;
    blocks.push(`<blockquote>${mathProse(quote.join('\n'), output)}</blockquote>`);
    quote = [];
  };
  const flushList = () => {
    if (!listType || !listItems.length) return;
    blocks.push(
      `<${listType}>${listItems.map((item) => `<li>${inlineMath(item, output)}</li>`).join('')}</${listType}>`,
    );
    listType = null;
    listItems = [];
  };
  const flushAll = () => {
    flushParagraph();
    flushQuote();
    flushList();
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fenceMatch = /^```([a-zA-Z0-9_-]*)\s*$/.exec(line);
    if (fenceMatch) {
      flushAll();
      const lang = fenceMatch[1] || 'text';
      const codeLines = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      blocks.push(
        `<pre data-language="${escapeHtml(lang)}"><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`,
      );
      continue;
    }
    if (/^\s*$/.test(line)) {
      flushParagraph();
      flushQuote();
      if (listType) {
        let nextIndex = i + 1;
        while (nextIndex < lines.length && /^\s*$/.test(lines[nextIndex])) nextIndex++;
        const nextLine = lines[nextIndex] || '';
        const continuesUl = listType === 'ul' && /^\s*[-*]\s+.+$/.test(nextLine);
        const continuesOl = listType === 'ol' && /^\s*\d+[.)]\s+.+$/.test(nextLine);
        if (!continuesUl && !continuesOl) flushList();
      }
      continue;
    }
    const headingMatch = /^(#{2,6})\s+(.+?)\s*$/.exec(line);
    if (headingMatch) {
      flushAll();
      const depth = headingMatch[1].length;
      blocks.push(`<h${depth}>${inlineMath(headingMatch[2], output)}</h${depth}>`);
      continue;
    }
    const quoteMatch = /^\s*>\s?(.*)$/.exec(line);
    if (quoteMatch) {
      flushParagraph();
      flushList();
      quote.push(quoteMatch[1]);
      continue;
    }
    const ulMatch = /^\s*[-*]\s+(.+)$/.exec(line);
    if (ulMatch) {
      flushParagraph();
      flushQuote();
      if (listType && listType !== 'ul') flushList();
      listType = 'ul';
      listItems.push(ulMatch[1]);
      continue;
    }
    const olMatch = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (olMatch) {
      flushParagraph();
      flushQuote();
      if (listType && listType !== 'ol') flushList();
      listType = 'ol';
      listItems.push(olMatch[1]);
      continue;
    }
    if (listType && /^\s{2,}\S/.test(line) && listItems.length) {
      listItems[listItems.length - 1] += '\n' + line.trim();
      continue;
    }
    flushList();
    flushQuote();
    paragraph.push(line);
  }
  flushAll();
  return blocks.join('');
}
