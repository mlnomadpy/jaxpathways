import { readFileSync } from 'node:fs';
import { Marked } from 'marked';
import { escapeHtml } from '../html.js';
import { renderMath } from '../math.js';

/** Render repository-owned Markdown; raw HTML is rejected to keep the content contract. */
export function assessmentContent(sourcePath, resolveLink = (href) => href) {
  const source = readFileSync(sourcePath, 'utf8');
  let heading = 0;
  const sections = [];
  const markdown = new Marked({
    extensions: [
      {
        name: 'courseDisplayMath',
        level: 'block',
        start(source) {
          return source.indexOf('\\[');
        },
        tokenizer(source) {
          const match = /^\\\[([\s\S]*?)\\\](?:\n|$)/.exec(source);
          if (match) return { type: 'courseDisplayMath', raw: match[0], text: match[1].trim() };
        },
        renderer(token) {
          return renderMath(token.text, true);
        },
      },
      {
        name: 'courseMath',
        level: 'inline',
        start(source) {
          return source.indexOf('\\(');
        },
        tokenizer(source) {
          const match = /^\\\(([\s\S]*?)\\\)/.exec(source);
          if (match) return { type: 'courseMath', raw: match[0], text: match[1] };
        },
        renderer(token) {
          return renderMath(token.text);
        },
      },
    ],
    renderer: {
      link({ href, tokens }) {
        return `<a href="${escapeHtml(resolveLink(href))}">${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, text }) {
        return `<img src="${escapeHtml(resolveLink(href))}" alt="${escapeHtml(text)}" loading="lazy" />`;
      },
      html() {
        throw new Error('Raw HTML is not supported in assessment sources');
      },
      heading({ tokens, depth }) {
        const label = this.parser.parseInline(tokens);
        const id = `part-${heading++}`;
        if (depth === 2) sections.push({ id, label });
        return `<h${depth} id="${id}">${label}</h${depth}>`;
      },
      code({ text, lang }) {
        const language =
          lang === 'python'
            ? 'python'
            : ['sh', 'bash', 'powershell'].includes(lang)
              ? 'shell'
              : 'text';
        return `<pre data-language="${language}"><code>${escapeHtml(text)}</code></pre>`;
      },
    },
  });
  const html = markdown.parse(source, { async: false });
  const end = html.indexOf('</h1>') + 5;
  return { heading: html.slice(0, end), body: html.slice(end), sections };
}
