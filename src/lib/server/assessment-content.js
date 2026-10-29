import { readFileSync } from 'node:fs';
import path from 'node:path';
import { Marked } from 'marked';
import { escapeHtml } from '../html.js';
import { renderMath } from '../math.js';

function positiveDimension(value) {
  return Number.isFinite(value) && value > 0 ? Math.ceil(value) : null;
}

function svgDimensions(bytes) {
  const source = bytes.toString('utf8');
  const root = source.match(/<svg\b[^>]*>/i)?.[0];
  if (!root) return null;
  const attribute = (name) => {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = root.match(new RegExp(`\\b${escaped}\\s*=\\s*(["'])(.*?)\\1`, 'i'));
    return match?.[2];
  };
  const lengthInPixels = (text) => {
    const match = /^\s*([0-9]+(?:\.[0-9]+)?)(px|in|cm|mm|pt|pc)?\s*$/i.exec(text || '');
    if (!match) return null;
    const value = Number(match[1]);
    const scale = { px: 1, in: 96, cm: 96 / 2.54, mm: 96 / 25.4, pt: 96 / 72, pc: 16 }[
      (match[2] || 'px').toLowerCase()
    ];
    return positiveDimension(value * scale);
  };
  const width = lengthInPixels(attribute('width'));
  const height = lengthInPixels(attribute('height'));
  if (width && height) return { width, height };

  const viewBox = (attribute('viewBox') || '')
    .trim()
    .split(/[\s,]+/)
    .map(Number);
  if (viewBox.length === 4 && viewBox.every(Number.isFinite)) {
    const width = positiveDimension(viewBox[2]);
    const height = positiveDimension(viewBox[3]);
    if (width && height) return { width, height };
  }
  return null;
}

function localImageDimensions(href, sourcePath) {
  if (/^(?:data:|https?:|\/\/)/i.test(href)) return null;
  const rawPath = href.split(/[?#]/, 1)[0];
  if (!rawPath) return null;
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(rawPath);
  } catch {
    return null;
  }
  const extension = path.extname(decodedPath).toLowerCase();
  if (!['.png', '.svg'].includes(extension)) return null;
  const imagePath = path.resolve(path.dirname(sourcePath), decodedPath);
  let bytes;
  try {
    bytes = readFileSync(imagePath);
  } catch {
    return null;
  }
  if (
    extension === '.png' &&
    bytes.length >= 24 &&
    bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) &&
    bytes.toString('ascii', 12, 16) === 'IHDR'
  ) {
    const width = positiveDimension(bytes.readUInt32BE(16));
    const height = positiveDimension(bytes.readUInt32BE(20));
    return width && height ? { width, height } : null;
  }
  if (extension === '.svg') return svgDimensions(bytes);
  return null;
}

/** Render repository-owned Markdown; raw HTML is rejected to keep the content contract. */
export function assessmentContent(
  sourcePath,
  resolveLink = (href) => href,
  mathOutput = 'htmlAndMathml',
) {
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
          return renderMath(token.text, true, mathOutput);
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
          return renderMath(token.text, false, mathOutput);
        },
      },
    ],
    renderer: {
      link({ href, tokens }) {
        return `<a href="${escapeHtml(resolveLink(href))}">${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, text }) {
        const resolvedHref = resolveLink(href);
        const resolved = escapeHtml(resolvedHref);
        const alt = escapeHtml(text);
        const dimensions = localImageDimensions(href, sourcePath);
        const intrinsicSize = dimensions
          ? ` width="${dimensions.width}" height="${dimensions.height}"`
          : '';
        if (/^data:/i.test(resolvedHref)) {
          return `<img src="${resolved}" alt="${alt}"${intrinsicSize} loading="lazy" />`;
        }
        const accessibleName = escapeHtml(
          text ? `Open full-size figure: ${text}` : 'Open full-size image',
        );
        return `<a href="${resolved}" aria-label="${accessibleName}" title="${accessibleName}"><img src="${resolved}" alt="${alt}"${intrinsicSize} loading="lazy" /></a>`;
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
