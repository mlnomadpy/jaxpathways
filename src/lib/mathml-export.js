/**
 * Normalize KaTeX MathML for EPUBCheck's MathML content model.
 *
 * KaTeX can emit three direct children in a script element when an operator
 * such as `\max` contributes its invisible function-application operator, and
 * a multi-letter operator such as `\bmod` can be nested inside `<mo>`. EPUB's
 * XHTML MathML schema requires fixed-arity script/fraction elements and token
 * content inside `<mo>`. This lightweight XML fragment parser preserves the
 * original token text and TeX annotation while correcting those structures.
 */
/**
 * @typedef {{raw?: string, tag: string, open: string, close?: string, selfClosing: boolean, children: MathNode[]}} MathNode
 * @param {string} source
 * @returns {string}
 */
export function normalizeEpubMathML(source) {
  /** @type {MathNode} */
  const root = { tag: '', open: '', selfClosing: false, children: [] };
  /** @type {MathNode[]} */
  const stack = [root];
  const tokens =
    String(source).match(
      /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>|<[^>]+>|[^<]+/g,
    ) || [];

  for (const token of tokens) {
    if (
      !token.startsWith('<') ||
      token.startsWith('<!--') ||
      token.startsWith('<![CDATA[') ||
      token.startsWith('<?')
    ) {
      stack[stack.length - 1].children.push(rawNode(token));
      continue;
    }
    if (token.startsWith('</')) {
      if (stack.length > 1) {
        const node = stack.pop();
        if (!node) continue;
        node.close = token;
      }
      continue;
    }
    const match = /^<([\w:.-]+)(?:\s[^<>]*?)?\s*(\/?)>$/.exec(token);
    if (!match) {
      stack[stack.length - 1].children.push(rawNode(token));
      continue;
    }
    const node = { tag: match[1], open: token, selfClosing: Boolean(match[2]), children: [] };
    stack[stack.length - 1].children.push(node);
    if (!node.selfClosing) stack.push(node);
  }

  /** @param {string} raw @returns {MathNode} */
  function rawNode(raw) {
    return { raw, tag: '', open: '', selfClosing: false, children: [] };
  }
  /** @param {MathNode} node @returns {boolean} */
  const isElement = (node) => Boolean(node.tag);
  /** @param {MathNode} node @returns {string} */
  const textContent = (node) => node.raw ?? (node.children || []).map(textContent).join('');
  /** @param {MathNode} node @returns {string} */
  const serialize = (node) => {
    if (node.raw !== undefined) return node.raw;
    return (
      node.open +
      (node.children || []).map(serialize).join('') +
      (node.selfClosing ? '' : node.close || `</${node.tag}>`)
    );
  };
  /** @param {string} tag @param {MathNode[]} children @returns {MathNode} */
  const element = (tag, children) => ({
    tag,
    open: `<${tag}>`,
    children,
    close: `</${tag}>`,
    selfClosing: false,
  });

  /** @param {MathNode} node */
  function normalize(node) {
    if (!isElement(node)) return;
    node.children.forEach(normalize);

    if (
      node.tag === 'mo' &&
      node.children.some(
        /** @param {MathNode} child */
        (child) => isElement(child) && !['mglyph', 'malignmark'].includes(child.tag),
      )
    ) {
      // MathML operator tokens contain text (or mglyph/malignmark), not an mrow.
      // Flatten token text while retaining the <mo>'s spacing/fence attributes.
      const tokenText = node.children.map(textContent).join('');
      node.children = [rawNode(tokenText)];
    }

    if (['msub', 'msup', 'mfrac'].includes(node.tag)) {
      const elementIndexes = node.children
        .map(
          /** @param {MathNode} child @param {number} index */
          (child, index) => (isElement(child) ? index : -1),
        )
        .filter((index) => index >= 0);
      if (elementIndexes.length > 2) {
        const firstScriptIndex = elementIndexes[1];
        const rest = node.children.slice(firstScriptIndex);
        const elementRest = rest.filter(isElement);
        // KaTeX's invisible function marker after a \max/\min subscript is
        // not part of the script glyph; original TeX remains in <annotation>.
        const last = elementRest.at(-1);
        if (
          node.tag === 'msub' &&
          elementRest[0]?.tag === 'mi' &&
          ['max', 'min'].includes(textContent(elementRest[0])) &&
          last?.tag === 'mo' &&
          textContent(last) === '⁡'
        ) {
          const lastIndex = rest.lastIndexOf(last);
          rest.splice(lastIndex, 1);
        }
        const remainingElements = rest.filter(isElement);
        if (remainingElements.length === 1) {
          node.children = [...node.children.slice(0, firstScriptIndex), ...rest];
        } else {
          node.children = [...node.children.slice(0, firstScriptIndex), element('mrow', rest)];
        }
      }
    }
  }

  root.children.forEach(normalize);
  return root.children.map(serialize).join('');
}
