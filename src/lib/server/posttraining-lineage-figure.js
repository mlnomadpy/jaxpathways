/** Group the one recorded bridge figure without changing generic Markdown image rendering. */
export function groupTransformerLineageFigure(html, { required = true } = {}) {
  const marker =
    /<h3 id="([^"]+)">Recorded v5 result<\/h3>\s*<p>((?:<a href="[^"]+"[^>]*>)?<img src="([^"]+)" alt="([^"]*)"[^>]*\/>?(?:<\/a>)?)<\/p>\s*<p><strong>Figure 1\.<\/strong>\s*([\s\S]*?)<\/p>/g;
  const matches = [...html.matchAll(marker)];
  if (matches.length === 0 && !required) return html;
  if (matches.length !== 1) {
    throw new Error(
      `Expected one recorded transformer-lineage figure group, found ${matches.length}`,
    );
  }
  const [match, titleId, image, , , caption] = matches[0];
  const figure = `<figure class="print-visual-intro" aria-labelledby="${titleId}"><h3 id="${titleId}">Recorded v5 result</h3><div class="figure-scroll" role="region" aria-label="Recorded v5 result" tabindex="0">${image}</div><figcaption><strong>Figure 1.</strong> ${caption}</figcaption></figure>`;
  return html.replace(match, figure);
}
