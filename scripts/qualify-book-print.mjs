import { chromium } from '@playwright/test';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { platform, release } from 'node:os';
import { resolve, join } from 'node:path';

// Print actual rendered chapters without replacing their markup or styles. The
// selection deliberately starts each chapter on a page, as the full book does.
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
};
const url = option('--url', 'http://127.0.0.1:4173/downloads/jax-foundations.html');
const output = resolve(option('--output', 'tmp/pdfs/book-print-baseline'));
const ids = ['welcome-01', 'optimization-09', 'transformers-04', 'tpu-06'];
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const sourceResponse = await page.request.get(url);
  if (!sourceResponse.ok()) throw new Error(`Book request failed: ${sourceResponse.status()}`);
  const source = await sourceResponse.body();
  const response = await page.goto(url, { waitUntil: 'load', timeout: 90000 });
  if (!response?.ok()) throw new Error(`Book request failed: ${response?.status()}`);
  const selection = await page.evaluate((wanted) => {
    const articles = Array.from(document.querySelectorAll('article'));
    const selected = wanted.map((id) => {
      const matches = articles.filter((article) =>
        article.querySelector('.chapter-kicker')?.textContent.trim().endsWith(`· ${id}`),
      );
      if (matches.length !== 1) throw new Error(`Expected one chapter for ${id}`);
      return matches[0];
    });
    const metadata = selected.map((article, index) => ({
      lessonId: wanted[index],
      articleId: article.id,
      title: article.querySelector('h1').textContent.trim(),
      figures: article.querySelectorAll('figure').length,
      codePanels: article.querySelectorAll('.code-panel').length,
      mathExpressions: article.querySelectorAll('.katex').length,
      learnerContracts: article.querySelectorAll('.learner-starter-contract').length,
      printedAppendix: article.querySelector('.print-solutions')?.id || null,
    }));
    // Preserve main/phase ancestors so contextual selectors still match. Only
    // unselected siblings are removed; each selected article stays untouched.
    const prune = (node) => {
      if (selected.includes(node)) return;
      for (const child of Array.from(node.childNodes)) {
        if (selected.some((article) => child.contains(article))) prune(child);
        else child.remove();
      }
    };
    prune(document.body);
    // Offscreen lazy images otherwise need not load before a print request.
    for (const image of document.images) image.loading = 'eager';
    return metadata;
  }, ids);
  await page.waitForFunction(
    () => document.fonts.status === 'loaded'
      && Array.from(document.images).every((image) => image.complete && image.naturalWidth > 0),
    null,
    { timeout: 30000 },
  );
  await page.emulateMedia({ media: 'print', colorScheme: 'light' });
  await page.waitForFunction(() =>
    Array.from(document.querySelectorAll('details:not(.solution-reveal)')).every(
      (details) => details.open,
    ),
  );
  const printState = await page.evaluate(() => ({
    inlineSolutionsHidden: Array.from(document.querySelectorAll('.solution-reveal')).every(
      (element) => getComputedStyle(element).display === 'none',
    ),
    appendicesVisible: Array.from(document.querySelectorAll('.print-solutions')).every(
      (element) => getComputedStyle(element).display === 'block',
    ),
  }));
  if (!printState.inlineSolutionsHidden || !printState.appendicesVisible
    || selection.some((chapter) => !chapter.printedAppendix)) {
    throw new Error('Print mode must hide interactive reveals and show every answer appendix');
  }
  const records = [];
  for (const format of ['A4', 'Letter']) {
    const filename = `book-samples-${format.toLowerCase()}.pdf`;
    const pdf = await page.pdf({
      path: join(output, filename),
      format,
      scale: 1,
      printBackground: true,
      displayHeaderFooter: false,
      preferCSSPageSize: false,
      tagged: true,
      outline: true,
    });
    records.push({
      format,
      filename,
      sha256: createHash('sha256').update(pdf).digest('hex'),
      bytes: pdf.length,
    });
  }
  const record = {
    schemaVersion: 1,
    executedAt: new Date().toISOString(),
    browser: await browser.version(),
    operatingSystem: { platform: platform(), release: release() },
    renderer: 'Isolated headless Chromium, Playwright page.pdf',
    sourceUrl: url,
    sourceHtmlSha256: createHash('sha256').update(source).digest('hex'),
    selection,
    printState,
    scriptSha256: createHash('sha256').update(await readFile(new URL(import.meta.url))).digest('hex'),
    settings: {
      scale: 1,
      printBackground: true,
      orientation: 'portrait',
      margins: 'Book @page CSS: 18mm',
      headerFooter: false,
      images: 'Selected chapter images loaded eagerly before printing',
    },
    artifacts: records,
    scope: 'Four selected actual chapters with original styles and ancestor contexts. Not a full-book pagination, native EPUB, screen-reader or human learner acceptance.',
    visualReview: 'Pending raster inspection; PDF creation alone is not acceptance.',
  };
  await writeFile(join(output, 'print-receipt.json'), `${JSON.stringify(record, null, 2)}\n`);
  console.log(`Created ${records.length} print samples in ${output}`);
} finally {
  await browser.close();
}
