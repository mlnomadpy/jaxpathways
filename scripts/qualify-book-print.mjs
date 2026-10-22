import { chromium } from '@playwright/test';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { platform, release } from 'node:os';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_CHAPTER_IDS = ['welcome-01', 'optimization-09', 'transformers-04', 'tpu-06'];
const FULL_BOOK_ARTICLE_COUNT = 104;
const PDF_TEXT_MAX_BUFFER_BYTES = 32 * 1024 * 1024;
const PDF_RENDER_TIMEOUT_MS = 240_000;
/** @param {string} value */
function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

/**
 * Parse print qualification options without launching a browser.
 * @param {string[]} argv
 */
export function parseQualificationOptions(argv) {
  const values = new Map();
  let all = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--all') {
      all = true;
      continue;
    }
    if (!['--url', '--output', '--format'].includes(arg)) {
      throw new Error(`Unknown print qualification option: ${arg}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Expected a value after ${arg}`);
    values.set(arg, value);
    index += 1;
  }

  const format = (values.get('--format') || 'both').toLowerCase();
  if (!['a4', 'letter', 'both'].includes(format)) {
    throw new Error('--format must be A4, Letter, or both');
  }
  const formats = format === 'both' ? ['A4', 'Letter'] : [format === 'a4' ? 'A4' : 'Letter'];
  return {
    all,
    format,
    formats,
    url: values.get('--url') || 'http://127.0.0.1:4173/downloads/jax-foundations.html',
    output: resolve(values.get('--output') || 'tmp/pdfs/book-print-baseline'),
    chapterIds: all ? [] : [...DEFAULT_CHAPTER_IDS],
  };
}

/** @param {string} pdfPath */
export function inspectPdf(pdfPath, { expectedVisuals = 0, expectedMechanismTitles = [] } = {}) {
  const info = execFileSync('pdfinfo', [pdfPath], { encoding: 'utf8' });
  const pageCount = Number(info.match(/^Pages:\s+(\d+)$/m)?.[1]);
  if (!Number.isInteger(pageCount) || pageCount < 1) {
    throw new Error(`Could not read PDF page count from ${pdfPath}`);
  }
  let extractedText;
  try {
    extractedText = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
      encoding: 'utf8',
      maxBuffer: PDF_TEXT_MAX_BUFFER_BYTES,
    });
  } catch (error) {
    if (error.code === 'ENOBUFS') {
      throw new Error(
        `pdftotext output exceeded the ${PDF_TEXT_MAX_BUFFER_BYTES}-byte inspection limit for ${pdfPath}`,
        { cause: error },
      );
    }
    throw error;
  }
  const pages = extractedText.split('\f');
  if (pages.at(-1)?.trim() === '') pages.pop();
  if (pages.length !== pageCount) {
    throw new Error(`pdftotext found ${pages.length} pages but pdfinfo found ${pageCount}`);
  }
  const numberedPages = pages.map((page, index) => {
    const label = `Page ${index + 1}`;
    if (!page.split(/\r?\n/).some((line) => line.trim() === label)) {
      throw new Error(`Print footer ${label} is missing from page ${index + 1}`);
    }
    return index + 1;
  });
  if (expectedVisuals || expectedMechanismTitles.length) {
    const visualCues = new Map([
      ['Predict before looking:', 'visual-prediction'],
      ['Plot from a recorded CPU computation', 'visual-caption'],
      ['Calculated scenario; no device execution was performed', 'visual-caption'],
      ['Conceptual workflow diagram', 'visual-caption'],
    ]);
    const visualCuePattern = new RegExp(
      [...visualCues.keys()]
        .sort((left, right) => right.length - left.length)
        .map((cue) => cue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|'),
      'g',
    );
    let visualCount = 0;
    for (const [pageIndex, page] of pages.entries()) {
      let pending = null;
      for (const match of page.matchAll(visualCuePattern)) {
        const cue = visualCues.get(match[0]);
        if (cue === 'visual-prediction') {
          if (pending) {
            throw new Error(
              `Visual print intro is split across pages; page ${pageIndex + 1} has an unpaired ${pending}`,
            );
          }
          pending = 'visual-caption';
          visualCount += 1;
        } else if (pending !== cue) {
          throw new Error(
            `Visual print intro is split or out of order on page ${pageIndex + 1}: expected ${pending || 'a prediction'}, found ${cue}`,
          );
        } else {
          pending = null;
        }
      }
      if (pending) {
        throw new Error(
          `Visual print intro is split across pages; page ${pageIndex + 1} ends before its ${pending}`,
        );
      }
    }
    if (visualCount !== expectedVisuals) {
      throw new Error(
        `PDF visual verification found ${visualCount}/${expectedVisuals} lesson visuals`,
      );
    }
    const normalizedPages = pages.map((page) => page.replace(/\s+/g, ' ').trim());
    for (const title of expectedMechanismTitles) {
      const normalizedTitle = title.replace(/\s+/g, ' ').trim();
      const matchingPages = normalizedPages.flatMap((page, index) =>
        page.includes(normalizedTitle) ? [index] : [],
      );
      if (matchingPages.length !== 1) {
        throw new Error(
          `Mechanism print intro title appears on ${matchingPages.length} pages: ${normalizedTitle}`,
        );
      }
      const pageText = normalizedPages[matchingPages[0]];
      const titleIndex = pageText.indexOf(normalizedTitle);
      const predictionIndex = pageText.indexOf('Predict:', titleIndex + normalizedTitle.length);
      const captionIndex = pageText.indexOf(
        'Architecture and dataflow mechanism diagram.',
        predictionIndex + 'Predict:'.length,
      );
      if (predictionIndex < 0 || captionIndex < 0) {
        throw new Error(
          `Mechanism print intro is split across pages: ${normalizedTitle} must share a page with its Predict label and diagram caption`,
        );
      }
    }
  }
  return { pageCount, numberedPages, extractedTextBytes: Buffer.byteLength(extractedText) };
}

/** @param {Promise<Buffer>} rendering @param {string} filename */
async function renderPdfWithTimeout(rendering, filename) {
  let timeout;
  try {
    return await Promise.race([
      rendering,
      new Promise((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error(`PDF render exceeded ${PDF_RENDER_TIMEOUT_MS} ms: ${filename}`)),
          PDF_RENDER_TIMEOUT_MS,
        );
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

/** @param {string[]} argv */
export async function qualifyBookPrint(argv = process.argv.slice(2)) {
  const options = parseQualificationOptions(argv);
  await mkdir(options.output, { recursive: true });
  const executablePath = process.env.CHROME_BIN;
  const browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const stylesheetResponses = [];
    page.on('response', (response) => {
      if (response.request().resourceType() === 'stylesheet') {
        stylesheetResponses.push(
          response
            .body()
            .then((body) => ({ url: response.url(), sha256: sha256(body), bytes: body.length }))
            .catch((error) => ({ url: response.url(), error: String(error) })),
        );
      }
    });

    const sourceResponse = await page.request.get(options.url);
    if (!sourceResponse.ok()) throw new Error(`Book request failed: ${sourceResponse.status()}`);
    const sourceHtml = await sourceResponse.body();
    const response = await page.goto(options.url, { waitUntil: 'load', timeout: 90000 });
    if (!response?.ok()) throw new Error(`Book request failed: ${response?.status()}`);

    const selection = await page.evaluate(
      ({ all, wanted, expectedCount }) => {
        const articles = Array.from(document.querySelectorAll('article'));
        const selected = all
          ? articles
          : wanted.map((id) => {
              const matches = articles.filter((article) =>
                article.querySelector('.chapter-kicker')?.textContent.trim().endsWith(`· ${id}`),
              );
              if (matches.length !== 1) throw new Error(`Expected one chapter for ${id}`);
              return matches[0];
            });
        if (all && selected.length !== expectedCount) {
          throw new Error(
            `Expected ${expectedCount} unpruned book articles, found ${selected.length}`,
          );
        }
        const metadata = selected.map((article, index) => ({
          lessonId: all ? null : wanted[index],
          articleId: article.id,
          title: article.querySelector('h1')?.textContent.trim() || '',
          figures: article.querySelectorAll('figure').length,
          codePanels: article.querySelectorAll('.code-panel').length,
          mathExpressions: article.querySelectorAll('.katex').length,
          learnerContracts: article.querySelectorAll('.learner-starter-contract').length,
          visualIntros: article.querySelectorAll('.lesson-visual > .print-visual-intro').length,
          mechanismTitles: Array.from(
            article.querySelectorAll('.lesson-mechanism > .print-visual-intro h3'),
            (heading) => heading.textContent.trim(),
          ),
          printedAppendix: article.querySelector('.print-solutions')?.id || null,
        }));
        if (!all) {
          const prune = (node) => {
            if (selected.includes(node)) return;
            for (const child of Array.from(node.childNodes)) {
              if (selected.some((article) => child.contains(article))) prune(child);
              else child.remove();
            }
          };
          prune(document.body);
        }
        for (const image of document.images) image.loading = 'eager';
        return {
          metadata,
          selectedCount: selected.length,
          documentArticleCount: articles.length,
          visualIntroCount: metadata.reduce((sum, article) => sum + article.visualIntros, 0),
          mechanismTitles: metadata.flatMap((article) => article.mechanismTitles),
        };
      },
      { all: options.all, wanted: options.chapterIds, expectedCount: FULL_BOOK_ARTICLE_COUNT },
    );

    await page.waitForFunction(
      () =>
        document.fonts.status === 'loaded' &&
        Array.from(document.images).every((image) => image.complete && image.naturalWidth > 0),
      null,
      { timeout: 30000 },
    );
    await page.emulateMedia({ media: 'print', colorScheme: 'light' });
    await page.waitForFunction(() =>
      Array.from(document.querySelectorAll('details:not(.solution-reveal)')).every(
        (details) => details.open,
      ),
    );
    const printState = await page.evaluate(() => {
      const prediction = document.querySelector('.visual-prediction');
      const figure =
        prediction?.closest('.lesson-visual')?.querySelector('figure') ||
        prediction?.closest('figure');
      return {
        inlineSolutionsHidden: Array.from(document.querySelectorAll('.solution-reveal')).every(
          (element) => getComputedStyle(element).display === 'none',
        ),
        appendicesVisible: Array.from(document.querySelectorAll('.print-solutions')).every(
          (element) => getComputedStyle(element).display === 'block',
        ),
        predictionBreakAfter: prediction ? getComputedStyle(prediction).breakAfter : null,
        figureBreakInside: figure ? getComputedStyle(figure).breakInside : null,
      };
    });
    if (
      !printState.inlineSolutionsHidden ||
      !printState.appendicesVisible ||
      selection.metadata.some((chapter) => !chapter.printedAppendix) ||
      !printState.predictionBreakAfter?.includes('avoid') ||
      printState.figureBreakInside !== 'avoid'
    ) {
      throw new Error(
        'Print mode must keep lesson visuals together and show separate answer appendices',
      );
    }

    const printCss = await readFile(new URL('../src/styles/print.css', import.meta.url));
    const stylesheets = await Promise.all(stylesheetResponses);
    const records = [];
    for (const format of options.formats) {
      const filename = `book-${options.all ? 'all' : 'samples'}-${format.toLowerCase()}.pdf`;
      const pdfPath = join(options.output, filename);
      const pdf = await renderPdfWithTimeout(
        page.pdf({
          path: pdfPath,
          format,
          scale: 1,
          printBackground: true,
          displayHeaderFooter: false,
          preferCSSPageSize: false,
          tagged: true,
          outline: true,
        }),
        filename,
      );
      const inspected = inspectPdf(pdfPath, {
        expectedVisuals: selection.visualIntroCount,
        expectedMechanismTitles: selection.mechanismTitles,
      });
      records.push({
        format,
        filename,
        sha256: sha256(pdf),
        bytes: pdf.length,
        pageCount: inspected.pageCount,
        extractedTextBytes: inspected.extractedTextBytes,
        footerNumberedPages: inspected.numberedPages,
        verifiedVisualIntros: selection.visualIntroCount,
        verifiedMechanismIntros: selection.mechanismTitles.length,
      });
    }
    const record = {
      schemaVersion: 2,
      executedAt: new Date().toISOString(),
      browser: await browser.version(),
      operatingSystem: { platform: platform(), release: release() },
      renderer: 'Isolated headless Chromium, Playwright page.pdf',
      source: {
        url: options.url,
        html: { sha256: sha256(sourceHtml), bytes: sourceHtml.length },
        printCss: {
          path: 'src/styles/print.css',
          sha256: sha256(printCss),
          bytes: printCss.length,
        },
        loadedStylesheets: stylesheets,
        script: {
          path: 'scripts/qualify-book-print.mjs',
          sha256: sha256(await readFile(new URL(import.meta.url))),
        },
      },
      options: {
        all: options.all,
        format: options.format,
        formats: options.formats,
        chapterIds: options.chapterIds,
        output: options.output,
      },
      selection: selection.metadata,
      selectedArticleCount: selection.selectedCount,
      documentArticleCount: selection.documentArticleCount,
      printState,
      settings: {
        scale: 1,
        printBackground: true,
        orientation: 'portrait',
        margins: 'print.css: 18mm top/side and 24mm bottom with a footer margin box',
        browserHeaderFooter: false,
        pdfRenderTimeoutMs: PDF_RENDER_TIMEOUT_MS,
        extractedPdfTextMaxBufferBytes: PDF_TEXT_MAX_BUFFER_BYTES,
        images: 'Selected images loaded eagerly before printing',
      },
      artifacts: records,
      scope: options.all
        ? 'All 104 original book articles printed without pruning.'
        : 'Four selected actual chapters with original styles and ancestor contexts.',
      visualReview:
        'Automated rendering and footer checks recorded; human visual review remains separate.',
    };
    await writeFile(
      join(options.output, 'print-receipt.json'),
      `${JSON.stringify(record, null, 2)}\n`,
    );
    console.log(`Created ${records.length} print samples in ${options.output}`);
  } finally {
    await browser.close();
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  await qualifyBookPrint();
}
