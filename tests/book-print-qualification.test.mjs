import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from '@playwright/test';
import postcss from 'postcss';
import { inspectPdf, parseQualificationOptions } from '../scripts/qualify-book-print.mjs';
import { lessonMechanism, lessonVisual } from '../src/lib/lesson-visuals.js';

test('print qualification defaults to four chapters and supports full-book format options', () => {
  const defaults = parseQualificationOptions([]);
  assert.equal(defaults.all, false);
  assert.deepEqual(defaults.chapterIds, [
    'welcome-01',
    'optimization-09',
    'transformers-04',
    'tpu-06',
  ]);
  assert.equal(defaults.format, 'both');
  assert.deepEqual(defaults.formats, ['A4', 'Letter']);

  const fullBookLetter = parseQualificationOptions([
    '--all',
    '--format',
    'Letter',
    '--output',
    '/tmp/book',
  ]);
  assert.equal(fullBookLetter.all, true);
  assert.deepEqual(fullBookLetter.chapterIds, []);
  assert.deepEqual(fullBookLetter.formats, ['Letter']);
  assert.throws(
    () => parseQualificationOptions(['--format', 'legal']),
    /must be A4, Letter, or both/,
  );
  assert.throws(() => parseQualificationOptions(['--all', '--format']), /Expected a value/);
});

test('print stylesheet reserves a footer and keeps complete visual intros together', async () => {
  const css = await readFile(new URL('../src/styles/print.css', import.meta.url), 'utf8');
  const root = postcss.parse(css);
  let footer = false;
  let introRule = false;
  let figureRule = false;
  let imageRule = false;

  root.walkAtRules('page', (pageRule) => {
    pageRule.walkAtRules('bottom-center', (marginBox) => {
      marginBox.walkDecls('content', (declaration) => {
        if (declaration.value.includes('counter(page)')) footer = true;
      });
    });
  });
  root.walkRules((rule) => {
    if (rule.selector === '.print-visual-intro') {
      const values = Object.fromEntries(
        rule.nodes.filter((node) => node.type === 'decl').map((node) => [node.prop, node.value]),
      );
      introRule =
        values['break-inside'] === 'avoid-page' && values['page-break-inside'] === 'avoid';
    }
    if (
      rule.selector
        .split(',')
        .map((selector) => selector.trim())
        .includes('figure:not(.lesson-mechanism):not(.print-visual-intro)')
    ) {
      const values = Object.fromEntries(
        rule.nodes.filter((node) => node.type === 'decl').map((node) => [node.prop, node.value]),
      );
      figureRule = values['break-inside'] === 'avoid';
    }
    if (rule.selector === '.print-visual-intro .figure-scroll img') {
      const values = Object.fromEntries(
        rule.nodes.filter((node) => node.type === 'decl').map((node) => [node.prop, node.value]),
      );
      imageRule = values['max-height'] === '175mm' && values['object-fit'] === 'contain';
    }
  });

  assert(footer, 'the @page bottom-center margin box includes the current page counter');
  assert(introRule, 'the title, prediction, image and caption stay in one print group');
  assert(figureRule, 'figures outside mechanism diagrams stay intact when they fit');
  assert(imageRule, 'tall print figures fit the page while preserving aspect ratio');
  assert.match(css, /margin:\s*18mm 18mm 24mm;/);
});

test('print CSS groups tall visuals and mechanism prompts and numbers every PDF page', async (context) => {
  const executable = execFileSync('which', ['pdftotext'], { encoding: 'utf8' }).trim();
  assert(executable, 'pdftotext is needed to inspect actual PDF page text');
  const css = await readFile(new URL('../src/styles/print.css', import.meta.url), 'utf8');
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}),
  });
  const directory = await mkdtemp(join(tmpdir(), 'jaxpathways-print-test-'));
  context.after(async () => {
    await browser.close();
    await rm(directory, { recursive: true, force: true });
  });

  const page = await browser.newPage();
  const tallSvg = `data:image/svg+xml;base64,${Buffer.from(
    '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="1200" viewBox="0 0 400 1200"><rect width="400" height="1200" fill="#d8e8f0"/><text x="16" y="52" font-size="24">Tall visual fixture</text></svg>',
  ).toString('base64')}`;
  const fixtureLesson = {
    id: 'print-fixture',
    contentHash: 'current',
    content: {
      visual: {
        kind: 'analytic',
        title: 'TALL VISUAL HEADING',
        prediction: 'Predict whether the tall plot rises.',
        alt: 'Tall plot fixture',
        reading: 'This short explanation follows the figure.',
        connection: 'This short connection follows the figure.',
      },
      diagram: {
        title: 'MECHANISM HEADING',
        prediction: 'What stays shared in the mechanism?',
        alt: 'Tall mechanism fixture',
        scope: 'Architecture and dataflow mechanism diagram.',
        reading: 'Mechanism reading follows its figure.',
      },
    },
    visualArtifact: {
      contentHash: 'current',
      image: tallSvg,
      mechanismImage: tallSvg,
      data: '/fixture.json',
    },
  };
  const imageResolver = (path) => path;
  await page.setContent(`<!doctype html><html><head></head><body>
    <div class="boundary-spacer"></div>
    ${lessonVisual(fixtureLesson, { offline: true, imageResolver })}
    ${lessonMechanism(fixtureLesson, { offline: true, imageResolver })}
  </body></html>`);
  await page.addStyleTag({ content: css });
  await page.addStyleTag({
    content: `.boundary-spacer { height: 210mm; }
      .lesson-visual, .lesson-mechanism { margin: 0; padding: 0; }
      .print-visual-intro h2, .print-visual-intro h3 { margin: 0; font: 12pt/1.2 Arial, sans-serif; }
      .print-visual-intro .visual-prediction { margin: 0; padding: 3mm; font: 10pt/1.2 Arial, sans-serif; }
      .lesson-visual .print-visual-intro figure, .lesson-mechanism .print-visual-intro figure { margin: 0; }
      .print-visual-intro figcaption { margin: 0; }`,
  });
  await page.emulateMedia({ media: 'print' });

  const structure = await page.evaluate(() => {
    const visualIntro = document.querySelector('.lesson-visual > .print-visual-intro');
    const mechanism = document.querySelector('.lesson-mechanism');
    const mechanismIntro = mechanism.querySelector(':scope > .print-visual-intro');
    const mechanismHeading = mechanismIntro.querySelector('h3');
    const mechanismCaption = mechanismIntro.querySelector('figcaption');
    const mechanismImage = mechanismIntro.querySelector('.figure-scroll img');
    const mechanismReading = mechanism.querySelector('[id$="-reading"]');
    return {
      visualParts: ['h2', '.visual-prediction', 'figure'].every((selector) =>
        visualIntro.querySelector(selector),
      ),
      walkthroughOutside: !visualIntro.contains(document.querySelector('.figure-walkthrough')),
      mechanismParts: ['h3', '.visual-prediction', '.figure-scroll img', 'figcaption'].every(
        (selector) => mechanismIntro.querySelector(selector),
      ),
      mechanismFigureCount: mechanism.querySelectorAll('figure').length,
      mechanismCaptionIsLastFigureChild:
        mechanismCaption.parentElement === mechanismIntro &&
        mechanismIntro.lastElementChild === mechanismCaption,
      mechanismFigureLabelResolves:
        mechanismIntro.getAttribute('aria-labelledby') === mechanismHeading.id &&
        document.getElementById(mechanismIntro.getAttribute('aria-labelledby')) ===
          mechanismHeading,
      mechanismImageAltAndReadingReference:
        mechanismImage.getAttribute('alt') === 'Tall mechanism fixture' &&
        document.getElementById(mechanismImage.getAttribute('aria-describedby')) ===
          mechanismReading,
      mechanismReadingOutside: !mechanismIntro.contains(mechanismReading),
    };
  });
  assert.deepEqual(structure, {
    visualParts: true,
    walkthroughOutside: true,
    mechanismParts: true,
    mechanismFigureCount: 1,
    mechanismCaptionIsLastFigureChild: true,
    mechanismFigureLabelResolves: true,
    mechanismImageAltAndReadingReference: true,
    mechanismReadingOutside: true,
  });
  await page.evaluate(() => {
    for (const image of document.images) image.loading = 'eager';
  });
  await page.waitForFunction(() => Array.from(document.images).every((image) => image.complete));

  const activePath = join(directory, 'active.pdf');
  await page.pdf({ path: activePath, format: 'Letter', displayHeaderFooter: false });
  const activeText = execFileSync('pdftotext', ['-layout', activePath, '-'], { encoding: 'utf8' });
  const activeInspection = inspectPdf(activePath, {
    expectedVisuals: 1,
    expectedMechanismTitles: ['MECHANISM HEADING'],
  });
  const activePageCount = Number(
    execFileSync('pdfinfo', [activePath], { encoding: 'utf8' }).match(/^Pages:\s+(\d+)$/m)?.[1],
  );
  const activePages = activeText.split('\f').filter((text) => text.trim());
  assert(activePageCount >= 2);
  assert.equal(activePages.length, activePageCount);
  const pageContaining = (pages, phrase) => {
    const matchingPages = pages.flatMap((text, index) => (text.includes(phrase) ? [index] : []));
    assert.equal(matchingPages.length, 1, `${phrase} appears on exactly one page`);
    return matchingPages[0];
  };
  assert.equal(
    pageContaining(activePages, 'TALL VISUAL HEADING'),
    pageContaining(activePages, 'Calculated scenario; no device execution was performed'),
    'the heading, prediction and plot provenance stay together even with a tall image',
  );
  assert.equal(activeInspection.numberedPages.length, activePageCount);
  assert.equal(
    pageContaining(activePages, 'MECHANISM HEADING'),
    pageContaining(activePages, 'Architecture and dataflow mechanism diagram.'),
    'the mechanism heading and prompt stay with its diagram caption',
  );
  activePages.forEach((pageText, index) => {
    assert(
      pageText.split(/\r?\n/).some((line) => line.trim() === `Page ${index + 1}`),
      `page ${index + 1} has its CSS generated footer`,
    );
  });

  await page.addStyleTag({
    content: `@media print {
      .print-visual-intro { break-inside: auto !important; page-break-inside: auto !important; }
      .print-visual-intro .figure-scroll img { max-height: none !important; }
    }`,
  });
  const splitPath = join(directory, 'split-intro.pdf');
  await page.pdf({ path: splitPath, format: 'Letter', displayHeaderFooter: false });
  assert.throws(
    () =>
      inspectPdf(splitPath, {
        expectedVisuals: 1,
        expectedMechanismTitles: ['MECHANISM HEADING'],
      }),
    /Visual print intro is split|Mechanism print intro is split/,
    'the PDF qualifier rejects a tall figure split from its heading and prediction',
  );
  const splitText = execFileSync('pdftotext', ['-layout', splitPath, '-'], { encoding: 'utf8' });
  const splitPages = splitText.split('\f').filter((text) => text.trim());
  assert.notEqual(
    pageContaining(splitPages, 'TALL VISUAL HEADING'),
    pageContaining(splitPages, 'Calculated scenario; no device execution was performed'),
    'disabling the grouping rule reproduces a split visual intro',
  );

  await page.addStyleTag({
    content: `@page { @bottom-center { content: none !important; } }
      @media print {
        .visual-prediction { break-after: auto !important; }
        figure { break-inside: auto !important; }
      }`,
  });

  const disabledPath = join(directory, 'disabled.pdf');
  await page.pdf({ path: disabledPath, format: 'Letter', displayHeaderFooter: false });
  const disabledText = execFileSync('pdftotext', ['-layout', disabledPath, '-'], {
    encoding: 'utf8',
  });
  assert.doesNotMatch(disabledText, /\bPage\s+\d+\b/);
  assert(activeInspection.pageCount >= 2);

  const largePage = await browser.newPage();
  const largeTextPages = Array.from(
    { length: 24 },
    () => `<section class="large-text">${'large text '.repeat(10_000)}</section>`,
  ).join('');
  await largePage.setContent(`<!doctype html><html><body>${largeTextPages}</body></html>`);
  await largePage.addStyleTag({ content: css });
  await largePage.addStyleTag({
    content:
      '.large-text { min-height: 220mm; break-after: page; font: 1pt/1.05 monospace; overflow-wrap: anywhere; }',
  });
  const largePdfPath = join(directory, 'large-extracted-text.pdf');
  await largePage.pdf({ path: largePdfPath, format: 'Letter', displayHeaderFooter: false });
  const largeInspection = inspectPdf(largePdfPath);
  assert(
    largeInspection.extractedTextBytes > 1024 * 1024,
    `fixture extraction exceeds Node's default 1 MiB buffer (${largeInspection.extractedTextBytes} bytes)`,
  );
  assert.equal(largeInspection.numberedPages.length, largeInspection.pageCount);
});
