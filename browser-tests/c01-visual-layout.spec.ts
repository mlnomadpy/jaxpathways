import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const retrievalFigure =
  '**/project-assets/embedding-model-300m/reference/evidence/final8/figures/final-retrieval.svg*';
const retrievalFigureFile = fileURLToPath(
  new URL(
    '../projects/embedding-model-300m/reference/evidence/final8/figures/final-retrieval.svg',
    import.meta.url,
  ),
);

test('C01 project plan wraps its evidence path and keeps tables inside the mobile page', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/project-plans/embedding-model-300m.html');
  await expect(page.locator('main').first()).toBeVisible();
  await expect(page.locator('.material-status')).toContainText('evidence index');
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBe(true);

  const evidenceTable = page.locator('#assessment article table').first();
  await expect(evidenceTable).toBeVisible();
  const tableBoundary = await evidenceTable.evaluate((table) => ({
    clientWidth: table.clientWidth,
    scrollWidth: table.scrollWidth,
    overflowX: getComputedStyle(table).overflowX,
  }));
  expect(tableBoundary.overflowX).toBe('auto');
  expect(tableBoundary.clientWidth).toBeLessThanOrEqual(390);
});

test('C01 local figure reserves its decoded geometry on narrow and desktop viewports', async ({
  page,
}) => {
  for (const width of [390, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    let markArrived!: () => void;
    let releaseRequest!: () => void;
    const arrived = new Promise<void>((resolve) => {
      markArrived = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      releaseRequest = resolve;
    });
    await page.route(retrievalFigure, async (route) => {
      markArrived();
      await gate;
      await route.fulfill({
        path: retrievalFigureFile,
        contentType: 'image/svg+xml',
        headers: { 'cache-control': 'no-store' },
      });
    });

    await page.goto('/project-plans/embedding-model-300m.html');
    const image = page.getByAltText('Measured final8 retrieval metrics').first();
    await expect(image).toHaveAttribute('width', '900');
    await expect(image).toHaveAttribute('height', '520');
    await image.evaluate((element, query) => {
      const imageElement = element as HTMLImageElement;
      const source = imageElement.getAttribute('src');
      if (!source) throw new Error('C01 retrieval figure is missing its source');
      imageElement.removeAttribute('src');
      imageElement.setAttribute('src', `${source}?geometry-check=${query}`);
    }, width);
    await image.scrollIntoViewIfNeeded();
    await arrived;

    const before = await image.evaluate((element) => {
      const imageElement = element as HTMLImageElement;
      const figureParagraph = imageElement.closest('p');
      return {
        imageHeight: imageElement.getBoundingClientRect().height,
        naturalHeight: imageElement.naturalHeight,
        nextTop: figureParagraph?.nextElementSibling?.getBoundingClientRect().top ?? null,
      };
    });
    expect(before.imageHeight).toBeGreaterThan(0);
    expect(before.naturalHeight).toBe(0);

    releaseRequest();
    await expect
      .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalHeight))
      .toBe(520);
    const after = await image.evaluate((element) => {
      const imageElement = element as HTMLImageElement;
      const figureParagraph = imageElement.closest('p');
      return {
        imageHeight: imageElement.getBoundingClientRect().height,
        nextTop: figureParagraph?.nextElementSibling?.getBoundingClientRect().top ?? null,
      };
    });
    expect(Math.abs(after.imageHeight - before.imageHeight)).toBeLessThanOrEqual(1);
    if (before.nextTop !== null && after.nextTop !== null) {
      expect(Math.abs(after.nextTop - before.nextTop)).toBeLessThanOrEqual(1);
    }
    await page.unroute(retrievalFigure);
  }
});
