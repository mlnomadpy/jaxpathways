import { expect, test } from '@playwright/test';

test('the optional local recovery guide presents the observed restart boundary and readable figure', async ({
  page,
}) => {
  await page.goto('/project-guides/sharded-training-multiprocess.html');
  await expect(page.locator('.page-intro')).toContainText('local POSIX CPU processes');
  await expect(page.locator('.page-intro')).toContainText(
    'Windows process recovery remains unqualified',
  );
  const image = page.locator('img[src$="multiprocess/figures/recovery-sequence.svg"]');
  await image.scrollIntoViewIfNeeded();
  await expect(image).toBeVisible();
  await expect
    .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalWidth))
    .toBeGreaterThan(0);
  const boundary = await image.boundingBox();
  expect(boundary).not.toBeNull();
  expect(boundary!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBe(true);
  await expect(image).toHaveAttribute('alt', /Both cases restart the whole cluster/);
  const link = page.getByRole('link', {
    name: /^Open full-size figure: Two observed kill-and-restart sequences/,
  });
  await link.focus();
  await expect(link).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(
    /\/project-assets\/sharded-training\/multiprocess\/figures\/recovery-sequence\.svg$/,
  );
  await page.goBack();
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
});
