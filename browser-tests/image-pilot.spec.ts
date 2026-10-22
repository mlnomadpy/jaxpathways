import { expect, test } from '@playwright/test';

test('the image pilot guide is discoverable and its seven figures fit desktop and mobile', async ({
  page,
}) => {
  await page.goto('/project-plans/image-model-300m.html');
  await expect(
    page.getByRole('link', { name: 'Read the flower-model pilot guide and figures' }),
  ).toBeVisible();
  await page.getByRole('link', { name: 'Read the flower-model pilot guide and figures' }).click();
  await expect(page).toHaveURL(/project-guides\/image-model-300m-pilot\.html$/);
  await expect(page.locator('.page-intro')).toContainText('classification gate failed');
  await expect(page.locator('.page-intro')).toContainText('full 300M capstone remains Planned');

  const figures = page.locator('section[aria-labelledby="pilot-figures"] figure');
  await expect(figures).toHaveCount(7);
  const firstImage = figures.first().locator('img');
  await expect(firstImage).toHaveAttribute('width', '900');
  await expect(firstImage).toHaveAttribute('height', '500');
  const fullSizeLink = page.getByRole('link', {
    name: 'Open full-size figure: MAE masked reconstruction loss by one-based update, measured as mean squared normalized-pixel target.',
    exact: true,
  });
  await fullSizeLink.focus();
  await expect(fullSizeLink).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/loss-mae\.svg$/);
  const figureResponse = await page.request.get(page.url());
  expect(figureResponse.ok()).toBe(true);
  expect(figureResponse.headers()['content-type']).toContain('image/svg+xml');
  await page.goBack();
  await expect(page).toHaveURL(/project-guides\/image-model-300m-pilot\.html$/);
  await expect(page.locator('h1')).toBeVisible();

  for (const figure of await figures.all()) {
    const image = figure.locator('img');
    await image.scrollIntoViewIfNeeded();
    await expect
      .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalWidth))
      .toBeGreaterThan(0);
    const bounds = await image.evaluate((element) => {
      const imageElement = element as HTMLImageElement;
      const rectangle = imageElement.getBoundingClientRect();
      return { width: rectangle.width, viewport: window.innerWidth };
    });
    expect(bounds.width).toBeGreaterThan(0);
    expect(bounds.width).toBeLessThanOrEqual(bounds.viewport + 1);
  }
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBe(true);

  await page.goto('/projects.html');
  await expect(
    page.getByRole('link', { name: 'Read the flower-model pilot guide and figures' }),
  ).toBeVisible();
  await page.getByRole('link', { name: 'Read the flower-model pilot guide and figures' }).click();
  await expect(page).toHaveURL(/project-guides\/image-model-300m-pilot\.html$/);
});
