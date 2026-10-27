import { expect, test } from '@playwright/test';

test('sequence PPO stages name their edit file and explain an accessible GAE figure', async ({
  page,
}) => {
  await page.goto('/project.html?id=posttraining-audit');
  const sequence = page.locator('#stage-5');
  await expect(sequence.locator('p strong code').first()).toHaveText(
    'projects/posttraining-audit/starter/sequence_ppo.py',
  );
  await expect(sequence.locator('.stage-code-header')).toContainText('starter/sequence_ppo.py');
  await expect(sequence.locator('.project-command pre')).toContainText(
    'tests/check_sequence_ppo.py --implementation starter --stage 5',
  );
  await expect(sequence.locator('.stage-solution-details')).toHaveCount(0);
  await expect(page.locator('#stage-1 p strong code').first()).toHaveText(
    'projects/posttraining-audit/starter/model.py',
  );
  await expect(page.locator('.project-tree')).toContainText('starter/sequence_ppo.py');

  await page.goto('/project-guides/posttraining-audit.html');
  await expect(page.locator('.project-guide-brief')).toContainText('starter/sequence_ppo.py');
  await expect(page.locator('.katex-error')).toHaveCount(0);
  expect(await page.locator('article .katex').count()).toBeGreaterThan(20);
  await expect(page.locator('article')).toContainText(/both stops set the trace continuation/i);
  await expect(page.locator('article')).toContainText('does not measure unseen-task');
  const sourceHref = await page
    .getByRole('link', { name: 'independent plotting script', exact: true })
    .getAttribute('href');
  expect(sourceHref).toMatch(/\/render_gae\.py$/);
  const sourceResponse = await page.request.get(sourceHref!);
  expect(sourceResponse.ok()).toBe(true);
  expect(await sourceResponse.text()).toContain('def calculate(fixture):');

  const image = page.getByRole('img', { name: /^Two illustrative responses/ });
  await image.scrollIntoViewIfNeeded();
  await expect(image).toHaveAttribute('width', '1152');
  await expect(image).toHaveAttribute('height', '615');
  await expect
    .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalWidth))
    .toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
    true,
  );
  const fullSize = page.getByRole('link', {
    name: /^Open full-size figure: Two illustrative responses/,
  });
  await fullSize.focus();
  await expect(fullSize).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(
    /project-assets\/posttraining-audit\/figures\/gae-eos-timeout\.svg$/,
  );
  const response = await page.request.get(page.url());
  expect(response.ok()).toBe(true);
  expect(response.headers()['content-type']).toContain('image/svg+xml');
});
