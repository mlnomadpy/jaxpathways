import { createHash } from 'node:crypto';
import { expect, test } from '@playwright/test';

const portableApprovalSha = '28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b';
const learnerApprovalSha = '72ecd919095e2fcc4bc095afee6f27a74571ba61aa1a56ae7c9ca442399c3c48';

async function expectApprovalDownload(
  page: import('@playwright/test').Page,
  linkName: string,
  expectedSha: string,
  expectedAuthorization: 'portable' | 'learner',
) {
  const link = page.getByRole('link', { name: linkName, exact: true });
  const href = await link.getAttribute('href');
  expect(href).toMatch(/^\/project-assets\/posttraining-audit\/transformer_lineage\//);

  const response = await page.request.get(href!);
  expect(response.ok()).toBe(true);
  const bytes = await response.body();
  expect(createHash('sha256').update(bytes).digest('hex')).toBe(expectedSha);

  const approval = JSON.parse(bytes.toString('utf8')) as Record<string, unknown>;
  expect(approval.runAuthorized).toBe(false);
  if (expectedAuthorization === 'learner') {
    expect(approval.learnerVerificationAuthorized).toBe(true);
  }
}

test('catalog and parent guide expose the optional appendix without changing Stage 5', async ({
  page,
}) => {
  await page.goto('/projects.html');
  const projectCard = page.locator('#project-posttraining-audit');
  await expect(
    projectCard.getByRole('link', { name: 'Saved decoder PPO appendix', exact: true }),
  ).toHaveAttribute('href', '/project-guides/posttraining-transformer-lineage.html');

  await page.goto('/project-guides/posttraining-audit.html');
  await expect(
    page.getByRole('link', {
      name: 'trace sequence PPO with the saved 10,880-parameter decoder',
      exact: true,
    }),
  ).toHaveAttribute('href', '/project-guides/posttraining-transformer-lineage.html');

  await page.goto('/project.html?id=posttraining-audit');
  await expect(page.locator('#stage-5 p strong code').first()).toHaveText(
    'projects/posttraining-audit/starter/sequence_ppo.py',
  );
});

test('appendix status, accessible figure, bounded layout, and approval downloads are explicit', async ({
  page,
}) => {
  await page.goto('/project-guides/posttraining-transformer-lineage.html');

  const preparation = page.locator('[aria-labelledby="preparation-project-status-heading"]');
  await expect(preparation.locator('[data-maturity="available-draft"]')).toHaveCount(1);
  const extension = page.locator(
    '[data-material-key="extension:posttraining-transformer-lineage"]',
  );
  await expect(extension.locator('[data-maturity="in-development"]')).toHaveCount(1);
  await expect(extension).toContainText('heldout reward quality gate failed');
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await expect(page.locator('article .katex').first()).toBeVisible();

  await expectApprovalDownload(page, 'portable non-run approval', portableApprovalSha, 'portable');
  await expectApprovalDownload(page, 'learner mechanics approval', learnerApprovalSha, 'learner');

  const figure = page.locator('figure.print-visual-intro[aria-labelledby="part-9"]');
  await expect(figure).toHaveCount(1);
  await expect(figure.locator('figcaption')).toContainText('strict positive improvement floor');
  const image = figure.getByRole('img', { name: /^Mean child-minus-parent fitted-reward change/ });
  await image.scrollIntoViewIfNeeded();
  await expect(image).toHaveAttribute('width', '1000');
  await expect(image).toHaveAttribute('height', '620');
  await expect
    .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalWidth))
    .toBe(1000);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
    true,
  );

  const fullSize = figure.getByRole('link', { name: /^Open full-size figure:/ });
  await fullSize.focus();
  await expect(fullSize).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(
    /project-assets\/posttraining-audit\/transformer_lineage\/teaching\/v5\/figure\.svg$/,
  );
  const svgResponse = await page.request.get(page.url());
  expect(svgResponse.ok()).toBe(true);
  expect(svgResponse.headers()['content-type']).toContain('image/svg+xml');
});

test('standalone book keeps the appendix figure grouped, embedded, and within the viewport', async ({
  page,
}) => {
  await page.goto('/downloads/jax-foundations.html#appendix-posttraining-lineage');

  const appendix = page.locator('#appendix-posttraining-lineage');
  await expect(appendix).toBeVisible();
  const figure = appendix.locator('figure.print-visual-intro');
  await expect(figure).toHaveCount(1);
  await expect(figure.locator('h3')).toHaveText('Recorded v5 result');
  await expect(figure.locator('figcaption')).toContainText('second inner optimizer epoch');
  expect(await figure.evaluate((element) => element.lastElementChild?.tagName.toLowerCase())).toBe(
    'figcaption',
  );

  const image = figure.locator('img');
  await expect(image).toHaveCount(1);
  await expect(image).toHaveAttribute('src', /^data:image\/svg\+xml;base64,/);
  await image.scrollIntoViewIfNeeded();
  await expect
    .poll(() => image.evaluate((element) => (element as HTMLImageElement).naturalWidth))
    .toBe(1000);
  const bounds = await figure.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(await page.evaluate(() => innerWidth + 1));

  expect(await appendix.locator('math').count()).toBeGreaterThan(0);
  expect(
    await appendix.locator('math annotation[encoding="application/x-tex"]').count(),
  ).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
    true,
  );
});
