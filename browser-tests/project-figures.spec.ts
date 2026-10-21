import { test, expect } from '@playwright/test';

for (const figure of [
  {
    guide: '/project-guides/text-harness-foundations.html',
    label: 'Measured text-harness foundations pilot',
    asset: '/project-assets/text-harness/foundations/reference/pilot-metrics.svg',
    type: 'image/svg+xml',
  },
  {
    guide: '/project-guides/scientific-inverse.html',
    label: 'Adaptive solver log-rate sensitivities',
    asset: '/project-assets/scientific-inverse/reference/diffrax/sensitivity.png',
    type: 'image/png',
  },
  {
    guide: '/project-plans/causal-model-300m.html',
    label:
      'Training losses decrease overall, while all six pilot branches remain below both held-out QA floors',
    asset:
      '/project-assets/causal-model-300m/reference/evidence/c02-pilot/figures/c02-training-and-qa.svg',
    type: 'image/svg+xml',
  },
  {
    guide: '/project-plans/causal-model-300m.html',
    label:
      'Bits per normalized held-out byte for six pilot branches, Pythia-14M and a train-only unigram control',
    asset:
      '/project-assets/causal-model-300m/reference/evidence/c02-pilot/figures/c02-byte-rates.svg',
    type: 'image/svg+xml',
  },
]) {
  test(`${figure.guide}: ${figure.label} opens the actual full-size figure with the keyboard`, async ({ page }) => {
    await page.goto(figure.guide);
    const link = page.getByRole('link', {
      name: `Open full-size figure: ${figure.label}`,
      exact: true,
    });
    await link.focus();
    await expect(link).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(
      new RegExp(figure.asset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$'),
    );
    const response = await page.request.get(page.url());
    expect(response.ok()).toBe(true);
    expect(response.headers()['content-type']).toContain(figure.type);
    await page.goBack();
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  });
}
