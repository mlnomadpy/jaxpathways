import { expect, test } from '@playwright/test';

test('standalone reader print lifecycle opens reasoning and restores every details state', async ({
  page,
}) => {
  await page.goto('/downloads/jax-foundations.html');
  await expect(page.locator('html')).toHaveAttribute('data-reader-ready', 'true');

  const initialStates = await page.evaluate(() => {
    const reasoning = [
      ...document.querySelectorAll<HTMLDetailsElement>('details:not(.solution-reveal)'),
    ];
    const solutions = [...document.querySelectorAll<HTMLDetailsElement>('details.solution-reveal')];
    if (reasoning.length < 2 || solutions.length < 2) {
      throw new Error('The reader needs at least two reasoning and two solution disclosures.');
    }

    reasoning.forEach((details, index) => {
      details.open = index === 0;
    });
    solutions.forEach((details, index) => {
      details.open = index === 0;
    });

    return [...document.querySelectorAll<HTMLDetailsElement>('details')].map((details) => ({
      id: details.id,
      className: details.className,
      open: details.open,
    }));
  });

  await page.evaluate(() => {
    window.dispatchEvent(new Event('beforeprint'));
    window.dispatchEvent(new Event('beforeprint'));
  });
  await page.emulateMedia({ media: 'print' });

  const printState = await page.evaluate(() => ({
    reasoningOpen: [
      ...document.querySelectorAll<HTMLDetailsElement>('details:not(.solution-reveal)'),
    ].every((details) => details.open),
    solutionStates: [
      ...document.querySelectorAll<HTMLDetailsElement>('details.solution-reveal'),
    ].map((details) => details.open),
    solutionDisplay: getComputedStyle(document.querySelector('.solution-reveal')!).display,
    appendixDisplay: getComputedStyle(document.querySelector('.print-solutions')!).display,
    starterTokenColors: Object.fromEntries(
      ['syntax-keyword', 'syntax-string', 'syntax-comment'].map((tokenClass) => {
        const token = document.querySelector(`.starter-scaffold .${tokenClass}`);
        return [tokenClass, token ? getComputedStyle(token).color : null];
      }),
    ),
  }));

  expect(printState.reasoningOpen).toBe(true);
  expect(printState.solutionStates).toContain(true);
  expect(printState.solutionStates).toContain(false);
  expect(printState.solutionDisplay).toBe('none');
  expect(printState.appendixDisplay).toBe('block');
  expect(printState.starterTokenColors).toEqual({
    'syntax-keyword': 'rgb(17, 17, 17)',
    'syntax-string': 'rgb(17, 17, 17)',
    'syntax-comment': 'rgb(17, 17, 17)',
  });

  await page.evaluate(() => {
    window.dispatchEvent(new Event('afterprint'));
    window.dispatchEvent(new Event('afterprint'));
  });
  await page.emulateMedia({ media: 'screen' });

  const restoredStates = await page.evaluate(() =>
    [...document.querySelectorAll<HTMLDetailsElement>('details')].map((details) => ({
      id: details.id,
      className: details.className,
      open: details.open,
    })),
  );
  expect(restoredStates).toEqual(initialStates);
});
