import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

async function expectSingleNonOverlappingDock(page: Page) {
  const dock = page.locator('#lesson-stage-bar');
  await expect(dock).toBeVisible();
  await expect(page.locator('#lesson-stage-bar:visible')).toHaveCount(1);
  await expect(page.locator('#activity-bar')).toHaveCount(0);

  const overlappingControls = await dock.evaluate((element) => {
    const controls = [
      ...element.querySelectorAll<HTMLElement>(
        '.stage-bar-context, .stage-bar-pills a, .stage-bar-navigation a',
      ),
    ].filter((control) => !control.hidden && control.getClientRects().length > 0);
    const boxes = controls.map((control) => ({
      label: control.textContent?.trim() || control.id,
      rect: control.getBoundingClientRect(),
    }));
    return boxes.flatMap((first, index) =>
      boxes.slice(index + 1).flatMap((second) => {
        const width =
          Math.min(first.rect.right, second.rect.right) -
          Math.max(first.rect.left, second.rect.left);
        const height =
          Math.min(first.rect.bottom, second.rect.bottom) -
          Math.max(first.rect.top, second.rect.top);
        return width > 1 && height > 1 ? [[first.label, second.label]] : [];
      }),
    );
  });

  expect(overlappingControls).toEqual([]);
}

for (const viewport of [
  { width: 1280, height: 620 },
  { width: 390, height: 620 },
]) {
  test(`one stage dock keeps keyboard navigation usable at ${viewport.width}×${viewport.height}`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await page.goto('/lesson-welcome-01.html');

    const dock = page.locator('#lesson-stage-bar');
    const buildPill = dock.locator('[data-stage-pill="Build"]');
    const previous = dock.locator('#previous-activity');
    const next = dock.locator('#next-activity');
    await expect(buildPill).toHaveAttribute('href', /#section-\d+/);
    await expect(previous).toBeHidden();
    await expect(next).toHaveAttribute('href', /#section-\d+/);
    await expect(next).toHaveText('Next: Build');
    await expect(
      page.getByRole('group', { name: 'Previous and next lesson activities' }),
    ).toBeVisible();
    await expectSingleNonOverlappingDock(page);

    await next.focus();
    await page.keyboard.press('Enter');
    await expect(dock.locator('#stage-bar-active-label')).toContainText('Stage 2 of');
    await expect(previous).toBeVisible();
    await expect(previous).toHaveText('Previous: Understand');
    await previous.focus();
    await page.keyboard.press('Enter');
    await expect(dock.locator('#stage-bar-active-label')).toContainText('Stage 1 of');
    await expect(previous).toBeHidden();

    await dock.locator('[data-stage-pill="Evidence"]').click();
    await expect(dock.locator('#stage-bar-active-label')).toContainText(/Stage \d+ of/);
    await expect(next).toHaveText('Next lesson');
    await expect(next).toHaveAttribute('href', /lesson-[\w-]+\.html/);
    await expectSingleNonOverlappingDock(page);
    const destination = await next.getAttribute('href');
    expect(destination).toBeTruthy();
    await next.focus();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(new URL(destination!, page.url()).href);
    await expect(page.locator('#lesson-stage-bar')).toBeVisible();
    await page.goBack();
    await expect(page.locator('#stage-bar-active-label')).toContainText('Keep Your Evidence');
  });
}

test('the final lesson sends its last activity to the course map', async ({ page }) => {
  const course = JSON.parse(
    readFileSync(new URL('../curriculum/course.json', import.meta.url), 'utf8'),
  ) as {
    phases: { id: string; lessons: { id: string; status: string }[] }[];
  };
  const finalLesson = course.phases
    .flatMap((phase) => phase.lessons)
    .filter((lesson) => lesson.status === 'authored')
    .at(-1)!;
  const initialVisiblePhase = course.phases.find((phase) =>
    phase.lessons.some((lesson) => lesson.status === 'authored'),
  )!;
  const firstVisibleLesson = initialVisiblePhase.lessons.find(
    (lesson) => lesson.status === 'authored',
  )!;
  await page.goto(`/lesson-${finalLesson.id}.html`);
  const dock = page.locator('#lesson-stage-bar');
  await dock.locator('[data-stage-pill="Evidence"]').click();
  const next = dock.locator('#next-activity');
  await expect(next).toHaveText('Course map');
  await expect(next).toHaveAttribute('href', 'course.html');
  await next.focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL((url) => {
    return (
      url.pathname === '/course.html' &&
      url.searchParams.get('phase') === initialVisiblePhase.id &&
      [...url.searchParams.keys()].every((key) => key === 'phase')
    );
  });
  const phaseCard = page.locator(`#phase-${initialVisiblePhase.id}`);
  await expect(phaseCard).toBeVisible();
  await expect(phaseCard).toHaveAttribute('open', '');
  await expect(phaseCard.locator(`#${firstVisibleLesson.id}`)).toBeVisible();
});
