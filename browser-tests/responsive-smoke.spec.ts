import { expect, test } from '@playwright/test';

const pages = [
  '/',
  '/course.html',
  '/lesson-welcome-01.html',
  '/project.html?id=regression-audit',
  '/careers.html',
  '/notebook.html',
];

for (const path of pages) {
  test(`${path} renders without horizontal overflow and exposes a main landmark`, async ({
    page,
  }) => {
    await page.goto(path);
    await expect(page.locator('main').first()).toBeVisible();
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
      )
      .toBe(true);
  });
}

test('catalog filters are keyboard reachable and have accessible names', async ({ page }) => {
  await page.goto('/course.html');
  await expect(page.locator('#course-view-status')).toContainText('available lesson drafts');
  const filters = page.locator('[data-course-filter]');
  await expect(filters.first()).toHaveAccessibleName(/^All\s*\(\d+\)$/);
  await page.keyboard.press('Tab');
  await expect(page.locator(':focus')).toBeVisible();
  const before = await page.locator('.phase-card:visible').count();
  const foundations = page.locator('[data-course-filter="foundation"]');
  await foundations.click();
  await expect(foundations).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(async () => page.locator('.phase-card:visible').count()).toBeLessThan(before);
});

test('catalog search opens and closes with keyboard focus restored', async ({ page }) => {
  await page.goto('/course.html');
  await expect(page.locator('#course-view-status')).toContainText('available lesson drafts');
  const trigger = page.locator('#course-search-open');
  await trigger.focus();
  await page.keyboard.press('Enter');
  const dialog = page.locator('#course-search-dialog');
  const input = page.locator('#course-search-input');
  await expect(dialog).toBeVisible();
  await expect(input).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
});

test('project details remain usable at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/project.html?id=regression-audit');
  await expect(page.locator('main').first()).toBeVisible();
  await expect(page.locator('#project h1')).not.toContainText('Loading');
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBe(true);
});

test('project catalog links have real targets at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/projects.html');
  const links = page.locator('.project-entry a');
  await expect(links.first()).toBeVisible();
  await expect
    .poll(() =>
      links.evaluateAll((items) =>
        items.every((link) => Boolean(link.getAttribute('href')?.trim())),
      ),
    )
    .toBe(true);
  await expect(page.locator('.project-entry a:not([href])')).toHaveCount(0);
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1))
    .toBe(true);
});

test('mobile reader places its outline before the article and stage jumps clear the sticky bar', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/lesson-welcome-01.html');
  const outline = page.locator('.reader-outline');
  await expect(outline).toBeVisible();
  const followsOutline = await page.evaluate(() => {
    const node = document.querySelector('.reader-outline');
    const body = document.querySelector('#lesson-body');
    return Boolean(
      node && body && node.compareDocumentPosition(body) & Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });
  expect(followsOutline).toBe(true);
  const build = page.locator('#lesson-stage-bar [data-stage-pill="Build"]');
  await expect(build).toHaveAttribute('href', /#section-\d+/);
  await build.click();
  const target = page.locator(`${await build.getAttribute('href')}`);
  await expect(target).toBeVisible();
  await expect(page.locator('#stage-bar-active-label')).toContainText('Stage 2 of');
  const clearOfBar = await page.evaluate(() => {
    const heading = document.querySelector('[data-activity-group="Build"]');
    const bar = document.querySelector('#lesson-stage-bar');
    return Boolean(
      heading &&
      bar &&
      heading.getBoundingClientRect().top >= bar.getBoundingClientRect().bottom - 1,
    );
  });
  expect(clearOfBar).toBe(true);
});

test('a failed navigation manifest leaves server-rendered lesson content readable', async ({
  page,
}) => {
  await page.route('**/api/v1/course-navigation.json', (route) => route.abort());
  await page.goto('/lesson-welcome-01.html');
  await expect(page.locator('main').first()).toBeVisible();
  await expect(page.locator('article')).toBeVisible();
  await expect(page.locator('h1').first()).toBeVisible();
});

test('legacy lesson URLs redirect to the canonical page and preserve route, search, and section', async ({
  page,
}) => {
  await page.goto('/lesson.html?lesson=welcome-03&path=foundations&find=backend#section-2');
  await expect(page).toHaveURL(/\/lesson-welcome-03\.html\?path=foundations&find=backend#section-2$/);
  await expect(page.locator('#lesson-title')).toContainText('Move your experiment to a TPU');
});
