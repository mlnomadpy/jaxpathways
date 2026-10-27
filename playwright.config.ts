import { defineConfig, devices } from '@playwright/test';

const portText = process.env.PLAYWRIGHT_PORT;
const port = portText === undefined ? 4173 : Number(portText);
if (
  !Number.isInteger(port) ||
  port < 1 ||
  port > 65535 ||
  (portText !== undefined && !/^\d+$/.test(portText))
) {
  throw new Error('PLAYWRIGHT_PORT must be an integer from 1 to 65535.');
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: './browser-tests',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...(process.env.CHROME_BIN
      ? { launchOptions: { executablePath: process.env.CHROME_BIN } }
      : {}),
  },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile-chromium', use: { ...devices['iPhone 13'], defaultBrowserType: 'chromium' } },
  ],
  webServer: {
    command: `npm run preview -- --host 127.0.0.1 --port ${port} --ignore-lock`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
