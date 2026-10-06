import { defineConfig } from 'astro/config';

export default defineConfig({
  site: process.env.SITE_URL || 'https://mlnomadpy.github.io',
  base: process.env.BASE_PATH || '/',
  output: 'static',
  trailingSlash: 'never',
  build: { format: 'file', inlineStylesheets: 'never' },
  server: { host: '127.0.0.1', port: 4173 },
});
