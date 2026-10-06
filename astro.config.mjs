import { defineConfig } from 'astro/config';

export default defineConfig({
  site: process.env.SITE_URL || 'https://www.tahabouhsine.com',
  base: process.env.BASE_PATH || '/',
  output: 'static',
  trailingSlash: 'never',
  build: { format: 'file', inlineStylesheets: 'never' },
  server: { host: '127.0.0.1', port: 4173 },
  vite: {
    optimizeDeps: { include: ['katex'] },
    plugins: [
      {
        name: 'separate-dev-build-dependency-caches',
        config: (_config, { command }) => ({
          // Builds must not replace dependencies served by an open dev session.
          cacheDir: `node_modules/.vite/${command}`,
        }),
      },
    ],
  },
});
