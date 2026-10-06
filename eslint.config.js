import js from '@eslint/js';
import globals from 'globals';
export default [
  { ignores: ['node_modules/**', 'dist/**', '.astro/**', 'public/**', 'scripts/migrate-*.mjs'] },
  {
    files: ['src/**/*.js', 'scripts/**/*.mjs', 'astro.config.mjs'],
    ...js.configs.recommended,
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
    },
    rules: {
      'no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', ignoreRestSiblings: true, caughtErrors: 'none' },
      ],
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
];
