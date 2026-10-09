import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

// Keep the saved HTML's equations readable without a network connection or asset folder.
const require = createRequire(import.meta.url);
const katexPath = require.resolve('katex/dist/katex.min.css');
const katexCss = readFileSync(katexPath, 'utf8').replace(
  /src:url\((fonts\/[^)]+\.woff2)\)[^;}]+/g,
  (_, font) =>
    `src:url(data:font/woff2;base64,${readFileSync(join(dirname(katexPath), font)).toString('base64')}) format("woff2")`,
);
const styles = [
  'tokens.css',
  'features/lesson-visuals.css',
  'features/math.css',
  'shared/code.css',
  'print.css',
];
export const readerStyles = [
  katexCss,
  ...styles.map((file) => readFileSync(join(process.cwd(), 'src/styles', file), 'utf8')),
].join('\n');
