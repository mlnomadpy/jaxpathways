import fs from 'node:fs';
import { bookGuide } from '../src/lib/server/book-guides.js';
import { normalizeEpubMathML } from '../src/lib/mathml-export.js';
const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
const output = {};
for (const [id, source] of Object.entries({ workflow: 'content/guides/project-workflow.md', experiments: 'projects/reproducible-experiments/README.md' })) {
  output[id] = normalizeEpubMathML(bookGuide(source, course, { mathOutput: 'mathml' }).replace(/<(hr|br|img)(\s[^>]*?)?\s*\/?\s*>/g, '<$1$2 />'));
}
process.stdout.write(JSON.stringify(output));
