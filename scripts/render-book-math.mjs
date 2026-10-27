// One batch process keeps EPUB generation independent of a browser or CDN.
import fs from 'node:fs';
import { inlineMath, renderMath } from '../src/lib/math.js';
import { normalizeEpubMathML } from '../src/lib/mathml-export.js';
const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
const prose = {}, equations = {};
function visit(value, key = '') {
  if (['code','solution','buildCode','formula'].includes(key)) return;
  if (typeof value === 'string') {
    if (key === 'math') equations[value] = normalizeEpubMathML(renderMath(value, true, 'mathml'));
    else
      for (const paragraph of value.split('\n\n')) {
        if (/\\\(|\\\[|\$\$|`|\*\*|\[[^\]\n]+\]\([^)]+\)/.test(paragraph))
          prose[paragraph] = normalizeEpubMathML(inlineMath(paragraph, 'mathml'));
      }
  } else if (Array.isArray(value)) value.forEach(item => visit(item, key));
  else if (value && typeof value === 'object') for (const [childKey, child] of Object.entries(value)) visit(child, childKey);
}
visit(course);
process.stdout.write(JSON.stringify({prose, equations}));
