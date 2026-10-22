import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import { bookGuide } from '../src/lib/server/book-guides.js';
import { groupTransformerLineageFigure } from '../src/lib/server/posttraining-lineage-figure.js';
import { projectGuide } from '../src/lib/server/project-guide.js';

const require = createRequire(import.meta.url);
const katex = require('katex');
const root = process.cwd();
const sourcePath = 'projects/posttraining-audit/transformer_lineage/TEACHING.md';
const rightsPath = 'projects/posttraining-audit/transformer_lineage/teaching/PARENT_RIGHTS.md';
const portableApproval =
  'projects/posttraining-audit/transformer_lineage/portable-nonrun-approval.json';
const learnerApproval =
  'projects/posttraining-audit/transformer_lineage/learner-mechanics-approval.json';
const teaching = fs.readFileSync(sourcePath, 'utf8');
const rights = fs.readFileSync(rightsPath, 'utf8');
const course = JSON.parse(fs.readFileSync('curriculum/course.json', 'utf8'));

function localMarkdownTargets(sourcePathname, markdown) {
  const targets = [];
  for (const [, href] of markdown.matchAll(/\[[^\]]+\]\(([^)]+)\)/g)) {
    if (/^(?:https?:|mailto:|data:|#)/i.test(href)) continue;
    targets.push(path.posix.normalize(path.posix.join(path.posix.dirname(sourcePathname), href)));
  }
  return targets;
}

test('optional appendix route is separate from Stage 5 and reports the failed quality floor', () => {
  const page = fs.readFileSync(
    'src/pages/project-guides/posttraining-transformer-lineage.astro',
    'utf8',
  );
  const mapper = fs.readFileSync('src/lib/server/project-guide.js', 'utf8');
  const card = fs.readFileSync('src/components/projects/ProjectEntryCard.astro', 'utf8');
  const projectReadme = fs.readFileSync('projects/posttraining-audit/README.md', 'utf8');
  const review = JSON.parse(fs.readFileSync('curriculum/content-review.json', 'utf8'));
  const phase = course.phases.find(({ id }) => id === 'posttraining');
  const projectReview = review.projects.find(({ id }) => id === 'posttraining-audit');
  const authoredLessons = course.phases
    .flatMap(({ lessons }) => lessons)
    .filter(({ status }) => status === 'authored');

  assert.match(
    mapper,
    /transformer_lineage\/TEACHING\.md[\s\S]*project-guides\/posttraining-transformer-lineage\.html/,
  );
  assert.match(
    card,
    /project\.id === 'posttraining-audit'[\s\S]*posttraining-transformer-lineage\.html/,
  );
  assert.match(projectReadme, /transformer_lineage\/TEACHING\.md/);
  assert.match(projectReadme, /bounded mechanics and replay checks that passed/i);
  assert.match(projectReadme, /strict heldout reward-quality floor failed/i);
  assert.match(projectReadme, /Human-preference quality, retained behavior, human evaluation/i);
  assert.match(phase.studyGuide.furtherWork, /optional saved-decoder appendix/i);
  assert.match(
    phase.studyGuide.furtherWork,
    /project-guides\/posttraining-transformer-lineage\.html/,
  );
  assert.match(phase.studyGuide.furtherWork, /strict heldout reward quality floor failed/i);
  assert.match(page, /data-material-key="extension:posttraining-transformer-lineage"/);
  assert.match(page, /kind="projects" id="text-harness"/);
  assert.match(page, /strict heldout[\s\S]*quality floor failed/i);
  assert.match(page, /description="[^"]+"/);
  assert.equal(projectReview.level, 'available-draft');
  assert.equal(authoredLessons.length, 104);
  assert.equal(new Set(authoredLessons.map(({ id }) => id)).size, 104);
  assert.match(teaching, /strict heldout reward quality floor remains \*\*failed\*\*/i);
  assert.match(teaching, /does not establish natural-language RLHF quality/i);
  assert.match(teaching, /eight outer updates executed in memory/i);
  assert.match(teaching, /only the update-4 checkpoint was durable/i);
  assert.match(teaching, /no matched baseline response, matched-quality comparison/i);
  assert.doesNotMatch(teaching, /eight prompts were processed in memory/i);
});

test('teaching and parent-rights local links resolve in the extracted project ZIP', () => {
  const included = new Set();
  const projectRoot = path.resolve('projects/posttraining-audit');
  for (const entry of fs.readdirSync(projectRoot, { recursive: true })) {
    if (entry.endsWith('.pyc')) continue;
    const absolute = path.join(projectRoot, entry);
    if (!fs.statSync(absolute).isFile()) continue;
    included.add(path.relative(root, absolute).split(path.sep).join('/'));
  }
  for (const target of [
    ...localMarkdownTargets(sourcePath, teaching),
    ...localMarkdownTargets(rightsPath, rights),
    ...localMarkdownTargets(
      'projects/posttraining-audit/README.md',
      fs.readFileSync('projects/posttraining-audit/README.md', 'utf8'),
    ),
  ]) {
    const absolute = path.resolve(root, target);
    const repoPath = path.relative(root, absolute).split(path.sep).join('/');
    if (repoPath.startsWith('projects/text-harness/foundations/')) {
      assert.ok(
        repoPath ===
          'projects/text-harness/foundations/reference/run/inference-package/package.json' ||
          repoPath.endsWith('/config.py') ||
          repoPath.endsWith('/data.py') ||
          repoPath.endsWith('/model.py') ||
          repoPath.endsWith('/train.py') ||
          repoPath.endsWith('/params.npz') ||
          repoPath.endsWith('/tokenizer.json'),
        `unexpected parent dependency in Markdown: ${repoPath}`,
      );
      assert.ok(fs.existsSync(repoPath), `missing pinned parent package/source: ${repoPath}`);
      continue;
    }
    assert.ok(
      included.has(repoPath),
      `local Markdown target is absent from extracted ZIP: ${repoPath}`,
    );
  }
  assert.match(
    teaching,
    /github\.com\/mlnomadpy\/jaxpathways\/blob\/f858816ead940aeb969e23fbc32df232da51196f/,
  );
  const pinnedManifestUrl =
    'https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/reference/run/inference-package/package.json';
  assert.ok(teaching.includes(pinnedManifestUrl));
  assert.ok(rights.includes(pinnedManifestUrl));
  assert.doesNotMatch(teaching, /sha256sum/);
  assert.match(rights, /gutenberg\.org\/ebooks\/11/);
  assert.match(rights, /does not license or authorize release of a child checkpoint/i);
});

test('POSIX candidate hash command works when the candidate path contains spaces', () => {
  const hashCommand = teaching.match(/CANDIDATE_SHA=\$\((python -c '[^\n]+' "\$CANDIDATE")\)/)?.[1];
  assert.ok(hashCommand, 'guide includes a portable Python SHA-256 command');
  const directory = fs.mkdtempSync(path.join(tmpdir(), 'candidate hash test '));
  const candidate = path.join(directory, 'learner candidate.py');
  const bytes = Buffer.from('candidate bytes with spaces in its path\n');
  fs.writeFileSync(candidate, bytes);
  try {
    const actual = execFileSync('sh', ['-c', hashCommand], {
      encoding: 'utf8',
      env: { ...process.env, CANDIDATE: candidate },
    }).trim();
    assert.equal(actual, createHash('sha256').update(bytes).digest('hex'));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('bundled approvals are exact-hash and reachable from the extracted guide', () => {
  const sha256 = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  assert.equal(
    sha256(portableApproval),
    '28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b',
  );
  assert.equal(
    sha256(learnerApproval),
    '72ecd919095e2fcc4bc095afee6f27a74571ba61aa1a56ae7c9ca442399c3c48',
  );
  assert.match(teaching, /--stage preflight/);
  assert.match(teaching, /--stage semantic/);
  assert.match(teaching, /--stage learner-check/);
  assert.match(teaching, /one outer update, two optimizer epochs, and four reward-fit steps/i);
  assert.match(teaching, /does not score heldout quality/i);
});

test('all appendix math renders with KaTeX and the EPUB renderer preserves MathML', () => {
  const expressions = [...teaching.matchAll(/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g)];
  assert.ok(
    expressions.length >= 15,
    `expected worked equations and inline variables, found ${expressions.length}`,
  );
  for (const [, inline, display] of expressions) {
    assert.doesNotThrow(() =>
      katex.renderToString(inline ?? display, { throwOnError: true, strict: 'error' }),
    );
  }

  const rendered = bookGuide(sourcePath, course, { mathOutput: 'mathml' });
  assert.match(rendered, /<math[\s>]/);
  assert.match(rendered, /application\/x-tex/);
  assert.match(rendered, /strict heldout reward quality floor remains <strong>failed<\/strong>/i);
  assert.match(rendered, /eligible four-action support/i);
  assert.equal((rendered.match(/<figure class="print-visual-intro"/g) || []).length, 1);
  assert.match(
    rendered,
    /<figure class="print-visual-intro" aria-labelledby="projects-posttraining-audit-transformer-lineage-TEACHING-md-part-\d+"><h3 id="projects-posttraining-audit-transformer-lineage-TEACHING-md-part-\d+">Recorded v5 result<\/h3><div class="figure-scroll"[^>]*><img src="data:image\/svg\+xml;base64,[^"]+" alt="Mean child-minus-parent fitted-reward change is zero on dev and heldout; the strict positive heldout floor fails\. Primary signed policy loss from inner epoch 2 by update, averaged per valid action\. Exact restricted KL per valid action is 0\.000123197, below its 0\.8 ceiling\."[^>]*\/><\/div><figcaption><strong>Figure 1\.<\/strong>/,
  );
  assert.match(
    rendered,
    /<figcaption><strong>Figure 1\.<\/strong>[\s\S]*?<\/figcaption><\/figure>/,
  );

  const online = projectGuide({ id: 'posttraining-audit', source: sourcePath }, course);
  assert.equal((online.body.match(/<figure class="print-visual-intro"/g) || []).length, 1);
  assert.match(
    online.body,
    /<a href="\/project-assets\/posttraining-audit\/transformer_lineage\/teaching\/v5\/figure\.svg"/,
  );
  assert.match(
    online.body,
    /<figcaption><strong>Figure 1\.<\/strong>[\s\S]*?<\/figcaption><\/figure>/,
  );
});

test('the bridge source inventory covers eight child inputs and the exact seven parent files', () => {
  const child = [
    'README.md',
    'check.py',
    'evidence-schema.json',
    'fixture.json',
    'recipe.json',
    'requirements.txt',
    'solution.py',
    'starter.py',
  ].map((name) => `projects/posttraining-audit/transformer_lineage/${name}`);
  const parents = [
    'projects/text-harness/foundations/config.py',
    'projects/text-harness/foundations/data.py',
    'projects/text-harness/foundations/model.py',
    'projects/text-harness/foundations/train.py',
    'projects/text-harness/foundations/reference/run/inference-package/package.json',
    'projects/text-harness/foundations/reference/run/inference-package/params.npz',
    'projects/text-harness/foundations/reference/run/inference-package/tokenizer.json',
  ];
  const sourceSet = new Set([...child, ...parents]);
  assert.equal(sourceSet.size, 15);
  for (const source of sourceSet)
    assert.ok(fs.statSync(source).isFile(), `missing bridge input: ${source}`);

  const builder = fs.readFileSync('scripts/build-distribution.py', 'utf8');
  const checker = fs.readFileSync('scripts/check-distribution.py', 'utf8');
  for (const parent of parents) {
    const suffix = parent.replace('projects/text-harness/foundations/', '');
    assert.ok(
      builder.includes(suffix) || builder.includes(parent),
      `builder does not include parent input ${parent}`,
    );
    assert.ok(
      checker.includes(suffix) || checker.includes(parent),
      `checker does not pin parent input ${parent}`,
    );
  }
  assert.match(builder, /posttraining_parent_files\(ROOT\)/);
  assert.match(checker, /check_posttraining_parent_bundle\(bundle, ROOT\)/);
});

test('the offline book routes the third appendix and keeps existing SVG project figures embeddable', () => {
  const reader = fs.readFileSync('src/pages/downloads/jax-foundations.astro', 'utf8');
  const renderer = fs.readFileSync('scripts/render-book-guides.mjs', 'utf8');
  const builder = fs.readFileSync('scripts/build-distribution.py', 'utf8');
  const checker = fs.readFileSync('scripts/check-distribution.py', 'utf8');

  assert.match(reader, /appendix-posttraining-lineage/);
  assert.match(reader, /transformerLineage[\s\S]*bookGuide/);
  assert.match(reader, /guideMarkup\([\s\S]*transformer_lineage\/TEACHING\.md/);
  assert.match(renderer, /posttrainingTransformerLineage[\s\S]*transformer_lineage\/TEACHING\.md/);
  assert.match(builder, /appendix-posttraining-lineage\.xhtml/);
  assert.match(builder, /appendix-posttraining-lineage[\s\S]*spine/);
  assert.match(builder, /lineage=book_guides\['posttrainingTransformerLineage'\]/);
  assert.match(checker, /appendix-posttraining-lineage/);

  const projectGuide = fs.readFileSync('projects/posttraining-audit/README.md', 'utf8');
  const renderedProject = bookGuide('projects/posttraining-audit/README.md', course, {
    mathOutput: 'mathml',
  });
  assert.match(projectGuide, /!\[[^\]]+\]\(figures\/gae-eos-timeout\.svg\)/);
  assert.match(renderedProject, /src="data:image\/svg\+xml;base64,/);
  assert.match(renderedProject, /optional appendix/i);
});

test('book guide canonicalizes the three Astro URL forms and EPUB relative paths', () => {
  const reader = fs.readFileSync('src/pages/downloads/jax-foundations.astro', 'utf8');
  assert.match(
    reader,
    /new URL\('\.\.\/\.\.\/\.\.\/content\/guides\/project-workflow\.md',[\s\S]*?\)\.pathname/,
  );
  assert.match(
    reader,
    /new URL\('\.\.\/\.\.\/\.\.\/projects\/reproducible-experiments\/README\.md',[\s\S]*?\)\s*\.pathname/,
  );
  assert.match(
    reader,
    /guideMarkup\(\s*transformerLineage,[\s\S]*?new URL\([\s\S]*?transformer_lineage\/TEACHING\.md',[\s\S]*?import\.meta\.url,[\s\S]*?\),/,
    'the HTML appendix passes a file URL directly to bookGuide',
  );

  const astroModuleUrl = pathToFileURL(path.resolve('src/pages/downloads/jax-foundations.astro'));
  const actualHtmlSources = [
    new URL('../../../content/guides/project-workflow.md', astroModuleUrl).pathname,
    new URL('../../../projects/reproducible-experiments/README.md', astroModuleUrl).pathname,
    new URL('../../../projects/posttraining-audit/transformer_lineage/TEACHING.md', astroModuleUrl),
  ];
  const relativeSources = [
    'content/guides/project-workflow.md',
    'projects/reproducible-experiments/README.md',
    sourcePath,
  ];
  for (const [index, actualSource] of actualHtmlSources.entries()) {
    const relative = bookGuide(relativeSources[index], course, { mathOutput: 'mathml' });
    const actual = bookGuide(actualSource, course, { mathOutput: 'mathml' });
    assert.equal(actual, relative, `Astro source form ${index + 1} matches EPUB relative source`);
    assert.doesNotMatch(actual, /\/private\/tmp\/jaxpathways-transformer-bridge-review/);
  }

  const absoluteSource = path.resolve(sourcePath);
  const relative = bookGuide(sourcePath, course, { mathOutput: 'mathml' });
  const absolute = bookGuide(absoluteSource, course, { mathOutput: 'mathml' });
  assert.equal(absolute, relative);
  assert.equal((absolute.match(/<figure class="print-visual-intro"/g) || []).length, 1);
  assert.match(absolute, /src="data:image\/svg\+xml;base64,/);
  assert.match(absolute, /project-guides\/posttraining-audit\.html/);
  assert.match(absolute, /lesson-[a-z0-9-]+\.html/);
  assert.throws(
    () => bookGuide(path.resolve('../outside/TEACHING.md'), course),
    /outside the repository/,
  );
  assert.throws(
    () => bookGuide(new URL('https://example.com/TEACHING.md'), course),
    /file protocol/,
  );
});

test('the appendix figure title, image and caption render as one accessible figure group', () => {
  const online = `<h3 id="part-9">Recorded v5 result</h3><p><a href="/project-assets/posttraining-audit/transformer_lineage/teaching/v5/figure.svg" aria-label="Open full-size figure" title="Open full-size figure"><img src="/project-assets/posttraining-audit/transformer_lineage/teaching/v5/figure.svg" alt="Heldout reward floor fails; restricted KL is below its ceiling." width="1000" height="620" loading="lazy" /></a></p><p><strong>Figure 1.</strong> The strict heldout reward improvement floor failed. Exact restricted conditional KL is measured over eligible actions at sampled child-visited contexts.</p>`;
  const offline = online
    .replace(/<a href="[^"]+"[^>]*>/, '')
    .replace('</a>', '')
    .replaceAll(
      '/project-assets/posttraining-audit/transformer_lineage/teaching/v5/figure.svg',
      'data:image/svg+xml;base64,PHN2Zz4=',
    );
  for (const rendered of [
    groupTransformerLineageFigure(online),
    groupTransformerLineageFigure(offline),
  ]) {
    const figure = rendered.match(/<figure class="print-visual-intro"[\s\S]*?<\/figure>/)?.[0];
    assert.ok(figure, 'figure title, image and caption should share one print group');
    assert.match(figure, /aria-labelledby="part-9"/);
    assert.match(figure, /<h3 id="part-9">Recorded v5 result<\/h3>/);
    assert.match(
      figure,
      /<div class="figure-scroll"[^>]*>(?:<a [^>]*>)?<img[^>]+alt="Heldout reward floor fails; restricted KL is below its ceiling\."/,
    );
    assert.match(
      figure,
      /<figcaption><strong>Figure 1\.<\/strong> The strict heldout reward improvement floor failed\./,
    );
    assert.ok(figure.indexOf('<figcaption>') > figure.indexOf('<img'));
    assert.ok(
      figure.endsWith('</figcaption></figure>'),
      'caption must remain the final figure child',
    );
    assert.equal((rendered.match(/<figure\b/g) || []).length, 1);
  }
  assert.throws(() => groupTransformerLineageFigure('<h3>Unrelated title</h3>'), /found 0/);
});
