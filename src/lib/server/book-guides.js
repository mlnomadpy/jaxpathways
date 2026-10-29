import path from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { assessmentContent } from './assessment-content.js';
import { groupTransformerLineageFigure } from './posttraining-lineage-figure.js';

const canonical = (destination) =>
  new URL(destination, 'https://www.tahabouhsine.com/jaxpathways/').href;
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

function repositorySource(source) {
  let sourcePath = source;
  if (source instanceof URL) {
    if (source.protocol !== 'file:') {
      throw new Error(`Book guide source URL must use file protocol: ${source.protocol}`);
    }
    sourcePath = fileURLToPath(source);
  }
  if (typeof sourcePath !== 'string') {
    throw new TypeError('Book guide source must be a path string or file URL');
  }
  const absolute = path.isAbsolute(sourcePath)
    ? path.resolve(sourcePath)
    : path.resolve(repositoryRoot, sourcePath);
  const relative = path.relative(repositoryRoot, absolute);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`Book guide source is outside the repository: ${source}`);
  }
  return relative.split(path.sep).join('/');
}

/** Keep the same Markdown/math semantics in HTML and EPUB appendices. */
export function bookGuide(source, course, { mathOutput = 'htmlAndMathml' } = {}) {
  const sourcePath = repositorySource(source);
  const resolveLink = (href) => {
    if (/^(?:https?:|mailto:|data:|#)/i.test(href)) return href;
    const target = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), href));
    if (/\.(png|svg|jpe?g)$/.test(href)) {
      const physicalTarget = path.resolve(repositoryRoot, target);
      const relativeTarget = path.relative(repositoryRoot, physicalTarget);
      if (relativeTarget === '..' || relativeTarget.startsWith(`..${path.sep}`))
        throw new Error(`Book guide asset is outside the repository: ${href}`);
      const mime = href.endsWith('.svg')
        ? 'image/svg+xml'
        : href.endsWith('.png')
          ? 'image/png'
          : 'image/jpeg';
      return `data:${mime};base64,${readFileSync(physicalTarget).toString('base64')}`;
    }
    const lesson = course.phases
      .flatMap((phase) => phase.lessons)
      .find((item) => target.startsWith(item.path + '/'));
    if (lesson) return canonical(`lesson-${lesson.id}.html`);
    const project = /^projects\/([^/]+)\/README\.md$/.exec(target);
    if (project) return canonical(`project-guides/${project[1]}.html`);
    if (
      /^(?:lesson(?:-[a-z0-9-]+)?|phase-guides|course|pathways|project|tracks|project-guides|downloads|developer|assessments|project-workflow)(?:\/|\.|\?)/.test(
        href,
      )
    )
      return canonical(href);
    if (target.startsWith('content/guides/') && target.endsWith('.md'))
      return canonical(path.posix.basename(target, '.md') + '.html');
    return `https://github.com/mlnomadpy/jaxpathways/blob/main/${target.split('/').map(encodeURIComponent).join('/')}`;
  };
  const rendered = assessmentContent(
    path.resolve(repositoryRoot, sourcePath),
    resolveLink,
    mathOutput,
  );
  const prefix = sourcePath.replace(/[^a-z0-9]/gi, '-');
  const prefixed = (rendered.heading + rendered.body)
    .replace(/id="part-(\d+)"/g, `id="${prefix}-part-$1"`)
    .replace(/href="#part-(\d+)"/g, `href="#${prefix}-part-$1"`);
  return sourcePath === 'projects/posttraining-audit/transformer_lineage/TEACHING.md'
    ? groupTransformerLineageFigure(prefixed, { required: true })
    : prefixed;
}
