import path from 'node:path';
import { readFileSync } from 'node:fs';
import { assessmentContent } from './assessment-content.js';

const canonical = (destination) =>
  new URL(destination, 'https://www.tahabouhsine.com/jaxpathways/').href;

/** Keep the same Markdown/math semantics in HTML and EPUB appendices. */
export function bookGuide(source, course, { mathOutput = 'htmlAndMathml' } = {}) {
  const resolveLink = (href) => {
    if (/^(?:https?:|mailto:|data:|#)/i.test(href)) return href;
    const target = path.posix.normalize(path.posix.join(path.posix.dirname(source), href));
    if (/\.(png|svg|jpe?g)$/.test(href)) {
      const mime = href.endsWith('.svg')
        ? 'image/svg+xml'
        : href.endsWith('.png')
          ? 'image/png'
          : 'image/jpeg';
      return `data:${mime};base64,${readFileSync(target).toString('base64')}`;
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
  const rendered = assessmentContent(source, resolveLink, mathOutput);
  const prefix = source.replace(/[^a-z0-9]/gi, '-');
  return (rendered.heading + rendered.body)
    .replace(/id="part-(\d+)"/g, `id="${prefix}-part-$1"`)
    .replace(/href="#part-(\d+)"/g, `href="#${prefix}-part-$1"`);
}
