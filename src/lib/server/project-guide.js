import path from 'node:path';
import { assessmentContent } from './assessment-content.js';
import { siteUrl } from '../urls.js';

/** Resolve repository Markdown links into the static course, including Pages bases. */
export function projectGuide(project, course, plannedIds = []) {
  return assessmentContent(project.source, (href) => {
    if (/^(https?:|#)/.test(href)) return href;
    const target = path.posix.normalize(path.posix.join(path.posix.dirname(project.source), href));
    const projectReadme = /^projects\/([^/]+)\/README\.md$/.exec(target);
    if (projectReadme) {
      const directory = plannedIds.includes(projectReadme[1]) ? 'project-plans' : 'project-guides';
      return siteUrl(`${directory}/${projectReadme[1]}.html`);
    }
    const assessment = /^assessments\/([^/]+)\.md$/.exec(target);
    if (assessment) return siteUrl(`assessments/${assessment[1]}.html`);
    const phase = course.phases.find(
      (phase) => target === `phases/${phase.number}-${phase.id}/README.md`,
    );
    if (phase) return siteUrl(`course.html#phase-${phase.id}`);
    const lesson = course.phases
      .flatMap((phase) => phase.lessons)
      .find((lesson) => target.startsWith(lesson.path + '/'));
    if (lesson) return siteUrl(`lesson.html?lesson=${lesson.id}`);
    if (target.startsWith(`projects/${project.id}/`)) {
      return siteUrl(
        `project-assets/${project.id}/${target.slice(`projects/${project.id}/`.length)}`,
      );
    }
    throw new Error(`Unresolved project guide link: ${project.id}: ${href}`);
  });
}
