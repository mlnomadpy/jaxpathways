import path from 'node:path';
import { existsSync } from 'node:fs';
import { assessmentContent } from './assessment-content.js';
import { lessonLink, siteUrl } from '../urls.js';
import projectPlans from '../../../curriculum/project-plans.json' with { type: 'json' };

/** Resolve repository Markdown links into the static course, including Pages bases. */
export function projectGuide(project, course, plannedIds = projectPlans.map((plan) => plan.id)) {
  return assessmentContent(project.source, (href) => {
    if (/^(https?:|#)/.test(href)) return href;
    const target = path.posix.normalize(path.posix.join(path.posix.dirname(project.source), href));
    if (target === 'projects/text-harness/foundations/README.md') {
      return siteUrl('project-guides/text-harness-foundations.html');
    }
    const projectReadme = /^projects\/([^/]+)\/README\.md$/.exec(target);
    if (projectReadme) {
      const directory = plannedIds.includes(projectReadme[1]) ? 'project-plans' : 'project-guides';
      return siteUrl(`${directory}/${projectReadme[1]}.html`);
    }
    const guide = /^content\/guides\/([^/]+)\.md$/.exec(target);
    if (guide) return siteUrl(`${guide[1]}.html`);
    const assessment = /^assessments\/([^/]+)\.md$/.exec(target);
    if (assessment) return siteUrl(`assessments/${assessment[1]}.html`);
    const phase = course.phases.find(
      (phase) => target === `phases/${phase.number}-${phase.id}/README.md`,
    );
    if (phase) return siteUrl(`phase-guides/${phase.id}.html`);
    const lesson = course.phases
      .flatMap((phase) => phase.lessons)
      .find((lesson) => target.startsWith(lesson.path + '/'));
    if (lesson) return siteUrl(lessonLink(lesson.id));
    const projectAsset = /^projects\/([^/]+)\/(.+)$/.exec(target);
    if (
      projectAsset &&
      existsSync(target) &&
      (projectAsset[1] === project.id ||
        plannedIds.includes(projectAsset[1]) ||
        course.projectIds.includes(projectAsset[1]))
    ) {
      return siteUrl(`project-assets/${projectAsset[1]}/${projectAsset[2]}`);
    }
    if (target.startsWith('docs/') && existsSync(target)) {
      return `https://github.com/mlnomadpy/jaxpathways/blob/main/${target.split('/').map(encodeURIComponent).join('/')}`;
    }
    throw new Error(`Unresolved project guide link: ${project.id}: ${href}`);
  });
}
