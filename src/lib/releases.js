import snapshots from '../../curriculum/release-snapshots.json' with { type: 'json' };
import { escapeHtml } from './html.js';
import { siteUrl } from './urls.js';
import repository from '../data/repository.json' with { type: 'json' };
export const currentRelease = snapshots[0];
export const revisionKey = 'jaxpathways-revisions-v1';
export const releaseKey = 'jaxpathways-release-v1';
export const feedbackAccess =
  repository.visibility === 'public'
    ? ''
    : 'The repository is currently private; GitHub issue access is limited to collaborators.';

export function revisionMarkup(kind, id) {
  const revision = currentRelease[kind]?.[id];
  if (!revision) return '';
  return `<p class="revision-stamp" data-revision-key="${escapeHtml(kind + ':' + id)}" data-revision-hash="${revision.hash}"><a href="${siteUrl('updates.html')}#v${revision.version}">Revision v${revision.version} · <time datetime="${revision.date}">${revision.date}</time></a><span data-revision-notice hidden>Updated since your last visit. Review the changes; your progress is saved.</span></p>`;
}

export function feedbackUrl(title = 'Course feedback', page = '') {
  const query = new URLSearchParams({
    template: 'course-feedback.yml',
    title: `[Feedback] ${title}`,
    context: `${page ? new URL(siteUrl(page), 'https://www.tahabouhsine.com').href : title}\nCourse release: v${currentRelease.version}`,
  });
  return `https://github.com/mlnomadpy/jaxpathways/issues/new?${query}`;
}
