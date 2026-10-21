import releaseNotes from '../../curriculum/releases.json' with { type: 'json' };
import snapshots from '../../curriculum/release-snapshots.json' with { type: 'json' };
import { escapeHtml } from './html.js';
import { siteUrl } from './urls.js';
import repository from '../data/repository.json' with { type: 'json' };
import { releaseKey, revisionKey } from './release-keys.js';
export const currentRelease = snapshots[0];
export const latestReleaseNote = releaseNotes[0];
export { releaseKey, revisionKey };
export const feedbackAccess =
  repository.visibility === 'public'
    ? ''
    : 'The repository is currently private; GitHub issue access is limited to collaborators.';

/** @param {string} [version] */
export function editionUrl(version = currentRelease.version) {
  return `${siteUrl('updates.html')}#v${encodeURIComponent(version)}`;
}

/**
 * @param {'lessons' | 'phases'} kind
 * @param {string} id
 */
export function revisionMarkup(kind, id) {
  const collection =
    /** @type {Record<string, { hash: string; version: string; date: string }>} */ (
      currentRelease[kind]
    );
  const revision = collection?.[id];
  if (!revision) return '';
  return `<p class="revision-stamp" data-revision-key="${escapeHtml(kind + ':' + id)}" data-revision-hash="${revision.hash}"><a href="${editionUrl(revision.version)}">Revision v${revision.version} · <time datetime="${revision.date}">${revision.date}</time></a><span data-revision-notice hidden>Updated since your last visit. Review the changes; your progress is saved.</span></p>`;
}

/**
 * @param {string} [title]
 * @param {string} [page]
 */
export function feedbackUrl(title = 'Course feedback', page = '') {
  const query = new URLSearchParams({
    template: 'course-feedback.yml',
    title: `[Feedback] ${title}`,
    context: `${page ? new URL(siteUrl(page), 'https://www.tahabouhsine.com').href : title}\nCourse release: v${currentRelease.version}`,
  });
  return `https://github.com/mlnomadpy/jaxpathways/issues/new?${query}`;
}
