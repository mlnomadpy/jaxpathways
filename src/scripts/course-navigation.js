import { siteUrl } from '../lib/urls.js';

/** Navigate from other features without importing the catalog's DOM controller. */
export function navigateToCourse({ path, phase, roadmap = false } = {}) {
  const params = new URLSearchParams();
  if (path) params.set('path', path);
  if (phase) params.set('phase', phase);
  if (roadmap) params.set('roadmap', '1');
  const query = params.toString();
  location.href = siteUrl(`course.html${query ? '?' + query : ''}#curriculum`);
}
