/**
 * Resolve site-local links against Astro's deployment base, preserving query/hash.
 * @param {string} [path]
 */
export function siteUrl(path = '') {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;
  const base = (typeof import.meta.env !== 'undefined' && import.meta.env.BASE_URL) || '/';
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}

/**
 * Permanent, pre-rendered lesson pages; the old query reader remains a compatibility entry.
 * @param {string} lessonId
 * @param {string} [pathwayId]
 */
export function lessonLink(lessonId, pathwayId = 'all') {
  return `lesson-${encodeURIComponent(lessonId)}.html${pathwayId === 'all' ? '' : '?path=' + encodeURIComponent(pathwayId)}`;
}

/**
 * Keep pathway, search and saved section when opening a historical reader link.
 * @param {string} lessonId
 * @param {string | URL} href
 */
export function legacyLessonUrl(lessonId, href) {
  const old = new URL(href);
  const next = new URL(`lesson-${encodeURIComponent(lessonId)}.html`, old);
  old.searchParams.delete('lesson');
  next.search = old.searchParams.toString();
  next.hash = old.hash;
  return next.href;
}
