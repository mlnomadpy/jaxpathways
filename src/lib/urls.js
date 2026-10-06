/** Resolve site-local links against Astro's deployment base, preserving query/hash. */
export function siteUrl(path = '') {
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(path)) return path;
  const base = import.meta.env?.BASE_URL || '/';
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}

/** Existing top-level lesson URLs stay stable across local and Pages deployments. */
export function lessonLink(lessonId, pathwayId = 'all') {
  return `lesson.html?lesson=${encodeURIComponent(lessonId)}${pathwayId === 'all' ? '' : '&path=' + encodeURIComponent(pathwayId)}`;
}
