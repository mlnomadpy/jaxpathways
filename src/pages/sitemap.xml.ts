import type { APIRoute } from 'astro';
import { authoredLessons, course } from '../lib/course-data.js';
import { currentRelease } from '../lib/releases.js';
import { siteUrl } from '../lib/urls.js';

// Derive static routes from the same source manifests as the pages; omit personal workspaces.
const pages = import.meta.glob('./**/*.astro');
import projectRegistry from '../../public/api/v1/projects.json';
import plans from '../../curriculum/project-plans.json';
import assessments from '../../curriculum/assessments.json';
import tracks from '../../curriculum/modality-tracks.json';
export const GET: APIRoute = ({ site }) => {
  const staticPaths = Object.keys(pages)
    .map((path) => path.slice(2).replace(/\.astro$/, '.html'))
    .filter(
      (path) =>
        !/[\[\]]/.test(path) &&
        !['lesson.html', 'project.html', 'notebook.html', 'organizer.html'].includes(path) &&
        !path.startsWith('downloads/'),
    );
  const urls = new Map(
    staticPaths.map((path) => [path, path === 'updates.html' ? currentRelease.date : undefined]),
  );
  for (const lesson of authoredLessons)
    urls.set(
      `lesson-${lesson.id}.html`,
      currentRelease.lessons[lesson.id as keyof typeof currentRelease.lessons]?.date,
    );
  for (const phase of course.phases)
    urls.set(
      `phase-guides/${phase.id}.html`,
      currentRelease.phases[phase.id as keyof typeof currentRelease.phases]?.date,
    );
  for (const project of projectRegistry.projects)
    urls.set(`project-guides/${project.id}.html`, undefined);
  for (const plan of plans) urls.set(`project-plans/${plan.id}.html`, undefined);
  for (const track of tracks.tracks) urls.set(`tracks/${track.id}.html`, undefined);
  for (const assessment of assessments) urls.set(assessment.url, undefined);
  const xml = (text: string) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${[...urls].map(([path, date]) => `<url><loc>${xml(new URL(siteUrl(path === 'index.html' ? '' : path), site).href)}</loc>${date ? `<lastmod>${date}</lastmod>` : ''}</url>`).join('\n')}</urlset>`,
    { headers: { 'Content-Type': 'application/xml; charset=utf-8' } },
  );
};
