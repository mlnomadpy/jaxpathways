import { careerCoverage } from '../careers.js';
import { escapeHtml } from '../html.js';
import { inlineMath } from '../math.js';

/** @param {import('../../types/course').Course} course */
export function renderDomains(course) {
  const availability = (/** @type {string[]} */ ids) => careerCoverage(ids, course);
  /** @type {Record<string, string>} */
  const labels = {
    training: 'Train a model',
    inference: 'Deploy a model',
    ops: 'Scale and operate workloads',
    research: 'Explore scientific computing',
  };
  return course.domains
    .map((d) => {
      const route =
          course.pathways.find(
            (r) => r.id === (d.id === 'training' ? 'models' : d.defaultPathwayId),
          ) || course.pathways[0],
        a = availability(route.phaseIds);
      return `<article class="domain-card">
    <h3>${escapeHtml(labels[d.id] || d.title)}</h3>
    <p>${inlineMath(d.description)}</p>
    <p class="availability-label">${a.planned ? 'Partly available' : 'Available lesson drafts'} · ${a.available} lessons</p>
    <a class="button" href="pathways.html?path=${encodeURIComponent(route.id)}">Explore path</a>
    </article>`;
    })
    .join('');
}
