import { careerCoverage } from '../careers.js';
import { escapeHtml } from '../html.js';

export function renderDomains(course) {
  const availability = (ids) => careerCoverage(ids, course);
  const labels = {
    training: 'Train a model',
    inference: 'Deploy a model',
    ops: 'Scale and operate workloads',
    research: 'Explore scientific computing',
  };
  return course.domains
    .map((d) => {
      const route = course.pathways.find(
          (r) => r.id === (d.id === 'training' ? 'models' : d.defaultPathwayId),
        ),
        a = availability(route.phaseIds);
      return `<article class="domain-card">
    <h3>${escapeHtml(labels[d.id] || d.title)}</h3>
    <p>${escapeHtml(d.description)}</p>
    <p class="availability-label">${a.planned ? 'Partly available' : 'Available lesson drafts'} · ${a.available} lessons</p>
    <a class="button" href="pathways.html?path=${encodeURIComponent(route.id)}">Explore path</a>
    </article>`;
    })
    .join('');
}
