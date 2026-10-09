import { escapeHtml } from '../html.js';
import { inlineMath } from '../math.js';
import { courseTypeCue } from '../runtime.js';
import { lessonLink } from '../urls.js';
import { routeViewContext } from './routes.js';

/**
 * @param {import('../../types/course').Pathway} route
 * @param {import('../../types/course').Course} course
 * @param {boolean} hasNext
 */
export function renderPathway(route, course, hasNext) {
  const id = route.id;
  const { phaseById, sources, availability, trackLinks, engineeringLinks, focusSummary } =
    routeViewContext(course);
  const a = availability(route.phaseIds),
    projectId =
      route.capstone.projectId ||
      route.phaseIds
        .map((id) => phaseById(id).projectId)
        .filter(Boolean)
        .at(-1);
  return `<p class="page-eyebrow"><a href="pathways.html" class="back-link">All learning paths</a><span aria-hidden="true"> · </span><span>Learning path</span></p>
    <div class="path-detail-heading">
    <div>
    <h1 tabindex="-1">${escapeHtml(route.title)}</h1>
    <p class="intro">${inlineMath(route.description)}</p>
    <p class="availability-label">${a.planned ? 'Partly available' : 'Available lesson drafts'} · ${a.available} lessons${a.planned ? ` · ${a.planned} planned ${a.planned === 1 ? 'topic' : 'topics'}` : ''}</p>
    <p>${inlineMath(route.prerequisites)}</p>
    </div>
    <div class="path-outcome">
    <span class="branch-glyph" aria-hidden="true">↳</span>
    <h2>${escapeHtml(route.capstone.title)}</h2>
    <p>${route.capstone.projectId ? 'Project workspace available' : 'Final project is planned. Available preparation is shown below.'}</p>${projectId ? `<a href="project.html?id=${encodeURIComponent(projectId)}">Inspect available project</a>` : ''}</div>
    </div>
    <div class="actions">${hasNext ? '<button type="button" id="use-path" class="primary">Use this path & start</button>' : ''}<a class="button" href="course.html?path=${encodeURIComponent(id)}">Browse lessons</a>
    </div>
    ${route.practicalGuide ? `<p><a class="button" href="${escapeHtml(route.practicalGuide.url)}">${escapeHtml(route.practicalGuide.title)}</a></p><p>${inlineMath(route.practicalGuide.scope)}</p>` : ''}
    ${(route.additionalGuides || []).map((guide) => `<p><a class="button" href="${escapeHtml(guide.url)}">${escapeHtml(guide.title)}</a></p><p>${inlineMath(guide.scope)}</p>`).join('')}
    <p id="path-save-status" role="status"></p>${focusSummary(route)}<div class="role-columns">
    <div>
    <h2>Who this path is for</h2>
    <p>${inlineMath(route.audience)}</p>
    <h3>Your first useful artifact</h3>
    <p>${inlineMath(route.firstArtifact)}</p>
    </div>
    <div>
    <h2>How to study</h2>
    <p>${inlineMath(route.studyAdvice)}</p>${trackLinks(route.modalityTrackIds)}</div>
    </div>${engineeringLinks(route)}<h2>Your journey</h2>
    <ol class="path-journey">${route.phaseIds
      .map((phaseId) => {
        const p = phaseById(phaseId),
          cue = courseTypeCue(p),
          a = availability([phaseId]);
        return `<li>
    <a href="course.html?path=${encodeURIComponent(id)}&phase=${encodeURIComponent(phaseId)}${a.available ? '' : '&roadmap=1'}">
    <span class="course-cue-row"><span class="course-type-pill" data-course-kind="${escapeHtml(cue.kind)}">${escapeHtml(cue.badge)}</span><span class="course-meta-pill">${escapeHtml(cue.paceLabel)}</span></span>
    <strong>${escapeHtml(p.number)} · ${escapeHtml(p.title)}</strong>
    <span>${a.available} available drafts${a.planned ? ' / ' + a.planned + ' planned' : ''}</span>
    </a>
    </li>`;
      })
      .join('')}</ol>
    <h2>Evidence to collect</h2>
    <ul>${route.capstone.evidence.map((item) => `<li>${inlineMath(item)}</li>`).join('')}</ul>${
      route.capstone.projectId
        ? `<p>
    <a class="button" href="project.html?id=${encodeURIComponent(route.capstone.projectId)}">Open final project</a>
    </p>`
        : ''
    }${
      route.capstone.assessmentDraft
        ? `<p>
    <a href="${escapeHtml(route.capstone.assessmentDraft.url)}">Synthesis assessment · review draft</a>
    </p>`
        : ''
    }${(route.assessmentExtensions || [])
      .map(
        (a) => `<p>
    <a href="${escapeHtml(a.url)}">${escapeHtml(a.title)} · review draft</a>
    </p>`,
      )
      .join('')}<details>
    <summary>Choose a different starting step</summary>
    <p>Start where you need practice. Earlier steps are still available; skipping does not mark them complete.</p>
    <div class="starting-steps">${route.phaseIds
      .map((phaseId) => {
        const p = phaseById(phaseId),
          l = p.lessons.find((l) => l.status === 'authored');
        return l ? `<a href="${lessonLink(l.id, id)}">${escapeHtml(p.title)}</a>` : '';
      })
      .join('')}</div>
    </details>
    <details>
    <summary>Source and project expectations</summary>
    <p>${inlineMath(route.capstone.assessment)}</p>
    <a href="${sources(route)}">Path source on GitHub</a>
    </details>`;
}
