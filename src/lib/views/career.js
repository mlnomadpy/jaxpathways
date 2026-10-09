import { escapeHtml } from '../html.js';
import { inlineMath } from '../math.js';
import { routeViewContext } from './routes.js';

/**
 * @param {import('../../types/course').Role} role
 * @param {import('../../types/course').Course} course
 */
export function renderCareer(role, course) {
  const { phaseById, routeName, availabilityText, trackLinks, engineeringLinks, focusSummary } =
    routeViewContext(course);
  const route =
    course.pathways.find((route) => route.id === role.defaultPathwayId) || course.pathways[0];
  const capstone = route.capstone;
  return `<div class="career-heading">
    <div>
    <p class="page-eyebrow"><button type="button" id="close-career" class="career-link" aria-label="Back to career routes">All career routes</button><span aria-hidden="true"> · </span><span>Career route</span></p>
    <h3>${escapeHtml(role.title)}</h3>
    </div>
    </div>
    <p>${inlineMath(role.description)}</p>
    <h4>A problem you might work on</h4>
    <p>${inlineMath(role.workExample)}</p>${focusSummary(route)}${trackLinks(role.modalityTrackIds)}${engineeringLinks(route)}<p class="route-availability">${availabilityText(route.phaseIds)}. Counts include shared foundations.</p>
    <div class="role-columns">
    <div>
    <h4>Skills you’ll connect</h4>
    <ul>${role.skills.map((s) => `<li>${inlineMath(s)}</li>`).join('')}</ul>
    <h4>Prerequisites</h4>
    <p>${inlineMath(role.background)}</p>
    </div>
    <div>
    <h4>Your portfolio project</h4>
    <p>${inlineMath(role.portfolio)}</p>${
      capstone.projectId
        ? `<p>
    <a href="project.html?id=${encodeURIComponent(capstone.projectId)}">Open the guided portfolio project</a>
    </p>`
        : ''
    }${
      capstone.assessmentDraft
        ? `<p>
    <a href="${escapeHtml(capstone.assessmentDraft.url)}">Review the synthesis assessment</a>
    </p>`
        : ''
    }<h4>Demonstrate the capability</h4>
    <p>${inlineMath(role.readiness)}</p>
    </div>
    </div>
    <h4>Questions to defend in a review</h4>
    <ul>${role.reviewQuestions.map((q) => `<li>${inlineMath(q)}</li>`).join('')}</ul>
    <p class="soft">${inlineMath(role.scopeNote)}</p>
    <h4 class="career-subhead">Your route in milestones</h4>
    <ol class="career-milestones">${role.milestones
      .map(
        (m) => `<li>
    <h4>${inlineMath(m.title)}</h4>
    <p>${inlineMath(m.deliverable)}</p>
    <p class="milestone-availability">${availabilityText(m.phaseIds)}</p>
    <div>${m.phaseIds.map((id) => `<button data-phase="${id}" class="phase-chip">${phaseById(id).number} ${escapeHtml(phaseById(id).title)}</button>`).join('')}</div>
    <details>
    <summary>What to demonstrate</summary>
    <ul>${m.criteria.map((s) => `<li>${inlineMath(s)}</li>`).join('')}</ul>
    </details>
    </li>`,
      )
      .join('')}</ol>
    <details class="starting-check">
    <summary>Find my starting point</summary>
    <p>Check what you can already explain and demonstrate. This suggests where to begin; it is not a graded assessment.</p>
    <div class="starting-checks">${role.startingPointChecks
      .map(
        (check) => `<label>
    <input type="checkbox" value="${check.phaseId}">${inlineMath(check.question)}</label>`,
      )
      .join('')}</div>
    </details>
    <p id="career-start-note" class="career-start-note" aria-live="polite"></p>
    <div class="career-actions">
    <button id="career-start" class="primary">Start this career route</button>
    <button id="career-save">Save my plan</button>
    <button id="career-download">Download plan</button>
    <button id="career-share">Copy route link</button>
    </div>
    <p id="career-status" class="soft" role="status"></p>${
      role.pathwayIds.length > 1
        ? `<p class="soft">Other route options:</p>
    <div class="actions">${role.pathwayIds
      .filter((id) => id !== role.defaultPathwayId)
      .map((id) => `<button data-choice="${id}">${escapeHtml(routeName(id))}</button>`)
      .join('')}</div>`
        : ''
    }<p class="soft">Milestone availability is shown above. Portfolio briefs without a linked staged project remain planned. Public checks and self-reported work do not establish reviewed readiness.</p>`;
}
