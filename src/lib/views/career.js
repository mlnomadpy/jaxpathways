import { escapeHtml } from '../html.js';
import { routeViewContext } from './routes.js';

export function renderCareer(role, course) {
  const { phaseById, routeName, availabilityText, trackLinks, engineeringLinks, focusSummary } =
    routeViewContext(course);
  const capstone = course.pathways.find((route) => route.id === role.defaultPathwayId).capstone;
  return `<div class="career-heading">
    <div>
    <p class="edition">Career route</p>
    <h3>${escapeHtml(role.title)}</h3>
    </div>
    <button id="close-career" aria-label="Back to career routes">All career routes</button>
    </div>
    <p>${escapeHtml(role.description)}</p>
    <h4>A problem you might work on</h4>
    <p>${escapeHtml(role.workExample)}</p>${focusSummary(course.pathways.find((r) => r.id === role.defaultPathwayId))}${trackLinks(role.modalityTrackIds)}${engineeringLinks(course.pathways.find((r) => r.id === role.defaultPathwayId))}<p class="route-availability">${availabilityText(course.pathways.find((r) => r.id === role.defaultPathwayId).phaseIds)}. Counts include shared foundations.</p>
    <div class="role-columns">
    <div>
    <h4>Skills you’ll connect</h4>
    <ul>${role.skills.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
    <h4>Prerequisites</h4>
    <p>${escapeHtml(role.background)}</p>
    </div>
    <div>
    <h4>Your portfolio project</h4>
    <p>${escapeHtml(role.portfolio)}</p>${
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
    <p>${escapeHtml(role.readiness)}</p>
    </div>
    </div>
    <h4>Questions to defend in a review</h4>
    <ul>${role.reviewQuestions.map((q) => `<li>${escapeHtml(q)}</li>`).join('')}</ul>
    <p class="soft">${escapeHtml(role.scopeNote)}</p>
    <h4 class="career-subhead">Your route in milestones</h4>
    <ol class="career-milestones">${role.milestones
      .map(
        (m) => `<li>
    <h4>${escapeHtml(m.title)}</h4>
    <p>${escapeHtml(m.deliverable)}</p>
    <p class="milestone-availability">${availabilityText(m.phaseIds)}</p>
    <div>${m.phaseIds.map((id) => `<button data-phase="${id}" class="phase-chip">${phaseById(id).number} ${escapeHtml(phaseById(id).title)}</button>`).join('')}</div>
    <details>
    <summary>What to demonstrate</summary>
    <ul>${m.criteria.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
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
    <input type="checkbox" value="${check.phaseId}">${escapeHtml(check.question)}</label>`,
      )
      .join('')}</div>
    </details>
    <p id="career-start-note" class="career-start-note" aria-live="polite">
    </p>
    <div class="career-actions">
    <button id="career-start" class="primary">Start this career route</button>
    <button id="career-save">Save my plan</button>
    <button id="career-download">Download plan</button>
    <button id="career-share">Copy route link</button>
    </div>
    <p id="career-status" class="soft" role="status">
    </p>${
      role.pathwayIds.length > 1
        ? `<p class="soft">Other route options:</p>
    <div class="actions">${role.pathwayIds
      .filter((id) => id !== role.defaultPathwayId)
      .map((id) => `<button data-choice="${id}">${escapeHtml(routeName(id))}</button>`)
      .join('')}</div>`
        : ''
    }<p class="soft">Milestone availability is shown above. Portfolio briefs without a linked staged project remain planned. Public checks and self-reported work do not establish reviewed readiness.</p>`;
}
