import { courseState, phaseById } from '../lib/course-state.js';
import {
  careerCoverage,
  pathwayCoverage,
  careerStartingPhase,
  careerPlanMarkdown,
} from '../lib/careers.js';
import { $ } from '../lib/dom.js';
import { escapeHtml } from '../lib/html.js';
import { openRoute } from './catalog.js';

function setupChoices() {
  const routeName = (id) => courseState.course.pathways.find((r) => r.id === id).title;
  const sources = (route) =>
    `https://github.com/mlnomadpy/jaxpathways/blob/main/learning-paths/${route.id}.json`;
  const availability = (ids) => careerCoverage(ids, courseState.course);
  const availabilityText = (ids) => {
    const a = availability(ids);
    return `${a.available} available lesson${a.available === 1 ? '' : 's'} · ${a.planned} unwritten · ${a.projects.length} staged project${a.projects.length === 1 ? '' : 's'}`;
  };
  const trackLinks = (ids = []) =>
    ids.length
      ? `<p>Apply this work in a guided project plan:</p><div class="actions">${ids.map((id) => `<a class="button" href="tracks/${encodeURIComponent(id)}.html">${escapeHtml(id)} guide</a>`).join('')}</div>`
      : '';
  const engineeringLinks = (route) => {
    const ids = route.engineeringLessonIds || [];
    if (!ids.length) return '';
    const lessons = courseState.course.phases.flatMap((phase) => phase.lessons);
    return `<h4>Engineering skills to connect</h4><p>Follow the listed lesson prerequisites when extending your core route.</p><ul>${ids.map((id) => `<li><a href="lesson.html?lesson=${encodeURIComponent(id)}">${escapeHtml(lessons.find((lesson) => lesson.id === id).title)}</a></li>`).join('')}</ul><p><a href="project-guides/engineering-release.html">Build a tracked and containerized release</a></p>`;
  };
  const focusSummary = (route) => {
    const { focus, preparation } = pathwayCoverage(route, courseState.course);
    return `<p class="route-availability">Focus phases: ${focus.available} available lessons · ${focus.planned} planned. Preparation: ${preparation.available} available · ${preparation.planned} planned.</p>`;
  };
  if ($('#domain-cards')) {
    const labels = {
      training: 'Train a model',
      inference: 'Deploy a model',
      ops: 'Scale and operate workloads',
      research: 'Explore scientific computing',
    };
    $('#domain-cards').innerHTML = courseState.course.domains
      .map((d) => {
        const route = courseState.course.pathways.find(
            (r) => r.id === (d.id === 'training' ? 'models' : d.defaultPathwayId),
          ),
          a = availability(route.phaseIds);
        return `<article class="domain-card"><h3>${escapeHtml(labels[d.id] || d.title)}</h3><p>${escapeHtml(d.description)}</p><p class="availability-label">${a.planned ? 'Partly available' : 'Available lesson drafts'} · ${a.available} lessons</p><a class="button" href="pathways.html?path=${encodeURIComponent(route.id)}">Explore path</a></article>`;
      })
      .join('');
  }
  if ($('#role-detail')) {
    if ($('#career-links'))
      $('#career-links').innerHTML = courseState.course.roles
        .map((r) => `<button data-role="${r.id}">${escapeHtml(r.title)}</button>`)
        .join('');
    function showRole(id) {
      const role = courseState.course.roles.find((r) => r.id === id);
      if (!role) return;
      const detail = $('#role-detail');
      const capstone = courseState.course.pathways.find(
        (r) => r.id === role.defaultPathwayId,
      ).capstone;
      detail.hidden = false;
      if ($('#role-cards')) document.body.classList.add('career-open');
      document
        .querySelectorAll('#career-links [data-role],#role-cards [data-role]')
        .forEach((button) =>
          button.setAttribute('aria-expanded', String(button.dataset.role === id)),
        );
      detail.innerHTML = `<div class="career-heading"><div><p class="edition">Career route</p><h3>${escapeHtml(role.title)}</h3></div><button id="close-career" aria-label="Back to career routes">All career routes</button></div><p>${escapeHtml(role.description)}</p><h4>A problem you might work on</h4><p>${escapeHtml(role.workExample)}</p>${focusSummary(courseState.course.pathways.find((r) => r.id === role.defaultPathwayId))}${trackLinks(role.modalityTrackIds)}${engineeringLinks(courseState.course.pathways.find((r) => r.id === role.defaultPathwayId))}<p class="route-availability">${availabilityText(courseState.course.pathways.find((r) => r.id === role.defaultPathwayId).phaseIds)}. Counts include shared foundations.</p><div class="role-columns"><div><h4>Skills you’ll connect</h4><ul>${role.skills.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}</ul><h4>Prerequisites</h4><p>${escapeHtml(role.background)}</p></div><div><h4>Your portfolio project</h4><p>${escapeHtml(role.portfolio)}</p>${capstone.projectId ? `<p><a href="project.html?id=${encodeURIComponent(capstone.projectId)}">Open the guided portfolio project</a></p>` : ''}${capstone.assessmentDraft ? `<p><a href="${escapeHtml(capstone.assessmentDraft.url)}">Review the synthesis assessment</a></p>` : ''}<h4>Demonstrate the capability</h4><p>${escapeHtml(role.readiness)}</p></div></div><h4>Questions to defend in a review</h4><ul>${role.reviewQuestions.map((q) => `<li>${escapeHtml(q)}</li>`).join('')}</ul><p class="soft">${escapeHtml(role.scopeNote)}</p><h4 class="career-subhead">Your route in milestones</h4><ol class="career-milestones">${role.milestones.map((m) => `<li><h4>${escapeHtml(m.title)}</h4><p>${escapeHtml(m.deliverable)}</p><p class="milestone-availability">${availabilityText(m.phaseIds)}</p><div>${m.phaseIds.map((id) => `<button data-phase="${id}" class="phase-chip">${phaseById(id).number} ${escapeHtml(phaseById(id).title)}</button>`).join('')}</div><details><summary>What to demonstrate</summary><ul>${m.criteria.map((s) => `<li>${escapeHtml(s)}</li>`).join('')}</ul></details></li>`).join('')}</ol><details class="starting-check"><summary>Find my starting point</summary><p>Check what you can already explain and demonstrate. This suggests where to begin; it is not a graded assessment.</p><div class="starting-checks">${role.startingPointChecks.map((check) => `<label><input type="checkbox" value="${check.phaseId}">${escapeHtml(check.question)}</label>`).join('')}</div></details><p id="career-start-note" class="career-start-note" aria-live="polite"></p><div class="career-actions"><button id="career-start" class="primary">Start this career route</button><button id="career-save">Save my plan</button><button id="career-download">Download plan</button><button id="career-share">Copy route link</button></div><p id="career-status" class="soft" role="status"></p>${
        role.pathwayIds.length > 1
          ? `<p class="soft">Other route options:</p><div class="actions">${role.pathwayIds
              .filter((id) => id !== role.defaultPathwayId)
              .map((id) => `<button data-choice="${id}">${escapeHtml(routeName(id))}</button>`)
              .join('')}</div>`
          : ''
      }<p class="soft">Milestone availability is shown above. Portfolio briefs without a linked staged project remain planned. Public checks and self-reported work do not establish reviewed readiness.</p>`;
      if ($('#domain-detail')) $('#domain-detail').hidden = true;
      const roleUrl = new URL(location.href);
      roleUrl.searchParams.set('role', role.id);
      if (roleUrl.href !== location.href) history.pushState(null, '', roleUrl);
      const heading = detail.querySelector('h3');
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
      let known = [];
      try {
        const saved = JSON.parse(localStorage.getItem('jaxpathways-career-plan-v1'));
        if (saved?.roleId === role.id && Array.isArray(saved.knownPhaseIds))
          known = saved.knownPhaseIds;
      } catch {}
      detail
        .querySelectorAll('.starting-checks input')
        .forEach((input) => (input.checked = known.includes(input.value)));
      const selectedPhase = () =>
        careerStartingPhase(
          role,
          [...detail.querySelectorAll('.starting-checks input:checked')].map(
            (input) => input.value,
          ),
          courseState.course,
        );
      const updateStart = () => {
        const phase = phaseById(selectedPhase()),
          lesson = phase.lessons.find((l) => l.status === 'authored');
        $('#career-start-note').textContent = lesson
          ? 'Suggested first lesson: ' +
            lesson.title +
            '. Earlier prerequisites remain available for review.'
          : 'This starting phase is unwritten. View its roadmap and choose available preparation.';
        $('#career-start').textContent = lesson
          ? 'Start: ' + lesson.title
          : 'View this planned milestone';
      };
      detail
        .querySelectorAll('.starting-checks input')
        .forEach((input) => (input.onchange = updateStart));
      updateStart();
      $('#close-career').onclick = () => {
        detail.hidden = true;
        document.body.classList.remove('career-open');
        const url = new URL(location.href);
        url.searchParams.delete('role');
        history.pushState(null, '', url);
        if ($('#career-browser')?.tagName === 'DETAILS') $('#career-browser').open = true;
        document
          .querySelectorAll('[data-role]')
          .forEach((button) => button.setAttribute('aria-expanded', 'false'));
        const origin = document.querySelector(`[data-role="${role.id}"]`);
        if (origin) origin.focus();
      };
      $('#career-start').onclick = () => {
        const phase = phaseById(selectedPhase());
        const lesson = phase.lessons.find((l) => l.status === 'authored');
        if (lesson) {
          location.href = `lesson.html?path=${encodeURIComponent(role.defaultPathwayId)}&lesson=${encodeURIComponent(lesson.id)}`;
        } else {
          location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(phase.id)}&roadmap=1#curriculum`;
        }
      };
      $('#career-save').onclick = () => {
        try {
          localStorage.setItem(
            'jaxpathways-career-plan-v1',
            JSON.stringify({
              version: 1,
              roleId: role.id,
              pathwayId: role.defaultPathwayId,
              startPhaseId: selectedPhase(),
              knownPhaseIds: [...detail.querySelectorAll('.starting-checks input:checked')].map(
                (input) => input.value,
              ),
            }),
          );
          const raw = localStorage.getItem('jaxpathways-notebook-v1');
          const notebook = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
          notebook.route = role.defaultPathwayId;
          localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(notebook));
          $('#career-status').textContent =
            'Plan saved on this device. Find it in My learning notebook.';
        } catch {
          $('#career-status').textContent =
            'Device storage is unavailable. Download your plan to keep it.';
        }
      };
      $('#career-download').onclick = () => {
        const url = URL.createObjectURL(
          new Blob([careerPlanMarkdown(role, courseState.course, selectedPhase())], {
            type: 'text/markdown',
          }),
        );
        const a = document.createElement('a');
        a.href = url;
        a.download = role.id + '-learning-plan.md';
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      };
      $('#career-share').onclick = async () => {
        const url = new URL('careers.html', location.href);
        url.searchParams.set('role', role.id);
        url.hash = 'roles';
        try {
          await navigator.clipboard.writeText(url.href);
          $('#career-status').textContent = 'Career route link copied.';
        } catch {
          $('#career-status').textContent = url.href;
        }
      };
      detail
        .querySelectorAll('[data-choice]')
        .forEach((b) => (b.onclick = () => openRoute(b.dataset.choice)));
      detail.querySelectorAll('[data-phase]').forEach(
        (b) =>
          (b.onclick = () => {
            location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(b.dataset.phase)}#curriculum`;
          }),
      );
      detail.scrollIntoView({
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
        block: 'start',
      });
    }
    document
      .querySelectorAll('[data-role]')
      .forEach((b) => (b.onclick = () => showRole(b.dataset.role)));
    window.addEventListener('popstate', () => {
      const id = new URLSearchParams(location.search).get('role');
      if (id) showRole(id);
      else {
        $('#role-detail').hidden = true;
        document.body.classList.remove('career-open');
      }
    });
    const selected = new URLSearchParams(location.search).get('role');
    if (selected && !new URLSearchParams(location.search).has('phase')) showRole(selected);
  }
  if ($('#routes')) {
    function showPath() {
      const id = new URLSearchParams(location.search).get('path'),
        route = courseState.course.pathways.find((r) => r.id === id),
        detail = $('#path-detail');
      if (!route) {
        detail.hidden = true;
        document.body.classList.remove('path-open');
        return;
      }
      document.body.classList.add('path-open');
      detail.hidden = false;
      const a = availability(route.phaseIds),
        projectId =
          route.capstone.projectId ||
          route.phaseIds
            .map((id) => phaseById(id).projectId)
            .filter(Boolean)
            .at(-1);
      const available = route.phaseIds
        .flatMap((id) => phaseById(id).lessons)
        .filter((l) => l.status === 'authored');
      const next =
        available.find((l) => !courseState.lessonProgress.lessons[l.id]?.evidenceSaved) ||
        available[0];
      detail.innerHTML = `<a href="pathways.html" class="back-link">All learning paths</a><div class="path-detail-heading"><div><h1 tabindex="-1">${escapeHtml(route.title)}</h1><p class="intro">${escapeHtml(route.description)}</p><p class="availability-label">${a.planned ? 'Partly available' : 'Available lesson drafts'} · ${a.available} lessons · ${a.planned} planned ${a.planned === 1 ? 'topic' : 'topics'}</p><p>${escapeHtml(route.prerequisites)}</p></div><div class="path-outcome"><span class="branch-glyph" aria-hidden="true">↳</span><h2>${escapeHtml(route.capstone.title)}</h2><p>${route.capstone.projectId ? 'Project workspace available' : 'Final project is planned. Available preparation is shown below.'}</p>${projectId ? `<a href="project.html?id=${encodeURIComponent(projectId)}">Inspect available project</a>` : ''}</div></div><div class="actions">${next ? '<button type="button" id="use-path" class="primary">Use this path & start</button>' : ''}<a class="button" href="course.html?path=${encodeURIComponent(id)}">Browse lessons</a></div><p id="path-save-status" role="status"></p>${focusSummary(route)}<div class="role-columns"><div><h2>Who this path is for</h2><p>${escapeHtml(route.audience)}</p><h3>Your first useful artifact</h3><p>${escapeHtml(route.firstArtifact)}</p></div><div><h2>How to study</h2><p>${escapeHtml(route.studyAdvice)}</p>${trackLinks(route.modalityTrackIds)}</div></div>${engineeringLinks(route)}<h2>Your journey</h2><ol class="path-journey">${route.phaseIds
        .map((phaseId) => {
          const p = phaseById(phaseId),
            a = availability([phaseId]);
          return `<li><a href="course.html?path=${encodeURIComponent(id)}&phase=${encodeURIComponent(phaseId)}${a.available ? '' : '&roadmap=1'}"><strong>${escapeHtml(p.title)}</strong><span>${a.available} available drafts${a.planned ? ' / ' + a.planned + ' planned' : ''}</span></a></li>`;
        })
        .join(
          '',
        )}</ol><h2>Evidence to collect</h2><ul>${route.capstone.evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>${route.capstone.projectId ? `<p><a class="button" href="project.html?id=${encodeURIComponent(route.capstone.projectId)}">Open final project</a></p>` : ''}${route.capstone.assessmentDraft ? `<p><a href="${escapeHtml(route.capstone.assessmentDraft.url)}">Synthesis assessment · review draft</a></p>` : ''}${(route.assessmentExtensions || []).map((a) => `<p><a href="${escapeHtml(a.url)}">${escapeHtml(a.title)} · review draft</a></p>`).join('')}<details><summary>Choose a different starting step</summary><p>Start where you need practice. Earlier steps are still available; skipping does not mark them complete.</p><div class="starting-steps">${route.phaseIds
        .map((phaseId) => {
          const p = phaseById(phaseId),
            l = p.lessons.find((l) => l.status === 'authored');
          return l
            ? `<a href="lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(l.id)}">${escapeHtml(p.title)}</a>`
            : '';
        })
        .join(
          '',
        )}</div></details><details><summary>Source and project expectations</summary><p>${escapeHtml(route.capstone.assessment)}</p><a href="${sources(route)}">Path source on GitHub</a></details>`;
      if (next)
        $('#use-path').onclick = () => {
          try {
            const raw = localStorage.getItem('jaxpathways-notebook-v1');
            const state = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
            state.route = id;
            localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(state));
            location.href = `lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(next.id)}`;
          } catch {
            $('#path-save-status').innerHTML =
              `This path could not be saved on this browser. <a href="lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(next.id)}">Continue without saving</a>`;
          }
        };
      detail.querySelector('h1').focus({ preventScroll: true });
    }
    showPath();
    window.addEventListener('popstate', showPath);
  }
}

function startPath(id) {
  const route = courseState.course.pathways.find((r) => r.id === id);
  if (!route) return;
  const lesson = route.phaseIds
    .flatMap((id) => phaseById(id).lessons)
    .find((l) => l.status === 'authored');
  location.href = lesson
    ? `lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(lesson.id)}`
    : `course.html?path=${encodeURIComponent(id)}&roadmap=1`;
}

export { setupChoices, startPath };
