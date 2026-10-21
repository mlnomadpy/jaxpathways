import { parseProjects } from '../lib/manifests.js';
import { enhanceChoices } from './choices.js';
import { initReadingTools } from './reading-tools.js';
import { fetchJson } from '../lib/http.js';
import { renderProject, projectCommand as command } from '../lib/views/project.js';
import { bindClipboard } from './browser-actions.js';
import { CourseCode } from '../lib/code.js';
import { requireElement } from '../lib/dom.js';
export async function initProject() {
  const target = requireElement('#project');
  try {
    const projects = parseProjects(await fetchJson('api/v1/projects.json', { cache: 'no-store' })),
      id = new URLSearchParams(location.search).get('id') || 'regression-audit';
    const p = projects.find((p) => p.id === id);
    if (!p) {
      target.innerHTML = '<h1>Project not found</h1><a href="course.html">Return to the course</a>';
      return;
    }
    const project = p;
    document.title = project.title + ' — JAX Pathways';
    target.innerHTML = renderProject(project);
    function bindCopy() {
      bindClipboard({
        root: target,
        selector: '.copy-command',
        source: (button) => button.closest('.project-command')?.querySelector('code') || null,
        feedback: (button) =>
          button.closest('.project-command')?.querySelector('.copy-status') || null,
        success: 'Command copied.',
        failure: 'Select the command and copy it manually.',
      });
    }
    function environment() {
      const windows =
        /** @type {HTMLSelectElement} */ (requireElement('#project-os')).value === 'windows';
      const py = windows ? '.\\.venv\\Scripts\\python.exe' : 'python';
      const setup = windows
        ? project.environment.setupCommands.windows
        : project.environment.setupCommands.unix;
      const reference = target.querySelector('.project-reference code');
      if (reference)
        reference.textContent = (
          project.referenceCommand ||
          `python3 projects/${project.id}/tests/check.py --implementation solution --stage all`
        ).replace(/^python3/, py);
      requireElement('#project-environment').innerHTML =
        command(
          setup,
          windows ? 'PowerShell · extracted top folder' : 'Terminal · extracted top folder',
        ) +
        `<p>Use the virtual environment interpreter for every stage check. The pinned package set declares Python ${project.environment.pythonCompatibility}; recorded course receipts use Python 3.14.3.</p>`;
      target.querySelectorAll('.project-stages .project-command').forEach((panel, i) => {
        const stage = project.stages[i];
        if (!stage) return;
        const code = panel.querySelector('code');
        const label = panel.querySelector('span');
        if (!code || !label) return;
        code.textContent = stage.command.replace(/^python3/, py);
        label.textContent =
          'Stage ' + stage.id + ' · ' + (windows ? 'PowerShell' : 'activated terminal');
      });
      bindCopy();
      CourseCode.paint(target, 'shell');
    }
    requireElement('#project-os').addEventListener('change', environment);
    environment();
    enhanceChoices(target);
    initReadingTools();
    if (/^#stage-[0-9]+$/.test(location.hash))
      document
        .getElementById(location.hash.slice(1))
        ?.scrollIntoView({ block: 'start', behavior: 'instant' });
    const key = 'jaxpathways-project-' + project.id;
    /** @type {string[]} */
    let saved = [];
    try {
      const stored = localStorage.getItem(key);
      const value = stored ? JSON.parse(stored) : null;
      if (Array.isArray(value))
        saved = value.filter(
          /** @param {unknown} id @returns {id is string} */
          (id) => typeof id === 'string' && project.stages.some((stage) => stage.id === id),
        );
    } catch {}
    const statusNode = requireElement('#project-status');
    const status = () =>
      (statusNode.textContent = `${saved.length}/${project.stages.length} stages completed. Save your terminal outputs and diagnostic notes in your portfolio.`);
    /** @type {NodeListOf<HTMLInputElement>} */ (target.querySelectorAll('[data-stage]')).forEach(
      (input) => {
        const stageId = input.dataset.stage;
        if (!stageId) return;
        input.checked = saved.includes(stageId);
        input.addEventListener('change', () => {
          const previous = saved;
          saved = [
            .../** @type {NodeListOf<HTMLInputElement>} */ (
              target.querySelectorAll('[data-stage]:checked')
            ),
          ]
            .map((item) => item.dataset.stage)
            .filter((id) => typeof id === 'string');
          try {
            localStorage.setItem(key, JSON.stringify(saved));
            status();
          } catch {
            saved = previous;
            input.checked = saved.includes(stageId);
            statusNode.textContent =
              'Browser storage is unavailable. This change was not saved; keep your own stage report.';
          }
        });
      },
    );
    status();
  } catch {
    target.innerHTML =
      '<h1>The project could not load</h1><p>Refresh this page or <a href="course.html">return to the course</a>.</p>';
  }
}
