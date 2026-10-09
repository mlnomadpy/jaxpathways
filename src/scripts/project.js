import { parseProjects } from '../lib/manifests.js';
import { enhanceChoices } from './choices.js';
import { initReadingTools } from './reading-tools.js';
import { fetchJson } from '../lib/http.js';
import { renderProject, projectCommand as command } from '../lib/views/project.js';
import { bindClipboard } from './browser-actions.js';
import { CourseCode } from '../lib/code.js';
export async function initProject() {
  const target = document.querySelector('#project');
  try {
    const projects = parseProjects(await fetchJson('api/v1/projects.json', { cache: 'no-store' })),
      id = new URLSearchParams(location.search).get('id') || 'regression-audit';
    const p = projects.find((p) => p.id === id);
    if (!p) {
      target.innerHTML = '<h1>Project not found</h1><a href="course.html">Return to the course</a>';
      return;
    }
    document.title = p.title + ' — JAX Pathways';
    target.innerHTML = renderProject(p);
    function bindCopy() {
      bindClipboard({
        root: target,
        selector: '.copy-command',
        source: (button) => button.closest('.project-command').querySelector('code'),
        feedback: (button) => button.closest('.project-command').querySelector('.copy-status'),
        success: 'Command copied.',
        failure: 'Select the command and copy it manually.',
      });
    }
    function environment() {
      const windows = document.querySelector('#project-os').value === 'windows';
      const py = windows ? '.\\.venv\\Scripts\\python.exe' : 'python';
      const setup = windows
        ? 'py -3 -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements-cpu.txt'
        : 'python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-cpu.txt';
      const reference = target.querySelector('.project-reference code');
      if (reference)
        reference.textContent = (
          p.referenceCommand ||
          `python3 projects/${p.id}/tests/check.py --implementation solution --stage all`
        ).replace(/^python3/, py);
      document.querySelector('#project-environment').innerHTML =
        command(
          setup,
          windows ? 'PowerShell · extracted top folder' : 'Terminal · extracted top folder',
        ) +
        '<p>Use the environment interpreter for every check. The CPU examples were tested with Python 3.14.3 (Python 3.11+ supported). PowerShell instructions have not been execution-validated on Windows.</p>';
      target.querySelectorAll('.project-stages .project-command').forEach((panel, i) => {
        panel.querySelector('code').textContent = p.stages[i].command.replace(/^python3/, py);
        panel.querySelector('span').textContent =
          'Stage ' + p.stages[i].id + ' · ' + (windows ? 'PowerShell' : 'activated terminal');
      });
      bindCopy();
      CourseCode.paint(target, 'shell');
    }
    document.querySelector('#project-os').onchange = environment;
    environment();
    enhanceChoices(target);
    initReadingTools();
    if (/^#stage-[0-9]+$/.test(location.hash))
      document
        .getElementById(location.hash.slice(1))
        ?.scrollIntoView({ block: 'start', behavior: 'instant' });
    const key = 'jaxpathways-project-' + p.id;
    let saved = [];
    try {
      const value = JSON.parse(localStorage.getItem(key));
      if (Array.isArray(value)) saved = value.filter((id) => p.stages.some((s) => s.id === id));
    } catch {}
    const status = () =>
      (document.querySelector('#project-status').textContent =
        `${saved.length}/${p.stages.length} stages self-reported. Your evidence stays separate from reviewed capability.`);
    target.querySelectorAll('[data-stage]').forEach((input) => {
      input.checked = saved.includes(input.dataset.stage);
      input.onchange = () => {
        const previous = saved;
        saved = [...target.querySelectorAll('[data-stage]:checked')].map((i) => i.dataset.stage);
        try {
          localStorage.setItem(key, JSON.stringify(saved));
          status();
        } catch {
          saved = previous;
          input.checked = saved.includes(input.dataset.stage);
          document.querySelector('#project-status').textContent =
            'Browser storage is unavailable. This change was not saved; keep your own stage report.';
        }
      };
    });
    status();
  } catch {
    target.innerHTML =
      '<h1>The project could not load</h1><p>Refresh this page or <a href="course.html">return to the course</a>.</p>';
  }
}
