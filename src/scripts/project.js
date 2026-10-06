import { CourseCode } from '../lib/code.js';
export async function initProject() {
  const escape = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  const target = document.querySelector('#project');
  const command = (code, label) =>
    `<div class="project-command"><div><span>${escape(label)}</span><button class="copy-command" type="button">Copy command</button></div><pre><code>${escape(code)}</code></pre><p class="copy-status" role="status"></p></div>`;
  try {
    const response = await fetch('api/v1/projects.json', { cache: 'no-store' });
    if (!response.ok) throw Error();
    const { projects } = await response.json(),
      id = new URLSearchParams(location.search).get('id') || 'regression-audit';
    const p = projects.find((p) => p.id === id);
    if (!p) {
      target.innerHTML = '<h1>Project not found</h1><a href="course.html">Return to the course</a>';
      return;
    }
    document.title = p.title + ' — JAX Pathways';
    target.innerHTML = `<a href="course.html?path=${escape(p.pathwayId)}">Course / ${escape(p.pathwayId)}</a><p class="edition">Integration project · ${escape(p.hardware)}</p><h1>${escape(p.title)}</h1><p>Build your implementation, check each stage, and explain the evidence.</p><div class="actions"><a class="button primary" href="project-guides/${escape(p.id)}.html">Read the guide and figures</a><a class="button" href="downloads/${escape(p.id)}.zip" download>Download project workspace</a><a class="button" href="lesson.html?path=${escape(p.pathwayId)}&lesson=${escape(p.requiredLessonIds[0])}">Learn the prerequisites</a></div><details><summary>Required lessons</summary><ul>${p.requiredLessonIds.map((id) => `<li><a href="lesson.html?path=${p.pathwayId}&lesson=${escape(id)}">${escape(id)}</a></li>`).join('')}</ul></details><details class="project-setup"><summary>Set up your workspace</summary><p>${escape(p.setupNote)}</p><label for="project-os">Terminal instructions</label><select id="project-os"><option value="unix">macOS / Linux</option><option value="windows">Windows PowerShell</option></select><div id="project-environment"></div><p>Work in <code>${escape(p.workspaceFile)}</code>. Implement the TODO functions before running the stage checks. The untouched starter deliberately fails.</p>${p.prepare ? `<h3>Prepare your own file</h3>${p.prepare.map((c) => command(c.command, c.label)).join('')}` : ''}<details class="project-reference"><summary>Check the reference implementation separately</summary>${command(p.referenceCommand || `python3 projects/${p.id}/tests/check.py --implementation solution --stage all`, 'Reference command')}<p>This checks the provided solution. It does not establish that your own implementation passes.</p></details></details><ol class="project-stages">${p.stages.map((s) => `<li id="stage-${escape(s.id)}"><h2>${escape(s.title)}</h2><p><strong>Keep:</strong> ${escape(s.evidence)}</p>${command(s.command, 'Stage ' + s.id + ' · macOS / Linux')}<p class="expected-output"><strong>Successful check:</strong><br><code>${escape(s.expected)}</code><br>The process exits with code 0. Numeric reports can vary within the checked tolerances.</p><details><summary>If this stage fails</summary><p>${escape(s.diagnosis)}</p><p>Keep the command and the final error line. A missing module usually means the selected interpreter is outside your course environment; NotImplementedError means the starter function still needs your implementation.</p></details><label class="stage-check"><input type="checkbox" data-stage="${s.id}"> I ran this stage on my implementation and kept its evidence</label><p><a href="notebook.html?project=${escape(p.id)}&stage=${s.id}&route=${escape(p.pathwayId)}#portfolio">Attach stage ${s.id} evidence</a></p></li>`).join('')}</ol><p id="project-status" role="status"></p>${p.assessmentId ? `<section><h2>Transfer what you learned</h2><p>Use changed inputs, independent calculations and deliberate failures to assess the combined skills.</p><a class="button" href="assessments/${escape(p.assessmentId)}.html">Open synthesis assessment</a><p class="soft">Editorial review draft. A reviewer must inspect your actual evidence; stage ticks do not award a pass.</p></section>` : ''}<h2>Review your portfolio</h2><ul>${p.rubric.map((c) => `<li>${escape(c)}</li>`).join('')}</ul><p class="soft">Stage ticks record your report of practice. Public checks establish the specified synthetic contracts; reviewed capability and TPU validation require separate evidence.</p><a class="button" href="notebook.html?project=${escape(p.id)}&route=${escape(p.pathwayId)}#portfolio">Add project evidence</a>`;
    function bindCopy() {
      target.querySelectorAll('.copy-command').forEach(
        (b) =>
          (b.onclick = async () => {
            const panel = b.closest('.project-command');
            try {
              await navigator.clipboard.writeText(panel.querySelector('code').textContent);
              panel.querySelector('.copy-status').textContent = 'Command copied.';
            } catch {
              panel.querySelector('.copy-status').textContent =
                'Select the command and copy it manually.';
            }
          }),
      );
    }
    function environment() {
      const windows = document.querySelector('#project-os').value === 'windows';
      const py = windows ? '.\\.venv\\Scripts\\python.exe' : 'python';
      const setup = windows
        ? 'py -3.14 -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements-cpu.txt'
        : 'python3.14 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-cpu.txt';
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
        '<p>Use the environment interpreter for every check. The CPU examples were tested with Python 3.14.3. PowerShell instructions have not been execution-validated on Windows.</p>';
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
