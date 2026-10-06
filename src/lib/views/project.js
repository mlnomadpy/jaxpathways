import { escapeHtml as escape } from '../html.js';

/** @param {string} code
 * @param {string} label
 */
export const projectCommand = (code, label) =>
  `<div class="project-command">
    <div>
    <span>${escape(label)}</span>
    <button class="copy-command" type="button">Copy command</button>
    </div>
    <pre><code>${escape(code)}</code></pre>
    <p class="copy-status" role="status"></p>
    </div>`;

/** @param {import('../../types/manifests').ProjectManifest} p */
export function renderProject(p) {
  const command = projectCommand;
  return `<a href="course.html?path=${escape(p.pathwayId)}">Course / ${escape(p.pathwayId)}</a>
    <p class="edition">Integration project · ${escape(p.hardware)}</p>
    <h1>${escape(p.title)}</h1>
    <p>Build your implementation, check each stage, and explain the evidence.</p>
    <div class="actions">
    <a class="button primary" href="project-guides/${escape(p.id)}.html">Read the guide and figures</a>
    <a class="button" href="downloads/${escape(p.id)}.zip" download>Download project workspace</a>
    <a class="button" href="lesson.html?path=${escape(p.pathwayId)}&lesson=${escape(p.requiredLessonIds[0])}">Learn the prerequisites</a>
    </div>
    <details>
    <summary>Required lessons</summary>
    <ul>${p.requiredLessonIds
      .map(
        (id) => `<li>
    <a href="lesson.html?path=${escape(p.pathwayId)}&lesson=${escape(id)}">${escape(id)}</a>
    </li>`,
      )
      .join('')}</ul>
    </details>
    <details class="project-setup">
    <summary>Set up your workspace</summary>
    <p>${escape(p.setupNote)}</p>
    <label for="project-os">Terminal instructions</label>
    <select data-choices="inline" data-choice-label="Terminal instructions" id="project-os">
    <option value="unix">macOS / Linux</option>
    <option value="windows">Windows PowerShell</option>
    </select>
    <div id="project-environment"></div>
    <p>Work in <code>${escape(p.workspaceFile)}</code>. Implement the TODO functions before running the stage checks. The untouched starter deliberately fails.</p>${p.prepare ? `<h3>Prepare your own file</h3>${p.prepare.map((c) => command(c.command, c.label)).join('')}` : ''}<details class="project-reference">
    <summary>Check the reference implementation separately</summary>${command(p.referenceCommand || `python3 projects/${p.id}/tests/check.py --implementation solution --stage all`, 'Reference command')}<p>This checks the provided solution. It does not establish that your own implementation passes.</p>
    </details>
    </details>
    <ol class="project-stages">${p.stages
      .map(
        (s) => `<li id="stage-${escape(s.id)}">
    <h2>${escape(s.title)}</h2>
    <p>
    <strong>Keep:</strong> ${escape(s.evidence)}</p>${command(s.command, 'Stage ' + s.id + ' · macOS / Linux')}<p class="expected-output">
    <strong>Successful check:</strong>
    <br>
    <code>${escape(s.expected)}</code>
    <br>The process exits with code 0. Numeric reports can vary within the checked tolerances.</p>
    <details>
    <summary>If this stage fails</summary>
    <p>${escape(s.diagnosis)}</p>
    <p>Keep the command and the final error line. A missing module usually means the selected interpreter is outside your course environment; NotImplementedError means the starter function still needs your implementation.</p>
    </details>
    <label class="stage-check">
    <input type="checkbox" data-stage="${escape(s.id)}"> I ran this stage on my implementation and kept its evidence</label>
    <p>
    <a href="notebook.html?project=${escape(p.id)}&stage=${escape(s.id)}&route=${escape(p.pathwayId)}#portfolio">Attach stage ${s.id} evidence</a>
    </p>
    </li>`,
      )
      .join('')}</ol>
    <p id="project-status" role="status"></p>${
      p.assessmentId
        ? `<section>
    <h2>Transfer what you learned</h2>
    <p>Use changed inputs, independent calculations and deliberate failures to assess the combined skills.</p>
    <a class="button" href="assessments/${escape(p.assessmentId)}.html">Open synthesis assessment</a>
    <p class="soft">Editorial review draft. A reviewer must inspect your actual evidence; stage ticks do not award a pass.</p>
    </section>`
        : ''
    }<h2>Review your portfolio</h2>
    <ul>${p.rubric.map((c) => `<li>${escape(c)}</li>`).join('')}</ul>
    <p class="soft">Stage ticks record your report of practice. Public checks establish the specified synthetic contracts; reviewed capability and TPU validation require separate evidence.</p>
    <a class="button" href="notebook.html?project=${escape(p.id)}&route=${escape(p.pathwayId)}#portfolio">Add project evidence</a>`;
}
