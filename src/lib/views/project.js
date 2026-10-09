import { extractCodeApis } from '../code-teaching.js';
import { escapeHtml as escape } from '../html.js';
import { inlineMath } from '../math.js';
import { lessonLink } from '../urls.js';

/** @param {import('../../types/manifests').ProjectStage} s
 * @param {string} workspaceFile
 */
function renderStageCodeWalkthrough(s, workspaceFile) {
  if (!s.codeGuide && !s.starterSnippet && !s.solutionSnippet) return '';
  const apis = extractCodeApis(s.solutionSnippet || s.starterSnippet || '');
  return `<div class="stage-code-walkthrough">
    <p class="stage-code-kicker">How to write the Stage ${escape(s.id)} code</p>
    ${s.codeGuide ? `<p class="stage-code-guide">${inlineMath(s.codeGuide)}</p>` : ''}
    ${
      apis.length
        ? `<ul class="stage-api-list">${apis
            .map((api) => `<li><code>${escape(api.token)}</code> — ${inlineMath(api.summary)}</li>`)
            .join('')}</ul>`
        : ''
    }
    ${
      s.starterSnippet
        ? `<div class="stage-starter-code">
    <div class="stage-code-header"><span>Starter scaffold in ${escape(workspaceFile)} (with step-by-step comments)</span></div>
    <pre data-language="python"><code>${escape(s.starterSnippet)}</code></pre>
    </div>`
        : ''
    }
    ${
      s.solutionSnippet
        ? `<details class="stage-solution-details">
    <summary>Compare with Stage ${escape(s.id)} reference implementation (try implementing first)</summary>
    <pre data-language="python"><code>${escape(s.solutionSnippet)}</code></pre>
    </details>`
        : ''
    }
    </div>`;
}

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
export function projectFileLayout(p) {
  const prepCmd = p.prepare?.[0]?.command || '';
  const match = /^cp\s+(\S+)\s+(\S+)/.exec(prepCmd);
  const starterFile = match
    ? match[1]
    : p.workspaceFile.includes('/starter/')
      ? p.workspaceFile
      : `projects/${p.id}/starter/${(p.workspaceFile.split('/').pop() || 'model.py').replace(/^my_/, '')}`;
  const solutionFile = starterFile.replace('/starter/', '/solution/');
  const checkerFile = `projects/${p.id}/tests/check.py`;
  const readmeFile = p.source || `projects/${p.id}/README.md`;
  const starterBasename = starterFile.split('/').pop() || 'model.py';
  const prefix = `projects/${p.id}/`;
  const workspaceRelative = p.workspaceFile.startsWith(prefix)
    ? p.workspaceFile.slice(prefix.length)
    : p.workspaceFile;
  const copiesStarter = starterFile !== p.workspaceFile;
  const treeLines = [
    '<extracted-zip-or-repo-root>/          # 1. Run all terminal commands from here',
    '├── requirements-cpu.txt               # 2. Install pinned CPU dependencies in .venv',
    `└── projects/${p.id}/`,
    '    ├── README.md                      # Full specification, math contracts & figures',
    ...(copiesStarter
      ? [
          `    ├── starter/${starterBasename.padEnd(22, ' ')} # Read-only starter template with TODOs`,
          `    ├── ${workspaceRelative.padEnd(30, ' ')} # <-- PUT YOUR CODE HERE (edit this file)`,
        ]
      : [
          `    ├── ${workspaceRelative.padEnd(30, ' ')} # <-- PUT YOUR CODE HERE (starter with TODOs)`,
        ]),
    `    ├── tests/check.py                 # Automated verifier (--stage 1 .. ${p.stages.length})`,
    `    └── solution/${starterBasename.padEnd(21, ' ')} # Reference solution (--implementation solution)`,
  ];
  return {
    starterFile,
    workspaceFile: p.workspaceFile,
    solutionFile,
    checkerFile,
    readmeFile,
    copiesStarter,
    tree: treeLines.join('\n'),
  };
}

/** @param {import('../../types/manifests').ProjectManifest} p */
export function renderProject(p) {
  const command = projectCommand;
  const layout = projectFileLayout(p);
  return `<div class="page-intro">
    <p class="page-eyebrow"><a href="projects.html">All projects</a><span aria-hidden="true"> · </span><a href="course.html?path=${escape(p.pathwayId)}">Course / ${escape(p.pathwayId)}</a><span aria-hidden="true"> · </span><span>Integration project · ${escape(p.hardware)}</span></p>
    <h1>${escape(p.title)}</h1>
    <p class="intro">${inlineMath(p.setupNote)}</p>
    <div class="actions">
    <a class="button primary" href="project-guides/${escape(p.id)}.html">Read the guide and figures</a>
    <a class="button" href="downloads/${escape(p.id)}.zip" download>Download project workspace</a>
    <a class="button" href="${lessonLink(p.requiredLessonIds[0], p.pathwayId)}">Learn the prerequisites</a>
    </div>
    </div>
    <section class="project-blueprint" aria-labelledby="project-blueprint-heading">
    <div class="project-blueprint-header">
    <span class="stage-pill">Project Blueprint · ${p.stages.length} verified stages</span>
    <h2 id="project-blueprint-heading">What to do, where to put your code &amp; how to run checks</h2>
    <p class="blueprint-lead">Every project bundle contains a starter template, your editable implementation file, an automated stage checker, and an instructor reference solution. Follow this layout so every stage command finds your code on the first run.</p>
    </div>
    <div class="project-blueprint-grid">
    <div class="project-blueprint-card">
    <h3>1. What you will build &amp; prove</h3>
    <p>You will work through <strong>${p.stages.length} progressive stages</strong>. Each stage tests a specific mathematical and engineering contract before unlocking the next:</p>
    <ul class="blueprint-list">${p.rubric.map((c) => `<li>${inlineMath(c)}</li>`).join('')}</ul>
    </div>
    <div class="project-blueprint-card">
    <h3>2. Where to put your code</h3>
    <p class="code-target-callout">Write your code in: <code>${escape(layout.workspaceFile)}</code></p>
    <p>${
      layout.copiesStarter
        ? `Copy the starter scaffold <code>${escape(layout.starterFile)}</code> into <code>${escape(layout.workspaceFile)}</code> before editing. Keep <code>${escape(layout.starterFile)}</code> untouched so you can always compare against the original function signatures.`
        : `Open <code>${escape(layout.workspaceFile)}</code> in your editor and implement the functions marked with <code>NotImplementedError</code> one stage at a time.`
    }</p>
    <pre class="project-tree"><code>${escape(layout.tree)}</code></pre>
    </div>
    </div>
    <ol class="project-workflow-strip">
    <li><strong>Step 1 · Unpack &amp; activate</strong><span>Extract <code>downloads/${escape(p.id)}.zip</code> (or open the repository root) and activate your <code>.venv</code> with <code>requirements-cpu.txt</code>.</span></li>
    <li><strong>Step 2 · Prepare your file</strong><span>${layout.copiesStarter ? `Copy <code>${escape(layout.starterFile)}</code> to <code>${escape(layout.workspaceFile)}</code> and open <code>${escape(layout.workspaceFile)}</code> in your editor.` : `Open <code>${escape(layout.workspaceFile)}</code> in your editor and locate the Stage 1 function.`}</span></li>
    <li><strong>Step 3 · Implement &amp; check each stage</strong><span>Write the code for Stage 1 in <code>${escape(layout.workspaceFile)}</code>, run <code>${escape(layout.checkerFile)}</code> from the top folder, and repeat through Stage ${p.stages.length}.</span></li>
    <li><strong>Step 4 · Record evidence</strong><span>Save the passing terminal output and changed-condition diagnosis in your <a href="notebook.html?project=${escape(p.id)}&route=${escape(p.pathwayId)}#portfolio">Notebook Portfolio</a>.</span></li>
    </ol>
    </section>
    <details>
    <summary>Required lessons (${p.requiredLessonIds.length})</summary>
    <ul>${p.requiredLessonIds
      .map(
        (id) => `<li>
    <a href="${lessonLink(id, p.pathwayId)}">${escape(id)}</a>
    </li>`,
      )
      .join('')}</ul>
    </details>
    <details class="project-setup" open>
    <summary>Set up your workspace &amp; copy starter file</summary>
    <p>${inlineMath(p.setupNote)}</p>
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
    <h2>${inlineMath(s.title)}</h2>
    <p>
    <strong>Edit in <code>${escape(p.workspaceFile)}</code> · Keep as evidence:</strong> ${inlineMath(s.evidence)}</p>${renderStageCodeWalkthrough(s, p.workspaceFile)}${command(s.command, 'Stage ' + s.id + ' · macOS / Linux')}<p class="expected-output">
    <strong>Successful check:</strong>
    <br>
    <code>${escape(s.expected)}</code>
    <br>The process exits with code 0. Numeric reports can vary within the checked tolerances.</p>
    <details>
    <summary>If this stage fails</summary>
    <p>${inlineMath(s.diagnosis)}</p>
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
    <ul>${p.rubric.map((c) => `<li>${inlineMath(c)}</li>`).join('')}</ul>
    <p class="soft">Stage ticks record your report of practice. Public checks establish the specified synthetic contracts; reviewed capability and TPU validation require separate evidence.</p>
    <a class="button" href="notebook.html?project=${escape(p.id)}&route=${escape(p.pathwayId)}#portfolio">Add project evidence</a>`;
}
