/**
 * @param {import('./learner-records.js').EvidenceEntry[]} entries
 * @param {{
 *   routes: { id: string; title: string }[];
 *   lessons: import('../types/course').Lesson[];
 *   projects: import('../types/manifests').ProjectManifest[];
 *   lessonProgress?: import('./learner-records.js').LessonProgressState;
 *   projectProgress?: Record<string, string[]>;
 *   baseUrl: string;
 *   generatedAt: string;
 * }} context
 */
function evidenceReviewPacket(
  entries,
  { routes, lessons, projects, lessonProgress, projectProgress, baseUrl, generatedAt },
) {
  if (!Array.isArray(entries) || !entries.length) throw Error('Choose at least one artifact');
  /** @param {unknown} value */
  const plain = (value) =>
    String(value ?? '')
      .replace(/\r?\n/g, ' ')
      .replace(/[\\`*{}\[\]()<>#!|~]/g, '\\$&');
  /** @param {unknown} value */
  const literal = (value) => {
    const text = String(value ?? '');
    const longest = Math.max(2, ...(text.match(/`+/g) || []).map((run) => run.length));
    const fence = '`'.repeat(longest + 1);
    return `${fence}text\n${text}\n${fence}`;
  };
  /**
   * @param {string} page
   * @param {Record<string, string>} params
   */
  const courseUrl = (page, params) => {
    const url = new URL(page, baseUrl);
    if (!/^https?:$/.test(url.protocol)) throw Error('Unsupported course link');
    url.search = new URLSearchParams(params).toString();
    return url.href;
  };
  /**
   * @param {string} label
   * @param {string} url
   */
  const link = (label, url) => {
    if (!/^https?:\/\//i.test(url)) throw Error('Unsupported artifact link');
    return `[${plain(label)}](<${String(url)
      .replace(/</g, '%3C')
      .replace(/>/g, '%3E')
      .replace(/[\r\n]/g, '')}>)`;
  };
  let text = `# Portfolio evidence review packet\n\nGenerated: ${plain(generatedAt)}\nArtifacts included: ${entries.length}\n\nAttach your source code, environment specification, and terminal output files alongside this packet when sharing it for code review. All reviewer checklist items below start unchecked.\n\n## How to use this packet\n\n1. Attach the code and output files referenced in each artifact entry below.\n2. Run the stage verification commands and inspect the changed-condition experiments.\n3. Record review notes and follow-up actions in the checklist below.\n\n`;
  entries.forEach((entry, index) => {
    const route = routes.find((r) => r.id === entry.route),
      lesson = lessons.find((l) => l.id === entry.lessonId),
      project = projects.find((p) => p.id === entry.projectId),
      stage = project?.stages.find((s) => s.id === entry.stageId);
    text += `## ${index + 1}. ${plain(entry.title)}\n\n### Saved record metadata\n\n- Artifact ID: ${plain(entry.id)}\n- Created: ${plain(entry.date)}\n- Last edited: ${plain(entry.updatedAt || entry.date)}\n- Pathway: ${plain(route?.title || entry.route)}\n- Portfolio status: saved in local workspace\n`;
    text += entry.url
      ? `- Artifact: ${link('Open learner artifact', entry.url)}\n`
      : '- Artifact link: not supplied; attach code and output files manually\n';
    if (entry.lessonId) {
      text += `- Lesson: ${link(lesson?.title || entry.lessonId, courseUrl('lesson.html', { path: entry.route, lesson: entry.lessonId }))}\n`;
      const progress = lessonProgress?.lessons[entry.lessonId];
      text += `- Lesson checkpoint: ${progress?.checkpointPassed ? 'passed' : 'not yet passed'}\n- Exercise status: ${progress?.evidenceSaved ? 'completed' : 'in progress'}\n`;
    }
    if (entry.projectId) {
      text += `- Project: ${link(project?.title || entry.projectId, courseUrl('project.html', { id: entry.projectId }))}\n`;
      if (entry.stageId) {
        text += `- Stage: ${link(stage?.title || 'Stage ' + entry.stageId, courseUrl('project.html', { id: entry.projectId }) + '#stage-' + encodeURIComponent(entry.stageId))}\n- Stage status: ${(projectProgress?.[entry.projectId] || []).includes(entry.stageId) ? 'completed' : 'in progress'}\n`;
      }
    }
    text += `\n### Recorded engineering notes\n\n${literal(entry.note)}\n\n`;
    if (stage) {
      text += `### Stage verification contract\n\nEvidence required: ${plain(stage.evidence)}\n\nVerification command:\n\n${literal(stage.command)}\n\nExpected passing output:\n\n${literal(stage.expected || 'Consult the project README for expected output.')}\n\n`;
    }
    text +=
      '### Reviewer criteria — leave unchecked until inspected\n\n- [ ] Inspect the linked or attached implementation; identify the exact revision reviewed.\n- [ ] Record the runtime, package versions, command, and actual observed output, or explain why execution was not possible.\n- [ ] Compare a numerical or behavioral result with an independent expectation and explain the tolerance.\n- [ ] Inspect a changed condition or deliberate failure and the evidence for its diagnosis.\n- [ ] Separate the demonstrated behavior from hardware, data, or generalization claims that remain untested.\n';
    if (lesson) {
      text += `- [ ] Verify lesson evidence: ${plain(lesson.evidence)}\n- [ ] Probe understanding beyond a quiz: ${plain(lesson.check)}\n`;
    }
    if (project) {
      for (const criterion of project.rubric || []) text += `- [ ] ${plain(criterion)}\n`;
    }
    if (entry.lessonId && !lesson)
      text +=
        '- [ ] Recover the lesson contract; this lesson is unavailable in the current catalog.\n';
    if (entry.projectId && !project)
      text +=
        '- [ ] Recover the project rubric; this project is unavailable in the current catalog.\n';
    if (entry.stageId && !stage)
      text +=
        '- [ ] Recover the stage contract; this stage is unavailable in the current catalog.\n';
    text +=
      '\n### Reviewer observations — not supplied by the application\n\n- Reviewer, if someone agrees to review: ____________________\n- Date and implementation revision: ____________________\n- Command, environment, and hardware actually used: ____________________\n- Output or files independently observed: ____________________\n- Checks that could not be run, and why: ____________________\n- Misconception, failure, or correction requested: ____________________\n- Follow-up evidence to inspect: ____________________\n\n';
  });
  return text;
}

export { evidenceReviewPacket };
