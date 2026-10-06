function evidenceReviewPacket(
  entries,
  { routes, lessons, projects, lessonProgress, projectProgress, baseUrl, generatedAt },
) {
  if (!Array.isArray(entries) || !entries.length) throw Error('Choose at least one artifact');
  const plain = (value) =>
    String(value ?? '')
      .replace(/\r?\n/g, ' ')
      .replace(/[\\`*{}\[\]()<>#!|~]/g, '\\$&');
  const literal = (value) => {
    const text = String(value ?? '');
    const longest = Math.max(2, ...(text.match(/`+/g) || []).map((run) => run.length));
    const fence = '`'.repeat(longest + 1);
    return `${fence}text\n${text}\n${fence}`;
  };
  const courseUrl = (page, params) => {
    const url = new URL(page, baseUrl);
    if (!/^https?:$/.test(url.protocol)) throw Error('Unsupported course link');
    url.search = new URLSearchParams(params).toString();
    return url.href;
  };
  const link = (label, url) => {
    if (!/^https?:\/\//i.test(url)) throw Error('Unsupported artifact link');
    return `[${plain(label)}](<${String(url)
      .replace(/</g, '%3C')
      .replace(/>/g, '%3E')
      .replace(/[\r\n]/g, '')}>)`;
  };
  let text = `# Learner-owned evidence review packet\n\nGenerated: ${plain(generatedAt)}\nArtifacts included: ${entries.length}\n\nThis is a handoff document, not an assessment result. The application has read saved records to assemble it. It has not opened artifact links, run the learner's code, verified outputs, assigned a reviewer, or approved competency. Nothing has been sent externally. All reviewer criteria below start unchecked.\n\nCourse links refer to the installation used for export. Localhost links require a local course server. Keep the actual code, environment file, and output files with this packet; exporting links does not copy their contents.\n\n## How to use this packet\n\n1. The learner adds missing code/output files and identifies what needs feedback.\n2. A reviewer independently inspects or runs the work and records observations below.\n3. The learner keeps corrections and follow-up evidence. Completing this document does not automatically change app progress or award a certificate.\n\n`;
  entries.forEach((entry, index) => {
    const route = routes.find((r) => r.id === entry.route),
      lesson = lessons.find((l) => l.id === entry.lessonId),
      project = projects.find((p) => p.id === entry.projectId),
      stage = project?.stages.find((s) => s.id === entry.stageId);
    text += `## ${index + 1}. ${plain(entry.title)}\n\n### Saved record metadata\n\n- Artifact ID: ${plain(entry.id)}\n- Created: ${plain(entry.date)}\n- Last edited: ${plain(entry.updatedAt || entry.date)}\n- Pathway: ${plain(route?.title || entry.route)}\n- Record status: self-reported; no review recorded by this application\n`;
    text += entry.url
      ? `- Artifact: ${link('Open learner artifact', entry.url)}\n`
      : '- Artifact link: not supplied; attach code and output files manually\n';
    if (entry.lessonId) {
      text += `- Lesson: ${link(lesson?.title || entry.lessonId, courseUrl('lesson.html', { path: entry.route, lesson: entry.lessonId }))}\n`;
      const progress = lessonProgress?.lessons[entry.lessonId];
      text += `- Local formative checkpoint record: ${progress?.checkpointPassed ? 'marked passed' : 'not marked passed'}; this is not artifact verification\n- Local exercise tick: ${progress?.evidenceSaved ? 'self-reported complete' : 'not marked complete'}\n`;
    }
    if (entry.projectId) {
      text += `- Project: ${link(project?.title || entry.projectId, courseUrl('project.html', { id: entry.projectId }))}\n`;
      if (entry.stageId) {
        text += `- Stage: ${link(stage?.title || 'Stage ' + entry.stageId, courseUrl('project.html', { id: entry.projectId }) + '#stage-' + encodeURIComponent(entry.stageId))}\n- Local stage tick: ${(projectProgress?.[entry.projectId] || []).includes(entry.stageId) ? 'self-reported complete' : 'not marked complete'}\n`;
      }
    }
    text += `\n### Learner's reported observations\n\nThe text below is a learner claim, preserved literally. No command execution or linked result has been observed by this export.\n\n${literal(entry.note)}\n\n`;
    if (stage) {
      text += `### Public stage contract — expected, not observed\n\nEvidence requested: ${plain(stage.evidence)}\n\nSuggested command from the course:\n\n${literal(stage.command)}\n\nExpected passing output (not a result for this learner):\n\n${literal(stage.expected || 'Consult the project README; no expected output is recorded here.')}\n\n`;
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
