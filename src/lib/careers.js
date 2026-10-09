/**
 * @typedef {{ id: string, number?: string, title?: string, projectId?: string, additionalProjectIds?: string[], lessons: { status: string }[] }} CoveragePhase
 * @typedef {{ phases: CoveragePhase[], pathways?: import('../types/course').Pathway[], modalityTracks?: import('../types/course').ModalityTrack[] }} CoverageCourse
 */

/**
 * @param {string[]} phaseIds
 * @param {CoverageCourse} courseData
 */
function careerCoverage(phaseIds, courseData) {
  const phases = phaseIds
    .map((id) => courseData.phases.find((p) => p.id === id))
    .filter(/** @returns {p is CoveragePhase} */ (p) => Boolean(p));
  const lessons = phases.flatMap((p) => p.lessons),
    applied = phases.filter((p) => Number(p.number) >= 5).flatMap((p) => p.lessons);
  return {
    available: lessons.filter((l) => l.status === 'authored').length,
    planned: lessons.filter((l) => l.status !== 'authored').length,
    appliedAvailable: applied.filter((l) => l.status === 'authored').length,
    appliedPlanned: applied.filter((l) => l.status !== 'authored').length,
    projects: [
      ...new Set(
        phases
          .flatMap((p) => [p.projectId, ...(p.additionalProjectIds || [])])
          .filter(/** @returns {id is string} */ (id) => Boolean(id)),
      ),
    ],
  };
}

/**
 * @param {{ phaseIds: string[], focusPhaseIds?: string[] }} route
 * @param {CoverageCourse} courseData
 */
function pathwayCoverage(route, courseData) {
  const focusIds =
    route.focusPhaseIds ||
    route.phaseIds.filter(
      (id) => Number(courseData.phases.find((phase) => phase.id === id)?.number) >= 5,
    );
  return {
    focus: careerCoverage(focusIds, courseData),
    preparation: careerCoverage(
      route.phaseIds.filter((id) => !focusIds.includes(id)),
      courseData,
    ),
  };
}

/**
 * @param {import('../types/course').Role} role
 * @param {string[]} knownPhaseIds
 * @param {import('../types/course').Course} courseData
 */
function careerStartingPhase(role, knownPhaseIds, courseData) {
  const route =
    courseData.pathways.find((r) => r.id === role.defaultPathwayId) || courseData.pathways[0];
  const known = new Set(knownPhaseIds);
  return route.phaseIds.find((id) => !known.has(id)) || route.phaseIds[0];
}

/**
 * @param {import('../types/course').Role} role
 * @param {import('../types/course').Course} courseData
 * @param {string} startPhaseId
 */
function careerPlanMarkdown(role, courseData, startPhaseId) {
  const route =
    courseData.pathways.find((r) => r.id === role.defaultPathwayId) || courseData.pathways[0];
  const coverage = pathwayCoverage(route, courseData);
  const phase = (/** @type {string} */ id) =>
    courseData.phases.find((p) => p.id === id) || courseData.phases[0];
  const guides = (role.modalityTrackIds || []).map((id) => {
    const track = (courseData.modalityTracks || []).find((track) => track.id === id);
    return (
      id +
      ' — python scripts/course.py guide ' +
      id +
      ' (' +
      (track?.harnessProjectId
        ? 'runnable harness: ' + track.harnessProjectId
        : 'connected harness planned') +
      ')'
    );
  });
  return `# ${role.title}\n\n${role.description}\n\nWork example: ${role.workExample}\n\nPreparation: ${role.background}\n\nFirst artifact: ${route.firstArtifact}\n\nFocus phases: ${coverage.focus.available} available lessons; ${coverage.focus.planned} planned. Preparation: ${coverage.preparation.available} available; ${coverage.preparation.planned} planned.\n\n${route.studyAdvice}\n\nProject guides: ${guides.join('; ') || 'No modality track; follow the domain milestones below.'}\n\n## Suggested starting point\n\n${phase(startPhaseId).number}: ${phase(startPhaseId).title}\nRecommended starting phase based on your selected prerequisite topics.\n\n## Work you will practice\n\n${role.responsibilities.map((s) => '- ' + s).join('\n')}\n\n## Route through the shared course\n\n${route.phaseIds.map((id) => '- ' + phase(id).number + ': ' + phase(id).title).join('\n')}\n\n## Milestones\n\n${role.milestones.map((m, i) => `### ${i + 1}. ${m.title}\n\nBuild: ${m.deliverable}\n\nAvailable lessons: ${careerCoverage(m.phaseIds, courseData).available}; staged projects: ${careerCoverage(m.phaseIds, courseData).projects.length}.\n\nVerify:\n${m.criteria.map((s) => '- ' + s).join('\n')}`).join('\n\n')}\n\n## Portfolio project\n\n${role.portfolio}\n\nProject workspace: projects/${route.capstone.projectId}/README.md\nAssessment: ${route.capstone.assessmentDraft?.source || 'Synthesis assessment'}\n\nDemonstrate: ${role.readiness}\n\n## Questions to defend in a review\n\n${role.reviewQuestions.map((q) => '- ' + q).join('\n')}\n\n## Engineering extensions\n\n${(route.engineeringLessonIds || []).map((id) => '- ' + id).join('\n')}\n\nFollow each lesson’s prerequisites. Connected project: projects/engineering-release/README.md\n\n## Evidence to collect\n\n${route.capstone.evidence.map((item) => '- ' + item).join('\n')}\n\n${role.scopeNote}\n\nContent status: ${careerCoverage(route.phaseIds, courseData).available} lessons available along this route.\n`;
}

export { pathwayCoverage, careerCoverage, careerStartingPhase, careerPlanMarkdown };
