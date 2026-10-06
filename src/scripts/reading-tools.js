export function initReadingTools() {
  const assessment = document.querySelector('#assessment article');
  if (assessment && !document.querySelector('.assessment-task-links')) {
    const tasks = [...assessment.querySelectorAll('h2')].filter((h) =>
      /^Task \d/.test(h.textContent),
    );
    if (tasks.length) {
      const nav = document.createElement('nav');
      nav.className = 'assessment-task-links';
      nav.setAttribute('aria-label', 'Assessment tasks');
      for (const h of tasks) {
        const a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent;
        nav.append(a);
      }
      const contents = document.querySelector('.assessment-contents');
      if (contents) {
        const details = document.createElement('details'),
          summary = document.createElement('summary');
        details.className = 'assessment-all-sections';
        summary.textContent = 'All assessment sections';
        contents.before(nav, details);
        details.append(summary, contents);
      }
    }
  }
  const project = document.querySelector('.project-stages');
  if (project && !document.querySelector('.project-stage-links')) {
    const nav = document.createElement('nav');
    nav.className = 'project-stage-links';
    nav.setAttribute('aria-label', 'Project stages');
    for (const stage of project.children) {
      const a = document.createElement('a');
      a.href = '#' + stage.id;
      a.textContent = stage.querySelector('h2')?.textContent || stage.id;
      nav.append(a);
    }
    project.before(nav);
  }
}
