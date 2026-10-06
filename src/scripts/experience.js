/* Visible choices adapt existing state controls without changing stored records. */
export function initExperience() {
  const adapters = new WeakMap();
  function enhanceChoice(select) {
    if (select.dataset.internal !== undefined || adapters.has(select)) return;
    const label = select.closest('label');
    const title = (
      label?.textContent ||
      document.querySelector(`label[for="${select.id}"]`)?.textContent ||
      'Choose an option'
    )
      .replace(select.textContent, '')
      .trim();
    if (label) {
      const wrapper = document.createElement('div');
      wrapper.className = 'choice-control';
      label.replaceWith(wrapper);
      wrapper.append(select);
    } else document.querySelector(`label[for="${select.id}"]`)?.remove();
    select.hidden = true;
    select.setAttribute('aria-hidden', 'true');
    select.tabIndex = -1;
    const fieldset = document.createElement('fieldset');
    fieldset.className = 'visible-choices';
    const legend = document.createElement('legend');
    legend.textContent = title;
    fieldset.append(legend);
    const choices = document.createElement('div');
    choices.className = 'choice-options';
    fieldset.append(choices);
    select.after(fieldset);
    const long = [
      'club-route',
      'evidence-route',
      'evidence-lesson',
      'evidence-project',
      'evidence-stage',
    ].includes(select.id);
    let disclosure = null,
      summary = null;
    if (long) {
      disclosure = document.createElement('details');
      disclosure.className = 'context-choices';
      summary = document.createElement('summary');
      disclosure.append(summary, fieldset);
      select.after(disclosure);
    }
    const sync = () => {
      for (const radio of choices.querySelectorAll('input')) {
        radio.checked = radio.value === select.value;
        radio.closest('label').classList.toggle('selected', radio.checked);
      }
      if (summary)
        summary.textContent = `${title}: ${select.selectedOptions[0]?.textContent || 'None'}`;
      const context = document.getElementById('work-context');
      if (context) {
        context.textContent = [
          'evidence-route',
          'evidence-lesson',
          'evidence-project',
          'evidence-stage',
        ]
          .map((id) => {
            const control = document.getElementById(id);
            return control?.value ? control.selectedOptions[0]?.textContent : '';
          })
          .filter(Boolean)
          .join(' / ');
      }
      if (select.id === 'active-route') {
        const title = document.getElementById('active-route-title');
        if (title) title.textContent = select.selectedOptions[0]?.textContent || '';
      }
    };
    const render = () => {
      const signature = [...select.options]
        .map((o) => o.value + '\u0000' + o.textContent)
        .join('\u0001');
      if (choices.dataset.signature !== signature) {
        choices.dataset.signature = signature;
        choices.replaceChildren();
        for (const option of select.options) {
          const item = document.createElement('label'),
            radio = document.createElement('input'),
            text = document.createElement('span');
          radio.type = 'radio';
          radio.name = 'choice-' + select.id;
          radio.value = option.value;
          radio.disabled = option.disabled;
          text.textContent = option.textContent;
          item.append(radio, text);
          choices.append(item);
          radio.addEventListener('change', () => {
            select.value = radio.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            sync();
          });
        }
      }
      sync();
    };
    const descriptor = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
    Object.defineProperty(select, 'value', {
      configurable: true,
      get() {
        return descriptor.get.call(this);
      },
      set(value) {
        descriptor.set.call(this, value);
        sync();
      },
    });
    select.addEventListener('change', sync);
    select.form?.addEventListener('reset', () => requestAnimationFrame(sync));
    new MutationObserver(render).observe(select, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    adapters.set(select, render);
    render();
  }
  function header() {
    const container = document.querySelector('body > header'),
      nav = container?.querySelector('nav');
    if (!nav) return;
    nav.id = 'main-navigation';
    const button = document.createElement('button');
    button.className = 'navigation-toggle';
    button.type = 'button';
    button.textContent = 'Menu';
    button.setAttribute('aria-controls', nav.id);
    button.setAttribute('aria-expanded', 'false');
    container.insertBefore(button, nav);
    const toggle = (open) => {
      container.classList.toggle('navigation-open', open);
      button.setAttribute('aria-expanded', String(open));
      button.textContent = open ? 'Close menu' : 'Menu';
    };
    button.onclick = () => toggle(button.getAttribute('aria-expanded') !== 'true');
    container.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && button.getAttribute('aria-expanded') === 'true') {
        toggle(false);
        button.focus();
      }
    });
    matchMedia('(min-width: 761px)').addEventListener('change', (e) => {
      if (e.matches) toggle(false);
    });
  }
  function revealContext() {
    const params = new URLSearchParams(location.search),
      save = document.getElementById('save-work');
    if (save && (params.has('lesson') || params.has('project') || location.hash === '#portfolio'))
      save.open = true;
    if (location.hash === '#backups') {
      const panel = document.getElementById('backup-panel');
      if (panel) panel.open = true;
    }
  }
  function readingTools() {
    const assessment = document.querySelector('#assessment article');
    if (assessment) {
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
  function enhance() {
    document.querySelectorAll('select').forEach(enhanceChoice);
    readingTools();
  }
  function enhancePage() {
    header();
    revealContext();
    enhance();
    // Only examine newly inserted DOM; adapters observe their own option updates.
    new MutationObserver((records) => {
      if (
        records.some((r) =>
          [...r.addedNodes].some(
            (n) =>
              n.nodeType === 1 &&
              (n.matches('select,.project-stages') || n.querySelector('select,.project-stages')),
          ),
        )
      )
        enhance();
    }).observe(document.body, { childList: true, subtree: true });
    document.addEventListener('click', (e) => {
      if (e.target.closest('.artifact-actions button')?.textContent === 'Edit') {
        const save = document.getElementById('save-work');
        if (save) save.open = true;
      }
    });
  }
  enhancePage();
}
