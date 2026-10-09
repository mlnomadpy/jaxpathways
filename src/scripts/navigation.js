const initialized = new WeakSet();

function initTheme(/** @type {HTMLElement} */ header) {
  const toggleButton = /** @type {HTMLButtonElement | null} */ (
    header.querySelector('[data-theme-toggle]')
  );
  if (!toggleButton) return;
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem('jaxpathways-theme');
    if (saved === 'dark' || saved === 'light') root.dataset.theme = saved;
  } catch {}
  const activeTheme = () => {
    if (root.dataset.theme === 'dark' || root.dataset.theme === 'light') return root.dataset.theme;
    return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  };
  const sync = () => {
    const dark = activeTheme() === 'dark';
    const actionLabel = dark ? 'Switch to light theme' : 'Switch to dark theme';
    toggleButton.setAttribute('aria-pressed', String(dark));
    toggleButton.setAttribute('aria-label', actionLabel);
    toggleButton.setAttribute('title', actionLabel);
  };
  sync();
  toggleButton.addEventListener('click', () => {
    const next = activeTheme() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try {
      localStorage.setItem('jaxpathways-theme', next);
    } catch {}
    sync();
  });
  if (typeof matchMedia === 'function') {
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', sync);
  }
}

/** Enhance server-rendered navigation while keeping links usable without JavaScript. */
export function initNavigation() {
  const header = /** @type {HTMLElement | null} */ (document.querySelector('.site-header'));
  const button = /** @type {HTMLButtonElement | null} */ (
    header?.querySelector('.navigation-toggle')
  );
  const nav = /** @type {HTMLElement | null} */ (header?.querySelector('#main-navigation'));
  if (!header || !button || !nav || initialized.has(header)) return;
  initialized.add(header);
  initTheme(header);

  /**
   * @param {boolean} open
   * @param {boolean} [restoreFocus]
   */
  const toggle = (open, restoreFocus = false) => {
    header.classList.toggle('navigation-open', open);
    button.setAttribute('aria-expanded', String(open));
    button.textContent = open ? 'Close menu' : 'Menu';
    if (restoreFocus) button.focus();
  };
  button.hidden = false;
  button.addEventListener('click', () => {
    toggle(button.getAttribute('aria-expanded') !== 'true');
  });
  header.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && button.getAttribute('aria-expanded') === 'true') {
      toggle(false, true);
    }
  });
  nav.addEventListener('click', (event) => {
    const target = /** @type {Element | null} */ (event.target);
    if (target?.closest?.('a')) toggle(false);
  });
  matchMedia('(min-width: 761px)').addEventListener('change', (event) => {
    if (event.matches) toggle(false);
  });
}
