const initialized = new WeakSet();

/** Enhance server-rendered navigation while keeping links usable without JavaScript. */
export function initNavigation() {
  const header = document.querySelector('.site-header');
  const button = header?.querySelector('.navigation-toggle');
  const nav = header?.querySelector('#main-navigation');
  if (!button || !nav || initialized.has(header)) return;
  initialized.add(header);

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
    if (event.target?.closest?.('a')) toggle(false);
  });
  matchMedia('(min-width: 761px)').addEventListener('change', (event) => {
    if (event.matches) toggle(false);
  });
}
