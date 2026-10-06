/* Copy a command or prompt; installation and chat remain user initiated. */
export function initHomeCopy() {
  for (const button of document.querySelectorAll('[data-home-copy]')) {
    button.addEventListener('click', async () => {
      const id = button.dataset.homeCopy;
      const source = document.getElementById(id);
      const status = document.getElementById(`${id}-status`);
      if (!source || !status) return;
      try {
        await navigator.clipboard.writeText(source.textContent);
        status.textContent =
          id === 'tutor-install'
            ? 'Copied. Run it in the course folder’s terminal.'
            : 'Copied. Paste it into your agent chat.';
      } catch {
        status.textContent = 'Select the text above and copy it manually.';
      }
    });
  }
}
