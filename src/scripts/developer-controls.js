export function initResourceCopy() {
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.addEventListener('click', async () => {
      const command = document.getElementById(button.dataset.copy).textContent;
      const status =
        button.closest('.resource-code').querySelector('.copy-status') ||
        document.getElementById('resource-copy-status');
      try {
        await navigator.clipboard.writeText(command);
        status.dataset.state = 'success';
        status.textContent = 'Command copied. Run it from your course checkout when you are ready.';
      } catch {
        status.dataset.state = 'error';
        status.textContent =
          'Clipboard access is unavailable. Select the command text and copy it manually.';
      }
    });
  }
}
