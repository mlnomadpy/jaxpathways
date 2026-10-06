function bindCodeCopy() {
  document.querySelectorAll('.copy-snippet').forEach(
    (button) =>
      (button.onclick = async () => {
        const panel = button.closest('.code-panel'),
          feedback = panel.querySelector('.copy-feedback');
        try {
          await navigator.clipboard.writeText(panel.querySelector('code').textContent);
          feedback.textContent = 'Code copied.';
        } catch {
          feedback.textContent = 'Clipboard unavailable. Select the code and copy it manually.';
        }
      }),
  );
}

export { bindCodeCopy };
