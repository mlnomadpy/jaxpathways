const adapters = new WeakMap();

export function enhanceChoices(root = document) {
  const selects = root.querySelectorAll('select[data-choices]');
  for (const select of selects) enhanceChoice(select);
}

export function refreshChoices(select) {
  adapters.get(select)?.();
}

function enhanceChoice(select) {
  if (!select.dataset.choices || adapters.has(select)) return;
  const label = select.closest('label');
  const title = select.dataset.choiceLabel;
  if (!title) return;
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
  const long = select.dataset.choices === 'disclosure';
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
  };
  const render = () => {
    const signature = [...select.options]
      .map((o) => JSON.stringify([o.value, o.textContent, o.disabled, select.disabled]))
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
        radio.disabled = select.disabled || option.disabled;
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
  select.addEventListener('change', sync);
  select.form?.addEventListener('reset', () => requestAnimationFrame(render));
  new MutationObserver(render).observe(select, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    attributeFilter: ['disabled', 'label', 'value', 'selected'],
  });
  adapters.set(select, render);
  render();
}
