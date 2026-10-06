// @ts-check
import { createRadioOption } from '../components/ui/radio-group.js';

/** @type {WeakMap<HTMLSelectElement, () => void>} */
const adapters = new WeakMap();
/** @param {HTMLOptionElement} option */
const optionLabel = (option) => option.getAttribute('label') ?? option.textContent ?? '';

/** @param {Document | Element} [root] */
export /** @param {HTMLSelectElement} select */
function enhanceChoices(root = document) {
  const selects = root.querySelectorAll('select');
  for (const select of selects) enhanceChoice(select);
}

/** @param {HTMLSelectElement} select */
export function refreshChoices(select) {
  adapters.get(select)?.();
}

/** @param {HTMLSelectElement} select */
function enhanceChoice(select) {
  if (!select.id || !select.dataset.choices || adapters.has(select)) return;
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
  choices.className = 'choice-options ui-radio-group';
  choices.dataset.variant = select.dataset.choices === 'disclosure' ? 'list' : 'segmented';
  fieldset.append(choices);
  select.after(fieldset);
  const long = select.dataset.choices === 'disclosure';
  /** @type {HTMLDetailsElement | null} */
  let disclosure = null;
  /** @type {HTMLElement | null} */
  let summary = null;
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
      radio.closest('label')?.classList.toggle('selected', radio.checked);
    }
    if (summary) {
      const selected = [...select.options].find((option) => option.value === select.value);
      summary.textContent = `${title}: ${selected ? optionLabel(selected) : 'None'}`;
    }
  };
  const render = () => {
    const signature = [...select.options]
      .map((o) =>
        JSON.stringify([
          o.value,
          optionLabel(o),
          o.disabled,
          o.closest('optgroup')?.disabled,
          select.disabled,
          select.required,
        ]),
      )
      .join('\u0001');
    if (choices.dataset.signature !== signature) {
      choices.dataset.signature = signature;
      choices.replaceChildren();
      for (const option of select.options) {
        const { label: item, input: radio } = createRadioOption(select.ownerDocument, {
          name: 'choice-' + select.id,
          value: option.value,
          label: optionLabel(option),
          disabled:
            select.disabled || option.disabled || Boolean(option.closest('optgroup')?.disabled),
          required: select.required,
        });
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
    attributeFilter: ['disabled', 'label', 'value', 'selected', 'required'],
  });
  adapters.set(select, render);
  render();
}
