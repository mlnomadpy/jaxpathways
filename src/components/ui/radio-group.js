// @ts-check
import { escapeHtml } from '../../lib/html.js';

/**
 * @typedef {{ value: string, label: string, disabled?: boolean }} RadioOption
 * @typedef {{ name: string, legend: string, options: RadioOption[], value?: string,
 * required?: boolean, disabled?: boolean }} RadioGroupProps
 */

/**
 * Labels are plain text by default. A trusted renderer can supply sanitized rich labels.
 * @param {RadioGroupProps} props
 * @param {(label: string) => string} [renderLabel]
 */
export function renderRadioGroup(props, renderLabel = escapeHtml) {
  return `<fieldset class="ui-radio-group" data-variant="answers"><legend>${escapeHtml(props.legend)}</legend>${props.options.map((option) => `<label><input type="radio" name="${escapeHtml(props.name)}" value="${escapeHtml(option.value)}"${props.required ? ' required' : ''}${props.disabled || option.disabled ? ' disabled' : ''}${props.value === option.value ? ' checked' : ''}><span>${renderLabel(option.label)}</span></label>`).join('')}</fieldset>`;
}

/**
 * DOM adapter for options populated or refreshed in the browser; no HTML injection.
 * @param {Document} document
 * @param {RadioOption & { name: string, required?: boolean }} option
 */
export function createRadioOption(document, option) {
  const label = document.createElement('label');
  const input = document.createElement('input');
  const text = document.createElement('span');
  input.type = 'radio';
  input.name = option.name;
  input.value = option.value;
  input.required = Boolean(option.required);
  input.disabled = Boolean(option.disabled);
  text.textContent = option.label;
  label.append(input, text);
  return { label, input };
}
