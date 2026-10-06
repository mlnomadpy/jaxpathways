import test from 'node:test';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';
import { renderRadioGroup, createRadioOption } from '../src/components/ui/radio-group.js';

test('radio group escapes user content and preserves form state', () => {
  const window = new Window();
  window.document.body.innerHTML = `<form>${renderRadioGroup({
    name: 'answer',
    legend: '<choose>',
    required: true,
    value: 'one',
    options: [
      { value: 'one', label: '<img src=x onerror=alert(1)>' },
      { value: 'two', label: 'Two', disabled: true },
    ],
  })}</form>`;
  assert.equal(window.document.querySelector('legend').textContent, '<choose>');
  assert.equal(window.document.querySelector('img'), null);
  const radios = window.document.querySelectorAll('input');
  assert.equal(radios[0].checked, true);
  assert.equal(radios[0].required, true);
  assert.equal(radios[1].disabled, true);
  assert.equal(new window.FormData(window.document.querySelector('form')).get('answer'), 'one');
  window.close();
});

test('dynamic option labels are text and disabled state is retained', () => {
  const window = new Window();
  const { label, input } = createRadioOption(window.document, {
    name: 'setting',
    value: 'x',
    label: '<script>bad()</script>',
    disabled: true,
  });
  assert.equal(label.querySelector('script'), null);
  assert.equal(label.textContent, '<script>bad()</script>');
  assert.equal(input.disabled, true);
  window.close();
});
