import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';
import { enhanceChoices } from '../src/scripts/choices.js';
import { populateSelect, setSelectValue } from '../src/scripts/browser-actions.js';

test('choice enhancement is opt-in, repeatable and keeps native select properties', async () => {
  const window = new Window();
  globalThis.document = window.document;
  globalThis.MutationObserver = window.MutationObserver;
  globalThis.Event = window.Event;
  globalThis.requestAnimationFrame = window.requestAnimationFrame.bind(window);
  document.body.innerHTML =
    '<form><label>Terminal<select id="terminal" data-choices="inline" data-choice-label="Terminal instructions"><option value="unix">Unix</option><option value="windows">Windows</option></select></label><label>Plain<select id="plain"><option>Keep native</option></select></label></form>';
  const select = document.querySelector('#terminal');
  void select.value;
  const nativeValue = Object.getOwnPropertyDescriptor(select, 'value');
  try {
    enhanceChoices();
    enhanceChoices();
    assert.equal(document.querySelectorAll('fieldset').length, 1);
    assert.equal(document.querySelector('legend').textContent, 'Terminal instructions');
    assert.equal(document.querySelector('#plain').hidden, false);
    assert.deepEqual(Object.getOwnPropertyDescriptor(select, 'value'), nativeValue);
    setSelectValue(select, 'windows');
    assert.equal(document.querySelector('input[value="windows"]').checked, true);
    let changes = 0;
    select.addEventListener('change', () => changes++);
    const unix = document.querySelector('input[value="unix"]');
    unix.checked = true;
    unix.dispatchEvent(new Event('change', { bubbles: true }));
    assert.equal(select.value, 'unix');
    assert.equal(changes, 1);
    select.options[1].disabled = true;
    await window.happyDOM.whenAsyncComplete();
    assert.equal(document.querySelector('input[value="windows"]').disabled, true);
    select.disabled = true;
    await window.happyDOM.whenAsyncComplete();
    assert.equal(document.querySelector('input[value="unix"]').disabled, true);
    select.disabled = false;
    populateSelect(select, [{ id: 'new', title: 'Changed option' }]);
    assert.equal(document.querySelector('input[value="new"]').checked, true);
    assert.equal(document.querySelectorAll('.choice-options input').length, 1);
  } finally {
    await window.happyDOM.abort();
  }
});

test('disclosure choices refresh after a form reset without changing stored records', async () => {
  const window = new Window();
  globalThis.document = window.document;
  globalThis.MutationObserver = window.MutationObserver;
  globalThis.Event = window.Event;
  globalThis.requestAnimationFrame = window.requestAnimationFrame.bind(window);
  document.body.innerHTML =
    '<form><select id="route" data-choices="disclosure" data-choice-label="Pathway"><option value="a" selected>First</option><option value="b">Second</option></select></form>';
  try {
    enhanceChoices();
    const select = document.querySelector('select');
    setSelectValue(select, 'b');
    assert.equal(document.querySelector('summary').textContent, 'Pathway: Second');
    document.querySelector('form').reset();
    await window.happyDOM.whenAsyncComplete();
    assert.equal(select.value, 'a');
    assert.equal(document.querySelector('input[value="a"]').checked, true);
    assert.equal(document.querySelector('summary').textContent, 'Pathway: First');
  } finally {
    await window.happyDOM.abort();
  }
});
