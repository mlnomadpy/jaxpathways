# UI component layer

The first primitives live in `src/components/ui`. They are local source files, designed to be edited like shadcn components, using Astro and native HTML instead of introducing React hydration. This is an initial component layer, not an installation of shadcn/ui or a completed migration of all app controls.

## ChoiceField

```astro
---
import ChoiceField from '../components/ui/ChoiceField.astro';
---
<ChoiceField id="duration" name="duration" label="Schedule" variant="segmented">
  <option value="4">4 weeks</option>
  <option value="8" selected>8 weeks</option>
</ChoiceField>
```

Props: `id` and `label` are required; `variant` is `segmented` (default) or `list`; `name`, `required`, and `disabled` are optional. Use the named `label` slot for additional caption content. Default slot accepts native options. Lists use a disclosure for many or long choices. Segments wrap at narrow widths.

Organizer and notebook controls use this component. Existing page controllers can populate its native select and listen to change events. `enhanceChoices()` converts the select using the shared DOM option primitive; `refreshChoices(select)` refreshes a dynamic control. The native select remains the no-JavaScript fallback.

## RadioGroup

```astro
---
import RadioGroup from '../components/ui/RadioGroup.astro';
---
<RadioGroup
  name="answer"
  legend="Choose one answer"
  required
  options={[{ value: 'a', label: 'First answer' }, { value: 'b', label: 'Second answer' }]}
/>
```

Optional `value` selects an initial answer; `disabled` disables the group; options can be individually disabled. Native radios supply keyboard behavior, exclusivity and validation. The `answers` variant uses lettered rows; the letters are visual markers, not keyboard shortcuts.

String-rendered lesson content uses the same `renderRadioGroup()` function as the Astro component. Labels are escaped by default. Its optional label renderer is only for trusted, sanitized markup (the lesson renderer supplies `inlineMath`). Client-populated settings use `createRadioOption()`, which writes labels with `textContent`.

## Styling and behavior

Shared styles are in `src/styles/shared/choices.css`; colors and typography come from `src/styles/tokens.css`. `ui-radio-group` and `data-variant` define the public styling hooks. Page CSS controls placement, not selection state. Motion lasts 200ms, responds only to interaction and is disabled for reduced-motion preferences. Keyboard focus and forced-color selection remain visible.

Next components should follow the same model: local source, explicit props and variants, accessible native behavior, and documented examples. Buttons, text fields and disclosures still use shared CSS and have not yet been migrated to this API.
