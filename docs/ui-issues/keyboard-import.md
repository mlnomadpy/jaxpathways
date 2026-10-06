<!-- jax-ui-audit:keyboard-import -->

Priority: Medium

## Problem

Visible import label targets a hidden file input without keyboard activation.

## Acceptance criteria

Focusable import button opens chooser; invalid files report errors without changing existing state.

## Audit source

UI/UX audit dated 2026-10-03, docs/ui-ux-audit-2026-10-03.md. Findings come from browser inspection and source review; formal assistive-technology verification and failure injection remain separate.

Local implementation work may address this issue before publication. Keep the issue open until the changes are available and its acceptance checks pass.
