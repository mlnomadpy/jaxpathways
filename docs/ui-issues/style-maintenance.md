<!-- jax-ui-audit:style-maintenance -->

Priority: Medium

## Problem

Reader download contrast bug demonstrates specificity collisions across old dark and new light CSS.

## Acceptance criteria

Component tokens and specificity are deliberate; remove obsolete duplicate rules after visual regression checks without framework rewrite.

## Audit source

UI/UX audit dated 2026-10-03, docs/ui-ux-audit-2026-10-03.md. Findings come from browser inspection and source review; formal assistive-technology verification and failure injection remain separate.

Local implementation work may address this issue before publication. Keep the issue open until the changes are available and its acceptance checks pass.
