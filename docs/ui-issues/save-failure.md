<!-- jax-ui-audit:save-failure -->

Priority: High

## Problem

Evidence form reports saved and resets even when save returns false.

## Acceptance criteria

Failed persistence retains input and offers export; import reports actual persistence outcome.

## Audit source

UI/UX audit dated 2026-10-03, docs/ui-ux-audit-2026-10-03.md. Findings come from browser inspection and source review; formal assistive-technology verification and failure injection remain separate.

Local implementation work may address this issue before publication. Keep the issue open until the changes are available and its acceptance checks pass.
