<!-- jax-ui-audit:catalog-state -->

Priority: Medium

## Problem

Route selection resets after reload; filters are not serialized.

## Acceptance criteria

Path, topic, environment and roadmap state round-trip through URL and history.

## Audit source

UI/UX audit dated 2026-10-03, docs/ui-ux-audit-2026-10-03.md. Findings come from browser inspection and source review; formal assistive-technology verification and failure injection remain separate.

Local implementation work may address this issue before publication. Keep the issue open until the changes are available and its acceptance checks pass.
