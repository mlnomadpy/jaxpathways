# Open issue review, 6 October 2026

The repository has 28 open issues, all created from the 3 October UI audit. None currently has a label or follow-up comment. The original audit described 38 authored lessons and two staged projects; the current course has 97 authored lessons and 19 staged projects. Treat the issue descriptions as historical observations, then verify each acceptance criterion against the deployed revision.

This is a source and automated-test review, not a fresh screen-reader or device audit. No issues were closed or edited. An implementation below is a candidate for acceptance testing, not proof that every original criterion passes.

## Recommended priorities

1. Verify learner data preservation first: issues #6, #7 and #8. Implementations and failure-injection tests exist, but losing a learner's notes is more costly than a visual defect.
2. Finish the reading experience: #19 and #27. The downloadable HTML book lacked the richer code presentation used by the lesson reader. Math fonts/styles also depended on external assets when saving just the HTML. The current change addresses those book-specific gaps; screen-reader, zoom and print-layout review remain separate.
3. Verify the full start, practice, recover and resume journey: #3, #13, #14, #16 and #23. A correct component is not enough if the learner loses context between pages or extracted workspaces.
4. Separate pedagogical work from UI fixes: #20 and #22 combine authored guidance with human assessment and review operations. Keep reviewer validation and actual review handoff as explicit outstanding work.
5. Reconcile the backlog with the deployed product. Use labels such as `bug`, `accessibility`, `learning-content`, and `needs-verification`; attach the deployed revision and a concrete result before closing an issue.

## Issue-by-issue assessment

| Issue | Current evidence | Assessment |
| --- | --- | --- |
| [#1 Domain exploration](https://github.com/mlnomadpy/jaxpathways/issues/1) | The old choices handler has been replaced; domain and route UI now have separate controllers. | Reproduce all four domain flows on the deployed site before closing. |
| [#2 Download contrast](https://github.com/mlnomadpy/jaxpathways/issues/2) | Shared button styles and lesson controls have changed since the audit. | Needs computed default, hover and focus contrast checks. |
| [#3 Section history](https://github.com/mlnomadpy/jaxpathways/issues/3) | Reader handles section hashes; browser migration tests cover permanent lesson links retaining section and route. | Candidate for closing after back/forward and reload verification. |
| [#4 Catalog URL state](https://github.com/mlnomadpy/jaxpathways/issues/4) | Browser migration tests cover route, roadmap and history filtering under a deployment base. | Candidate for acceptance verification across all filters. |
| [#5 Mobile command overflow](https://github.com/mlnomadpy/jaxpathways/issues/5) | Code containers use local horizontal overflow and width constraints. | Needs a real narrow-screen check, especially long commands. |
| [#6 Complete backup](https://github.com/mlnomadpy/jaxpathways/issues/6) | Version 2 includes notebook, lesson progress, projects, career plan and reading state. Round-trip and compatibility tests exist. | Strong closure candidate after an exported-file smoke check. |
| [#7 Failed persistence](https://github.com/mlnomadpy/jaxpathways/issues/7) | Failure-injection tests retain evidence inputs and exercise backup persistence rollback. | Strong closure candidate; verify export remains usable when storage fails. |
| [#8 Keyboard import](https://github.com/mlnomadpy/jaxpathways/issues/8) | Backup workflow supports preview, cancellation and invalid-file rejection. | Verify the actual chooser trigger and focus using a keyboard. |
| [#9 Pathway source](https://github.com/mlnomadpy/jaxpathways/issues/9) | `routeViewContext` builds `learning-paths/<id>.json` URLs. | Narrow, likely resolved; inspect all ten generated links. |
| [#10 Project route IDs](https://github.com/mlnomadpy/jaxpathways/issues/10) | Canonical project routes and prerequisite links are validated/rendered. | Verify invalid-route rejection and preserved context before closure. |
| [#11 Career availability](https://github.com/mlnomadpy/jaxpathways/issues/11) | Coverage derives from current manifests; tests distinguish focus phases from preparation. | Historical problem largely addressed; review every milestone label. |
| [#12 Focused homepage](https://github.com/mlnomadpy/jaxpathways/issues/12) | Course catalog, careers and projects have separate pages. | Recheck mobile hierarchy and legacy links; avoid another speculative redesign. |
| [#13 Resume position](https://github.com/mlnomadpy/jaxpathways/issues/13) | Reader stores lesson, path and section; home continuation has a migration test. | Candidate for closure after an actual cross-page resume. |
| [#14 Career handoff](https://github.com/mlnomadpy/jaxpathways/issues/14) | Career tests choose authored prerequisite starts and preserve route plans. | Verify role URL to first lesson end to end. |
| [#15 Long lesson hierarchy](https://github.com/mlnomadpy/jaxpathways/issues/15) | Reader groups navigation and marks the current section. | Needs a mobile reading session with a long lesson, not just DOM assertions. |
| [#16 Practice setup](https://github.com/mlnomadpy/jaxpathways/issues/16) | Setup help, OS-specific commands, beginner workspace and course workspace exist. | Verify extraction and notebook launch from a clean learner directory. |
| [#17 Prerequisite links](https://github.com/mlnomadpy/jaxpathways/issues/17) | Reader now constructs links to prerequisite lessons/phases. | Narrow closure candidate after checking authored/planned destinations. |
| [#18 Explanatory diagrams](https://github.com/mlnomadpy/jaxpathways/issues/18) | All authored lessons have visual artifacts; mechanism figures, captions and interpretation have automated checks. | Original two-figure count is obsolete. Retain subject-matter review of explanatory quality. |
| [#19 Code presentation](https://github.com/mlnomadpy/jaxpathways/issues/19) | Lesson reader already has highlighting, labels, copy controls and cumulative builds; downloadable book lagged behind. | Active reader improvement. Split any remaining build-workflow gaps from book typography. |
| [#20 Feedback and assessments](https://github.com/mlnomadpy/jaxpathways/issues/20) | Formative explanations and synthesis assessment drafts exist; review status stays explicit. | Partially addressed. Authorship is not expert review or proof of readiness. |
| [#21 Contextual evidence](https://github.com/mlnomadpy/jaxpathways/issues/21) | Evidence supports associations, editing, filtering, undo and self-reported status. | Candidate for closure after edit/delete/undo and backup round-trip checks. |
| [#22 Review handoff](https://github.com/mlnomadpy/jaxpathways/issues/22) | UI says not reviewed; review packets exist. | Split resolved wording from any future human review submission/response workflow. |
| [#23 Project guidance](https://github.com/mlnomadpy/jaxpathways/issues/23) | Workbench includes setup and stage commands; project guides and standalone ZIPs exist. | Check pass/fail diagnosis and extracted-folder commands per project. |
| [#24 Search](https://github.com/mlnomadpy/jaxpathways/issues/24) | Search covers prose, diagnostics, experiments, practice and aliases with excerpts. | Partial: search returns excerpts but no matched section target. Add section links and fuller API/code indexing. |
| [#25 Cohort planning](https://github.com/mlnomadpy/jaxpathways/issues/25) | Planner has its own page and tests for duration budgets, prerequisite blocking and unwritten work. | Strong closure candidate after checking the rendered weekly plan. |
| [#26 Resource discovery](https://github.com/mlnomadpy/jaxpathways/issues/26) | Learner resources page exposes books, tutor skills, workspaces and CLI guidance. | Historical absence is resolved; verify navigation and downloads. |
| [#27 Accessibility](https://github.com/mlnomadpy/jaxpathways/issues/27) | Targeted statuses, focus and current-location semantics exist in source. | Keep open until actual assistive-technology, zoom, contrast and touch verification is recorded. |
| [#28 Style consolidation](https://github.com/mlnomadpy/jaxpathways/issues/28) | Shared/component/feature CSS separation has regression tests. | Substantial progress. Remaining visual collisions require rendered-page checks. |

## Additional defect found during the reader work

An exact-source check exposed a shared syntax-highlighting bug: Python numeric literals containing underscores could disappear from displayed code. For example, `100_000_000.` became `.`. The fix preserves every character, and the regression suite compares all highlighted book blocks with their canonical lesson sources. This belongs with #19 but is a code-integrity defect, not just visual polish.

## Scope advice

Do not start 28 new implementations. First close verified historical defects, split mixed issues, and keep a short list of reproducible remaining problems. Add independent content-review work for careers and TPU qualification, rather than implying that a UI audit validates those subjects. A small release that improves reading, preserves learner work and supplies a complete practice journey will help learners more than expanding the catalog again.
