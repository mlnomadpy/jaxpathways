# Book reading qualification

Structural export checks are automated. This matrix records reading-environment evidence separately and must be updated after actual native checks.

| Environment | Evidence | Status |
| --- | --- | --- |
| Standalone HTML, embedded fonts and figures | `tests/course-reader.test.mjs`, `scripts/check-distribution.py` | Automated structural checks; native visual result recorded separately |
| EPUB package and MathML conformance | Official EPUBCheck 5.4.0, edition 0.13.0: zero errors and zero warnings; [report](evidence/course-review-0.13/epubcheck-report.json) | Passed automated conformance |
| Apple Books EPUB | Native attempt blocked by the locked desktop on 2026-10-21; no retained reading session | Pending |
| Second EPUB reader, such as Thorium | No retained native reading session | Pending |
| Screen reader equation speech and reading order | No retained assistive-technology session | Pending |
| Chromium print-to-PDF, A4 and Letter | Edition 0.11.1: four selected chapters, 72 A4 / 79 Letter pages; contact-sheet inspection of all 151 sample pages, focused full-resolution math/code/figure checks, and glyph-bound checks. [Print receipt](evidence/reader-parity-0.11.1/print-receipt.json), [geometry check](evidence/reader-parity-0.11.1/geometry-check.json) | Passed within the selected Chromium sample scope; full-book and other engines pending |

For each session record edition, source revision, application and version, OS, font size/zoom, viewport, page size, margins and screenshots or a PDF. Use welcome-01, optimization-09, transformers-04 and tpu-06 as beginner, math, long-code and multi-panel samples. Check TOC navigation, next/previous reading order, the solution appendix, figures and captions, equation semantics, zoom and code overflow. For assistive technology, record what is actually spoken rather than inferring it from MathML presence.

For print use A4 and US Letter, portrait, 100% scale, default margins and backgrounds enabled. Inspect page breaks through equations, long code and figure-caption pairs. Record exact pages of clipping. Create focused issues with reproductions, fix them, and rerun that environment before marking it supported.

The [native qualification issue](https://github.com/mlnomadpy/jaxpathways/issues/56) remains open. This matrix is a work protocol, not a claim that pending applications have been tested.

The 0.11.1 sample repair adds all authored learner contracts to the standalone HTML reader and prints chapter-labeled solutions and checkpoint reasoning. The inspected samples found and repaired split reasoning boxes, split answer blocks, and low-contrast starter syntax. The final ink-color change preserved every sampled glyph position and text value; the final starter, math and figure rasters are retained beside the receipt. This is internal automated/visual QA in isolated Chromium, not native EPUB qualification, actual assistive-technology speech, human teaching acceptance or inspection of every full-book page.
