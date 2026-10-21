# Book reading qualification

Structural export checks are automated. This matrix records reading-environment evidence separately and must be updated after actual native checks.

| Environment | Evidence | Status |
| --- | --- | --- |
| Standalone HTML, embedded fonts and figures | `tests/course-reader.test.mjs`, `scripts/check-distribution.py` | Automated structural checks; native visual result recorded separately |
| Apple Books EPUB | No retained native reading session | Pending |
| Second EPUB reader, such as Thorium | No retained native reading session | Pending |
| Screen reader equation speech and reading order | No retained assistive-technology session | Pending |
| Chrome print-to-PDF pagination | No retained printed PDF inspection | Pending |

For each session record edition, source revision, application and version, OS, font size/zoom, viewport, page size, margins and screenshots or a PDF. Use welcome-01, optimization-09, transformers-04 and tpu-06 as beginner, math, long-code and multi-panel samples. Check TOC navigation, next/previous reading order, the solution appendix, figures and captions, equation semantics, zoom and code overflow. For assistive technology, record what is actually spoken rather than inferring it from MathML presence.

For print use A4 and US Letter, portrait, 100% scale, default margins and backgrounds enabled. Inspect page breaks through equations, long code and figure-caption pairs. Record exact pages of clipping. Create focused issues with reproductions, fix them, and rerun that environment before marking it supported.

The [native qualification issue](https://github.com/mlnomadpy/jaxpathways/issues/56) remains open. This matrix is a work protocol, not a claim that pending applications have been tested.
