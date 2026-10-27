# Full-book Chromium print baseline, edition 0.16.0

The entire frozen HTML book was printed without pruning chapters: 1,254 A4 pages and 1,368 Letter pages. The PDFs remain private review artifacts under `/private/tmp/jaxpathways-full-print-016/output/pdf/`; the receipt pins their hashes and the exact source HTML.

All pages were scanned for text glyphs beyond their media boxes: none were found, and no page had an empty text extraction. Two contact sheets cover 34 selected pages, including math, a Transformer diagram, code, solution appendices, TPU precision and book boundaries. This is a full-book geometry check with sampled visual review, not every-page visual acceptance.

Changes requested: Letter page 223 leaves the least-squares figure title and prediction before the chart on page 224. The export also has no page-number footer. Repair and reprint the next candidate; keep this frozen baseline unchanged. Native EPUB, screen-reader speech and other print engines remain unqualified.

The first two harness attempts stopped before producing PDFs: one sandbox launch refusal, then an evicted large navigation response. The final harness reads source bytes through an API response before navigating; both completed PDFs use the same exact frozen HTML. Logs are preserved.
