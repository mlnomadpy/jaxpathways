# Google Rich Results Test validation, 2026-10-20

Test target: published JAX Pathways v0.7.1; correction: v0.7.2. Tests use Google's smartphone URL inspection through the actual Rich Results Test browser UI. Local JSON parsing and schema-vocabulary checks are separate evidence.

## Published baseline

| Published page | Google result | Finding |
| --- | --- | --- |
| `/course.html` | [Result](https://search.google.com/test/rich-results/result?id=0Lk7iSNuhIDzO7xbOp5Uyg) | 45 valid items: one Breadcrumb, 22 Course list items and 22 Paywalled Content items. Non-critical issues in the paywall items. |
| `/phase-guides/arrays.html` | [Result](https://search.google.com/test/rich-results/result?id=AVv4XLHBBNiLxnxod1VaNw) | Three valid items. The course's paywall interpretation reports four optional `Invalid object type for field "hasPart"` issues, one for each lesson. |
| `/lesson-arrays-01.html` | [Result](https://search.google.com/test/rich-results/result?id=6qffrShDJ9a1P9db0PpTZw) | Three valid items, with non-critical issues in the embedded phase Course's paywall interpretation. |
| `/book.html` | [Result](https://search.google.com/test/rich-results/result?id=H-_5cczMjps0P5yn04E-hw) | One valid Breadcrumb item, no reported warnings. Book is not a separately supported rich-result type in this tool. |

Initial catalog attempts encountered Google's reCAPTCHA connection error. A subsequent normal URL test completed; no CAPTCHA challenge was solved or bypassed.

## Cause and correction

The lesson list used Schema.org's general-purpose `Course.hasPart`, containing `LearningResource` objects. This is a valid CreativeWork relationship in Schema.org, but Google's paywall validator also interprets this property and expects paywall page elements. The free course therefore acquired unrelated optional paywall warnings.

The correction uses the course-specific `syllabusSections` property with typed `Syllabus` entries. Each outline has its lesson URL, title and learning objective. It has its own `#syllabus` identity; the actual lesson retains its `#lesson` identity and `isPartOf` course relationship. Free-access declarations remain true. No paid-access fragments or CSS selectors are fabricated.

A regression check rejects `Course.hasPart` and verifies typed syllabus entries, nonempty descriptions and canonical lesson destinations across the generated phase pages.

## Candidate validation before publication

The corrected phase's complete generated HTML was submitted through Google's CODE mode: [candidate result](https://search.google.com/test/rich-results/result?id=ukp8k0m8nlIqEnowI0ZYQw). Google reported three valid items with no non-critical warnings. The same shared course-schema function is used by the catalog, all 22 phase pages and embedded phase entities in the 103 lessons.

## Live retest

The final published URLs must be retested after v0.7.2 deployment. Candidate CODE results are not represented as live URL results. The checked examples cover the shared catalog, phase, lesson and book templates; no claim is made that Google individually tested all 103 lesson URLs.

Google result links expire after approximately 90 days. Rich Results Test checks supported search features, not every Schema.org type or property and not indexing or ranking guarantees. The standalone download reader deliberately remains noindex; the indexable book overview is the search entry point.

## References

- [Google: Rich Results Test and supported types](https://support.google.com/webmasters/answer/7445569)
- [Google: Subscription and paywalled content](https://developers.google.com/search/docs/appearance/structured-data/paywalled-content)
- [Schema.org: syllabusSections](https://schema.org/syllabusSections)
- [Schema.org: Syllabus](https://schema.org/Syllabus)
