# Publishing and course versions

The deployed course lives at https://www.tahabouhsine.com/jaxpathways/.
GitHub Actions builds Astro with `BASE_PATH=/jaxpathways` and that site's origin.

## Create a release

1. Finish the canonical edits in `curriculum/`, `phases/*/*/lesson.json`, projects and learning paths. Regenerate figures and CPU execution receipts if teaching or executable material changes, following `content/README.md`.
2. Bump the package version with `npm version patch --no-git-tag-version` (or `minor` for a substantial expansion).
3. Prepend a matching version, ISO date, title, summary and learner-facing `changes` list to `curriculum/releases.json`. Describe what changed and what learners should revisit. Keep previous entries.
4. Run `npm run release:prepare`. This adds an immutable source snapshot to `curriculum/release-snapshots.json`, identifies revised lessons/phases and retains unchanged revisions. Do not edit an already published snapshot to bypass a failed check.
5. Run `npm run build`, `npm run check` and `npm run check:deployment`. The deployment check exercises `/jaxpathways`, checks canonical/sitemap URLs and restores the local root build afterward.
6. Commit the sources, snapshot, package version and generated companions. Push `main`; verify both the build and Pages deployment succeed and inspect the public version and lesson URLs. Tag that tested commit with the matching `vX.Y.Z` tag.

`npm run build` fails when tracked course sources differ from the latest release snapshot. Stable source hashes ignore JSON formatting and object-key order. Rendering a diagram or rebuilding the website alone does not create a lesson revision. Website-only releases still need an explicit release note and package version when publishing a new version.

## Reader notices

- The footer always links to `updates.html` and the current version.
- Returning readers see a dismissible banner for an unacknowledged release. Existing readers from before v0.2.0 are recognized through their saved reading record.
- A changed lesson or opened phase compares its source hash with the version last visited in that browser. A closed phase in the catalog is not marked as visited.
- These are local browser notices, not email/push subscriptions. Clearing storage or moving devices resets notice history. Checkpoints, exercise evidence and saved reading positions are never reset by a content update.
- v0.2.0 establishes the baseline; prior edit dates are not reconstructed.

## Search discovery

Every authored lesson has a pre-rendered `lesson-<id>.html` page. Existing `lesson.html?lesson=<id>` bookmarks forward to it while retaining pathway, search term and section. The query reader is excluded from indexing; use the permanent pages in new links.

Shared layouts supply unique page metadata, canonical URLs, Open Graph previews and appropriate WebSite/Course/LearningResource JSON-LD. A sitemap derives its routes from the page and curriculum manifests; accurate lesson/phase revision dates supply `lastmod`. Personal workspaces, the query reader and the duplicate printable book are excluded. No ratings, accreditations or search-ranking claims are generated.

The project ships `sitemap.xml` and `robots.txt`, but crawlers read robots.txt at the **domain root**, not `/jaxpathways/robots.txt`. The existing root robots currently allows crawling and advertises the portfolio's master sitemaps. Add `https://www.tahabouhsine.com/jaxpathways/sitemap.xml` to that master sitemap or submit it in the domain's Google Search Console. This repository does not own the portfolio's root robots file. Search Console URL inspection and indexing cannot be verified from a successful build alone.

Implementation references: [Google's JavaScript SEO guidance](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics), [canonical URL guidance](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls), and [sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap).

## Feedback

The footer and each lesson link to the GitHub issue form in `.github/ISSUE_TEMPLATE/course-feedback.yml`. Lesson links include their URL and release version. No issue is submitted automatically.

`src/data/repository.json` records the verified repository visibility. While it is private, the interface explicitly explains that issues are limited to collaborators. Change visibility only with the owner's authorization, then update that record. A public feedback repository is another option if the source should remain private.
