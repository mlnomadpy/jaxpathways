# JAX atlas: landing page design

The course should feel like a place to understand computation and choose meaningful work. The tutor is the way to begin, while the page shows the material and outcomes that justify beginning.

## Tokens and composition

- Canvas: pale lavender `#eeebfa`; paper: `#ffffff`; ink: `#252139`.
- Main identity: atlas purple `#30216b`; transformations: violet `#6043c4`.
- Annotation: orange `#d96828`, limited to diagram markers and large decorative detail.
- Display and controls: Avenir Next, with Trebuchet MS and sans-serif fallbacks. Prose: Georgia for short course explanations. Monospace reserved for code and diagram notation.
- Left-aligned large course promise beside a branching JAX diagram. The diagram is the signature gesture. No decorative gradients or stock imagery.
- Compact tutor command band, followed by visible work directions, real curriculum rows, two project previews and one offline book.

```text
Brand                  Course / Learning paths / Careers / My learning / Resources

Learn JAX.                         function → grad ─┐
From arrays                        function → vmap ┼→ model
 to models.                        function → jit ─┘
[Start with a tutor] [Explore course]

Tutor command + required workspace download + start prompt

Choose your work: Training / Inference / Research / Systems
Real outcomes + availability + direct path/career links

A look inside the course: ordered real lesson previews

Regression experiment             Neural classifier
Concept diagram / project link    Concept diagram / project link

Offline edition: visual book + EPUB / printable reader
Source / maintainer / feedback
```

## Review against the brief

The diagram, code band and real lesson sequence are specific to JAX. Keep the existing branching mark and color family so the new landing still belongs to the course. Use different structures for different purposes: a learning diagram, compact setup band, destination list, lesson rows and project surfaces. Avoid repeatedly enclosing explanatory copy in identical cards. Career links remain both in the header and beside each work direction. All choices are visible; setup details are optional and prerequisites stay beside the command.

Availability and lesson previews are generated from the canonical curriculum. Project visuals explain their mathematical task; they do not invent execution results. The existing course workspace is required for the local npx command.

## Implementation status

Implemented in `site/index.html` and standalone `site/home.css`. The desktop diagram has a simplified mobile companion. `scripts/render-home.mjs` renders real lesson previews and path availability during the existing build. Legacy course-overview and other-ways anchors, the saved reading panel, and tutor copy IDs remain available.

Astro migration is deferred at the user's request: finish the landing first. The page sections and source-derived renderer provide a clear boundary for later Astro components; no framework dependencies or deployment changes were added.

Verification: `npm run build` and `npm run check` pass (29 tests). Additional checks cover generated-section idempotence, local files and query IDs, anchor destinations, four directions, six authored lesson previews, and clipboard success/denial behavior. Browser access to the local preview was denied, so desktop/mobile screenshots and real browser interaction checks remain outstanding.

## Astro migration

The landing is now composed in `src/pages/index.astro` from `src/components/home/`, with copy in `src/data/home.json` and styles in `src/styles/home/`. This supersedes the implementation paths and migration deferral above. See [the migration guide](astro-migration.md).
