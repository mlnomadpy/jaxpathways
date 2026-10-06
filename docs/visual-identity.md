# Visual identity: the transformation atlas

The course should be recognizable through its subject: a small function becomes a differentiable, batched, compiled program, then a system. The signature is a branching computational diagram, not a generic dashboard illustration.

Palette: atlas lavender #eeebfa, paper #ffffff, diagram indigo #30216b, transformation violet #6043c4, annotation orange #d96828, reading ink #252139. Orange marks explanatory annotations, never small low-contrast body text. Dark violet carries action and selection.

Type: Trebuchet MS and Avenir Next for a rounded, approachable display/interface voice; Georgia for sustained lesson reading; system monospace only for code and mathematical notation. Display typography is large and compact; lesson typography remains quiet.

Layout: the homepage opens as one broad lavender diagram workspace with the learning promise on the left and three connected transformations on the right. Code is a secondary inspectable layer. Work domains become broad annotated route entries rather than identical SaaS cards. Reader navigation is a lavender atlas margin around white reading paper.

    promise / start       f(x) ─┬─ grad
                               ├─ vmap
                               └─ jit
    shared foundation ── work domains
    practical project     course entry

Critique before implementation: a cream field guide with serif headlines would repeat the generic editorial style. Rejected it. A dark terminal with neon accents would repeat the developer-site template. Rejected it. Keep the conceptual branch diagram as the single bold visual gesture; avoid ornamental grids, unexplained numbering, gradients, decorative badges and unrelated illustration.

Use the same branching mark in navigation and the same visual grammar in instructional diagrams. Keep actual lesson/exam availability and self-reported progress honest. Respect reduced motion and keyboard focus. Do not rearrange controls or remove course features solely to achieve the look.

## Implemented and checked

The custom branching SVG mark is used in site navigation and favicon. The homepage diagram connects the function to grad/vmap/jit; selection updates the diagram emphasis, example code, input/output preview and lesson link. Code is inspectable through a disclosure. Domain availability stays accessible in its own disclosure.

Identity styles load last across the app and generated assessment pages. The reader uses lavender navigation margins and white reading paper, with matching concept figures. Returning-learner continuation was reduced to an 87px utility strip in the checked phone view.

Build and all 23 tests passed. Desktop and 390px homepage/reader received visual review; course, career, learning and resource pages showed no horizontal document overflow at 390px. Independent design critique caught the continuation size and inherited green disclosure rule; both were corrected. No lesson content, completion flags or runtime receipts were changed by the design work.

Previews: preview-atlas-home.png, preview-atlas-mobile.png and preview-atlas-reader.png. Publication remains pending.
