# Course architecture adapted for JAX

Reference inspected: [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/3be078b37ffd8f0c04953c0678e48f5c6d0c7775), particularly its CNN lesson, lesson reader, build script, public-resource contract, lesson runner, pathway manifests, and tutor instructions. This project uses original JAX content and implementation.

| Reference pattern | JAX adaptation | Status |
|---|---|---|
| Phase-based lesson bundles | Narrative, code, notebook, quiz, and evidence template in each authored lesson folder | 19 bundles |
| Shared route manifests | Pathways reference shared phases and prerequisites | 10 manifests |
| Dedicated reader with phase menu and page outline | Full-page reader with direct links and route-aware navigation | Implemented |
| Problem, explanation, implementation, application, deliverable | Nine array/transformation lessons include derivations, experiments, transfer practice, and diagnosis | Depth rewrite for nine; ten authored lessons still need it |
| Public machine-readable resources | Static catalog, pathways, roles, lesson JSON, OpenAPI, and agent index | Generated resources |
| Repository lesson runner | CLI discovery, reading, CPU execution, quizzes, and career plans | Implemented |
| Tutor skills and persistent tutor state | Four learner tutors and one course authoring skill | Packaged; installation optional |
| Interactive concept figures | Analytic gradient figure with keyboard control | One figure |
| Books and project stages | Offline HTML/EPUB and four-stage foundation project | Implemented; CPU reference checked |
| Translation and advanced deployment | Require reviewed translated content and hosting-specific implementation | Pending |
| CI validation | Build, contracts, CPU lessons, and reference project checks | Workflow added; remote execution pending |

The static course resources do not reproduce the reference’s server-side content negotiation. They are files served by the existing site host. Quiz correctness, self-reported exercise evidence, CPU execution, TPU execution, expert review, and pathway readiness remain distinct. No blanket feature parity is claimed.

## Authoring and validation

Edit the course manifest, pathway manifests, and authored lesson JSON. Run `npm run build`, `npm run check`, and `npm run smoke:cpu`. The build generates website data and readable/executable lesson companions. Checks compare downloadable and repository artifacts, retain planned status, reject unknown execution targets, and verify route prerequisites.
