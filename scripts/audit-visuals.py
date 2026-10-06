"""Audit every curriculum entry against actual figures, execution records and teaching explanations."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
course=json.loads((ROOT/'public/curriculum.json').read_text())
before={r['id']:r for r in json.loads((ROOT/'docs/visual-audit-before.json').read_text())}
concepts=json.loads((ROOT/'src/data/concept-figures.json').read_text())
planned={
'welcome-03':('CPU versus real TPU execution boundary','Show device discovery and a verified computation on the named TPU runtime; installation logs alone do not establish successful execution.'),
'recovery-05':('Checkpoint event timeline','Contrast requested, writing and completed checkpoints; mark which failures can be recovered.'),
'performance-03':('Annotated real profiler trace','Connect host dispatch, device work and synchronization; label the captured workload and hardware.'),
'performance-04':('Roofline with measured workload points','Derive arithmetic intensity, then compare measured throughput with the named device limits.'),
'performance-05':('Memory versus recomputation plot','Compare peak memory and synchronized step time on the same workload; show what was rematerialized.'),
'distributed-03':('Collective communication diagram and byte counts','Track ownership before and after each collective and distinguish algorithmic volume from measured latency.'),
'distributed-04':('Multi-host failure and recovery timeline','Show checkpoint completion, host loss and restart; require a real multi-host experiment for recovery claims.'),
'science-01':('Trajectory with conservation-error plot','Connect a state update to physical motion and a known invariant.'),
'science-02':('Ensemble trajectories and time-axis diagram','Distinguish the independent batch axis from recurrent simulation time.'),
'science-03':('Solver gradient versus finite differences','Show sensitivity to parameters and integration step size alongside an independent derivative check.'),
'science-04':('Observed and fitted trajectories with residuals','Separate parameter recovery, prediction error and scale-out performance evidence.'),
'probability-01':('Density and empirical histogram','Explain probability mass, density units and why histogram height is not a point probability.'),
'probability-02':('Posterior regression bands','Show predictive uncertainty separately from uncertainty in the mean function.'),
'probability-03':('Trace, autocorrelation and chain diagnostics','Connect mixing failures to correlated draws; do not infer convergence from a smooth histogram alone.'),
'probability-04':('Posterior predictive checks','Compare observed statistics with replicated draws and explain an approximation failure.'),
'rl-01':('State-transition and reward timeline','Follow one actual rollout with observations, actions, rewards and termination flags.'),
'rl-02':('Batched rollout grid','Show which axis indexes environments and which indexes time, including reset boundaries.'),
'rl-03':('Policy-ratio and clipping plot','Explain the surrogate objective on both sides of its clipping interval with measured update diagnostics.'),
'rl-04':('Return distributions across seeds','Report spread and evaluation protocol; do not substitute one best training curve.'),
'kernels-01':('Grid-to-array tile mapping','Label program IDs, block origins and boundary masks before presenting kernel code.'),
'kernels-02':('Kernel output and reference error map','Pair the first real TPU result with an independent numerical reference; keep CPU illustrations labeled.'),
'kernels-03':('Memory and pipeline timeline','Distinguish conceptual overlap from measured device traces and expose synchronization boundaries.'),
'kernels-04':('Correctness-versus-throughput comparison','Show size/dtype coverage and repeat measurements on the named device rather than one speedup.'),
'internals-02':('Forward and reverse sensitivity graph','Connect primal values, tangents and cotangents with independently checked directional products.'),
'internals-03':('Custom derivative versus numerical reference','Expose domain boundaries and check the custom rule at smooth points and changed inputs.'),
'internals-04':('Source-to-jaxpr-to-lowering map','Follow the same operation through representations without treating text size as runtime cost.'),
'deployment-02':('Adaptation loss and held-out quality curves','Separate the training objective from downstream task quality and include an unadapted baseline.'),
'deployment-04':('Latency-throughput saturation curve','Measure a named request distribution, batch policy and hardware; include tail latency.'),
'operations-01':('Job lifecycle and resource ownership diagram','Identify provisioning, readiness, execution and teardown boundaries with actual status evidence.'),
'operations-02':('Correlated metrics and log timeline','Tie a workload failure to relevant signals rather than showing a decorative dashboard.'),
'operations-03':('Incident and recovery timeline','Separate detection, diagnosis, mitigation and validated recovery in one reproducible scenario.'),
'operations-04':('Artifact lineage and rollout comparison','Connect versioned inputs to deployed output and a concrete rollback checkpoint.'),
'operations-05':('Utilization and cost per useful result','Define the denominator and include idle time; distinguish quoted prices from measured consumption.')}
rows=[]
for phase in course['phases']:
 for lesson in phase['lessons']:
  old=before.get(lesson['id'], {'interactiveFigure':False,'shapeSketches':None})
  old_visual=bool(old['interactiveFigure'] or lesson['id'] in concepts)
  if lesson['status']=='authored':
   visual=lesson['content']['visual'];artifact=lesson.get('visualArtifact');execution=lesson.get('execution')
   notebook=json.loads((ROOT/lesson['artifacts']['notebookSource']).read_text())
   code=[c for c in notebook['cells'] if c['cell_type']=='code']
   record=json.loads((ROOT/lesson['path']/'outputs/visual.json').read_text())
   assert record['contentHash']==lesson['contentHash']
   assert artifact and execution and execution['contentHash']==lesson['contentHash']
   assert all(c['execution_count'] is not None for c in code)
   assert any(o.get('output_type')=='display_data' and 'image/png' in o.get('data',{}) for c in code for o in c['outputs'])
   rows.append(dict(id=lesson['id'],phase=phase['id'],title=lesson['title'],status='authored',previousVisual=old_visual,baselinePresent=lesson['id'] in before,previousShapeSketches=old['shapeSketches'],visual=visual['title'],kind=visual['kind'],explanation=visual['connection'],figureData=lesson['path']+'/outputs/visual.json',executedNotebook=True,review='CPU executed; figure/lesson relationship inspected; learner review pending'))
  else:
   title,explanation=planned[lesson['id']]
   rows.append(dict(id=lesson['id'],phase=phase['id'],title=lesson['title'],status='planned',previousVisual=False,visual=title,explanation=explanation,review='Specification only: lesson and runtime evidence missing'))
report={'scope':f'All {len(rows)} canonical entries across {len(course["phases"])} phases and all {len(course["pathways"])} routes. Existing authored sources, generated outputs and artifact provenance were inspected. This is a visual teaching and delivery audit, not independent expert or learner validation.',
 'findings':['The follow-up interpretation audit rewrites every authored walkthrough against its source computation and saved plot data; see docs/plot-interpretation-audit.md for corrected misconceptions and review limits.','The original reader exposed expected output and pass receipts but not captured executions.','Downloaded notebooks originally had empty execution counts and output arrays.','Existing interactive and conceptual figures covered only a small subset; shape sketches remained code-like text.','Books omitted the existing client-rendered figures.','Planned entries have no lesson or execution to visualize and must remain clearly planned.'],
 'counts':{'entries':len(rows),'authored':sum(r['status']=='authored' for r in rows),'planned':sum(r['status']=='planned' for r in rows),'previousFigureCoverage':sum(r['previousVisual'] for r in rows),'executedPlots':sum(r.get('kind')=='executed' for r in rows),'conceptualDiagrams':sum(r.get('kind')=='conceptual' for r in rows)},'lessons':rows}
(ROOT/'docs/visual-learning-audit.json').write_text(json.dumps(report,indent=2)+'\n')
md='# Visual learning audit\n\n'+report['scope']+'\n\n## Findings and changes\n\n'+''.join('- '+s+'\n' for s in report['findings'])+'\nAll '+str(report['counts']['authored'])+' authored lessons now have a purpose-specific figure, a prediction prompt, reading guidance and an explicit connection to the computation. There are '+str(report['counts']['executedPlots'])+' plots from recorded CPU computations and '+str(report['counts']['conceptualDiagrams'])+' conceptual workflow diagrams. Each lesson also exposes captured script output; standalone notebooks include executed cells and actual Matplotlib image outputs, with plotting code that can be rerun. Browser, printable book, EPUB and Markdown companions carry the figures.\n\nPlots show recorded reference runs, not code running in the browser. CPU timing figures identify their measurement boundary and do not establish accelerator or edge-device performance. Conceptual arrows are never labeled measured results. Source hashes prevent stale results from being attached to revised content.\n\n## Every authored lesson\n\n| Lesson | Previous visual | New figure | Why it belongs here |\n|---|---|---|---|\n'
for r in rows:
 if r['status']=='authored':md+=f"| {r['id']}: {r['title']} | {'New since baseline' if not r['baselinePresent'] else 'Existing diagram' if r['previousVisual'] else 'No figure'} | {r['visual']} ({r['kind']}) | {r['explanation'].replace(chr(10), ' ').replace('|', '&#124;')} |\n"
md+='\n## Planned lessons: visual requirements before authoring\n\nThese are teaching specifications, not available lessons or fabricated results.\n\n| Lesson | Proposed visual | Explanation and evidence needed |\n|---|---|---|\n'
for r in rows:
 if r['status']=='planned':md+=f"| {r['id']}: {r['title']} | {r['visual']} | {r['explanation'].replace(chr(10), ' ').replace('|', '&#124;')} |\n"
md+='\n## Remaining review work\n\n- Learner walkthroughs should check whether the prediction prompts and legends resolve the intended misconceptions.\n- Dense or multi-panel figures provide full-size links and a keyboard-scrollable region on narrow screens. Actual browser layout review remains separate from artifact inspection and DOM checks.\n- Future experiments should add task-appropriate distributions, uncertainty or error bars; do not invent them for deterministic toy examples.\n- Further lesson authoring must supply matching execution and figure evidence before being presented as executed material.\n\n## Reproduction\n\nAuthor `content.visual` in each lesson JSON. Run `npm run figures:build`, then `npm run content:build`, `npm run smoke:cpu`, and `npm run build`. The smoke run stores actual per-cell notebook outputs. Run `npm run check` and `npm run audit:visuals`. Figure generation requires the pinned Matplotlib dependency in `requirements-cpu.txt`.\n'
(ROOT/'docs/visual-learning-audit.md').write_text(md)
print(report['counts'])
