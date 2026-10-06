"""Audit every canonical entry and its real teaching artifacts, without executing code."""
import argparse, collections, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',default='docs/content-audit.json');p.add_argument('--markdown',default='docs/content-audit.md');args=p.parse_args()
course=json.loads((ROOT/'curriculum/course.json').read_text());receipt=json.loads((ROOT/'curriculum/validation.json').read_text())
records=[]
for phase in course['phases']:
 for l in phase['lessons']:
  source=ROOT/l['path']/'lesson.json';c=json.loads(source.read_text())['content'] if source.exists() else {}
  generic=[field for field in ['exercise','evidence','check'] if l.get(field)==l['objective'] or 'A reproducible experiment for “' in l.get(field,'') or 'Explain your result for “' in l.get(field,'')]
  artifacts={k:(ROOT/path if k.endswith('Source') or k=='source' else ROOT/'public'/path).exists() for k,path in l.get('artifacts',{}).items()}
  missing=[name for name in ['problem','objectives','sections','experiments','practice','diagnosis','references'] if not c.get(name)]
  tier='missing lesson' if not c else 'setup walkthrough' if c.get('setupBundle') else 'starter only' if missing else 'teaching structure present'
  content_hash=hashlib.sha256(json.dumps(c,separators=(',',':'),ensure_ascii=False).encode()).hexdigest() if c else None
  runtime=next((r for r in receipt['lessons'] if r['lessonId']==l['id']),None)
  generated=ROOT/'public/api/v1/lessons'/f"{l['id']}.json"
  generated_hash=json.loads(generated.read_text()).get('contentHash') if generated.exists() else None
  records.append(dict(id=l['id'],title=l['title'],phase=phase['id'],path=l['path'],declaredStatus=l['status'],contentTier=tier,genericManifestFields=generic,missingTeachingComponents=missing,artifacts=artifacts,sectionCount=len(c.get('sections',[])),experimentCount=len(c.get('experiments',[])),practiceCount=len(c.get('practice',[])),diagramCount=sum(bool(s.get('formula')) for s in c.get('sections',[]))+bool(c.get('figure')),runtime=('logical CPU reference passed' if runtime.get('runtime',{}).get('deviceCount',1)>1 else 'CPU reference passed') if runtime and runtime['contentHash']==generated_hash and generated_hash==content_hash else 'not validated for current content',review='not independently reviewed',sourcePresent=bool(c)))
projects=list((ROOT/'projects').glob('*/project.json'))
assessments=json.loads((ROOT/'curriculum/assessments.json').read_text())
report={'scope':'All canonical lesson entries and every existing lesson.json; source/companion availability and teaching structure. Structural presence is not an editorial approval. CPU receipts validate instructor examples only. No TPU execution or independent technical review.', 'lessons':records,'totals':dict(collections.Counter(r['contentTier'] for r in records)),'integration':{'projectManifests':len(projects),'pathwayManifests':len(list((ROOT/'learning-paths').glob('*.json'))),'assessmentStatuses':dict(collections.Counter(a['status'] for a in assessments)),'tpuValidatedLessons':0,'independentlyReviewedLessons':0}}
(ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
md='# Whole-content audit\n\n'+report['scope']+'\n\n## Actual source coverage\n\n'+ '\n'.join(f'- {n} {kind}' for kind,n in report['totals'].items())+f'\n\n{len(projects)} staged integration project manifests exist. {len(assessments)} synthesis assessment review drafts exist; no TPU execution receipts exist. Pathway and career descriptions distinguish available projects from remaining hardware and review evidence.\n\n## Every entry\n\n| Lesson | Actual material | Missing teaching components | Generic fields | Runtime |\n|---|---|---|---|---|\n'
for r in records:md+=f"| {r['id']}: {r['title']} | {r['contentTier']} | {', '.join(r['missingTeachingComponents']) or 'Structure present; editorial review still needed'} | {', '.join(r['genericManifestFields']) or 'None detected'} | {r['runtime']} |\n"
md+='\n## Remediation priorities\n\n1. Preserve a connected, independently checked foundation: arrays → transformations → explicit state → optimization → regression audit.\n2. Verify each connected modality harness through its complete lifecycle and retain declared data/device limits.\n3. Keep route projects, synthesis drafts and phase integration checks aligned with their implemented scope. A manifest naming them is not an implementation.\n4. Add targeted diagrams, benchmark/profile artifacts and real TPU execution with supported environments.\n5. Obtain independent technical review and learner walkthroughs. Do not relabel structural checks as review.\n'
(ROOT/args.markdown).write_text(md);print(report['totals'])
