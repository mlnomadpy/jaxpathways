"""Build offline reading and portable learning bundles from canonical content."""
import hashlib
import html
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/downloads'
OUT.mkdir(exist_ok=True)
course = json.loads((ROOT / 'public/curriculum.json').read_text())
lessons = [lesson for phase in course['phases'] for lesson in phase['lessons'] if lesson['status'] == 'authored']
esc = html.escape
math_markup = json.loads(subprocess.check_output(["node", "scripts/render-book-math.mjs"], cwd=ROOT, text=True))
def rich(text):
    return math_markup["prose"].get(text, esc(text))
style = 'figure{margin:1.5rem 0}figure img{width:100%;height:auto}figcaption{font-size:.85em}body{max-width:760px;margin:3rem auto;padding:0 1.5rem;font:18px/1.7 Georgia,serif;color:#172334}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.3}pre{padding:1rem;background:#eef1f5;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}article{page-break-before:always;margin-top:4rem}a{color:#2456a6}@media print{nav{display:none}body{margin:0;max-width:none}}'
intro = f'<h1>JAX Pathways: course reader</h1><p>{len(lessons)} lesson drafts from the shared course. Availability of any remaining lessons and projects is recorded in the catalog. Reference execution status is recorded separately in validation.json; TPU execution and expert review remain outstanding.</p><p>Work through each exercise before reading its solution. Checkpoint answers are formative. Keep your code, outputs, and diagnosis as evidence.</p>'

def prose(text):
    return ''.join(f'<p>{rich(p)}</p>' for p in text.split('\n\n'))

def code_block(text):
    return '<pre><code>'+esc(text)+'</code></pre>'

def chapter(lesson):
    body = f'<h1>{esc(lesson["title"])}</h1><p>{esc(lesson["objective"])}</p>'
    c = lesson['content']
    if c.get('objectives'):
        body += '<h2>What you will be able to do</h2><ul>'+''.join('<li>'+rich(o)+'</li>' for o in c['objectives'])+'</ul>'
    if c.get('problem'): body += '<h2>The problem</h2>'+prose(c['problem'])
    body += '<h2>The idea</h2>'+prose(c['idea'])
    for section in c.get('sections',[]):
        body += '<h2>'+esc(section['title'])+'</h2>'+prose(section['body'])
        if section.get('id')=='guided-reasoning' and c.get('diagram'):
            d=c['diagram']; artifact=lesson.get('visualArtifact',{})
            body += '<h3>'+esc(d['title'])+'</h3><p><strong>Predict:</strong> '+rich(d['prediction'])+'</p>'
            if artifact.get('mechanismPng'):
                body += '<figure><img src="figures/'+esc(lesson['id'])+'-mechanism.png" alt="'+esc(d['title'])+'"/><figcaption>'+esc(d['scope'])+'</figcaption></figure>'
            body += prose(d['reading'])
        if section.get('check'):
            check=section['check']
            body += '<h3>Pause and reason</h3>'+prose(check['prompt'])+'<details><summary>Compare your reasoning</summary>'+prose(check['answer'])+'</details>'
        for command in section.get('commands',[]):
            body += '<h3>'+esc(command['label'])+'</h3>'+code_block(command['code'])+'<p><strong>Expected:</strong></p>'+prose(command['expected'])
        if section.get('math'): body += math_markup['equations'][section['math']]
        if section.get('formula'): body += code_block(section['formula'])
    if c.get('visual') and lesson.get('visualArtifact'):
        visual=c['visual']; artifact=lesson['visualArtifact']
        scope='Recorded CPU computation' if visual['kind']=='executed' else 'Conceptual workflow diagram'
        body += '<h2>'+esc(visual['title'])+'</h2><p><strong>Predict:</strong> '+rich(visual['prediction'])+'</p>'
        body += '<figure><img src="figures/'+esc(lesson['id'])+'.png" alt="'+esc(visual['title'])+'"/><figcaption>'+scope+' · JAX '+esc(artifact['environment']['jax'])+' · '+esc(artifact['generatedAt'])+'</figcaption></figure>'
        body += '<h3>Read the figure</h3>'+prose(visual['reading'])+'<h3>Connect it to the computation</h3>'+prose(visual['connection'])
    for step in c.get('buildSteps',[]):
        body += '<h2>'+esc(step['title'])+'</h2>'+prose(step['instruction'])+code_block(step['code'])+prose(step['explanation'])
    if c.get('buildCode'): body += '<h2>Build a numerical estimate</h2>'+code_block(c['buildCode'])+prose(c['buildOutput'])
    body += '<h2>Run the example</h2>'+code_block(c['code'])+prose(c['output'])
    if lesson.get('execution'):
        run=lesson['execution']
        body += '<h2>Recorded reference execution</h2><p>Saved instructor CPU run, including assertions and reference solutions. '+esc(run['executedAt'])+'</p>'+code_block(run['stdout'])
    for experiment in c.get('experiments',[]):
        body += '<h2>'+esc(experiment['title'])+'</h2><p><strong>Predict before running:</strong> '+rich(experiment['prediction'])+'</p>'+code_block(experiment['code'])+prose(experiment['output'])+prose(experiment['explanation'])
    body += '<h2>Your exercise</h2>'+prose(c['exercise'])+'<h3>Reference solution</h3>'+code_block(c['solution'])
    for practice in c.get('practice',[]):
        body += '<h2>'+esc(practice['title'])+'</h2><p>'+esc(practice['difficulty'])+'</p>'+prose(practice['prompt'])+'<h3>Hint</h3>'+prose(practice['hint'])+'<h3>Reference solution</h3>'+code_block(practice['solution'])+prose(practice['explanation'])
    body += f'<h2>Checkpoint</h2><p>{rich(c["question"])}</p><ol>' + ''.join(f'<li>{rich(o)}</li>' for o in c['options']) + '</ol>'
    body += f'<p>Answer: {rich(c["options"][c["answer"]])}. {rich(c["explanation"])}</p>'
    if c.get('diagnosis'): body += '<h2>Diagnose the result</h2>'+prose(c['diagnosis'])
    if c.get('takeaways'): body += '<h2>Carry forward</h2><ul>'+''.join('<li>'+rich(t)+'</li>' for t in c['takeaways'])+'</ul>'
    body += '<h2>Your evidence</h2>'+prose(lesson['evidence'])
    if c.get('references'): body += '<h2>Primary references</h2><ul>'+''.join('<li><a href="'+esc(r['url'])+'">'+esc(r['title'])+'</a></li>' for r in c['references'])+'</ul>'
    return body

chapters = [chapter(l) for l in lessons]
nav = '<nav><h2>Contents</h2><ol>' + ''.join(f'<li><a href="#chapter-{i}">{esc(l["title"])}</a></li>' for i,l in enumerate(lessons)) + '</ol></nav>'
xhtml = lambda body: '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head><title>JAX Pathways</title><link rel="stylesheet" href="style.css"/></head><body>'+body+'</body></html>'
identity = hashlib.sha256(json.dumps(lessons,sort_keys=True).encode()).hexdigest()
with ZipFile(OUT / 'jax-foundations.epub','w',compression=ZIP_DEFLATED) as book:
    book.writestr('mimetype','application/epub+zip',compress_type=ZIP_STORED)
    book.writestr('META-INF/container.xml','<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    book.writestr('OEBPS/style.css',style)
    book.writestr('OEBPS/intro.xhtml',xhtml(intro))
    book.writestr('OEBPS/nav.xhtml',xhtml('<nav xmlns:epub="http://www.idpf.org/2007/ops" epub:type="toc"><h1>Contents</h1><ol><li><a href="intro.xhtml">About this edition</a></li>'+''.join(f'<li><a href="chapter-{i}.xhtml">{esc(l["title"])}</a></li>' for i,l in enumerate(lessons))+'</ol></nav>'))
    for i,body in enumerate(chapters): book.writestr(f'OEBPS/chapter-{i}.xhtml',xhtml(body))
    items=''.join(f'<item id="c{i}" href="chapter-{i}.xhtml" media-type="application/xhtml+xml"'+(' properties="mathml"' if '<math ' in body else '')+'/>' for i,body in enumerate(chapters))
    for lesson in lessons:
        if lesson.get('visualArtifact'):
            book.write(ROOT/'public'/lesson['visualArtifact']['png'], 'OEBPS/figures/'+lesson['id']+'.png')
            items += '<item id="figure-'+lesson['id']+'" href="figures/'+lesson['id']+'.png" media-type="image/png"/>'
        if lesson.get('visualArtifact',{}).get('mechanismPng'):
            book.write(ROOT/'public'/lesson['visualArtifact']['mechanismPng'], 'OEBPS/figures/'+lesson['id']+'-mechanism.png')
            items += '<item id="mechanism-'+lesson['id']+'" href="figures/'+lesson['id']+'-mechanism.png" media-type="image/png"/>'
    spine=''.join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    book.writestr('OEBPS/package.opf',f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book-id">urn:sha256:{identity}</dc:identifier><dc:title>JAX Pathways: course reader</dc:title><dc:language>en</dc:language><meta property="dcterms:modified">2026-10-03T00:00:00Z</meta></metadata><manifest><item id="intro" href="intro.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="style" href="style.css" media-type="text/css"/>{items}</manifest><spine><itemref idref="intro"/>{spine}</spine></package>')
for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
    project_path = ROOT/manifest
    project = json.loads(project_path.read_text())
    with ZipFile(OUT/(project['id']+'.zip'),'w',compression=ZIP_DEFLATED) as bundle:
        for path in sorted(project_path.parent.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                bundle.write(path,path.relative_to(ROOT))
        bundle.write(ROOT/'requirements-cpu.txt','requirements-cpu.txt')
with ZipFile(OUT/'jax-tutor-skills.zip','w',compression=ZIP_DEFLATED) as bundle:
    for path in sorted((ROOT/'skills').glob('*/SKILL.md')): bundle.write(path,path.relative_to(ROOT))
    for path in sorted((ROOT/'skills').glob('*/references/*.md')): bundle.write(path,path.relative_to(ROOT))
    bundle.write(ROOT/'skills/README.md','README.md')
print(f'Built offline HTML/EPUB ({len(lessons)} lessons), staged project, and tutor bundles.')

setup=next(l for l in lessons if l['id']=='welcome-01')
with ZipFile(OUT/'jax-start-here.zip','w',compression=ZIP_DEFLATED) as bundle:
    bundle.writestr('jax-start-here/first_experiment.py',setup['content']['code']+'\n')
    bundle.write(ROOT/'requirements-cpu.txt','jax-start-here/requirements-cpu.txt')
    setup_text=(ROOT/setup['artifacts']['source']).read_text().replace('../outputs/figure.svg','figure.svg').replace('../outputs/mechanism.svg','mechanism.svg')
    bundle.writestr('jax-start-here/START-HERE.md',setup_text)
    if setup.get('visualArtifact'):bundle.write(ROOT/setup['path']/'outputs/figure.svg','jax-start-here/figure.svg')
    if setup.get('visualArtifact',{}).get('mechanismImage'):bundle.write(ROOT/setup['path']/'outputs/mechanism.svg','jax-start-here/mechanism.svg')

# A portable course root for tutors: skills alone do not include their lesson sources.
workspace_files = {ROOT/'requirements-cpu.txt', ROOT/'scripts/course.py',
                   ROOT/'public/curriculum.json', ROOT/'public/validation.json'}
for directory in ('curriculum', 'learning-paths', 'phases', 'projects', 'skills', 'assessments', 'resources/tpu-gcp', 'content/guides'):
    workspace_files.update(path for path in (ROOT/directory).rglob('*')
                           if path.is_file() and path.suffix in ('.md', '.json', '.py', '.ipynb', '.txt', '.svg', '.png')
                           and '__pycache__' not in path.parts)
# Registered project bundles include their executed binary artifacts and media.
for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
    workspace_files.update(path for path in (ROOT/manifest).parent.rglob('*')
                           if path.is_file() and '__pycache__' not in path.parts)
for directory in ('public/figures','public/executions','public/guides'):
    workspace_files.update(path for path in (ROOT/directory).glob('*') if path.is_file())
start_guide = '''# Start JAX Pathways with your agent

This folder contains the current course curriculum, lesson sources, projects and tutor skills.

1. Extract the archive. Open this jaxpathways folder in your coding agent.
2. Open a terminal in this folder. With Node.js/npm available, run:

```sh
npx skills add . --skill start-learning learn-jax jax-course-guide check-jax
```

Choose your agent when prompted. Installation is scoped to this project; it does not install Python.

3. In the agent chat say:

> Use start-learning to help me learn JAX from this workspace.

Codex also supports $start-learning; Claude Code supports /start-learning.
Keep this folder open so the tutor can read the course files. It starts with guided setup and
keeps your plan in LEARNING.md. Say “Use learn-jax to continue my plan” to resume.

Read skills/README.md for the full skill instructions. requirements-cpu.txt records the course's
tested Python package versions; the setup lesson explains installation. Browser progress and
the tutor's study plan do not synchronize automatically. Planned topics are not authored lessons.
This is a portable learning workspace, not a Git checkout or a website build environment.
'''
with ZipFile(OUT/'jax-course-workspace.zip','w',compression=ZIP_DEFLATED) as bundle:
    bundle.writestr('jaxpathways/START-HERE.md',start_guide)
    for path in sorted(workspace_files):
        bundle.write(path,'jaxpathways/'+str(path.relative_to(ROOT)))
print('Built course workspace with canonical lessons, projects and tutor skills.')
