"""Check downloadable books and bundles, including EPUB internal references."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/downloads'
course=json.loads((ROOT/'public/curriculum.json').read_text())
authored=[l for p in course['phases'] for l in p['lessons'] if l['status']=='authored']
release=json.loads((ROOT/'curriculum/releases.json').read_text())[0]
site_hosts={'www.tahabouhsine.com','tahabouhsine.com'}

class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids=set()
    def handle_starttag(self, tag, attrs):
        self.ids.update(value for key,value in attrs if key=='id' and value)

def check_published_link(href, source):
    link=urlsplit(href)
    if link.scheme not in ('http','https') or link.hostname not in site_hosts:
        return
    site_prefix='/jaxpathways/'
    assert link.path.startswith(site_prefix), f'{source}: same-site URL outside deployment prefix: {href}'
    # Astro's URL base does not introduce a matching directory inside dist.
    target=ROOT/'dist'/link.path[len(site_prefix):]
    assert target.is_file(), f'{source}: same-site target missing from built site: {href}'
    if link.fragment and target.suffix=='.html':
        parser=IdCollector()
        parser.feed(target.read_text())
        assert link.fragment in parser.ids, f'{source}: same-site fragment missing from built site: {href}'
def assert_narrative(source, text, chapter):
    # Visible prose and code survive; each TeX token survives in MathML annotation.
    pattern = re.compile(r"(`+)([^\n]*?)\1|\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\[([^\n\]]+)\]\(([^)\s]+)\)", re.S)
    annotations = {node.text for node in chapter.iter('{http://www.w3.org/1998/Math/MathML}annotation')}
    cursor = 0
    for match in pattern.finditer(source):
        assert source[cursor:match.start()].replace('**', '') in text, 'missing surrounding math prose'
        if match.group(1): assert match.group(2) in text, 'missing inline code'
        elif match.group(6):
            links = [node for node in chapter.iter('{http://www.w3.org/1999/xhtml}a') if node.attrib.get('href') == match.group(7)]
            for link in links:
                try: assert_narrative(match.group(6), ''.join(link.itertext()), link)
                except AssertionError: continue
                break
            else: raise AssertionError('missing narrative link or link label')
        else: assert next(group for group in match.groups()[2:5] if group is not None) in annotations, 'missing MathML expression'
        cursor = match.end()
    assert source[cursor:].replace('**', '') in text, 'missing narrative text'

with ZipFile(OUT/'jax-foundations.epub') as book:
    assert book.infolist()[0].filename=='mimetype'
    assert book.infolist()[0].compress_type==ZIP_STORED
    assert book.read('mimetype')==b'application/epub+zip'
    assert book.testzip() is None
    for name in book.namelist():
        if name.endswith(('.xhtml','.xml','.opf')): ET.fromstring(book.read(name))
    package=ET.fromstring(book.read('OEBPS/package.opf'))
    ns={'o':'http://www.idpf.org/2007/opf','dc':'http://purl.org/dc/elements/1.1/'}
    assert package.find('o:metadata/dc:creator',ns).text=='Taha Bouhsine'
    assert package.find('o:metadata/dc:publisher',ns).text=='JAX Pathways'
    assert package.find('o:metadata/dc:source',ns).text=='https://www.tahabouhsine.com/jaxpathways/book.html'
    assert package.find('o:metadata/dc:description',ns).text
    items={item.attrib['id']:item for item in package.findall('o:manifest/o:item',ns)}
    for item in items.values(): assert 'OEBPS/'+item.attrib['href'] in book.namelist()
    for item in package.findall('o:spine/o:itemref',ns): assert item.attrib['idref'] in items
    assert len([name for name in book.namelist() if name.startswith('OEBPS/chapter-')])==len(authored)
    assert f'course reader {release["version"]}' in ''.join(package.find('o:metadata/dc:title',ns).itertext())
    assert package.find('o:metadata/o:meta[@property="dcterms:modified"]',ns).text==release['date']+'T00:00:00Z'
    assert all(f'OEBPS/{name}.xhtml' in book.namelist() for name in ['solutions','appendix-workflow','appendix-experiments'])
    solutions_text=''.join(ET.fromstring(book.read('OEBPS/solutions.xhtml')).itertext())
    assert all(lesson['content']['solution'] in solutions_text and lesson['content']['code'] in solutions_text for lesson in authored)
    for index, lesson in enumerate(authored):
        chapter=ET.fromstring(book.read(f'OEBPS/chapter-{index}.xhtml'))
        text=''.join(chapter.itertext())
        assert chapter.find('.//{http://www.w3.org/1999/xhtml}head/{http://www.w3.org/1999/xhtml}title').text==lesson['title']
        assert lesson['content']['solution'] not in text, f'{lesson["id"]}: solution must be in separate appendix'
        assert lesson['content']['code'] not in text, f'{lesson["id"]}: assembled example must be in separate appendix'
        annotations={node.text for node in chapter.iter('{http://www.w3.org/1998/Math/MathML}annotation')}
        if annotations: assert 'mathml' in items[f'c{index}'].attrib.get('properties','').split()
        if lesson['content'].get('visual'):
            v=lesson['content']['visual']
            image=chapter.find('.//{http://www.w3.org/1999/xhtml}img[@src="figures/'+lesson['id']+'.png"]')
            assert image is not None and image.attrib['src']=='figures/'+lesson['id']+'.png', f'{lesson["id"]}: missing book figure; regenerate its evidence, then rebuild the downloads'
            assert image.attrib.get('alt')==v.get('alt',v['title'])
            assert book.read('OEBPS/'+image.attrib['src'])==(ROOT/lesson['path']/'outputs/figure.png').read_bytes()
            for field in ['prediction','reading','connection']:
                for paragraph in v[field].split('\n\n'):assert_narrative(paragraph,text,chapter)
            assert lesson['execution']['stdout'] in solutions_text
        if lesson['content'].get('diagram'):
            diagram=lesson['content']['diagram']
            image=chapter.find('.//{http://www.w3.org/1999/xhtml}img[@src="figures/'+lesson['id']+'-mechanism.png"]')
            assert image is not None, lesson['id']+': missing mechanism image'
            assert image.attrib.get('alt')==diagram.get('alt',diagram['title'])
            assert book.read('OEBPS/'+image.attrib['src'])==(ROOT/lesson['path']/'outputs/mechanism.png').read_bytes()
            for field in ['prediction','reading']:
                for paragraph in diagram[field].split('\n\n'): assert_narrative(paragraph,text,chapter)
        for section in lesson['content'].get('sections',[]):
            if section.get('math'): assert section['math'] in annotations, f'{lesson["id"]}: missing display equation'
            if section.get('check'):
                for field in ['prompt','answer']:
                    for paragraph in section['check'][field].split('\n\n'): assert_narrative(paragraph,text,chapter)
        for section in lesson['content'].get('sections',[]):
            for paragraph in section['body'].split('\n\n'):
                assert_narrative(paragraph, text, chapter)
        for section in lesson['content'].get('sections',[]):
            for command in section.get('commands',[]):
                assert command['code'] in text, f'{lesson["id"]}: missing literal shell command'
                for paragraph in command['expected'].split('\n\n'):
                    assert_narrative(paragraph, text, chapter)
        for step in lesson['content'].get('buildSteps',[]):
            assert step['code'] in text
            for field in ['instruction','explanation']:
                for paragraph in step[field].split('\n\n'):
                    assert_narrative(paragraph, text, chapter)
        for experiment in lesson['content'].get('experiments',[]):
            assert experiment['code'] in text, f'{lesson["id"]}: missing book experiment'
        assert all(practice['solution'] in solutions_text for practice in lesson['content'].get('practice',[]))
    for name in book.namelist():
        if name.endswith('.xhtml'):
            for a in ET.fromstring(book.read(name)).iter('{http://www.w3.org/1999/xhtml}a'):
                href=a.attrib['href']
                check_published_link(href, f'EPUB {name}')
                if not urlsplit(href).scheme and not href.startswith('//'):
                    assert 'OEBPS/'+href.split('#')[0] in book.namelist()
for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
    project_path=ROOT/manifest
    project=json.loads(project_path.read_text())
    with ZipFile(OUT/(project['id']+'.zip')) as bundle:
        assert len(bundle.namelist())==len(set(bundle.namelist())), f'{project["id"]}: duplicate archive paths'
        for path in project_path.parent.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                assert bundle.read(str(path.relative_to(ROOT)))==path.read_bytes()
        requirements=project['environment']['requirementsPath']
        assert bundle.read(requirements)==(ROOT/requirements).read_bytes()
        for extra in {
            'kernel-audit': ('resources/tpu-gcp/requirements-qualification.txt',),
        }.get(project['id'], ()):
            assert bundle.read(extra)==(ROOT/extra).read_bytes(), f'{project["id"]}: missing extra environment file {extra}'
        assert set(project['requiredLessonIds'])<=set(l['id'] for l in authored)
with ZipFile(OUT/'jax-tutor-skills.zip') as bundle:
    for path in (ROOT/'skills').rglob('*.md'):
        assert bundle.read('README.md' if path == ROOT/'skills/README.md' else str(path.relative_to(ROOT)))==path.read_bytes()
projects=json.loads((ROOT/'public/api/v1/projects.json').read_text())
project=json.loads((ROOT/'projects/regression-audit/project.json').read_text())
assert project in projects['projects']
assert set(project['requiredLessonIds'])<=set(l['id'] for l in authored)
print(f'OK: {len(authored)} offline chapters, EPUB XML/references, project/tutor bundle fidelity, authored project prerequisites.')

html_book=(ROOT/'dist/downloads/jax-foundations.html').read_text()
assert all('src="'+value+'"' in html_book for value in re.findall(r'<img\b[^>]*\bsrc="([^"]+)"',html_book))
assert not re.search(r'<img\b[^>]*\bsrc="(?!data:image/)',html_book), 'standalone book has an external image dependency'
assert '<link rel="stylesheet"' not in html_book, 'standalone book has an external stylesheet dependency'
assert 'appendix-project-workflow' in html_book and 'appendix-experiments' in html_book
assert f'v{release["version"]}' in html_book
print('OK: standalone HTML has embedded figure assets and course appendices.')

with ZipFile(OUT/'jax-start-here.zip') as bundle:
    setup=next(l for l in authored if l['id']=='welcome-01')
    expected={'jax-start-here/first_experiment.py','jax-start-here/requirements-beginner.txt','jax-start-here/START-HERE.md','jax-start-here/figure.svg'}
    if setup['content'].get('diagram'):
        expected.add('jax-start-here/mechanism.svg')
        assert bundle.read('jax-start-here/mechanism.svg')==(ROOT/setup['path']/'outputs/mechanism.svg').read_bytes()
    assert set(bundle.namelist())==expected
    assert bundle.read('jax-start-here/first_experiment.py').decode()==setup['content']['code']+'\n'
    assert bundle.read('jax-start-here/requirements-beginner.txt')==(ROOT/'requirements-beginner.txt').read_bytes()
    setup_guide = bundle.read('jax-start-here/START-HERE.md').decode()
    assert 'requirements-beginner.txt' in setup_guide
    assert setup_guide==(ROOT/setup['artifacts']['source']).read_text().replace('../outputs/figure.svg','figure.svg').replace('../outputs/mechanism.svg','mechanism.svg')
    assert bundle.read('jax-start-here/figure.svg')==(ROOT/setup['path']/'outputs/figure.svg').read_bytes()
print('OK: beginner workspace source/package/instruction fidelity.')

with ZipFile(OUT/'jax-course-workspace.zip') as bundle:
    assert bundle.testzip() is None
    workspace_names=bundle.namelist()
    assert len(workspace_names)==len(set(workspace_names)), 'course workspace contains duplicate archive paths'
    names=set(workspace_names)
    assert 'jaxpathways/START-HERE.md' in names
    assert 'npx skills add . --skill start-learning learn-jax jax-course-guide check-jax' in bundle.read('jaxpathways/START-HERE.md').decode()
    for source in ('curriculum/course.json','curriculum/projects.json','requirements-cpu.txt','scripts/course.py','public/curriculum.json','public/validation.json'):
        assert bundle.read('jaxpathways/'+source)==(ROOT/source).read_bytes()
    for skill in ('start-learning','learn-jax','jax-course-guide','check-jax'):
        source=f'skills/{skill}/SKILL.md'
        assert bundle.read('jaxpathways/'+source)==(ROOT/source).read_bytes()
    for lesson in authored:
        for source in (lesson['path']+'/lesson.json',lesson['artifacts']['source'],lesson['artifacts']['scriptSource'],lesson['artifacts']['notebookSource'],lesson['artifacts']['quizSource'],lesson['artifacts']['evidenceSource']):
            assert bundle.read('jaxpathways/'+source)==(ROOT/source).read_bytes()
    for lesson in authored:
        for name in ('figure.svg','figure.png','visual.json','execution.json')+(('mechanism.svg','mechanism.png') if lesson['content'].get('diagram') else ()):
            source=lesson['path']+'/outputs/'+name
            assert bundle.read('jaxpathways/'+source)==(ROOT/source).read_bytes()
    for manifest in json.loads((ROOT/'curriculum/projects.json').read_text()):
        for path in (ROOT/manifest).parent.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                assert bundle.read('jaxpathways/'+str(path.relative_to(ROOT)))==path.read_bytes()
    for source in (
        'resources/tpu-gcp/collectives_target.py',
        'resources/tpu-gcp/check_collectives_target.py',
        'resources/tpu-gcp/requirements-qualification.txt',
        'phases/09-distributed/03-communication-efficient-algorithms/code/main.py',
    ):
        assert bundle.read('jaxpathways/'+source)==(ROOT/source).read_bytes()
    for repository_only in (
        'resources/tpu-gcp/multihost_target.py',
        'resources/tpu-gcp/check_multihost_target.py',
    ):
        assert 'jaxpathways/'+repository_only not in names, f'multi-host target requires coordinated workers and shared GCS, so it remains repository-only: {repository_only}'
    assert all(name.startswith('jaxpathways/') and not any(part in ('.git','.venv','__pycache__','LEARNING.md') for part in Path(name).parts) for name in names)
print('OK: course workspace includes tutor prerequisites, authored lesson artifacts and project sources without local learner state.')

with ZipFile(OUT/'jax-tpu-gcp.zip') as bundle:
    names=bundle.namelist()
    assert len(names)==len(set(names)), 'TPU/GCP workspace contains duplicate archive paths'
    prefix='jax-tpu-gcp/'
    for source in (
        'resources/tpu-gcp/collectives_target.py',
        'resources/tpu-gcp/check_collectives_target.py',
        'resources/tpu-gcp/requirements-qualification.txt',
        'phases/09-distributed/03-communication-efficient-algorithms/code/main.py',
    ):
        assert bundle.read(prefix+source)==(ROOT/source).read_bytes(), f'TPU/GCP workspace is missing exact source: {source}'
    for repository_only in (
        'resources/tpu-gcp/multihost_target.py',
        'resources/tpu-gcp/check_multihost_target.py',
    ):
        assert prefix+repository_only not in names, f'multi-host target requires coordinated workers and shared GCS, so it remains repository-only: {repository_only}'
print('OK: TPU workspace has its pinned single-host target, checker and exact base lesson source; coordinated multi-host files remain repository-only.')

assessment_api=json.loads((ROOT/'public/api/v1/assessments.json').read_text())
for assessment in assessment_api['assessments']:
    assert assessment['status'] in ('active', 'review-draft')
    route=next(r for r in course['pathways'] if r['id']==assessment.get('pathwayId',assessment['id']))
    assert assessment['projectId'] in [p['id'] for p in projects['projects'] if p['pathwayId']==route['id']]
    if assessment['scope']=='math-extension':
        assert assessment['phaseId'] in route['phaseIds']
        assert any(a['id']==assessment['id'] and a['url']==assessment['url'] for a in route['assessmentExtensions'])
    else:
        assert route['capstone']['assessmentDraft']['url']==assessment['url']
    project=next(p for p in projects['projects'] if p['id']==assessment['projectId'])
    if assessment['scope']!='math-extension':
        assert project['assessmentId']==assessment['id']
    assert (ROOT/assessment['source']).read_bytes()==(ROOT/'public/assessments'/Path(assessment['source']).name).read_bytes()
    assert (ROOT/'dist'/assessment['url']).is_file()
    notes=assessment['id']+'-reviewer.md'
    assert (ROOT/'assessments'/notes).read_bytes()==(ROOT/'public/assessments'/notes).read_bytes()
print(f"OK: {len(assessment_api['assessments'])} synthesis assessments match source, pathway and project.")
