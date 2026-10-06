"""Inventory source structure; this does not claim close reading or technical validation."""
import argparse, subprocess
import collections, hashlib, json, re, statistics
from pathlib import Path
parser=argparse.ArgumentParser(description='Structurally inventory English lessons without executing reference code.')
parser.add_argument('checkout',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.checkout.resolve()
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
records=[]
for path in sorted((root/'phases').glob('*/*/docs/en.md')):
    text=path.read_text();base=path.parent.parent
    headers=re.findall(r'^(#{1,4})\s+(.+)$',text,re.M)
    fences=re.findall(r'^```([^\n]*)\n(.*?)^```\s*$',text,re.M|re.S)
    sections=re.split(r'^##\s+',text,flags=re.M)[1:]
    problem=next((s.split('\n',1)[1] for s in sections if s.lower().startswith('the problem\n')), '')
    exercises=next((s.split('\n',1)[1] for s in sections if s.lower().startswith('exercises\n')), '')
    figures=[body.strip() for lang,body in fences if lang.strip()=='figure']
    record=dict(path=str(base.relative_to(root)),title=next((h for level,h in headers if level=='#'),base.name),words=len(text.split()),bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest(),headings=[h for level,h in headers if level=='##'],subheadings=[h for level,h in headers if level in ('###','####')],languages=collections.Counter(lang.strip() or 'plain' for lang,body in fences),figures=figures,mermaid=sum(lang.strip()=='mermaid' for lang,body in fences),stepHeadings=sum(bool(re.search(r'\bstep\b',h,re.I)) for level,h in headers),tables=len(re.findall(r'^\|\s*[-:]' ,text,re.M)),links=len(re.findall(r'\]\(https?://',text)),problemLead=' '.join(problem.split()[:20]),exerciseLead=' '.join(exercises.split()[:25]),bundle={name:sum(p.is_file() for p in (base/name).rglob('*')) if (base/name).is_dir() else int((base/name).is_file()) for name in ['code','notebook','notebooks','outputs','tests','quiz.json']})
    records.append(record)
phases={}
for record in records:phases.setdefault(record['path'].split('/')[1],[]).append(record)
report={'referenceRevision':revision,'method':'Every English phase lesson file read and structurally indexed. Statistical inventory is separate from human/agent close reading and technical verification. No reference code executed.','lessons':records,'phaseSummary':{p:{'lessons':len(ls),'words':sum(l['words'] for l in ls),'medianWords':int(statistics.median(l['words'] for l in ls)),'mermaidBlocks':sum(l['mermaid'] for l in ls),'interactiveSlots':sum(len(l['figures']) for l in ls),'stepHeadings':sum(l['stepHeadings'] for l in ls)} for p,ls in phases.items()}}
args.output.write_text(json.dumps(report,indent=2)+'\n')
for p,summary in report['phaseSummary'].items():print(p,json.dumps(summary))
print('TOTAL',len(records),'lessons',sum(l['words'] for l in records),'words',sum(l['mermaid'] for l in records),'mermaid',sum(len(l['figures']) for l in records),'figure slots')
