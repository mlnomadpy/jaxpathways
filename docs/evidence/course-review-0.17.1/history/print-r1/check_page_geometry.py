from pathlib import Path
import pdfplumber,hashlib,json,datetime
root=Path('/private/tmp/jaxpathways-print-0.17.1-r1/output/pdf')
report={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All PDF pages checked for text glyphs outside the media box; this is not full visual review.','documents':[]}
needles=['Least squares, rank, and conditioning','Least squares balances unavoidable residuals','Train, checkpoint, and generate','bfloat16, int8 precision, and XProf','A Transformer block and mixed precision','Optional appendix: a saved decoder enters sequence PPO','Optional saved-decoder','Predict the boundaries','Calculate EOS and timeout by hand','Implement the six learner contracts','Recorded v5 result','Mean reward improvement','Treat this as a small synthetic mechanics experiment']
for path in sorted(root.glob('*.pdf')):
 entry={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':0,'characters':0,'outsideMediaBox':[],'emptyTextPages':[],'chapterPages':{}}
 with pdfplumber.open(path) as pdf:
  entry['pages']=len(pdf.pages)
  for i,page in enumerate(pdf.pages,1):
   chars=page.chars
   entry['characters']+=len(chars)
   outside=[c for c in chars if c['text'].strip() and (c['x0'] < -0.5 or c['x1'] > page.width+0.5 or c['top'] < -0.5 or c['bottom'] > page.height+0.5)]
   if outside:entry['outsideMediaBox'].append({'page':i,'count':len(outside),'examples':[{k:c[k] for k in ('text','x0','x1','top','bottom')} for c in outside[:6]]})
   if not chars:entry['emptyTextPages'].append(i)
   text=page.extract_text() or ''
   normalized=' '.join(text.split())
   for needle in needles:
    if needle.lower() in normalized.lower():entry['chapterPages'].setdefault(needle,[]).append(i)
   if i%100==0:print(path.name,i,flush=True)
   page.close()
 report['documents'].append(entry)
 (root/'geometry-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
