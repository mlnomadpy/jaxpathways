"""Independent saved-output aggregation; no candidate scoring imports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import string
from tokenizers import Tokenizer
from transformers import AutoTokenizer

OUT=Path(__file__).resolve().parent
PYTHIA=Path('/private/tmp/jaxpathways-c02-pythia-qa-20261021.json')
HARNESS=Path('/private/tmp/jaxpathways-c02-lm-eval-20261021-final.json')
BASE=Path('/private/tmp/jaxpathways-c02-baseline-pythia14m')
DATA=Path('/private/tmp/jaxpathways-c02-datafreeze-luna-20261021')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gold={r['question_id']:r for r in map(json.loads,(DATA/'qa-final.jsonl').read_text().splitlines())}
def normalize(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def score(p,answers):
    p=normalize(p); ep=[];fp=[]
    for answer in answers:
        g=normalize(answer);overlap=sum((Counter(p.split())&Counter(g.split())).values())
        ep.append(float(p==g));fp.append(0. if not overlap else 2*overlap/(len(p.split())+len(g.split())))
    return max(ep),max(fp)
p=json.loads(PYTHIA.read_text());t=Tokenizer.from_file(str(BASE/'tokenizer.json'))
configured_tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
assert p['checkpoint_sha256']==sha(BASE/'model.safetensors')
assert p['tokenizer_sha256']==sha(BASE/'tokenizer.json')
assert p['data_recipe_sha256']==sha(DATA/'frozen_recipe.json')
assert len(p['per_question'])==128 and {r['question_id'] for r in p['per_question']}==set(gold)
ems=[];f1s=[]
for r in p['per_question']:
    g=gold[r['question_id']]
    prompt=f"Question: {g['question']}\nContext: {g['context']}\nAnswer:"
    assert r['prompt']==prompt and r['human_answers']==g['answers'] and r['article_id']==g['article_id']
    assert r['prompt_token_ids']==t.encode(prompt).ids
    ids=r['generated_token_ids'];assert 0<len(ids)<=32 and 0 not in ids[:-1]
    assert r['prediction']==configured_tokenizer.decode(ids,skip_special_tokens=True).strip()
    assert r['finish_reason']==('eos' if ids[-1]==0 else 'max_new_tokens')
    em,f1=score(r['prediction'],g['answers'])
    assert em==r['exact_match'] and math.isclose(f1,r['token_f1'],abs_tol=1e-15)
    ems.append(em);f1s.append(f1)
assert sum(ems)/128==p['exact_match']
assert math.isclose(sum(f1s)/128,p['token_f1'],abs_tol=1e-15)
h=json.loads(HARNESS.read_text());tasks={}
for task,rows in h['result']['samples'].items():
    assert len(rows)==8 and [r['doc_id'] for r in rows]==list(range(8))
    accs=[];norms=[]
    for r in rows:
        choices=r['doc']['choices'] if task=='hellaswag' else r['doc']['choices']['text']
        scores=[v[0] for v in r['filtered_resps']]
        assert len(scores)==len(choices)
        raw=max(range(len(scores)),key=lambda i:scores[i])
        norm=max(range(len(scores)),key=lambda i:scores[i]/len(choices[i]))
        acc=float(raw==r['target']);an=float(norm==r['target'])
        assert acc==r['acc'] and an==r['acc_norm']
        accs.append(acc);norms.append(an)
    actual={'examples':8,'accuracy':sum(accs)/8,'accuracy_character_normalized':sum(norms)/8}
    assert actual['accuracy']==h['result']['results'][task]['acc,none']
    assert actual['accuracy_character_normalized']==h['result']['results'][task]['acc_norm,none']
    tasks[task]=actual
result={'status':'pass','scope':'Saved Pythia QA prompt/answer/tokenizer/checkpoint identity and all 128 scores; native task raw-versus-character-normalized choice aggregation on 8 stored samples each. No independent model execution, harness source provenance, task dataset reconstruction or official benchmark qualification.', 'input_sha256':{PYTHIA.name:sha(PYTHIA),HARNESS.name:sha(HARNESS)},'pythia':{'questions':128,'exact_match':sum(ems)/128,'token_f1':sum(f1s)/128},'native_task_smoke':tasks}
(OUT/'baseline-output-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
