"""Replay the first final question of each branch; independent prefix oracle."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import jax

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'source'))
from config import ModelConfig
from training import load_checkpoint
from reference_math import numpy_reference
import model

RUN=Path('/private/tmp/jaxpathways-c02-pilot-luna-approved-mechanics-20261021')
EVAL=Path('/private/tmp/jaxpathways-c02-evaluation-20261021-final2')
rows=list(map(json.loads,(EVAL/'qa-generations.jsonl').read_text().splitlines()))
selected={}
for row in rows:
    selected.setdefault(row['branch'],row)
record={'scope':'One preselected first final QA record per branch: fresh checkpoint reload, full 32-token cached replay, independent NumPy full-prefix first-four-token oracle. Does not replay all 768 records or reconstruct corpus NLL.', 'selection':'First serialized question in each branch, selected before this replay.', 'cases':[]}
for name,row in selected.items():
    identity=json.loads((RUN/name/'identity.json').read_text())
    state=load_checkpoint(RUN/name/'final-state.npz',identity,expected_order=list(range(len(identity['schedule_sample_ids']))),expected_seen_ids=identity['schedule_sample_ids'])
    cfg=ModelConfig(**identity['model_config'])
    params=state['params']
    if state['adapters'] is not None:
        # Independent explicit rank-four alpha-eight merge, no candidate merger.
        params=jax.tree.map(lambda a:np.asarray(a).copy(),params)
        for block,adapter in zip(params['blocks'],state['adapters']['blocks'],strict=True):
            for projection in ('q_proj','v_proj'):
                block[projection]=block[projection]+np.float32(2)*(np.asarray(adapter[projection]['A'])@np.asarray(adapter[projection]['B']))
    prompt=list(row['prompt_token_ids'])
    observed=model.generate(params,prompt,cfg,max_new_tokens=32,eos_token_id=2)
    assert observed==row['generated_token_ids'],(name,observed,row['generated_token_ids'])
    independent=[]
    for _ in range(min(4,len(observed))):
        tokens=np.array([prompt+independent],dtype=np.int32)
        logits=numpy_reference(params,tokens,cfg,np.ones_like(tokens,dtype=bool),np.ones_like(tokens),np.arange(tokens.shape[1])[None,:])[0,-1]
        logits[0]=-np.inf
        independent.append(int(np.argmax(logits)))
    assert independent==observed[:len(independent)],(name,independent,observed[:4])
    record['cases'].append({'branch':name,'question_id':row['question_id'],'checkpoint_sha256':hashlib.sha256((RUN/name/'final-state.npz').read_bytes()).hexdigest(),'full_cached_replay_tokens':len(observed),'independent_prefix_tokens':independent})
    print(name,'passed',flush=True)
record['status']='pass'
record['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'source/model.py',OUT/'source/config.py',OUT/'source/training.py',OUT/'source/reference_math.py',Path(__file__)]}
(OUT/'selected-generation-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
