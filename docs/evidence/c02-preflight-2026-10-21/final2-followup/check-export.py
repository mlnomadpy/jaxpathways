"""Exercise the full frozen exporter CLI with actual pilot inputs."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys

OUT=Path(__file__).resolve().parent
DATA=Path('/private/tmp/jaxpathways-c02-datafreeze-luna-20261021')
RUN=Path('/private/tmp/jaxpathways-c02-pilot-luna-approved-mechanics-20261021')
EVAL=Path('/private/tmp/jaxpathways-c02-evaluation-20261021-final2')
BRANCH='sft_full_from_domain'
original=json.loads((EVAL/'metrics.json').read_text())
record={'scope':'Full exporter CLI against real frozen data/checkpoint and evaluation receipts, including seven stale receipt mutations. No model publication.', 'source_sha256':hashlib.sha256((OUT/'source/export_consumer.py').read_bytes()).hexdigest(),'cases':[]}
paths=[('recipe_sha256',),('tokenizer_sha256',),('evaluator_source_sha256',),('scorer','reference_bytes_sha256'),('branches',BRANCH,'checkpoint_sha256'),('branches',BRANCH,'checkpoint_identity_sha256'),('branches',BRANCH,'candidate_source_sha256')]
for index,path in enumerate(paths):
    metrics=copy.deepcopy(original)
    cursor=metrics
    for part in path[:-1]: cursor=cursor[part]
    cursor[path[-1]]='0'*64
    evaluation=OUT/f'negative-evaluation-{index}'
    evaluation.mkdir(exist_ok=True)
    (evaluation/'metrics.json').write_text(json.dumps(metrics))
    target=OUT/f'must-not-create-{index}'
    assert not target.exists()
    command=[sys.executable,str(OUT/'source/export_consumer.py'),'--data',str(DATA),'--run',str(RUN),'--evaluation',str(evaluation),'--branch',BRANCH,'--output',str(target)]
    result=subprocess.run(command,cwd='/private/tmp',capture_output=True,text=True,timeout=60)
    assert result.returncode != 0 and not target.exists(),(path,result.stdout,result.stderr)
    assert 'ValueError: evaluation' in result.stderr,(path,result.stderr)
    record['cases'].append({'altered_field':'.'.join(path),'exit_code':result.returncode,'output_absent':True,'error':result.stderr.strip().splitlines()[-1]})
target=OUT/'positive-local-package'
command=[sys.executable,str(OUT/'source/export_consumer.py'),'--data',str(DATA),'--run',str(RUN),'--evaluation',str(EVAL),'--branch',BRANCH,'--output',str(target)]
result=subprocess.run(command,cwd='/private/tmp',capture_output=True,text=True,timeout=60)
assert result.returncode == 0,(result.stdout,result.stderr)
manifest=json.loads((target/'manifest.json').read_text())
assert manifest['evaluation_metrics_sha256'] == hashlib.sha256((EVAL/'metrics.json').read_bytes()).hexdigest()
assert manifest['checkpoint_sha256']==original['branches'][BRANCH]['checkpoint_sha256']
record['positive_control']={'accepted':True,'evaluation_quality_still_failed':True,'manifest_sha256':hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()}
record['status']='pass'
(OUT/'export-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
