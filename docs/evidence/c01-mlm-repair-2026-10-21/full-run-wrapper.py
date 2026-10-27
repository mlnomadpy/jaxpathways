import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/private/tmp/jaxpathways-reader-review-fix')
project = root / 'projects/embedding-model-300m'
out = Path('/private/tmp/jaxpathways-c01-root-checker-repair2')
receipt = json.loads((project / 'reference/evidence/final8/sourcebound-receipt.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
sources = {relative: sha(project / relative) for relative in receipt['source_files_sha256']}
assert sources['tests/check.py'] == 'a3867129f43df1a1b6d8bf671d449488987e1339f7abfcae0caf35fc88c6d952'
env = dict(os.environ, JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
base = ['/private/tmp/jaxpathways-embedding-c01-env/bin/python', str(project / 'tests/check.py'), '--stage', 'all', '--data', '/private/tmp/jaxpathways-scifact-c01-frozen-final', '--baseline-snapshot', '/private/tmp/jaxpathways-v013-c01-current-check/full-run/baseline_snapshot']
for name, implementation in [('zero-head', '/private/tmp/jaxpathways-c01-root-mlm-probe/zero_mlm_logits.py'), ('reference', 'solution')]:
    output = out / name
    command = [*base, '--implementation', implementation, '--output', str(output)]
    with (out / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert {relative: sha(project / relative) for relative in sources} == sources
    binding = dict(checker_sha256=sources['tests/check.py'], exit_code=result.returncode, command=command,
                   log_sha256=sha(out / (name + '.log')), source_files_sha256=sources)
    if name == 'zero-head':
        assert result.returncode != 0
        log = (out / (name + '.log')).read_text()
        assert '_check_mlm_oracle_and_gradients' in log and 'Mismatched elements' in log
        assert not any(output.iterdir()), 'Negative reached resume or optimization output'
        binding.update(candidate_sha256=sha(Path(implementation)), rejected_before_optimization=True,
                       reason='Independent raw MLM logit comparison rejects the same constant-zero candidate previously accepted by the full checker.')
    else:
        assert result.returncode == 0, (out / (name + '.log')).read_text()[-5000:]
        record = output / 'independent-check.json'
        binding.update(independent_check=json.loads(record.read_text()), independent_check_sha256=sha(record))
        assert binding['independent_check']['status'] == 'pass'
    (out / (name + '-verification.json')).write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'case': name, 'exit_code': result.returncode, 'verification': str(out / (name + '-verification.json'))}), flush=True)
