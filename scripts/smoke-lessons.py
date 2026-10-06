"""Execute authored script and notebook companions on CPU; emit a bounded validation receipt."""
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
course = json.loads((ROOT / 'public/curriculum.json').read_text())
env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
records = []
for phase in course['phases']:
    for lesson in phase['lessons']:
        if lesson['status'] != 'authored':
            continue
        script = ROOT / lesson['artifacts']['scriptSource']
        notebook = json.loads((ROOT / lesson['artifacts']['notebookSource']).read_text())
        code = '\n\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
        # Notebook cells execute in order with shared state, just as in a fresh kernel.
        runtime_probe = "\nimport jax, json\nprint('COURSE_RUNTIME:' + json.dumps({'backend':jax.default_backend(),'deviceCount':jax.local_device_count()}))"
        observed = []
        notebook_result = None
        script_stdout = ''
        with tempfile.TemporaryDirectory() as temporary:
            notebook_output = Path(temporary) / 'execution.json'
            for kind in ['script', 'notebook']:
                command = ([sys.executable, '-c', script.read_text() + runtime_probe] if kind == 'script' else
                           [sys.executable, str(ROOT/'scripts/execute-notebook.py'), str(ROOT/lesson['artifacts']['notebookSource']), str(notebook_output)])
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    print(f"FAIL {lesson['id']} ({kind})\n{result.stdout}\n{result.stderr}", file=sys.stderr)
                    sys.exit(1)
                if kind == 'script':
                    marker = next(line.removeprefix('COURSE_RUNTIME:') for line in result.stdout.splitlines() if line.startswith('COURSE_RUNTIME:'))
                    runtime = json.loads(marker)
                    script_stdout = '\n'.join(line for line in result.stdout.splitlines() if not line.startswith('COURSE_RUNTIME:')) + '\n'
                else:
                    notebook_result = json.loads(notebook_output.read_text())
                    runtime = notebook_result['runtime']
                expected = lesson['content'].get('runtime', {}).get('deviceCount', 1)
                if runtime['backend'] != 'cpu' or runtime['deviceCount'] != expected:
                    raise RuntimeError(f"{lesson['id']}: expected CPU with {expected} devices, observed {runtime}")
                observed.append(runtime)
        execution = {'schemaVersion': 1, 'lessonId': lesson['id'], 'contentHash': lesson['contentHash'],
                     'executedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'environment': {**observed[0], 'python': platform.python_version(),
                                     **{name: importlib.metadata.version(name) for name in ['jax','numpy','matplotlib']}},
                     'stdout': script_stdout, 'notebookCells': notebook_result['notebookCells'],
                     'scope': 'Recorded instructor reference run; all script assertions and notebook code cells passed. No accelerator performance claim.'}
        (ROOT / lesson['path'] / 'outputs/execution.json').write_text(json.dumps(execution, indent=2) + '\n')
        records.append({'lessonId': lesson['id'], 'contentHash': lesson['contentHash'], 'script': 'passed', 'notebookCode': 'passed', 'runtime': observed[0]})
        print(f"PASS {lesson['id']}: script + notebook code", flush=True)
import jax
import numpy
import importlib.metadata
receipt = {'backend': 'cpu', 'python': platform.python_version(), 'jax': jax.__version__, 'numpy': numpy.__version__, 'packages': {name:importlib.metadata.version(name) for name in ['jax','numpy','optax','flax','grain','orbax-checkpoint','matplotlib']}, 'validatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Worked examples and reference solutions executed in fresh processes. No TPU validation or learner assessment.', 'lessons': records}
(ROOT / 'curriculum/validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(f"OK: {len(records)} CPU lessons; {len(records)*2} executions")
