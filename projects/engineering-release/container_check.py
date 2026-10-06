"""Qualify only the teaching image created here; no published ports or external deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def command(args, **kw):
    return subprocess.run(args, text=True, capture_output=True, check=True, timeout=120, **kw)


def run(model_path):
    model_path = Path(model_path).resolve()
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = json.loads(model_path.read_text())
    image = 'jaxpathways-engineering:course'
    command(['docker', 'build', '-t', image, str(ROOT)])
    inspected = json.loads(command(['docker', 'image', 'inspect', image]).stdout)[0]
    common = ['--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--mount', f'type=bind,source={model_path},target=/model/model.json,readonly', '-e', 'MODEL_PATH=/model/model.json']
    args = ['docker', 'run', '--rm', '-i', *common, '-e', f'MODEL_SHA256={sha}', image]
    values = [-.37, 0., 2.2]
    result = json.loads(command([*args, '--predict'], input=json.dumps({'inputs': values})).stdout)
    expected = [model['weight'] * x + model['bias'] for x in values]
    assert result['predictions'] == expected
    readiness = command([*args, '--check']).stdout.strip()
    bad = subprocess.run(['docker', 'run', '--rm', *common, '-e', 'MODEL_SHA256=' + '0' * 64, image, '--check'], text=True, capture_output=True, timeout=30)
    assert bad.returncode != 0 and 'model digest mismatch' in bad.stderr
    container = command(['docker', 'run', '--detach', *common, '-e', f'MODEL_SHA256={sha}', image]).stdout.strip()
    try:
        # Poll inside this network-isolated container. The local machine publishes no port.
        probe = '''import json,time,urllib.request,urllib.error,os
for attempt in range(40):
    try:
        ready=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/ready',timeout=1).read());break
    except OSError:time.sleep(.1)
else:raise RuntimeError('service did not become ready')
assert ready['ready'] is True and os.getuid()==10001
request=urllib.request.Request('http://127.0.0.1:8080/predict',data=json.dumps({'inputs':[0.,1.]}).encode(),headers={'Content-Type':'application/json'})
response=json.loads(urllib.request.urlopen(request,timeout=2).read())
bad=urllib.request.Request('http://127.0.0.1:8080/predict',data=b'{"inputs":[]}',headers={'Content-Type':'application/json'})
try:urllib.request.urlopen(bad,timeout=2)
except urllib.error.HTTPError as error:assert error.code==400
else:raise AssertionError('invalid request accepted')
print(json.dumps({'uid':os.getuid(),'ready':ready,'response':response,'invalid_request_status':400}))'''
        http = json.loads(command(['docker', 'exec', container, 'python', '-c', probe]).stdout)
        assert http['response']['predictions'] == [model['bias'], model['weight'] + model['bias']]
    finally:
        command(['docker', 'rm', '-f', container])
    return {'image_id': inspected['Id'], 'os': inspected['Os'], 'architecture': inspected['Architecture'], 'configured_user': inspected['Config']['User'], 'model_sha256': sha, 'predictions': result['predictions'], 'readiness': readiness, 'http': http, 'bad_digest_rejected': True, 'source_sha256': hashlib.sha256((ROOT/'service.py').read_bytes()).hexdigest(), 'dockerfile_sha256': hashlib.sha256((ROOT/'Dockerfile').read_bytes()).hexdigest(), 'scope': 'Actual local Docker CPU image, CLI and internal HTTP checks. No published port, cloud rollout or load benchmark.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', required=True); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = run(args.model); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
