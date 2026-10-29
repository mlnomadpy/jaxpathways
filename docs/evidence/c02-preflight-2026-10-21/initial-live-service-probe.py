from pathlib import Path
import hashlib
import http.client
import json
import socket
import subprocess
import time

PROJECT = Path('/Users/tahabsn/Documents/ChatGPT/jaxpathways/projects/causal-model-300m')
PACKAGE = Path('/private/tmp/jaxpathways-c02-evaluation-20261021-final2/consumer-sft-full-domain')
OUT = Path('/private/tmp/jaxpathways-c02-live-service-probe')
OUT.mkdir(exist_ok=True)
PORT = 48127
SOURCES = ['serve_local.py', 'consumer.py', 'model.py', 'config.py']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {name: sha(PROJECT / name) for name in SOURCES}


def request(method, path, value=None, timeout=30):
    connection = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    try:
        body = None if value is None else json.dumps(value).encode()
        connection.request(method, path, body=body, headers={'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        data = response.read()
        return {'status': response.status, 'body': json.loads(data)}
    finally:
        connection.close()


record = {'package': str(PACKAGE), 'port': PORT, 'sourceSha256': before, 'scope': 'Actual loopback listener smoke and one partial-header isolation probe; not production or target qualification.'}
with (OUT / 'server.log').open('w') as log:
    process = subprocess.Popen(['/private/tmp/jaxpathways-foundations-venv/bin/python', str(PROJECT / 'serve_local.py'), '--package', str(PACKAGE), '--port', str(PORT)], stdout=log, stderr=subprocess.STDOUT, cwd='/private/tmp')
    record['ownedPid'] = process.pid
    try:
        deadline = time.monotonic() + 45
        while True:
            try:
                record['health'] = request('GET', '/health', timeout=1)
                break
            except (OSError, http.client.HTTPException):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('server failed to become ready')
                time.sleep(0.1)
        record['invalidType'] = request('POST', '/generate', {'prompt': 'hello', 'max_new_tokens': True})
        record['valid'] = request('POST', '/generate', {'prompt': 'Question: Who wrote this? Context: A writer created a book.\nAnswer:', 'max_new_tokens': 2})
        idle = socket.create_connection(('127.0.0.1', PORT), timeout=2)
        try:
            idle.sendall(b'GET /health HTTP/1.1\r\n')
            time.sleep(0.1)
            try:
                record['healthDuringPartialHeader'] = request('GET', '/health', timeout=0.7)
            except (OSError, http.client.HTTPException) as error:
                record['healthDuringPartialHeader'] = {'errorType': type(error).__name__, 'error': str(error)}
        finally:
            idle.close()
        record['healthAfterRelease'] = request('GET', '/health', timeout=3)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        record['ownedServerStopped'] = process.poll() is not None
        record['sourceUnchanged'] = before == {name: sha(PROJECT / name) for name in SOURCES}
        (OUT / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
