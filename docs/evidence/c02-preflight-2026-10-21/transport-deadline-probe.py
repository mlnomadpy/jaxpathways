from pathlib import Path
import hashlib
import http.client
import json
import socket
import sys
import threading
import time

PROJECT = Path('/Users/tahabsn/Documents/ChatGPT/jaxpathways/projects/causal-model-300m')
sys.path.insert(0, str(PROJECT))
import serve_local


class UnusedModel:
    manifest = {'status': 'transport probe only', 'branch': 'probe'}

    def generate(self, *args, **kwargs):
        raise AssertionError('incomplete request must never start inference')


serve_local.MAX_REQUEST_SECONDS = 0.15
server = serve_local.BoundedHTTPServer(('127.0.0.1', 0), serve_local.make_handler(UnusedModel()))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
record = {'serveSourceSha256': hashlib.sha256((PROJECT / 'serve_local.py').read_bytes()).hexdigest(), 'declaredLifetimeSeconds': 0.15, 'scope': 'Actual loopback incomplete-body deadline probe; no model inference.'}
peer = socket.create_connection(('127.0.0.1', server.server_port), timeout=2)
try:
    peer.sendall(b'POST /generate HTTP/1.0\r\nContent-Length: 8192\r\n\r\n{')
    started = time.monotonic()
    for _ in range(8):
        time.sleep(0.1)
        peer.sendall(b' ')
    record['observedElapsedSeconds'] = time.monotonic() - started
    peer.settimeout(0.2)
    try:
        record['responseOrEOF'] = peer.recv(4096).decode(errors='replace')
    except socket.timeout:
        record['responseOrEOF'] = 'no response or EOF before probe timeout'
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=1)
    connection.request('GET', '/health')
    response = connection.getresponse()
    record['health'] = {'status': response.status, 'body': json.loads(response.read())}
    connection.close()
finally:
    peer.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    record['ownedServerStopped'] = not thread.is_alive()
path = Path('/private/tmp/jaxpathways-c02-transport-deadline-result.json')
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
