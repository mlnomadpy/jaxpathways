"""Independent real-socket bounded parser probe against a frozen source copy."""
from pathlib import Path
import hashlib
import http.client
import json
import socket
import sys
import threading
import time

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT/'source'))
import serve_local

class NoInference:
    manifest = {'status': 'parser-only probe', 'branch': 'none'}
    def generate(self, *args, **kwargs):
        raise AssertionError('incomplete request must not enter inference')

serve_local.MAX_REQUEST_SECONDS = .15
server = serve_local.BoundedHTTPServer(('127.0.0.1', 0), serve_local.make_handler(NoInference()))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
record = {'scope': 'Actual loopback parser admission/header/body deadline with a controlled 150 ms budget and no inference. Does not establish an interruptible native JAX operation.', 'serve_source_sha256': hashlib.sha256((OUT/'source/serve_local.py').read_bytes()).hexdigest(), 'cases': []}
def health():
    c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        c.request('GET','/health'); r=c.getresponse(); body=r.read()
        assert r.status == 200
        return json.loads(body)
    finally:
        c.close()
try:
    for name, payload in [('incomplete_body', b'POST /generate HTTP/1.0\r\nContent-Length: 8192\r\n\r\n{'), ('incomplete_header', b'POST /generate HTTP/1.0\r\nX-Incomplete: ')]:
        peer=socket.create_connection(('127.0.0.1',server.server_port),timeout=2)
        started=time.monotonic(); peer.sendall(payload); peer.settimeout(.02)
        response=b''
        try:
            while time.monotonic()-started < .65:
                try:
                    fragment=peer.recv(4096)
                    response+=fragment
                    if not fragment: break
                except socket.timeout:
                    try: peer.sendall(b' ')
                    except OSError: break
            elapsed=time.monotonic()-started
            assert elapsed < .5, (name,elapsed,response)
            assert b'408 Request Timeout' in response, (name,response)
            record['cases'].append({'name':name,'elapsed_seconds':elapsed,'response':response.decode(errors='replace'),'recovery_health':health()})
        finally:
            peer.close()
    # Two incomplete clients consume exactly the admission capacity. A third
    # must get a complete, correctly framed 503 body immediately.
    peers=[]
    try:
        for _ in range(2):
            p=socket.create_connection(('127.0.0.1',server.server_port),timeout=2)
            p.sendall(b'GET /health HTTP/1.0\r\nX-Incomplete: '); peers.append(p)
        deadline=time.monotonic()+.1
        while len(server._leases) < 2 and time.monotonic() < deadline: time.sleep(.001)
        c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=1)
        try:
            c.request('GET','/health');r=c.getresponse();body=r.read()
            assert r.status == 503 and json.loads(body)['error']=='request capacity full'
            record['overload']={'status':r.status,'body':json.loads(body)}
        finally: c.close()
    finally:
        for p in peers: p.close()
    time.sleep(.2)
    record['final_health']=health()
    record['status']='pass'
finally:
    server.shutdown();server.server_close();thread.join(timeout=3)
    record['owned_server_stopped']=not thread.is_alive()
    (OUT/'transport-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
