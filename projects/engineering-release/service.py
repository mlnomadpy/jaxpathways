"""A deliberately small stdlib inference boundary shared by local and container checks."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def load_model(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('model digest mismatch')
    model = json.loads(raw)
    if set(model) != {'schema', 'weight', 'bias'} or model['schema'] != 1:
        raise ValueError('unsupported model schema')
    if any(type(model[k]) not in (float, int) or not math.isfinite(model[k]) for k in ('weight', 'bias')):
        raise ValueError('invalid coefficients')
    return model


def predict(model, request):
    if set(request) != {'inputs'} or not isinstance(request['inputs'], list) or not 1 <= len(request['inputs']) <= 32:
        raise ValueError('expected one to 32 input numbers')
    values = request['inputs']
    if any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 100 for x in values):
        raise ValueError('inputs must be finite numbers in [-100,100]')
    predictions = [model['weight'] * x + model['bias'] for x in values]
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError('nonfinite prediction')
    return {'predictions': predictions}


def main():
    model = load_model(os.environ['MODEL_PATH'], os.environ['MODEL_SHA256'])
    if '--check' in sys.argv:
        assert math.isfinite(predict(model, {'inputs': [0.]})['predictions'][0])
        print('ready: artifact verified and prediction completed')
    elif '--predict' in sys.argv:
        print(json.dumps(predict(model, json.load(sys.stdin)), allow_nan=False))
    else:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                if self.path != '/ready':
                    self.reply(404, {'error': 'unknown route'})
                    return
                predict(model, {'inputs': [0.]})
                self.reply(200, {'ready': True})
            def do_POST(self):
                if self.path != '/predict':
                    self.reply(404, {'error': 'unknown route'})
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 4096:
                        raise ValueError('request size')
                    request = json.loads(self.rfile.read(size))
                    self.reply(200, predict(model, request))
                except (ValueError, TypeError, KeyError):
                    self.reply(400, {'error': 'invalid input contract'})
            def log_message(self, *args): pass  # Do not emit raw request content.
        HTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8080'))), Handler).serve_forever()


if __name__ == '__main__':
    main()
