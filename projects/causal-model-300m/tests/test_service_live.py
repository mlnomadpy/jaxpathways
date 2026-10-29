"""Loopback integration checks for queue, timeout, cancellation and isolation.

Run from the project environment where local socket tests are permitted:
`python projects/causal-model-300m/tests/test_service_live.py`.
"""

from __future__ import annotations

from http.client import HTTPConnection
import json
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch

PROJECT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import serve_local


class GatedModel:
    manifest = {"status": "live test model", "branch": "test"}

    def __init__(self):
        self.started = 0
        self.condition = threading.Condition()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def wait_for_started(self, count, timeout=3):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.started < count and time.monotonic() < deadline:
                self.condition.wait(deadline - time.monotonic())
            return self.started >= count

    def generate(self, prompt, max_new_tokens, cancel_event=None):
        with self.condition:
            self.started += 1
            self.condition.notify_all()
        if prompt in {"hold", "cancel", "timeout"}:
            while not self.release.is_set() and not cancel_event.is_set():
                cancel_event.wait(0.01)
            if prompt == "cancel" and cancel_event.is_set():
                self.cancelled.set()
                raise RuntimeError("generation cancelled by client")
            if prompt == "timeout" and cancel_event.is_set():
                raise RuntimeError("generation cancelled by deadline")
        return {"text": prompt, "token_ids": [3], "finish_reason": "length"}


class LiveServiceTests(unittest.TestCase):
    def setUp(self):
        self.model = GatedModel()
        self.server = serve_local.BoundedHTTPServer(
            ("127.0.0.1", 0), serve_local.make_handler(self.model)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_port = self.server.server_port

    def tearDown(self):
        self.model.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, path, payload=None, timeout=3):
        connection = HTTPConnection("127.0.0.1", self.base_port, timeout=timeout)
        body = None if payload is None else json.dumps(payload).encode()
        connection.request(
            method, path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_partial_header_does_not_block_health(self):
        partial = socket.create_connection(("127.0.0.1", self.base_port), timeout=2)
        partial.sendall(b"GET /health HTTP/1.0\r\n")
        status, body = self.request("GET", "/health", timeout=0.7)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["branch"], "test")
        partial.close()

    def test_absolute_admission_deadline_covers_slow_headers_and_body(self):
        with patch.object(serve_local, "MAX_REQUEST_SECONDS", 0.15):
            partial_post_header = socket.create_connection(
                ("127.0.0.1", self.base_port), timeout=2
            )
            partial_post_header.sendall(
                b"POST /generate HTTP/1.0\r\n"
                b"Content-Length: 999999\r\n"
                b"Host: localhost"
            )
            partial_post_header.settimeout(1.0)
            started = time.monotonic()
            partial_header_response = bytearray()
            while True:
                chunk = partial_post_header.recv(4096)
                if not chunk:
                    break
                partial_header_response.extend(chunk)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertIn(b"408 Request Timeout", partial_header_response)
            partial_post_header.close()

            header = socket.create_connection(("127.0.0.1", self.base_port), timeout=2)
            header.sendall(b"POST /generate HTTP/1.0\r\nHost:")
            started = time.monotonic()
            header.settimeout(1.0)
            header_bytes = bytearray()
            while True:
                chunk = header.recv(4096)
                if not chunk:
                    break
                header_bytes.extend(chunk)
            self.assertLess(time.monotonic() - started, 0.8)
            header.close()

            body = socket.create_connection(("127.0.0.1", self.base_port), timeout=2)
            body.sendall(
                b"POST /generate HTTP/1.0\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 128\r\n\r\n"
                b'{"prompt":"unfinished'
            )
            started = time.monotonic()
            body.settimeout(1.0)
            response = bytearray()
            while True:
                chunk = body.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertIn(b"408 Request Timeout", response)
            self.assertIn(b"request lifetime exceeded", response)
            body.close()

        status, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(self.model.started, 0)

    def test_worker_limit_returns_503_and_releases_after_requests(self):
        results = []

        def hold_request():
            results.append(self.request("POST", "/generate", {"prompt": "hold"}))

        workers = [threading.Thread(target=hold_request) for _ in range(2)]
        for worker in workers:
            worker.start()
        self.assertTrue(self.model.wait_for_started(2))
        try:
            status, _ = self.request("GET", "/health")
            self.assertEqual(status, 503)
        finally:
            self.model.release.set()
            for worker in workers:
                worker.join(timeout=3)
        self.assertEqual([status for status, _ in results], [200, 200])

    def test_client_disconnect_cancels_only_its_generation(self):
        connection = HTTPConnection("127.0.0.1", self.base_port, timeout=3)
        payload = json.dumps({"prompt": "cancel"}).encode()
        connection.putrequest("POST", "/generate")
        connection.putheader("Content-Length", str(len(payload)))
        connection.endheaders(payload)
        self.assertTrue(self.model.wait_for_started(1))
        connection.close()
        self.assertTrue(self.model.cancelled.wait(timeout=2))
        status, body = self.request("POST", "/generate", {"prompt": "ok"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["text"], "ok")

    def test_generation_deadline_returns_gateway_timeout(self):
        with patch.object(serve_local, "MAX_REQUEST_SECONDS", 0.15):
            status, body = self.request("POST", "/generate", {"prompt": "timeout"})
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(body)["error"], "request lifetime exceeded")


if __name__ == "__main__":
    unittest.main()
