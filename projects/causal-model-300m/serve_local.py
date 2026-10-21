"""Bounded, single-process localhost consumer for a C02 pilot package."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from threading import Event, Semaphore, Thread, Timer

from consumer import load_package

MAX_BODY_BYTES = 8192
MAX_ACTIVE_REQUESTS = 2
MAX_REQUEST_SECONDS = 30.0
SOCKET_IDLE_TIMEOUT_SECONDS = 2.0


@dataclass
class RequestLease:
    """Absolute admission deadline shared by parsing and model generation."""

    deadline: float
    expired: Event = field(default_factory=Event)
    timer: Timer | None = None


class BoundedHTTPServer(ThreadingHTTPServer):
    """Loopback server with a small worker bound, reject-on-full and idle timeout."""

    request_queue_size = 2
    daemon_threads = True

    def __init__(self, address, handler):
        if address[0] != "127.0.0.1":
            raise ValueError("C02 pilot service must bind only to IPv4 loopback")
        self._workers = Semaphore(MAX_ACTIVE_REQUESTS)
        self._lease_lock = threading.Lock()
        self._leases = {}
        super().__init__(address, handler)

    def request_lease(self, request):
        with self._lease_lock:
            return self._leases.get(request.fileno())

    def _expire_request(self, request, lease):
        lease.expired.set()
        try:
            # Wake a stalled request-line/header/body read while preserving the
            # write half so the handler can return a bounded timeout response.
            request.shutdown(socket.SHUT_RD)
        except OSError:
            pass

    def process_request(self, request, client_address):
        if not self._workers.acquire(blocking=False):
            try:
                request.settimeout(0.5)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 33\r\nConnection: close\r\n\r\n"
                    b'{"error":"request capacity full"}'
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        lease = RequestLease(deadline=time.monotonic() + MAX_REQUEST_SECONDS)
        lease.timer = Timer(
            MAX_REQUEST_SECONDS, self._expire_request, args=(request, lease)
        )
        lease.timer.daemon = True
        with self._lease_lock:
            self._leases[request.fileno()] = lease
        lease.timer.start()
        lease_key = request.fileno()
        try:
            super().process_request(request, client_address)
        except BaseException:
            lease.timer.cancel()
            with self._lease_lock:
                self._leases.pop(lease_key, None)
            self._workers.release()
            raise

    def process_request_thread(self, request, client_address):
        lease_key = request.fileno()
        lease = self.request_lease(request)
        try:
            request.settimeout(SOCKET_IDLE_TIMEOUT_SECONDS)
            super().process_request_thread(request, client_address)
        finally:
            if lease and lease.timer:
                lease.timer.cancel()
            with self._lease_lock:
                self._leases.pop(lease_key, None)
            self._workers.release()


def make_handler(loaded_model):
    class Handler(BaseHTTPRequestHandler):
        server_version = "C02PilotLocal/1"
        protocol_version = "HTTP/1.0"

        def _write(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def do_GET(self):
            self.close_connection = True
            lease = self.server.request_lease(self.connection)
            if lease and lease.expired.is_set():
                self._write(408, {"error": "request lifetime exceeded"})
                return
            if self.path != "/health":
                self._write(404, {"error": "route not found"})
                return
            self._write(200, _health_payload(loaded_model))

        def do_POST(self):
            self.close_connection = True
            lease = self.server.request_lease(self.connection)
            if lease is None or lease.expired.is_set():
                self._write(408, {"error": "request lifetime exceeded"})
                return
            if self.path != "/generate":
                self._write(404, {"error": "route not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY_BYTES:
                self._write(
                    413,
                    {"error": f"request body must be at most {MAX_BODY_BYTES} bytes"},
                )
                return
            try:
                if lease.expired.is_set():
                    self._write(408, {"error": "request lifetime exceeded"})
                    return
                raw_body = self.rfile.read(length)
                if lease.expired.is_set() or len(raw_body) != length:
                    self._write(408, {"error": "request lifetime exceeded"})
                    return
                request = json.loads(raw_body)
                if not isinstance(request, dict) or set(request) - {
                    "prompt",
                    "max_new_tokens",
                }:
                    raise ValueError("send only prompt and max_new_tokens")
                cancelled = Event()
                client_disconnected = Event()
                timed_out = Event()
                monitor_stop = Event()

                def watch_disconnect():
                    while not monitor_stop.wait(0.05):
                        if lease.expired.is_set():
                            timed_out.set()
                            cancelled.set()
                            return
                        try:
                            readable, _, _ = select.select([self.connection], [], [], 0)
                            if (
                                readable
                                and self.connection.recv(1, socket.MSG_PEEK) == b""
                            ):
                                if not lease.expired.is_set():
                                    client_disconnected.set()
                                    cancelled.set()
                                return
                        except (OSError, ValueError):
                            cancelled.set()
                            return

                monitor = Thread(target=watch_disconnect, daemon=True)
                monitor.start()
                if lease.expired.is_set():
                    timed_out.set()
                    cancelled.set()
                try:
                    result = loaded_model.generate(
                        request.get("prompt"),
                        request.get("max_new_tokens", 32),
                        cancel_event=cancelled,
                    )
                except RuntimeError as error:
                    if "cancelled" not in str(error):
                        raise
                    if timed_out.is_set():
                        self._write(504, {"error": "request lifetime exceeded"})
                    return
                finally:
                    monitor_stop.set()
                    monitor.join(timeout=0.2)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
                TimeoutError,
            ) as error:
                self._write(
                    408 if isinstance(error, TimeoutError) else 400,
                    (
                        {"error": "request body read timed out"}
                        if isinstance(error, TimeoutError)
                        else {"error": str(error)}
                    ),
                )
                return
            if lease.expired.is_set():
                timed_out.set()
            if timed_out.is_set():
                self._write(504, {"error": "request lifetime exceeded"})
                return
            if client_disconnected.is_set():
                return
            self._write(200, result)

        def log_message(self, _format, *_args):
            return

    return Handler


def _health_payload(loaded_model):
    manifest = loaded_model.manifest
    manifest_path = Path(getattr(loaded_model, "package_dir", ".")) / "manifest.json"
    manifest_bytes = (
        manifest_path.read_bytes()
        if manifest_path.is_file()
        else json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "status": "ready",
        "model_status": manifest["status"],
        "branch": manifest.get("branch", "local-test-model"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "request_limits": {
            "body_bytes": MAX_BODY_BYTES,
            "active_requests": MAX_ACTIVE_REQUESTS,
            "lifetime_seconds": MAX_REQUEST_SECONDS,
            "cancellation": "cooperative-between-generation-steps",
            "max_new_tokens": 32,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8127)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    loaded_model = load_package(args.package)
    server = BoundedHTTPServer(("127.0.0.1", args.port), make_handler(loaded_model))
    print(
        f"serving package {loaded_model.manifest['branch']} on 127.0.0.1:{args.port}; "
        f"max active requests={MAX_ACTIVE_REQUESTS}, request body={MAX_BODY_BYTES} bytes, "
        f"request lifetime={MAX_REQUEST_SECONDS:.0f}s; Ctrl-C to stop",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
