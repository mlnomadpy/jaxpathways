"""Offline package controls and bounded localhost endpoint checks."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Event
from types import MethodType
import unittest

import jax
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import consumer
import model
from config import ModelConfig
from export_consumer import _encode
from unittest.mock import patch
from serve_local import RequestLease, make_handler


class FakeThread:
    def __init__(self, target, daemon=False):
        self.target = target

    def start(self):
        return None

    def join(self, timeout=None):
        return None


class ConsumerTests(unittest.TestCase):
    def make_package(self, root):
        config = ModelConfig(
            vocab_size=4,
            context=16,
            width=8,
            layers=2,
            heads=2,
            mlp_width=12,
            dropout=0.0,
        )
        params = model.initialize(jax.random.PRNGKey(4), config)
        arrays = {}
        spec = _encode(params, arrays)
        with (root / "parameters.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        (root / "parameters.json").write_text(json.dumps(spec))
        (root / "config.json").write_text(json.dumps(config.as_dict()))
        tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3}, unk_token="[UNK]"
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(root / "tokenizer.json"))
        for name in (
            "config.py",
            "consumer.py",
            "model.py",
            "run_pilot.py",
            "serve_local.py",
            "training.py",
        ):
            shutil.copyfile(PROJECT / name, root / name)
        shutil.copyfile(PROJECT / "requirements-c02.txt", root / "requirements.txt")
        files = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "config.py",
                "config.json",
                "consumer.py",
                "model.py",
                "parameters.json",
                "parameters.npz",
                "requirements.txt",
                "run_pilot.py",
                "serve_local.py",
                "tokenizer.json",
                "training.py",
            )
        }
        manifest = {
            "format": "c02-custom-consumer-v1",
            "candidate_source_sha256": consumer._candidate_hash(),
            "files": files,
            "parameter_arrays": sorted(arrays),
            "status": "test package",
        }
        (root / "README.md").write_text("test package")
        (root / "RIGHTS.md").write_text("test notice")
        manifest["files"]["README.md"] = hashlib.sha256(
            (root / "README.md").read_bytes()
        ).hexdigest()
        manifest["files"]["RIGHTS.md"] = hashlib.sha256(
            (root / "RIGHTS.md").read_bytes()
        ).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))
        return config, params

    def test_cold_package_generation_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, params = self.make_package(root)
            loaded = consumer.load_package(root)
            prompt = "hello"
            ids = [1, *loaded.tokenizer.encode(prompt).ids]
            expected = model.generate(params, ids, config, 2, eos_token_id=2)
            actual = loaded.generate(prompt, max_new_tokens=2)
            self.assertEqual(actual["token_ids"], expected)
            with self.assertRaisesRegex(ValueError, "integer from 0 through 32"):
                loaded.generate(prompt, max_new_tokens=True)
            cancellable = loaded._generate_cancellable(ids, 2, Event())
            self.assertEqual(cancellable, expected)
            cancelled = Event()
            cancelled.set()
            with patch.object(model, "prefill", side_effect=AssertionError):
                with self.assertRaisesRegex(RuntimeError, "generation cancelled"):
                    loaded._generate_cancellable(ids, 2, cancelled)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"].pop("config.json")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not cover"):
                consumer.load_package(root)
            manifest["files"]["config.json"] = hashlib.sha256(
                (root / "config.json").read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity check"):
                consumer.load_package(root)

    def test_localhost_handler_routes_and_bounds_without_external_listener(self):
        class StubModel:
            manifest = {"status": "bounded local pilot"}

            def generate(self, prompt, max_new_tokens, cancel_event=None):
                if not prompt or not 0 <= max_new_tokens <= 32:
                    raise ValueError("invalid generation request")
                return {"text": "ok", "token_ids": [3], "finish_reason": "length"}

        handler = make_handler(StubModel())

        def request(path, body):
            instance = object.__new__(handler)
            instance.path = path
            lease = RequestLease(deadline=9999999999)
            instance.server = type(
                "TestServer", (), {"request_lease": lambda *_: lease}
            )()
            instance.connection = object()
            instance.headers = {"Content-Length": str(len(body))}
            instance.rfile = BytesIO(body)
            instance.wfile = BytesIO()
            statuses = []
            instance.send_response = MethodType(
                lambda _self, status: statuses.append(status), instance
            )
            instance.send_header = MethodType(lambda *_args: None, instance)
            instance.end_headers = MethodType(lambda *_args: None, instance)
            instance._write = MethodType(
                lambda self, status, payload: (
                    statuses.append(status),
                    self.wfile.write(json.dumps(payload).encode()),
                ),
                instance,
            )
            return instance, statuses

        health, _ = request("/health", b"")
        handler.do_GET(health)
        self.assertEqual(json.loads(health.wfile.getvalue())["status"], "ready")
        valid, _ = request(
            "/generate", json.dumps({"prompt": "hello", "max_new_tokens": 2}).encode()
        )
        with patch("serve_local.Thread", FakeThread):
            handler.do_POST(valid)
        self.assertEqual(json.loads(valid.wfile.getvalue())["text"], "ok")
        oversized, rejected = request("/generate", b" " * 8193)
        handler.do_POST(oversized)
        self.assertEqual(rejected, [413])


if __name__ == "__main__":
    unittest.main()
