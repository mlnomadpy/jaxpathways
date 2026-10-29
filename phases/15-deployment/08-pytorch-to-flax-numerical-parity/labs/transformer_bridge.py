"""Bounded PyTorch -> Flax NNX decoder conversion fixture.

This is an educational conversion for one tiny architecture, not a generic
checkpoint importer. It creates a local source checkpoint so no model download
or credentials are required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

torch.set_num_threads(1)

VOCAB = {"<pad>": 0, "red": 1, "green": 2, "blue": 3, "gold": 4, "black": 5, "white": 6}
PROCESSOR_ID = "toy-wordpiece-v1:" + json.dumps(
    VOCAB, sort_keys=True, separators=(",", ":")
)
D, HEADS, MLP, VOCAB_SIZE, MAX_POS, EPS = 8, 2, 16, len(VOCAB), 6, 1e-5
ATOL, RTOL = 3e-6, 3e-5


class TorchBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(D, 3 * D)
        self.attn_out = torch.nn.Linear(D, D)
        self.mlp_in = torch.nn.Linear(D, MLP)
        self.mlp_out = torch.nn.Linear(MLP, D)
        self.token_embedding = torch.nn.Embedding(VOCAB_SIZE, D, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(MAX_POS, D)

    def forward(self, token_ids, position_ids, keep_mask, *, wrong_mask=False):
        x = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        residual = x
        n = torch.nn.functional.layer_norm(x, (D,), eps=EPS)
        qkv = self.qkv(n)
        q, k, v = qkv.reshape(*qkv.shape[:-1], 3, HEADS, D // HEADS).permute(
            2, 0, 3, 1, 4
        )
        scores = q @ k.transpose(-1, -2) / np.sqrt(D // HEADS)
        causal = torch.tril(
            torch.ones(token_ids.shape[1], token_ids.shape[1], dtype=torch.bool)
        )
        allowed = causal[None, None, :, :] & keep_mask[:, None, None, :]
        if wrong_mask:
            allowed = causal[None, None, :, :] & (~keep_mask[:, None, None, :])
        probs = torch.softmax(
            scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1
        )
        attended = (probs @ v).transpose(1, 2).reshape(*token_ids.shape, D)
        attn = self.attn_out(attended)
        residual_attn = residual + attn
        n2 = torch.nn.functional.layer_norm(residual_attn, (D,), eps=EPS)
        hidden = torch.nn.functional.gelu(self.mlp_in(n2), approximate="none")
        mlp = self.mlp_out(hidden)
        output = residual_attn + mlp
        logits = output @ self.token_embedding.weight.T
        return {
            "embedding": x,
            "norm1": n,
            "qkv": qkv,
            "attention": attended,
            "attn_projection": attn,
            "residual1": residual_attn,
            "norm2": n2,
            "mlp_hidden": hidden,
            "mlp_projection": mlp,
            "block": output,
            "logits": logits,
        }


class NNXBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(2026)
        self.token_embedding = nnx.Embed(VOCAB_SIZE, D, rngs=rngs)
        self.position_embedding = nnx.Embed(MAX_POS, D, rngs=rngs)
        self.qkv = nnx.Linear(D, 3 * D, rngs=rngs)
        self.attn_out = nnx.Linear(D, D, rngs=rngs)
        self.mlp_in = nnx.Linear(D, MLP, rngs=rngs)
        self.mlp_out = nnx.Linear(MLP, D, rngs=rngs)

    def __call__(
        self,
        token_ids,
        position_ids,
        keep_mask,
        *,
        wrong_head_layout=False,
        wrong_mask=False,
    ):
        x = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        residual = x
        n = (x - jnp.mean(x, axis=-1, keepdims=True)) * jax.lax.rsqrt(
            jnp.mean(
                jnp.square(x - jnp.mean(x, axis=-1, keepdims=True)),
                axis=-1,
                keepdims=True,
            )
            + EPS
        )
        qkv = self.qkv(n)
        b, t, _ = qkv.shape
        if wrong_head_layout:
            # Deliberate plausible error: interpret fused [Q_all,K_all,V_all] as
            # per-head [Q,K,V]. Shapes remain valid but head/channel meaning changes.
            q, k, v = qkv.reshape(b, t, HEADS, 3, D // HEADS).transpose(3, 0, 2, 1, 4)
        else:
            q, k, v = qkv.reshape(b, t, 3, HEADS, D // HEADS).transpose(2, 0, 3, 1, 4)
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D // HEADS)
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))
        valid_keys = ~keep_mask if wrong_mask else keep_mask
        allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
        scores = jnp.where(allowed, scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores, axis=-1)
        attended = (
            jnp.einsum("bhts,bhsd->bhtd", probs, v)
            .transpose(0, 2, 1, 3)
            .reshape(b, t, D)
        )
        attn = self.attn_out(attended)
        residual_attn = residual + attn
        n2 = (
            residual_attn - jnp.mean(residual_attn, axis=-1, keepdims=True)
        ) * jax.lax.rsqrt(
            jnp.mean(
                jnp.square(
                    residual_attn - jnp.mean(residual_attn, axis=-1, keepdims=True)
                ),
                axis=-1,
                keepdims=True,
            )
            + EPS
        )
        hidden = jax.nn.gelu(self.mlp_in(n2), approximate=False)
        mlp = self.mlp_out(hidden)
        output = residual_attn + mlp
        logits = output @ self.token_embedding.embedding[...].T  # tied decoder head
        return {
            "embedding": x,
            "norm1": n,
            "qkv": qkv,
            "attention": attended,
            "attn_projection": attn,
            "residual1": residual_attn,
            "norm2": n2,
            "mlp_hidden": hidden,
            "mlp_projection": mlp,
            "block": output,
            "logits": logits,
        }


def source_to_target(state):
    expected = {
        "qkv.weight": (3 * D, D),
        "qkv.bias": (3 * D,),
        "attn_out.weight": (D, D),
        "attn_out.bias": (D,),
        "mlp_in.weight": (MLP, D),
        "mlp_in.bias": (MLP,),
        "mlp_out.weight": (D, MLP),
        "mlp_out.bias": (D,),
        "token_embedding.weight": (VOCAB_SIZE, D),
        "position_embedding.weight": (MAX_POS, D),
    }
    if set(state) != set(expected):
        raise ValueError("source state keys do not match bounded decoder contract")
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape or state[name].dtype != torch.float32:
            raise ValueError(f"invalid source tensor {name}")
    target = NNXBlock()
    # PyTorch Linear is [out,in]; NNX Linear kernel is [in,out].
    # qkv rows are contiguous [Q_all,K_all,V_all], each containing all heads.
    mapping = [
        ("qkv.weight", target.qkv.kernel, "transpose (3D,D)->(D,3D)"),
        ("attn_out.weight", target.attn_out.kernel, "transpose (D,D)"),
        ("mlp_in.weight", target.mlp_in.kernel, "transpose (MLP,D)->(D,MLP)"),
        ("mlp_out.weight", target.mlp_out.kernel, "transpose (D,MLP)->(MLP,D)"),
    ]
    for src_name, dst, _ in mapping:
        dst[...] = jnp.asarray(state[src_name].detach().cpu().numpy().T.copy())
    for name, module, attr in [
        ("qkv.bias", target.qkv, "bias"),
        ("attn_out.bias", target.attn_out, "bias"),
        ("mlp_in.bias", target.mlp_in, "bias"),
        ("mlp_out.bias", target.mlp_out, "bias"),
    ]:
        getattr(module, attr)[...] = jnp.asarray(
            state[name].detach().cpu().numpy().copy()
        )
    target.token_embedding.embedding[...] = jnp.asarray(
        state["token_embedding.weight"].detach().cpu().numpy().copy()
    )
    target.position_embedding.embedding[...] = jnp.asarray(
        state["position_embedding.weight"].detach().cpu().numpy().copy()
    )
    return target


def compare(reference, actual):
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    ref, got = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    absolute = float(np.max(np.abs(ref - got)))
    relative = float(np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-12))
    passed = bool(
        ref.shape == got.shape
        and np.all(np.abs(ref - got) <= ATOL + RTOL * np.abs(ref))
    )
    return {"absolute": absolute, "relative_l2": relative, "passed": passed}


def first_divergence(reference_outputs, actual_outputs):
    for name in reference_outputs:
        report = compare(reference_outputs[name], actual_outputs[name])
        if not report["passed"]:
            return {"layer": name, **report}
    return {"layer": None, "passed": True}


def assign_weights(model, arrays):
    for module, attr, name in [
        (model.qkv, "kernel", "qkv_kernel"),
        (model.qkv, "bias", "qkv_bias"),
        (model.attn_out, "kernel", "attn_out_kernel"),
        (model.attn_out, "bias", "attn_out_bias"),
        (model.mlp_in, "kernel", "mlp_in_kernel"),
        (model.mlp_in, "bias", "mlp_in_bias"),
        (model.mlp_out, "kernel", "mlp_out_kernel"),
        (model.mlp_out, "bias", "mlp_out_bias"),
    ]:
        getattr(module, attr)[...] = jnp.asarray(arrays[name])
    model.token_embedding.embedding[...] = jnp.asarray(arrays["token_embedding"])
    model.position_embedding.embedding[...] = jnp.asarray(arrays["position_embedding"])


def child_reload(archive_path, input_path):
    with np.load(archive_path, allow_pickle=False) as z:
        if str(z["processor_id"].item()) != PROCESSOR_ID:
            raise ValueError("processor identity mismatch")
        if int(z["schema"].item()) != 1:
            raise ValueError("unknown converted artifact schema")
        model = NNXBlock()
        arrays = {
            k: z[k].copy() for k in z.files if k not in ("schema", "processor_id")
        }
        assign_weights(model, arrays)
        x = np.load(input_path, allow_pickle=False)
        got = model(
            jnp.asarray(x["ids"]),
            jnp.asarray(x["positions"]),
            jnp.asarray(x["keep_mask"]),
        )
        print(json.dumps({k: np.asarray(got[k]).tolist() for k in ("block", "logits")}))


def run():
    torch.manual_seed(83)
    source = TorchBlock().eval()
    with torch.no_grad():
        source.token_embedding.weight[0].zero_()
    # Saved local fixture, then load it back as a real state_dict boundary.
    ids = np.array([[1, 4, 2, 0], [3, 1, 5, 6]], np.int64)
    positions = np.array(
        [[0, 2, 3, 5], [1, 0, 4, 2]], np.int64
    )  # intentionally nonsequential
    keep_mask = ids != 0
    inputs = (torch.tensor(ids), torch.tensor(positions), torch.tensor(keep_mask))
    with tempfile.TemporaryDirectory(prefix="jaxpathways-transformer-bridge-") as td:
        root = Path(td)
        source_path, converted_path, input_path = (
            root / "source.pt",
            root / "nnx.npz",
            root / "inputs.npz",
        )
        torch.save(source.state_dict(), source_path)
        state = torch.load(source_path, map_location="cpu", weights_only=True)
        target = source_to_target(state)
        np.savez(input_path, ids=ids, positions=positions, keep_mask=keep_mask)
        torch_outputs = source(*inputs)
        flax_outputs = target(
            jnp.asarray(ids), jnp.asarray(positions), jnp.asarray(keep_mask)
        )
        reports = []
        for name in torch_outputs:
            report = compare(torch_outputs[name].detach().numpy(), flax_outputs[name])
            reports.append((name, report))
            if not report["passed"]:
                raise AssertionError(f"first divergence at {name}: {report}")
        # The output projection must use the same value as the embedding table.
        tied_logits = flax_outputs["block"] @ target.token_embedding.embedding[...].T
        np.testing.assert_allclose(tied_logits, flax_outputs["logits"], atol=0, rtol=0)
        # Nonuniform mixed-sign cotangent makes this a directional VJP, not merely sum(outputs).
        cotangent = np.linspace(
            -0.7, 1.1, 2 * 4 * VOCAB_SIZE, dtype=np.float32
        ).reshape(2, 4, VOCAB_SIZE)
        tx = torch.tensor(ids, requires_grad=False)
        source_input = source.token_embedding(tx) + source.position_embedding(
            torch.tensor(positions)
        )
        source_input = source_input.detach().requires_grad_(True)

        # A small helper factors the block from token embeddings so input VJP is directly observable.
        def torch_from_embedding(z, wrong_mask=False):
            residual = z
            n = torch.nn.functional.layer_norm(z, (D,), eps=EPS)
            qkv = source.qkv(n)
            q, k, v = qkv.reshape(2, 4, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
            scores = q @ k.transpose(-1, -2) / np.sqrt(D // HEADS)
            allowed = (
                torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
                & torch.tensor(keep_mask)[:, None, None, :]
            )
            if wrong_mask:
                allowed = (
                    torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
                    & (~torch.tensor(keep_mask))[:, None, None, :]
                )
            probs = torch.softmax(
                scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), -1
            )
            a = (probs @ v).transpose(1, 2).reshape(2, 4, D)
            r1 = z + source.attn_out(a)
            n2 = torch.nn.functional.layer_norm(r1, (D,), eps=EPS)
            h = torch.nn.functional.gelu(source.mlp_in(n2), approximate="none")
            return r1 + source.mlp_out(h)

        torch_scalar = (
            torch_from_embedding(source_input)
            @ source.token_embedding.weight.T
            * torch.tensor(cotangent)
        ).sum()
        torch_grad = torch.autograd.grad(torch_scalar, source_input)[0].numpy()

        def jax_scalar(z):
            # Norm + block from inputs; use token IDs only to preserve same padding mask.
            x = z
            n = (x - jnp.mean(x, -1, keepdims=True)) * jax.lax.rsqrt(
                jnp.mean((x - jnp.mean(x, -1, keepdims=True)) ** 2, -1, keepdims=True)
                + EPS
            )
            qkv = target.qkv(n).reshape(2, 4, 3, HEADS, D // HEADS)
            q, k, v = qkv.transpose(2, 0, 3, 1, 4)
            scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(D // HEADS)
            allowed = (
                jnp.tril(jnp.ones((4, 4), bool))[None, None]
                & jnp.asarray(keep_mask)[:, None, None, :]
            )
            p = jax.nn.softmax(
                jnp.where(allowed, scores, jnp.finfo(scores.dtype).min), -1
            )
            a = (
                jnp.einsum("bhts,bhsd->bhtd", p, v)
                .transpose(0, 2, 1, 3)
                .reshape(2, 4, D)
            )
            r1 = x + target.attn_out(a)
            n2 = (r1 - jnp.mean(r1, -1, keepdims=True)) * jax.lax.rsqrt(
                jnp.mean((r1 - jnp.mean(r1, -1, keepdims=True)) ** 2, -1, keepdims=True)
                + EPS
            )
            h = jax.nn.gelu(target.mlp_in(n2), approximate=False)
            block = r1 + target.mlp_out(h)
            return jnp.sum(
                (block @ target.token_embedding.embedding[...].T)
                * jnp.asarray(cotangent)
            )

        jax_grad = np.asarray(
            jax.grad(jax_scalar)(jnp.asarray(source_input.detach().numpy()))
        )
        vjp = compare(torch_grad, jax_grad)
        if not vjp["passed"]:
            raise AssertionError(f"nonuniform input VJP mismatch: {vjp}")
        # Same shapes, wrong QKV/head interpretation and inverted pad-mask meaning must fail.
        bad_heads = target(
            jnp.asarray(ids),
            jnp.asarray(positions),
            jnp.asarray(keep_mask),
            wrong_head_layout=True,
        )
        head_report = compare(
            torch_outputs["logits"].detach().numpy(), bad_heads["logits"]
        )
        head_first = first_divergence(torch_outputs, bad_heads)
        assert (
            not head_report["passed"]
            and bad_heads["logits"].shape == flax_outputs["logits"].shape
        )
        bad_mask = target(
            jnp.asarray(ids),
            jnp.asarray(positions),
            jnp.asarray(keep_mask),
            wrong_mask=True,
        )
        mask_report = compare(
            torch_outputs["logits"].detach().numpy(), bad_mask["logits"]
        )
        mask_first = first_divergence(torch_outputs, bad_mask)
        assert (
            not mask_report["passed"]
            and bad_mask["logits"].shape == flax_outputs["logits"].shape
        )
        # Save NNX arrays and reload by a new Python process, not the live graph.
        np.savez(
            converted_path,
            qkv_kernel=np.asarray(target.qkv.kernel),
            qkv_bias=np.asarray(target.qkv.bias),
            attn_out_kernel=np.asarray(target.attn_out.kernel),
            attn_out_bias=np.asarray(target.attn_out.bias),
            mlp_in_kernel=np.asarray(target.mlp_in.kernel),
            mlp_in_bias=np.asarray(target.mlp_in.bias),
            mlp_out_kernel=np.asarray(target.mlp_out.kernel),
            mlp_out_bias=np.asarray(target.mlp_out.bias),
            token_embedding=np.asarray(target.token_embedding.embedding),
            position_embedding=np.asarray(target.position_embedding.embedding),
            processor_id=np.array(PROCESSOR_ID),
            schema=np.array(1),
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--reload",
                str(converted_path),
                str(input_path),
            ],
            check=True,
            text=True,
            capture_output=True,
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            timeout=60,
        )
        reloaded = json.loads(proc.stdout)
        np.testing.assert_allclose(
            reloaded["logits"], flax_outputs["logits"], atol=0, rtol=0
        )
        print("First divergent layer: none; all ordered boundaries pass.")
        print(
            "Source -> target tensor/layout map:",
            json.dumps(
                [
                    [
                        "qkv.weight",
                        "qkv.kernel",
                        "[3D,D] -> [D,3D], transpose; rows [Q_all,K_all,V_all]",
                    ],
                    ["qkv.bias", "qkv.bias", "[3D] -> [3D], unchanged"],
                    ["attn_out.weight", "attn_out.kernel", "[D,D] -> [D,D], transpose"],
                    ["mlp_in.weight", "mlp_in.kernel", "[MLP,D] -> [D,MLP], transpose"],
                    [
                        "mlp_out.weight",
                        "mlp_out.kernel",
                        "[D,MLP] -> [MLP,D], transpose",
                    ],
                    [
                        "token_embedding.weight",
                        "token_embedding.embedding",
                        "[V,D] unchanged; also tied output head",
                    ],
                    [
                        "position_embedding.weight",
                        "position_embedding.embedding",
                        "[P,D] unchanged",
                    ],
                ]
            ),
        )
        print(
            "Per-layer max absolute / relative-L2:",
            json.dumps({k: v for k, v in reports}),
        )
        print("Nonuniform VJP:", json.dumps(vjp))
        print("Same-shape QKV/head fault first divergence:", json.dumps(head_first))
        print("Same-shape mask fault first divergence:", json.dumps(mask_first))
        print(
            "Positions: explicit nonsequential position_ids; pad keys excluded; processor identity checked."
        )
        print(
            "Tied token/output embedding identity and fresh-process save/reload: PASS"
        )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--reload":
        child_reload(sys.argv[2], sys.argv[3])
    else:
        run()
