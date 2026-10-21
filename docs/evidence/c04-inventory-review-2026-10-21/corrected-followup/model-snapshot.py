"""Small temporal acoustic encoder used for mechanics and CPU pilot checks."""
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from config import BLANK_ID, N_MELS, PILOT_MODEL, TOKEN_TO_ID


def init_linear(key, in_dim, out_dim, scale=None):
    scale = scale or (1.0 / math.sqrt(in_dim))
    return {"w": jax.random.normal(key, (in_dim, out_dim)) * scale,
            "b": jnp.zeros((out_dim,), jnp.float32)}


def init_layer_norm(width):
    return {"scale": jnp.ones((width,), jnp.float32), "bias": jnp.zeros((width,), jnp.float32)}


def init_model(key, config=None, vocab_size=None):
    """Create an explicit parameter tree, including a separate acoustic decoder."""
    c = dict(PILOT_MODEL if config is None else config)
    vocab_size = vocab_size or len(TOKEN_TO_ID)
    keys = iter(jax.random.split(key, 3 + c["blocks"] * 6))
    front = []
    width_in = N_MELS
    for _ in range(2):
        width_out = c["frontend_channels"]
        front.append({"w": jax.random.normal(next(keys), (3, width_in, width_out)) / math.sqrt(3 * width_in),
                      "norm": init_layer_norm(width_out)})
        width_in = width_out
    blocks = []
    d, h = c["width"], c["heads"]
    for _ in range(c["blocks"]):
        blocks.append({
            "n1": init_layer_norm(d), "qkv": init_linear(next(keys), d, 3 * d), "attn_out": init_linear(next(keys), d, d),
            "n2": init_layer_norm(d), "ff1": init_linear(next(keys), d, c["mlp_width"]), "ff2": init_linear(next(keys), c["mlp_width"], d),
        })
    return {"frontend": front, "input_proj": init_linear(next(keys), c["frontend_channels"], d),
            "blocks": blocks, "final_norm": init_layer_norm(d),
            "ctc_head": init_linear(next(keys), d, vocab_size),
            "acoustic_decoder": init_linear(next(keys), d, N_MELS)}


def _norm(x, p, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * p["scale"] + p["bias"]


def conv_lengths(lengths, kernel=3, stride=2):
    """Valid counts for the fixed left/right padded stride-2 alignment."""
    return jnp.floor_divide(lengths + stride - 1, stride)


def _conv_explicit_symmetric(x, w, stride=2):
    """Kernel-3 output j is centered at input 2*j, independent of batch width."""
    if int(w.shape[0]) != 3 or stride != 2:
        raise ValueError("pilot frontend requires kernel=3 and stride=2")
    return jax.lax.conv_general_dilated(x, w, (stride,), ((1, 1),),
                                        dimension_numbers=("NWC", "WIO", "NWC"))


def _position_encoding(length, width, dtype):
    pos = jnp.arange(length, dtype=jnp.float32)[:, None]
    dim = jnp.arange(width, dtype=jnp.float32)[None, :]
    rate = jnp.exp(-math.log(10_000.0) * (2 * jnp.floor(dim / 2) / width))
    angle = pos * rate
    return jnp.where((dim.astype(jnp.int32) % 2) == 0, jnp.sin(angle), jnp.cos(angle)).astype(dtype)


def encode(params, features, frame_lengths, *, train=False, dropout_key=None, dropout_rate=None):
    """Encode temporal log-mel frames; returns encoded states and valid lengths."""
    c = PILOT_MODEL
    x = jnp.asarray(features, jnp.float32)
    lengths = jnp.asarray(frame_lengths, jnp.int32)
    if x.ndim != 3 or x.shape[-1] != N_MELS or lengths.shape != (x.shape[0],):
        raise ValueError("expected (batch,time,80) features and valid frame lengths")
    x = jnp.where(jnp.arange(x.shape[1])[None, :, None] < lengths[:, None, None], x, 0.0)
    for layer in params["frontend"]:
        x = _conv_explicit_symmetric(x, layer["w"])
        x = jax.nn.gelu(_norm(x, layer["norm"]))
        lengths = conv_lengths(lengths)
        x = jnp.where(jnp.arange(x.shape[1])[None, :, None] < lengths[:, None, None], x, 0.0)
    x = jnp.einsum("btd,df->btf", x, params["input_proj"]["w"]) + params["input_proj"]["b"]
    x = x + _position_encoding(x.shape[1], x.shape[2], x.dtype)[None, :, :]
    valid = jnp.arange(x.shape[1])[None, :] < lengths[:, None]
    x = jnp.where(valid[:, :, None], x, 0.0)
    rate = c["dropout"] if dropout_rate is None else dropout_rate
    keys = jax.random.split(dropout_key, len(params["blocks"]) * 2) if train else [None] * (len(params["blocks"]) * 2)
    for i, block in enumerate(params["blocks"]):
        x0 = _norm(x, block["n1"])
        qkv = jnp.einsum("btd,df->btf", x0, block["qkv"]["w"]) + block["qkv"]["b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        b, t, d = q.shape
        h = c["heads"]
        q = q.reshape(b, t, h, d // h).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, h, d // h).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, h, d // h).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(d // h)
        scores = jnp.where(valid[:, None, None, :], scores, -1e30)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhij,bhjd->bhid", weights, v).transpose(0, 2, 1, 3).reshape(b, t, d)
        attn = jnp.einsum("btd,df->btf", attended, block["attn_out"]["w"]) + block["attn_out"]["b"]
        if train and rate:
            keep = jax.random.bernoulli(keys[2 * i], 1.0 - rate, attn.shape)
            attn = jnp.where(keep, attn / (1.0 - rate), 0.0)
        x = jnp.where(valid[:, :, None], x + attn, 0.0)
        ff = jax.nn.gelu(_norm(x, block["n2"]) @ block["ff1"]["w"] + block["ff1"]["b"])
        ff = ff @ block["ff2"]["w"] + block["ff2"]["b"]
        if train and rate:
            keep = jax.random.bernoulli(keys[2 * i + 1], 1.0 - rate, ff.shape)
            ff = jnp.where(keep, ff / (1.0 - rate), 0.0)
        x = jnp.where(valid[:, :, None], x + ff, 0.0)
    encoded = _norm(x, params["final_norm"])
    encoded = jnp.where(valid[:, :, None], encoded, 0.0)
    return encoded, lengths


def forward(params, features, frame_lengths, *, train=False, dropout_key=None):
    encoded, lengths = encode(params, features, frame_lengths, train=train, dropout_key=dropout_key)
    logits = encoded @ params["ctc_head"]["w"] + params["ctc_head"]["b"]
    reconstruction = encoded @ params["acoustic_decoder"]["w"] + params["acoustic_decoder"]["b"]
    return {"logits": logits, "encoded": encoded, "reconstruction": reconstruction, "lengths": lengths}


def parameter_count(tree):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(tree))


def ctc_head_only(params):
    """Return a head-only tree for an optimizer with no accidental encoder leaves."""
    return {"ctc_head": params["ctc_head"]}
