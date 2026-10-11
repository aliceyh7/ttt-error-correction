#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic_jax_ckpt.py -- build a *byte-format-accurate* fake EasyLM/JAX TTT
checkpoint (random weights) so the converter can be tested without downloading
the real 4.4 GB files.

The writer reimplements `flax.serialization.to_bytes` + EasyLM's
`StreamingCheckpointer.save_train_state_to_file` (ttt/infra/checkpoint.py:44):

    for key, value in flatten_dict(train_state).items():
        fout.write(packer.pack((key, to_bytes(value))))

with `to_bytes(arr) = msgpack.packb(ExtType(1, packb((shape, dtype.name, arr.tobytes('C')))))`.

Parameter names/shapes are taken from the flax module tree:
    ttt_layer.py:33  get_multi_head_params  -> head-stacked params whose *name*
                     contains a '/', e.g. "ttt_norm/scale", "learnable_ttt_lr/kernel"
    ttt_layer.py:139 TTTBase.setup, :354 TTTLinearBase.setup, :425 TTTMLPBase.setup
    ttt_layer.py:398 TTTLinear.setup_qkvo   -> wq + conv_q + conv_k (share_qk), wg
    model.py:379     ConvModule             -> conv/conv1, conv/conv_norm  (pre_conv)
    model.py:610     Block                  -> seq_norm, ffn_norm, feed_forward/w{1,2,3}
    model.py:809     Model                  -> model/wte/embedding, model/ln_f/kernel

Optimizer moments and `step` are also emitted for `--kind train_state`, so the
converter's "skip everything that is not params/params/model" path is exercised.
"""

from __future__ import annotations

import argparse
import os

import msgpack
import numpy as np

EXT_NDARRAY = 1


def _to_bytes(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "bfloat16":
        u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
        raw = (u32 >> np.uint32(16)).astype("<u2").tobytes()   # truncate, like jnp.bfloat16
        name = "bfloat16"
    else:
        arr = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
        raw, name = arr.tobytes("C"), arr.dtype.name
    inner = msgpack.packb((list(arr.shape), name, raw), use_bin_type=True)
    return msgpack.packb(msgpack.ExtType(EXT_NDARRAY, inner), use_bin_type=True)


def bf16_round_trip(arr: np.ndarray) -> np.ndarray:
    """The exact fp32 values a bf16-stored array will have after conversion."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return ((u32 >> np.uint32(16)).astype(np.uint32) << np.uint32(16)).view(np.float32)


def default_cfg(**over):
    cfg = dict(
        vocab_size=97, hidden_size=64, num_heads=4, intermediate_size=128,
        num_hidden_layers=2, mini_batch_size=8, conv_kernel=4,
        ttt_layer_type="linear", share_qk=True, use_gate=True, pre_conv=True,
        tie_word_embeddings=True, rope_theta=10000.0, ttt_base_lr=1.0,
        rms_norm_eps=1e-6,
    )
    cfg.update(over)
    return cfg


def build_params(cfg, seed=0):
    """Return {flat_flax_key: np.float32 array} in NATIVE FLAX LAYOUT."""
    rng = np.random.default_rng(seed)
    C, nh = cfg["hidden_size"], cfg["num_heads"]
    f = C // nh
    I, K, W = cfg["intermediate_size"], cfg["mini_batch_size"], cfg["conv_kernel"]
    n = lambda *s: rng.normal(0, 0.02, size=s).astype(np.float32)

    p = {
        "model/wte/embedding": n(cfg["vocab_size"], C),
        "model/ln_f/kernel": (1.0 + 0.1 * rng.normal(size=C)).astype(np.float32),
    }
    if not cfg["tie_word_embeddings"]:
        p["lm_head/kernel"] = n(C, cfg["vocab_size"])

    for i in range(cfg["num_hidden_layers"]):
        pre = f"model/h/{i}/"
        p[pre + "seq_norm/kernel"] = (1.0 + 0.1 * rng.normal(size=C)).astype(np.float32)
        p[pre + "ffn_norm/kernel"] = (1.0 + 0.1 * rng.normal(size=C)).astype(np.float32)
        p[pre + "feed_forward/w1/kernel"] = n(C, I)
        p[pre + "feed_forward/w2/kernel"] = n(I, C)
        p[pre + "feed_forward/w3/kernel"] = n(C, I)
        if cfg["pre_conv"]:
            p[pre + "conv/conv_norm/kernel"] = (1.0 + 0.1 * rng.normal(size=C)).astype(np.float32)
            p[pre + "conv/conv1/kernel"] = n(W, 1, C)          # flax [width, in/g, out]
            p[pre + "conv/conv1/bias"] = n(C)
        s = pre + "seq_modeling_block/"
        p[s + "wq/kernel"] = n(C, C)
        p[s + "wv/kernel"] = n(C, C)
        p[s + "wo/kernel"] = n(C, C)
        if cfg["share_qk"]:
            p[s + "conv_q/kernel"] = n(W, 1, C)
            p[s + "conv_q/bias"] = n(C)
            p[s + "conv_k/kernel"] = n(W, 1, C)
            p[s + "conv_k/bias"] = n(C)
        else:
            p[s + "wk/kernel"] = n(C, C)
        if cfg["use_gate"]:
            p[s + "wg/kernel"] = n(C, C)
        p[s + "ttt_norm/scale"] = (1.0 + 0.3 * rng.normal(size=(nh, f))).astype(np.float32)
        p[s + "ttt_norm/bias"] = (0.05 * rng.normal(size=(nh, f))).astype(np.float32)
        p[s + "post_norm/scale"] = (1.0 + 0.1 * rng.normal(size=C)).astype(np.float32)
        p[s + "post_norm/bias"] = (0.05 * rng.normal(size=C)).astype(np.float32)
        p[s + "learnable_ttt_lr/kernel"] = n(nh, C, 1)         # flax [nh, in, out=1]
        p[s + "learnable_ttt_lr/bias"] = n(nh, 1)
        p[s + "learnable_token_idx"] = (0.1 * rng.normal(size=K)).astype(np.float32)
        if cfg["ttt_layer_type"] == "linear":
            p[s + "ttt_dense_0"] = n(nh, f, f)
            p[s + "ttt_bias_0"] = n(nh, 1, f)
        else:
            p[s + "ttt_dense_0"] = n(nh, f, 4 * f)
            p[s + "ttt_bias_0"] = n(nh, 1, 4 * f)
            p[s + "ttt_dense_1"] = n(nh, 4 * f, f)
            p[s + "ttt_bias_1"] = n(nh, 1, f)
    return p


def write_ckpt(path, params, cfg, dtype="bfloat16", kind="train_state"):
    """Write the EasyLM streaming msgpack file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    packer = msgpack.Packer(use_bin_type=True)
    prefix = ("params", "params") if kind == "train_state" else ("params",)
    with open(path, "wb") as fout:
        if kind == "train_state":
            fout.write(packer.pack((["step"], _to_bytes(np.array(1234, np.int32), "int32"))))
        for k, v in params.items():
            # NOTE: flax params whose *name* contains '/' are single dict keys
            # (ttt_layer.py:33).  Rebuild that here so the file matches what
            # flatten_dict really emits for the released checkpoints.
            key = list(prefix) + _split_flax_key(k)
            fout.write(packer.pack((key, _to_bytes(v, dtype))))
        if kind == "train_state":
            # a couple of Adam moments, to exercise the "skip non-model" path
            for k, v in list(params.items())[:3]:
                for mom in ("mu", "nu"):
                    key = ["opt_state", "1", "0", mom, "params", "params"] + _split_flax_key(k)
                    fout.write(packer.pack((key, _to_bytes(np.zeros_like(v), dtype))))
    return path


# the flax param names that are a single dict key despite containing '/'
_FUSED = ("ttt_norm/scale", "ttt_norm/bias",
          "learnable_ttt_lr/kernel", "learnable_ttt_lr/bias")


def _split_flax_key(k):
    for fused in _FUSED:
        if k.endswith("/" + fused):
            head = k[: -len(fused) - 1]
            return head.split("/") + [fused]
    return k.split("/")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ttt-layer-type", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--no-share-qk", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-pre-conv", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--kind", default="train_state", choices=["train_state", "params"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = default_cfg(ttt_layer_type=a.ttt_layer_type,
                      share_qk=not a.no_share_qk,
                      use_gate=not a.no_gate,
                      pre_conv=not a.no_pre_conv)
    p = build_params(cfg, a.seed)
    write_ckpt(a.out, p, cfg, a.dtype, a.kind)
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes, {len(p)} params, {cfg})")


if __name__ == "__main__":
    main()
