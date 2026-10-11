#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_jax_to_pt.py -- convert an OFFICIAL EasyLM/JAX TTT checkpoint
(`streaming_train_state` / `streaming_params`, as released under
huggingface.co/Test-Time-Training) into an HF/PyTorch directory that
`ttt-lm-pytorch`'s `TTTForCausalLM.from_pretrained` can load directly.

    python3 convert_jax_to_pt.py IN_CKPT OUT_DIR [--verify] [--dtype float32]

WHY THIS EXISTS
---------------
The released TTT LM checkpoints are EasyLM streaming msgpack files holding a
*flax* parameter tree.  The reference PyTorch implementation (ttt.py) is a
faithful re-implementation with different module names and different array
layout conventions.  Neither repo ships a converter.  This file is that
converter, written by reading `ttt-lm-jax/ttt/models/{model.py,ttt_layer.py}`
and `ttt-lm-pytorch/ttt.py` side by side.

FILE FORMAT (EasyLM StreamingCheckpointer, ttt/infra/checkpoint.py:44)
---------------------------------------------------------------------
The file is a *concatenation* of msgpack objects, one per parameter:

    packer.pack((flat_key_tuple, flax.serialization.to_bytes(array)))

`to_bytes` is itself msgpack, with ndarrays encoded as ExtType(code=1) whose
payload is `msgpack.packb((shape, dtype.name, arr.tobytes('C')))`.  bfloat16
arrives as raw 2-byte little-endian values; we widen them to fp32 by placing
them in the high half of a uint32 (bf16 is literally the top 16 bits of fp32).

`flat_key_tuple` for a `streaming_train_state` starts with ('params','params',...)
for the model weights; ('step',) and the optimizer moments are also present and
are skipped.  A `streaming_params` file has the ('params',...) prefix only.

Some flax parameter *names* themselves contain a '/', because
`ttt_layer.get_multi_head_params` (ttt_layer.py:33) creates head-stacked params
with `self.param("ttt_norm/scale", ...)`.  We therefore join the whole key tuple
with '/' and parse the resulting string, which is correct either way.

MEMORY
------
The file is streamed one parameter at a time and the output is written as
*sharded* safetensors (HF `model.safetensors.index.json` format), flushing a
shard whenever it exceeds --max-shard-size.  Peak RSS is therefore about one
shard plus one parameter, not the whole model -- important because a 760M
`streaming_train_state` is ~4.4 GB on disk (params + Adam moments) and the fp32
weights alone are ~3 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import msgpack
except ImportError:  # pragma: no cover
    sys.exit("need msgpack:  python3 -m pip install msgpack")

try:
    from safetensors.numpy import save_file as _save_file
except ImportError:  # pragma: no cover
    sys.exit("need safetensors:  python3 -m pip install safetensors")


# --------------------------------------------------------------------------
# 1.  msgpack / flax decoding
# --------------------------------------------------------------------------

_EXT_NDARRAY = 1          # flax.serialization._MsgpackExtType.ndarray
_EXT_NPSCALAR = 3         # flax.serialization._MsgpackExtType.npscalar


def _bf16_bytes_to_f32(raw: bytes, shape) -> np.ndarray:
    """bfloat16 is the high 16 bits of a float32; widen without ml_dtypes."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).reshape(shape)


_NP_DTYPES = {
    "float32": "<f4", "float64": "<f8", "float16": "<f2",
    "int32": "<i4", "int64": "<i8", "uint32": "<u4", "uint8": "|u1",
    "int8": "|i1", "int16": "<i2", "uint16": "<u2", "bool": "|b1",
}


def _decode_array(blob: bytes) -> np.ndarray:
    """Decode one `flax.serialization.to_bytes(array)` payload to fp32/native numpy."""
    obj = msgpack.unpackb(blob, raw=False, strict_map_key=False)
    if isinstance(obj, msgpack.ExtType):
        if obj.code not in (_EXT_NDARRAY, _EXT_NPSCALAR):
            raise ValueError(f"unexpected msgpack ExtType code {obj.code}")
        shape, dtype_name, raw = msgpack.unpackb(obj.data, raw=False, strict_map_key=False)
        shape = tuple(int(s) for s in shape)
        if dtype_name == "bfloat16":
            return _bf16_bytes_to_f32(raw, shape)
        if dtype_name not in _NP_DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r}")
        return np.frombuffer(raw, dtype=_NP_DTYPES[dtype_name]).reshape(shape)
    # plain python scalar (e.g. `step`)
    return np.asarray(obj)


def iter_jax_params(path, prefixes=(("params", "params"), ("params",), ())):
    """Yield (joined_key, np.ndarray) for every model parameter in the file.

    Streams: only one parameter is held at a time.  `prefixes` is tried in
    order; the first prefix that any key in the file matches wins, so this
    works for both `streaming_train_state` (params/params/...) and
    `streaming_params` (params/... or bare).
    """
    chosen = None
    with open(path, "rb") as fin:
        unpacker = msgpack.Unpacker(fin, raw=False, read_size=8 << 20,
                                    max_buffer_size=0, strict_map_key=False)
        for item in unpacker:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise ValueError(
                    f"{path} is not an EasyLM streaming checkpoint "
                    f"(top-level item is {type(item).__name__}, expected a (key, blob) pair)"
                )
            key, blob = item
            key = tuple(str(k) for k in key)
            if chosen is None:
                for p in prefixes:
                    if key[: len(p)] == p and len(key) > len(p):
                        # require it to look like the model tree, not the optimizer
                        rest = key[len(p):]
                        if rest[0] in ("model", "lm_head"):
                            chosen = p
                            break
                if chosen is None:
                    continue
            if key[: len(chosen)] != chosen:
                continue
            rest = key[len(chosen):]
            if not rest or rest[0] not in ("model", "lm_head"):
                continue      # optimizer moments / step / anything else
            yield "/".join(rest), _decode_array(blob)
    if chosen is None:
        raise ValueError(
            f"found no model parameters in {path}; expected keys under "
            f"params/params/model/... (streaming_train_state) or params/model/..."
        )


def iter_raw_headers(path):
    """Yield (key_tuple, shape, dtype_name, payload_bytes) for EVERY record in
    the file, model weights and optimizer state alike, without materialising
    the arrays.  Used by --list-keys, which is the diagnostic to run first if
    the conversion complains about a checkpoint this converter has not seen."""
    with open(path, "rb") as fin:
        unpacker = msgpack.Unpacker(fin, raw=False, read_size=8 << 20,
                                    max_buffer_size=0, strict_map_key=False)
        for key, blob in unpacker:
            key = tuple(str(k) for k in key)
            obj = msgpack.unpackb(blob, raw=False, strict_map_key=False)
            if isinstance(obj, msgpack.ExtType):
                shape, dtype_name, raw = msgpack.unpackb(obj.data, raw=False,
                                                         strict_map_key=False)
                yield key, tuple(int(x) for x in shape), dtype_name, len(raw)
            else:
                yield key, (), type(obj).__name__, 0


def list_keys(path, model_only=False):
    total = 0
    print(f"{'key':<70s} {'shape':>22s} {'dtype':>10s} {'bytes':>12s}  mapped-to")
    for key, shape, dt, nb in iter_raw_headers(path):
        joined = "/".join(key)
        hit = None
        for pre in ("params/params/", "params/"):
            if joined.startswith(pre):
                hit = map_key(joined[len(pre):])
                if hit:
                    break
        if model_only and hit is None:
            continue
        total += nb
        print(f"{joined:<70s} {str(shape):>22s} {dt:>10s} {nb:>12d}  "
              f"{hit[0] + '  [' + (hit[1] or 'copy') + ']' if hit else '-'}")
    print(f"\ntotal payload {total / 1e9:.3f} GB")


# --------------------------------------------------------------------------
# 2.  the name / layout map
# --------------------------------------------------------------------------
#
# transform codes
#   ""      : copy as-is
#   "T"     : flax Dense kernel [in, out]  ->  torch Linear weight [out, in]
#   "CONV"  : flax Conv kernel [width, in/groups=1, C] -> torch Conv1d [C, 1, width]
#   "LR"    : head-stacked Dense kernel [nh, width, 1] -> torch [nh, 1, width]

_LAYER_RE = re.compile(r"^model/h/(\d+)/(.*)$")

_BLOCK_MAP = {
    "seq_norm/kernel":            ("seq_norm.weight", ""),
    "ffn_norm/kernel":            ("ffn_norm.weight", ""),
    # SwiGLU: jax  w2(silu(w1 x) * w3 x)  ==  torch down(act(gate x) * up x)
    "feed_forward/w1/kernel":     ("mlp.gate_proj.weight", "T"),
    "feed_forward/w2/kernel":     ("mlp.down_proj.weight", "T"),
    "feed_forward/w3/kernel":     ("mlp.up_proj.weight", "T"),
    # pre_conv (Mamba-style) block, model.py:379 ConvModule
    "conv/conv_norm/kernel":      ("conv.norm.weight", ""),
    "conv/conv1/kernel":          ("conv.conv.weight", "CONV"),
    "conv/conv1/bias":            ("conv.conv.bias", ""),
}

_TTT_MAP = {
    "wq/kernel":                    ("q_proj.weight", "T"),
    "wk/kernel":                    ("k_proj.weight", "T"),   # only when share_qk=False
    "wv/kernel":                    ("v_proj.weight", "T"),
    "wo/kernel":                    ("o_proj.weight", "T"),
    "wg/kernel":                    ("g_proj.weight", "T"),   # use_gate=True
    "conv_q/kernel":                ("conv_q.weight", "CONV"),
    "conv_q/bias":                  ("conv_q.bias", ""),
    "conv_k/kernel":                ("conv_k.weight", "CONV"),
    "conv_k/bias":                  ("conv_k.bias", ""),
    "ttt_dense_0":                  ("W1", ""),
    "ttt_bias_0":                   ("b1", ""),
    "ttt_dense_1":                  ("W2", ""),
    "ttt_bias_1":                   ("b2", ""),
    "ttt_norm/scale":               ("ttt_norm_weight", ""),
    "ttt_norm/bias":                ("ttt_norm_bias", ""),
    "post_norm/scale":              ("post_norm.weight", ""),
    "post_norm/bias":               ("post_norm.bias", ""),
    "learnable_ttt_lr/kernel":      ("learnable_ttt_lr_weight", "LR"),
    "learnable_ttt_lr/bias":        ("learnable_ttt_lr_bias", ""),
    "learnable_token_idx":          ("learnable_token_idx", ""),
}

_TOP_MAP = {
    "model/wte/embedding": ("model.embed_tokens.weight", ""),
    "model/ln_f/kernel":   ("model.norm.weight", ""),
    "lm_head/kernel":      ("lm_head.weight", "T"),
}


def map_key(joined: str):
    """flax flat key -> (torch state_dict key, transform code).  None if unknown."""
    if joined in _TOP_MAP:
        return _TOP_MAP[joined]
    m = _LAYER_RE.match(joined)
    if not m:
        return None
    idx, rest = m.group(1), m.group(2)
    if rest in _BLOCK_MAP:
        suffix, tr = _BLOCK_MAP[rest]
        return f"model.layers.{idx}.{suffix}", tr
    if rest.startswith("seq_modeling_block/"):
        sub = rest[len("seq_modeling_block/"):]
        if sub in _TTT_MAP:
            suffix, tr = _TTT_MAP[sub]
            return f"model.layers.{idx}.seq_modeling_block.{suffix}", tr
    return None


def apply_transform(arr: np.ndarray, code: str, key: str) -> np.ndarray:
    if code == "":
        return arr
    if code == "T":
        if arr.ndim != 2:
            raise ValueError(f"{key}: expected 2-D Dense kernel, got {arr.shape}")
        return np.ascontiguousarray(arr.T)
    if code == "CONV":
        # flax [width, in/groups=1, C]  ->  torch [C, 1, width]
        if arr.ndim != 3:
            raise ValueError(f"{key}: expected 3-D conv kernel, got {arr.shape}")
        return np.ascontiguousarray(arr.transpose(2, 1, 0))
    if code == "LR":
        # head-stacked Dense kernel [nh, width, 1] -> torch [nh, 1, width]
        if arr.ndim != 3:
            raise ValueError(f"{key}: expected 3-D lr-gate kernel, got {arr.shape}")
        return np.ascontiguousarray(arr.transpose(0, 2, 1))
    raise ValueError(f"unknown transform {code!r}")


# --------------------------------------------------------------------------
# 3.  config inference
# --------------------------------------------------------------------------

def infer_config(shapes, args):
    """shapes: {torch_key: tuple} for every converted tensor."""
    def need(k):
        if k not in shapes:
            raise SystemExit(f"checkpoint is missing {k!r}; cannot infer the config")
        return shapes[k]

    vocab_size, hidden_size = need("model.embed_tokens.weight")
    n_layers = 1 + max(
        int(k.split(".")[2]) for k in shapes if k.startswith("model.layers.")
    )
    inter = need("model.layers.0.mlp.gate_proj.weight")[0]

    w1 = "model.layers.0.seq_modeling_block.W1"
    if w1 not in shapes:
        raise SystemExit(
            "no seq_modeling_block/ttt_dense_0 in this checkpoint -- it is a "
            "self-attention (Transformer) baseline, not a TTT model; "
            "ttt-lm-pytorch cannot represent it."
        )
    n_heads, head_dim, w1_out = shapes[w1]
    if head_dim * n_heads != hidden_size:
        raise SystemExit(
            f"inconsistent shapes: W1 says {n_heads} heads x {head_dim} dim "
            f"= {n_heads * head_dim}, embedding says hidden_size={hidden_size}"
        )
    has_w2 = "model.layers.0.seq_modeling_block.W2" in shapes
    ttt_layer_type = "mlp" if has_w2 else "linear"
    if ttt_layer_type == "linear" and w1_out != head_dim:
        raise SystemExit(f"TTT-Linear W1 should be square, got {shapes[w1]}")
    if ttt_layer_type == "mlp" and w1_out != 4 * head_dim:
        raise SystemExit(f"TTT-MLP W1 should be [nh, f, 4f], got {shapes[w1]}")

    share_qk = "model.layers.0.seq_modeling_block.conv_q.weight" in shapes
    if share_qk and "model.layers.0.seq_modeling_block.k_proj.weight" in shapes:
        raise SystemExit("checkpoint has BOTH conv_q/conv_k and wk; cannot infer share_qk")
    if not share_qk and "model.layers.0.seq_modeling_block.k_proj.weight" not in shapes:
        raise SystemExit("checkpoint has neither conv_q/conv_k nor wk")
    use_gate = "model.layers.0.seq_modeling_block.g_proj.weight" in shapes
    pre_conv = "model.layers.0.conv.conv.weight" in shapes

    conv_kernel = args.conv_kernel
    if share_qk:
        conv_kernel = shapes["model.layers.0.seq_modeling_block.conv_q.weight"][2]
    elif pre_conv:
        conv_kernel = shapes["model.layers.0.conv.conv.weight"][2]

    mini_batch_size = shapes["model.layers.0.seq_modeling_block.learnable_token_idx"][0]

    tie = "lm_head.weight" not in shapes
    if not tie and args.tie_word_embeddings is not None:
        tie = args.tie_word_embeddings

    cfg = {
        "architectures": ["TTTForCausalLM"],
        "model_type": "ttt",
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(inter),
        "num_hidden_layers": int(n_layers),
        "num_attention_heads": int(n_heads),
        "hidden_act": "silu",
        "max_position_embeddings": int(args.max_position_embeddings),
        "initializer_range": 0.02,
        "rms_norm_eps": float(args.rms_norm_eps),
        "use_cache": False,
        "pad_token_id": None,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pretraining_tp": 1,
        "tie_word_embeddings": bool(tie),
        "rope_theta": float(args.rope_theta),
        "use_gate": bool(use_gate),
        "share_qk": bool(share_qk),
        "ttt_layer_type": ttt_layer_type,
        "ttt_base_lr": float(args.ttt_base_lr),
        "mini_batch_size": int(mini_batch_size),
        "pre_conv": bool(pre_conv),
        "conv_kernel": int(conv_kernel),
        "scan_checkpoint_group_size": 0,
        "torch_dtype": args.dtype,
    }
    return cfg


def expected_keys(cfg):
    """The exact torch state_dict key set implied by `cfg`, derived from
    ttt-lm-pytorch/ttt.py (TTTForCausalLM/TTTModel/Block/TTTBase)."""
    ks = {"model.embed_tokens.weight", "model.norm.weight"}
    if not cfg["tie_word_embeddings"]:
        ks.add("lm_head.weight")
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        ks |= {f"{p}.seq_norm.weight", f"{p}.ffn_norm.weight",
               f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight",
               f"{p}.mlp.down_proj.weight"}
        if cfg["pre_conv"]:
            ks |= {f"{p}.conv.norm.weight", f"{p}.conv.conv.weight", f"{p}.conv.conv.bias"}
        s = f"{p}.seq_modeling_block"
        ks |= {f"{s}.q_proj.weight", f"{s}.v_proj.weight", f"{s}.o_proj.weight",
               f"{s}.learnable_token_idx", f"{s}.learnable_ttt_lr_weight",
               f"{s}.learnable_ttt_lr_bias", f"{s}.ttt_norm_weight",
               f"{s}.ttt_norm_bias", f"{s}.post_norm.weight", f"{s}.post_norm.bias",
               f"{s}.W1", f"{s}.b1"}
        if cfg["share_qk"]:
            ks |= {f"{s}.conv_q.weight", f"{s}.conv_q.bias",
                   f"{s}.conv_k.weight", f"{s}.conv_k.bias"}
        else:
            ks.add(f"{s}.k_proj.weight")
        if cfg["use_gate"]:
            ks.add(f"{s}.g_proj.weight")
        if cfg["ttt_layer_type"] == "mlp":
            ks |= {f"{s}.W2", f"{s}.b2"}
    return ks


# --------------------------------------------------------------------------
# 4.  sharded writer
# --------------------------------------------------------------------------

class ShardWriter:
    """Accumulate tensors and write safetensors shards, HF index format.

    Each shard is written to disk (under a temporary name) AS SOON AS it fills
    up and is then dropped from memory, so peak RSS is one shard plus the
    tensor currently being converted -- not the whole model.  The final HF
    names embed the shard count, which is only known at the end, so the files
    are renamed in `finalize`.
    """

    def __init__(self, out_dir, max_bytes, dtype):
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.dtype = dtype
        self.buf, self.buf_bytes = {}, 0
        self.written = []          # [(tmp_path, [keys])]
        self.weight_map = {}
        self.total_bytes = 0
        os.makedirs(out_dir, exist_ok=True)
        # remove weights from any earlier run: a leftover shard from a run with a
        # different shard count would silently shadow the new ones
        for f in os.listdir(out_dir):
            if (f.startswith(".shard-") and f.endswith(".tmp")) or \
               (f.startswith("model") and f.endswith(".safetensors")):
                os.remove(os.path.join(out_dir, f))

    def cleanup(self):
        for tmp, _ in self.written:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def add(self, key, arr):
        arr = np.ascontiguousarray(arr.astype(self.dtype, copy=False))
        self.buf[key] = arr
        self.buf_bytes += arr.nbytes
        self.total_bytes += arr.nbytes
        if self.buf_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        tmp = os.path.join(self.out_dir, f".shard-{len(self.written):05d}.tmp")
        _save_file(self.buf, tmp, metadata={"format": "pt"})
        self.written.append((tmp, list(self.buf)))
        self.buf, self.buf_bytes = {}, 0      # free the shard

    def finalize(self):
        self.flush()
        n = len(self.written)
        if n == 0:
            raise SystemExit("nothing to write")
        names = (["model.safetensors"] if n == 1 else
                 [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)])
        for (tmp, keys), name in zip(self.written, names):
            os.replace(tmp, os.path.join(self.out_dir, name))
            for k in keys:
                self.weight_map[k] = name
        idx = os.path.join(self.out_dir, "model.safetensors.index.json")
        if n > 1:
            with open(idx, "w") as f:
                json.dump({"metadata": {"total_size": self.total_bytes},
                           "weight_map": self.weight_map}, f, indent=1)
        elif os.path.exists(idx):
            os.remove(idx)        # stale index from an earlier sharded run
        return names


# --------------------------------------------------------------------------
# 5.  driver
# --------------------------------------------------------------------------

def convert(args):
    dtype = {"float32": np.float32, "float16": np.float16}.get(args.dtype)
    if dtype is None:
        raise SystemExit("--dtype must be float32 or float16 (bf16 output is not "
                         "useful here: the probe runs in fp32/fp64)")

    writer = ShardWriter(args.out_dir, int(args.max_shard_size * 1e9), args.dtype)
    try:
        return _convert(args, writer, dtype)
    except BaseException:
        writer.cleanup()          # do not leave half-written shards behind
        raise


def _convert(args, writer, dtype):
    shapes, unknown, n = {}, [], 0
    lm_head = None

    for joined, arr in iter_jax_params(args.checkpoint):
        hit = map_key(joined)
        if hit is None:
            unknown.append(joined)
            continue
        tkey, code = hit
        out = apply_transform(arr, code, joined)
        shapes[tkey] = tuple(int(s) for s in out.shape)
        n += 1
        if tkey == "lm_head.weight":
            lm_head = out.astype(dtype, copy=False)     # decide tying later
            continue
        writer.add(tkey, out.astype(dtype, copy=False))
        if args.verbose:
            print(f"  {joined:<60s} {str(arr.shape):>22s} -> {tkey:<58s} {out.shape}")

    if unknown:
        print(f"[warn] {len(unknown)} unmapped parameter(s) were skipped:", file=sys.stderr)
        for u in sorted(set(unknown))[:20]:
            print(f"       {u}", file=sys.stderr)

    cfg = infer_config(shapes, args)

    if lm_head is not None and not cfg["tie_word_embeddings"]:
        writer.add("lm_head.weight", lm_head)
    elif lm_head is not None:
        # checkpoint carried an untied head but the user asked to tie: drop it
        shapes.pop("lm_head.weight", None)
        cfg = infer_config(shapes, args)

    want = expected_keys(cfg)
    got = set(shapes) - ({"lm_head.weight"} if cfg["tie_word_embeddings"] else set())
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        msg = ["converted state dict does not match the TTTForCausalLM key set:"]
        if missing:
            msg.append(f"  missing ({len(missing)}): " + ", ".join(missing[:12]))
        if extra:
            msg.append(f"  unexpected ({len(extra)}): " + ", ".join(extra[:12]))
        raise SystemExit("\n".join(msg))

    names = writer.finalize()
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    # a generation_config keeps transformers quiet on some versions
    with open(os.path.join(args.out_dir, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2, "_from_model_config": True}, f, indent=1)

    print(f"[convert] {n} parameters -> {len(names)} shard(s) "
          f"({writer.total_bytes / 1e9:.2f} GB {args.dtype}) in {args.out_dir}")
    keys = ("ttt_layer_type", "share_qk", "use_gate", "pre_conv", "conv_kernel",
            "mini_batch_size", "num_hidden_layers", "hidden_size",
            "num_attention_heads", "intermediate_size", "vocab_size",
            "tie_word_embeddings")
    print("[config]  " + json.dumps({k: cfg[k] for k in keys}))
    return cfg


def verify(args, cfg):
    """Load the converted directory with the real ttt.py and run a short forward."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    repos = os.environ.get("TTCL_REPOS")
    if repos:
        sys.path.append(os.path.join(repos, "ttt-lm-pytorch"))
    try:
        import ttt as ttt_mod
    except ImportError:
        raise SystemExit(
            "--verify needs ttt.py (the ttt-lm-pytorch model definition).  It is "
            "vendored inside validation_kit/; if you are running this file from "
            "somewhere else, set TTCL_REPOS to the directory containing your "
            "ttt-lm-pytorch clone."
        )

    print("[verify] loading with TTTForCausalLM.from_pretrained ...", flush=True)
    model = ttt_mod.TTTForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.float32)
    model.eval()
    L = max(2 * int(model.config.mini_batch_size), 32)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, int(model.config.vocab_size), (1, L), generator=g)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True)
    if model.config.tie_word_embeddings:
        a = model.lm_head.weight.data_ptr()
        b = model.model.embed_tokens.weight.data_ptr()
        if a != b:
            raise SystemExit(
                "[verify] FAILED: config says tie_word_embeddings but transformers "
                "did not tie lm_head to the embedding -- the head would be RANDOM. "
                "Re-run the conversion with --tie-word-embeddings false, or check "
                "the transformers version (4.4x is what this kit expects)."
            )
        print("[verify] lm_head is tied to model.embed_tokens (shared storage)")
    lg = out.logits
    ok = bool(torch.isfinite(lg).all())
    print(f"[verify] logits {tuple(lg.shape)}  finite={ok}  "
          f"mean={lg.mean().item():+.4f} std={lg.std().item():.4f} "
          f"absmax={lg.abs().max().item():.3f}")
    if not ok:
        raise SystemExit("[verify] FAILED: non-finite logits")
    # a real checkpoint should be far better than uniform on its own samples
    with torch.no_grad():
        nll = torch.nn.functional.cross_entropy(
            lg[0, :-1].float(), ids[0, 1:]).item()
    import math
    print(f"[verify] ppl on random ids = {math.exp(nll):.1f} "
          f"(uniform baseline = {model.config.vocab_size})")
    print("[verify] OK")
    return model


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="path to streaming_train_state / streaming_params")
    p.add_argument("out_dir", nargs="?", default=None,
                   help="output HF model directory (omit with --list-keys)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--max-shard-size", type=float, default=1.0,
                   help="GB per safetensors shard (default 1.0).  Shards are "
                        "written and freed as they fill, so this is roughly the "
                        "converter's peak memory; lower it on a small machine.")
    p.add_argument("--rope-theta", type=float, default=10000.0,
                   help="not stored in the checkpoint; 10000 for every released TTT config")
    p.add_argument("--ttt-base-lr", type=float, default=1.0,
                   help="not stored in the checkpoint; 1.0 for every released TTT config")
    p.add_argument("--rms-norm-eps", type=float, default=1e-6)
    p.add_argument("--max-position-embeddings", type=int, default=2048,
                   help="inert in ttt.py (RoPE is per mini-batch); metadata only")
    p.add_argument("--conv-kernel", type=int, default=4)
    p.add_argument("--tie-word-embeddings", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=None, help="override; default: tie iff no lm_head/kernel in file")
    p.add_argument("--list-keys", action="store_true",
                   help="DIAGNOSTIC: dump every record in the checkpoint with its "
                        "shape, dtype and the torch name it maps to, then exit")
    p.add_argument("--list-model-keys", action="store_true",
                   help="like --list-keys but only the parameters that map")
    p.add_argument("--verify", action="store_true",
                   help="load the result with ttt.py and run a forward pass")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"no such file: {args.checkpoint}")
    if args.list_keys or args.list_model_keys:
        list_keys(args.checkpoint, model_only=args.list_model_keys)
        return
    if not args.out_dir:
        raise SystemExit("out_dir is required (or use --list-keys)")
    cfg = convert(args)
    if args.verify:
        verify(args, cfg)


if __name__ == "__main__":
    main()
