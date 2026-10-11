#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_conversion.py -- numerical equivalence test for convert_jax_to_pt.py.

WHAT IS TESTED
--------------
For each of several architecture variants:

  1. a synthetic checkpoint is written in the *exact* EasyLM streaming-msgpack
     byte format (make_synthetic_jax_ckpt.py), bf16 payloads included;
  2. it is converted with convert_jax_to_pt.py;
  3. the result is loaded by the AUTHORS' UNMODIFIED ttt.py via
     TTTForCausalLM.from_pretrained;
  4. its float32 forward pass is compared, tensor by tensor and end to end,
     against jax_ref_numpy.py -- a float64 NumPy transcription of the flax
     model run on the *native, unconverted* flax parameter tree.

Because step 4 consumes the flax-layout arrays directly, ANY error in the
name map, in a Dense transpose, in the depthwise-conv layout, in the
head-stacked LR-gate transpose or in the W1/b1 head reshape shows up as a
large relative error.  Every stage is checked, not just the logits:

    embedding -> [pre_conv] -> seq_norm -> {q,k,v} + conv + RoPE + eta ->
    TTT inner loop -> post_norm -> gate -> o_proj -> residual ->
    ffn_norm -> SwiGLU -> residual -> ... -> final norm -> logits

Run:  python3 test_conversion.py [--repos DIR_CONTAINING_ttt-lm-pytorch]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert_jax_to_pt as conv                     # noqa: E402
import jax_ref_numpy as ref                          # noqa: E402
import make_synthetic_jax_ckpt as mk                 # noqa: E402


def import_ttt(repos=None):
    """Prefer a vendored ttt.py next to this file; else $TTCL_REPOS/ttt-lm-pytorch."""
    if os.path.exists(os.path.join(HERE, "ttt.py")):
        sys.path.insert(0, HERE)
    else:
        repos = repos or os.environ.get("TTCL_REPOS", "")
        d = os.path.join(repos, "ttt-lm-pytorch")
        if not os.path.exists(os.path.join(d, "ttt.py")):
            sys.exit(f"cannot find ttt.py (vendored or in {d})")
        sys.path.insert(0, d)
    import ttt as m
    return m


def relerr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


# --------------------------------------------------------------------------
def run_variant(name, cfg_over, ttt_mod, tmp, seq_len=24, batch=2, tol=3e-5,
                dtype="bfloat16", kind="train_state", verbose=False):
    cfg = mk.default_cfg(**cfg_over)
    ck = os.path.join(tmp, f"{name}.msgpack")
    out = os.path.join(tmp, f"{name}_hf")

    params = mk.build_params(cfg, seed=7)
    mk.write_ckpt(ck, params, cfg, dtype=dtype, kind=kind)

    # the reference must see exactly the values that survive the bf16 round trip
    if dtype == "bfloat16":
        params = {k: mk.bf16_round_trip(v) for k, v in params.items()}

    argv = [ck, out, "--dtype", "float32"]
    if verbose:
        argv.append("--verbose")
    conv.main(argv)

    import json
    with open(os.path.join(out, "config.json")) as f:
        hf_cfg = json.load(f)

    # ---- the config the converter inferred must match the truth -----------
    checks = {
        "vocab_size": cfg["vocab_size"], "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_heads"],
        "mini_batch_size": cfg["mini_batch_size"], "conv_kernel": cfg["conv_kernel"],
        "ttt_layer_type": cfg["ttt_layer_type"], "share_qk": cfg["share_qk"],
        "use_gate": cfg["use_gate"], "pre_conv": cfg["pre_conv"],
        "tie_word_embeddings": cfg["tie_word_embeddings"],
    }
    for k, v in checks.items():
        assert hf_cfg[k] == v, f"[{name}] config.{k} = {hf_cfg[k]!r}, expected {v!r}"

    # ---- load with the authors' unmodified ttt.py -------------------------
    model = ttt_mod.TTTForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg["vocab_size"], size=(batch, seq_len)).astype(np.int64)

    # ---- reference (float64 numpy on the native flax tree) ----------------
    rcfg = dict(cfg)
    rcfg["ttt_base_lr"] = 1.0
    ref_logits, ref_hidden = ref.forward(params, ids, rcfg)

    # ---- torch, with per-stage taps --------------------------------------
    taps = {}
    hooks = []

    def tap(nm):
        def fn(_m, _i, o):
            taps[nm] = (o[0] if isinstance(o, tuple) else o).detach().numpy()
        return fn

    for i, lyr in enumerate(model.model.layers):
        hooks.append(lyr.seq_norm.register_forward_hook(tap(f"seq_norm.{i}")))
        hooks.append(lyr.seq_modeling_block.register_forward_hook(tap(f"ttt.{i}")))
        hooks.append(lyr.mlp.register_forward_hook(tap(f"mlp.{i}")))
        hooks.append(lyr.register_forward_hook(tap(f"block.{i}")))
        if cfg["pre_conv"]:
            # ttt.py's `Conv` overrides __call__ (not forward), so nn.Module hooks
            # never fire on it; tap the inner Conv1d instead -- which is exactly
            # the tensor whose weight layout the conversion has to get right.
            hooks.append(lyr.conv.conv.register_forward_hook(tap(f"conv1d.{i}")))
    with torch.no_grad():
        o = model(input_ids=torch.from_numpy(ids), use_cache=False, return_dict=True)
    for h in hooks:
        h.remove()
    pt_logits = o.logits.numpy()

    # ---- reference, same taps --------------------------------------------
    results = []
    emb = params["model/wte/embedding"].astype(np.float64)
    h = emb[ids]
    for i in range(cfg["num_hidden_layers"]):
        pre = f"model/h/{i}/"
        if cfg["pre_conv"]:
            c = ref.rms_norm(h, params[pre + "conv/conv_norm/kernel"], cfg["rms_norm_eps"])
            c = ref.causal_depthwise_conv(c, params[pre + "conv/conv1/kernel"],
                                          params[pre + "conv/conv1/bias"])
            pt_c = taps[f"conv1d.{i}"][..., : c.shape[1]].transpose(0, 2, 1)
            results.append((f"pre_conv.{i}", relerr(pt_c, c)))
            h = h + c
        hp = ref.rms_norm(h, params[pre + "seq_norm/kernel"], cfg["rms_norm_eps"])
        results.append((f"seq_norm.{i}", relerr(taps[f"seq_norm.{i}"], hp)))
        sp = pre + "seq_modeling_block/"
        pblk = {k[len(sp):]: v for k, v in params.items() if k.startswith(sp)}
        t = ref.ttt_block(pblk, hp, rcfg)
        results.append((f"ttt.{i}", relerr(taps[f"ttt.{i}"], t)))
        h = h + t
        ff = ref.rms_norm(h, params[pre + "ffn_norm/kernel"], cfg["rms_norm_eps"])
        m_ = ref.swiglu(ff, params[pre + "feed_forward/w1/kernel"],
                        params[pre + "feed_forward/w2/kernel"],
                        params[pre + "feed_forward/w3/kernel"])
        results.append((f"mlp.{i}", relerr(taps[f"mlp.{i}"], m_)))
        h = h + m_
        results.append((f"block.{i}", relerr(taps[f"block.{i}"], h)))
    results.append(("logits", relerr(pt_logits, ref_logits)))

    worst = max(r for _, r in results)
    print(f"\n[{name}]  cfg={ {k: cfg[k] for k in ('ttt_layer_type','share_qk','use_gate','pre_conv','tie_word_embeddings')} }  "
          f"store={dtype} file={kind}")
    for nm, r in results:
        flag = "ok " if r <= tol else "FAIL"
        print(f"    {flag} {nm:<14s} rel_err = {r:.3e}")
    assert worst <= tol, f"[{name}] worst relative error {worst:.3e} > {tol:.1e}"
    shutil.rmtree(out, ignore_errors=True)
    os.remove(ck)
    return worst


MUTATIONS = {
    # each returns a wrong apply_transform; every one MUST be detected
    "drop the LR-gate transpose":
        lambda o: (lambda a, c, k: o(a, "" if c == "LR" else c, k)),
    "drop the Dense transpose":
        lambda o: (lambda a, c, k: o(a, "" if c == "T" else c, k)),
    "conv permuted (2,0,1) not (2,1,0)":
        lambda o: (lambda a, c, k: np.ascontiguousarray(a.transpose(2, 0, 1))
                   if c == "CONV" else o(a, c, k)),
    "conv taps time-flipped":
        lambda o: (lambda a, c, k: np.ascontiguousarray(a.transpose(2, 1, 0)[:, :, ::-1])
                   if c == "CONV" else o(a, c, k)),
    "W1 head-inner dims transposed":
        lambda o: (lambda a, c, k: np.ascontiguousarray(a.transpose(0, 2, 1))
                   if k.endswith("ttt_dense_0") else o(a, c, k)),
}

# mutations of the NAME map rather than the layout map
MAP_MUTATIONS = {
    "conv_q and conv_k swapped":
        lambda o: (lambda j: (o(j)[0].replace("conv_q", "conv_K")
                              .replace("conv_k", "conv_q")
                              .replace("conv_K", "conv_k"), o(j)[1])
                   if o(j) else None),
    "gate_proj and up_proj swapped (w1 <-> w3)":
        lambda o: (lambda j: (o(j)[0].replace("gate_proj", "GATE")
                              .replace("up_proj", "gate_proj")
                              .replace("GATE", "up_proj"), o(j)[1])
                   if o(j) else None),
}


def run_mutations(ttt_mod, tol):
    """Negative control: break the layout map on purpose and confirm the test
    notices.  A test that cannot fail proves nothing."""
    import contextlib
    import io
    import convert_jax_to_pt as C
    orig = C.apply_transform
    print("\n=== negative controls (each SHOULD be caught) ===")
    orig_map = C.map_key
    bad = []
    cases = ([(n, "apply_transform", f) for n, f in MUTATIONS.items()]
             + [(n, "map_key", f) for n, f in MAP_MUTATIONS.items()])
    for name, attr, make in cases:
        tmp = tempfile.mkdtemp(prefix="ttt_mut_")
        base = orig if attr == "apply_transform" else orig_map
        setattr(C, attr, make(base))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_variant("mut", {}, ttt_mod, tmp, tol=tol)
            print(f"    NOT CAUGHT  {name}")
            bad.append(name)
        except Exception as e:
            why = str(e).strip().splitlines()[0][:64]
            print(f"    caught      {name:<42s} ({type(e).__name__}: {why})")
        finally:
            setattr(C, attr, base)
            shutil.rmtree(tmp, ignore_errors=True)
    assert not bad, f"the test is blind to: {bad}"
    print("    all negative controls caught")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=None)
    ap.add_argument("--tol", type=float, default=3e-5)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-mutations", action="store_true",
                    help="skip the negative-control pass")
    a = ap.parse_args(argv)
    ttt_mod = import_ttt(a.repos)

    tmp = tempfile.mkdtemp(prefix="ttt_conv_test_")
    try:
        worsts = []
        # 1. the released TTT-Linear 760m architecture (share_qk / gate / pre_conv)
        worsts.append(run_variant("linear_released", {}, ttt_mod, tmp, tol=a.tol))
        # 2. the released TTT-MLP architecture
        worsts.append(run_variant("mlp_released", {"ttt_layer_type": "mlp"},
                                  ttt_mod, tmp, tol=a.tol))
        # 3. the *_base architecture: separate wk, no conv, no gate
        worsts.append(run_variant("linear_base",
                                  {"share_qk": False, "use_gate": False, "pre_conv": False},
                                  ttt_mod, tmp, tol=a.tol))
        # 4. untied lm_head + fp32 storage + streaming_params (no params/params prefix)
        worsts.append(run_variant("untied_fp32_params",
                                  {"tie_word_embeddings": False},
                                  ttt_mod, tmp, tol=a.tol,
                                  dtype="float32", kind="params"))
        print(f"\nALL VARIANTS PASS  (worst relative error {max(worsts):.3e} "
              f"<= tol {a.tol:.1e})")
        if not a.no_mutations:
            run_mutations(ttt_mod, a.tol)
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"(kept {tmp})")


if __name__ == "__main__":
    main()
