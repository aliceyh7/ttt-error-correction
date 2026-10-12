#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_delta_probe.py -- run `delta_fraction_probe` on a converted OFFICIAL
TTT checkpoint and emit the per-layer delta_fraction table alongside the
CV(gamma^2) that the theory says controls it.

    delta_fraction = || g_descent - g_hebbian || / || g_descent ||

measured in float64 at every (layer, mini-batch) call site of
`ttt.ln_fused_l2_bwd`, on REAL tokens, with the model otherwise completely
unmodified (the probe computes the Hebbian gradient as a side calculation and
returns the descent gradient, so the forward pass is bit-identical to the
stock model).

The claim under test (ttt_arms.py, "THE FUSED-LAYERNORM SUBTLETY"): under the
shipped fused-LayerNorm inner loss the error-correction ("delta") term is
annihilated exactly when `ttt_norm_weight` is channel-uniform, so the natural
control parameter is CV(gamma^2), not eta*||k||^2.  This script measures both
on the real weights and prints them side by side.

    python3 run_delta_probe.py --checkpoint OUT_DIR --text corpus.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ttt_arms import (  # noqa: E402
    HEBBIAN_MODES, _gamma_cv2, delta_fraction_probe, import_ttt,
)


# ------------------------------------------------------------------ tokens ---
def get_tokens(args, vocab_size):
    if args.token_file:
        ids = torch.load(args.token_file)
        return torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    if args.random_tokens:
        g = torch.Generator().manual_seed(args.seed)
        return torch.randint(0, vocab_size, (args.random_tokens,), generator=g)
    if not args.text:
        raise SystemExit("need --text CORPUS.txt (or --token-file / --random-tokens)")
    with open(args.text, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(text, return_tensors="pt").input_ids.reshape(-1)
    if int(ids.max()) >= vocab_size:
        raise SystemExit(
            f"tokenizer produced id {int(ids.max())} but the model vocab is "
            f"{vocab_size}; the released TTT models are Llama-2 tokenized "
            f"(--tokenizer NousResearch/Llama-2-7b-hf)"
        )
    return ids


# -------------------------------------------------------------------- main ---
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="converted HF directory (output of convert_jax_to_pt.py)")
    p.add_argument("--text", help="plain-text corpus")
    p.add_argument("--token-file", help="pre-tokenised .pt of ids")
    p.add_argument("--random-tokens", type=int, default=None,
                   help="SANITY ONLY: probe on uniformly random ids")
    p.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--max-length", type=int, default=2048, help="probe window length")
    p.add_argument("--windows", type=int, default=4,
                   help="number of disjoint windows to average over")
    p.add_argument("--hebbian-mode", default="drop_pred", choices=HEBBIAN_MODES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--eta-probe", action="store_true",
                   help="also measure eta*||k||^2 per layer (needs run_ablation_ladder.py)")
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ttt_mod = import_ttt()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    t0 = time.time()
    model = ttt_mod.TTTForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    model.to(args.device)
    cfg = model.config
    nl = int(cfg.num_hidden_layers)
    print(f"[model] {args.checkpoint}  ttt_layer_type={cfg.ttt_layer_type} "
          f"layers={nl} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"share_qk={cfg.share_qk} use_gate={cfg.use_gate} pre_conv={cfg.pre_conv} "
          f"mini_batch_size={cfg.mini_batch_size}  ({time.time() - t0:.1f}s to load)",
          flush=True)

    tokens = get_tokens(args, int(cfg.vocab_size))
    L = args.max_length
    nwin = min(args.windows, max(1, tokens.numel() // L))
    print(f"[tokens] {tokens.numel()} ids -> {nwin} window(s) of {L}", flush=True)

    # ---- static weight statistics (no data needed) ------------------------
    norm_stats = {}
    for m in model.modules():
        if isinstance(m, ttt_mod.TTTBase):
            g = m.ttt_norm_weight.detach().to(torch.float64)
            b = m.ttt_norm_bias.detach().to(torch.float64)
            norm_stats[str(m.layer_idx)] = {
                "gamma_rms": float(g.pow(2).mean().sqrt()),
                "gamma_cv2": _gamma_cv2(m.ttt_norm_weight),
                "beta_rms": float(b.pow(2).mean().sqrt()),
                "gamma_min": float(g.min()), "gamma_max": float(g.max()),
            }

    # ---- the probe --------------------------------------------------------
    acc = {li: {"df": [], "gn": [], "tn": []} for li in range(nl)}
    t0 = time.time()
    with delta_fraction_probe(ttt_mod, hebbian_mode=args.hebbian_mode) as rec:
        with torch.no_grad():
            for w in range(nwin):
                rec.clear()
                ids = tokens[w * L:(w + 1) * L].unsqueeze(0).to(args.device)
                model(input_ids=ids, use_cache=False, return_dict=True)
                if not rec or len(rec) % nl != 0:
                    raise SystemExit(
                        f"probe recorded {len(rec)} calls, not a multiple of "
                        f"{nl} layers; cannot attribute per layer"
                    )
                per = len(rec) // nl
                for li in range(nl):
                    sl = rec[li * per:(li + 1) * per]
                    acc[li]["df"] += [s["delta_fraction"] for s in sl]
                    acc[li]["gn"] += [s["g_norm"] for s in sl]
                    acc[li]["tn"] += [s["target_norm"] for s in sl]
                print(f"  window {w + 1}/{nwin} done ({time.time() - t0:.1f}s)", flush=True)

    per_layer = {}
    for li in range(nl):
        d = acc[li]["df"]
        per_layer[str(li)] = {
            "delta_fraction_mean": statistics.fmean(d),
            "delta_fraction_median": statistics.median(d),
            "delta_fraction_max": max(d),
            "delta_fraction_min": min(d),
            "mean_grad_norm": statistics.fmean(acc[li]["gn"]),
            "mean_target_norm": statistics.fmean(acc[li]["tn"]),
            "n_minibatches": len(d),
            **norm_stats.get(str(li), {}),
        }

    # ---- optional eta*||k||^2 cross-reference -----------------------------
    eta_stats = {}
    if args.eta_probe:
        try:
            from run_ablation_ladder import EtaProbe
            kind = "ttt_linear" if cfg.ttt_layer_type == "linear" else "ttt_mlp"
            with EtaProbe(model, ttt_mod, kind, max_batches=4) as pr:
                with torch.no_grad():
                    model(input_ids=tokens[:L].unsqueeze(0).to(args.device),
                          use_cache=False, return_dict=True)
            eta_stats = pr.summary()
        except Exception as e:
            print(f"[eta] skipped: {e}")

    # ---- report -----------------------------------------------------------
    print("\n=== per-layer delta_fraction  vs  CV(gamma^2) ===")
    print("  ||g_descent - g_hebbian|| / ||g_descent||, float64, real tokens\n")
    hdr = (f"  {'layer':>5} {'delta_frac':>11} {'median':>10} {'max':>10} "
           f"{'CV(gamma^2)':>12} {'gamma_rms':>10} {'beta_rms':>9} "
           f"{'|g|':>10} {'|target|':>10}")
    if eta_stats:
        hdr += f" {'eta*key^2':>10}"
    print(hdr)
    for li in range(nl):
        v = per_layer[str(li)]
        line = (f"  {li:>5} {v['delta_fraction_mean']:>11.4g} "
                f"{v['delta_fraction_median']:>10.4g} {v['delta_fraction_max']:>10.4g} "
                f"{v['gamma_cv2']:>12.4g} {v['gamma_rms']:>10.4g} {v['beta_rms']:>9.3g} "
                f"{v['mean_grad_norm']:>10.4g} {v['mean_target_norm']:>10.4g}")
        if eta_stats:
            e = eta_stats.get(str(li), eta_stats.get(li, {}))
            line += f" {e.get('mean', float('nan')):>10.4g}"
        print(line)

    df = [per_layer[str(li)]["delta_fraction_mean"] for li in range(nl)]
    cv = [per_layer[str(li)]["gamma_cv2"] for li in range(nl)]
    print(f"\n  delta_fraction  : min={min(df):.4g}  median={statistics.median(df):.4g}  "
          f"max={max(df):.4g}  mean={statistics.fmean(df):.4g}")
    print(f"  CV(gamma^2)     : min={min(cv):.4g}  median={statistics.median(cv):.4g}  "
          f"max={max(cv):.4g}")
    if nl > 2 and statistics.pstdev(cv) > 0 and statistics.pstdev(df) > 0:
        mc, md = statistics.fmean(cv), statistics.fmean(df)
        num = sum((c - mc) * (d - md) for c, d in zip(cv, df))
        den = math.sqrt(sum((c - mc) ** 2 for c in cv) * sum((d - md) ** 2 for d in df))
        print(f"  Pearson r( CV(gamma^2), delta_fraction ) over layers = {num / den:+.3f}")

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "hebbian_mode": args.hebbian_mode,
        "max_length": L, "windows": nwin, "dtype": args.dtype,
        "config": {k: getattr(cfg, k) for k in
                   ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "intermediate_size", "vocab_size", "ttt_layer_type",
                    "ttt_base_lr", "mini_batch_size", "use_gate", "share_qk",
                    "pre_conv", "conv_kernel", "rope_theta")},
        "delta_term_per_layer": per_layer,
        "eta_k2_per_layer": eta_stats,
        "summary": {
            "delta_fraction_min": min(df), "delta_fraction_max": max(df),
            "delta_fraction_mean": statistics.fmean(df),
            "gamma_cv2_min": min(cv), "gamma_cv2_max": max(cv),
        },
    }
    path = args.out or os.path.join(os.getcwd(), "delta_fraction_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
