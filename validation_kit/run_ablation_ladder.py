#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_ladder.py -- HEADLINE experiment: five-arm ablation of the TTT
inner-loop update on a trained checkpoint, inference only.

    arm 1  descent   the reference update, unmodified                (baseline)
    arm 2  hebbian   error-correction term deleted, additive write kept
    arm 3  ascent    inner gradient negated (expansion, not contraction)
    arm 4  frozen    no inner update at all (fast weights stay at learned init)

All four run on the SAME frozen checkpoint and the SAME token stream.  Only the
inner update rule differs; see ttt_arms.py for the (one-function) patch and for
the fused-LayerNorm gradient decomposition that defines the Hebbian arm.

Perplexity is reported AS A FUNCTION OF CONTEXT POSITION, because the number of
compounded fast-weight updates grows with context: if the delta term does any
work, the descent/Hebbian gap must grow with the chunk index.  A single scalar
perplexity per arm averages that signal away.

Output:
    ablation_results.json          all buckets, all arms, plus per-layer eta*||k||^2
    ablation_ppl_vs_context.png    ppl (and delta-NLL) vs context position

EXAMPLE (researcher's Mac)
    python3 run_ablation_ladder.py \
        --checkpoint Test-Time-Training/ttt-linear-760m-pile-8k \
        --model ttt_linear --tokenizer meta-llama/Llama-2-7b-hf \
        --text corpus.txt --max-length 2048 --max-tokens 200000 --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ttt_arms import (  # noqa: E402
    ARMS, HEBBIAN_MODES, INNER_LOSSES, NORMMATCH_LEVELS, NORMMATCH_LOG, _gamma_cv2,
    arm as arm_ctx, delta_fraction_probe, import_ttt,
)


# ------------------------------------------------------------------ model IO --
def load_model(checkpoint, model_kind, device, dtype, ttt_mod, config_overrides=None):
    """Load a trained TTT checkpoint (HF id or local dir).  Outer weights frozen."""
    layer_type = "linear" if model_kind == "ttt_linear" else "mlp"
    kwargs = dict(torch_dtype=dtype)
    model = ttt_mod.TTTForCausalLM.from_pretrained(checkpoint, **kwargs)
    if model.config.ttt_layer_type != layer_type:
        raise SystemExit(
            f"checkpoint declares ttt_layer_type={model.config.ttt_layer_type!r} "
            f"but --model {model_kind} implies {layer_type!r}"
        )
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(model.config, k, v)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def build_random_model(ttt_mod, size="125m", model_kind="ttt_linear", seed=0,
                       vocab=512, n_layers=None, device="cpu", dtype=torch.float32):
    """A randomly-initialised model, for validation only (no HF download)."""
    torch.manual_seed(seed)
    cfg_kw = dict(ttt_mod.TTT_STANDARD_CONFIGS[size])
    if n_layers is not None:
        cfg_kw["num_hidden_layers"] = n_layers
    cfg = ttt_mod.TTTConfig(
        vocab_size=vocab,
        ttt_layer_type=("linear" if model_kind == "ttt_linear" else "mlp"),
        use_cache=False,
        **cfg_kw,
    )
    m = ttt_mod.TTTForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device=device, dtype=dtype)


# --------------------------------------------------------------- tokenisation --
def get_tokens(args):
    """Return a 1-D LongTensor of token ids."""
    if args.token_file:
        ids = torch.load(args.token_file)
        return torch.as_tensor(ids, dtype=torch.long).reshape(-1)

    text = None
    if args.text:
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif args.dataset:
        from datasets import load_dataset

        parts = args.dataset.split(":")
        name, cfgname = parts[0], (parts[1] if len(parts) > 1 else None)
        split = parts[2] if len(parts) > 2 else "test"
        ds = load_dataset(name, cfgname, split=split)
        col = args.dataset_column or ("text" if "text" in ds.column_names else ds.column_names[0])
        text = "\n\n".join(ds[col])
    else:
        raise SystemExit("need one of --text / --dataset / --token-file")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(text, return_tensors="pt").input_ids.reshape(-1)
    return ids


# ------------------------------------------------------- eta * ||k||^2 probe --
class EtaProbe:
    """Measure the effective inner step size per layer on the *real* token stream.

    Hooks TTTBase.get_ttt_inputs (ttt.py:810), whose return value already
    contains the exact XK, eta, token_eta and ttt_lr_eta that the inner loop
    uses -- no re-derivation of the gate or the key path is needed.

      eta_diag[j]  = token_eta[j] * ttt_lr_eta[j]          (ttt.py:829, diagonal)
                   = token_idx[j] * base_lr * sigmoid(gate_j) / head_dim
      eta_raw[j]   = ttt_lr_eta[j]                          (token_idx factor removed;
                                                             this is the quantity
                                                             reported by exp2's
                                                             measure_eta_k2.py)

    TTT-Linear : effective key = k          -> eta * ||k||^2
    TTT-MLP    : effective key for W2 = X2 = gelu(k W1 + b1)  (ttt.py:1129)
                 -> eta * ||X2||^2 , evaluated with W1,b1 at the layer's learned
                 init (the fast weights drift during the sequence; this is the
                 same point estimate exp2 reports, so the two are comparable).
    """

    def __init__(self, model, ttt_mod, model_kind, max_batches=8):
        self.model, self.ttt_mod, self.model_kind = model, ttt_mod, model_kind
        self.max_batches = max_batches
        self.stats = {}       # layer_idx -> dict of running sums
        self._handles = []
        self._saved = None

    def __enter__(self):
        base = self.ttt_mod.TTTBase
        self._saved = base.get_ttt_inputs
        outer = self

        def wrapped(self_layer, inputs, mini_batch_size, cache_params):
            out = outer._saved(self_layer, inputs, mini_batch_size, cache_params)
            try:
                outer._record(self_layer, out)
            except Exception as e:  # never let the probe break the eval
                outer.stats.setdefault("_errors", []).append(repr(e))
            return out

        base.get_ttt_inputs = wrapped
        return self

    def __exit__(self, *exc):
        self.ttt_mod.TTTBase.get_ttt_inputs = self._saved
        return False

    @torch.no_grad()
    def _record(self, layer, out):
        li = layer.layer_idx
        st = self.stats.setdefault(
            li, {"n": 0, "sum": 0.0, "sumsq": 0.0, "max": 0.0, "vals": [],
                 "raw_sum": 0.0, "sig_sum": 0.0, "batches": 0}
        )
        if st["batches"] >= self.max_batches:
            return
        st["batches"] += 1

        XK = out["XK"].to(torch.float32)                 # [B,nh,NMB,K,f]
        eta = out["eta"].to(torch.float32)               # [B,nh,NMB,K,K]
        ttt_lr = out["ttt_lr_eta"].to(torch.float32)     # [B,nh,NMB,1,K]
        token_eta = out["token_eta"].to(torch.float32)   # [B,nh,NMB,K,1]
        K = XK.shape[-2]
        eta_diag = eta[..., torch.arange(K), torch.arange(K)]   # [B,nh,NMB,K]
        eta_raw = ttt_lr[..., 0, :]                              # [B,nh,NMB,K]

        if self.model_kind == "ttt_mlp":
            W1 = layer.W1.to(torch.float32)                      # [nh,f,4f]
            b1 = layer.b1.to(torch.float32)                      # [nh,1,4f]
            Z1 = torch.einsum("bhnkf,hfg->bhnkg", XK, W1) + b1[None, :, None, :, :]
            X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
            key2 = (X2 ** 2).sum(-1)
            st["key_name"] = "||X2||^2 (post-GELU hidden; effective key of W2)"
        else:
            key2 = (XK ** 2).sum(-1)
            st["key_name"] = "||k||^2"

        eff = (eta_diag * key2).reshape(-1)
        eff_raw = (eta_raw * key2).reshape(-1)
        st["n"] += eff.numel()
        st["sum"] += float(eff.sum())
        st["sumsq"] += float((eff ** 2).sum())
        st["max"] = max(st["max"], float(eff.max()))
        st["raw_sum"] += float(eff_raw.sum())
        st["sig_sum"] += float((eta_raw * layer.head_dim / layer.config.ttt_base_lr).sum())
        if len(st["vals"]) < 200000:
            st["vals"].append(eff[:: max(1, eff.numel() // 4000)].cpu())
        st["token_eta_last"] = float(token_eta[..., -1, 0].mean())
        st["head_dim"] = layer.head_dim
        st["base_lr"] = layer.config.ttt_base_lr

    def summary(self):
        out = {}
        for li, st in self.stats.items():
            if not isinstance(li, int) or st["n"] == 0:
                continue
            v = torch.cat(st["vals"]) if st["vals"] else torch.zeros(1)
            q = torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
            mean = st["sum"] / st["n"]
            out[str(li)] = {
                "quantity": f"eta_diag * {st['key_name']}",
                "mean": mean,
                "std": math.sqrt(max(0.0, st["sumsq"] / st["n"] - mean ** 2)),
                "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
                "p75": float(q[3]), "p95": float(q[4]), "max": st["max"],
                "mean_no_token_idx": st["raw_sum"] / st["n"],
                "mean_gate_sigmoid": st["sig_sum"] / st["n"],
                "head_dim": st["head_dim"], "base_lr": st["base_lr"],
                "n_samples": st["n"],
            }
        return out


# ------------------------------------------------------------------ scoring --
@torch.no_grad()
def score_arm(model, tokens, args, device):
    """Per-token NLL for every scored position, bucketed by in-window position.

    Returns (nll_by_pos_sum, count_by_pos, total_nll, total_tokens, seconds).
    Windows are disjoint by default (stride == max_length) so that the in-window
    position index IS the context length -- which is the axis the claim lives on.
    """
    L, stride = args.max_length, (args.stride or args.max_length)
    n_tok = tokens.numel()
    starts = list(range(0, max(1, n_tok - 1), stride))
    starts = [s for s in starts if s + 2 <= n_tok]
    if args.max_windows:
        starts = starts[: args.max_windows]

    nll_sum = torch.zeros(L, dtype=torch.float64)
    nll_cnt = torch.zeros(L, dtype=torch.float64)
    flat_nll, flat_pos = [], []
    hidden = []                      # final hidden state, first N windows only
    t0 = time.time()
    done_tokens = 0

    hook, sink = None, []
    nwin_h = getattr(args, "divergence_windows", 0) or 0
    if nwin_h:
        def _cap(mod, inp, out):
            sink.append(out.detach()[0].float().cpu())
        hook = model.model.norm.register_forward_hook(_cap)

    for wi, s in enumerate(starts):
        ids = tokens[s : s + L].to(device)
        if ids.numel() < 2:
            break
        ids = ids.unsqueeze(0)
        out = model(input_ids=ids, use_cache=False, return_dict=True)
        logits = out.logits[0].float()                     # [l, V]
        tgt = ids[0, 1:]
        lp = torch.log_softmax(logits[:-1], dim=-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [l-1]
        nll = nll.double().cpu()
        # position p means: token p is predicted from a context of p tokens
        pos = torch.arange(1, nll.numel() + 1)
        nll_sum.index_add_(0, pos, nll)
        nll_cnt.index_add_(0, pos, torch.ones_like(nll))
        flat_nll.append(nll)
        flat_pos.append(pos)
        if sink:
            if wi < nwin_h:
                hidden.append(sink[-1])
            sink.clear()
        done_tokens += nll.numel()
        if args.verbose:
            print(f"    window {wi + 1}/{len(starts)}  tokens={done_tokens}", flush=True)
        if args.max_tokens and done_tokens >= args.max_tokens:
            break
    if hook is not None:
        hook.remove()
    return {
        "hidden": hidden,
        "nll_sum": nll_sum, "nll_cnt": nll_cnt,
        "flat_nll": torch.cat(flat_nll) if flat_nll else torch.zeros(0, dtype=torch.float64),
        "flat_pos": torch.cat(flat_pos) if flat_pos else torch.zeros(0, dtype=torch.long),
        "total_nll": float(nll_sum.sum()), "n_tokens": float(nll_cnt.sum()),
        "seconds": time.time() - t0,
    }


def bucketize(nll_sum, nll_cnt, mini_batch_size, n_pos_buckets=16):
    """Two bucketings: log-spaced by absolute position, and by chunk index."""
    L = nll_sum.numel()
    valid = nll_cnt > 0
    idx = torch.nonzero(valid).squeeze(-1)
    if idx.numel() == 0:
        return {"by_position": [], "by_chunk": []}
    lo, hi = int(idx.min()), int(idx.max())

    edges = torch.unique(
        torch.logspace(math.log10(max(1, lo)), math.log10(hi + 1), n_pos_buckets + 1).long()
    ).tolist()
    by_pos = []
    for a, b in zip(edges[:-1], edges[1:]):
        c = float(nll_cnt[a:b].sum())
        if c == 0:
            continue
        s = float(nll_sum[a:b].sum())
        by_pos.append({"pos_lo": a, "pos_hi": b - 1, "pos_mid": (a + b - 1) / 2.0,
                       "n": c, "mean_nll": s / c, "ppl": math.exp(s / c)})

    n_chunks = (L + mini_batch_size - 1) // mini_batch_size
    by_chunk = []
    for ci in range(n_chunks):
        a, b = ci * mini_batch_size, min(L, (ci + 1) * mini_batch_size)
        c = float(nll_cnt[a:b].sum())
        if c == 0:
            continue
        s = float(nll_sum[a:b].sum())
        by_chunk.append({"chunk": ci, "pos_lo": a, "pos_hi": b - 1, "n": c,
                         "mean_nll": s / c, "ppl": math.exp(s / c)})
    return {"by_position": by_pos, "by_chunk": by_chunk}


def paired_delta(arm_run, base_run, mini_batch_size, n_pos_buckets=16):
    """Token-paired delta-NLL vs the baseline arm.

    Every arm scores the identical windows in the identical order, so the flat
    per-token NLL vectors align element-wise and the difference is paired.  The
    paired standard error is what tells you whether a small delta is a real
    effect or sampling noise -- essential for the "descent == Hebbian" claim,
    which is a claim that a difference is ABSENT.
    """
    a, b = arm_run["flat_nll"], base_run["flat_nll"]
    pos = arm_run["flat_pos"]
    n = min(a.numel(), b.numel())
    d = a[:n] - b[:n]
    pos = pos[:n]

    def stat(mask):
        x = d[mask]
        if x.numel() == 0:
            return None
        m = float(x.mean())
        se = float(x.std(unbiased=True) / math.sqrt(x.numel())) if x.numel() > 1 else float("nan")
        return {"n": int(x.numel()), "delta_nll": m, "se": se,
                "t": (m / se if se and se > 0 else float("nan")),
                "ppl_ratio": math.exp(m)}

    out = {"overall": stat(torch.ones_like(pos, dtype=torch.bool))}
    lo, hi = int(pos.min()), int(pos.max())
    edges = sorted(set(
        torch.logspace(math.log10(max(1, lo)), math.log10(hi + 1),
                       n_pos_buckets + 1).long().tolist()))
    bypos = []
    for x, y in zip(edges[:-1], edges[1:]):
        s = stat((pos >= x) & (pos < y))
        if s:
            s.update(pos_lo=x, pos_hi=y - 1, pos_mid=(x + y - 1) / 2.0)
            bypos.append(s)
    out["by_position"] = bypos
    ch = pos // mini_batch_size
    bych = []
    for ci in range(int(ch.max()) + 1):
        s = stat(ch == ci)
        if s:
            s.update(chunk=ci)
            bych.append(s)
    out["by_chunk"] = bych
    # compounding: OLS slope of paired delta-NLL on chunk index, and the
    # early-half / late-half contrast (both are "does the gap grow?" statistics)
    if len(bych) >= 4:
        xs = torch.tensor([float(s["chunk"]) for s in bych])
        ys = torch.tensor([s["delta_nll"] for s in bych])
        w = torch.tensor([float(s["n"]) for s in bych])
        xm = float((xs * w).sum() / w.sum())
        ym = float((ys * w).sum() / w.sum())
        den = float((w * (xs - xm) ** 2).sum())
        out["slope_per_chunk"] = float((w * (xs - xm) * (ys - ym)).sum() / den) if den else None
        mid = bych[len(bych) // 2]["chunk"]
        e, l = stat(ch < mid), stat(ch >= mid)
        out["early_half"], out["late_half"] = e, l
        if e and l:
            se = math.sqrt(e["se"] ** 2 + l["se"] ** 2)
            out["late_minus_early"] = {
                "delta": l["delta_nll"] - e["delta_nll"], "se": se,
                "t": (l["delta_nll"] - e["delta_nll"]) / se if se > 0 else float("nan"),
            }
    return out


def hidden_divergence(arm_run, base_run, mini_batch_size):
    """Position-resolved relative divergence of the model's final hidden state.

    ||h_arm(p) - h_desc(p)|| / ||h_desc(p)|| per token position, bucketed by
    chunk index.  Much more sensitive than perplexity (no softmax, no averaging
    over the vocabulary), and it is the metric to use on synthetic /
    randomly-initialised checkpoints, where the LM head is uninformative and the
    per-token NLL is dominated by noise.
    """
    ha, hb = arm_run.get("hidden"), base_run.get("hidden")
    if not ha or not hb:
        return None
    a = torch.cat(ha, 0).double()
    b = torch.cat(hb[: len(ha)], 0).double()
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    rel = (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-30)
    L = ha[0].shape[0]
    pos = torch.arange(n) % L
    ch = pos // mini_batch_size
    by_chunk = []
    for ci in range(int(ch.max()) + 1):
        m = ch == ci
        if m.any():
            by_chunk.append({"chunk": ci, "rel": float(rel[m].mean())})
    return {"overall": float(rel.mean()),
            "first_chunk": by_chunk[0]["rel"] if by_chunk else None,
            "final_chunk": by_chunk[-1]["rel"] if by_chunk else None,
            "growth_ratio": (by_chunk[-1]["rel"] / by_chunk[0]["rel"]
                             if by_chunk and by_chunk[0]["rel"] > 0 else None),
            "by_chunk": by_chunk}


# --------------------------------------------------------------------- plot --
def make_plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in results["arms"] if results["per_arm"][a].get("by_position")]
    if not arms:
        return None
    base = results.get("baseline_arm", "descent")
    dv0 = results.get("deltas_vs_descent", {})
    npan = 3 if any("hidden_divergence" in v for v in dv0.values()) else 2
    fig, ax = plt.subplots(1, npan, figsize=(6.2 * npan, 4.6))
    colors = {"descent": "#1b6ca8", "hebbian": "#d1495b", "ascent": "#edae49",
              "frozen": "#6a6a6a"}
    for a in arms:
        bp = results["per_arm"][a]["by_position"]
        x = [b["pos_mid"] for b in bp]
        ax[0].plot(x, [b["ppl"] for b in bp], marker="o", ms=3.5, lw=1.6,
                   label=a, color=colors.get(a))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("context position (tokens seen)")
    ax[0].set_ylabel("perplexity")
    ax[0].set_title(title)
    ax[0].grid(alpha=0.25, which="both")
    ax[0].legend(frameon=False, fontsize=9)

    dv = results.get("deltas_vs_descent", {})
    for a in arms:
        if a == base or a not in dv:
            continue
        bp = dv[a]["by_position"]
        x = [b["pos_mid"] for b in bp]
        y = [b["delta_nll"] for b in bp]
        e = [b["se"] for b in bp]
        ax[1].errorbar(x, y, yerr=e, marker="o", ms=3.5, lw=1.6, capsize=2,
                       label=f"{a} - {base}", color=colors.get(a))
    if dv:
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("context position (tokens seen)")
        ax[1].set_ylabel(r"$\Delta$ mean NLL vs %s (nats/token)" % base)
        ax[1].set_title("compounding: divergence vs context length")
        ax[1].grid(alpha=0.25, which="both")
    if npan == 3:
        for a in arms:
            hd = dv0.get(a, {}).get("hidden_divergence")
            if not hd:
                continue
            bc = hd["by_chunk"]
            ax[2].plot([b["chunk"] + 1 for b in bc], [b["rel"] for b in bc],
                       marker="o", ms=3.5, lw=1.6, label=f"{a} vs {base}",
                       color=colors.get(a))
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        ax[2].set_xlabel("mini-batch (chunk) index + 1")
        ax[2].set_ylabel(r"$\|h_{arm}-h_{desc}\|/\|h_{desc}\|$")
        ax[2].set_title("functional divergence of the hidden state")
        ax[2].grid(alpha=0.25, which="both")
        ax[2].legend(frameon=False, fontsize=9)
        ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- main --
def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="HF id (e.g. Test-Time-Training/ttt-linear-760m-pile-8k) or local dir; "
                        "or 'random:<size>' for a smoke test with random weights")
    p.add_argument("--model", choices=["ttt_linear", "ttt_mlp"], default="ttt_linear")
    p.add_argument("--arms", default="descent,hebbian,hebbian_nm,ascent,frozen")
    p.add_argument("--hebbian-mode", default="drop_pred", choices=HEBBIAN_MODES,
                   help="how to delete the error-correction term under the fused LN "
                        "(see ttt_arms.py docstring)")
    p.add_argument("--normmatch-level", default="token", choices=NORMMATCH_LEVELS,
                   help="hebbian_nm arm: match the descent gradient norm per token "
                        "(every rank-one write keeps its Frobenius norm) or per "
                        "mini-batch block (see ttt_arms.py docstring)")
    p.add_argument("--inner-loss", default="ln_fused", choices=INNER_LOSSES,
                   help="'ln_fused' = the shipped inner loss.  'plain_mse' is a "
                        "DIAGNOSTIC that strips the LayerNorm from the inner loss, "
                        "recovering the textbook delta rule for which eta*||k||^2 is "
                        "the correct control parameter (positive control).")
    p.add_argument("--text", help="path to a plain-text corpus")
    p.add_argument("--dataset", help="HF dataset spec name[:config[:split]], e.g. "
                                     "wikitext:wikitext-103-raw-v1:test")
    p.add_argument("--dataset-column", default=None)
    p.add_argument("--token-file", help="pre-tokenised .pt of ids (skips the tokenizer)")
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=None,
                   help="window stride; default = --max-length (disjoint windows)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="stop after this many scored tokens PER ARM")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out-dir", default=HERE)
    p.add_argument("--tag", default="")
    p.add_argument("--eta-probe-batches", type=int, default=4)
    p.add_argument("--no-eta-probe", action="store_true")
    p.add_argument("--mini-batch-size", type=int, default=None,
                   help="override config.mini_batch_size (chunk size); default: checkpoint value")
    p.add_argument("--divergence-windows", type=int, default=1,
                   help="capture the final hidden state for this many windows per arm "
                        "and report ||h_arm - h_descent||/||h_descent|| vs context "
                        "position; 0 disables (saves L*hidden_size*4 bytes/window)")
    p.add_argument("--random-tokens", type=int, default=None,
                   help="VALIDATION ONLY: score this many uniformly-random token ids "
                        "instead of a corpus (the arms are still paired on the same ids)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; expected from {ARMS}")

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    ttt_mod = import_ttt()

    if args.checkpoint.startswith("random:"):
        size = args.checkpoint.split(":", 1)[1] or "125m"
        model = build_random_model(ttt_mod, size=size, model_kind=args.model,
                                   device=args.device, dtype=dtype,
                                   vocab=512, n_layers=2)
    else:
        model = load_model(args.checkpoint, args.model, args.device, dtype, ttt_mod)
    if args.mini_batch_size:
        model.config.mini_batch_size = args.mini_batch_size
        for m in model.modules():
            if isinstance(m, ttt_mod.TTTBase):
                m.mini_batch_size = args.mini_batch_size
    mbs = model.config.mini_batch_size

    if args.random_tokens or (args.checkpoint.startswith("random:") and
                              not (args.text or args.dataset or args.token_file)):
        n = args.random_tokens or args.max_length * 2
        g = torch.Generator().manual_seed(args.seed)
        tokens = torch.randint(0, int(model.config.vocab_size), (n,), generator=g)
    else:
        tokens = get_tokens(args)
    print(f"[tokens] {tokens.numel()} ids; window={args.max_length} "
          f"stride={args.stride or args.max_length} mini_batch_size={mbs}", flush=True)

    results = {
        "checkpoint": args.checkpoint, "model": args.model, "arms": arms,
        "hebbian_mode": args.hebbian_mode, "normmatch_level": args.normmatch_level,
        "baseline_arm": "descent",
        "max_length": args.max_length, "stride": args.stride or args.max_length,
        "inner_loss": args.inner_loss,
        "mini_batch_size": mbs, "dtype": args.dtype, "device": args.device,
        "n_tokens_available": int(tokens.numel()),
        "config": {k: getattr(model.config, k) for k in
                   ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "ttt_layer_type", "ttt_base_lr", "mini_batch_size",
                    "use_gate", "share_qk", "pre_conv")},
        "per_arm": {},
    }

    # ---- eta*||k||^2 on the real token stream (descent arm, unmodified path) --
    if not args.no_eta_probe:
        with EtaProbe(model, ttt_mod, args.model, max_batches=args.eta_probe_batches) as pr:
            with torch.no_grad():
                ids = tokens[: args.max_length].unsqueeze(0).to(args.device)
                model(input_ids=ids, use_cache=False, return_dict=True)
        results["eta_k2_per_layer"] = pr.summary()
        vals = [v["mean"] for v in results["eta_k2_per_layer"].values()]
        if vals:
            print(f"[eta*key^2] per-layer mean: min={min(vals):.3g} "
                  f"median={sorted(vals)[len(vals)//2]:.3g} max={max(vals):.3g}", flush=True)

        # --- how much of the inner gradient IS the error-correction term? ----
        nl = model.config.num_hidden_layers
        with delta_fraction_probe(ttt_mod, hebbian_mode=args.hebbian_mode) as rec:
            with torch.no_grad():
                ids = tokens[: args.max_length].unsqueeze(0).to(args.device)
                model(input_ids=ids, use_cache=False, return_dict=True)
        # one call per mini-batch per layer, emitted in layer order
        per_layer = {}
        if rec and len(rec) % nl == 0:
            per = len(rec) // nl
            for li in range(nl):
                sl = rec[li * per : (li + 1) * per]
                per_layer[str(li)] = {
                    "delta_fraction_mean": sum(s["delta_fraction"] for s in sl) / len(sl),
                    "delta_fraction_max": max(s["delta_fraction"] for s in sl),
                    "gamma_cv2": sl[0]["gamma_cv2"],
                    "mean_grad_norm": sum(s["g_norm"] for s in sl) / len(sl),
                    "mean_target_norm": sum(s["target_norm"] for s in sl) / len(sl),
                    "n_minibatches": len(sl),
                }
        results["delta_term_per_layer"] = per_layer
        # also record gamma / beta of every TTT layer straight from the weights
        results["ttt_norm_stats"] = {}
        for m in model.modules():
            if isinstance(m, ttt_mod.TTTBase):
                g = m.ttt_norm_weight.detach().to(torch.float64)
                b = m.ttt_norm_bias.detach().to(torch.float64)
                results["ttt_norm_stats"][str(m.layer_idx)] = {
                    "gamma_rms": float(g.pow(2).mean().sqrt()),
                    "gamma_cv2": _gamma_cv2(m.ttt_norm_weight),
                    "beta_rms": float(b.pow(2).mean().sqrt()),
                }
        if per_layer:
            df = [v["delta_fraction_mean"] for v in per_layer.values()]
            print(f"[delta-term] ||g_desc - g_hebb||/||g_desc|| per layer: "
                  f"min={min(df):.3g} median={sorted(df)[len(df)//2]:.3g} "
                  f"max={max(df):.3g}", flush=True)

    # ---- the four arms ------------------------------------------------------
    runs = {}
    for a in arms:
        print(f"[arm] {a}" + (f" ({args.hebbian_mode})" if a == "hebbian" else "")
              + (f" ({args.hebbian_mode}, norm-matched per {args.normmatch_level})"
                 if a == "hebbian_nm" else ""), flush=True)
        NORMMATCH_LOG.clear()
        with arm_ctx(a, ttt_mod, hebbian_mode=args.hebbian_mode,
                     inner_loss=args.inner_loss, normmatch_level=args.normmatch_level):
            run = score_arm(model, tokens, args, args.device)
        runs[a] = run
        if a == "hebbian_nm" and NORMMATCH_LOG:
            nl_ = model.config.num_hidden_layers
            rm = [e["ratio_mean"] for e in NORMMATCH_LOG]
            per_layer_ratio = {}
            M = max(1, args.max_length // mbs)          # mini-batches per full window
            if len(NORMMATCH_LOG) % (nl_ * M) == 0:
                # within a forward pass calls are layer-major: layer 0's M
                # mini-batches, then layer 1's, ...  (same order delta_fraction_probe relies on)
                for i, e in enumerate(NORMMATCH_LOG):
                    li = (i % (nl_ * M)) // M
                    per_layer_ratio.setdefault(str(li), []).append(e["ratio_mean"])
                per_layer_ratio = {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v),
                                       "n": len(v)} for k, v in per_layer_ratio.items()}
            results["normmatch_ratio"] = {
                "level": args.normmatch_level,
                "n_calls": len(NORMMATCH_LOG),
                "mean": sum(rm) / len(rm),
                "median": sorted(rm)[len(rm) // 2],
                "p05_of_call_p05": sorted(e["ratio_p05"] for e in NORMMATCH_LOG)[len(rm) // 20],
                "p95_of_call_p95": sorted(e["ratio_p95"] for e in NORMMATCH_LOG)[-max(1, len(rm) // 20)],
                "per_layer": per_layer_ratio,
            }
            print(f"       applied ratio ||g_desc||/||g_hebb||: mean={results['normmatch_ratio']['mean']:.4f} "
                  f"median={results['normmatch_ratio']['median']:.4f} "
                  f"(orthogonal components -> ~1.04 at 27% corrective share; "
                  f"opposed -> ~0.79; parallel -> ~1.37)", flush=True)
        tot, ntok, secs = run["total_nll"], run["n_tokens"], run["seconds"]
        rec = {
            "total_nll": tot, "n_scored_tokens": ntok,
            "ppl": math.exp(tot / ntok) if ntok else float("nan"),
            "mean_nll": tot / ntok if ntok else float("nan"),
            "seconds": secs,
            "tokens_per_second": ntok / secs if secs > 0 else float("nan"),
            **bucketize(run["nll_sum"], run["nll_cnt"], mbs),
        }
        results["per_arm"][a] = rec
        print(f"       ppl={rec['ppl']:.4f}  tokens={int(ntok)}  "
              f"{secs:.1f}s ({rec['tokens_per_second']:.0f} tok/s)", flush=True)

    # ---- token-paired deltas vs baseline ------------------------------------
    if "descent" in runs:
        results["deltas_vs_descent"] = {}
        for a in arms:
            if a == "descent":
                continue
            pd = paired_delta(runs[a], runs["descent"], mbs)
            pd["delta_ppl"] = results["per_arm"][a]["ppl"] - results["per_arm"]["descent"]["ppl"]
            hd = hidden_divergence(runs[a], runs["descent"], mbs)
            if hd:
                pd["hidden_divergence"] = hd
            results["deltas_vs_descent"][a] = pd

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    jpath = os.path.join(args.out_dir, f"ablation_results{tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=1)
    ppath = os.path.join(args.out_dir, f"ablation_ppl_vs_context{tag}.png")
    try:
        make_plot(results, ppath, f"{args.model} / {os.path.basename(args.checkpoint)}")
    except Exception as e:
        print(f"[plot] skipped: {e}")
        ppath = None

    print("\n=== summary (paired vs descent; se = paired standard error) ===")
    print(f"  {'arm':9s} {'ppl':>11s} {'dNLL':>10s} {'se':>9s} {'t':>8s} "
          f"{'slope/chunk':>12s} {'late-early':>11s}")
    for a in arms:
        r = results["per_arm"][a]
        d = results.get("deltas_vs_descent", {}).get(a)
        if not d:
            print(f"  {a:9s} {r['ppl']:11.4f}")
            continue
        o = d["overall"]
        lme = d.get("late_minus_early", {}).get("delta", float("nan"))
        sl = d.get("slope_per_chunk")
        hd = d.get("hidden_divergence")
        print(f"  {a:9s} {r['ppl']:11.4f} {o['delta_nll']:+10.5f} {o['se']:9.5f} "
              f"{o['t']:8.1f} {(sl if sl is not None else float('nan')):12.2e} "
              f"{lme:+11.5f}" +
              (f"   relH={hd['overall']:.3e} (chunk0 {hd['first_chunk']:.2e} -> "
               f"last {hd['final_chunk']:.2e})" if hd else ""))
    ek = results.get("eta_k2_per_layer", {})
    dt = results.get("delta_term_per_layer", {})
    ns = results.get("ttt_norm_stats", {})
    if ek or dt:
        print("\n=== per-layer cross-reference table ===")
        print(f"  {'layer':>5} {'eta*key^2 mean':>15} {'median':>10} {'p95':>10} "
              f"{'gamma_rms':>10} {'CV(gamma^2)':>12} {'delta_frac':>11}")
        rows = []
        for li in sorted(set(ek) | set(dt), key=lambda s: int(s)):
            e, d, n = ek.get(li, {}), dt.get(li, {}), ns.get(li, {})
            rows.append({"layer": int(li),
                         "eta_key2_mean": e.get("mean"), "eta_key2_median": e.get("median"),
                         "eta_key2_p95": e.get("p95"), "gamma_rms": n.get("gamma_rms"),
                         "gamma_cv2": n.get("gamma_cv2"),
                         "delta_fraction": d.get("delta_fraction_mean")})
            f = lambda v: ("%10.4g" % v) if v is not None else " " * 10
            print(f"  {li:>5} {f(e.get('mean')):>15} {f(e.get('median'))} "
                  f"{f(e.get('p95'))} {f(n.get('gamma_rms'))} "
                  f"{f(n.get('gamma_cv2')):>12} {f(d.get('delta_fraction_mean')):>11}")
        results["cross_reference_table"] = rows
        with open(jpath, "w") as f2:
            json.dump(results, f2, indent=1)

    print(f"\nwrote {jpath}" + (f" and {ppath}" if ppath else ""))
    return results


if __name__ == "__main__":
    main()
