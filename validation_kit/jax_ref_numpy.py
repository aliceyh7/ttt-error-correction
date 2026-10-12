#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jax_ref_numpy.py -- a minimal, dependency-free NumPy transcription of the FLAX
TTT forward pass (ttt-lm-jax/ttt/models/{model.py,ttt_layer.py}).

Purpose: this is the *reference side* of the conversion equivalence test.  To
keep the kit free of a JAX/flax dependency, instead of running the real flax model
we transcribe it operation by operation and run it on the *unconverted*, native
flax-layout parameter tree.  If `convert_jax_to_pt.py` gets a transpose, a conv
layout, a head reshape or a name wrong, the PyTorch model's output will disagree
with this reference.

Everything here is computed in float64 so that the comparison against the fp32
torch model is limited by the torch side only.

Transcription map (flax source -> function here)
    model.py:307 RMSNorm.__call__            -> rms_norm
    model.py:379 ConvModule.__call__         -> causal_depthwise_conv (+ rms_norm)
    model.py:327 SwiGLUMLP.__call__          -> swiglu
    ttt_layer.py:54 precompute_freqs_cis     -> freqs_cis
    ttt_layer.py:64 apply_rotary_emb         -> apply_rope
    ttt_layer.py:206 TTTBase.get_ttt_inputs  -> _ttt_inputs
    ttt_layer.py:224 TTTBase.get_eta         -> _ttt_inputs (eta block)
    ttt_layer.py:363 TTTLinearBase.process_mini_batch -> _mini_batch_linear
    ttt_layer.py:437 TTTMLPBase.process_mini_batch    -> _mini_batch_mlp
    ttt_layer.py:318 TTTBase.__call__        -> ttt_block
    model.py:610  Block.__call__             -> block
    model.py:809  Model.__call__ / CausalLM  -> forward
"""

from __future__ import annotations

import numpy as np

F64 = np.float64


# ------------------------------------------------------------------ norms ---
def rms_norm(x, kernel, eps=1e-6):
    """model.py:307.  x: [..., C], kernel: [C]"""
    x = x.astype(F64)
    return x * (1.0 / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps)) * kernel


def layer_norm(x, scale, bias, eps=1e-6):
    """flax nn.LayerNorm with default epsilon=1e-6."""
    x = x.astype(F64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * scale + bias


def layer_norm_vjp(x, cot, scale, eps=1e-6):
    """jax.vjp of `layer_norm` w.r.t. x, evaluated at cotangent `cot`.

    d/dx of  y = scale*(x-mu)/s + bias,  s = sqrt(var+eps):

        dL/dx = ( gxhat - mean(gxhat) - xhat * mean(gxhat*xhat) ) / s ,
        gxhat = cot * scale

    which is byte-for-byte the expression the PyTorch reference fuses into
    `ln_fused_l2_bwd` (ttt-lm-pytorch/ttt.py:478-505).
    """
    x = x.astype(F64)
    D = x.shape[-1]
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    s = np.sqrt(var + eps)
    xhat = (x - mu) / s
    gxhat = cot * scale
    return (gxhat - gxhat.mean(-1, keepdims=True)
            - xhat * (gxhat * xhat).mean(-1, keepdims=True)) / s


# ------------------------------------------------------------ activations ---
def gelu_tanh(x):
    """flax nn.gelu(approximate=True) == torch F.gelu(approximate='tanh')."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def diff_gelu(x):
    """ttt_layer.py:82 (== ttt.py gelu_bwd)."""
    t = np.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - t * t) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + t)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------- rope ---
def freqs_cis(dim, end, theta=10000.0):
    """ttt_layer.py:54"""
    f = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2].astype(F64) / dim))
    t = np.arange(end).astype(F64)
    ang = np.outer(t, f)
    return np.cos(ang) + 1j * np.sin(ang)          # [end, dim/2]


def apply_rope(x, fc):
    """ttt_layer.py:64.  x: [B, N, nh, f], fc: [B, N, f/2] complex."""
    r = x.astype(F64).reshape(*x.shape[:-1], -1, 2)
    z = r[..., 0] + 1j * r[..., 1]                 # [B, N, nh, f/2]
    z = z * fc[:, :, None, :]
    return np.stack((z.real, z.imag), -1).reshape(x.shape)


# ------------------------------------------------------------------- conv ---
def causal_depthwise_conv(x, kernel, bias):
    """flax nn.Conv(padding='CAUSAL', feature_group_count=C).

    x: [B, N, C];  kernel: [W, 1, C] (the native flax layout);  bias: [C]
        out[t, c] = sum_j kernel[j, 0, c] * x[t + j - (W-1), c] + bias[c]
    """
    x = x.astype(F64)
    B, N, C = x.shape
    W = kernel.shape[0]
    pad = np.concatenate([np.zeros((B, W - 1, C), F64), x], axis=1)
    out = np.zeros((B, N, C), F64)
    for j in range(W):
        out += pad[:, j:j + N, :] * kernel[j, 0, :]
    return out + bias


def swiglu(x, w1, w2, w3):
    """model.py:327:  w2(silu(w1 x) * w3 x)"""
    x = x.astype(F64)
    return (silu(x @ w1) * (x @ w3)) @ w2


# ------------------------------------------------------------- TTT layer ----
def _ttt_inputs(p, hp, cfg):
    """ttt_layer.py:206 get_ttt_inputs + :224 get_eta.

    Returns XQ, XK, XV as [B, nh, NMB, K, f] and eta as [B, nh, NMB, K, K].
    """
    B, N, C = hp.shape
    nh, K = cfg["num_heads"], cfg["mini_batch_size"]
    f = C // nh
    NMB = N // K

    if cfg["share_qk"]:
        xqk = hp @ p["wq/kernel"]
        XQ = causal_depthwise_conv(xqk, p["conv_q/kernel"], p["conv_q/bias"])
        XK = causal_depthwise_conv(xqk, p["conv_k/kernel"], p["conv_k/bias"])
    else:
        XQ = hp @ p["wq/kernel"]
        XK = hp @ p["wk/kernel"]
    XV = hp @ p["wv/kernel"]

    XQ, XK, XV = (a.reshape(B, N, nh, f) for a in (XQ, XK, XV))

    pos = np.arange(N) % K
    fc = freqs_cis(f, 2 * K, cfg["rope_theta"])[pos][None].repeat(B, 0)
    XQ = apply_rope(XQ, fc)
    XK = apply_rope(XK, fc)

    def split_mb(a):                                # ttt_layer.py:196
        return a.reshape(B, NMB, K, nh, f).transpose(0, 3, 1, 2, 4)

    XQ, XK, XV = split_mb(XQ), split_mb(XK), split_mb(XV)

    # --- eta (ttt_layer.py:224) ---
    X = hp.reshape(B, NMB, K, C)
    # vmap over heads of Dense(width -> 1): kernel [nh, C, 1], bias [nh, 1]
    lr = np.einsum("bnkc,hcd->bhnkd", X, p["learnable_ttt_lr/kernel"]) \
        + p["learnable_ttt_lr/bias"].reshape(1, nh, 1, 1, 1)
    lr = sigmoid(lr).transpose(0, 1, 2, 4, 3)       # [B, nh, NMB, 1, K]
    tok = p["learnable_token_idx"] + 1.0 / np.arange(1, K + 1).astype(F64)
    tok = np.clip(tok, 0.0, None)
    eta = (cfg["ttt_base_lr"] * tok).reshape(1, 1, 1, K, 1) * lr / f
    return XQ, XK, XV, eta


def _mini_batch_linear(p, W1, b1, XQ, XK, XV, eta):
    """ttt_layer.py:363, for one (batch, head).  Arrays are [K, f]."""
    g, b = p["ttt_norm/scale"], p["ttt_norm/bias"]
    sq_eta = eta                                    # [K, K]
    last_eta = eta[-1][:, None]                     # [K, 1]

    X1 = XK
    Z1 = X1 @ W1 + b1
    out = layer_norm(Z1, g, b)
    tgt = XV - XK
    gout = out - tgt
    gZ1 = layer_norm_vjp(Z1, gout, g)

    X1b = XQ
    tril = np.tril(np.ones((X1.shape[0], X1.shape[0]), F64))
    Attn1 = np.tril(X1b @ X1.T)
    b1b = b1 - (sq_eta * tril) @ gZ1
    Z1b = X1b @ W1 - (sq_eta * Attn1) @ gZ1 + b1b
    outb = layer_norm(Z1b, g, b)

    W1n = W1 - (last_eta * X1).T @ gZ1
    b1n = b1 - np.sum(last_eta * gZ1, 0, keepdims=True)
    return (W1n, b1n), X1b + outb


def _mini_batch_mlp(p, W1, b1, W2, b2, XQ, XK, XV, eta):
    """ttt_layer.py:437, for one (batch, head)."""
    g, b = p["ttt_norm/scale"], p["ttt_norm/bias"]
    sq_eta = eta
    last_eta = eta[-1][:, None]

    X1 = XK
    Z1 = X1 @ W1 + b1
    X2 = gelu_tanh(Z1)
    Z2 = X2 @ W2 + b2
    out = layer_norm(Z2, g, b)
    tgt = XV - X1
    gout = out - tgt
    gZ2 = layer_norm_vjp(Z2, gout, g)
    gZ1 = (gZ2 @ W2.T) * diff_gelu(Z1)

    X1b = XQ
    tril = np.tril(np.ones((X1.shape[0], X1.shape[0]), F64))
    Attn1 = np.tril(X1b @ X1.T)
    b1b = b1 - (sq_eta * tril) @ gZ1
    Z1b = X1b @ W1 - (sq_eta * Attn1) @ gZ1 + b1b
    X2b = gelu_tanh(Z1b)
    Attn2 = np.tril(X2b @ X2.T)
    b2b = b2 - (sq_eta * tril) @ gZ2
    Z2b = X2b @ W2 - (sq_eta * Attn2) @ gZ2 + b2b
    outb = layer_norm(Z2b, g, b)

    W1n = W1 - (last_eta * X1).T @ gZ1
    b1n = b1 - np.sum(last_eta * gZ1, 0, keepdims=True)
    W2n = W2 - (last_eta * X2).T @ gZ2
    b2n = b2 - np.sum(last_eta * gZ2, 0, keepdims=True)
    return (W1n, b1n, W2n, b2n), X1b + outb


def ttt_block(p, hp, cfg):
    """ttt_layer.py:318 TTTBase.__call__ (+ TTTLinear/TTTMLP overrides)."""
    B, N, C = hp.shape
    nh, K = cfg["num_heads"], cfg["mini_batch_size"]
    f = C // nh
    NMB = N // K
    XQ, XK, XV, eta = _ttt_inputs(p, hp, cfg)

    Z = np.zeros((B, nh, NMB, K, f), F64)
    for bi in range(B):
        for h in range(nh):
            ph = {"ttt_norm/scale": p["ttt_norm/scale"][h],
                  "ttt_norm/bias": p["ttt_norm/bias"][h]}
            if cfg["ttt_layer_type"] == "linear":
                st = (p["ttt_dense_0"][h].astype(F64), p["ttt_bias_0"][h].astype(F64))
                for n in range(NMB):
                    st, o = _mini_batch_linear(ph, st[0], st[1],
                                               XQ[bi, h, n], XK[bi, h, n],
                                               XV[bi, h, n], eta[bi, h, n])
                    Z[bi, h, n] = o
            else:
                st = (p["ttt_dense_0"][h].astype(F64), p["ttt_bias_0"][h].astype(F64),
                      p["ttt_dense_1"][h].astype(F64), p["ttt_bias_1"][h].astype(F64))
                for n in range(NMB):
                    st, o = _mini_batch_mlp(ph, *st,
                                            XQ[bi, h, n], XK[bi, h, n],
                                            XV[bi, h, n], eta[bi, h, n])
                    Z[bi, h, n] = o

    Z = Z.reshape(B, nh, N, f).transpose(0, 2, 1, 3).reshape(B, N, C)
    Z = layer_norm(Z, p["post_norm/scale"], p["post_norm/bias"])
    if cfg["use_gate"]:
        Z = gelu_tanh(hp @ p["wg/kernel"]) * Z
    return Z @ p["wo/kernel"]


# ----------------------------------------------------------------- model ----
def block(params, i, h, cfg):
    """model.py:610 Block.__call__"""
    pre = f"model/h/{i}/"
    if cfg["pre_conv"]:
        c = rms_norm(h, params[pre + "conv/conv_norm/kernel"], cfg["rms_norm_eps"])
        c = causal_depthwise_conv(c, params[pre + "conv/conv1/kernel"],
                                  params[pre + "conv/conv1/bias"])
        h = h + c
    hp = rms_norm(h, params[pre + "seq_norm/kernel"], cfg["rms_norm_eps"])
    sp = pre + "seq_modeling_block/"
    p = {k[len(sp):]: v for k, v in params.items() if k.startswith(sp)}
    h = h + ttt_block(p, hp, cfg)
    ff = rms_norm(h, params[pre + "ffn_norm/kernel"], cfg["rms_norm_eps"])
    return h + swiglu(ff, params[pre + "feed_forward/w1/kernel"],
                      params[pre + "feed_forward/w2/kernel"],
                      params[pre + "feed_forward/w3/kernel"])


def forward(params, input_ids, cfg):
    """model.py:809 Model + :872 CausalLM.  Returns (logits, final_hidden)."""
    emb = params["model/wte/embedding"].astype(F64)
    h = emb[input_ids]
    for i in range(cfg["num_hidden_layers"]):
        h = block(params, i, h, cfg)
    h = rms_norm(h, params["model/ln_f/kernel"], cfg["rms_norm_eps"])
    head = params["lm_head/kernel"] if "lm_head/kernel" in params else emb.T
    return h @ head.astype(F64), h
