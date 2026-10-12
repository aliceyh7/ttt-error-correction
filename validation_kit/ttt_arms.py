#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-time interventions at the released fused LayerNorm gradient.

The historical names hebbian and hebbian_nm are retained for compatibility
with saved runs. The paper calls these prediction removed and norm matched.

For z = f_W(k), y = gamma * zhat + beta, and target = v - k,
    g = P_epsilon(zhat)[gamma * (y - target)] / std.
The drop_pred arm retains only the target component, so the removed component
includes both gamma**2 * zhat and gamma * beta after the Jacobian. The drop_w
alternative retains the beta component. These interventions need not agree.
Both retain state dependence through the LayerNorm Jacobian and std.

With gamma**2 = c and beta = 0, the removed component is
    c * epsilon * zhat / std**3.
Cancellation is exact at epsilon zero with nonzero prediction variance.
Gain variation alone does not determine the component because alignment with
activations matters. The pure alternative bypasses the normalization Jacobian,
but a nonlinear inner model can still introduce parameter dependence.

The patch affects the output and carried-state computations in both TTTLinear
and TTTMLP. For MLP it also propagates through the hidden-layer chain rule.
Norm matching preserves each token/head gradient norm at the current modified
state, not the summed chunk-update norm or the separate baseline trajectory.
"""

from __future__ import annotations

import contextlib
import os
import sys

import torch

REPOS = os.environ.get("TTCL_REPOS", "")
# Prefer the ttt.py vendored next to this file, so the kit is self-contained and
# needs no clone of ttt-lm-pytorch.  The vendored copy is byte-identical to
# upstream (see MANIFEST.txt); fall back to $TTCL_REPOS/ttt-lm-pytorch when absent.
_HERE = os.path.dirname(os.path.abspath(__file__))
TTT_PT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "ttt.py")) \
    else os.path.join(REPOS, "ttt-lm-pytorch")

ARMS = ("descent", "hebbian", "hebbian_nm", "ascent", "frozen")
HEBBIAN_MODES = ("drop_pred", "drop_w", "pure")
NORMMATCH_LEVELS = ("token", "block")

# Ratio log for the norm-matched arm: one entry per kernel call (= per layer per
# mini-batch), in call order.  run_ablation_ladder.py drains it per arm.
NORMMATCH_LOG = []


def import_ttt():
    """Import the authors' reference ttt.py verbatim (no edits on disk)."""
    if TTT_PT_DIR not in sys.path:
        sys.path.insert(0, TTT_PT_DIR)
    import ttt as ttt_mod  # noqa: E402

    return ttt_mod


# ---------------------------------------------------------------- arm kernels --
# Each kernel has the exact signature of ttt.ln_fused_l2_bwd (ttt.py:478).


def _ln_stats(x, gamma, beta, eps):
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    return D, std, x_hat


def _project(D, std, x_hat, grad_x_hat):
    """The LN Jacobian-vector product, byte-for-byte the expression at
    ttt.py:495-503."""
    return (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )


def make_hebbian_bwd(mode="drop_pred"):
    """Return an ln_fused_l2_bwd replacement with the error-correction
    (prediction-dependent) part of grad_output deleted."""
    if mode not in HEBBIAN_MODES:
        raise ValueError(f"hebbian mode must be one of {HEBBIAN_MODES}, got {mode}")

    def ln_fused_l2_bwd_hebbian(x, l2_target, gamma, beta, eps=1e-6):
        D, std, x_hat = _ln_stats(x, gamma, beta, eps)
        if mode == "drop_pred":
            grad_output = -l2_target                      # delete y = gamma*Zhat + beta
        elif mode == "drop_w":
            grad_output = beta - l2_target                # delete only gamma*Zhat
        else:  # 'pure'
            return gamma * (-l2_target)                   # + no LN Jacobian at all
        grad_x_hat = grad_output * gamma
        return _project(D, std, x_hat, grad_x_hat)

    ln_fused_l2_bwd_hebbian._arm = f"hebbian[{mode}]"
    return ln_fused_l2_bwd_hebbian


def make_normmatched_hebbian_bwd(original, mode="drop_pred", level="token", log=None):
    """Hebbian kernel rescaled so the update has the *norm* of the unmodified one.

    Deleting the error-correction term changes both the direction and the
    magnitude of the inner gradient, so the plain Hebbian penalty conflates the
    two.  This arm keeps the Hebbian *direction* and restores the descent
    *magnitude*:

        g_nm = g_hebb * ||g_desc|| / ||g_hebb||

    level='token'  (default): the norm is taken per token over the head-feature
        dim, i.e. g[..., t, :].  Because the fast-weight update is a sum of
        rank-one writes  -eta_t * x_t g_t^T  with ||x_t g_t^T||_F = ||x_t|| ||g_t||,
        this makes EVERY rank-one write have exactly the Frobenius norm of the
        corresponding write in the descent arm (x_t and eta_t are shared).  The
        norm of the summed mini-batch update is then matched up to the
        cross-terms between tokens, which the arms do not control.
    level='block': the norm is taken over the whole [K, f] mini-batch block per
        (batch, head), so the summed update is scaled by one scalar per
        mini-batch; exact matching of ||sum_t eta_t x_t g_t^T|| would need x and
        eta, which the one-function patch point does not see.

    The applied ratio is appended to `log` (default: NORMMATCH_LOG) as a float
    per call (mean over the batch/head/token entries of that call) together
    with its spread, so the geometry of the two components can be reported.
    """
    if level not in NORMMATCH_LEVELS:
        raise ValueError(f"normmatch level must be one of {NORMMATCH_LEVELS}")
    hebb = make_hebbian_bwd(mode)
    rec = NORMMATCH_LOG if log is None else log

    def ln_fused_l2_bwd_hebbian_nm(x, l2_target, gamma, beta, eps=1e-6):
        g_desc = original(x, l2_target, gamma, beta, eps)
        g_hebb = hebb(x, l2_target, gamma, beta, eps)
        dims = (-1,) if level == "token" else (-2, -1)
        n_desc = g_desc.norm(dim=dims, keepdim=True)
        n_hebb = g_hebb.norm(dim=dims, keepdim=True)
        ratio = n_desc / (n_hebb + 1e-30)
        with torch.no_grad():
            r = ratio.detach().to(torch.float64).flatten()
            rec.append({"ratio_mean": float(r.mean()), "ratio_median": float(r.median()),
                        "ratio_p05": float(r.quantile(0.05)), "ratio_p95": float(r.quantile(0.95)),
                        "n": int(r.numel())})
        return g_hebb * ratio.to(g_hebb.dtype)

    ln_fused_l2_bwd_hebbian_nm._arm = f"hebbian_nm[{mode},{level}]"
    return ln_fused_l2_bwd_hebbian_nm


def make_ascent_bwd(original):
    """Negate the inner gradient: W <- W + eta k g^T (expansion)."""

    def ln_fused_l2_bwd_ascent(x, l2_target, gamma, beta, eps=1e-6):
        return -original(x, l2_target, gamma, beta, eps)

    ln_fused_l2_bwd_ascent._arm = "ascent"
    return ln_fused_l2_bwd_ascent


def make_frozen_bwd():
    """Zero inner gradient => W1_last == W1_init, b1_last == b1_init exactly, and
    Z*_bar collapses to the fast-weight-at-init forward pass."""

    def ln_fused_l2_bwd_frozen(x, l2_target, gamma, beta, eps=1e-6):
        return torch.zeros_like(x)

    ln_fused_l2_bwd_frozen._arm = "frozen"
    return ln_fused_l2_bwd_frozen


# -------------------------------------------------- plain-MSE inner loss ------
# DIAGNOSTIC / POSITIVE-CONTROL MODE, not the shipped model.
#
# Removing the LayerNorm from the inner loss turns the update into the textbook
# delta rule  W <- W(I - eta k k^T) + eta k (v-k-b)^T , for which eta*||k||^2 IS
# the correct small-step parameter.  Running the same four arms under this loss
# is how we show that the harness detects the effect *when the effect is there*,
# and that the LN is the reason it is absent in the real model.
INNER_LOSSES = ("ln_fused", "plain_mse")


def plain_mse_bwd(x, l2_target, gamma, beta, eps=1e-6):
    return x - l2_target


def plain_mse_hebbian_bwd(x, l2_target, gamma, beta, eps=1e-6):
    return -l2_target


def build_arm_fn(arm, ttt_mod, hebbian_mode="drop_pred", inner_loss="ln_fused",
                 normmatch_level="token"):
    if inner_loss not in INNER_LOSSES:
        raise ValueError(f"inner_loss must be one of {INNER_LOSSES}")
    if inner_loss == "plain_mse":
        base, hebb = plain_mse_bwd, plain_mse_hebbian_bwd
    else:
        base = getattr(ttt_mod, "_ln_fused_l2_bwd_original", ttt_mod.ln_fused_l2_bwd)
        hebb = make_hebbian_bwd(hebbian_mode)
    if arm == "descent":
        return base                                       # for ln_fused: same object
    if arm == "hebbian":
        return hebb
    if arm == "hebbian_nm":
        if inner_loss == "plain_mse":
            raise ValueError("hebbian_nm is implemented for the ln_fused inner loss only")
        return make_normmatched_hebbian_bwd(base, hebbian_mode, normmatch_level)
    if arm == "ascent":
        return make_ascent_bwd(base)
    if arm == "frozen":
        return make_frozen_bwd()
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


@contextlib.contextmanager
def arm(name, ttt_mod=None, hebbian_mode="drop_pred", inner_loss="ln_fused",
        normmatch_level="token"):
    """Install an inner-update arm for the duration of the block.

    With inner_loss='ln_fused' (the default, = the shipped model), `descent`
    re-installs the *identical function object*, so arm-1 numerics are bitwise
    identical to the unpatched model by construction.
    """
    ttt_mod = ttt_mod or import_ttt()
    if not hasattr(ttt_mod, "_ln_fused_l2_bwd_original"):
        ttt_mod._ln_fused_l2_bwd_original = ttt_mod.ln_fused_l2_bwd
    saved = ttt_mod.ln_fused_l2_bwd
    ttt_mod.ln_fused_l2_bwd = build_arm_fn(name, ttt_mod, hebbian_mode, inner_loss,
                                           normmatch_level)
    try:
        yield ttt_mod
    finally:
        ttt_mod.ln_fused_l2_bwd = saved


# ------------------------------------------------- delta-term contribution ----
@contextlib.contextmanager
def delta_fraction_probe(ttt_mod=None, hebbian_mode="drop_pred", sink=None):
    """Run the UNMODIFIED (descent) model while recording, per call site, how much
    of the inner gradient the error-correction term actually accounts for:

        delta_fraction = || g_descent - g_hebbian || / || g_descent ||

    This is the direct, per-layer, on-real-tokens version of the paper's claim.
    Calls arrive in layer order within a forward pass (one call per mini-batch
    per layer), so the caller can attribute them; run_ablation_ladder.py does the
    attribution by wrapping it around a single forward pass.
    """
    ttt_mod = ttt_mod or import_ttt()
    if not hasattr(ttt_mod, "_ln_fused_l2_bwd_original"):
        ttt_mod._ln_fused_l2_bwd_original = ttt_mod.ln_fused_l2_bwd
    base = ttt_mod._ln_fused_l2_bwd_original
    hebb = make_hebbian_bwd(hebbian_mode)
    rec = [] if sink is None else sink

    def measuring(x, l2_target, gamma, beta, eps=1e-6):
        g = base(x, l2_target, gamma, beta, eps)
        with torch.no_grad():
            # recompute in float64: for channel-uniform gamma the descent and
            # Hebbian gradients agree to machine precision, so a float32
            # measurement of their difference is pure cancellation noise.
            x64, t64 = x.detach().double(), l2_target.detach().double()
            g64, b64 = gamma.detach().double(), beta.detach().double()
            gn = base(x64, t64, g64, b64, eps)
            dn = gn - hebb(x64, t64, g64, b64, eps)
            rec.append({
                "delta_fraction": float(dn.norm() / (gn.norm() + 1e-30)),
                "g_norm": float(gn.norm()),
                "target_norm": float(l2_target.detach().to(torch.float64).norm()),
                "gamma_cv2": _gamma_cv2(gamma),
            })
        return g

    saved = ttt_mod.ln_fused_l2_bwd
    ttt_mod.ln_fused_l2_bwd = measuring
    try:
        yield rec
    finally:
        ttt_mod.ln_fused_l2_bwd = saved


def _gamma_cv2(gamma):
    """Population CV of gamma squared across channels, averaged over heads.

    This parameter summary is not a calibrated correction gate. The projected
    prediction component also depends on activation alignment and epsilon.
    """
    g2 = gamma.detach().to(torch.float64).reshape(gamma.shape[0], -1) ** 2
    m = g2.mean(-1)
    s = g2.std(-1, unbiased=False)
    return float((s / (m.abs() + 1e-30)).mean())


# --------------------------------------------------------- state recorder ------
class StateRecorder:
    """Record the fast-weight carry after every mini-batch of every `scan` call.

    Implemented by wrapping the module-level `scan` (ttt.py:431), which both
    TTTLinear.ttt (ttt.py:1053) and TTTMLP.ttt (ttt.py:1252) route through.
    Used only for validation (arm-4 must leave the fast weights untouched).
    """

    def __init__(self, ttt_mod=None, keys=("W1_states", "b1_states", "W2_states", "b2_states")):
        self.ttt_mod = ttt_mod or import_ttt()
        self.keys = keys
        self.calls = []          # list per scan() call, each a list per mini-batch
        self._saved = None

    def __enter__(self):
        mod = self.ttt_mod
        self._saved = mod.scan
        outer = self

        def scan_recording(f, init, xs, out, checkpoint_group=0):
            log = []
            outer.calls.append(log)

            def f_rec(carry, x):
                new_carry, y = f(carry, x)
                log.append(
                    {
                        k: new_carry[k].detach().clone()
                        for k in outer.keys
                        if k in new_carry
                    }
                )
                return new_carry, y

            return outer._saved(f_rec, init, xs, out, checkpoint_group)

        mod.scan = scan_recording
        return self

    def __exit__(self, *exc):
        self.ttt_mod.scan = self._saved
        return False

    def max_drift(self, call_idx=0):
        """max_t max_ij |W(t)_ij - W(0)_ij| over the recorded mini-batches, and
        the same relative to ||W(0)||."""
        log = self.calls[call_idx]
        out = {}
        for k in self.keys:
            if not log or k not in log[0]:
                continue
            w0 = log[0][k]
            absd = max(float((s[k] - w0).abs().max()) for s in log)
            spread = max(
                float((log[i][k] - log[j][k]).abs().max())
                for i in range(len(log))
                for j in range(len(log))
            )
            out[k] = {
                "abs_drift_vs_first_chunk": absd,
                "max_pairwise_spread": spread,
                "w_absmax": float(w0.abs().max()),
                "n_chunks": len(log),
            }
        return out
