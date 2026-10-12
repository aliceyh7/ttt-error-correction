#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ttt_arms.py -- five-arm ablation of the TTT inner-loop update.

The ablation is applied to the *reference* implementation
(ttt-lm-pytorch/ttt.py) at inference time only.  Outer weights are frozen; the
only thing that changes between arms is the inner-loop update rule.

WHY A ONE-FUNCTION PATCH IS SUFFICIENT
--------------------------------------
Read ttt.py carefully and you find that *every* consumer of the inner gradient,
in both the dual and the primal form, and in both TTTLinear and TTTMLP, is a
function of a single tensor produced by a single module-level function:

    ttt.py:975   grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ...)   # TTTLinear
    ttt.py:1137  grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ...)   # TTTMLP
    ttt.py:1139  grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.T * gelu_bwd(Z1)          # TTTMLP chain rule

Consumers:
    TTTLinear dual   : ttt.py:981 (b1_bar), 983 (Z1_bar), 987 (W1_last), 989 (b1_last)
    TTTLinear primal : ttt.py:1003-1007 (grad_W1, grad_b1) -> 1010-1019
    TTTMLP    dual   : ttt.py:1144,1146 (b1_bar,Z1_bar), 1152,1154 (b2_bar,Z2_bar),
                       1158-1164 (W1_last,b1_last,W2_last,b2_last)
    TTTMLP    primal : ttt.py:1181-1194 -> 1196-1213

`compute_mini_batch` is a closure that resolves `ln_fused_l2_bwd` as a *module
global* at call time.  Rebinding that global therefore changes the inner update
consistently everywhere -- dual form, primal form, output path (`Z*_bar`) and
carried state (`W*_last`) -- with zero risk of the arms differing from the
baseline anywhere except in the intended term.  We never fork or paraphrase the
file.

THE FUSED-LAYERNORM SUBTLETY
----------------------------
The inner loss is NOT  1/2 || W k + b - (v - k) ||^2 .  It is

    l(W) = 1/2 || LN_{gamma,beta}( f_W(k) ) - (v - k) ||^2                (ttt.py:970/975)

so the "prediction" seen by the L2 loss is LN(Z), not Z, and the gradient that
multiplies the key in the outer product is pushed back through the LN Jacobian
(ttt.py:478-505):

    Zhat   = (Z - mu) / std
    y      = gamma * Zhat + beta                          # the prediction
    gout   = y - target                                   # ttt.py:493
    gxhat  = gamma * gout                                 # ttt.py:494
    g      = (1/D)(D*gxhat - sum(gxhat) - Zhat*sum(gxhat*Zhat)) / std      # ttt.py:495-503
           = P(Zhat) gxhat / std ,  P(Zhat) = I - 11^T/D - Zhat Zhat^T/D

and the fast-weight update is  W <- W - eta * k g^T  (ttt.py:987).

Substituting y = gamma*Zhat + beta splits `gout` into a prediction part and a
target part:

    gout = [gamma * Zhat]  +  [beta - target]
            ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
            depends on W      independent of W

Only the first bracket carries W.  It is the *sole* source of the contraction
W(I - eta k k^T) in the linear-no-LN limit: with gamma=1, beta=0, LN=identity,
g = Z - target = W^T k + b - (v-k) and W <- W - eta k (k^T W)^T - eta k (b-v+k)^T
  = W(I - eta k k^T) + eta k (v - k - b)^T .
So "delete the error-correction term" == "delete the W-dependent part of gout".

Three defensible ways to do that under a fused LN; all are implemented, the
first is the default:

  hebbian_mode='drop_pred'  (PRIMARY)
        gout := -target.
        Deletes the whole prediction y = gamma*Zhat + beta.  This is the exact
        analogue of "zero out the prediction slot of the delta rule", it reduces
        *exactly* to W <- W + eta v k^T in the LN->identity limit, and it is the
        most conservative reading of "prediction-dependent part".
        Residual W-dependence: the LN Jacobian P(Zhat)/std is still evaluated at
        the live Z.  That factor is a rank-(D-2) projection times a scalar
        1/std; it rescales and deflates the write but it is NOT an error
        correction (it does not depend on the target, and it cannot produce the
        k k^T contraction, which requires gout to be linear in W^T k).

  hebbian_mode='drop_w'     (ALTERNATIVE, --hebbian-mode drop_w)
        gout := beta - target.
        Deletes only gamma*Zhat, i.e. exactly and only the tensor factor that
        carries W.  Differs from 'drop_pred' by the constant -beta*(...)/std
        write.  For the released checkpoints beta = ttt_norm_bias, initialised
        to 0 (ttt.py:698-699) and typically small, so the two agree closely; we
        report both so the claim does not rest on the choice.

  hebbian_mode='pure'       (ALTERNATIVE, --hebbian-mode pure)
        g := gamma * (-target)   (no LN projection, no 1/std).
        Removes *all* dependence of the update on the fast weights, so the layer
        becomes an exact outer-product (Hebbian) associative memory
        W <- W + eta k (gamma*(v-k))^T .  Use this one when the claim to be
        tested is literally "the layer is a Hebbian memory"; use 'drop_pred'
        when the claim is "the error-correction term does no work".

CONSEQUENCE THAT THE EXPERIMENT MUST NOT MISS
---------------------------------------------
Under this fused LN the error-correction term is not merely *small* when
eta*||k||^2 is small -- for channel-uniform `ttt_norm_weight` it is EXACTLY ZERO
at every step size.  Substituting gout_pred = gamma * Zhat into ttt.py:495-503:

    P(Zhat)[gamma^2 * Zhat] = gamma^2*Zhat - mean(gamma^2*Zhat) 1
                              - Zhat * mean(gamma^2 * Zhat^2)

and for gamma^2 = c (constant across channels) this is
    c*Zhat - c*mean(Zhat)*1 - Zhat*c*mean(Zhat^2) = c*Zhat - 0 - c*Zhat = 0
because LayerNorm guarantees mean(Zhat)=0 and mean(Zhat^2)=1.

What survives is proportional to the channel-to-channel VARIATION of gamma^2:
with gamma^2 = c(1+delta), mean(delta)=0,

    P(Zhat)[gamma^2 Zhat] = c * P(Zhat)[delta * Zhat]  = O(c * std(delta) * sqrt(D))

so the natural dimensionless control parameter for "does the delta term do any
work" is

    rho_delta  ~  ||gamma||_rms * CV(gamma^2) * sqrt(head_dim) / ||v - k||

NOT eta*||k||^2.  Indeed ||Z|| (and hence ||k||) cancels out of the ratio
entirely: the prediction is LN-bounded at ||gamma||*sqrt(D) while the target
v - k is not, so *increasing* ||k|| makes the update MORE Hebbian, not less.
This is measured directly by `delta_fraction_probe` below and cross-checked in
validate_arms.py.  The claim of the paper survives -- the layer really is a
Hebbian outer-product write -- but the stated reason (small eta*||k||^2) is not
the operative mechanism in the reference implementation.

TTT-MLP
-------
The delta-rule theory applies to the *final* fast-weight layer W2 (ttt.py:1162):

    W2 <- W2 - eta * X2^T grad_l_wrt_Z2 ,   X2 = gelu(k W1 + b1)  (ttt.py:1129)

so the effective key is the post-GELU hidden activation X2 (width 4*head_dim),
NOT k, and the relevant small-step quantity is eta*||X2||^2, not eta*||k||^2.
Patching `ln_fused_l2_bwd` gives the intended arm on W2 exactly.  It also
propagates through the chain rule at ttt.py:1139 to W1.  That is deliberate and
correct (the arms ablate the *inner loss gradient*), but note the honest caveat:
the W1 update of TTT-MLP is a backprop-through-GELU term and is never a pure
outer-product rule in any arm, because grad_l_wrt_Z1 still contains W2_init and
gelu'(Z1).  Only the W2 update is a clean delta rule.

CHUNK-BOUNDARY STATE
--------------------
`scan` (ttt.py:431-458) carries params_dict across mini-batches.  Per mini-batch
the state is advanced once, at the last token, using `last_eta` (ttt.py:985,
1156) = eta[:, :, -1, :, None] = token_eta[K-1] * ttt_lr_eta[:] -- the last row
of the [K,K] eta matrix, i.e. every token's lr weighted by the *last* token's
token_idx factor.  Within a mini-batch the per-token weights W(t) are never
materialised in the dual form; they are folded into Z1_bar via the causal
`Attn1` matrix (ttt.py:979-983).  Both the dual output path and the carried
state are functions of grad_l_wrt_Z*, hence both are covered by the patch.
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
    """Coefficient of variation of gamma^2 across channels, averaged over heads.

    THE control parameter for the error-correction term under a fused LN.  With
    P(Zhat) u = u - mean(u) 1 - Zhat (Zhat.u)/D  (ttt.py:495-503), the
    prediction part of grad_output contributes

        P(Zhat)[gamma^2 * Zhat] / std

    and for CHANNEL-UNIFORM gamma this is IDENTICALLY ZERO, because
    mean(Zhat) = 0 and (Zhat.Zhat)/D = 1.  What survives is proportional to the
    channel-to-channel *variation* of gamma^2 -- not to eta*||k||^2, and not to
    ||Z|| (hence not to ||k||) at all.
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
