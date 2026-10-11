"""E1: the norm-matched Hebbian arm for the inference-time ladder.

Drop-in for your harness. Where the existing Hebbian arm does
    dW = dW_additive(...)                         # corrective term deleted
the norm-matched arm does
    dW = norm_match(dW_additive, dW_full)         # same direction, full update's norm

`norm_match` works on tensors of shape [..., d_out, d_in]: the Frobenius norm is taken over
the trailing two dims, so per-head / per-layer / per-mini-batch granularity is whatever the
leading dims are. Log the returned ratio: its distribution is itself a result (see
EXPERIMENTS.md E1 for what to expect).
"""
import torch


def norm_match(dW_add: torch.Tensor, dW_full: torch.Tensor, eps: float = 1e-12):
    """Rescale dW_add so ||dW_add||_F == ||dW_full||_F over the trailing two dims.

    Returns (dW_scaled, ratio) with ratio = ||dW_full||_F / ||dW_add||_F, shape [...].
    """
    n_full = torch.linalg.matrix_norm(dW_full, ord="fro")   # [...]
    n_add = torch.linalg.matrix_norm(dW_add, ord="fro")
    ratio = n_full / (n_add + eps)
    return dW_add * ratio[..., None, None], ratio


def corrective_fraction(dW_add: torch.Tensor, dW_full: torch.Tensor, eps: float = 1e-12):
    """The quantity reported in Sec. 4.2: ||dW_full - dW_add||_F / ||dW_full||_F."""
    return torch.linalg.matrix_norm(dW_full - dW_add, ord="fro") / \
        (torch.linalg.matrix_norm(dW_full, ord="fro") + eps)


if __name__ == "__main__":
    # Self-test: synthetic updates with a known corrective share.
    torch.manual_seed(0)
    B, H, do, di = 4, 8, 64, 64
    add = torch.randn(B, H, do, di)
    corr = torch.randn(B, H, do, di)
    corr = corr / torch.linalg.matrix_norm(corr, ord="fro")[..., None, None] \
        * 0.37 * torch.linalg.matrix_norm(add, ord="fro")[..., None, None]   # 37% of ||add||
    full = add - corr
    scaled, ratio = norm_match(add, full)
    assert torch.allclose(torch.linalg.matrix_norm(scaled, ord="fro"),
                          torch.linalg.matrix_norm(full, ord="fro"), rtol=1e-5)
    # direction unchanged
    cos = (scaled * add).sum((-1, -2)) / (torch.linalg.matrix_norm(scaled, ord="fro")
                                          * torch.linalg.matrix_norm(add, ord="fro"))
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)
    print("ok; ratio mean %.4f (random-direction corrective: expect ~sqrt(1+0.37^2)=%.4f)"
          % (ratio.mean(), (1 + 0.37 ** 2) ** 0.5))
    print("corrective fraction mean %.3f" % corrective_fraction(add, full).mean())
