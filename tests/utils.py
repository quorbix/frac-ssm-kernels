"""
Shared test utilities
"""

import math

import torch
import torch.nn.functional as F


# Frac SSM defaults
_TAU = (1.0, 65536.0)
_DT = (1e-4, 1.0)
_ALPHA = (0.20, 0.95)
_LAM = (0.25, 4.0)


# ============================================================================
# Input generators
# ============================================================================

def random_log_rho(B, L, M, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(
        B, L, M, device=device, dtype=torch.float32, generator=g,
    ).clamp(0.01, 0.99).log()


def model_like_log_rho(B, L, M, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    tau = torch.linspace(math.log(_TAU[0]), math.log(_TAU[1]), M,
                         device=device, dtype=torch.float32).exp()

    raw = torch.randn(B, L, device=device, dtype=torch.float32, generator=g)
    raw = F.avg_pool1d(raw[:, None, :], 4, stride=1, padding=2)[..., :L].squeeze(1)
    dt = F.softplus(raw).clamp(*_DT)

    def _u(lo, hi):
        return lo + (hi - lo) * torch.rand(B, L, device=device, dtype=torch.float32, generator=g)
    alpha, lam = _u(*_ALPHA), _u(*_LAM)
    lambda_factor = lam ** (1.0 / alpha)

    rho = torch.exp(-dt[..., None] * lambda_factor[..., None] / tau).clamp(1e-6, 1.0 - 1e-6)
    return rho.log()


LOG_RHO_GENS = {"random": random_log_rho, "model_like": model_like_log_rho}


# ============================================================================
# Tolerance
# ============================================================================


def is_cuda_oom(exc):
    seen = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, torch.OutOfMemoryError) or "out of memory" in str(cur).lower():
            return True
        stack.extend((getattr(cur, "__cause__", None), getattr(cur, "__context__", None)))
    return False
