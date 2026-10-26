"""
Three runnable backends: 1) torch 2) triton 3) cute_sm90  (needs nvidia-cutlass-dsl and device with sm90 cc)

There are some older cute kernels at ``frac_ssm/ops/cute/variants/``. They are slower than than the cute_sm90, and aren't used as backends.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

BACKENDS = ("torch", "triton", "cute_sm90")

IMPLS: dict[str, str] = {
    "torch": "frac_ssm.ops.refs.frac_scan",
    "triton": "frac_ssm.ops.triton.frac_scan",
    "cute_sm90": "frac_ssm.ops.cute.frac_scan",
}


def available(backend: str) -> tuple[bool, str | None]:
    """
    Whether this backend can run in the current environment.
    """
    _check_backend(backend)
    if backend == "cute_sm90":
        if importlib.util.find_spec("cutlass") is None:
            return False, (
                "nvidia-cutlass-dsl is not installed; pip install frac-ssm[cute]"
            )
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA device required"
        cap = torch.cuda.get_device_capability()
        if cap != (9, 0):
            return False, f"Device with sm90 cc is required, found sm_{cap[0]}{cap[1]}"

    return True, None


def load(backend: str | None = None) -> ModuleType:
    """Load a backend module; ``None`` picks cute_sm90 if usable, else triton."""
    if backend is None:
        backend = "cute_sm90" if available("cute_sm90")[0] else "triton"
    else:
        _check_backend(backend)
        ok, reason = available(backend)
        if not ok:
            raise RuntimeError(f"Backend {backend!r} is unavailable: {reason}")
    return importlib.import_module(IMPLS[backend])


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available backends: {', '.join(BACKENDS)}"
        )
