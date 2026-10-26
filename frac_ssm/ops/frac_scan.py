"""Public fractional scan API."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch

from .registry import IMPLS, available, load


_LOGGER = logging.getLogger(__name__)
_IMPL_ENV = "FRAC_SSM_IMPL"
_BACKEND_BY_MODULE = {module: backend for backend, module in IMPLS.items()}
_selection_logged = False


def cute_supported(*_args, **_kwargs) -> tuple[bool, str | None]:
    """Return whether the CuTe backend is available in this environment."""
    return available("cute_sm90")


def _log_selection(backend: str, reason: str) -> None:
    global _selection_logged
    if not _selection_logged:
        _LOGGER.info("Using frac_scan backend %r (%s)", backend, reason)
        _selection_logged = True


@torch.compiler.disable
def _select_frac_scan(impl: str | None) -> Callable:
    if impl is not None:
        module = load(impl)
        reason = "selected by the impl argument"
    elif (env_impl := os.environ.get(_IMPL_ENV)) is not None:
        module = load(env_impl)
        reason = f"selected by {_IMPL_ENV}"
    else:
        cute_ok, cute_reason = cute_supported()
        module = load(None)
        if cute_ok:
            reason = "automatically selected because CuTe is available"
        else:
            reason = f"automatic Triton fallback: {cute_reason}"

    backend = _BACKEND_BY_MODULE[module.__name__]
    _log_selection(backend, reason)
    return module.frac_scan


def frac_scan(
    u,
    c,
    kappa,
    rho,
    chunk_size,
    initial_state=None,
    return_final_state=False,
    cu_seqlens=None,
    impl=None,
):
    """Run the fractional scan with the selected backend.

    Backend selection follows ``impl`` > ``FRAC_SSM_IMPL`` > automatic
    selection. Variable-length sequences are reserved for a future release.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens is not supported yet")

    fn = _select_frac_scan(impl)
    return fn(
        u,
        c,
        kappa,
        rho,
        chunk_size,
        initial_state,
        return_final_state,
    )


__all__ = ["cute_supported", "frac_scan"]
