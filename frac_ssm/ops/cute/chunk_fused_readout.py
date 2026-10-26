"""Persistent fused chunk readout and shared input handling."""

import math

import torch

from frac_ssm.ops.cute._chunk_fused_readout.persistent_bwd import _frac_chunk_fused_readout_bwd as _persistent_bwd
from frac_ssm.ops.cute._chunk_fused_readout.persistent_fwd import _frac_chunk_fused_readout_fwd as _persistent_fwd


def _aligned_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Return contiguous storage with the 16-byte alignment required by TMA."""
    x = x.contiguous()
    return x if x.data_ptr() % 16 == 0 else x.clone()


def _frac_chunk_fused_readout_fwd(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return _persistent_fwd(
        _aligned_contiguous(u),
        _aligned_contiguous(c),
        _aligned_contiguous(kappa),
        _aligned_contiguous(states_acc),
        _aligned_contiguous(log_cum_rho),
        chunk_size,
    )


def _frac_chunk_fused_readout_bwd(d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return _persistent_bwd(
        _aligned_contiguous(d_h),
        _aligned_contiguous(u),
        _aligned_contiguous(c),
        _aligned_contiguous(kappa),
        _aligned_contiguous(states_acc),
        _aligned_contiguous(log_cum_rho),
        chunk_size,
    )


@torch.library.custom_op("frac_ssm::chunk_fused_readout_fwd_cute", mutates_args=())
def _chunk_fused_readout_fwd_op(
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    states_acc: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _frac_chunk_fused_readout_fwd(
        u, c, kappa, states_acc, log_cum_rho, chunk_size,
    )


@_chunk_fused_readout_fwd_op.register_fake
def _(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return torch.empty_like(u)


@torch.library.custom_op("frac_ssm::chunk_fused_readout_bwd_cute", mutates_args=())
def _chunk_fused_readout_bwd_op(
    d_h: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    states_acc: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_fused_readout_bwd(
        d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size,
    )


@_chunk_fused_readout_bwd_op.register_fake
def _(d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return (
        torch.empty_like(u),
        torch.empty_like(c),
        torch.empty_like(kappa),
        torch.empty_like(states_acc),
        torch.empty_like(log_cum_rho),
    )


def _chunk_fused_readout_setup_ctx(ctx, inputs, output):
    u, c, kappa, states_acc, log_cum_rho, chunk_size = inputs
    ctx.save_for_backward(u, c, kappa, states_acc, log_cum_rho)
    ctx.chunk_size = chunk_size


def _chunk_fused_readout_backward(ctx, d_h):
    u, c, kappa, states_acc, log_cum_rho = ctx.saved_tensors
    return (
        *_chunk_fused_readout_bwd_op(
            d_h, u, c, kappa, states_acc, log_cum_rho, ctx.chunk_size,
        ),
        None,
    )


_chunk_fused_readout_fwd_op.register_autograd(
    _chunk_fused_readout_backward, setup_context=_chunk_fused_readout_setup_ctx,
)


def _validate_inputs(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    """Validate the shared input contract for fused chunk readout."""
    assert u.dim() == 3, f"u must be 3D (B, L, R), got {u.dim()}D"
    assert c.dim() == 3, f"c must be 3D (B, L, M), got {c.dim()}D"
    assert kappa.dim() == 3, f"kappa must be 3D (B, L, M), got {kappa.dim()}D"
    assert states_acc.dim() == 4, (
        f"states_acc must be 4D (B, N, R, M), got {states_acc.dim()}D"
    )
    assert log_cum_rho.dim() == 3, (
        f"log_cum_rho must be 3D (B, L, M), got {log_cum_rho.dim()}D"
    )
    assert all(x.is_cuda for x in (u, c, kappa, states_acc, log_cum_rho)), (
        "inputs must be on CUDA"
    )
    assert u.dtype in (torch.float16, torch.bfloat16), (
        f"u must be float16 or bfloat16, got {u.dtype}"
    )
    assert c.dtype == kappa.dtype and c.dtype in (u.dtype, torch.float32), (
        "c and kappa dtype must match u dtype or be float32"
    )
    assert states_acc.dtype in (u.dtype, torch.float32), (
        f"states_acc dtype must be {u.dtype} or float32, got {states_acc.dtype}"
    )
    assert log_cum_rho.dtype == torch.float32, (
        f"log_cum_rho dtype must be float32, got {log_cum_rho.dtype}"
    )
    assert isinstance(chunk_size, int) and chunk_size in (32, 64, 128), (
        f"chunk_size must be 32, 64 or 128, got {chunk_size}"
    )

    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    assert c.shape == (B, L, M) and kappa.shape == (B, L, M), (
        "c and kappa shapes must match (B, L, M)"
    )
    assert log_cum_rho.shape == (B, L, M), (
        "log_cum_rho shape must match (B, L, M)"
    )
    assert states_acc.shape == (B, nchunks, R, M), (
        "states_acc shape must be (B, nchunks, R, M)"
    )


def chunk_fused_readout(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    """Compute fused chunk readout with the persistent implementation."""
    _validate_inputs(u, c, kappa, states_acc, log_cum_rho, chunk_size)
    return _chunk_fused_readout_fwd_op(
        u, c, kappa, states_acc, log_cum_rho, chunk_size,
    )


__all__ = ["chunk_fused_readout"]
