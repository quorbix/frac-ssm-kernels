"""Warp-specialized fused chunk readout."""

import torch

from frac_ssm.ops.cute._chunk_fused_readout.warp_specialized_bwd import _frac_chunk_fused_readout_bwd
from frac_ssm.ops.cute._chunk_fused_readout.warp_specialized_fwd import _frac_chunk_fused_readout_fwd
from frac_ssm.ops.cute.chunk_fused_readout import _aligned_contiguous, _chunk_fused_readout_setup_ctx, _validate_inputs


@torch.library.custom_op(
    "frac_ssm::chunk_fused_readout_fwd_cute_warp_specialized",
    mutates_args=(),
)
def _chunk_fused_readout_warp_specialized_fwd_op(
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    states_acc: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _frac_chunk_fused_readout_fwd(
        _aligned_contiguous(u),
        _aligned_contiguous(c),
        _aligned_contiguous(kappa),
        _aligned_contiguous(states_acc),
        _aligned_contiguous(log_cum_rho),
        chunk_size,
    )


@_chunk_fused_readout_warp_specialized_fwd_op.register_fake
def _(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return torch.empty_like(u)


@torch.library.custom_op(
    "frac_ssm::chunk_fused_readout_bwd_cute_warp_specialized",
    mutates_args=(),
)
def _chunk_fused_readout_warp_specialized_bwd_op(
    d_h: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    states_acc: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_fused_readout_bwd(
        _aligned_contiguous(d_h),
        _aligned_contiguous(u),
        _aligned_contiguous(c),
        _aligned_contiguous(kappa),
        _aligned_contiguous(states_acc),
        _aligned_contiguous(log_cum_rho),
        chunk_size,
    )


@_chunk_fused_readout_warp_specialized_bwd_op.register_fake
def _(d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size):
    return (
        torch.empty_like(u),
        torch.empty_like(c),
        torch.empty_like(kappa),
        torch.empty_like(states_acc),
        torch.empty_like(log_cum_rho),
    )


def _chunk_fused_readout_warp_specialized_backward(ctx, d_h):
    u, c, kappa, states_acc, log_cum_rho = ctx.saved_tensors
    return (
        *_chunk_fused_readout_warp_specialized_bwd_op(
            d_h,
            u,
            c,
            kappa,
            states_acc,
            log_cum_rho,
            ctx.chunk_size,
        ),
        None,
    )


_chunk_fused_readout_warp_specialized_fwd_op.register_autograd(
    _chunk_fused_readout_warp_specialized_backward,
    setup_context=_chunk_fused_readout_setup_ctx,
)


def chunk_fused_readout(u, c, kappa, states_acc, log_cum_rho, chunk_size):
    """Compute fused chunk readout with the warp-specialized implementation."""
    _validate_inputs(u, c, kappa, states_acc, log_cum_rho, chunk_size)
    return _chunk_fused_readout_warp_specialized_fwd_op(
        u,
        c,
        kappa,
        states_acc,
        log_cum_rho,
        chunk_size,
    )


__all__ = ["chunk_fused_readout"]
