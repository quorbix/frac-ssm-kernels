"""CuTe kernels for the fractional scan."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .chunk_cumsum import _frac_chunk_cumsum_bwd, _frac_chunk_cumsum_fwd
from .chunk_state import _frac_chunk_state_bwd, _frac_chunk_state_fwd
from .state_passing import _frac_state_passing_bwd, _frac_state_passing_fwd
from .chunk_fused_readout import _frac_chunk_fused_readout_bwd, _frac_chunk_fused_readout_fwd


@torch.library.custom_op("frac_ssm::frac_scan_fwd_cute", mutates_args=())
def _frac_scan_fwd_op(
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    rho: torch.Tensor,
    chunk_size: int,
    initial_state: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    log_rho = torch.log(rho.clamp(min=1e-6))
    log_cum_rho, chunk_decay = _frac_chunk_cumsum_fwd(log_rho, chunk_size)
    states = _frac_chunk_state_fwd(
        u,
        kappa,
        log_cum_rho,
        chunk_size,
        state_fp32=u.dtype != torch.bfloat16,
    )
    states_acc, final_state = _frac_state_passing_fwd(
        states,
        chunk_decay,
        initial_state,
    )
    h = _frac_chunk_fused_readout_fwd(
        u,
        c,
        kappa,
        states_acc,
        log_cum_rho,
        chunk_size,
    )
    return h, final_state, log_cum_rho, chunk_decay, states_acc


@_frac_scan_fwd_op.register_fake
def _(u, c, kappa, rho, chunk_size, initial_state):
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    states_dtype = u.dtype if u.dtype == torch.bfloat16 else torch.float32
    return (
        torch.empty_like(u),
        torch.empty(B, R, M, device=u.device, dtype=states_dtype),
        torch.empty(B, L, M, device=u.device, dtype=torch.float32),
        torch.empty(B, nchunks, M, device=u.device, dtype=torch.float32),
        torch.empty(
            B, nchunks, R, M, device=u.device, dtype=states_dtype,
        ),
    )


@torch.library.custom_op("frac_ssm::frac_scan_bwd_cute", mutates_args=())
def _frac_scan_bwd_op(
    d_h: torch.Tensor,
    d_final_state: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    rho: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_decay: torch.Tensor,
    states_acc: torch.Tensor,
    chunk_size: int,
    has_initial_state: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    d_u_scan, d_c, d_kappa_scan, d_states_acc, d_lcr_scan = (
        _frac_chunk_fused_readout_bwd(
            d_h,
            u,
            c,
            kappa,
            states_acc,
            log_cum_rho,
            chunk_size,
        )
    )
    d_states, d_chunk_decay, d_initial_state = _frac_state_passing_bwd(
        d_states_acc,
        d_final_state,
        states_acc,
        chunk_decay,
        has_initial_state,
    )
    d_u_state, d_kappa_state, d_lcr_state = _frac_chunk_state_bwd(
        d_states,
        u,
        kappa,
        log_cum_rho,
        chunk_size,
    )

    d_u = (d_u_scan + d_u_state).to(u.dtype)
    d_kappa = d_kappa_scan + d_kappa_state
    d_log_rho = _frac_chunk_cumsum_bwd(
        d_lcr_scan + d_lcr_state,
        d_chunk_decay,
        chunk_size,
    )
    rho_clamped = rho.clamp(min=1e-6)
    d_rho = torch.where(
        rho >= 1e-6,
        d_log_rho / rho_clamped,
        torch.zeros_like(d_log_rho),
    )
    if d_initial_state is None:
        d_initial_state = torch.empty(
            0, device=u.device, dtype=torch.float32,
        )
    return d_u, d_c, d_kappa, d_rho, d_initial_state


@_frac_scan_bwd_op.register_fake
def _(
    d_h,
    d_final_state,
    u,
    c,
    kappa,
    rho,
    log_cum_rho,
    chunk_decay,
    states_acc,
    chunk_size,
    has_initial_state,
):
    B, _, R = u.shape
    M = c.shape[-1]
    d_initial_shape = (B, R, M) if has_initial_state else (0,)
    return (
        torch.empty_like(u),
        torch.empty_like(c),
        torch.empty_like(kappa),
        torch.empty_like(rho),
        torch.empty(
            d_initial_shape, device=u.device,
            dtype=states_acc.dtype if has_initial_state else torch.float32,
        ),
    )


class FracChunkScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, c, kappa, rho, chunk_size, initial_state=None):
        assert rho.dtype == torch.float32, (
            "rho must be float32 for numerical stability"
        )
        if initial_state is not None:
            assert initial_state.dtype == torch.float32, (
                "initial_state must be float32"
            )

        u = u.contiguous()
        c = c.contiguous()
        kappa = kappa.contiguous()
        rho = rho.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()

        h, final_state, log_cum_rho, chunk_decay, states_acc = (
            _frac_scan_fwd_op(
                u,
                c,
                kappa,
                rho,
                chunk_size,
                initial_state,
            )
        )
        ctx.save_for_backward(
            u,
            c,
            kappa,
            rho,
            log_cum_rho,
            chunk_decay,
            states_acc,
        )
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = initial_state is not None
        return h, final_state

    @staticmethod
    def backward(ctx, d_h, d_final_state):
        (
            u,
            c,
            kappa,
            rho,
            log_cum_rho,
            chunk_decay,
            states_acc,
        ) = ctx.saved_tensors

        if d_final_state is None:
            d_final_state = torch.zeros_like(states_acc[:, 0])

        d_u, d_c, d_kappa, d_rho, d_initial_state = _frac_scan_bwd_op(
            d_h.contiguous(),
            d_final_state.contiguous(),
            u,
            c,
            kappa,
            rho,
            log_cum_rho,
            chunk_decay,
            states_acc,
            ctx.chunk_size,
            ctx.has_initial_state,
        )
        if not ctx.has_initial_state:
            d_initial_state = None
        return d_u, d_c, d_kappa, d_rho, None, d_initial_state


def frac_scan(
    u,
    c,
    kappa,
    rho,
    chunk_size,
    initial_state=None,
    return_final_state=False,
):
    """
    Fractional scan using CuTe kernels with a custom backward.

    Args:
        u: [B, L, R] - input features
        c: [B, L, M] - read mixing coefficients
        kappa: [B, L, M] - write mixing coefficients
        rho: [B, L, M] - decay factors
        chunk_size: int
        initial_state: [B, R, M] or None
        return_final_state: bool

    Returns:
        h: [B, L, R] - output
        final_state: [B, R, M] (only if return_final_state=True)
    """
    assert isinstance(chunk_size, int) and chunk_size in (32, 64, 128), (
        f"chunk_size must be 32, 64, or 128, got {chunk_size}"
    )
    h, final_state = FracChunkScanFunction.apply(
        u, c, kappa, rho, chunk_size, initial_state,
    )
    if return_final_state:
        return h, final_state
    return h


__all__ = ["FracChunkScanFunction", "frac_scan"]
