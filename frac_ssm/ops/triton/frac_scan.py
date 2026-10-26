"""
Triton kernels for the fractional scan.

Combines:
- Kernel 1: chunk_cumsum (Triton)
- Kernel 2: chunk_state (Triton)
- Kernel 3: state_passing (Triton)
- Kernel 4: chunk_W (Triton)
- Kernel 5: chunk_readout (Triton)
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.runtime._allocation as _triton_alloc

# Patch NullAllocator to use fallback allocation in autograd threads
# (ContextVar-based allocator doesn't propagate to PyTorch autograd threads)
_triton_alloc.NullAllocator.__call__ = lambda self, size, align, stream: \
    torch.empty(size, device="cuda", dtype=torch.int8)

from .chunk_cumsum import _frac_chunk_cumsum_bwd, _frac_chunk_cumsum_fwd
from .chunk_readout import _frac_chunk_readout_bwd, _frac_chunk_readout_fwd
from .chunk_state import _frac_chunk_state_bwd, _frac_chunk_state_fwd
from .chunk_w import _frac_chunk_W_bwd, _frac_chunk_W_fwd
from .state_passing import _frac_state_passing_bwd, _frac_state_passing_fwd

# ============================================================================
# Autograd wrapper
# ============================================================================

class FracChunkScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, c, kappa, rho, chunk_size, initial_state=None):

        assert rho.dtype == torch.float32, "rho must be float32 for numerical stability"
        if initial_state is not None:
            assert initial_state.dtype == torch.float32, "initial_state must be float32"
        log_rho = torch.log(rho.clamp(min=1e-6))
        log_cum_rho, chunk_decay = _frac_chunk_cumsum_fwd(log_rho, chunk_size)
        states = _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32=u.dtype != torch.bfloat16)
        states_acc, final_state = _frac_state_passing_fwd(states, chunk_decay, initial_state)
        W = _frac_chunk_W_fwd(c, kappa, log_cum_rho, chunk_size)
        h = _frac_chunk_readout_fwd(
            u, c, states_acc, W, log_cum_rho, chunk_size,
        )

        ctx.save_for_backward(u, c, kappa, rho, log_cum_rho, chunk_decay, states, states_acc, W)
        ctx.chunk_size = chunk_size
        ctx.has_initial_state = initial_state is not None

        return h, final_state

    @staticmethod
    def backward(ctx, d_h, d_final_state):
        u, c, kappa, rho, log_cum_rho, chunk_decay, states, states_acc, W = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        has_initial_state = ctx.has_initial_state

        assert rho.dtype == torch.float32, "rho must be float32 for numerical stability"
        if d_final_state is None:
            d_final_state = torch.zeros_like(states_acc[:, 0])

        # Backward: chunk_readout
        d_u_scan, d_c_scan, d_states_acc, d_lcr_scan, d_W = _frac_chunk_readout_bwd(
            d_h, u, c, states_acc, W, log_cum_rho, chunk_size
        )

        # Backward: chunk_W
        d_c_W, d_kappa_W, d_lcr_W = _frac_chunk_W_bwd(
            d_W, c, kappa, log_cum_rho, chunk_size
        )

        # Combine scan and W gradients for log_cum_rho
        d_log_cum_rho = d_lcr_scan + d_lcr_W

        # Backward: state_passing - Triton
        d_states, d_chunk_decay, d_initial_state = _frac_state_passing_bwd(
            d_states_acc, d_final_state, states_acc, chunk_decay, has_initial_state
        )

        # Backward: chunk_state - Triton
        d_u_state, d_kappa_state, d_log_cum_rho_state = _frac_chunk_state_bwd(
            d_states, u, kappa, log_cum_rho, chunk_size
        )

        # Backward: cumsum - Triton
        d_log_cum_rho_total = d_log_cum_rho + d_log_cum_rho_state
        rho_clamped = rho.clamp(min=1e-6)
        d_log_rho = _frac_chunk_cumsum_bwd(d_log_cum_rho_total, d_chunk_decay, chunk_size)
        d_rho = torch.where(rho >= 1e-6, d_log_rho / rho_clamped, torch.zeros_like(d_log_rho))

        # Aggregate gradients
        d_u = (d_u_scan + d_u_state).to(u.dtype)
        d_c = d_c_scan + d_c_W
        d_kappa = d_kappa_W + d_kappa_state
        d_rho = d_rho

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
    Fractional scan using Triton kernels with a custom backward.

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
    h, final_state = FracChunkScanFunction.apply(u, c, kappa, rho, chunk_size, initial_state)
    if return_final_state:
        return h, final_state
    return h
