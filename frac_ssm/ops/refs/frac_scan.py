"""PyTorch reference for the fractional scan."""

import torch

from .chunk_cumsum import chunk_cumsum_ref
from .chunk_state import chunk_state_ref
from .state_passing import state_passing_ref
from .chunk_w import chunk_w_ref
from .chunk_readout import chunk_readout_ref


def frac_scan_ref(
    u,
    c,
    kappa,
    rho, # this is rho hat from the paper
    chunk_size,
    initial_state=None,
    return_final_state=False,
    upcast=False,
):
    """PyTorch reference for the end-to-end fractional chunked scan.

    If upcast, evaluate the logarithm in float64 and use each stage's upcast
    reference while preserving the float32 intermediate interfaces.
    """
    assert rho.dtype == torch.float32, "rho must be float32 for numerical stability"
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be float32"

    if upcast:
        log_rho = torch.log(rho.double().clamp(min=1e-6)).float()
    else:
        log_rho = torch.log(rho.clamp(min=1e-6))
    log_cum_rho, chunk_decay = chunk_cumsum_ref(
        log_rho, chunk_size, upcast=upcast,
    )
    states = chunk_state_ref(
        u, kappa, log_cum_rho, chunk_size,
        state_fp32=upcast or u.dtype != torch.bfloat16, upcast=upcast,
    )
    states_acc, final_state = state_passing_ref(
        states, chunk_decay, initial_state, upcast=upcast,
    )
    W = chunk_w_ref(
        c, kappa, log_cum_rho, chunk_size, upcast=upcast,
    )
    h = chunk_readout_ref(
        u, c, states_acc, W, log_cum_rho, chunk_size, upcast=upcast,
    )

    if return_final_state:
        return h, final_state
    return h


frac_scan = frac_scan_ref


__all__ = [
    "frac_scan",
    "frac_scan_ref",
]
