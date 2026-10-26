import math

import torch
import torch.nn.functional as F


def chunk_cumsum_ref(
    log_rho: torch.Tensor,  # (B, L, M)
    chunk_size: int,
    upcast: bool = False,
) -> tuple[
    torch.Tensor,  # (B, L, M)
    torch.Tensor,  # (B, ceil(L / chunk_size), M)
]:
    """PyTorch reference for chunk_cumsum.

    If upcast, compute in float64 and cast the outputs back to the input dtype.
    """
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size

    dtype_og = log_rho.dtype
    x = log_rho.double() if upcast else log_rho.float()
    if L_padded > L:
        x = F.pad(x, (0, 0, 0, L_padded - L))
    x = x.view(B, nchunks, chunk_size, M)

    log_cum_rho = torch.cumsum(x, dim=2)
    chunk_decay = log_cum_rho[:, :, -1].contiguous()
    log_cum_rho = log_cum_rho.reshape(B, L_padded, M)[:, :L, :]
    if upcast:
        log_cum_rho = log_cum_rho.to(dtype=dtype_og)
        chunk_decay = chunk_decay.to(dtype=dtype_og)
    return log_cum_rho, chunk_decay
