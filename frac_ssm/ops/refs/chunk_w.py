import math

import torch
import torch.nn.functional as F


def chunk_w_ref(c, kappa, log_cum_rho, chunk_size, upcast=False):
    """PyTorch reference for chunk_w.

    If upcast, compute in float64 and cast the output back to float32.
    """
    B, L, M = log_cum_rho.shape
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size
    compute_dtype = torch.float64 if upcast else torch.float32

    if L < L_padded:
        pad_len = L_padded - L
        c = F.pad(c, (0, 0, 0, pad_len, 0, 0), value=0.0)
        kappa = F.pad(kappa, (0, 0, 0, pad_len, 0, 0), value=0.0)
        last = log_cum_rho[:, L - 1:L, :]
        log_cum_rho = torch.cat(
            [log_cum_rho, last.expand(-1, pad_len, -1)], dim=1,
        )

    c_chunked = c.view(B, nchunks, chunk_size, M).to(compute_dtype)
    kappa_chunked = kappa.view(B, nchunks, chunk_size, M).to(compute_dtype)
    lcr = log_cum_rho.view(B, nchunks, chunk_size, M).to(compute_dtype)

    diff = lcr.unsqueeze(3) - lcr.unsqueeze(2)  # [B, nchunks, t, k, M]
    positions = torch.arange(chunk_size, device=lcr.device)
    noncausal = positions[:, None] < positions[None, :]
    # Mask before exponentiating to avoid overflow above the diagonal.
    decay = torch.exp(diff.masked_fill(noncausal[..., None], float('-inf')))
    W = (c_chunked.unsqueeze(3) * decay * kappa_chunked.unsqueeze(2)).sum(-1)

    return W.float()
