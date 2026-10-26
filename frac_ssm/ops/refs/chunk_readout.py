import math

import torch
import torch.nn.functional as F


def chunk_readout_ref(
    u, c, states_acc, W, log_cum_rho, chunk_size, upcast=False,
):
    """PyTorch reference for chunk_readout.

    If upcast, evaluate the decay in float64 and the contractions in float32.
    Otherwise, use the input dtype for the contractions.
    """
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    L_padded = nchunks * chunk_size

    if L_padded > L:
        pad_len = L_padded - L
        u = F.pad(u, (0, 0, 0, pad_len), value=0.0)
        c = F.pad(c, (0, 0, 0, pad_len), value=0.0)
        log_cum_rho = F.pad(
            log_cum_rho, (0, 0, 0, pad_len), value=0.0,
        )

    u_chunked = u.view(B, nchunks, chunk_size, R)
    c_chunked = c.view(B, nchunks, chunk_size, M)
    log_cum_rho_chunked = log_cum_rho.view(B, nchunks, chunk_size, M)

    lcr_dtype = torch.float64 if upcast else torch.float32
    dot_dtype = torch.float32 if upcast else u.dtype
    decay = torch.exp(log_cum_rho_chunked.to(lcr_dtype)).float()
    c_decay = (c_chunked.float() * decay).to(dot_dtype)

    h_inter = torch.einsum("bntm,bnrm->bntr", c_decay, states_acc.to(dot_dtype))
    h_intra = torch.einsum("bntk,bnkr->bntr", W.to(dot_dtype), u_chunked.to(dot_dtype))
    return (h_inter + h_intra).reshape(B, L_padded, R)[:, :L, :]
