import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ],
    key=['chunk_size', 'M','M_BLOCK'],
)
@triton.jit(do_not_specialize=["L"])
def _frac_chunk_cumsum_fwd_kernel(rho_ptr, rho_cumsum_ptr, chunk_decay_ptr,
                                  rho_stride_b, rho_stride_l, rho_stride_m,
                                  rho_cumsum_stride_b, rho_cumsum_stride_l, rho_cumsum_stride_m,
                                  chunk_decay_stride_b, chunk_decay_stride_n, chunk_decay_stride_m,
                                  L, M:tl.constexpr, M_BLOCK:tl.constexpr, chunk_size:tl.constexpr):
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid_chunk = tl.program_id(axis=1).to(tl.int64)

    bos = pid_chunk * chunk_size
    rho_ptr += pid_batch * rho_stride_b + bos * rho_stride_l
    rho_cumsum_ptr += pid_batch * rho_cumsum_stride_b + bos * rho_cumsum_stride_l
    chunk_decay_ptr += pid_batch * chunk_decay_stride_b + pid_chunk * chunk_decay_stride_n

    offset_c = tl.arange(0, chunk_size)
    mask_c = (bos + offset_c) < L

    offset_m = tl.arange(0, M_BLOCK)
    mask_m = offset_m < M

    x = tl.load(rho_ptr + (offset_c[:, None] * rho_stride_l) + (offset_m[None, :] * rho_stride_m), mask=mask_c[:, None] & mask_m[None, :], other=0.0).to(tl.float32)

    chunk_decay = tl.sum(x, axis=0)
    x = tl.cumsum(x, axis=0)

    tl.store(rho_cumsum_ptr + (offset_c[:, None] * rho_cumsum_stride_l) + (offset_m[None, :] * rho_cumsum_stride_m), x, mask=mask_c[:, None] & mask_m[None, :])
    tl.store(chunk_decay_ptr + (offset_m * chunk_decay_stride_m), chunk_decay, mask=mask_m)
    
def _frac_chunk_cumsum_fwd(
    log_rho: torch.Tensor,  # (B, L, M)
    chunk_size: int,
) -> tuple[
    torch.Tensor,  # (B, L, M)
    torch.Tensor,  # (B, ceil(L / chunk_size), M)
]:
    B, L, M = log_rho.shape # (B, L, M)
    nchunks = math.ceil(L / chunk_size)

    log_cumsum_rho = torch.empty((B, L, M), dtype=torch.float32, device=log_rho.device) # (B, L, M)
    chunk_decay = torch.empty((B, nchunks, M), dtype=torch.float32, device=log_rho.device) # (B, N_c, M)
    M_BLOCK = triton.next_power_of_2(M)
    grid = (B, nchunks)
    _frac_chunk_cumsum_fwd_kernel[grid](log_rho, log_cumsum_rho, chunk_decay,
                                        log_rho.stride(0), log_rho.stride(1), log_rho.stride(2),
                                        log_cumsum_rho.stride(0), log_cumsum_rho.stride(1), log_cumsum_rho.stride(2),
                                        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
                                        L, M, M_BLOCK, chunk_size)

    return log_cumsum_rho, chunk_decay

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in [1, 2, 3, 4]
    ],
    key=['chunk_size', 'M', 'M_BLOCK'],
)
@triton.jit(do_not_specialize=["L"])
def _frac_chunk_cumsum_bwd_kernel(
    d_log_cum_rho_ptr, d_chunk_decay_ptr, d_log_rho_ptr,
    dlcr_stride_b, dlcr_stride_l, dlcr_stride_m,
    dcd_stride_b, dcd_stride_n, dcd_stride_m,
    dlr_stride_b, dlr_stride_l, dlr_stride_m,
    L, M: tl.constexpr, M_BLOCK: tl.constexpr, chunk_size: tl.constexpr
):
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid_chunk = tl.program_id(axis=1).to(tl.int64)

    bos = pid_chunk * chunk_size
    d_log_cum_rho_ptr += pid_batch * dlcr_stride_b + bos * dlcr_stride_l
    d_chunk_decay_ptr += pid_batch * dcd_stride_b + pid_chunk * dcd_stride_n
    d_log_rho_ptr += pid_batch * dlr_stride_b + bos * dlr_stride_l

    offset_c = tl.arange(0, chunk_size)
    offset_m = tl.arange(0, M_BLOCK)
    mask_c = (bos + offset_c) < L
    mask_m = offset_m < M

    x = tl.load(d_log_cum_rho_ptr + offset_c[:, None] * dlcr_stride_l + offset_m[None, :] * dlcr_stride_m,
                mask=mask_c[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
    dcd = tl.load(d_chunk_decay_ptr + offset_m * dcd_stride_m, mask=mask_m, other=0.0).to(tl.float32)

    # Reverse cumsum, then add dcd (equivalent to adding dcd to last row before cumsum)
    d_log_rho = tl.cumsum(x, axis=0, reverse=True) + dcd[None, :]

    tl.store(
        d_log_rho_ptr + offset_c[:, None] * dlr_stride_l + offset_m[None, :] * dlr_stride_m,
        d_log_rho, mask=mask_c[:, None] & mask_m[None, :]
    )


def _frac_chunk_cumsum_bwd(
    d_log_cum_rho: torch.Tensor,  # (B, L, M)
    d_chunk_decay: torch.Tensor,  # (B, ceil(L / chunk_size), M)
    chunk_size: int,
) -> torch.Tensor:  # (B, L, M)
    B, L, M = d_log_cum_rho.shape
    nchunks = math.ceil(L / chunk_size)

    d_log_rho = torch.empty((B, L, M), dtype=torch.float32, device=d_log_cum_rho.device)
    M_BLOCK = triton.next_power_of_2(M)
    grid = (B, nchunks)

    _frac_chunk_cumsum_bwd_kernel[grid](
        d_log_cum_rho, d_chunk_decay, d_log_rho,
        d_log_cum_rho.stride(0), d_log_cum_rho.stride(1), d_log_cum_rho.stride(2),
        d_chunk_decay.stride(0), d_chunk_decay.stride(1), d_chunk_decay.stride(2),
        d_log_rho.stride(0), d_log_rho.stride(1), d_log_rho.stride(2),
        L, M, M_BLOCK, chunk_size
    )

    return d_log_rho


# ============================================================================
# Public API and autograd wrappers
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_cumsum_fwd", mutates_args=())
def _chunk_cumsum_fwd_op(
    log_rho: torch.Tensor,  # (B, L, M)
    chunk_size: int,
) -> tuple[
    torch.Tensor,  # (B, L, M)
    torch.Tensor,  # (B, ceil(L / chunk_size), M)
]:
    return _frac_chunk_cumsum_fwd(log_rho, chunk_size)


@_chunk_cumsum_fwd_op.register_fake
def _(log_rho, chunk_size):
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)
    return (
        torch.empty(B, L, M, dtype=torch.float32, device=log_rho.device),
        torch.empty(B, nchunks, M, dtype=torch.float32, device=log_rho.device),
    )


@torch.library.custom_op("frac_ssm::chunk_cumsum_bwd", mutates_args=())
def _chunk_cumsum_bwd_op(
    d_log_cum_rho: torch.Tensor,  # (B, L, M)
    d_chunk_decay: torch.Tensor,  # (B, ceil(L / chunk_size), M)
    chunk_size: int,
) -> torch.Tensor:  # (B, L, M)
    return _frac_chunk_cumsum_bwd(d_log_cum_rho, d_chunk_decay, chunk_size)


@_chunk_cumsum_bwd_op.register_fake
def _(d_log_cum_rho, d_chunk_decay, chunk_size):
    B, L, M = d_log_cum_rho.shape
    return torch.empty(B, L, M, dtype=torch.float32, device=d_log_cum_rho.device)


def _chunk_cumsum_setup_ctx(ctx, inputs, output):
    log_rho, chunk_size = inputs
    ctx.chunk_size = chunk_size
    ctx.input_dtype = log_rho.dtype


def _chunk_cumsum_backward(ctx, d_log_cum_rho, d_chunk_decay):
    d_log_rho = _chunk_cumsum_bwd_op(
        d_log_cum_rho.contiguous(), d_chunk_decay.contiguous(),
        ctx.chunk_size,
    )
    return d_log_rho.to(ctx.input_dtype), None


_chunk_cumsum_fwd_op.register_autograd(_chunk_cumsum_backward, setup_context=_chunk_cumsum_setup_ctx)


def chunk_cumsum(log_rho, chunk_size):
    """Per-chunk log-cumsum.

    Args:
        log_rho: (B, L, M), float32 on CUDA.
        chunk_size: int.

    Returns:
        log_cum_rho: (B, L, M) fp32 — within-chunk cumulative sum of log_rho.
        chunk_decay: (B, nchunks, M) fp32 — sum of log_rho over each full chunk.
    """
    assert log_rho.dim() == 3, f"log_rho must be 3D (B, L, M), got {log_rho.dim()}D"
    assert log_rho.is_cuda, "log_rho must be on CUDA"
    assert log_rho.dtype == torch.float32, f"log_rho dtype must be float32, got {log_rho.dtype}"
    assert (
        isinstance(chunk_size, int)
        and chunk_size > 0
        and (chunk_size & (chunk_size - 1)) == 0
    ), f"chunk_size must be a positive power of two, got {chunk_size}"
    return _chunk_cumsum_fwd_op(log_rho, chunk_size)
