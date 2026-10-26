"""
Frac Chunk W Forward & Backward Kernels (Kernel 4)

Computes: W[t,k] = 1[k <= t] * sum_m c[t,m] * exp(lcr[t,m] - lcr[k,m]) * kappa[k,m]
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_T": block_size_t, "BLOCK_SIZE_K": block_size_k},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_t in [32, 64]
        for block_size_k in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["chunk_size", "M"],
)
@triton.jit
def _frac_chunk_W_fwd_kernel(
    c_ptr, kappa_ptr, log_cum_rho_ptr, W_ptr,
    # c strides
    c_stride_B, c_stride_chunk, c_stride_t, c_stride_m,
    # kappa strides
    kappa_stride_B, kappa_stride_chunk, kappa_stride_k, kappa_stride_m,
    # lcr strides
    lcr_stride_B, lcr_stride_chunk, lcr_stride_pos, lcr_stride_m,
    # W strides
    W_stride_B, W_stride_chunk, W_stride_t, W_stride_k,
    # Dimensions
    L, M: tl.constexpr,
    chunk_size: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # W[t, k] = 1[k <= t] * sum_m c[t, m] * exp(lcr[t, m] - lcr[k, m]) * kappa[k, m]
    LOG2_E: tl.constexpr = 1.4426950408889634

    pid_b = tl.program_id(axis=0)
    pid_chunk = tl.program_id(axis=1)
    pid_tile = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_chunk_i64 = pid_chunk.to(tl.int64)

    num_tiles_k = tl.cdiv(chunk_size, BLOCK_SIZE_K)
    pid_t = pid_tile // num_tiles_k
    pid_k = pid_tile % num_tiles_k

    # Causal: skip tiles entirely above diagonal
    if pid_k * BLOCK_SIZE_K >= (pid_t + 1) * BLOCK_SIZE_T:
        return

    # Compiler hints
    tl.assume(pid_t >= 0)
    tl.assume(pid_k >= 0)
    tl.assume(c_stride_t > 0)
    tl.assume(c_stride_m > 0)
    tl.assume(kappa_stride_k > 0)
    tl.assume(kappa_stride_m > 0)
    tl.assume(lcr_stride_pos > 0)
    tl.assume(lcr_stride_m > 0)
    tl.assume(W_stride_t > 0)
    tl.assume(W_stride_k > 0)

    # Base pointers
    c_ptr += pid_b_i64 * c_stride_B + pid_chunk_i64 * c_stride_chunk
    kappa_ptr += pid_b_i64 * kappa_stride_B + pid_chunk_i64 * kappa_stride_chunk
    lcr_ptr = log_cum_rho_ptr + pid_b_i64 * lcr_stride_B + pid_chunk_i64 * lcr_stride_chunk
    W_ptr += pid_b_i64 * W_stride_B + pid_chunk_i64 * W_stride_chunk

    # Offsets
    offs_t = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    chunk_len = tl.minimum(chunk_size, L - pid_chunk * chunk_size)
    mask_t = offs_t < chunk_len
    mask_k = offs_k < chunk_len

    # Accumulator for W[t, k]
    acc = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_K), dtype=tl.float32)

    offs_m = tl.arange(0, M)
    c_block = tl.load(c_ptr + offs_t[:, None] * c_stride_t + offs_m[None, :] * c_stride_m,
                        mask=mask_t[:, None], other=0.0).to(tl.float32)      # [T, M]
    lcr_t_block = tl.load(lcr_ptr + offs_t[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                        mask=mask_t[:, None], other=0.0).to(tl.float32)      # [T, M]
    kappa_block = tl.load(kappa_ptr + offs_k[:, None] * kappa_stride_k + offs_m[None, :] * kappa_stride_m,
                        mask=mask_k[:, None], other=0.0).to(tl.float32)      # [K, M]
    lcr_k_block = tl.load(lcr_ptr + offs_k[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                        mask=mask_k[:, None], other=0.0).to(tl.float32)      # [K, M]

    # This per-mode loop is intended for small M; larger M needs a different design.
    for m in range(M):
        sel = (offs_m == m).to(tl.float32)
        c_m = tl.sum(c_block * sel[None, :], axis=1)
        lcr_t_m = tl.sum(lcr_t_block * sel[None, :], axis=1)
        kappa_m = tl.sum(kappa_block * sel[None, :], axis=1)
        lcr_k_m = tl.sum(lcr_k_block * sel[None, :], axis=1)

        decay = tl.math.exp2((lcr_t_m[:, None] - lcr_k_m[None, :]) * LOG2_E)
        acc += c_m[:, None] * decay * kappa_m[None, :]

    # Mask entries above the diagonal; this is a no-op for fully causal tiles
    if pid_k * BLOCK_SIZE_K < (pid_t + 1) * BLOCK_SIZE_T:
        causal_mask = offs_t[:, None] >= offs_k[None, :]
        acc = tl.where(causal_mask, acc, 0.0)

    # Store W[t, k]
    W_ptrs = W_ptr + offs_t[:, None] * W_stride_t + offs_k[None, :] * W_stride_k
    tl.store(W_ptrs, acc.to(W_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_k[None, :])


def _frac_chunk_W_fwd(c, kappa, log_cum_rho, chunk_size):
    """
    W computation: W[t,k] = 1[k <= t] * sum_m c[t,m] * exp(lcr[t,m] - lcr[k,m]) * kappa[k,m]

    Args:
        c: (B, L, M)
        kappa: (B, L, M)
        log_cum_rho: (B, L, M)
        chunk_size: int

    Returns:
        W: (B, nchunks, chunk_size, chunk_size)
    """
    B, L, M = c.shape
    nchunks = math.ceil(L / chunk_size)

    # Initialize to zero because skipped upper tiles and padded positions are not written
    W = torch.zeros((B, nchunks, chunk_size, chunk_size), dtype=torch.float32, device=c.device)

    grid = lambda META: (
        B, nchunks,
        triton.cdiv(chunk_size, META['BLOCK_SIZE_T']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_K'])
    )
    _frac_chunk_W_fwd_kernel[grid](
        c, kappa, log_cum_rho, W,
        c.stride(0), chunk_size * c.stride(1), c.stride(1), c.stride(2),
        kappa.stride(0), chunk_size * kappa.stride(1), kappa.stride(1), kappa.stride(2),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        L, M, chunk_size,
    )

    return W

# ============================================================================
# Backward kernels
# ============================================================================
# d_c[p,m] = sum_{q<=p} d_W[p,q] * exp(lcr[p,m] - lcr[q,m]) * kappa[q,m]
# d_kappa[p,m] = sum_{t>=p} d_W[t,p] * c[t,m] * exp(lcr[t,m] - lcr[p,m])
# d_lcr[p,m] = c[p,m] * d_c[p,m] - kappa[p,m] * d_kappa[p,m]
#
# Each kernel processes all M modes, reusing each d_W tile across modes.

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_PQ": block_size_pq},
            num_warps=num_warps,
            num_stages=2,
        )
        for block_size_pq in [32, 64]
        for num_warps in [4, 8]
    ],
    key=["chunk_size", "M"],
)
@triton.jit
def _frac_chunk_W_bwd_dc_kernel(
    d_W_ptr, kappa_ptr, lcr_ptr, d_c_ptr,
    dW_stride_B, dW_stride_chunk, dW_stride_t, dW_stride_k,
    ck_stride_B, ck_stride_chunk, ck_stride_pos, ck_stride_m,
    lcr_stride_B, lcr_stride_chunk, lcr_stride_pos, lcr_stride_m,
    out_stride_B, out_stride_chunk, out_stride_pos, out_stride_m,
    L, M: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_SIZE_PQ: tl.constexpr
):
    """d_c[p,m] = sum_{q<=p} d_W[p,q] * exp(lcr[p,m] - lcr[q,m]) * kappa[q,m]
    Loads d_W once per q-block, processes all M modes internally."""
    pid_b = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_p = tl.program_id(2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_chunk_i64 = pid_chunk.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_chunk * chunk_size)
    d_W_ptr += pid_b_i64 * dW_stride_B + pid_chunk_i64 * dW_stride_chunk
    kappa_ptr += pid_b_i64 * ck_stride_B + pid_chunk_i64 * ck_stride_chunk
    lcr_ptr += pid_b_i64 * lcr_stride_B + pid_chunk_i64 * lcr_stride_chunk
    d_c_ptr += pid_b_i64 * out_stride_B + pid_chunk_i64 * out_stride_chunk

    offs_p = pid_p * BLOCK_SIZE_PQ + tl.arange(0, BLOCK_SIZE_PQ)
    offs_m = tl.arange(0, M)
    mask_p = offs_p < chunk_len

    # Pre-load lcr for p-positions [P, M]
    lcr_p_all = tl.load(lcr_ptr + offs_p[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                        mask=mask_p[:, None], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_PQ, M), dtype=tl.float32)
    Q_END = (pid_p + 1) * BLOCK_SIZE_PQ

    for q_start in tl.range(0, Q_END, BLOCK_SIZE_PQ):
        offs_q = q_start + tl.arange(0, BLOCK_SIZE_PQ)
        mask_q = offs_q < chunk_len

        # Load d_W ONCE for all modes
        d_W_pq = tl.load(d_W_ptr + offs_p[:, None] * dW_stride_t + offs_q[None, :] * dW_stride_k,
                        mask=mask_p[:, None] & mask_q[None, :], other=0.0).to(tl.float32)  # [P, Q]

        # Load lcr_q and kappa_q for all modes [Q, M]
        lcr_q_all = tl.load(lcr_ptr + offs_q[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                            mask=mask_q[:, None], other=0.0).to(tl.float32)
        kappa_q_all = tl.load(kappa_ptr + offs_q[:, None] * ck_stride_pos + offs_m[None, :] * ck_stride_m,
                             mask=mask_q[:, None], other=0.0).to(tl.float32)

        # Process each mode
        for m in range(M):
            sel = (offs_m == m).to(tl.float32)
            lcr_p_m = tl.sum(lcr_p_all * sel[None, :], axis=1)
            lcr_q_m = tl.sum(lcr_q_all * sel[None, :], axis=1)
            kappa_q_m = tl.sum(kappa_q_all * sel[None, :], axis=1)

            decay = tl.exp(lcr_p_m[:, None] - lcr_q_m[None, :])
            if q_start == pid_p * BLOCK_SIZE_PQ:
                decay = tl.where(offs_p[:, None] >= offs_q[None, :], decay, 0.0)

            contrib = tl.sum(d_W_pq * decay * kappa_q_m[None, :], axis=1)
            acc += contrib[:, None] * sel[None, :]

    tl.store(d_c_ptr + offs_p[:, None] * out_stride_pos + offs_m[None, :] * out_stride_m,
            acc.to(d_c_ptr.dtype.element_ty), mask=mask_p[:, None])


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_PQ": block_size_pq},
            num_warps=num_warps,
            num_stages=2,
        )
        for block_size_pq in [32, 64]
        for num_warps in [4, 8]
    ],
    key=["chunk_size", "M"],
)
@triton.jit
def _frac_chunk_W_bwd_dkappa_kernel(
    d_W_ptr, c_ptr, lcr_ptr, d_kappa_ptr,
    dW_stride_B, dW_stride_chunk, dW_stride_t, dW_stride_k,
    ck_stride_B, ck_stride_chunk, ck_stride_pos, ck_stride_m,
    lcr_stride_B, lcr_stride_chunk, lcr_stride_pos, lcr_stride_m,
    out_stride_B, out_stride_chunk, out_stride_pos, out_stride_m,
    L, M: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK_SIZE_PQ: tl.constexpr,
):
    """d_kappa[p,m] = sum_{t>=p} d_W[t,p] * c[t,m] * exp(lcr[t,m] - lcr[p,m])
    Loads d_W once per t-block, processes all M modes internally."""
    pid_b = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_p = tl.program_id(2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_chunk_i64 = pid_chunk.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_chunk * chunk_size)
    d_W_ptr += pid_b_i64 * dW_stride_B + pid_chunk_i64 * dW_stride_chunk
    c_ptr += pid_b_i64 * ck_stride_B + pid_chunk_i64 * ck_stride_chunk
    lcr_ptr += pid_b_i64 * lcr_stride_B + pid_chunk_i64 * lcr_stride_chunk
    d_kappa_ptr += pid_b_i64 * out_stride_B + pid_chunk_i64 * out_stride_chunk

    offs_p = pid_p * BLOCK_SIZE_PQ + tl.arange(0, BLOCK_SIZE_PQ)
    offs_m = tl.arange(0, M)
    mask_p = offs_p < chunk_len

    # Pre-load lcr for p-positions [P, M]
    lcr_p_all = tl.load(lcr_ptr + offs_p[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                        mask=mask_p[:, None], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_PQ, M), dtype=tl.float32)
    Q_START = pid_p * BLOCK_SIZE_PQ

    for t_start in tl.range(Q_START, chunk_len, BLOCK_SIZE_PQ):
        offs_t = t_start + tl.arange(0, BLOCK_SIZE_PQ)
        mask_t = offs_t < chunk_len

        # Load d_W ONCE for all modes
        d_W_tp = tl.load(d_W_ptr + offs_t[:, None] * dW_stride_t + offs_p[None, :] * dW_stride_k,
                        mask=mask_t[:, None] & mask_p[None, :], other=0.0).to(tl.float32)  # [T, P]

        # Load lcr_t and c_t for all modes [T, M]
        lcr_t_all = tl.load(lcr_ptr + offs_t[:, None] * lcr_stride_pos + offs_m[None, :] * lcr_stride_m,
                            mask=mask_t[:, None], other=0.0).to(tl.float32)
        c_t_all = tl.load(c_ptr + offs_t[:, None] * ck_stride_pos + offs_m[None, :] * ck_stride_m,
                          mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Process each mode
        for m in range(M):
            sel = (offs_m == m).to(tl.float32)
            lcr_p_m = tl.sum(lcr_p_all * sel[None, :], axis=1)
            lcr_t_m = tl.sum(lcr_t_all * sel[None, :], axis=1)
            c_t_m = tl.sum(c_t_all * sel[None, :], axis=1)

            decay = tl.exp(lcr_t_m[:, None] - lcr_p_m[None, :])

            # Masked t lanes load lcr_t=0; for a very negative lcr_p,
            # exp(-lcr_p) can overflow and make 0 * 0 * inf = NaN.
            decay = tl.where(mask_t[:, None], decay, 0.0)
            if t_start == pid_p * BLOCK_SIZE_PQ:
                decay = tl.where(offs_t[:, None] >= offs_p[None, :], decay, 0.0)

            contrib = tl.sum(d_W_tp * c_t_m[:, None] * decay, axis=0)
            acc += contrib[:, None] * sel[None, :]

    tl.store(d_kappa_ptr + offs_p[:, None] * out_stride_pos + offs_m[None, :] * out_stride_m,
             acc.to(d_kappa_ptr.dtype.element_ty), mask=mask_p[:, None])


def _frac_chunk_W_bwd(d_W, c, kappa, log_cum_rho, chunk_size):
    B, L, M = c.shape
    nchunks = math.ceil(L / chunk_size)
    d_c_acc = torch.empty((B, L, M), dtype=torch.float32, device=c.device)
    d_kappa_acc = torch.empty((B, L, M), dtype=torch.float32, device=c.device)
    def grid_dc(META):
        return (B, nchunks, triton.cdiv(chunk_size, META['BLOCK_SIZE_PQ']))
    _frac_chunk_W_bwd_dc_kernel[grid_dc](
        d_W, kappa, log_cum_rho, d_c_acc,
        d_W.stride(0), d_W.stride(1), d_W.stride(2), d_W.stride(3),
        kappa.stride(0), chunk_size * kappa.stride(1), kappa.stride(1), kappa.stride(2),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        d_c_acc.stride(0), chunk_size * d_c_acc.stride(1), d_c_acc.stride(1), d_c_acc.stride(2),
        L, M, chunk_size)
    def grid_dkappa(META):
        return (B, nchunks, triton.cdiv(chunk_size, META['BLOCK_SIZE_PQ']))
    _frac_chunk_W_bwd_dkappa_kernel[grid_dkappa](
        d_W, c, log_cum_rho, d_kappa_acc,
        d_W.stride(0), d_W.stride(1), d_W.stride(2), d_W.stride(3),
        c.stride(0), chunk_size * c.stride(1), c.stride(1), c.stride(2),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        d_kappa_acc.stride(0), chunk_size * d_kappa_acc.stride(1), d_kappa_acc.stride(1), d_kappa_acc.stride(2),
        L, M, chunk_size)
    d_lcr = c.float() * d_c_acc - kappa.float() * d_kappa_acc
    return d_c_acc.to(c.dtype), d_kappa_acc.to(kappa.dtype), d_lcr


# ============================================================================
# Public API and autograd wrappers
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_w_fwd", mutates_args=())
def _chunk_w_fwd_op(
    c: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _frac_chunk_W_fwd(c, kappa, log_cum_rho, chunk_size)


@_chunk_w_fwd_op.register_fake
def _(c, kappa, log_cum_rho, chunk_size):
    B, L, _ = c.shape
    nchunks = math.ceil(L / chunk_size)
    return torch.empty(
        B, nchunks, chunk_size, chunk_size,
        device=c.device, dtype=torch.float32,
    )


@torch.library.custom_op("frac_ssm::chunk_w_bwd", mutates_args=())
def _chunk_w_bwd_op(
    d_W: torch.Tensor,
    c: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_W_bwd(d_W, c, kappa, log_cum_rho, chunk_size)


@_chunk_w_bwd_op.register_fake
def _(d_W, c, kappa, log_cum_rho, chunk_size):
    return torch.empty_like(c), torch.empty_like(kappa), torch.empty_like(log_cum_rho)


def _chunk_w_setup_ctx(ctx, inputs, output):
    c, kappa, log_cum_rho, chunk_size = inputs
    ctx.save_for_backward(c, kappa, log_cum_rho)
    ctx.chunk_size = chunk_size


def _chunk_w_backward(ctx, d_W):
    c, kappa, log_cum_rho = ctx.saved_tensors
    return (
        *_chunk_w_bwd_op(
            d_W.contiguous(), c, kappa, log_cum_rho, ctx.chunk_size
        ),
        None,
    )


_chunk_w_fwd_op.register_autograd(
    _chunk_w_backward, setup_context=_chunk_w_setup_ctx
)


def chunk_w(c, kappa, log_cum_rho, chunk_size):
    """Compute the causal per-chunk interaction matrix."""
    assert c.dim() == 3, f"c must be 3D (B, L, M), got {c.dim()}D"
    assert kappa.dim() == 3, f"kappa must be 3D (B, L, M), got {kappa.dim()}D"
    assert log_cum_rho.dim() == 3, (
        f"log_cum_rho must be 3D (B, L, M), got {log_cum_rho.dim()}D"
    )
    assert c.is_cuda and kappa.is_cuda and log_cum_rho.is_cuda, (
        "inputs must be on CUDA"
    )
    assert c.dtype == kappa.dtype, (
        f"c and kappa must have the same dtype, got {c.dtype} and {kappa.dtype}"
    )
    assert c.dtype in (torch.float32, torch.bfloat16), (
        f"c and kappa dtype must be float32 or bfloat16, got {c.dtype}"
    )
    assert log_cum_rho.dtype == torch.float32, (
        f"log_cum_rho dtype must be float32, got {log_cum_rho.dtype}"
    )
    M = c.shape[-1]
    assert M > 0 and (M & (M - 1)) == 0, (
        f"M must be a positive power of two, got {M}"
    )
    assert chunk_size > 0 and chunk_size % 32 == 0, (
        f"chunk_size must be a positive multiple of 32, got {chunk_size}"
    )
    return _chunk_w_fwd_op(c, kappa, log_cum_rho, chunk_size)
