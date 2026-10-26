"""Triton forward and backward kernels for chunk readout.

Within each chunk, with lcr = log_cum_rho:
    h[t,r] = h_inter[t,r] + h_intra[t,r]
    h_inter[t,r] = sum_m c[t,m] * exp(lcr[t,m]) * states_acc[r,m]
    h_intra[t,r] = sum_{k<=t} W[t,k] * u[k,r]
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


if hasattr(tl, "make_tensor_descriptor"):
    _make_tensor_descriptor = tl.make_tensor_descriptor
elif hasattr(tl, "_experimental_make_tensor_descriptor"):
    _make_tensor_descriptor = tl._experimental_make_tensor_descriptor
else:
    _make_tensor_descriptor = None


def _supports_tma(device):
    return (
        _make_tensor_descriptor is not None
        and hasattr(triton, "set_allocator")
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 9
    )


def _set_tma_allocator(device):
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, device=device, dtype=torch.uint8,
        )
    )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "GROUP_SIZE_M": 8,
            },
            num_warps=num_warps,
            num_stages=3,
        )
        for block_size_m in [64, 128]
        for block_size_n in [64, 128, 256]
        for block_size_k in [32, 64]
        for num_warps in [4, 8]
    ],
    key=["chunk_size", "R", "M", "USE_TMA"],
)
@triton.jit
def _frac_chunk_readout_fwd_kernel(
    u_ptr, c_ptr, states_acc_ptr, W_ptr, log_cum_rho_ptr, h_ptr,
    stride_u_b, stride_u_c, stride_u_t, stride_u_r,
    stride_c_b, stride_c_c, stride_c_t, stride_c_m,
    stride_sa_b, stride_sa_c, stride_sa_r, stride_sa_m,
    stride_W_b, stride_W_c, stride_W_t, stride_W_k,
    stride_lcr_b, stride_lcr_c, stride_lcr_t, stride_lcr_m,
    stride_h_b, stride_h_c, stride_h_t, stride_h_r,
    L, chunk_size: tl.constexpr, R: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_DSTATE: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    # h = (c * exp(lcr)) @ states_acc.T + W @ u
    LOG2_E: tl.constexpr = 1.4426950408889634

    pid_c = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)

    # Grouped tile ordering for L2 reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(chunk_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(R, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compiler hints
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_W_t > 0)
    tl.assume(stride_W_k > 0)
    tl.assume(stride_u_t > 0)
    tl.assume(stride_u_r > 0)
    tl.assume(stride_h_t > 0)
    tl.assume(stride_h_r > 0)

    W_ptr += pid_b_i64 * stride_W_b + pid_c_i64 * stride_W_c
    u_ptr += pid_b_i64 * stride_u_b + pid_c_i64 * stride_u_c
    h_ptr += pid_b_i64 * stride_h_b + pid_c_i64 * stride_h_c
    c_ptr += pid_b_i64 * stride_c_b + pid_c_i64 * stride_c_c
    log_cum_rho_ptr += pid_b_i64 * stride_lcr_b + pid_c_i64 * stride_lcr_c
    states_acc_ptr += pid_b_i64 * stride_sa_b + pid_c_i64 * stride_sa_c

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)

    # h_inter = (c * exp(lcr)) @ states_acc.T
    lcr_m = tl.load(
        log_cum_rho_ptr + offs_m[:, None] * stride_lcr_t + offs_dstate[None, :] * stride_lcr_m,
        mask=(offs_m[:, None] < chunk_len) & (offs_dstate[None, :] < M), other=0.0
    ).to(tl.float32)
    scale_m = tl.math.exp2(lcr_m * LOG2_E)

    c = tl.load(
        c_ptr + offs_m[:, None] * stride_c_t + offs_dstate[None, :] * stride_c_m,
        mask=(offs_m[:, None] < chunk_len) & (offs_dstate[None, :] < M), other=0.0
    ).to(tl.float32)

    states_acc_ptrs = states_acc_ptr + offs_dstate[:, None] * stride_sa_m + offs_n[None, :] * stride_sa_r
    prev_states = tl.load(
        states_acc_ptrs,
        mask=(offs_dstate[:, None] < M) & (offs_n[None, :] < R), other=0.0
    )

    c_scaled = c * scale_m
    acc = tl.dot(
        c_scaled,
        prev_states.to(tl.float32),
    )

    # h_intra = W @ u. For this row tile, causality bounds k by its largest t.
    K_MAX = tl.minimum((pid_m + 1) * BLOCK_SIZE_M, chunk_len)

    if USE_TMA:
        W_desc = _make_tensor_descriptor(
            W_ptr, shape=[chunk_size, chunk_size], strides=[stride_W_t, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        u_desc = _make_tensor_descriptor(
            u_ptr, shape=[chunk_len, R], strides=[stride_u_t, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
        for k in tl.range(0, K_MAX, BLOCK_SIZE_K):
            W_block = W_desc.load([pid_m * BLOCK_SIZE_M, k])
            u_block = u_desc.load([k, pid_n * BLOCK_SIZE_N]).to(tl.float32)
            W_block = W_block.to(tl.float32)
            acc = tl.dot(W_block, u_block, acc)
    else:
        W_block_ptr = tl.make_block_ptr(
            base=W_ptr, shape=(chunk_size, chunk_size), strides=(stride_W_t, stride_W_k),
            offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0),
        )
        u_block_ptr = tl.make_block_ptr(
            base=u_ptr, shape=(chunk_len, R), strides=(stride_u_t, stride_u_r),
            offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0),
        )
        for k in tl.range(0, K_MAX, BLOCK_SIZE_K):
            W_block = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option="zero")
            u_block = tl.load(u_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            W_block = W_block.to(tl.float32)
            acc = tl.dot(W_block, u_block, acc)
            W_block_ptr = tl.advance(W_block_ptr, (0, BLOCK_SIZE_K))
            u_block_ptr = tl.advance(u_block_ptr, (BLOCK_SIZE_K, 0))

    h_block_ptr = tl.make_block_ptr(
        base=h_ptr, shape=(chunk_len, R), strides=(stride_h_t, stride_h_r),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
    )
    tl.store(h_block_ptr, acc.to(h_ptr.dtype.element_ty), boundary_check=(0, 1))


def _frac_chunk_readout_fwd(u, c, states_acc, W, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)

    h = torch.empty((B, L, R), device=u.device, dtype=u.dtype)

    use_tma = (
        _supports_tma(u.device)
        and u.is_contiguous()
        and W.is_contiguous()
    )
    if use_tma:
        _set_tma_allocator(u.device)

    def grid(META):
        num_pid_m = triton.cdiv(chunk_size, META['BLOCK_SIZE_M'])
        num_pid_n = triton.cdiv(R, META['BLOCK_SIZE_N'])
        return (num_pid_m * num_pid_n, nchunks, B)

    _frac_chunk_readout_fwd_kernel[grid](
        u, c, states_acc, W, log_cum_rho, h,
        u.stride(0), chunk_size * u.stride(1), u.stride(1), u.stride(2),
        c.stride(0), chunk_size * c.stride(1), c.stride(1), c.stride(2),
        states_acc.stride(0), states_acc.stride(1), states_acc.stride(2), states_acc.stride(3),
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        h.stride(0), chunk_size * h.stride(1), h.stride(1), h.stride(2),
        L, chunk_size, R, M,
        # Pad the mode dimension to a supported dot width; masked lanes load zero.
        BLOCK_SIZE_DSTATE=max(16, triton.next_power_of_2(M)),
        USE_TMA=use_tma,
    )

    return h



# ============================================================================
# Chunk readout backward
# ============================================================================
# Math below uses chunked views with c_decay = c * exp(lcr):
#   From h_inter = einsum('bntm,bnrm->bntr', c_decay, states_acc):
#       d_c_decay = einsum('bntr,bnrm->bntm', d_h, states_acc)
#       d_states_acc = einsum('bntr,bntm->bnrm', d_h, c_decay)
#       d_c_inter = d_c_decay * exp(lcr)
#       d_lcr_scan = d_c_decay * c_decay
#   From h_intra = einsum('bntk,bnkr->bntr', W, u):
#       d_u = einsum('bntk,bntr->bnkr', W, d_h)  # W.T @ d_h
#       d_W = einsum('bntr,bnkr->bntk', d_h, u)
#
# d_c_inter and d_lcr_scan are direct gradients through h_inter. In the full
# pipeline, d_W propagates through chunk_w and produces additional gradients.
#
# Chunked shapes used above:
#   d_h, u:                             [B, N, C, R]
#   c, lcr, c_decay:                    [B, N, C, M]
#   states_acc, d_states_acc:           [B, N, R, M]
#   W, d_W:                             [B, N, C, C]
#   d_u:                                [B, N, C, R]
#   d_c_inter, d_lcr_scan:              [B, N, C, M]
#
# Flat API shapes use L in place of N * C and crop the final chunk to length L.
# ============================================================================


# ----- Backward Kernel: d_u = W^T @ d_h -----
# W is lower triangular, so W^T is upper triangular
# d_u[k,r] = sum_{t>=k} W[t,k] * d_h[t,r]
# Exploits upper triangular structure by starting loop from diagonal.

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "GROUP_SIZE_M": 8,
            },
            num_warps=num_warps,
            num_stages=3,
        )
        for block_size_m in [64, 128]
        for block_size_n in [128, 256]
        for block_size_k in [32, 64]
        for num_warps in [4, 8]
    ],
    key=["chunk_size", "R"],
)
@triton.jit
def _frac_chunk_readout_bwd_du_kernel(
    d_h_ptr, W_ptr, d_u_ptr,
    stride_dh_b, stride_dh_c, stride_dh_t, stride_dh_r,
    stride_W_b, stride_W_c, stride_W_t, stride_W_k,
    stride_du_b, stride_du_c, stride_du_t, stride_du_r,
    L, chunk_size: tl.constexpr, R: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Compute d_u = W^T @ d_h. Exploits upper triangular W^T via causal loop bounds."""
    pid_c = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(chunk_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(R, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    W_ptr += pid_b_i64 * stride_W_b + pid_c_i64 * stride_W_c
    d_h_ptr += pid_b_i64 * stride_dh_b + pid_c_i64 * stride_dh_c
    d_u_ptr += pid_b_i64 * stride_du_b + pid_c_i64 * stride_du_c

    # W.T is upper triangular, so start the t reduction at the diagonal
    K_MIN = pid_m * BLOCK_SIZE_M

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    W_block_ptr = tl.make_block_ptr(
        base=W_ptr, shape=(chunk_size, chunk_size), strides=(stride_W_t, stride_W_k),
        offsets=(K_MIN, pid_m * BLOCK_SIZE_M), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M), order=(1, 0),
    )
    d_h_block_ptr = tl.make_block_ptr(
        base=d_h_ptr, shape=(chunk_len, R), strides=(stride_dh_t, stride_dh_r),
        offsets=(K_MIN, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0),
    )
    for t in tl.range(K_MIN, chunk_len, BLOCK_SIZE_K):
        W_block = tl.load(W_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        d_h_block = tl.load(d_h_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        acc = tl.dot(tl.trans(W_block), d_h_block, acc)
        W_block_ptr = tl.advance(W_block_ptr, (BLOCK_SIZE_K, 0))
        d_h_block_ptr = tl.advance(d_h_block_ptr, (BLOCK_SIZE_K, 0))

    d_u_block_ptr = tl.make_block_ptr(
        base=d_u_ptr, shape=(chunk_len, R), strides=(stride_du_t, stride_du_r),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
    )
    tl.store(d_u_block_ptr, acc.to(d_u_ptr.dtype.element_ty), boundary_check=(0, 1))


# ----- Backward Kernel: h_inter (d_c_inter, d_lcr_scan) -----

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_T": block_size_t, "BLOCK_SIZE_R": block_size_r},
            num_warps=num_warps,
            num_stages=3,
        )
        for block_size_t in [32, 64, 128]
        for block_size_r in [128, 256]
        for num_warps in [4, 8]
    ],
    key=["chunk_size", "R", "M"],
)
@triton.jit
def _frac_chunk_readout_bwd_inter_kernel(
    d_h_ptr, c_ptr, states_acc_ptr, log_cum_rho_ptr,
    d_c_inter_ptr, d_lcr_scan_ptr,
    stride_dh_b, stride_dh_c, stride_dh_t, stride_dh_r,
    stride_c_b, stride_c_c, stride_c_t, stride_c_m,
    stride_sa_b, stride_sa_c, stride_sa_r, stride_sa_m,
    stride_lcr_b, stride_lcr_c, stride_lcr_t, stride_lcr_m,
    stride_dc_b, stride_dc_c, stride_dc_t, stride_dc_m,
    stride_dlcr_b, stride_dlcr_c, stride_dlcr_t, stride_dlcr_m,
    L, chunk_size: tl.constexpr, R: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    """Compute d_c_inter and d_lcr_scan from h_inter backward.

    d_c_inter = (d_h @ states_acc) * exp(lcr)
    d_lcr_scan = d_c_inter * c
    """
    LOG2_E: tl.constexpr = 1.4426950408889634

    pid_c = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    pid_t = tl.program_id(axis=0)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)

    d_h_ptr += pid_b_i64 * stride_dh_b + pid_c_i64 * stride_dh_c
    c_ptr += pid_b_i64 * stride_c_b + pid_c_i64 * stride_c_c
    states_acc_ptr += pid_b_i64 * stride_sa_b + pid_c_i64 * stride_sa_c
    log_cum_rho_ptr += pid_b_i64 * stride_lcr_b + pid_c_i64 * stride_lcr_c
    d_c_inter_ptr += pid_b_i64 * stride_dc_b + pid_c_i64 * stride_dc_c
    d_lcr_scan_ptr += pid_b_i64 * stride_dlcr_b + pid_c_i64 * stride_dlcr_c

    offs_t = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    c = tl.load(
        c_ptr + offs_t[:, None] * stride_c_t + offs_m[None, :] * stride_c_m,
        mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M), other=0.0
    ).to(tl.float32)
    lcr = tl.load(
        log_cum_rho_ptr + offs_t[:, None] * stride_lcr_t + offs_m[None, :] * stride_lcr_m,
        mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M), other=0.0
    ).to(tl.float32)
    decay = tl.math.exp2(lcr * LOG2_E)
    c_decay = c * decay

    d_c_decay = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_M), dtype=tl.float32)

    for r_start in tl.range(0, R, BLOCK_SIZE_R):
        offs_r = r_start + tl.arange(0, BLOCK_SIZE_R)
        d_h_block = tl.load(
            d_h_ptr + offs_t[:, None] * stride_dh_t + offs_r[None, :] * stride_dh_r,
            mask=(offs_t[:, None] < chunk_len) & (offs_r[None, :] < R), other=0.0
        ).to(tl.float32)
        states_acc_block = tl.load(
            states_acc_ptr + offs_r[:, None] * stride_sa_r + offs_m[None, :] * stride_sa_m,
            mask=(offs_r[:, None] < R) & (offs_m[None, :] < M), other=0.0
        ).to(tl.float32)
        d_c_decay += tl.dot(
            d_h_block,
            states_acc_block,
        )

    d_c_inter = d_c_decay * decay
    d_lcr_scan = d_c_decay * c_decay

    tl.store(
        d_c_inter_ptr + offs_t[:, None] * stride_dc_t + offs_m[None, :] * stride_dc_m,
        d_c_inter.to(d_c_inter_ptr.dtype.element_ty),
        mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M)
    )
    tl.store(
        d_lcr_scan_ptr + offs_t[:, None] * stride_dlcr_t + offs_m[None, :] * stride_dlcr_m,
        d_lcr_scan.to(d_lcr_scan_ptr.dtype.element_ty),
        mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M)
    )


# ----- Backward Kernel: d_states_acc -----

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_R": block_size_r, "BLOCK_SIZE_T": block_size_t},
            num_warps=4,
            num_stages=3,
        )
        for block_size_r in [64, 128]
        for block_size_t in [32, 64, 128]
    ],
    key=["chunk_size", "R", "M"],
)
@triton.jit
def _frac_chunk_readout_bwd_dstates_kernel(
    d_h_ptr, c_ptr, log_cum_rho_ptr, d_states_acc_ptr,
    stride_dh_b, stride_dh_c, stride_dh_t, stride_dh_r,
    stride_c_b, stride_c_c, stride_c_t, stride_c_m,
    stride_lcr_b, stride_lcr_c, stride_lcr_t, stride_lcr_m,
    stride_dsa_b, stride_dsa_c, stride_dsa_r, stride_dsa_m,
    L, chunk_size: tl.constexpr, R: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    """Compute d_states_acc = d_h^T @ c_decay, where c_decay = c * exp(lcr)."""
    LOG2_E: tl.constexpr = 1.4426950408889634

    pid_c = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)
    pid_r = tl.program_id(axis=0)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)

    d_h_ptr += pid_b_i64 * stride_dh_b + pid_c_i64 * stride_dh_c
    c_ptr += pid_b_i64 * stride_c_b + pid_c_i64 * stride_c_c
    log_cum_rho_ptr += pid_b_i64 * stride_lcr_b + pid_c_i64 * stride_lcr_c
    d_states_acc_ptr += pid_b_i64 * stride_dsa_b + pid_c_i64 * stride_dsa_c

    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    d_states_acc = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_M), dtype=tl.float32)

    for t_start in tl.range(0, chunk_len, BLOCK_SIZE_T):
        offs_t = t_start + tl.arange(0, BLOCK_SIZE_T)
        d_h_block = tl.load(
            d_h_ptr + offs_t[:, None] * stride_dh_t + offs_r[None, :] * stride_dh_r,
            mask=(offs_t[:, None] < chunk_len) & (offs_r[None, :] < R), other=0.0
        ).to(tl.float32)
        c_block = tl.load(
            c_ptr + offs_t[:, None] * stride_c_t + offs_m[None, :] * stride_c_m,
            mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M), other=0.0
        ).to(tl.float32)
        lcr_block = tl.load(
            log_cum_rho_ptr + offs_t[:, None] * stride_lcr_t + offs_m[None, :] * stride_lcr_m,
            mask=(offs_t[:, None] < chunk_len) & (offs_m[None, :] < M), other=0.0
        ).to(tl.float32)
        decay_block = tl.math.exp2(lcr_block * LOG2_E)
        c_decay_block = c_block * decay_block
        d_states_acc += tl.dot(
            tl.trans(d_h_block),
            c_decay_block,
        )

    tl.store(
        d_states_acc_ptr + offs_r[:, None] * stride_dsa_r + offs_m[None, :] * stride_dsa_m,
        d_states_acc.to(d_states_acc_ptr.dtype.element_ty),
        mask=(offs_r[:, None] < R) & (offs_m[None, :] < M)
    )


def _frac_chunk_readout_bwd(d_h, u, c, states_acc, W, log_cum_rho, chunk_size):
    """Backward for the inter- and intra-chunk readout contributions.

    Uses three Triton kernels followed by one PyTorch BMM:
    - d_u: W^T @ d_h (exploits upper triangular structure)
    - h_inter backward: d_c_inter, d_lcr_scan
    - d_states_acc: d_h^T @ c_decay
    - d_W: d_h @ u^T
    """
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)

    # Allocate outputs written by the Triton kernels
    d_u = torch.empty((B, L, R), device=u.device, dtype=u.dtype)
    d_c_inter = torch.empty((B, L, M), device=c.device, dtype=c.dtype)
    d_lcr_scan = torch.empty((B, L, M), device=log_cum_rho.device, dtype=log_cum_rho.dtype)
    d_states_acc = torch.empty_like(states_acc)

    # d_u = W.T @ d_h; start the t reduction at the diagonal
    def grid_du(META):
        num_pid_m = triton.cdiv(chunk_size, META['BLOCK_SIZE_M'])
        num_pid_n = triton.cdiv(R, META['BLOCK_SIZE_N'])
        return (num_pid_m * num_pid_n, nchunks, B)

    _frac_chunk_readout_bwd_du_kernel[grid_du](
        d_h, W, d_u,
        d_h.stride(0), chunk_size * d_h.stride(1), d_h.stride(1), d_h.stride(2),
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        d_u.stride(0), chunk_size * d_u.stride(1), d_u.stride(1), d_u.stride(2),
        L, chunk_size, R,
    )

    # Direct h_inter gradients: d_c_inter and d_lcr_scan
    def grid_inter(META):
        num_pid_t = triton.cdiv(chunk_size, META['BLOCK_SIZE_T'])
        return (num_pid_t, nchunks, B)

    _frac_chunk_readout_bwd_inter_kernel[grid_inter](
        d_h, c, states_acc, log_cum_rho,
        d_c_inter, d_lcr_scan,
        d_h.stride(0), chunk_size * d_h.stride(1), d_h.stride(1), d_h.stride(2),
        c.stride(0), chunk_size * c.stride(1), c.stride(1), c.stride(2),
        states_acc.stride(0), states_acc.stride(1), states_acc.stride(2), states_acc.stride(3),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        d_c_inter.stride(0), chunk_size * d_c_inter.stride(1), d_c_inter.stride(1), d_c_inter.stride(2),
        d_lcr_scan.stride(0), chunk_size * d_lcr_scan.stride(1), d_lcr_scan.stride(1), d_lcr_scan.stride(2),
        L, chunk_size, R, M,
        BLOCK_SIZE_M=triton.next_power_of_2(M),
    )

    # d_states_acc = d_h^T @ c_decay
    def grid_dstates(META):
        num_pid_r = triton.cdiv(R, META['BLOCK_SIZE_R'])
        return (num_pid_r, nchunks, B)

    _frac_chunk_readout_bwd_dstates_kernel[grid_dstates](
        d_h, c, log_cum_rho, d_states_acc,
        d_h.stride(0), chunk_size * d_h.stride(1), d_h.stride(1), d_h.stride(2),
        c.stride(0), chunk_size * c.stride(1), c.stride(1), c.stride(2),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        d_states_acc.stride(0), d_states_acc.stride(1), d_states_acc.stride(2), d_states_acc.stride(3),
        L, chunk_size, R, M,
        BLOCK_SIZE_M=triton.next_power_of_2(M),
    )

    # d_W = d_h @ u.T
    L_padded = nchunks * chunk_size
    if L < L_padded:
        pad_len = L_padded - L
        d_h_padded = F.pad(d_h, (0, 0, 0, pad_len), value=0.0)
        u_padded = F.pad(u, (0, 0, 0, pad_len), value=0.0)
    else:
        d_h_padded, u_padded = d_h, u
    d_h_bmm = d_h_padded.view(B * nchunks, chunk_size, R)
    u_bmm = u_padded.view(B * nchunks, chunk_size, R).transpose(1, 2)
    d_W = torch.bmm(d_h_bmm, u_bmm)
    d_W = d_W.view(B, nchunks, chunk_size, chunk_size)

    return d_u, d_c_inter, d_states_acc, d_lcr_scan, d_W


# ============================================================================
# Public API
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_readout_fwd", mutates_args=())
def _chunk_readout_fwd_op(
    u: torch.Tensor,
    c: torch.Tensor,
    states_acc: torch.Tensor,
    W: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _frac_chunk_readout_fwd(u, c, states_acc, W, log_cum_rho, chunk_size)


@_chunk_readout_fwd_op.register_fake
def _(u, c, states_acc, W, log_cum_rho, chunk_size):
    B, L, R = u.shape
    return torch.empty(B, L, R, device=u.device, dtype=u.dtype)


@torch.library.custom_op("frac_ssm::chunk_readout_bwd", mutates_args=())
def _chunk_readout_bwd_op(
    d_h: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    states_acc: torch.Tensor,
    W: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_readout_bwd(
        d_h, u, c, states_acc, W, log_cum_rho, chunk_size
    )


@_chunk_readout_bwd_op.register_fake
def _(d_h, u, c, states_acc, W, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = W.shape[1]
    return (
        torch.empty(B, L, R, device=u.device, dtype=u.dtype),
        torch.empty(B, L, M, device=c.device, dtype=c.dtype),
        torch.empty_like(states_acc),
        torch.empty(B, L, M, device=log_cum_rho.device, dtype=log_cum_rho.dtype),
        torch.empty(B, nchunks, chunk_size, chunk_size, device=W.device, dtype=d_h.dtype),
    )


def _chunk_readout_setup_ctx(ctx, inputs, output):
    u, c, states_acc, W, log_cum_rho, chunk_size = inputs
    ctx.save_for_backward(u, c, states_acc, W, log_cum_rho)
    ctx.chunk_size = chunk_size


def _chunk_readout_backward(ctx, d_h):
    u, c, states_acc, W, log_cum_rho = ctx.saved_tensors
    d_u, d_c, d_states_acc, d_log_cum_rho, d_W = _chunk_readout_bwd_op(
        d_h.contiguous(),
        u,
        c,
        states_acc,
        W,
        log_cum_rho,
        ctx.chunk_size,
    )
    return d_u, d_c, d_states_acc, d_W, d_log_cum_rho, None


_chunk_readout_fwd_op.register_autograd(_chunk_readout_backward, setup_context=_chunk_readout_setup_ctx)


def chunk_readout(u, c, states_acc, W, log_cum_rho, chunk_size):
    """Apply inter- and intra-chunk scan contributions."""
    assert u.dim() == 3, f"u must be 3D (B, L, R), got {u.dim()}D"
    assert c.dim() == 3, f"c must be 3D (B, L, M), got {c.dim()}D"
    assert states_acc.dim() == 4, f"states_acc must be 4D (B, N, R, M), got {states_acc.dim()}D"
    assert W.dim() == 4, f"W must be 4D (B, N, C, C), got {W.dim()}D"
    assert log_cum_rho.dim() == 3, f"log_cum_rho must be 3D (B, L, M), got {log_cum_rho.dim()}D"
    assert all(x.is_cuda for x in (u, c, states_acc, W, log_cum_rho)), "inputs must be on CUDA"
    assert states_acc.dtype in (u.dtype, torch.float32), (
        f"states_acc dtype must be {u.dtype} or float32, got {states_acc.dtype}"
    )
    assert W.dtype in (u.dtype, torch.float32), (
        f"W dtype must be {u.dtype} or float32, got {W.dtype}"
    )
    assert log_cum_rho.dtype == torch.float32, (
        f"log_cum_rho dtype must be float32, got {log_cum_rho.dtype}"
    )
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"

    B, L, R = u.shape
    M = c.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    assert c.shape == (B, L, M), "c shape must be (B, L, M)"
    assert log_cum_rho.shape == (B, L, M), (
        "log_cum_rho shape must be (B, L, M)"
    )
    assert states_acc.shape == (B, nchunks, R, M), (
        "states_acc shape must be (B, nchunks, R, M)"
    )
    assert W.shape == (B, nchunks, chunk_size, chunk_size), (
        "W shape must be (B, nchunks, chunk_size, chunk_size)"
    )
    return _chunk_readout_fwd_op(u, c, states_acc, W, log_cum_rho, chunk_size)
