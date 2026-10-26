
import math

import torch
import triton
import triton.language as tl




@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_R": block_size_r, "BLOCK_SIZE_K": block_size_k},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_r, block_size_k in [(128, 16), (64, 16), (64, 32)]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["chunk_size", "M", "R"],
)
@triton.jit
def _frac_chunk_state_fwd_kernel(u_ptr, kappa_ptr, log_cum_rho_ptr, state_ptr,
                                 u_stride_batch, u_stride_chunk, u_stride_chunk_size, u_stride_r,
                                 kappa_stride_batch, kappa_stride_chunk, kappa_stride_chunk_size, kappa_stride_m,
                                 log_cum_rho_stride_batch, log_cum_rho_stride_chunk, log_cum_rho_stride_chunk_size, log_cum_rho_stride_m,
                                 state_stride_batch, state_stride_chunk, state_stride_r, state_stride_m,
                                 L, R, chunk_size:tl.constexpr, M:tl.constexpr, M_BLOCK:tl.constexpr,
                                 BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
                                 ):
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    pid_r = tl.program_id(axis=0)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    u_ptr += pid_b_i64 * u_stride_batch + pid_c_i64 * u_stride_chunk 
    kappa_ptr += pid_b_i64 * kappa_stride_batch + pid_c_i64 * kappa_stride_chunk
    log_cum_rho_ptr += pid_b_i64 * log_cum_rho_stride_batch + pid_c_i64 * log_cum_rho_stride_chunk
    state_ptr += pid_b_i64 * state_stride_batch + pid_c_i64 * state_stride_chunk


    offset_m = tl.arange(0, M_BLOCK)
    offset_c = tl.arange(0, BLOCK_SIZE_K)
    r_ids = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)

    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)
    lcr_end = tl.load(log_cum_rho_ptr + (chunk_len-1)*log_cum_rho_stride_chunk_size + offset_m[None, :]*log_cum_rho_stride_m,
                      mask= (offset_m[None, :]<M), other=0.0).to(tl.float32) # (1, M)
    acc = tl.zeros((BLOCK_SIZE_R, M_BLOCK), dtype=tl.float32)
    for i in tl.range(0, chunk_size, BLOCK_SIZE_K):
        offset_c_i = i + offset_c
        mask_c = offset_c_i < chunk_len
        decay = tl.load(log_cum_rho_ptr+(offset_c_i[:, None])*log_cum_rho_stride_chunk_size+ offset_m[None, :]*log_cum_rho_stride_m,
                        mask=mask_c[:, None] & (offset_m[None, :] < M), other=0.0).to(tl.float32) # (BLOCK_SIZE_K, M)

        decay_to_end = tl.exp(lcr_end-decay) # (BLOCK_SIZE_K, M)

        kappa = tl.load(kappa_ptr+offset_c_i[:, None]*kappa_stride_chunk_size+ offset_m[None, :]*kappa_stride_m,
                    mask=mask_c[:, None] & (offset_m[None, :] < M), other=0.0) # (BLOCK_SIZE_K, M)

        scale = kappa * decay_to_end # (BLOCK_SIZE_K, M) - both float32

        u = tl.load(u_ptr + r_ids[:,None]*u_stride_r + offset_c_i[None,:]*u_stride_chunk_size,
                mask=(r_ids[:, None] < R) & mask_c[None, :], other=0.0).to(tl.float32) # (BLOCK_SIZE_R, BLOCK_SIZE_K)

        acc = tl.dot(u, scale, acc)
    tl.store(state_ptr + r_ids[:,None]*state_stride_r + offset_m[None,:]*state_stride_m, acc.to(state_ptr.dtype.element_ty),
            mask=(r_ids[:,None] < R) & (offset_m[None,:] < M))




def _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32=True):
    B, L, R = u.shape
    M = log_cum_rho.shape[-1]
    nchunks = math.ceil(L / chunk_size)

    states_dtype = torch.float32 if state_fp32 else u.dtype
    states = torch.empty((B, nchunks, R, M), device=u.device, dtype=states_dtype)

    grid = lambda META: (triton.cdiv(R, META['BLOCK_SIZE_R']), B, nchunks, )

    _frac_chunk_state_fwd_kernel[grid](u, kappa, log_cum_rho, states,
                                       u.stride(0), chunk_size * u.stride(1), u.stride(1), u.stride(2),
                                       kappa.stride(0), chunk_size * kappa.stride(1), kappa.stride(1), kappa.stride(2),
                                       log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
                                       states.stride(0), states.stride(1), states.stride(2), states.stride(3),
                                       L, R, chunk_size, M, triton.next_power_of_2(M))

    return states



@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_K": block_size_k, "BLOCK_SIZE_R": block_size_r},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_k in [32, 64, 128]
        for block_size_r in [64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["chunk_size", "M", "R"],
    reset_to_zero=["d_log_cum_rho_ptr"],
)
@triton.jit
def _frac_chunk_state_bwd_kernel(
    d_states_ptr, u_ptr, kappa_ptr, log_cum_rho_ptr,
    d_u_ptr, d_kappa_ptr, d_log_cum_rho_ptr,
    d_states_stride_batch, d_states_stride_chunk, d_states_stride_r, d_states_stride_m,
    u_stride_batch, u_stride_chunk, u_stride_chunk_size, u_stride_r,
    kappa_stride_batch, kappa_stride_chunk, kappa_stride_chunk_size, kappa_stride_m,
    log_cum_rho_stride_batch, log_cum_rho_stride_chunk, log_cum_rho_stride_chunk_size, log_cum_rho_stride_m,
    d_u_stride_batch, d_u_stride_chunk, d_u_stride_chunk_size, d_u_stride_r,
    d_kappa_stride_batch, d_kappa_stride_chunk, d_kappa_stride_chunk_size, d_kappa_stride_m,
    d_log_cum_rho_stride_batch, d_log_cum_rho_stride_chunk, d_log_cum_rho_stride_chunk_size, d_log_cum_rho_stride_m,
    L, R, chunk_size: tl.constexpr, M: tl.constexpr, M_BLOCK: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
):
    
    pid_k = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_c_i64 = pid_c.to(tl.int64)

    d_states_ptr += pid_b_i64 * d_states_stride_batch + pid_c_i64 * d_states_stride_chunk
    u_ptr += pid_b_i64 * u_stride_batch + pid_c_i64 * u_stride_chunk
    kappa_ptr += pid_b_i64 * kappa_stride_batch + pid_c_i64 * kappa_stride_chunk
    log_cum_rho_ptr += pid_b_i64 * log_cum_rho_stride_batch + pid_c_i64 * log_cum_rho_stride_chunk
    d_u_ptr += pid_b_i64 * d_u_stride_batch + pid_c_i64 * d_u_stride_chunk
    d_kappa_ptr += pid_b_i64 * d_kappa_stride_batch + pid_c_i64 * d_kappa_stride_chunk
    d_log_cum_rho_ptr += pid_b_i64 * d_log_cum_rho_stride_batch + pid_c_i64 * d_log_cum_rho_stride_chunk

    offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offset_m = tl.arange(0, M_BLOCK)
    offset_r = tl.arange(0, BLOCK_SIZE_R)
    chunk_len = tl.minimum(chunk_size, L - pid_c * chunk_size)
    mask_k = offset_k < chunk_len
    mask_m = offset_m < M

    lcr_end = tl.load(log_cum_rho_ptr + (chunk_len - 1) * log_cum_rho_stride_chunk_size + offset_m * log_cum_rho_stride_m,
                      mask=mask_m, other=0.0).to(tl.float32) # (M_BLOCK,)
    lcr = tl.load(log_cum_rho_ptr + offset_k[:, None] * log_cum_rho_stride_chunk_size + offset_m[None, :] * log_cum_rho_stride_m,
                  mask=mask_k[:, None] & mask_m[None, :], other=0.0).to(tl.float32) # (BLOCK_SIZE_K, M_BLOCK)

    decay_to_end = tl.exp(lcr_end[None, :] - lcr) # (BLOCK_SIZE_K, M_BLOCK)

    kappa = tl.load(kappa_ptr + offset_k[:, None] * kappa_stride_chunk_size + offset_m[None, :] * kappa_stride_m,
                    mask=mask_k[:, None] & mask_m[None, :], other=0.0).to(tl.float32) # [BLOCK_SIZE_K, M_BLOCK]

    scale = kappa * decay_to_end  # [BLOCK_SIZE_K, M_BLOCK]

    d_scale_acc = tl.zeros((BLOCK_SIZE_K, M_BLOCK), dtype=tl.float32)

    for r_start in tl.range(0, R, BLOCK_SIZE_R):
        r_ids = r_start + offset_r
        mask_r = r_ids < R

        u = tl.load(u_ptr + offset_k[:, None] * u_stride_chunk_size + r_ids[None, :] * u_stride_r,
                    mask=mask_k[:, None] & mask_r[None, :], other=0.0).to(tl.float32) # [BLOCK_SIZE_K, BLOCK_SIZE_R]


        d_states = tl.load(d_states_ptr + r_ids[:, None] * d_states_stride_r + offset_m[None, :] * d_states_stride_m,
                           mask=mask_r[:, None] & mask_m[None, :], other=0.0).to(tl.float32) # [BLOCK_SIZE_R, M_BLOCK]


        d_scale_acc += tl.dot(u, d_states) # [BLOCK_SIZE_K, M_BLOCK]


        # d_states_t = tl.trans(d_states)  # [M_BLOCK, BLOCK_SIZE_R] it is faster to load again for the transpose
        d_states_t = tl.load(d_states_ptr + offset_m[:, None] * d_states_stride_m + r_ids[None, :] * d_states_stride_r,
                        mask=mask_m[:, None] & mask_r[None, :], other=0.0).to(tl.float32)

        d_u_block = tl.dot(scale, d_states_t)  # [BLOCK_SIZE_K, BLOCK_SIZE_R]

        tl.store(d_u_ptr + offset_k[:, None] * d_u_stride_chunk_size + r_ids[None, :] * d_u_stride_r, d_u_block.to(d_u_ptr.dtype.element_ty),
                 mask=mask_k[:, None] & mask_r[None, :])

    d_kappa = d_scale_acc * decay_to_end
    d_exp = d_scale_acc * kappa * decay_to_end
    d_log_cum_rho = -d_exp


    tl.store(d_kappa_ptr + offset_k[:, None] * d_kappa_stride_chunk_size + offset_m[None, :] * d_kappa_stride_m, d_kappa.to(d_kappa_ptr.dtype.element_ty),
             mask=mask_k[:, None] & mask_m[None, :])

    ### This is faster than doing torch.sum outside the kernel ###

    # Store to all positions except the last valid position
    mask_not_last = mask_k & (offset_k != (chunk_len - 1)) # (BLOCK_SIZE_K,)
    tl.store(d_log_cum_rho_ptr + offset_k[:, None] * d_log_cum_rho_stride_chunk_size + offset_m[None, :] * d_log_cum_rho_stride_m,
             d_log_cum_rho.to(d_log_cum_rho_ptr.dtype.element_ty), mask=mask_not_last[:, None] & mask_m[None, :]) # (BLOCK_SIZE_K, M_BLOCK)

    # Compute partial sum for the last valid position (excluding it if in this block)
    is_last_pos = (offset_k == (chunk_len - 1)) # (BLOCK_SIZE_K,)
    d_exp_for_sum = tl.where(is_last_pos[:, None], tl.zeros_like(d_exp), d_exp) # (BLOCK_SIZE_K, M_BLOCK)
    d_exp_partial_sum = tl.sum(d_exp_for_sum, axis=0) # (M_BLOCK,)

    # All k-blocks atomically add their partial sums to the last valid position
    tl.atomic_add(d_log_cum_rho_ptr + (chunk_len - 1) * d_log_cum_rho_stride_chunk_size + offset_m * d_log_cum_rho_stride_m,
                  d_exp_partial_sum.to(d_log_cum_rho_ptr.dtype.element_ty), mask=mask_m) # (M_BLOCK,)

def _frac_chunk_state_bwd(d_states, u, kappa, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = kappa.shape[-1]
    nchunks = math.ceil(L / chunk_size)

    d_u = torch.empty((B, L, R), device=u.device, dtype=u.dtype)
    d_kappa = torch.empty((B, L, M), device=kappa.device, dtype=kappa.dtype)
    d_log_cum_rho = torch.zeros((B, L, M), device=log_cum_rho.device, dtype=torch.float32)

    M_BLOCK = max(16, triton.next_power_of_2(M))

    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_K']), B, nchunks)
    _frac_chunk_state_bwd_kernel[grid](
        d_states, u, kappa, log_cum_rho,
        d_u, d_kappa, d_log_cum_rho,
        d_states.stride(0), d_states.stride(1), d_states.stride(2), d_states.stride(3),
        u.stride(0), chunk_size * u.stride(1), u.stride(1), u.stride(2),
        kappa.stride(0), chunk_size * kappa.stride(1), kappa.stride(1), kappa.stride(2),
        log_cum_rho.stride(0), chunk_size * log_cum_rho.stride(1), log_cum_rho.stride(1), log_cum_rho.stride(2),
        d_u.stride(0), chunk_size * d_u.stride(1), d_u.stride(1), d_u.stride(2),
        d_kappa.stride(0), chunk_size * d_kappa.stride(1), d_kappa.stride(1), d_kappa.stride(2),
        d_log_cum_rho.stride(0), chunk_size * d_log_cum_rho.stride(1), d_log_cum_rho.stride(1), d_log_cum_rho.stride(2),
        L, R, chunk_size, M, M_BLOCK
    )

    return d_u, d_kappa, d_log_cum_rho.to(log_cum_rho.dtype)


# ============================================================================
# Public API and autograd wrappers
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_state_fwd", mutates_args=())
def _chunk_state_fwd_op(
    u: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
    state_fp32: bool,
) -> torch.Tensor:
    return _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32)


@_chunk_state_fwd_op.register_fake
def _(u, kappa, log_cum_rho, chunk_size, state_fp32):
    B, L, R = u.shape
    M = kappa.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    states_dtype = torch.float32 if state_fp32 else u.dtype
    return torch.empty(B, nchunks, R, M, device=u.device, dtype=states_dtype)


@torch.library.custom_op("frac_ssm::chunk_state_bwd", mutates_args=())
def _chunk_state_bwd_op(
    d_states: torch.Tensor,
    u: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_state_bwd(d_states, u, kappa, log_cum_rho, chunk_size)


@_chunk_state_bwd_op.register_fake
def _(d_states, u, kappa, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = kappa.shape[-1]
    return (
        torch.empty(B, L, R, device=u.device, dtype=u.dtype),
        torch.empty(B, L, M, device=kappa.device, dtype=kappa.dtype),
        torch.empty(B, L, M, device=log_cum_rho.device, dtype=log_cum_rho.dtype),
    )


def _chunk_state_setup_ctx(ctx, inputs, output):
    u, kappa, log_cum_rho, chunk_size, state_fp32 = inputs
    ctx.save_for_backward(u, kappa, log_cum_rho)
    ctx.chunk_size = chunk_size


def _chunk_state_backward(ctx, d_states):
    u, kappa, log_cum_rho = ctx.saved_tensors
    return (
        *_chunk_state_bwd_op(
            d_states.contiguous(), u, kappa, log_cum_rho, ctx.chunk_size
        ),
        None,
        None,
    )


_chunk_state_fwd_op.register_autograd(_chunk_state_backward, setup_context=_chunk_state_setup_ctx)


def chunk_state(u, kappa, log_cum_rho, chunk_size, state_fp32=True):
    """Compute per-chunk states for scan."""
    assert u.dim() == 3, f"u must be 3D (B, L, R), got {u.dim()}D"
    assert kappa.dim() == 3, f"kappa must be 3D (B, L, M), got {kappa.dim()}D"
    assert log_cum_rho.dim() == 3, f"log_cum_rho must be 3D (B, L, M), got {log_cum_rho.dim()}D"
    assert u.is_cuda and kappa.is_cuda and log_cum_rho.is_cuda, "inputs must be on CUDA"
    assert log_cum_rho.dtype == torch.float32, (
        f"log_cum_rho dtype must be float32, got {log_cum_rho.dtype}"
    )
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    return _chunk_state_fwd_op(u, kappa, log_cum_rho, chunk_size, state_fp32)
