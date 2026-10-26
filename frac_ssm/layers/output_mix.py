import torch
import triton
import triton.language as tl


# ============================================================================
# Autotune configs
# ============================================================================

fwd_configs = [
    triton.Config({'BLOCK_L': bl}, num_warps=nw, num_stages=ns)
    for bl in [32, 64, 128]
    for nw in [4, 8]
    for ns in [1, 2, 3]
]

bwd_configs = [
    triton.Config({'BLOCK_L': bl}, num_warps=nw, num_stages=ns)
    for bl in [32, 64, 128, 256]
    for nw in [4, 8]
    for ns in [1, 2, 3]
]


# ============================================================================
# Forward kernel: D skip + reshape
# mixed_flat[BH, L, R] + x_conv[B, L, h*R+r] * D[H, R] -> mixed[B, L, HR]
# ============================================================================

@triton.autotune(configs=fwd_configs, key=['R'])
@triton.jit
def _frac_skip_mix_fwd_kernel(
    # inputs
    mf_ptr, xcv_ptr, D_ptr,
    # output
    out_ptr,
    # strides: mixed_flat [BH, L, R]
    mf_s0, mf_s1,
    # strides: x_conv [B, L, HR]
    xcv_s0, xcv_s1,
    # strides: out [B, L, HR]
    out_s0, out_s1,
    # dims
    L,
    H: tl.constexpr,
    R: tl.constexpr,
    # meta
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    lmask = offs_l < L
    h_R = h * R

    for r_start in tl.range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        rmask = offs_r < R

        # D[h, r]
        d_vals = tl.load(D_ptr + h * R + offs_r, mask=rmask, other=0.0)

        # mixed_flat[bh, l, r]
        mf_vals = tl.load(
            mf_ptr + pid_bh * mf_s0 + offs_l[:, None] * mf_s1 + offs_r[None, :],
            mask=lmask[:, None] & rmask[None, :], other=0.0)

        # x_conv[b, l, h*R + r] (= u[b, l, h, r])
        u_vals = tl.load(
            xcv_ptr + b * xcv_s0 + offs_l[:, None] * xcv_s1 + (h_R + offs_r[None, :]),
            mask=lmask[:, None] & rmask[None, :], other=0.0)

        # mixed + u * D
        result = mf_vals + u_vals * d_vals[None, :]

        # Store to out[b, l, h*R + r]
        tl.store(
            out_ptr + b * out_s0 + offs_l[:, None] * out_s1 + (h_R + offs_r[None, :]),
            result.to(out_ptr.dtype.element_ty),
            mask=lmask[:, None] & rmask[None, :])


def _frac_skip_mix_fwd(mixed_flat, x_conv, D, num_heads, head_dim, out_dtype):
    BH, L, R = mixed_flat.shape
    B = BH // num_heads
    H = num_heads
    BLOCK_R = triton.next_power_of_2(R)

    mixed = torch.empty(B, L, H * R, device=mixed_flat.device, dtype=out_dtype)

    grid = lambda META: (BH, triton.cdiv(L, META['BLOCK_L']))
    _frac_skip_mix_fwd_kernel[grid](
        mixed_flat, x_conv, D,
        mixed,
        mixed_flat.stride(0), mixed_flat.stride(1),
        x_conv.stride(0), x_conv.stride(1),
        mixed.stride(0), mixed.stride(1),
        L, H, R,
        BLOCK_R=BLOCK_R,
    )
    return mixed


# ============================================================================
# Backward kernel
# ============================================================================

@triton.autotune(configs=bwd_configs, key=['R'], reset_to_zero=["d_D_ptr"])
@triton.jit
def _frac_skip_mix_bwd_kernel(
    # inputs
    d_out_ptr, xcv_ptr, D_ptr,
    # outputs
    d_mf_ptr, d_xcv_ptr, d_D_ptr,
    # strides: d_out [B, L, HR]
    d_out_s0, d_out_s1,
    # strides: x_conv [B, L, HR]
    xcv_s0, xcv_s1,
    # strides: d_xcv [B, L, HR]
    d_xcv_s0, d_xcv_s1,
    # strides: d_mf [BH, L, R]
    d_mf_s0, d_mf_s1,
    # dims
    L,
    H: tl.constexpr,
    R: tl.constexpr,
    # meta
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    lmask = offs_l < L
    h_R = h * R

    for r_start in tl.range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        rmask = offs_r < R

        # d_out[b, l, h*R + r]
        d_out_vals = tl.load(
            d_out_ptr + b * d_out_s0 + offs_l[:, None] * d_out_s1 + (h_R + offs_r[None, :]),
            mask=lmask[:, None] & rmask[None, :], other=0.0).to(tl.float32)

        # D[h, r]
        d_vals = tl.load(D_ptr + h * R + offs_r, mask=rmask, other=0.0).to(tl.float32)

        # x_conv[b, l, h*R + r]
        u_vals = tl.load(
            xcv_ptr + b * xcv_s0 + offs_l[:, None] * xcv_s1 + (h_R + offs_r[None, :]),
            mask=lmask[:, None] & rmask[None, :], other=0.0).to(tl.float32)

        # d_mixed_flat[bh, l, r] = d_out
        tl.store(
            d_mf_ptr + pid_bh * d_mf_s0 + offs_l[:, None] * d_mf_s1 + offs_r[None, :],
            d_out_vals.to(d_mf_ptr.dtype.element_ty),
            mask=lmask[:, None] & rmask[None, :])

        # d_x_conv[b, l, h*R + r] = d_out * D[h, r]
        d_xcv_vals = d_out_vals * d_vals[None, :]
        tl.store(
            d_xcv_ptr + b * d_xcv_s0 + offs_l[:, None] * d_xcv_s1 + (h_R + offs_r[None, :]),
            d_xcv_vals.to(d_xcv_ptr.dtype.element_ty),
            mask=lmask[:, None] & rmask[None, :])

        # d_D[h, r] += sum_l (d_out * x_conv) -- atomic across (b, l-tiles)
        d_D_block = tl.sum(
            tl.where(lmask[:, None] & rmask[None, :], d_out_vals * u_vals, 0.0),
            axis=0)
        tl.atomic_add(d_D_ptr + h * R + offs_r, d_D_block.to(tl.float32), mask=rmask)


def _frac_skip_mix_bwd(d_mixed, x_conv, D, num_heads, head_dim, mixed_flat_dtype):
    B, L, HR = d_mixed.shape
    H = num_heads
    R = head_dim
    BH = B * H
    BLOCK_R = triton.next_power_of_2(R)

    d_mixed_flat = torch.empty(BH, L, R, device=d_mixed.device, dtype=mixed_flat_dtype)
    d_x_conv = torch.empty_like(x_conv)
    d_D = torch.zeros(H, R, device=d_mixed.device, dtype=torch.float32)

    grid = lambda META: (BH, triton.cdiv(L, META['BLOCK_L']))
    _frac_skip_mix_bwd_kernel[grid](
        d_mixed, x_conv, D,
        d_mixed_flat, d_x_conv, d_D,
        d_mixed.stride(0), d_mixed.stride(1),
        x_conv.stride(0), x_conv.stride(1),
        d_x_conv.stride(0), d_x_conv.stride(1),
        d_mixed_flat.stride(0), d_mixed_flat.stride(1),
        L, H, R,
        BLOCK_R=BLOCK_R,
    )
    return d_mixed_flat, d_x_conv, d_D


# ============================================================================
# Autograd Function
# ============================================================================

class _FracPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed_flat, x_conv, D, num_heads, head_dim, out_dtype):
        mixed = _frac_skip_mix_fwd(mixed_flat, x_conv, D, num_heads, head_dim, out_dtype)
        ctx.save_for_backward(x_conv, D)
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.mixed_flat_dtype = mixed_flat.dtype
        return mixed

    @staticmethod
    def backward(ctx, d_mixed):
        x_conv, D = ctx.saved_tensors
        d_mixed_flat, d_x_conv, d_D = _frac_skip_mix_bwd(
            d_mixed, x_conv, D, ctx.num_heads, ctx.head_dim, ctx.mixed_flat_dtype)
        return d_mixed_flat, d_x_conv, d_D, None, None, None


# ============================================================================
# Public API
# ============================================================================

def frac_skip_mix(model, mixed_flat, x_conv, out_dtype=None):
    """D skip connection, reshape.

    Triton kernel for the D skip + reshape.
    """
    if out_dtype is None:
        out_dtype = x_conv.dtype
    mixed = _FracPostFn.apply(
        mixed_flat, x_conv, model.D,
        model.num_heads, model.head_dim, out_dtype,
    )
    return mixed
