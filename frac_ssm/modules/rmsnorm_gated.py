import torch
import torch.nn as nn

import triton.language as tl
import triton


def rmsnorm_gated_ref(hidden_states, weight, gate=None, eps=1e-6, *, upcast=False):
    """Reference matching Hugging Face MambaRMSNormGated when upcast=False.

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py#L2184-L2207
    """
    input_dtype = hidden_states.dtype
    compute_dtype = torch.float64 if upcast else torch.float32
    hidden_states = hidden_states.to(compute_dtype)

    if gate is not None:
        hidden_states = hidden_states * nn.functional.silu(gate.to(compute_dtype))
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)

    if upcast:
        return weight.to(compute_dtype) * hidden_states
    return weight * hidden_states.to(input_dtype)


@triton.jit
def rmsnorm_gated_fwd_kernel(
    X, W, Z, Y, Rstd,
    stride_x_row, stride_z_row, stride_y_row,
    M, D, eps,
    HAS_GATE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One-pass RMSNorm forward: load data once, compute variance and normalize."""
    row = tl.program_id(0).to(tl.int64)

    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_GATE:
        Z += row * stride_z_row

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load x once into registers
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # Apply gating before norm: x = silu(z) * x
    if HAS_GATE:
        z = tl.load(Z + cols, mask=mask, other=0.0).to(tl.float32)
        x = z * tl.sigmoid(z) * x

    # Compute rstd (x still in registers)
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply weight (x still in registers, no reload)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + cols, y, mask=mask)


def rmsnorm_gated_fwd(hidden_states, weight, gate=None, eps=1e-6):
    """One-pass RMSNorm forward with optional gating (before norm)."""
    shape_og = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, shape_og[-1])
    if hidden_states.stride(-1) != 1:
        hidden_states = hidden_states.contiguous()
    if gate is not None:
        gate = gate.reshape(-1, shape_og[-1])
        if gate.stride(-1) != 1:
            gate = gate.contiguous()

    M, D = hidden_states.shape
    output = torch.empty_like(hidden_states)
    rstd = torch.empty(M, device=hidden_states.device, dtype=torch.float32)

    has_gate = gate is not None
    if not has_gate:
        gate = hidden_states  # dummy, won't be used

    MAX_FUSED_SIZE = 65536 // hidden_states.element_size()
    BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError(f"RMSNorm doesn't support D={D} >= {BLOCK_D}")

    num_warps = min(max(BLOCK_D // 256, 1), 8)

    rmsnorm_gated_fwd_kernel[(M,)](
        hidden_states, weight, gate, output, rstd,
        hidden_states.stride(0), gate.stride(0), output.stride(0),
        M, D, eps,
        HAS_GATE=has_gate, BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return output.reshape(shape_og), rstd.reshape(shape_og[:-1])


bwd_config = [
    triton.Config({'BLOCK_L': BL, 'BLOCK_D': BD}, num_stages=ns, num_warps=nw)
    for BL in [32, 64]
    for BD in [32, 64, 128, 256]
    for nw in [4, 8]
    for ns in [3, 4, 5]
]


@triton.autotune(
    configs=bwd_config,
    key=['L', 'D'],
    reset_to_zero=['dw_ptr'],
)
@triton.jit
def rmsnorm_gated_bwd_kernel(dy_ptr, hidden_states_ptr, weight_ptr, rstd_ptr, gate_ptr, dx_ptr, dg_ptr, dw_ptr,
                             B, L, D, eps,
                             dy_strides_B, dy_strides_L, dy_strides_D,
                             hidden_states_strides_B, hidden_states_strides_L, hidden_states_strides_D,
                             weight_strides_D,
                             rstd_strides_B, rstd_strides_L,
                             gate_strides_B, gate_strides_L, gate_strides_D,
                             dx_strides_B, dx_strides_L, dx_strides_D,
                             dg_strides_B, dg_strides_L, dg_strides_D,
                             dw_strides_row, dw_strides_D,
                             HAS_GATE: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)

    tl.assume(pid_b >= 0)
    tl.assume(pid_l >= 0)
    tl.assume(dy_strides_L > 0)
    tl.assume(dy_strides_D > 0)
    tl.assume(hidden_states_strides_L > 0)
    tl.assume(hidden_states_strides_D > 0)
    tl.assume(dx_strides_L > 0)
    tl.assume(dx_strides_D > 0)

    dy_ptr += pid_b * dy_strides_B + pid_l * BLOCK_L * dy_strides_L
    hidden_states_ptr += pid_b * hidden_states_strides_B + pid_l * BLOCK_L * hidden_states_strides_L
    rstd_ptr += pid_b * rstd_strides_B + pid_l * BLOCK_L * rstd_strides_L
    dx_ptr += pid_b * dx_strides_B + pid_l * BLOCK_L * dx_strides_L
    dw_ptr += pid_b * dw_strides_row

    if HAS_GATE:
        gate_ptr += pid_b * gate_strides_B + pid_l * BLOCK_L * gate_strides_L
        dg_ptr += pid_b * dg_strides_B + pid_l * BLOCK_L * dg_strides_L

    offset_l = tl.arange(0, BLOCK_L)
    offset_d = tl.arange(0, BLOCK_D)
    mask_l = (pid_l * BLOCK_L + offset_l) < L
    rstd = tl.load(rstd_ptr + offset_l * rstd_strides_L, mask=mask_l, other=0.0).to(tl.float32)

    # Pass 1: compute dw and dot = sum_D(dx_norm * gated_x) for dx
    dot = tl.zeros((BLOCK_L,), dtype=tl.float32)
    for i in range(0, D, BLOCK_D):
        offset_d_block = i + offset_d
        mask_d = offset_d_block < D
        mask_ld = mask_l[:, None] & (mask_d[None, :])

        h_raw = tl.load(hidden_states_ptr + offset_l[:, None] * hidden_states_strides_L + offset_d_block[None,
                                                                                          :] * hidden_states_strides_D,
                        mask=mask_ld, other=0.0)
        h = h_raw.to(tl.float32)
        x = h
        if HAS_GATE:
            g = tl.load(gate_ptr + offset_l[:, None] * gate_strides_L + offset_d_block[None, :] * gate_strides_D,
                        mask=mask_ld, other=0.0).to(tl.float32)
            x = g * tl.sigmoid(g) * h

        dy = tl.load(dy_ptr + offset_l[:, None] * dy_strides_L + offset_d_block[None, :] * dy_strides_D,
                     mask=mask_ld, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offset_d_block * weight_strides_D, mask=mask_d, other=0.0).to(tl.float32)

        x_norm = x * rstd[:, None]

        dw_prod = dy * x_norm
        partial_dw = tl.sum(dw_prod, axis=0)
        tl.atomic_add(dw_ptr + offset_d_block * dw_strides_D, partial_dw, mask=mask_d)

        # dx_norm = dy * w
        dx_norm = dy * w[None, :]
        # dot = sum_D(dx_norm * x)  (x is gated input, not normalized)
        dot += tl.sum(dx_norm * x, axis=1)

    # Pass 2: compute dx (and dg if gated)
    for i in range(0, D, BLOCK_D):
        offset_d_block = i + offset_d
        mask_d = offset_d_block < D
        mask_ld = mask_l[:, None] & (mask_d[None, :])

        h = tl.load(hidden_states_ptr + offset_l[:, None] * hidden_states_strides_L + offset_d_block[None,
                                                                                      :] * hidden_states_strides_D,
                    mask=mask_ld, other=0.0).to(tl.float32)
        x = h
        if HAS_GATE:
            g = tl.load(gate_ptr + offset_l[:, None] * gate_strides_L + offset_d_block[None, :] * gate_strides_D,
                        mask=mask_ld, other=0.0).to(tl.float32)
            sig_g = tl.sigmoid(g)
            silu_g = g * sig_g
            x = silu_g * h

        dy = tl.load(dy_ptr + offset_l[:, None] * dy_strides_L + offset_d_block[None, :] * dy_strides_D,
                     mask=mask_ld, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offset_d_block * weight_strides_D, mask=mask_d, other=0.0).to(tl.float32)

        dx_norm = dy * w[None, :]
        # dx through RMSNorm: rstd * (dx_norm - rstd^2 / D * x * dot)
        dx = rstd[:, None] * (dx_norm - (rstd[:, None] * rstd[:, None] / D) * x * dot[:, None])

        if HAS_GATE:
            # dx is gradient w.r.t. gated x, now split into dh and dg
            # dh = dx * silu(g)
            dh = dx * silu_g
            # dg = dx * h * silu'(g) where silu'(g) = sig(g) * (1 + g*(1-sig(g)))
            silu_prime = sig_g * (1.0 + g * (1.0 - sig_g))
            dg = dx * h * silu_prime
            tl.store(dx_ptr + offset_l[:, None] * dx_strides_L + offset_d_block[None, :] * dx_strides_D,
                     dh, mask=mask_ld)
            tl.store(dg_ptr + offset_l[:, None] * dg_strides_L + offset_d_block[None, :] * dg_strides_D,
                     dg, mask=mask_ld)
        else:
            tl.store(dx_ptr + offset_l[:, None] * dx_strides_L + offset_d_block[None, :] * dx_strides_D,
                     dx, mask=mask_ld)


def rmsnorm_gated_bwd(dy, hidden_states, weight, rstd, gate=None, eps=1e-6):
    shape = hidden_states.shape
    D = shape[-1]
    L = shape[-2] if hidden_states.dim() > 1 else 1
    dy = dy.reshape(-1, L, D)
    hidden_states = hidden_states.reshape(-1, L, D)
    rstd = rstd.reshape(-1, L)
    if gate is not None:
        gate = gate.reshape(-1, L, D)

    dx = torch.empty_like(hidden_states)
    dw = torch.empty_like(weight)

    has_gate = gate is not None
    gate = gate if gate is not None else torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
    dg = torch.empty_like(gate) if has_gate else torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)

    gate_stride_B, gate_stride_L, gate_stride_D = gate.stride() if has_gate else (0, 0, 0)
    dg_stride_B, dg_stride_L, dg_stride_D = dg.stride() if has_gate else (0, 0, 0)
    B = hidden_states.shape[0]
    grid = lambda meta: (B, triton.cdiv(L, meta['BLOCK_L']))

    dw_partial = torch.zeros(B, D, device=hidden_states.device, dtype=torch.float32)
    rmsnorm_gated_bwd_kernel[grid](
        dy, hidden_states, weight, rstd, gate, dx, dg, dw_partial,
        B, L, D, eps,
        dy.stride(0), dy.stride(1), dy.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        weight.stride(0),
        rstd.stride(0), rstd.stride(1),
        gate_stride_B, gate_stride_L, gate_stride_D,
        dx.stride(0), dx.stride(1), dx.stride(2),
        dg_stride_B, dg_stride_L, dg_stride_D,
        dw_partial.stride(0), dw_partial.stride(1),
        HAS_GATE=has_gate,
    )
    dw = dw_partial.sum(dim=0).to(weight.dtype)

    if has_gate:
        return dx.reshape(shape), dw, dg.reshape(shape)

    return dx.reshape(shape), dw


class _RMSNormGatedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, gate, eps):
        output, rstd = rmsnorm_gated_fwd(hidden_states, weight, gate, eps)
        ctx.save_for_backward(hidden_states, weight, rstd, gate)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, d_output):
        hidden_states, weight, rstd, gate = ctx.saved_tensors
        grads = rmsnorm_gated_bwd(d_output, hidden_states, weight, rstd, gate, ctx.eps)
        if gate is not None:
            dx, dw, dg = grads
            return dx, dw, dg, None
        else:
            dx, dw = grads
            return dx, dw, None, None


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        return _RMSNormGatedFn.apply(hidden_states, self.weight, gate, self.variance_epsilon)