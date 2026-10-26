import torch
import torch.nn.functional as F
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
# Forward kernel
# ============================================================================

@triton.autotune(configs=fwd_configs, key=['M', 'R'])
@triton.jit
def _frac_compute_params_fwd_kernel(
    # inputs
    dyn_ptr, rw_ptr, mask_ptr, xcv_ptr,
    # weights / buffers
    dt_bias_ptr, read_gate_ptr, write_gate_ptr, log_tau_ptr,
    # outputs
    u_ptr, c_ptr, kappa_ptr, rho_ptr,
    # saved softmax normalization for backward
    r_amax_ptr, r_sum_ptr, w_amax_ptr, w_sum_ptr,
    # strides: dyn [B, L, 3H]
    dyn_s0, dyn_s1,
    # strides: rw [B, L, 2HM]
    rw_s0, rw_s1,
    # strides: mask [B, L]
    mask_s0, mask_s1,
    # strides: x_conv [B, L, HR]
    xcv_s0, xcv_s1,
    # strides: u [BH, L, R]
    u_s0, u_s1,
    # strides: c/kappa/rho [BH, L, M]
    out_s0, out_s1,
    # dims
    L,
    H: tl.constexpr,
    M: tl.constexpr,
    HM: tl.constexpr,
    R: tl.constexpr,
    # config scalars
    read_temp,
    dt_min, dt_max,
    alpha_min, alpha_max,
    lambda_min, lambda_max,
    # meta
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_m = tl.arange(0, M)
    lmask = offs_l < L

    # --- Constants (loaded once per thread block) ---
    dtb = tl.load(dt_bias_ptr + h).to(tl.float32)
    rg = tl.sigmoid(tl.load(read_gate_ptr).to(tl.float32))
    wg = tl.sigmoid(tl.load(write_gate_ptr).to(tl.float32))
    lt = tl.load(log_tau_ptr + offs_m).to(tl.float32)  # [M]
    tau = tl.exp(lt)                                     # [M]

    # --- Load dyn: 3 scattered scalars per position ---
    db = dyn_ptr + b * dyn_s0
    dt_r = tl.load(db + offs_l * dyn_s1 + h,         mask=lmask, other=0.0).to(tl.float32)
    al_r = tl.load(db + offs_l * dyn_s1 + (H + h),   mask=lmask, other=0.0).to(tl.float32)
    la_r = tl.load(db + offs_l * dyn_s1 + (2*H + h), mask=lmask, other=0.0).to(tl.float32)

    # --- dt = softplus(raw + bias).clamp ---
    sp = dt_r + dtb
    dt = tl.where(sp > 20.0, sp, tl.log(1.0 + tl.exp(sp)))
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)

    # --- alpha, lambda ---
    alpha = alpha_min + tl.sigmoid(al_r) * (alpha_max - alpha_min)
    lam = lambda_min + tl.sigmoid(la_r) * (lambda_max - lambda_min)

    # --- Load rw: 2*M values per position ---
    rb = rw_ptr + b * rw_s0
    rd = tl.load(rb + offs_l[:, None] * rw_s1 + (h * M + offs_m[None, :]),
                 mask=lmask[:, None], other=0.0).to(tl.float32)
    wr = tl.load(rb + offs_l[:, None] * rw_s1 + (HM + h * M + offs_m[None, :]),
                 mask=lmask[:, None], other=0.0).to(tl.float32)

    # --- prior = -alpha * log_tau ---
    prior = -alpha[:, None] * lt[None, :]

    # --- Read softmax (over M) — save amax/sum for backward ---
    rl = (prior + rg * rd) / read_temp
    r_amax_val = tl.max(rl, axis=1)          # [BLOCK_L]
    rl = rl - r_amax_val[:, None]
    re = tl.exp(rl)
    r_sum_val = tl.sum(re, axis=1)           # [BLOCK_L]
    c = re / r_sum_val[:, None]

    # --- Write softmax (over M) — save amax/sum for backward ---
    wl = prior + wg * wr
    w_amax_val = tl.max(wl, axis=1)          # [BLOCK_L]
    wl = wl - w_amax_val[:, None]
    we = tl.exp(wl)
    w_sum_val = tl.sum(we, axis=1)           # [BLOCK_L]
    ww = we / w_sum_val[:, None]

    # --- rho, kappa ---
    lf = tl.exp(tl.log(tl.maximum(lam, 1e-4)) / tl.maximum(alpha, 1e-4))
    ite = lf[:, None] / tl.maximum(tau[None, :], 1e-6)
    rho = tl.exp(-dt[:, None] * ite)
    rho = tl.minimum(tl.maximum(rho, 1e-6), 1.0 - 1e-6)
    kappa = (1.0 - rho) * ww

    # --- Attention mask ---
    if HAS_MASK:
        mv = tl.load(mask_ptr + b * mask_s0 + offs_l * mask_s1,
                     mask=lmask, other=0.0).to(tl.float32)
        kappa = kappa * mv[:, None]
        rho = rho * mv[:, None] + (1.0 - mv[:, None])

    # --- Store c, kappa, rho [BH, L, M] ---
    ob = pid_bh * out_s0
    so = offs_l[:, None] * out_s1 + offs_m[None, :]
    tl.store(c_ptr     + ob + so, c.to(c_ptr.dtype.element_ty),         mask=lmask[:, None])
    tl.store(kappa_ptr + ob + so, kappa.to(kappa_ptr.dtype.element_ty), mask=lmask[:, None])
    tl.store(rho_ptr   + ob + so, rho.to(rho_ptr.dtype.element_ty),     mask=lmask[:, None])

    # --- Store softmax normalization constants [BH, L] ---
    sm_base = pid_bh * L
    tl.store(r_amax_ptr + sm_base + offs_l, r_amax_val, mask=lmask)
    tl.store(r_sum_ptr  + sm_base + offs_l, r_sum_val,  mask=lmask)
    tl.store(w_amax_ptr + sm_base + offs_l, w_amax_val, mask=lmask)
    tl.store(w_sum_ptr  + sm_base + offs_l, w_sum_val,  mask=lmask)

    # --- u permute: x_conv[b, l, h*R + r] → u[bh, l, r] ---
    xcv_base = xcv_ptr + b * xcv_s0
    u_base = u_ptr + pid_bh * u_s0
    h_R = h * R
    for r_start in tl.range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        rmask = offs_r < R
        vals = tl.load(xcv_base + offs_l[:, None] * xcv_s1 + (h_R + offs_r[None, :]),
                       mask=lmask[:, None] & rmask[None, :], other=0.0)
        tl.store(u_base + offs_l[:, None] * u_s1 + offs_r[None, :],
                 vals, mask=lmask[:, None] & rmask[None, :])


def _frac_compute_params_fwd(dyn, rw, x_conv, attention_mask,
                             dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
                             num_heads, head_dim, num_modes, read_temp,
                             dt_min, dt_max, alpha_min, alpha_max, lambda_min, lambda_max):
    B, L, _ = x_conv.shape
    H = num_heads
    R = head_dim
    M = num_modes
    HM = H * M
    BH = B * H
    BLOCK_R = triton.next_power_of_2(R)

    out_dtype = x_conv.dtype
    u_flat     = torch.empty(BH, L, R, device=x_conv.device, dtype=out_dtype)
    c_flat     = torch.empty(BH, L, M, device=x_conv.device, dtype=torch.float32)
    kappa_flat = torch.empty(BH, L, M, device=x_conv.device, dtype=torch.float32)
    rho_flat   = torch.empty(BH, L, M, device=x_conv.device, dtype=torch.float32)

    # Softmax normalization constants saved for backward
    r_amax = torch.empty(BH, L, device=x_conv.device, dtype=torch.float32)
    r_sum  = torch.empty(BH, L, device=x_conv.device, dtype=torch.float32)
    w_amax = torch.empty(BH, L, device=x_conv.device, dtype=torch.float32)
    w_sum  = torch.empty(BH, L, device=x_conv.device, dtype=torch.float32)

    has_mask = attention_mask is not None
    if has_mask:
        ms0, ms1 = attention_mask.stride()
    else:
        attention_mask = c_flat
        ms0, ms1 = 0, 0

    grid = lambda META: (BH, triton.cdiv(L, META['BLOCK_L']))
    _frac_compute_params_fwd_kernel[grid](
        dyn, rw, attention_mask, x_conv,
        dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
        u_flat, c_flat, kappa_flat, rho_flat,
        r_amax, r_sum, w_amax, w_sum,
        dyn.stride(0), dyn.stride(1),
        rw.stride(0), rw.stride(1),
        ms0, ms1,
        x_conv.stride(0), x_conv.stride(1),
        u_flat.stride(0), u_flat.stride(1),
        c_flat.stride(0), c_flat.stride(1),
        L, H, M, HM, R,
        read_temp,
        dt_min, dt_max, alpha_min, alpha_max, lambda_min, lambda_max,
        BLOCK_R=BLOCK_R,
        HAS_MASK=has_mask,
    )

    return u_flat, c_flat, kappa_flat, rho_flat, r_amax, r_sum, w_amax, w_sum


# ============================================================================
# Backward kernel — single kernel, recomputes c/ww/rho from saved amax/sum
# ============================================================================

@triton.autotune(
    configs=bwd_configs,
    key=['M', 'R'],
    reset_to_zero=['d_dt_bias_ptr', 'd_read_gate_ptr', 'd_write_gate_ptr', 'd_log_tau_ptr'],
)
@triton.jit
def _frac_compute_params_bwd_kernel(
    # forward inputs (recompute)
    dyn_ptr, rw_ptr, mask_ptr,
    dt_bias_ptr, read_gate_ptr, write_gate_ptr, log_tau_ptr,
    # saved softmax normalization [BH, L]
    r_amax_ptr, r_sum_ptr, w_amax_ptr, w_sum_ptr,
    # upstream grads
    du_ptr, dc_ptr, dkappa_ptr, drho_ptr,
    # output grads (direct writes)
    d_dyn_ptr, d_rw_ptr, d_xcv_ptr,
    # output grads (atomic)
    d_dt_bias_ptr, d_read_gate_ptr, d_write_gate_ptr, d_log_tau_ptr,
    # strides: dyn / d_dyn [B, L, 3H]
    dyn_s0, dyn_s1,
    # strides: rw / d_rw [B, L, 2HM]
    rw_s0, rw_s1,
    # strides: mask [B, L]
    mask_s0, mask_s1,
    # strides: du [BH, L, R]
    du_s0, du_s1,
    # strides: dc/dkappa/drho [BH, L, M]
    out_s0, out_s1,
    # strides: d_x_conv [B, L, HR]
    dxcv_s0, dxcv_s1,
    # dims
    L,
    H: tl.constexpr,
    M: tl.constexpr,
    HM: tl.constexpr,
    R: tl.constexpr,
    # config scalars
    read_temp,
    dt_min, dt_max,
    alpha_min, alpha_max,
    lambda_min, lambda_max,
    # meta
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_m = tl.arange(0, M)
    lmask = offs_l < L

    EPS  = 1e-4
    EPS2 = 1e-6

    # ====================== Recompute full forward ======================

    dtb = tl.load(dt_bias_ptr + h).to(tl.float32)
    rg = tl.sigmoid(tl.load(read_gate_ptr).to(tl.float32))
    wg = tl.sigmoid(tl.load(write_gate_ptr).to(tl.float32))
    lt = tl.load(log_tau_ptr + offs_m).to(tl.float32)
    tau = tl.exp(lt)

    db = dyn_ptr + b * dyn_s0
    dt_r = tl.load(db + offs_l * dyn_s1 + h,         mask=lmask, other=0.0).to(tl.float32)
    al_r = tl.load(db + offs_l * dyn_s1 + (H + h),   mask=lmask, other=0.0).to(tl.float32)
    la_r = tl.load(db + offs_l * dyn_s1 + (2*H + h), mask=lmask, other=0.0).to(tl.float32)

    sp = dt_r + dtb
    dt_raw = tl.where(sp > 20.0, sp, tl.log(1.0 + tl.exp(sp)))
    dt = tl.minimum(tl.maximum(dt_raw, dt_min), dt_max)

    sig_a = tl.sigmoid(al_r)
    alpha = alpha_min + sig_a * (alpha_max - alpha_min)
    sig_l = tl.sigmoid(la_r)
    lam = lambda_min + sig_l * (lambda_max - lambda_min)

    alpha_c = tl.maximum(alpha, EPS)
    lam_c = tl.maximum(lam, EPS)
    log_lam_c = tl.log(lam_c)
    lf = tl.exp(log_lam_c / alpha_c)
    ite = lf[:, None] / tl.maximum(tau[None, :], EPS2)

    # Load rw logits
    rb = rw_ptr + b * rw_s0
    rd = tl.load(rb + offs_l[:, None] * rw_s1 + (h * M + offs_m[None, :]),
                 mask=lmask[:, None], other=0.0).to(tl.float32)
    wr = tl.load(rb + offs_l[:, None] * rw_s1 + (HM + h * M + offs_m[None, :]),
                 mask=lmask[:, None], other=0.0).to(tl.float32)

    prior = -alpha[:, None] * lt[None, :]

    # ---- Recompute softmax outputs from saved amax/sum ----
    sm_base = pid_bh * L

    r_amax_val = tl.load(r_amax_ptr + sm_base + offs_l, mask=lmask, other=0.0)
    r_sum_val  = tl.load(r_sum_ptr  + sm_base + offs_l, mask=lmask, other=0.0)
    rl = (prior + rg * rd) / read_temp
    c = tl.exp(rl - r_amax_val[:, None]) / r_sum_val[:, None]

    w_amax_val = tl.load(w_amax_ptr + sm_base + offs_l, mask=lmask, other=0.0)
    w_sum_val  = tl.load(w_sum_ptr  + sm_base + offs_l, mask=lmask, other=0.0)
    wl = prior + wg * wr
    ww = tl.exp(wl - w_amax_val[:, None]) / w_sum_val[:, None]

    # ---- Recompute rho ----
    rho_raw = tl.exp(-dt[:, None] * ite)
    rho = tl.minimum(tl.maximum(rho_raw, EPS2), 1.0 - EPS2)

    # ====================== Load upstream grads ======================

    ob = pid_bh * out_s0
    so = offs_l[:, None] * out_s1 + offs_m[None, :]
    dc_up     = tl.load(dc_ptr     + ob + so, mask=lmask[:, None], other=0.0).to(tl.float32)
    dkappa_up = tl.load(dkappa_ptr + ob + so, mask=lmask[:, None], other=0.0).to(tl.float32)
    drho_up   = tl.load(drho_ptr   + ob + so, mask=lmask[:, None], other=0.0).to(tl.float32)

    # ====================== Backward ======================

    # --- Mask backward ---
    if HAS_MASK:
        mv = tl.load(mask_ptr + b * mask_s0 + offs_l * mask_s1,
                     mask=lmask, other=0.0).to(tl.float32)
        dkappa_pre = dkappa_up * mv[:, None]
        drho_pre   = drho_up   * mv[:, None]
    else:
        dkappa_pre = dkappa_up
        drho_pre   = drho_up

    # --- kappa = (1 - rho) * ww ---
    d_ww        = dkappa_pre * (1.0 - rho)
    d_rho_total = drho_pre - dkappa_pre * ww

    # --- rho clamp (check pre-clamp value, matching PyTorch clamp gradient) ---
    rho_active = (rho_raw >= EPS2) & (rho_raw <= 1.0 - EPS2)
    d_rho_raw = tl.where(rho_active, d_rho_total, 0.0)

    # --- rho = exp(-dt * ite) ---
    d_z      = d_rho_raw * rho
    d_dt_rho = -tl.sum(d_z * ite, axis=1)
    d_ite    = -d_z * dt[:, None]

    # --- ite = lf / tau ---
    d_lf     = tl.sum(d_ite / tau[None, :], axis=1)
    d_lt_ite = -d_ite * ite

    # --- lf = exp(log(lam_c) / alpha_c) ---
    d_q        = d_lf * lf
    d_lam_lf   = tl.where(lam   > EPS, d_q / (alpha_c * lam_c), 0.0)
    d_alpha_lf = tl.where(alpha > EPS, -d_q * log_lam_c / (alpha_c * alpha_c), 0.0)

    # --- Write softmax backward ---
    s_w  = tl.sum(d_ww * ww, axis=1)[:, None]
    d_wl = ww * (d_ww - s_w)
    d_wr_out = d_wl * wg

    # --- Read softmax backward ---
    s_r     = tl.sum(dc_up * c, axis=1)[:, None]
    d_rl    = c * (dc_up - s_r)
    d_raw_r = d_rl / read_temp
    d_rd_out = d_raw_r * rg

    # --- prior = -alpha * lt ---
    d_prior       = d_raw_r + d_wl
    d_alpha_prior = -tl.sum(d_prior * lt[None, :], axis=1)
    d_lt_prior    = -alpha[:, None] * d_prior

    # --- Collect alpha, lambda ---
    d_alpha = d_alpha_prior + d_alpha_lf
    d_lam   = d_lam_lf

    # --- Through sigmoid+affine -> dyn grads ---
    d_al_r = d_alpha * (alpha_max - alpha_min) * sig_a * (1.0 - sig_a)
    d_la_r = d_lam   * (lambda_max - lambda_min) * sig_l * (1.0 - sig_l)

    # --- dt -> softplus -> clamp ---
    dt_active  = (dt_raw >= dt_min) & (dt_raw <= dt_max)
    sig_sp     = tl.where(sp > 20.0, 1.0, tl.sigmoid(sp))
    d_sp       = d_dt_rho * tl.where(dt_active, sig_sp, 0.0)

    # ====================== Store direct grads ======================

    d_db = d_dyn_ptr + b * dyn_s0
    tl.store(d_db + offs_l * dyn_s1 + h,
             d_sp.to(d_dyn_ptr.dtype.element_ty),   mask=lmask)
    tl.store(d_db + offs_l * dyn_s1 + (H + h),
             d_al_r.to(d_dyn_ptr.dtype.element_ty), mask=lmask)
    tl.store(d_db + offs_l * dyn_s1 + (2*H + h),
             d_la_r.to(d_dyn_ptr.dtype.element_ty), mask=lmask)

    d_rb = d_rw_ptr + b * rw_s0
    tl.store(d_rb + offs_l[:, None] * rw_s1 + (h * M + offs_m[None, :]),
             d_rd_out.to(d_rw_ptr.dtype.element_ty), mask=lmask[:, None])
    tl.store(d_rb + offs_l[:, None] * rw_s1 + (HM + h * M + offs_m[None, :]),
             d_wr_out.to(d_rw_ptr.dtype.element_ty), mask=lmask[:, None])

    # ====================== Atomic parameter grads ======================

    d_dtb_block = tl.sum(tl.where(lmask, d_sp, 0.0))
    tl.atomic_add(d_dt_bias_ptr + h, d_dtb_block.to(tl.float32))

    d_rg_block = tl.sum(tl.where(lmask, tl.sum(d_raw_r * rd, axis=1), 0.0))
    tl.atomic_add(d_read_gate_ptr, (d_rg_block * rg * (1.0 - rg)).to(tl.float32))

    d_wg_block = tl.sum(tl.where(lmask, tl.sum(d_wl * wr, axis=1), 0.0))
    tl.atomic_add(d_write_gate_ptr, (d_wg_block * wg * (1.0 - wg)).to(tl.float32))

    d_lt_total = d_lt_prior + d_lt_ite
    d_lt_block = tl.sum(tl.where(lmask[:, None], d_lt_total, 0.0), axis=0)
    tl.atomic_add(d_log_tau_ptr + offs_m, d_lt_block.to(tl.float32))

    # ====================== u permute backward ======================

    du_base   = du_ptr    + pid_bh * du_s0
    dxcv_base = d_xcv_ptr + b * dxcv_s0
    h_R = h * R
    for r_start in tl.range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        rmask  = offs_r < R
        du_vals = tl.load(du_base + offs_l[:, None] * du_s1 + offs_r[None, :],
                          mask=lmask[:, None] & rmask[None, :], other=0.0)
        tl.store(dxcv_base + offs_l[:, None] * dxcv_s1 + (h_R + offs_r[None, :]),
                 du_vals, mask=lmask[:, None] & rmask[None, :])


# ============================================================================
# Backward wrapper
# ============================================================================

def _frac_compute_params_bwd(dyn, rw, attention_mask,
                              dt_bias, read_residual_logit,
                              write_residual_scale, base_log_tau,
                              r_amax, r_sum, w_amax, w_sum,
                              du, dc, dkappa, drho,
                              num_heads, head_dim, num_modes, read_temp,
                              dt_min, dt_max, alpha_min, alpha_max,
                              lambda_min, lambda_max):
    B, L, _ = dyn.shape
    H = num_heads
    R = head_dim
    M = num_modes
    HM = H * M
    BH = B * H
    BLOCK_R = triton.next_power_of_2(R)

    d_dyn    = torch.empty_like(dyn)
    d_rw     = torch.empty_like(rw)
    d_x_conv = torch.empty(B, L, H * R, device=dyn.device, dtype=dyn.dtype)

    # Atomic accumulators
    d_dt_bias    = torch.zeros_like(dt_bias, dtype=torch.float32)
    d_read_gate  = torch.zeros_like(read_residual_logit, dtype=torch.float32)
    d_write_gate = torch.zeros_like(write_residual_scale, dtype=torch.float32)
    d_log_tau    = torch.zeros_like(base_log_tau, dtype=torch.float32)

    has_mask = attention_mask is not None
    if has_mask:
        ms0, ms1 = attention_mask.stride()
    else:
        attention_mask = d_dyn
        ms0, ms1 = 0, 0

    grid = lambda META: (BH, triton.cdiv(L, META['BLOCK_L']))
    _frac_compute_params_bwd_kernel[grid](
        dyn, rw, attention_mask,
        dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
        r_amax, r_sum, w_amax, w_sum,
        du, dc, dkappa, drho,
        d_dyn, d_rw, d_x_conv,
        d_dt_bias, d_read_gate, d_write_gate, d_log_tau,
        dyn.stride(0), dyn.stride(1),
        rw.stride(0), rw.stride(1),
        ms0, ms1,
        du.stride(0), du.stride(1),
        dc.stride(0), dc.stride(1),
        d_x_conv.stride(0), d_x_conv.stride(1),
        L, H, M, HM, R,
        read_temp,
        dt_min, dt_max, alpha_min, alpha_max, lambda_min, lambda_max,
        BLOCK_R=BLOCK_R,
        HAS_MASK=has_mask,
    )

    return d_dyn, d_rw, d_x_conv, d_dt_bias, d_read_gate, d_write_gate, d_log_tau


# ============================================================================
# Autograd Function
# ============================================================================

class _FracFusedParamsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dyn, rw, x_conv, attention_mask,
                dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
                num_heads, head_dim, num_modes, read_temp,
                dt_min, dt_max, alpha_min, alpha_max, lambda_min, lambda_max):
        has_mask = attention_mask is not None
        if not has_mask:
            attention_mask = dyn.new_empty(0)
        real_mask = attention_mask if has_mask else None
        (u_flat, c_flat, kappa_flat, rho_flat,
         r_amax, r_sum, w_amax, w_sum) = _frac_compute_params_fwd(
            dyn, rw, x_conv, real_mask,
            dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
            num_heads, head_dim, num_modes, read_temp,
            dt_min, dt_max, alpha_min, alpha_max, lambda_min, lambda_max,
        )
        # Save amax/sum instead of c/ww/rho 
        ctx.save_for_backward(dyn, rw, attention_mask,
                              dt_bias, read_residual_logit,
                              write_residual_scale, base_log_tau,
                              r_amax, r_sum, w_amax, w_sum)
        ctx.has_mask = has_mask
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.num_modes = num_modes
        ctx.read_temp = read_temp
        ctx.dt_min = dt_min
        ctx.dt_max = dt_max
        ctx.alpha_min = alpha_min
        ctx.alpha_max = alpha_max
        ctx.lambda_min = lambda_min
        ctx.lambda_max = lambda_max
        return u_flat, c_flat, kappa_flat, rho_flat

    @staticmethod
    def backward(ctx, du, dc, dkappa, drho):
        (dyn, rw, attention_mask,
         dt_bias, read_residual_logit,
         write_residual_scale, base_log_tau,
         r_amax, r_sum, w_amax, w_sum) = ctx.saved_tensors
        if not ctx.has_mask:
            attention_mask = None

        (d_dyn, d_rw, d_x_conv,
         d_dt_bias, d_read_gate,
         d_write_gate, d_log_tau) = _frac_compute_params_bwd(
            dyn, rw, attention_mask,
            dt_bias, read_residual_logit, write_residual_scale, base_log_tau,
            r_amax, r_sum, w_amax, w_sum,
            du, dc, dkappa, drho,
            ctx.num_heads, ctx.head_dim, ctx.num_modes, ctx.read_temp,
            ctx.dt_min, ctx.dt_max, ctx.alpha_min, ctx.alpha_max,
            ctx.lambda_min, ctx.lambda_max,
        )

        return (d_dyn, d_rw, d_x_conv, None,
                d_dt_bias, d_read_gate, d_write_gate, d_log_tau,
                None, None, None, None,
                None, None, None, None, None, None)


# ============================================================================
# Public API
# ============================================================================

def frac_compute_scan_params(model, x_conv, x, attention_mask=None):
    """Compute scan inputs (u, c, kappa, rho) from post-conv features.
    """
    dyn = F.linear(x, model.dyn_proj.weight, model.dyn_proj.bias)
    rw = F.linear(x_conv, model.rw_proj.weight)
    return _FracFusedParamsFn.apply(
        dyn, rw, x_conv, attention_mask,
        model.dt_bias, model.read_residual_logit, model.write_residual_scale,
        model.base_log_tau,
        model.num_heads, model.head_dim, model.num_modes, model.read_temp,
        model.config.frac_dt_min, model.config.frac_dt_max,
        model.config.frac_alpha_min, model.config.frac_alpha_max,
        model.config.frac_lambda_min, model.config.frac_lambda_max,
    )
