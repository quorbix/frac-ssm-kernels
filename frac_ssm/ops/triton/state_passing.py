import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_R": block_r}, num_warps=warps, num_stages=num_stages)
    for block_r in [32, 64, 128]
    for warps in [2, 4, 8]
    for num_stages in [1, 2, 3, 4, 5]
]


@triton.autotune(configs=configs, key=["M", "R"])
@triton.jit
def _frac_state_passing_fwd_kernel(states_ptr, chunk_decay_ptr, initial_state_ptr,
                                   states_acc_ptr, final_state_ptr,
                                   states_stride_batch, states_stride_chunk, states_stride_r, states_stride_m,
                                   chunk_decay_stride_batch, chunk_decay_stride_chunk, chunk_decay_stride_m,
                                   initial_state_stride_batch, initial_state_stride_r, initial_state_stride_m,
                                   states_acc_stride_batch, states_acc_stride_chunk, states_acc_stride_r, states_acc_stride_m,
                                   final_state_stride_batch, final_state_stride_r, final_state_stride_m,
                                   nchunks, R: tl.constexpr, M: tl.constexpr,
                                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, HAS_INITIAL_STATE: tl.constexpr,
                                   ACC_DTYPE: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_r_i64 = pid_r.to(tl.int64)
    pid_m_i64 = pid_m.to(tl.int64)

    states_ptr += pid_b_i64 * states_stride_batch + pid_r_i64 * BLOCK_SIZE_R * states_stride_r + pid_m_i64 * BLOCK_SIZE_M * states_stride_m
    chunk_decay_ptr += pid_b_i64 * chunk_decay_stride_batch + pid_m_i64 * BLOCK_SIZE_M * chunk_decay_stride_m
    states_acc_ptr += pid_b_i64 * states_acc_stride_batch + pid_r_i64 * BLOCK_SIZE_R * states_acc_stride_r + pid_m_i64 * BLOCK_SIZE_M * states_acc_stride_m
    final_state_ptr += pid_b_i64 * final_state_stride_batch + pid_r_i64 * BLOCK_SIZE_R * final_state_stride_r + pid_m_i64 * BLOCK_SIZE_M * final_state_stride_m

    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_r = tl.arange(0, BLOCK_SIZE_R)

    mask_r = (pid_r * BLOCK_SIZE_R + offset_r < R  )[:, None]        # [BLOCK_SIZE_R, 1]
    mask_m = (pid_m * BLOCK_SIZE_M + offset_m < M)[None, :]          # [1, BLOCK_SIZE_M]
    mask_rm = mask_r & mask_m   # [BLOCK_SIZE_R, BLOCK_SIZE_M] tile

    if HAS_INITIAL_STATE:
        initial_state_ptr += pid_b_i64 * initial_state_stride_batch + pid_r_i64 * BLOCK_SIZE_R * initial_state_stride_r + pid_m_i64 * BLOCK_SIZE_M * initial_state_stride_m
        H = tl.load(initial_state_ptr + offset_r[:, None] * initial_state_stride_r + offset_m[None, :] * initial_state_stride_m,
                    mask=mask_rm, other=0.0).to(ACC_DTYPE)
    else:
        H = tl.zeros((BLOCK_SIZE_R, BLOCK_SIZE_M), dtype=ACC_DTYPE)

    for i in range(0, nchunks):
        tl.store(states_acc_ptr + (i * states_acc_stride_chunk) + offset_r[:, None] * states_acc_stride_r + offset_m[None, :] * states_acc_stride_m, H,
            mask=mask_rm)


        state = tl.load(states_ptr + (i*states_stride_chunk) + offset_r[:, None] * states_stride_r + offset_m[None, :] * states_stride_m,
                        mask=mask_rm, other=0.0).to(ACC_DTYPE)
        chunk_decay = tl.load(chunk_decay_ptr + (i * chunk_decay_stride_chunk) + offset_m[None, :] * chunk_decay_stride_m,
                               mask=mask_m, other=0.0).to(tl.float32)
        chunk_decay = tl.exp(chunk_decay).to(ACC_DTYPE)


        H = chunk_decay * H + state


    tl.store(final_state_ptr + offset_r[:, None] * final_state_stride_r + offset_m[None, :] * final_state_stride_m, H,
             mask=mask_rm)

    

def _frac_state_passing_fwd(states, chunk_decay, initial_state=None, out_dtype=None):
    B, nchunks, R, M = states.shape

    out_dtype = states.dtype if out_dtype is None else out_dtype
    states_acc = torch.empty(B, nchunks, R, M, device=states.device, dtype=out_dtype)
    final_state = torch.empty(B, R, M, device=states.device, dtype=states.dtype)
    grid = lambda META: (B, triton.cdiv(R, META["BLOCK_SIZE_R"]), triton.cdiv(M, META["BLOCK_SIZE_M"]))

    if initial_state is None:
        initial_state_stride_b, initial_state_stride_r, initial_state_stride_m = 0,0,0
    else:
        initial_state_stride_b = initial_state.stride(0)
        initial_state_stride_r = initial_state.stride(1)
        initial_state_stride_m = initial_state.stride(2)

    _frac_state_passing_fwd_kernel[grid](states, chunk_decay, initial_state,
                                        states_acc, final_state,
                                        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
                                        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
                                        initial_state_stride_b, initial_state_stride_r, initial_state_stride_m,
                                        states_acc.stride(0), states_acc.stride(1), states_acc.stride(2), states_acc.stride(3),
                                        final_state.stride(0), final_state.stride(1), final_state.stride(2),
                                        nchunks, R, M, HAS_INITIAL_STATE=initial_state is not None,
                                        ACC_DTYPE=tl.bfloat16 if states.dtype == torch.bfloat16 else tl.float32)
    return states_acc, final_state



configs_bwd = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_R": block_r}, num_warps=warps, num_stages=num_stages)
    for block_r in [32, 64, 128]
    for warps in [2, 4, 8]
    for num_stages in [1, 2, 3, 4, 5]
]


@triton.autotune(configs=configs_bwd, key=["M", "R"], reset_to_zero=["d_chunk_decay_ptr"])
@triton.jit
def _frac_state_passing_bwd_kernel(
    d_states_acc_ptr, d_final_state_ptr, states_acc_ptr, chunk_decay_ptr,
    d_states_ptr, d_chunk_decay_ptr, d_initial_state_ptr,
    d_states_acc_stride_batch, d_states_acc_stride_chunk, d_states_acc_stride_r, d_states_acc_stride_m,
    d_final_state_stride_batch, d_final_state_stride_r, d_final_state_stride_m,
    states_acc_stride_batch, states_acc_stride_chunk, states_acc_stride_r, states_acc_stride_m,
    chunk_decay_stride_batch, chunk_decay_stride_chunk, chunk_decay_stride_m,
    d_states_stride_batch, d_states_stride_chunk, d_states_stride_r, d_states_stride_m,
    d_chunk_decay_stride_batch, d_chunk_decay_stride_chunk, d_chunk_decay_stride_m,
    d_initial_state_stride_batch, d_initial_state_stride_r, d_initial_state_stride_m,
    nchunks, R: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, HAS_INITIAL_STATE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    pid_b_i64 = pid_b.to(tl.int64)
    pid_r_i64 = pid_r.to(tl.int64)
    pid_m_i64 = pid_m.to(tl.int64)

    # Offset pointers for this block
    d_states_acc_ptr += pid_b_i64 * d_states_acc_stride_batch + pid_r_i64 * BLOCK_SIZE_R * d_states_acc_stride_r + pid_m_i64 * BLOCK_SIZE_M * d_states_acc_stride_m
    d_final_state_ptr += pid_b_i64 * d_final_state_stride_batch + pid_r_i64 * BLOCK_SIZE_R * d_final_state_stride_r + pid_m_i64 * BLOCK_SIZE_M * d_final_state_stride_m
    states_acc_ptr += pid_b_i64 * states_acc_stride_batch + pid_r_i64 * BLOCK_SIZE_R * states_acc_stride_r + pid_m_i64 * BLOCK_SIZE_M * states_acc_stride_m
    chunk_decay_ptr += pid_b_i64 * chunk_decay_stride_batch + pid_m_i64 * BLOCK_SIZE_M * chunk_decay_stride_m
    d_states_ptr += pid_b_i64 * d_states_stride_batch + pid_r_i64 * BLOCK_SIZE_R * d_states_stride_r + pid_m_i64 * BLOCK_SIZE_M * d_states_stride_m
    d_chunk_decay_ptr += pid_b_i64 * d_chunk_decay_stride_batch + pid_m_i64 * BLOCK_SIZE_M * d_chunk_decay_stride_m

    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_r = tl.arange(0, BLOCK_SIZE_R)

    mask_r = (pid_r * BLOCK_SIZE_R + offset_r < R)[:, None]
    mask_m = (pid_m * BLOCK_SIZE_M + offset_m < M)[None, :]
    mask_m_1d = pid_m * BLOCK_SIZE_M + offset_m < M  # 1D mask for atomic_add
    mask_rm = mask_r & mask_m

    dH = tl.load(d_final_state_ptr + offset_r[:, None] * d_final_state_stride_r + offset_m[None, :] * d_final_state_stride_m,
                 mask=mask_rm, other=0.0).to(ACC_DTYPE)

    for i in range(nchunks):
        n = nchunks - 1 - i

        H_before = tl.load(states_acc_ptr + (n * states_acc_stride_chunk) + offset_r[:, None] * states_acc_stride_r + offset_m[None, :] * states_acc_stride_m,
                           mask=mask_rm, other=0.0).to(ACC_DTYPE)

        cd = tl.load(chunk_decay_ptr + (n * chunk_decay_stride_chunk) + offset_m[None, :] * chunk_decay_stride_m,
                     mask=mask_m, other=0.0).to(tl.float32)
        decay = tl.exp(cd).to(ACC_DTYPE)

        # d_states[n] = dH because H_next = decay * H_before + states[n]
        tl.store(d_states_ptr + (n * d_states_stride_chunk) + offset_r[:, None] * d_states_stride_r + offset_m[None, :] * d_states_stride_m,
                 dH, mask=mask_rm)

        # d_chunk_decay[:, n] = (dH * decay * H_before).sum(dim=R) partial sum, so we need atomic_add
        d_cd_partial = tl.sum(dH * decay * H_before, axis=0)
        tl.atomic_add(d_chunk_decay_ptr + (n * d_chunk_decay_stride_chunk) + offset_m * d_chunk_decay_stride_m,
                      d_cd_partial, mask=mask_m_1d)

        d_states_acc_n = tl.load(d_states_acc_ptr + (n * d_states_acc_stride_chunk) + offset_r[:, None] * d_states_acc_stride_r + offset_m[None, :] * d_states_acc_stride_m,
                                 mask=mask_rm, other=0.0).to(ACC_DTYPE)
        dH = dH * decay + d_states_acc_n

    if HAS_INITIAL_STATE:
        d_initial_state_ptr += pid_b_i64 * d_initial_state_stride_batch + pid_r_i64 * BLOCK_SIZE_R * d_initial_state_stride_r + pid_m_i64 * BLOCK_SIZE_M * d_initial_state_stride_m
        tl.store(d_initial_state_ptr + offset_r[:, None] * d_initial_state_stride_r + offset_m[None, :] * d_initial_state_stride_m,
                 dH, mask=mask_rm)


def _frac_state_passing_bwd(d_states_acc, d_final_state, states_acc, chunk_decay, has_initial_state=False):
    B, nchunks, R, M = states_acc.shape

    d_states = torch.empty_like(states_acc)
    d_chunk_decay = torch.zeros((B, nchunks, M), device=chunk_decay.device, dtype=torch.float32)

    if has_initial_state:
        d_initial_state = torch.empty(B, R, M, device=states_acc.device, dtype=states_acc.dtype)
        d_init_stride_b, d_init_stride_r, d_init_stride_m = d_initial_state.stride(0), d_initial_state.stride(1), d_initial_state.stride(2)
    else:
        d_initial_state = None
        d_init_stride_b, d_init_stride_r, d_init_stride_m = 0, 0, 0

    grid = lambda META: (B, triton.cdiv(R, META["BLOCK_SIZE_R"]), triton.cdiv(M, META["BLOCK_SIZE_M"]))

    _frac_state_passing_bwd_kernel[grid](
        d_states_acc, d_final_state, states_acc, chunk_decay,
        d_states, d_chunk_decay, d_initial_state,
        d_states_acc.stride(0), d_states_acc.stride(1), d_states_acc.stride(2), d_states_acc.stride(3),
        d_final_state.stride(0), d_final_state.stride(1), d_final_state.stride(2),
        states_acc.stride(0), states_acc.stride(1), states_acc.stride(2), states_acc.stride(3),
        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
        d_states.stride(0), d_states.stride(1), d_states.stride(2), d_states.stride(3),
        d_chunk_decay.stride(0), d_chunk_decay.stride(1), d_chunk_decay.stride(2),
        d_init_stride_b, d_init_stride_r, d_init_stride_m,
        nchunks, R, M,
        HAS_INITIAL_STATE=has_initial_state,
        ACC_DTYPE=tl.bfloat16 if states_acc.dtype == torch.bfloat16 else tl.float32,
    )

    return d_states, d_chunk_decay.to(chunk_decay.dtype), d_initial_state


# ============================================================================
# Public API and autograd wrappers
# ============================================================================


@torch.library.custom_op("frac_ssm::state_passing_fwd", mutates_args=())
def _state_passing_fwd_op(
    states: torch.Tensor,
    chunk_decay: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _frac_state_passing_fwd(states, chunk_decay, initial_state)


@_state_passing_fwd_op.register_fake
def _(states, chunk_decay, initial_state):
    B, nchunks, R, M = states.shape
    return (
        torch.empty(B, nchunks, R, M, device=states.device, dtype=states.dtype),
        torch.empty(B, R, M, device=states.device, dtype=states.dtype),
    )


@torch.library.custom_op("frac_ssm::state_passing_fwd_noinit", mutates_args=())
def _state_passing_fwd_noinit_op(
    states: torch.Tensor,
    chunk_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _frac_state_passing_fwd(states, chunk_decay, None)


@_state_passing_fwd_noinit_op.register_fake
def _(states, chunk_decay):
    B, nchunks, R, M = states.shape
    return (
        torch.empty(B, nchunks, R, M, device=states.device, dtype=states.dtype),
        torch.empty(B, R, M, device=states.device, dtype=states.dtype),
    )


@torch.library.custom_op("frac_ssm::state_passing_bwd", mutates_args=())
def _state_passing_bwd_op(
    d_states_acc: torch.Tensor,
    d_final_state: torch.Tensor,
    states_acc: torch.Tensor,
    chunk_decay: torch.Tensor,
    has_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_states, d_chunk_decay, d_initial_state = _frac_state_passing_bwd(
        d_states_acc, d_final_state, states_acc, chunk_decay, has_initial_state
    )
    if d_initial_state is None:
        d_initial_state = torch.empty(0, device=states_acc.device, dtype=states_acc.dtype)
    return d_states, d_chunk_decay, d_initial_state


@_state_passing_bwd_op.register_fake
def _(d_states_acc, d_final_state, states_acc, chunk_decay, has_initial_state):
    B, nchunks, R, M = states_acc.shape
    d_initial_shape = (B, R, M) if has_initial_state else (0,)
    return (
        torch.empty_like(states_acc),
        torch.empty(B, nchunks, M, device=chunk_decay.device, dtype=chunk_decay.dtype),
        torch.empty(d_initial_shape, device=states_acc.device, dtype=states_acc.dtype),
    )


def _state_passing_setup_ctx(ctx, inputs, output):
    states, chunk_decay, initial_state = inputs
    states_acc, _ = output
    ctx.save_for_backward(states_acc, chunk_decay)


def _state_passing_noinit_setup_ctx(ctx, inputs, output):
    states, chunk_decay = inputs
    states_acc, _ = output
    ctx.save_for_backward(states_acc, chunk_decay)


def _state_passing_backward(ctx, d_states_acc, d_final_state):
    states_acc, chunk_decay = ctx.saved_tensors
    d_states, d_chunk_decay, d_initial_state = _state_passing_bwd_op(
        d_states_acc.contiguous(),
        d_final_state.contiguous(),
        states_acc,
        chunk_decay,
        True,
    )
    return d_states, d_chunk_decay, d_initial_state


def _state_passing_noinit_backward(ctx, d_states_acc, d_final_state):
    states_acc, chunk_decay = ctx.saved_tensors
    d_states, d_chunk_decay, _ = _state_passing_bwd_op(
        d_states_acc.contiguous(),
        d_final_state.contiguous(),
        states_acc,
        chunk_decay,
        False,
    )
    return d_states, d_chunk_decay


_state_passing_fwd_op.register_autograd(
    _state_passing_backward, setup_context=_state_passing_setup_ctx
)
_state_passing_fwd_noinit_op.register_autograd(
    _state_passing_noinit_backward, setup_context=_state_passing_noinit_setup_ctx
)


def state_passing(states, chunk_decay, initial_state=None):
    """Pass state between chunks."""
    assert states.dim() == 4, f"states must be 4D (B, N, R, M), got {states.dim()}D"
    assert chunk_decay.dim() == 3, f"chunk_decay must be 3D (B, N, M), got {chunk_decay.dim()}D"
    assert states.is_cuda and chunk_decay.is_cuda, "states and chunk_decay must be on CUDA"
    assert states.dtype in (torch.float32, torch.bfloat16), (
        f"states dtype must be float32 or bfloat16, got {states.dtype}"
    )
    assert chunk_decay.dtype == torch.float32, (
        f"chunk_decay dtype must be float32, got {chunk_decay.dtype}"
    )
    if initial_state is None:
        return _state_passing_fwd_noinit_op(states, chunk_decay)
    assert initial_state.dim() == 3, (
        f"initial_state must be 3D (B, R, M), got {initial_state.dim()}D"
    )
    assert initial_state.is_cuda, "initial_state must be on CUDA"
    assert initial_state.dtype in (torch.float32, torch.bfloat16), (
        f"initial_state dtype must be float32 or bfloat16, got {initial_state.dtype}"
    )
    return _state_passing_fwd_op(states, chunk_decay, initial_state)
