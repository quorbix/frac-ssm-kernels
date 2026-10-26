"""Fully Sequential CuTe implementation of state passing."""

import math
from typing import Union

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, const_expr
from cutlass import cute
from cutlass.cutlass_dsl import dsl_user_op

from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.utils import make_fake_tensor, torch2cute_dtype_map


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


class StatePassSeqFwd:
    def __init__(self, dtype, out_dtype, R, M):
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.R = R
        self.M = M

        self.R_TILE = 16
        self.R_TILES = math.ceil(R / self.R_TILE)
        self.M_TILE = 8 if M == 8 else 16
        self.M_TILES = M // self.M_TILE
        self.num_threads = self.R_TILE * self.M_TILE

        assert M == 8 or M % 16 == 0, f"M must be 8 or a multiple of 16, got {M}"

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,   # (B, nchunks, R, M)
        mCD: cute.Tensor,  # (B, nchunks, M), fp32
        mY: cute.Tensor,   # (B, nchunks, R, M)
        mFS: cute.Tensor,  # (B, R, M), fp32
        initial_state: Union[cute.Tensor, None],
        stream: cuda.CUstream,
    ):
        B = mX.shape[0]
        self.kernel(mX, mCD, mY, mFS, initial_state).launch(
            grid=[B, self.R_TILES, self.M_TILES],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mCD: cute.Tensor,
        mY: cute.Tensor,
        mFS: cute.Tensor,
        initial_state: Union[cute.Tensor, None],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_r_tile, bidx_m_tile = cute.arch.block_idx()

        nchunks = mX.shape[1]

        local_r = tidx // self.M_TILE
        local_m = tidx - local_r * self.M_TILE
        r = bidx_r_tile * self.R_TILE + local_r
        m = bidx_m_tile * self.M_TILE + local_m

        if r < self.R and m < self.M:
            running = 0.0
            if const_expr(initial_state is not None):
                running = initial_state[bidx_batch, r, m]

            for n in cutlass.range(0, nchunks, 1, unroll=1):
                mY[bidx_batch, n, r, m] = running.to(mY.element_type)
                decay = cute.exp(mCD[bidx_batch, n, m])
                state = mX[bidx_batch, n, r, m].to(Float32)
                running = decay * running + state

            mFS[bidx_batch, r, m] = running


class StatePassSeqBwd:
    def __init__(self, dtype, R, M):
        self.dtype = dtype
        self.R = R
        self.M = M

        self.R_TILE = 16
        self.R_TILES = math.ceil(R / self.R_TILE)
        self.M_TILE = 8 if M == 8 else 16
        self.M_TILES = M // self.M_TILE
        self.num_threads = self.R_TILE * self.M_TILE

        assert M == 8 or M % 16 == 0, f"M must be 8 or a multiple of 16, got {M}"

    @cute.jit
    def __call__(
        self,
        mDY: cute.Tensor,   # (B, nchunks, R, M), fp32
        mDFS: cute.Tensor,  # (B, R, M), fp32
        mY: cute.Tensor,    # (B, nchunks, R, M), fp32
        mCD: cute.Tensor,   # (B, nchunks, M), fp32
        mDX: cute.Tensor,   # (B, nchunks, R, M), fp32
        mDCD: cute.Tensor,  # (B, nchunks, M), fp32
        mDIS: Union[cute.Tensor, None],
        stream: cuda.CUstream,
    ):
        B = mY.shape[0]
        self.kernel(mDY, mDFS, mY, mCD, mDX, mDCD, mDIS).launch(
            grid=[B, self.R_TILES, self.M_TILES],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDY: cute.Tensor,
        mDFS: cute.Tensor,
        mY: cute.Tensor,
        mCD: cute.Tensor,
        mDX: cute.Tensor,
        mDCD: cute.Tensor,
        mDIS: Union[cute.Tensor, None],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_r_tile, bidx_m_tile = cute.arch.block_idx()
        nchunks = mDY.shape[1]

        local_r = tidx // self.M_TILE
        local_m = tidx - local_r * self.M_TILE
        r = bidx_r_tile * self.R_TILE + local_r
        m = bidx_m_tile * self.M_TILE + local_m

        if r < self.R and m < self.M:
            running = mDFS[bidx_batch, r, m]

            for i in cutlass.range(0, nchunks, 1, unroll=1):
                n = nchunks - 1 - i
                h_before = mY[bidx_batch, n, r, m].to(Float32)
                decay = cute.exp(mCD[bidx_batch, n, m])

                mDX[bidx_batch, n, r, m] = running.to(mDX.element_type)
                d_cd = running * decay * h_before
                cute.arch.atomic_add(elem_pointer(mDCD, (bidx_batch, n, m)).llvm_ptr,
                                    d_cd, sem="relaxed", scope="gpu")

                running = running * decay + mDY[bidx_batch, n, r, m].to(Float32)

            if const_expr(mDIS is not None):
                mDIS[bidx_batch, r, m] = running.to(mDIS.element_type)


@jit_cache
def _compile_state_passing_fwd(dtype, out_dtype, R, M, has_initial_state):
    B_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_states = make_fake_tensor(dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks_sym, M), divisibility=1)
    fake_states_acc = make_fake_tensor(out_dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_final_state = make_fake_tensor(Float32, (B_sym, R, M), divisibility=1)
    fake_initial_state = (
        make_fake_tensor(Float32, (B_sym, R, M), divisibility=1)
        if has_initial_state
        else None
    )

    return cute.compile(
        StatePassSeqFwd(dtype, out_dtype, R=R, M=M),
        fake_states,
        fake_chunk_decay,
        fake_states_acc,
        fake_final_state,
        fake_initial_state,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@jit_cache
def _compile_state_passing_bwd(dtype, R, M, has_initial_state):
    B_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_d_states_acc = make_fake_tensor(dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_d_final_state = make_fake_tensor(Float32, (B_sym, R, M), divisibility=1)
    fake_states_acc = make_fake_tensor(dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks_sym, M), divisibility=1)
    fake_d_states = make_fake_tensor(dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_d_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks_sym, M), divisibility=1)
    fake_d_initial_state = (
        make_fake_tensor(dtype, (B_sym, R, M), divisibility=1)
        if has_initial_state
        else None
    )

    return cute.compile(
        StatePassSeqBwd(dtype, R=R, M=M),
        fake_d_states_acc,
        fake_d_final_state,
        fake_states_acc,
        fake_chunk_decay,
        fake_d_states,
        fake_d_chunk_decay,
        fake_d_initial_state,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _frac_state_passing_fwd(states, chunk_decay, initial_state=None, out_dtype=None):
    B, nchunks, R, M = states.shape
    if M != 8 and M % 16 != 0:
        raise ValueError(f"cute.seq state_passing requires M=8 or M multiple of 16, got M={M}")

    out_dtype = states.dtype if out_dtype is None else out_dtype
    states_acc = torch.empty(B, nchunks, R, M, device=states.device, dtype=out_dtype)
    final_state = torch.empty(B, R, M, device=states.device, dtype=torch.float32)
    if B == 0:
        return states_acc, final_state

    dtype = torch2cute_dtype_map[states.dtype]
    cute_out_dtype = torch2cute_dtype_map[out_dtype]
    _compile_state_passing_fwd(dtype, cute_out_dtype, R, M, initial_state is not None)(
        states,
        chunk_decay,
        states_acc,
        final_state,
        initial_state,
    )
    return states_acc, final_state


def _frac_state_passing_bwd(
    d_states_acc,
    d_final_state,
    states_acc,
    chunk_decay,
    has_initial_state=False,
):
    B, nchunks, R, M = states_acc.shape
    if M != 8 and M % 16 != 0:
        raise ValueError(f"cute.seq state_passing requires M=8 or M multiple of 16, got M={M}")

    d_states = torch.empty_like(states_acc)
    d_chunk_decay = torch.zeros((B, nchunks, M), device=chunk_decay.device, dtype=torch.float32)
    d_initial_state = (
        torch.empty(B, R, M, device=states_acc.device, dtype=states_acc.dtype)
        if has_initial_state
        else None
    )
    if B == 0:
        return d_states, d_chunk_decay.to(chunk_decay.dtype), d_initial_state

    dtype = torch2cute_dtype_map[states_acc.dtype]
    _compile_state_passing_bwd(dtype, R, M, has_initial_state)(
        d_states_acc,
        d_final_state,
        states_acc,
        chunk_decay,
        d_states,
        d_chunk_decay,
        d_initial_state,
    )

    return d_states, d_chunk_decay.to(chunk_decay.dtype), d_initial_state


# ============================================================================
# Public API (autograd-wrapped)
# ============================================================================


@torch.library.custom_op("frac_ssm::state_passing_fwd_seq_cute", mutates_args=())
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
        torch.empty(B, R, M, device=states.device, dtype=torch.float32),
    )


@torch.library.custom_op("frac_ssm::state_passing_fwd_noinit_seq_cute", mutates_args=())
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
        torch.empty(B, R, M, device=states.device, dtype=torch.float32),
    )


@torch.library.custom_op("frac_ssm::state_passing_bwd_seq_cute", mutates_args=())
def _state_passing_bwd_op(
    d_states_acc: torch.Tensor,
    d_final_state: torch.Tensor,
    states_acc: torch.Tensor,
    chunk_decay: torch.Tensor,
    has_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_states, d_chunk_decay, d_initial_state = _frac_state_passing_bwd(
        d_states_acc,
        d_final_state,
        states_acc,
        chunk_decay,
        has_initial_state,
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
    assert states.dtype == torch.float32, f"states dtype must be float32, got {states.dtype}"
    assert chunk_decay.dtype == torch.float32, (
        f"chunk_decay dtype must be float32, got {chunk_decay.dtype}"
    )
    if initial_state is None:
        return _state_passing_fwd_noinit_op(states, chunk_decay)
    assert initial_state.dim() == 3, (
        f"initial_state must be 3D (B, R, M), got {initial_state.dim()}D"
    )
    assert initial_state.is_cuda, "initial_state must be on CUDA"
    assert initial_state.dtype == torch.float32, (
        f"initial_state dtype must be float32, got {initial_state.dtype}"
    )
    return _state_passing_fwd_op(states, chunk_decay, initial_state)
