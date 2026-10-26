import math
from typing import Union

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import BFloat16, Float32, const_expr
from cutlass import cute
from cutlass.cutlass_dsl import dsl_user_op

from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.utils import make_fake_tensor, torch2cute_dtype_map


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def state_fma(a, b, c, dtype: cutlass.Constexpr):
    """a * b + c in the carried-state dtype; BF16 states use a BF16 FMA as in Triton."""
    if const_expr(dtype is BFloat16):
        return cute.arch.fma_bf16(a, b, c)
    return a * b + c

"""
For both fwd and bwd, writing the first or last segment in the first pass to 
avoid replaying it in the third pass was slightly slower due to uneven work
around the barriers, so we replay all segments.
"""
class StatePassFwd:
    R_TILE = 16
    
    def __init__(self, dtype, out_dtype, R, M, nchunks, num_segments):
        self.dtype = dtype
        self.acc_dtype = BFloat16 if dtype is BFloat16 else Float32
        self.out_dtype = out_dtype
        self.R = R
        self.M = M
        self.nchunks = nchunks

        self.num_segments = num_segments
        self.R_TILES = math.ceil(R / self.R_TILE)
        self.M_TILE = 8 if M == 8 else 16
        self.M_TILES = M // self.M_TILE
        self.segment_nchunks = math.ceil(nchunks / self.num_segments)
        self.elems_per_segment = self.R_TILE * self.M_TILE
        self.num_threads = self.num_segments * self.elems_per_segment

        assert M == 8 or M % 16 == 0, f"M must be 8 or a multiple of 16, got {M}"
        assert self.num_segments in (1, 4), (
            f"num_segments must be 1 or 4, got {self.num_segments}"
        )

    @classmethod
    def _state_passing_num_segments(cls, B, R, M, nchunks):
        m_tile = 8 if M == 8 else 16
        num_ctas = B * math.ceil(R / cls.R_TILE) * (M // m_tile)
        if nchunks < 64:
            return 1
        if num_ctas > 256:
            return 1
        return 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,   # (B, nchunks, R, M)
        mCD: cute.Tensor,  # (B, nchunks, M), fp32
        mY: cute.Tensor,   # (B, nchunks, R, M)
        mFS: cute.Tensor,  # (B, R, M), states dtype
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

        segment = tidx // self.elems_per_segment
        elem = tidx % self.elems_per_segment
        local_r = elem // self.M_TILE
        local_m = elem % self.M_TILE
        r = bidx_r_tile * self.R_TILE + local_r
        m = bidx_m_tile * self.M_TILE + local_m

        if const_expr(self.num_segments == 1):
            # Baseline one thread owns one (batch, r, m) and scans all chunks.
            if r < self.R and m < self.M:
                running = self.acc_dtype(0.0)
                if const_expr(initial_state is not None):
                    running = initial_state[bidx_batch, r, m].to(self.acc_dtype)

                for n in cutlass.range(0, self.nchunks, 1, unroll=1):
                    mY[bidx_batch, n, r, m] = running.to(mY.element_type)
                    decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)
                    state = mX[bidx_batch, n, r, m].to(self.acc_dtype)
                    running = state_fma(decay, running, state, self.acc_dtype)

                mFS[bidx_batch, r, m] = running.to(mFS.element_type)
        else:
            # Parallel path computes segment summaries, then replays later segments with their carries.
            smem = cutlass.memory.SmemAllocator()
            smem_layout = cute.make_ordered_layout(
                (self.num_segments, self.R_TILE, self.M_TILE), order=(2, 1, 0)
            )
            sPartDecay = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sPartState = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sCarryDecay = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sCarryState = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )


            # First pass scans within segments.
            segment_start = segment * self.segment_nchunks
            segment_decay = 1.0
            segment_state = self.acc_dtype(0.0)
            for i_n in cutlass.range(0, self.segment_nchunks, 1, unroll=1):
                n = segment_start + i_n
                if n < self.nchunks and r < self.R and m < self.M:
                    decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)
                    state = mX[bidx_batch, n, r, m].to(self.acc_dtype)
                    segment_state = state_fma(decay, segment_state, state, self.acc_dtype)
                    segment_decay = decay.to(Float32) * segment_decay

            if r < self.R and m < self.M:
                sPartDecay[segment, local_r, local_m] = segment_decay
                sPartState[segment, local_r, local_m] = segment_state.to(Float32)

            cute.arch.barrier()

            # Second pass scans across segments.
            if segment == 0 and r < self.R and m < self.M:
                carry_decay = 1.0
                carry_state = self.acc_dtype(0.0)
                for i_segment in cutlass.range_constexpr(self.num_segments):
                    sCarryDecay[i_segment, local_r, local_m] = carry_decay
                    sCarryState[i_segment, local_r, local_m] = carry_state.to(Float32)

                    part_decay = sPartDecay[i_segment, local_r, local_m]
                    part_state = sPartState[i_segment, local_r, local_m]
                    carry_state = (part_decay * carry_state.to(Float32) + part_state).to(self.acc_dtype)
                    carry_decay = part_decay * carry_decay
                    
                # The final state is available without replay.
                init = self.acc_dtype(0.0)
                if const_expr(initial_state is not None):
                    init = initial_state[bidx_batch, r, m].to(self.acc_dtype)
                mFS[bidx_batch, r, m] = (carry_decay * init.to(Float32) + carry_state.to(Float32)).to(mFS.element_type)

            cute.arch.barrier()

            # Third pass replays the within-segment scan.
            if r < self.R and m < self.M:
                carry_decay = sCarryDecay[segment, local_r, local_m]
                carry_state = sCarryState[segment, local_r, local_m]

                init = self.acc_dtype(0.0)
                if const_expr(initial_state is not None):
                    init = initial_state[bidx_batch, r, m].to(self.acc_dtype)
                running = (carry_decay * init.to(Float32) + carry_state).to(self.acc_dtype)

                for i_n in cutlass.range(0, self.segment_nchunks, 1, unroll=1):
                    n = segment_start + i_n
                    if n < self.nchunks:
                        mY[bidx_batch, n, r, m] = running.to(mY.element_type)
                        decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)
                        state = mX[bidx_batch, n, r, m].to(self.acc_dtype)
                        running = state_fma(decay, running, state, self.acc_dtype)


class StatePassBwd:
    R_TILE = 16

    def __init__(self, dtype, R, M, nchunks, num_segments):
        self.dtype = dtype
        self.acc_dtype = BFloat16 if dtype is BFloat16 else Float32
        self.R = R
        self.M = M
        self.nchunks = nchunks

        self.num_segments = num_segments
        self.R_TILES = math.ceil(R / self.R_TILE)
        self.M_TILE = 8 if M == 8 else 16
        self.M_TILES = M // self.M_TILE
        self.segment_nchunks = math.ceil(nchunks / self.num_segments)
        self.elems_per_segment = self.R_TILE * self.M_TILE
        self.warps_per_segment = self.elems_per_segment // cute.arch.WARP_SIZE
        self.num_threads = self.num_segments * self.elems_per_segment

        assert M == 8 or M % 16 == 0, f"M must be 8 or a multiple of 16, got {M}"
        assert self.num_segments in (1, 4), (
            f"num_segments must be 1 or 4, got {self.num_segments}"
        )

    @classmethod
    def _state_passing_num_segments(cls, B, R, M, nchunks):
        m_tile = 8 if M == 8 else 16
        num_ctas = B * math.ceil(R / cls.R_TILE) * (M // m_tile)
        if nchunks < 64:
            return 1
        if num_ctas > 256:
            return 1
        return 4

    @cute.jit
    def __call__(
        self,
        mDY: cute.Tensor,   # (B, nchunks, R, M), states dtype
        mDFS: cute.Tensor,  # (B, R, M)
        mY: cute.Tensor,    # (B, nchunks, R, M), states dtype
        mCD: cute.Tensor,   # (B, nchunks, M), fp32
        mDX: cute.Tensor,   # (B, nchunks, R, M), states dtype
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

        segment = tidx // self.elems_per_segment
        elem = tidx % self.elems_per_segment
        local_r = elem // self.M_TILE
        local_m = elem % self.M_TILE
        r = bidx_r_tile * self.R_TILE + local_r
        m = bidx_m_tile * self.M_TILE + local_m

        smem = cutlass.memory.SmemAllocator()
        sDecayGrad = None
        if const_expr(self.acc_dtype is BFloat16):
            sDecayGrad = smem.allocate_tensor(
                BFloat16,
                cute.make_ordered_layout(
                    (2, self.num_segments, self.warps_per_segment, self.M_TILE), order=(3, 2, 1, 0),
                ),
                byte_alignment=16,
            )

        if const_expr(self.num_segments == 1):
            valid = r < self.R and m < self.M
            running = self.acc_dtype(0.0)
            if valid:
                running = mDFS[bidx_batch, r, m].to(self.acc_dtype)

            for i in cutlass.range(0, self.nchunks, 1, unroll=1):
                n = self.nchunks - 1 - i
                d_scaled = self.acc_dtype(0.0)
                h_before = self.acc_dtype(0.0)
                if valid:
                    h_before = mY[bidx_batch, n, r, m].to(self.acc_dtype)
                    d_states_acc = mDY[bidx_batch, n, r, m].to(self.acc_dtype)
                    decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)

                    mDX[bidx_batch, n, r, m] = running.to(mDX.element_type)
                    d_scaled = running * decay

                    running = state_fma(running, decay, d_states_acc, self.acc_dtype)
                self.add_decay_grad(
                    mDCD, sDecayGrad, d_scaled, h_before, bidx_batch, n, m, segment, elem, i % 2, True, valid,
                )

            if const_expr(mDIS is not None):
                if valid:
                    mDIS[bidx_batch, r, m] = running.to(mDIS.element_type)
        else:
            # Parallel path computes segment summaries, then replays segments with their carries.
            smem_layout = cute.make_ordered_layout(
                (self.num_segments, self.R_TILE, self.M_TILE), order=(2, 1, 0)
            )
            sPartDecay = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sPartState = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sCarryDecay = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )
            sCarryState = smem.allocate_tensor(
                Float32,
                smem_layout,
                byte_alignment=16,
            )

            # First pass scans within segments.
            segment_start = segment * self.segment_nchunks
            segment_end = segment_start + self.segment_nchunks
            if segment_end > self.nchunks:
                segment_end = self.nchunks

            segment_decay = 1.0
            segment_state = self.acc_dtype(0.0)
            for i_n in cutlass.range(0, self.segment_nchunks, 1, unroll=1):
                n = segment_end - 1 - i_n
                if n >= segment_start and r < self.R and m < self.M:
                    decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)
                    segment_state = state_fma(segment_state, decay, mDY[bidx_batch, n, r, m].to(self.acc_dtype), self.acc_dtype)
                    segment_decay = segment_decay * decay.to(Float32)

            if r < self.R and m < self.M:
                sPartDecay[segment, local_r, local_m] = segment_decay
                sPartState[segment, local_r, local_m] = segment_state.to(Float32)

            cute.arch.barrier()

            # Second pass scans across segments.
            if segment == 0 and r < self.R and m < self.M:
                carry_decay = 1.0
                carry_state = self.acc_dtype(0.0)
                for i_segment in cutlass.range_constexpr(self.num_segments):
                    rev_segment = self.num_segments - 1 - i_segment
                    sCarryDecay[rev_segment, local_r, local_m] = carry_decay
                    sCarryState[rev_segment, local_r, local_m] = carry_state.to(Float32)

                    part_decay = sPartDecay[rev_segment, local_r, local_m]
                    part_state = sPartState[rev_segment, local_r, local_m]
                    carry_state = (carry_state.to(Float32) * part_decay + part_state).to(self.acc_dtype)
                    carry_decay = carry_decay * part_decay

                if const_expr(mDIS is not None):
                    d_final = mDFS[bidx_batch, r, m].to(self.acc_dtype)
                    mDIS[bidx_batch, r, m] = (
                        d_final.to(Float32) * carry_decay + carry_state.to(Float32)
                    ).to(mDIS.element_type)

            cute.arch.barrier()

            # Third pass replays the within-segment scan.
            valid = r < self.R and m < self.M
            running = self.acc_dtype(0.0)
            if valid:
                d_final = mDFS[bidx_batch, r, m].to(self.acc_dtype)
                running = (
                    d_final.to(Float32) * sCarryDecay[segment, local_r, local_m]
                    + sCarryState[segment, local_r, local_m]
                ).to(self.acc_dtype)

            for i_n in cutlass.range(0, self.segment_nchunks, 1, unroll=1):
                n = segment_end - 1 - i_n
                in_segment = n >= segment_start
                d_scaled = self.acc_dtype(0.0)
                h_before = self.acc_dtype(0.0)
                if in_segment and valid:
                    h_before = mY[bidx_batch, n, r, m].to(self.acc_dtype)
                    d_states_acc = mDY[bidx_batch, n, r, m].to(self.acc_dtype)
                    decay = cute.exp(mCD[bidx_batch, n, m]).to(self.acc_dtype)

                    mDX[bidx_batch, n, r, m] = running.to(mDX.element_type)
                    d_scaled = running * decay

                    running = state_fma(running, decay, d_states_acc, self.acc_dtype)
                self.add_decay_grad(
                    mDCD, sDecayGrad, d_scaled, h_before, bidx_batch, n, m, segment, elem, i_n % 2, in_segment,
                    valid,
                )

    @cute.jit
    def add_decay_grad(
        self, mDCD, sDecayGrad, d_scaled, h_before, bidx_batch, n, m, segment, elem, slot, in_segment, valid,
    ):
        """Add d_scaled * h_before, summed over this segment's R tile, to the FP32 dChunkDecay[n, m].

        BF16 values are summed in BF16 within the tile, across lanes by butterfly shuffles
        and then across warps. FP32 values are added directly.
        """
        value = d_scaled * h_before
        if const_expr(self.acc_dtype is BFloat16):
            lane = elem % cute.arch.WARP_SIZE
            warp = elem // cute.arch.WARP_SIZE
            # Lanes that share m are M_TILE apart.
            for stage in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE // self.M_TILE))):
                other = cute.arch.shuffle_sync_bfly(
                    value.to(Float32), offset=self.M_TILE << stage,
                ).to(self.acc_dtype)
                if const_expr(stage == 0):
                    # Like Triton, each lane fuses its own product into the first add.
                    value = state_fma(d_scaled, h_before, other, self.acc_dtype)
                else:
                    value = value + other
            if lane < self.M_TILE:
                sDecayGrad[slot, segment, warp, lane] = value
            cute.arch.barrier()
            if warp == 0 and lane < self.M_TILE and in_segment:
                total = sDecayGrad[slot, segment, 0, lane]
                for w in cutlass.range_constexpr(1, self.warps_per_segment):
                    total = total + sDecayGrad[slot, segment, w, lane]
                cute.arch.atomic_add(
                    elem_pointer(mDCD, (bidx_batch, n, m)).llvm_ptr, total.to(Float32),
                    sem="relaxed", scope="gpu",
                )
        else:
            if in_segment and valid:
                cute.arch.atomic_add(
                    elem_pointer(mDCD, (bidx_batch, n, m)).llvm_ptr, value,
                    sem="relaxed", scope="gpu",
                )


@jit_cache
def _compile_state_passing_fwd(dtype, out_dtype, R, M, nchunks, initial_state_dtype, num_segments):
    B_sym = cute.sym_int()

    fake_states = make_fake_tensor(dtype, (B_sym, nchunks, R, M), divisibility=1)
    fake_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks, M), divisibility=1)
    fake_states_acc = make_fake_tensor(out_dtype, (B_sym, nchunks, R, M), divisibility=1)
    fake_final_state = make_fake_tensor(dtype, (B_sym, R, M), divisibility=1)
    fake_initial_state = (
        make_fake_tensor(initial_state_dtype, (B_sym, R, M), divisibility=1)
        if initial_state_dtype is not None
        else None
    )

    return cute.compile(
        StatePassFwd(dtype, out_dtype, R=R, M=M, nchunks=nchunks, num_segments=num_segments),
        fake_states,
        fake_chunk_decay,
        fake_states_acc,
        fake_final_state,
        fake_initial_state,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@jit_cache
def _compile_state_passing_bwd(dtype, d_final_state_dtype, R, M, nchunks, has_initial_state, num_segments):
    B_sym = cute.sym_int()

    fake_d_states_acc = make_fake_tensor(dtype, (B_sym, nchunks, R, M), divisibility=1)
    fake_d_final_state = make_fake_tensor(d_final_state_dtype, (B_sym, R, M), divisibility=1)
    fake_states_acc = make_fake_tensor(dtype, (B_sym, nchunks, R, M), divisibility=1)
    fake_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks, M), divisibility=1)
    fake_d_states = make_fake_tensor(dtype, (B_sym, nchunks, R, M), divisibility=1)
    fake_d_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks, M), divisibility=1)
    fake_d_initial_state = (
        make_fake_tensor(dtype, (B_sym, R, M), divisibility=1)
        if has_initial_state
        else None
    )

    return cute.compile(
        StatePassBwd(dtype, R=R, M=M, nchunks=nchunks, num_segments=num_segments),
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
        raise ValueError(f"cute.base state_passing requires M=8 or M multiple of 16, got M={M}")

    out_dtype = states.dtype if out_dtype is None else out_dtype
    states_acc = torch.empty(B, nchunks, R, M, device=states.device, dtype=out_dtype)
    final_state = torch.empty(B, R, M, device=states.device, dtype=states.dtype)
    if B == 0:
        return states_acc, final_state

    dtype = torch2cute_dtype_map[states.dtype]
    cute_out_dtype = torch2cute_dtype_map[out_dtype]
    initial_state_dtype = None if initial_state is None else torch2cute_dtype_map[initial_state.dtype]
    num_segments = StatePassFwd._state_passing_num_segments(B, R, M, nchunks)
    _compile_state_passing_fwd(
        dtype, cute_out_dtype, R, M, nchunks, initial_state_dtype, num_segments
    )(
        states, chunk_decay, states_acc, final_state, initial_state,
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
        raise ValueError(f"cute.base state_passing requires M=8 or M multiple of 16, got M={M}")

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
    d_final_state_dtype = torch2cute_dtype_map[d_final_state.dtype]
    num_segments = StatePassBwd._state_passing_num_segments(B, R, M, nchunks)
    _compile_state_passing_bwd(dtype, d_final_state_dtype, R, M, nchunks, has_initial_state, num_segments)(
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
# Public API (autograd-wrapped) and reference
# ============================================================================


@torch.library.custom_op("frac_ssm::state_passing_fwd_cute", mutates_args=())
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


@torch.library.custom_op("frac_ssm::state_passing_fwd_noinit_cute", mutates_args=())
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


@torch.library.custom_op("frac_ssm::state_passing_bwd_cute", mutates_args=())
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




if __name__ == "__main__":
    chunk_size = 32
    B, nchunks, R, M = 2, 1024, 64, 16
    device = "cuda"
    L = nchunks * chunk_size
    g = torch.Generator(device=device).manual_seed(0)
    states = 0.25 * torch.randn(B, nchunks, R, M, device=device, dtype=torch.float32, generator=g)
    import torch.nn.functional as F
    g_rho = torch.Generator(device=device).manual_seed(1)
    tau = torch.linspace(math.log(1.0), math.log(65536.0), M, device=device, dtype=torch.float32).exp()
    raw = torch.randn(B, L, device=device, dtype=torch.float32, generator=g_rho)
    raw = F.avg_pool1d(raw[:, None, :], 4, stride=1, padding=2)[..., :L].squeeze(1)
    dt = F.softplus(raw).clamp(1e-4, 1.0)
    alpha = 0.20 + (0.95 - 0.20) * torch.rand(B, L, device=device, dtype=torch.float32, generator=g_rho)
    lam = 0.25 + (4.0 - 0.25) * torch.rand(B, L, device=device, dtype=torch.float32, generator=g_rho)
    lambda_factor = lam.clamp_min(1e-4) ** (1.0 / alpha.clamp_min(1e-4))
    rho = torch.exp(-dt[..., None] * lambda_factor[..., None] / tau).clamp(1e-6, 1.0 - 1e-6)
    log_rho = rho.log()
    chunk_decay = log_rho.view(B, nchunks, chunk_size, M).sum(dim=2)
    initial_state = None
    for _ in range(10):
        states_acc, final_state = state_passing(states, chunk_decay, initial_state)

    torch.cuda.profiler.start()
    states_acc, final_state = state_passing(states, chunk_decay, initial_state)
    torch.cuda.profiler.stop()
