"""Warp-specialized SM90 backward for fused chunk readout.

Each kernel assigns one chunk to each CTA. The first computes D = dH @ U.T,
dE = dH @ cast(states_acc), and dStatesAcc = dH.T @ cast(C * exp(LCR)).
D and dE share an FP32 workspace of shape
(B, nchunks, chunk_size, chunk_size + M_PAD). The first chunk_size columns
hold D and the remaining columns hold dE, with only M state columns used.

In the second kernel, producer warps reconstruct W from exp(LCR[p] - LCR[q])
and reduce dC, dK, and dLCR. A consumer warp group computes dU = W.T @ dH.
W and the tensor-core operands use the input dtype. Accumulators and parameter
reductions use FP32. The intra-chunk dC term rounds to the input dtype before
the state contribution is added.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from itertools import product

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Float32
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import select_
from cutlass.tensor_utils import LayoutEnum

from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.utils import cvt_copy_r2s, make_fake_tensor, torch2cute_dtype_map


T_TILE = 64  # WGMMA output rows per instruction.
MMA_K = 16  # Reduction elements per WGMMA instruction.
M_PAD = 16  # State columns in the packed D/dE workspace and WGMMA operands.
SMEM_ALIGNMENT_BYTES = 128
PARAM_ALIGNMENT_BYTES = 16
REGISTERS_PER_SM = 65536
REGISTER_GRANULARITY = 8
MAX_REGISTERS_PER_THREAD = 255
MIN_REGISTER_ESTIMATE = 64
DDE_REGISTER_OVERHEAD = 32
DU_REGISTER_OVERHEAD = 32
PARAM_REGISTER_OVERHEAD = 16
SMEM_ESTIMATE_MARGIN_BYTES = 2048


@dataclass(frozen=True)
class ChunkFusedReadoutBwdDdeStateConfig:
    R_TILE: int = 128  # Reduction channels per D/dE tile and output rows per dStates tile.
    min_blocks_per_mp: int = 2


@dataclass(frozen=True)
class ChunkFusedReadoutBwdDuParamsConfig:
    R_TILE: int = 128  # Output channels per dU tile.
    num_producer_threads: int = 128
    min_blocks_per_mp: int = 2
    num_threads_Q: int = 32  # Lanes assigned to input positions for each output row.


class ChunkFusedReadoutBwdDdeState:
    """Compute packed D/dE and reuse the transposed dH tile for dStatesAcc."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutBwdDdeStateConfig,
        states_dtype: type[cutlass.Numeric],
    ):
        self.dtype = dtype
        self.states_dtype = states_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        self.DDE_T_TILE = max(T_TILE, chunk_size)
        self.R_TILE = config.R_TILE
        self.DDE_N_TILE = chunk_size + M_PAD
        self.num_r_tiles = math.ceil(R / self.R_TILE)

        self.num_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE
        self.tma_warp_idx = 0

    @cute.jit
    def __call__(
        self,
        mdH: cute.Tensor,         # (B, L, R)
        mU: cute.Tensor,          # (B, L, R)
        mC: cute.Tensor,          # (B, L, M)
        mStatesAcc: cute.Tensor,  # (B, nchunks, R, M)
        mLCR: cute.Tensor,        # (B, L, M), FP32
        mDDE: cute.Tensor,        # (B, nchunks, chunk_size, chunk_size + M_PAD), FP32
        mdStates: cute.Tensor,    # (B, nchunks, R, M)
        stream: cuda.CUstream,
    ):
        B, L, _ = mdH.shape
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM layouts ##################
        mma_shape = (self.DDE_T_TILE, self.DDE_N_TILE, self.R_TILE)
        dh_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_a(LayoutEnum.ROW_MAJOR, mma_shape, self.dtype, 1),
            (None, None, 0),
        )
        u_states_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_b(LayoutEnum.ROW_MAJOR, mma_shape, self.dtype, 1),
            (None, None, 0),
        )
        c_scaled_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_b(
                LayoutEnum.COL_MAJOR, (self.R_TILE, M_PAD, self.DDE_T_TILE), self.dtype, 1,
            ),
            (None, None, 0),
        )

        # TMA loads the chunk rows. Zero the remaining dH rows and append the
        # converted state tile to U to compute D and dE together.
        input_tile = (self.chunk_size, self.R_TILE)
        dh_load_smem_layout = cute.composition(
            dh_smem_layout, cute.make_layout(input_tile, stride=(1, self.DDE_T_TILE)),
        )
        u_load_smem_layout = cute.composition(
            u_states_smem_layout, cute.make_layout(input_tile, stride=(1, self.DDE_N_TILE)),
        )
        param_tile = (self.chunk_size, self.M)
        state_tile = (self.R_TILE, self.M)
        param_raw_smem_layout = cute.make_layout(param_tile, stride=(self.M, 1))
        states_raw_smem_layout = cute.make_layout(state_tile, stride=(self.M, 1))

        ################## TMA setup ##################
        tma_atom_dh, tma_tensor_dh = make_tma_load(mdH, dh_load_smem_layout, input_tile, (1, 2, 0))
        tma_atom_u, tma_tensor_u = make_tma_load(mU, u_load_smem_layout, input_tile, (1, 2, 0))
        tma_atom_c, tma_tensor_c = make_tma_load(mC, param_raw_smem_layout, param_tile, (1, 2, 0))
        tma_atom_lcr, tma_tensor_lcr = make_tma_load(
            mLCR, param_raw_smem_layout, param_tile, (1, 2, 0),
        )
        tma_atom_states, tma_tensor_states = make_tma_load(
            mStatesAcc, states_raw_smem_layout, state_tile, (2, 3, 1, 0),
        )

        ################## MMA setup ##################
        tiled_mma_dde = cute.make_tiled_mma(cute.nvgpu.warpgroup.MmaF16BF16Op(
            self.dtype, Float32, (T_TILE, self.DDE_N_TILE, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))
        tiled_mma_dstates = cute.make_tiled_mma(cute.nvgpu.warpgroup.MmaF16BF16Op(
            self.dtype, Float32, (T_TILE, M_PAD, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        ))

        ################## shared storage ##################
        @cute.struct
        class SharedStorage:
            operand_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            param_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sCRaw: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.chunk_size * self.M], SMEM_ALIGNMENT_BYTES,
            ]
            sLCRRaw: cute.struct.Align[
                cute.struct.MemRange[Float32, self.chunk_size * self.M], SMEM_ALIGNMENT_BYTES,
            ]
            sStatesRaw: cute.struct.Align[
                cute.struct.MemRange[self.states_dtype, self.R_TILE * self.M],
                SMEM_ALIGNMENT_BYTES,
            ]
            sdH: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(dh_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sUStates: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(u_states_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sCScaled: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(c_scaled_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## launch ##################
        self.kernel(
            mDDE, mdStates, L,
            tma_tensor_dh, tma_tensor_u, tma_tensor_c, tma_tensor_lcr, tma_tensor_states,
            tma_atom_dh, tma_atom_u, tma_atom_c, tma_atom_lcr, tma_atom_states,
            param_raw_smem_layout, states_raw_smem_layout, dh_smem_layout, dh_load_smem_layout,
            u_states_smem_layout, u_load_smem_layout, c_scaled_smem_layout,
            tiled_mma_dde, tiled_mma_dstates,
        ).launch(
            grid=(B * nchunks, 1, 1), block=(self.num_threads, 1, 1), stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self, mDDE: cute.Tensor, mdStates: cute.Tensor, L: cutlass.Int32,
        mdH_lrb: cute.Tensor, mU_lrb: cute.Tensor, mC_lmb: cute.Tensor,
        mLCR_lmb: cute.Tensor, mStatesAcc_rmnb: cute.Tensor,
        tma_atom_dh: cute.CopyAtom, tma_atom_u: cute.CopyAtom, tma_atom_c: cute.CopyAtom,
        tma_atom_lcr: cute.CopyAtom, tma_atom_states: cute.CopyAtom,
        param_raw_smem_layout: cute.Layout, states_raw_smem_layout: cute.Layout,
        dh_smem_layout: cute.ComposedLayout, dh_load_smem_layout: cute.ComposedLayout,
        u_states_smem_layout: cute.ComposedLayout, u_load_smem_layout: cute.ComposedLayout,
        c_scaled_smem_layout: cute.ComposedLayout,
        tiled_mma_dde: cute.TiledMma, tiled_mma_dstates: cute.TiledMma,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        nchunks = cute.ceil_div(L, self.chunk_size)
        batch, chunk = block_idx // nchunks, block_idx % nchunks
        chunk_start = chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        ################## SMEM allocations ##################
        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sdH = storage.sdH.get_tensor(dh_smem_layout.outer, swizzle=dh_smem_layout.inner)
        sdHLoad = cute.make_tensor(sdH.iterator, dh_load_smem_layout.outer)
        sdHTransposed = cute.make_tensor(
            sdH.iterator, cute.select(dh_smem_layout.outer, mode=[1, 0]),
        )
        sUStates = storage.sUStates.get_tensor(
            u_states_smem_layout.outer, swizzle=u_states_smem_layout.inner,
        )
        sULoad = cute.make_tensor(sUStates.iterator, u_load_smem_layout.outer)
        sCScaled = storage.sCScaled.get_tensor(
            c_scaled_smem_layout.outer, swizzle=c_scaled_smem_layout.inner,
        )
        sCRaw = storage.sCRaw.get_tensor(param_raw_smem_layout)
        sLCRRaw = storage.sLCRRaw.get_tensor(param_raw_smem_layout)
        sStatesRaw = storage.sStatesRaw.get_tensor(states_raw_smem_layout)

        ################## pipelines ##################
        param_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.param_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_warps),
            tx_count=self.chunk_size * self.M * (self.dtype.width // 8 + Float32.width // 8),
            defer_sync=True,
        )
        operand_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.operand_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_warps),
            tx_count=(
                2 * self.chunk_size * self.R_TILE * self.dtype.width // 8
                + self.R_TILE * self.M * self.states_dtype.width // 8
            ),
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()
        param_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        param_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        operand_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        operand_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        ################## MMA fragments ##################
        thr_mma_dde = tiled_mma_dde.get_slice(tidx)
        tCrdH = thr_mma_dde.make_fragment_A(thr_mma_dde.partition_A(sdH))
        tCrUStates = thr_mma_dde.make_fragment_B(thr_mma_dde.partition_B(sUStates))
        dde_acc_shape = (self.DDE_T_TILE, self.DDE_N_TILE)
        tCcDDE = thr_mma_dde.partition_C(cute.make_identity_tensor(dde_acc_shape))
        tCrDDE = thr_mma_dde.make_fragment_C(thr_mma_dde.partition_shape_C(dde_acc_shape))
        tCrDDE.fill(0.0)

        thr_mma_dstates = tiled_mma_dstates.get_slice(tidx)
        tCrdHTransposed = thr_mma_dstates.make_fragment_A(
            thr_mma_dstates.partition_A(sdHTransposed),
        )
        tCrCScaled = thr_mma_dstates.make_fragment_B(thr_mma_dstates.partition_B(sCScaled))
        dstates_acc_shape = (self.R_TILE, M_PAD)
        tCcdStates = thr_mma_dstates.partition_C(cute.make_identity_tensor(dstates_acc_shape))

        ################## parameter loads ##################
        self.zero_dh_padding(sdH, tidx)
        if warp_idx == self.tma_warp_idx:
            param_pipeline.producer_acquire(param_producer_state)
            barrier_ptr = param_pipeline.producer_get_barrier(param_producer_state)
            load_tile(
                tma_atom_c, mC_lmb[None, None, batch], sCRaw,
                (chunk_start, 0), (self.chunk_size, self.M),
                barrier_ptr,
            )
            load_tile(
                tma_atom_lcr, mLCR_lmb[None, None, batch], sLCRRaw,
                (chunk_start, 0), (self.chunk_size, self.M),
                barrier_ptr,
            )
            param_pipeline.producer_commit(param_producer_state)
        param_producer_state.advance()

        ################## channel loop ##################
        for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
            ################## operand loads ##################
            if warp_idx == self.tma_warp_idx:
                operand_pipeline.producer_acquire(operand_producer_state)
                barrier_ptr = operand_pipeline.producer_get_barrier(operand_producer_state)
                load_tile(
                    tma_atom_dh, mdH_lrb[None, None, batch], sdHLoad,
                    (chunk_start, r_tile_idx * self.R_TILE), (self.chunk_size, self.R_TILE),
                    barrier_ptr,
                )
                load_tile(
                    tma_atom_u, mU_lrb[None, None, batch], sULoad,
                    (chunk_start, r_tile_idx * self.R_TILE), (self.chunk_size, self.R_TILE),
                    barrier_ptr,
                )
                load_tile(
                    tma_atom_states, mStatesAcc_rmnb[None, None, chunk, batch], sStatesRaw,
                    (r_tile_idx * self.R_TILE, 0), (self.R_TILE, self.M),
                    barrier_ptr,
                )
                operand_pipeline.producer_commit(operand_producer_state)
            operand_producer_state.advance()
            if cutlass.const_expr(r_tile_idx == 0):
                param_pipeline.consumer_wait(param_consumer_state)
                self.scale_c(sCRaw, sLCRRaw, sCScaled, chunk_len, tidx)
                param_pipeline.consumer_release(param_consumer_state)
                param_consumer_state.advance()
            operand_pipeline.consumer_wait(operand_consumer_state)
            self.convert_state(sStatesRaw, sUStates, r_tile_idx, tidx)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()

            ################## D and dE WGMMA ##################
            # Accumulate across R tiles in increasing K order.
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(cute.size(tCrdH, mode=[2])):
                tiled_mma_dde.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, r_tile_idx != 0 or k_block_idx != 0,
                )
                cute.gemm(
                    tiled_mma_dde, tCrDDE, tCrdH[None, None, k_block_idx],
                    tCrUStates[None, None, k_block_idx], tCrDDE,
                )

            ################## state gradient WGMMA ##################
            # Each output channel tile reduces over the chunk's time dimension.
            tCrdStates = thr_mma_dstates.make_fragment_C(
                thr_mma_dstates.partition_shape_C(dstates_acc_shape),
            )
            for k_block_idx in cutlass.range_constexpr(cute.size(tCrdHTransposed, mode=[2])):
                tiled_mma_dstates.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(
                    tiled_mma_dstates, tCrdStates, tCrdHTransposed[None, None, k_block_idx],
                    tCrCScaled[None, None, k_block_idx], tCrdStates,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            ################## state gradient stores ##################
            self.store_state_gradient(tCrdStates, tCcdStates, mdStates, batch, chunk, r_tile_idx)
            operand_pipeline.consumer_release(operand_consumer_state)
            operand_consumer_state.advance()
            cute.arch.barrier()

        ################## workspace stores ##################
        for i in cutlass.range_constexpr(cute.size(tCrDDE)):
            p, column = tCcDDE[i]
            if p < self.chunk_size:
                mDDE[batch, chunk, p, column] = tCrDDE[i]
        cute.arch.barrier()

        ################## pipeline drain ##################
        if warp_idx == self.tma_warp_idx:
            operand_pipeline.producer_tail(operand_producer_state)
            param_pipeline.producer_tail(param_producer_state)

    @cute.jit
    def zero_dh_padding(self, sdH, tidx):
        # A 32-row chunk still needs 64 MMA rows. TMA leaves the padding untouched.
        if cutlass.const_expr(self.DDE_T_TILE > self.chunk_size):
            count = cutlass.const_expr((self.DDE_T_TILE - self.chunk_size) * self.R_TILE)
            for i in cutlass.range_constexpr(cute.ceil_div(count, self.num_threads)):
                idx = tidx + i * self.num_threads
                if idx < count:
                    sdH[self.chunk_size + idx // self.R_TILE, idx % self.R_TILE] = self.dtype(0.0)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()

    @cute.jit
    def scale_c(self, sCRaw, sLCRRaw, sCScaled, chunk_len, tidx):
        for i in cutlass.range_constexpr(cute.ceil_div(self.DDE_T_TILE * M_PAD, self.num_threads)):
            idx = tidx + i * self.num_threads
            p, m = idx // M_PAD, idx % M_PAD
            value = Float32(0.0)
            if p < chunk_len and m < self.M:
                value = sCRaw[p, m].to(Float32) * cute.math.exp2(
                    sLCRRaw[p, m] * math.log2(math.e), fastmath=True,
                )
            sCScaled[m, p] = value.to(self.dtype)

    @cute.jit
    def convert_state(self, sStatesRaw, sUStates, r_tile_idx: cutlass.Constexpr, tidx):
        for i in cutlass.range_constexpr(cute.ceil_div(self.R_TILE * M_PAD, self.num_threads)):
            idx = tidx + i * self.num_threads
            r, m = idx // M_PAD, idx % M_PAD
            value = self.dtype(0.0)
            if r_tile_idx * self.R_TILE + r < self.R and m < self.M:
                value = sStatesRaw[r, m].to(self.dtype)
            sUStates[self.chunk_size + m, r] = value

    @cute.jit
    def store_state_gradient(self, tCrdStates, tCcdStates, mdStates, batch, chunk, r_tile_idx):
        for i in cutlass.range_constexpr(cute.size(tCrdStates)):
            r, m = tCcdStates[i]
            if r_tile_idx * self.R_TILE + r < self.R and m < self.M:
                mdStates[batch, chunk, r_tile_idx * self.R_TILE + r, m] = tCrdStates[i].to(
                    mdStates.element_type,
                )


class ChunkFusedReadoutBwdDuParams:
    """Producer warps compute W/parameters while one warp group computes dU."""

    PARAM_PAIR_SIZE = 2  # A parameter value and its LCR value.
    PARAM_GRAD_PADDING = 1  # Extra FP32 element per gradient row.
    PARAM_PAIR_PADDING = 1  # Extra pair per row spreads accesses across banks.
    NUM_W_ACCUMULATORS = 4  # Independent reduction chains before rounding W.
    PRODUCER_BARRIER_ID = 1
    CONSUMER_BARRIER_ID = 2

    def __init__(
        self, dtype: type[cutlass.Numeric], R: int, M: int, chunk_size: int,
        config: ChunkFusedReadoutBwdDuParamsConfig,
    ):
        self.dtype = dtype
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        self.DU_T_TILE = max(T_TILE, chunk_size)
        self.R_TILE = config.R_TILE
        self.DDE_N_TILE = chunk_size + M_PAD
        self.num_r_tiles = math.ceil(R / self.R_TILE)

        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        self.num_producer_threads = config.num_producer_threads
        self.num_producer_warps = self.num_producer_threads // cute.arch.WARP_SIZE
        self.num_threads = self.num_consumer_threads + self.num_producer_threads
        self.producer_load_warp_idx = self.num_consumer_warps
        self.du_tma_warp_idx = 0
        assert config.num_threads_Q in (16, 32), "num_threads_Q must be 16 or 32"
        assert M <= config.num_threads_Q and chunk_size % config.num_threads_Q == 0

        self.producer_barrier = pipeline.NamedBarrier(
            self.PRODUCER_BARRIER_ID, self.num_producer_threads,
        )
        self.consumer_barrier = pipeline.NamedBarrier(
            self.CONSUMER_BARRIER_ID, self.num_consumer_threads,
        )

        self.param_grad_stride = M + self.PARAM_GRAD_PADDING
        self.param_pair_stride = self.PARAM_PAIR_SIZE * (M + self.PARAM_PAIR_PADDING)
        self.param_grad_elements = chunk_size * self.param_grad_stride

        self.num_param_iters = math.ceil(chunk_size * M / self.num_producer_threads)
        self.num_q_tiles = chunk_size // config.num_threads_Q
        self.num_threads_P = cute.arch.WARP_SIZE // config.num_threads_Q
        self.mma_k_blocks_per_t_tile = T_TILE // MMA_K

    @cute.jit
    def __call__(
        self,
        mdH: cute.Tensor,     # (B, L, R)
        mC: cute.Tensor,      # (B, L, M)
        mK: cute.Tensor,      # (B, L, M)
        mLCR: cute.Tensor,    # (B, L, M), FP32
        mDDE: cute.Tensor,    # (B, nchunks, chunk_size, chunk_size + M_PAD), FP32
        mdU: cute.Tensor,     # (B, L, R)
        mdC: cute.Tensor,     # (B, L, M)
        mdK: cute.Tensor,     # (B, L, M)
        mdLCR: cute.Tensor,   # (B, L, M), FP32
        stream: cuda.CUstream,
    ):
        B, L, _ = mdH.shape
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM layouts ##################
        mma_shape = (self.DU_T_TILE, self.R_TILE, self.chunk_size)
        w_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_a(LayoutEnum.COL_MAJOR, mma_shape, self.dtype, 1),
            (None, None, 0),
        )
        dh_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_b(LayoutEnum.COL_MAJOR, mma_shape, self.dtype, 1),
            (None, None, 0),
        )
        du_smem_layout = cute.slice_(
            sm90_utils.make_smem_layout_epi(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.DU_T_TILE, self.R_TILE), 1,
            ),
            (None, None, 0),
        )
        # Limit stores to the chunk rows so MMA padding cannot overwrite the next chunk.
        du_tile = (self.chunk_size, self.R_TILE)
        du_store_smem_layout = cute.composition(
            du_smem_layout, cute.make_layout(du_tile, stride=(1, self.DU_T_TILE)),
        )
        # The padded M stride spreads time rows across banks. Each C/K pair
        # includes LCR so the inner loop can load both values in 64 bits.
        param_tile = (self.chunk_size, self.M)
        param_grad_smem_layout = cute.make_layout(param_tile, stride=(self.param_grad_stride, 1))
        param_pair_smem_layout = cute.make_layout(
            (self.chunk_size, self.M, self.PARAM_PAIR_SIZE),
            stride=(self.param_pair_stride, self.PARAM_PAIR_SIZE, 1),
        )
        param_raw_smem_layout = cute.make_layout(param_tile, stride=(self.M, 1))
        dde_tile = (self.chunk_size, self.DDE_N_TILE)
        dde_smem_layout = cute.make_layout(dde_tile, stride=(self.DDE_N_TILE, 1))

        ################## TMA setup ##################
        tma_atom_dh, tma_tensor_dh = make_tma_load(
            mdH, dh_smem_layout, (self.R_TILE, self.chunk_size), (2, 1, 0),
        )
        mdU_lrb = cute.make_tensor(mdU.iterator, cute.select(mdU.layout, mode=[1, 2, 0]))
        tma_atom_du, tma_tensor_du = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mdU_lrb, du_store_smem_layout, du_tile,
        )
        tma_atom_c, tma_tensor_c = make_tma_load(mC, param_raw_smem_layout, param_tile, (1, 2, 0))
        tma_atom_k, tma_tensor_k = make_tma_load(mK, param_raw_smem_layout, param_tile, (1, 2, 0))
        tma_atom_lcr, tma_tensor_lcr = make_tma_load(
            mLCR, param_raw_smem_layout, param_tile, (1, 2, 0),
        )
        tma_atom_dde, tma_tensor_dde = make_tma_load(mDDE, dde_smem_layout, dde_tile, (2, 3, 1, 0))

        ################## MMA setup ##################
        tiled_mma_du = cute.make_tiled_mma(cute.nvgpu.warpgroup.MmaF16BF16Op(
            self.dtype, Float32, (T_TILE, self.R_TILE, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        ))

        ################## shared storage ##################
        @cute.struct
        class SharedStorage:
            w_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            dh_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            input_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(w_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]
            # Consumers finish the dH MMA before using the same slab for dU.
            sdHDUBuffer: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype, max(cute.cosize(dh_smem_layout), cute.cosize(du_smem_layout)),
                ],
                SMEM_ALIGNMENT_BYTES,
            ]
            sCPair: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_pair_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sKPair: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_pair_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sDDE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(dde_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]
            # Raw C and K occupy one FP32 slab and LCR occupies the other.
            # After staging the parameter pairs, the first slab holds dC.
            sdParams: cute.struct.Align[
                cute.struct.MemRange[Float32, 2 * self.param_grad_elements],
                SMEM_ALIGNMENT_BYTES,
            ]
            sdKPartials: cute.struct.Align[
                cute.struct.MemRange[Float32, self.param_grad_elements * self.num_producer_warps],
                PARAM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## launch ##################
        self.kernel(
            mdC, mdK, mdLCR, tma_atom_dh, tma_tensor_dh, tma_atom_du, tma_tensor_du,
            tma_atom_c, tma_tensor_c, tma_atom_k, tma_tensor_k,
            tma_atom_lcr, tma_tensor_lcr, tma_atom_dde, tma_tensor_dde,
            w_smem_layout, dh_smem_layout, du_smem_layout, du_store_smem_layout,
            param_grad_smem_layout, param_pair_smem_layout, param_raw_smem_layout,
            dde_smem_layout, tiled_mma_du,
        ).launch(
            grid=(B * nchunks, 1, 1), block=(self.num_threads, 1, 1), stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self, mdC: cute.Tensor, mdK: cute.Tensor, mdLCR: cute.Tensor,
        tma_atom_dh: cute.CopyAtom, mdH_rlb: cute.Tensor,
        tma_atom_du: cute.CopyAtom, mdU_lrb: cute.Tensor,
        tma_atom_c: cute.CopyAtom, mC_lmb: cute.Tensor,
        tma_atom_k: cute.CopyAtom, mK_lmb: cute.Tensor,
        tma_atom_lcr: cute.CopyAtom, mLCR_lmb: cute.Tensor,
        tma_atom_dde: cute.CopyAtom, mDDE_pjnb: cute.Tensor,
        w_smem_layout: cute.ComposedLayout, dh_smem_layout: cute.ComposedLayout,
        du_smem_layout: cute.ComposedLayout, du_store_smem_layout: cute.ComposedLayout,
        param_grad_smem_layout: cute.Layout, param_pair_smem_layout: cute.Layout,
        param_raw_smem_layout: cute.Layout, dde_smem_layout: cute.Layout,
        tiled_mma_du: cute.TiledMma,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        L = mdC.shape[1]
        nchunks = cute.ceil_div(L, self.chunk_size)
        batch, chunk = block_idx // nchunks, block_idx % nchunks
        chunk_start = chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        ################## SMEM allocations ##################
        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sW = storage.sW.get_tensor(w_smem_layout.outer, swizzle=w_smem_layout.inner)
        sdH = storage.sdHDUBuffer.get_tensor(dh_smem_layout.outer, swizzle=dh_smem_layout.inner)
        sdU = cute.make_tensor(
            cute.recast_ptr(sdH.iterator, du_smem_layout.inner, self.dtype), du_smem_layout.outer,
        )
        sdUStore = cute.make_tensor(sdU.iterator, du_store_smem_layout.outer)
        sCPair = storage.sCPair.get_tensor(param_pair_smem_layout)
        sKPair = storage.sKPair.get_tensor(param_pair_smem_layout)
        sDDE = storage.sDDE.get_tensor(dde_smem_layout)
        sdC = storage.sdParams.get_tensor(param_grad_smem_layout)
        sCRaw = cute.make_tensor(
            cute.recast_ptr(sdC.iterator, dtype=self.dtype), param_raw_smem_layout,
        )
        sKRaw = cute.make_tensor(sCRaw.iterator + self.chunk_size * self.M, param_raw_smem_layout)
        sLCRRaw = cute.make_tensor(sdC.iterator + self.chunk_size * self.M, param_raw_smem_layout)
        sdKPartials = storage.sdKPartials.get_tensor(cute.make_layout(
            (self.chunk_size, self.M, self.num_producer_warps),
            stride=(self.param_grad_stride, 1, self.param_grad_elements),
        ))

        ################## pipelines ##################
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
        )
        w_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.w_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
            defer_sync=True,
        )
        dh_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dh_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=self.R_TILE * self.chunk_size * self.dtype.width // 8, defer_sync=True,
        )
        input_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.input_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_warps,
            ),
            tx_count=(
                self.chunk_size * self.M * (2 * self.dtype.width // 8 + Float32.width // 8)
                + self.chunk_size * self.DDE_N_TILE * Float32.width // 8
            ),
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()

        if warp_idx >= self.producer_load_warp_idx:
            ################## producer prologue ##################
            producer_tidx = tidx - self.num_consumer_threads
            input_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            input_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            w_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            ################## parameter and workspace loads ##################
            w_pipeline.producer_acquire(w_producer_state)
            if warp_idx == self.producer_load_warp_idx:
                input_pipeline.producer_acquire(input_producer_state)
                barrier_ptr = input_pipeline.producer_get_barrier(input_producer_state)
                load_tile(
                    tma_atom_c, mC_lmb[None, None, batch], sCRaw,
                    (chunk_start, 0), (self.chunk_size, self.M),
                    barrier_ptr,
                )
                load_tile(
                    tma_atom_k, mK_lmb[None, None, batch], sKRaw,
                    (chunk_start, 0), (self.chunk_size, self.M),
                    barrier_ptr,
                )
                load_tile(
                    tma_atom_lcr, mLCR_lmb[None, None, batch], sLCRRaw,
                    (chunk_start, 0), (self.chunk_size, self.M),
                    barrier_ptr,
                )
                load_tile(
                    tma_atom_dde, mDDE_pjnb[None, None, chunk, batch], sDDE,
                    (0, 0), (self.chunk_size, self.DDE_N_TILE),
                    barrier_ptr,
                )
                input_pipeline.producer_commit(input_producer_state)
            input_producer_state.advance()
            input_pipeline.consumer_wait(input_consumer_state)

            ################## W and parameter gradients ##################
            self.stage_param_pairs(sCRaw, sKRaw, sLCRRaw, sCPair, sKPair, producer_tidx)
            # All raw reads must finish before dC reuses their slab.
            self.producer_barrier.arrive_and_wait()
            self.compute_w_and_param_partials(
                sW, sCPair, sKPair, sDDE, sdC, sdKPartials, chunk_len, producer_tidx,
            )
            cute.arch.fence_view_async_shared()
            w_pipeline.producer_commit(w_producer_state)
            w_producer_state.advance()
            self.producer_barrier.arrive_and_wait()

            ################## parameter gradient stores ##################
            # Parameter stores overlap the consumer's dU matrix products.
            self.store_param_gradients(
                sCPair, sKPair, sDDE, sdC, sdKPartials,
                mdC, mdK, mdLCR, batch, chunk_start, chunk_len, producer_tidx,
            )
            self.producer_barrier.arrive_and_wait()

            ################## producer drain ##################
            input_pipeline.consumer_release(input_consumer_state)
            input_consumer_state.advance()
            w_pipeline.producer_tail(w_producer_state)
            if warp_idx == self.producer_load_warp_idx:
                input_pipeline.producer_tail(input_producer_state)
        else:
            ################## consumer prologue ##################
            w_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dh_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            ################## MMA and copy fragments ##################
            thr_mma_du = tiled_mma_du.get_slice(tidx)
            tCrW = thr_mma_du.make_fragment_A(thr_mma_du.partition_A(sW))
            tCrdH = thr_mma_du.make_fragment_B(thr_mma_du.partition_B(sdH))
            r2s_atom = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(
                LayoutEnum.ROW_MAJOR.is_m_major_c(), num_matrices=4,
            ), self.dtype)
            tiled_copy_r2s = cute.make_tiled_copy_C(r2s_atom, tiled_mma_du)
            tRS_sdU = tiled_copy_r2s.get_slice(tidx).partition_D(sdU)

            for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
                ################## dH loads ##################
                if warp_idx == self.du_tma_warp_idx:
                    dh_pipeline.producer_acquire(dh_producer_state)
                    load_tile(
                        tma_atom_dh, mdH_rlb[None, None, batch], sdH,
                        (r_tile_idx * self.R_TILE, chunk_start), (self.R_TILE, self.chunk_size),
                        dh_pipeline.producer_get_barrier(dh_producer_state),
                    )
                    dh_pipeline.producer_commit(dh_producer_state)
                dh_producer_state.advance()
                if cutlass.const_expr(r_tile_idx == 0):
                    w_pipeline.consumer_wait(w_consumer_state)
                dh_pipeline.consumer_wait(dh_consumer_state)

                ################## dU WGMMA ##################
                tCrdU = thr_mma_du.make_fragment_C(
                    thr_mma_du.partition_shape_C((self.DU_T_TILE, self.R_TILE)),
                )
                self.gemm_du(tiled_mma_du, tCrW, tCrdH, tCrdU)
                dh_pipeline.consumer_release(dh_consumer_state)
                dh_consumer_state.advance()

                ################## dU stores ##################
                cvt_copy_r2s(tiled_copy_r2s, tCrdU, tRS_sdU)
                cute.arch.fence_view_async_shared()
                self.consumer_barrier.arrive_and_wait()
                if warp_idx == self.du_tma_warp_idx:
                    self.store_du(
                        tma_atom_du, mdU_lrb[None, None, batch], sdUStore,
                        store_pipeline, chunk_start, r_tile_idx,
                    )
                # The TMA store must finish before dH reuses the dU slab.
                self.consumer_barrier.arrive_and_wait()

            ################## consumer drain ##################
            w_pipeline.consumer_release(w_consumer_state)
            w_consumer_state.advance()
            if warp_idx == self.du_tma_warp_idx:
                dh_pipeline.producer_tail(dh_producer_state)

    @cute.jit
    def stage_param_pairs(self, sCRaw, sKRaw, sLCRRaw, sCPair, sKPair, producer_tidx):
        for i in cutlass.range_constexpr(self.num_param_iters):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.chunk_size * self.M:
                p, m = idx // self.M, idx % self.M
                sCPair[p, m, 0] = sCRaw[p, m].to(Float32)
                sKPair[p, m, 0] = sKRaw[p, m].to(Float32)
                sCPair[p, m, 1] = sLCRRaw[p, m]
                sKPair[p, m, 1] = sLCRRaw[p, m]

    @cute.jit
    def gemm_du(self, tiled_mma_du, tCrW, tCrdH, tCrdU):
        cute.nvgpu.warpgroup.fence()
        # W.T is upper triangular, so each row tile starts at its diagonal K block.
        for t_tile_idx in cutlass.range_constexpr(self.DU_T_TILE // T_TILE):
            first_k_block = cutlass.const_expr(t_tile_idx * self.mma_k_blocks_per_t_tile)
            tCrdURows = cute.local_tile(
                tCrdU, (tCrdU.shape[0], 1, tCrdU.shape[2]), (0, t_tile_idx, 0),
            )
            for k_block_idx in cutlass.range_constexpr(cute.size(tCrW, mode=[2])):
                if cutlass.const_expr(k_block_idx >= first_k_block):
                    tCrWRows = cute.local_tile(
                        tCrW[None, None, k_block_idx], (tCrW.shape[0], 1), (0, t_tile_idx),
                    )
                    tiled_mma_du.set(
                        cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != first_k_block,
                    )
                    cute.gemm(
                        tiled_mma_du, tCrdURows, tCrWRows, tCrdH[None, None, k_block_idx],
                        tCrdURows,
                    )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

    @cute.jit
    def store_du(self, tma_atom_du, mdU, sdU, store_pipeline, chunk_start, r_tile_idx):
        gdU = cute.local_tile(
            cute.domain_offset((chunk_start, r_tile_idx * self.R_TILE), mdU),
            (self.chunk_size, self.R_TILE), (0, 0),
        )
        bSG_sdU, bSG_gdU = cpasync.tma_partition(
            tma_atom_du, cta_coord=0, cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sdU, 0, 2),
            gmem_tensor=cute.group_modes(gdU, 0, 2),
        )
        store_pipeline.producer_acquire()
        cute.copy(tma_atom_du, bSG_sdU, bSG_gdU)
        store_pipeline.producer_commit()
        store_pipeline.producer_tail()

    @cute.jit
    def compute_w_and_param_partials(
        self, sW, sCPair, sKPair, sDDE, sdC, sdKPartials, chunk_len, producer_tidx,
    ):
        # Each lane owns q positions and retains their dK partials across p rows.
        lane_idx = cute.arch.lane_idx()
        producer_warp_idx = producer_tidx // cute.arch.WARP_SIZE
        q_lane = lane_idx % self.config.num_threads_Q
        p_lane = lane_idx // self.config.num_threads_Q
        rdK = cute.make_rmem_tensor((self.num_q_tiles, self.M), Float32)
        rdK.fill(0.0)

        # Only causal entries are overwritten below. Clear the rest for the dU MMA.
        for i in cutlass.range_constexpr(
            cute.ceil_div(self.DU_T_TILE * self.chunk_size, self.num_producer_threads),
        ):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.DU_T_TILE * self.chunk_size:
                sW[idx % self.DU_T_TILE, idx // self.DU_T_TILE] = self.dtype(0.0)
        self.producer_barrier.arrive_and_wait()

        for p_tile_idx in cutlass.range(
            cute.ceil_div(self.chunk_size, self.num_producer_warps * self.num_threads_P), unroll=1,
        ):
            p = (
                (p_tile_idx * self.num_producer_warps + producer_warp_idx) * self.num_threads_P
                + p_lane
            )
            rdC = cute.make_rmem_tensor((self.M,), Float32)
            rdC.fill(0.0)
            for q_tile_idx in cutlass.range_constexpr(self.num_q_tiles):
                q = q_lane + q_tile_idx * self.config.num_threads_Q
                rW = cute.make_rmem_tensor((self.NUM_W_ACCUMULATORS,), Float32)
                rW.fill(0.0)
                if p < chunk_len and q <= p:
                    d_pq = sDDE[p, q]
                    for m in cutlass.range_constexpr(self.M):
                        c_p, lcr_p = self.load_pair(sCPair, p, m)
                        k_q, lcr_q = self.load_pair(sKPair, q, m)
                        decay = cute.math.exp2((lcr_p - lcr_q) * math.log2(math.e), fastmath=True)
                        # Accumulate into independent sums to shorten the dependency chains.
                        rW[m % self.NUM_W_ACCUMULATORS] += c_p * k_q * decay
                        d_decay = d_pq * decay
                        rdC[m] += d_decay * k_q
                        rdK[q_tile_idx, m] += d_decay * c_p
                    # Add the chains in order before rounding W to the input dtype.
                    w_value = rW[0]
                    for i in cutlass.range_constexpr(1, self.NUM_W_ACCUMULATORS):
                        w_value += rW[i]
                    sW[q, p] = w_value.to(self.dtype)
            self.reduce_dc(rdC, sdC, p, q_lane, chunk_len)

        for q_tile_idx in cutlass.range_constexpr(self.num_q_tiles):
            for m in cutlass.range_constexpr(self.M):
                value = rdK[q_tile_idx, m]
                for stage in cutlass.range_constexpr(int(math.log2(self.num_threads_P))):
                    value += cute.arch.shuffle_sync_bfly(
                        value, offset=(cute.arch.WARP_SIZE // 2) >> stage,
                    )
                if p_lane == 0:
                    q = q_lane + q_tile_idx * self.config.num_threads_Q
                    sdKPartials[q, m, producer_warp_idx] = value

    @cute.jit
    def reduce_dc(self, rdC, sdC, p, q_lane, chunk_len):
        # Reduce across this row's q lanes and distribute the M results among them.
        for stage in cutlass.range_constexpr(int(math.log2(self.M))):
            half = cutlass.const_expr(self.M >> (stage + 1))
            offset = cutlass.const_expr((self.config.num_threads_Q // 2) >> stage)
            upper = (q_lane & offset) != 0
            for m in cutlass.range_constexpr(half):
                local = Float32(select_(upper, rdC[m + half], rdC[m]))
                send = Float32(select_(upper, rdC[m], rdC[m + half]))
                rdC[m] = local + cute.arch.shuffle_sync_bfly(send, offset=offset)
        lanes_per_m = cutlass.const_expr(self.config.num_threads_Q // self.M)
        for stage in cutlass.range_constexpr(int(math.log2(lanes_per_m))):
            offset = cutlass.const_expr(
                (self.config.num_threads_Q // 2) >> (int(math.log2(self.M)) + stage),
            )
            rdC[0] += cute.arch.shuffle_sync_bfly(rdC[0], offset=offset)
        if q_lane % lanes_per_m == 0 and p < chunk_len:
            sdC[p, q_lane // lanes_per_m] = rdC[0]

    @cute.jit
    def load_pair(self, sParamPair, p, m):
        # Load the FP32 parameter value and its LCR together in 64 bits.
        sParams = cute.make_tensor(
            sParamPair.iterator + cute.crd2idx((p, m, 0), sParamPair.layout),
            cute.make_layout(self.PARAM_PAIR_SIZE),
        )
        rParams = cute.make_rmem_tensor((self.PARAM_PAIR_SIZE,), Float32)
        s2r_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32,
            num_bits_per_copy=self.PARAM_PAIR_SIZE * Float32.width,
        )
        cute.copy(s2r_atom, sParams, rParams)
        return rParams[0], rParams[1]

    @cute.jit
    def store_param_gradients(
        self, sCPair, sKPair, sDDE, sdC, sdKPartials,
        mdC, mdK, mdLCR, batch, chunk_start, chunk_len, producer_tidx,
    ):
        # Avoid full unrolling so epilogue addresses do not stay live across W computation.
        for i in cutlass.range(self.num_param_iters, unroll=1):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.chunk_size * self.M:
                p, m = idx // self.M, idx % self.M
                if p < chunk_len:
                    c_p, lcr_p = self.load_pair(sCPair, p, m)
                    k_p, _ = self.load_pair(sKPair, p, m)
                    dc_intra = sdC[p, m]
                    dk = Float32(0.0)
                    # Sum warp partials in increasing warp order.
                    for warp_idx in cutlass.range_constexpr(self.num_producer_warps):
                        dk += sdKPartials[p, m, warp_idx]
                    decay = cute.math.exp2(lcr_p * math.log2(math.e), fastmath=True)
                    de = sDDE[p, self.chunk_size + m]
                    dlcr_intra = c_p * dc_intra - k_p * dk
                    # Round intra-chunk dC before adding the state term.
                    mdC[batch, chunk_start + p, m] = (
                        dc_intra.to(self.dtype).to(Float32) + de * decay
                    ).to(self.dtype)
                    mdK[batch, chunk_start + p, m] = dk.to(self.dtype)
                    mdLCR[batch, chunk_start + p, m] = dlcr_intra + de * c_p * decay


@cute.jit
def make_tma_load(mX: cute.Tensor, smem_layout, tile_shape, modes: cutlass.Constexpr):
    """Reorder global axes to match the shared tile without moving data."""
    mView = cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=modes))
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mView, smem_layout, tile_shape,
    )


@cute.jit
def load_tile(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, sX: cute.Tensor,
    offset, tile_shape, barrier_ptr,
):
    gX = cute.local_tile(cute.domain_offset(offset, tma_tensor), tile_shape, (0, 0))
    bGS_sX, bGS_gX = cpasync.tma_partition(
        tma_atom, cta_coord=0, cta_layout=cute.make_layout(1),
        smem_tensor=cute.group_modes(sX, 0, 2),
        gmem_tensor=cute.group_modes(gX, 0, 2),
    )
    cute.copy(tma_atom, bGS_gX, bGS_sX, tma_bar_ptr=barrier_ptr)


@jit_cache
def _compile_chunk_fused_readout_bwd_dde_state(dtype, states_dtype, R, M, chunk_size, config):
    B, L, nchunks = cute.sym_int(), cute.sym_int(), cute.sym_int()
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_u = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_states = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    fake_dde = make_fake_tensor(
        Float32, (B, nchunks, chunk_size, chunk_size + M_PAD), divisibility=1,
    )
    fake_dstates = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwdDdeState(dtype, R, M, chunk_size, config, states_dtype),
        fake_dh, fake_u, fake_c, fake_states, fake_lcr, fake_dde, fake_dstates,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@jit_cache
def _compile_chunk_fused_readout_bwd_du_params(dtype, R, M, chunk_size, config):
    B, L, nchunks = cute.sym_int(), cute.sym_int(), cute.sym_int()
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_k = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    fake_dde = make_fake_tensor(
        Float32, (B, nchunks, chunk_size, chunk_size + M_PAD), divisibility=1,
    )
    fake_du = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_dc = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_dk = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_dlcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwdDuParams(dtype, R, M, chunk_size, config),
        fake_dh, fake_c, fake_k, fake_lcr, fake_dde, fake_du, fake_dc, fake_dk, fake_dlcr,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


def dde_state_configs():
    return [
        AutotuneConfig(config=ChunkFusedReadoutBwdDdeStateConfig(
            R_TILE=R_TILE, min_blocks_per_mp=min_blocks_per_mp,
        ))
        for R_TILE, min_blocks_per_mp in product((64, 128), (1, 2, 3, 4, 6))
    ]


def du_params_configs():
    return [
        AutotuneConfig(config=ChunkFusedReadoutBwdDuParamsConfig(
            R_TILE=R_TILE,
            num_producer_threads=num_producer_threads,
            min_blocks_per_mp=min_blocks_per_mp,
            num_threads_Q=num_threads_Q,
        ))
        for R_TILE, num_producer_threads, min_blocks_per_mp, num_threads_Q in product(
            (64, 128), (64, 128, 256), (1, 2, 3), (16, 32),
        )
    ]


def _prune_dde_state(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    dh, c, states_acc, chunk_size = (args[k] for k in ("dh", "c", "states_acc", "chunk_size"))
    smem_limit = torch.cuda.get_device_properties(dh.device).shared_memory_per_block_optin
    M = c.shape[-1]
    DDE_T_TILE = max(T_TILE, chunk_size)
    DDE_N_TILE = chunk_size + M_PAD
    num_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
    input_bytes = dh.element_size()
    states_bytes = states_acc.element_size()
    fp32_bytes = Float32.width // 8

    valid_configs = []
    for candidate in configs:
        config = candidate.kwargs["config"]
        registers_per_thread = REGISTERS_PER_SM // (num_threads * config.min_blocks_per_mp)
        registers_per_thread = min(
            MAX_REGISTERS_PER_THREAD,
            registers_per_thread // REGISTER_GRANULARITY * REGISTER_GRANULARITY,
        )
        dde_acc_registers = DDE_T_TILE * DDE_N_TILE // num_threads
        dstates_acc_registers = config.R_TILE * M_PAD // num_threads
        required_registers = dde_acc_registers + dstates_acc_registers + DDE_REGISTER_OVERHEAD
        if registers_per_thread < required_registers:
            continue

        # Account for the operands, raw inputs, barriers and alignment margin.
        operand_smem_bytes = input_bytes * (
            DDE_T_TILE * config.R_TILE + DDE_N_TILE * config.R_TILE + DDE_T_TILE * M_PAD
        )
        states_smem_bytes = config.R_TILE * M * states_bytes
        param_smem_bytes = chunk_size * M * (input_bytes + fp32_bytes)
        smem_bytes = (
            operand_smem_bytes + states_smem_bytes + param_smem_bytes
            + SMEM_ESTIMATE_MARGIN_BYTES
        )
        if smem_bytes <= smem_limit:
            valid_configs.append(candidate)
    return valid_configs


def _prune_du_params(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    dh, c, chunk_size = (args[k] for k in ("dh", "c", "chunk_size"))
    smem_limit = torch.cuda.get_device_properties(dh.device).shared_memory_per_block_optin
    M = c.shape[-1]
    DU_T_TILE = max(T_TILE, chunk_size)
    DDE_N_TILE = chunk_size + M_PAD
    num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
    input_bytes = dh.element_size()
    fp32_bytes = Float32.width // 8
    param_grad_stride = M + ChunkFusedReadoutBwdDuParams.PARAM_GRAD_PADDING
    param_pair_stride = ChunkFusedReadoutBwdDuParams.PARAM_PAIR_SIZE * (
        M + ChunkFusedReadoutBwdDuParams.PARAM_PAIR_PADDING
    )

    valid_configs = []
    for candidate in configs:
        config = candidate.kwargs["config"]
        num_threads = config.num_producer_threads + num_consumer_threads
        num_producer_warps = config.num_producer_threads // cute.arch.WARP_SIZE
        registers_per_thread = REGISTERS_PER_SM // (num_threads * config.min_blocks_per_mp)
        registers_per_thread = min(
            MAX_REGISTERS_PER_THREAD,
            registers_per_thread // REGISTER_GRANULARITY * REGISTER_GRANULARITY,
        )
        du_acc_registers = DU_T_TILE * config.R_TILE // num_consumer_threads
        param_registers = chunk_size // config.num_threads_Q * M + M
        required_registers = max(
            MIN_REGISTER_ESTIMATE,
            du_acc_registers + DU_REGISTER_OVERHEAD,
            param_registers + PARAM_REGISTER_OVERHEAD,
        )
        if registers_per_thread < required_registers:
            continue

        w_smem_bytes = input_bytes * DU_T_TILE * chunk_size
        dh_du_smem_bytes = input_bytes * DU_T_TILE * config.R_TILE
        dk_partials_smem_bytes = fp32_bytes * chunk_size * param_grad_stride * num_producer_warps
        param_pairs_smem_bytes = 2 * fp32_bytes * chunk_size * param_pair_stride
        raw_params_smem_bytes = 2 * fp32_bytes * chunk_size * param_grad_stride
        dde_smem_bytes = fp32_bytes * chunk_size * DDE_N_TILE
        smem_bytes = (
            w_smem_bytes + dh_du_smem_bytes + dk_partials_smem_bytes
            + param_pairs_smem_bytes + raw_params_smem_bytes + dde_smem_bytes
            + SMEM_ESTIMATE_MARGIN_BYTES
        )
        if smem_bytes <= smem_limit:
            valid_configs.append(candidate)
    return valid_configs


def _prepare_dde_state(dh, u, c, states_acc, lcr, dde, dstates, *, chunk_size, config):
    return _compile_chunk_fused_readout_bwd_dde_state(
        torch2cute_dtype_map[dh.dtype], torch2cute_dtype_map[states_acc.dtype],
        dh.shape[-1], c.shape[-1], chunk_size, config,
    )


def _prepare_du_params(dh, c, k, lcr, dde, du, dc, dk, dlcr, *, chunk_size, config):
    return _compile_chunk_fused_readout_bwd_du_params(
        torch2cute_dtype_map[dh.dtype], dh.shape[-1], c.shape[-1], chunk_size, config,
    )


@autotune(
    configs=dde_state_configs(), key=["chunk_size", ("dh", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": _prune_dde_state}, cache_results=False,
)
def _tune_dde_state(dh, u, c, states_acc, lcr, dde, dstates, *, chunk_size, config):
    kernel = _prepare_dde_state(
        dh, u, c, states_acc, lcr, dde, dstates, chunk_size=chunk_size, config=config,
    )
    kernel(dh, u, c, states_acc, lcr, dde, dstates)


@autotune(
    configs=du_params_configs(), key=["chunk_size", ("dh", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": _prune_du_params}, cache_results=False,
)
def _tune_du_params(dh, c, k, lcr, dde, du, dc, dk, dlcr, *, chunk_size, config):
    kernel = _prepare_du_params(
        dh, c, k, lcr, dde, du, dc, dk, dlcr, chunk_size=chunk_size, config=config,
    )
    kernel(dh, c, k, lcr, dde, du, dc, dk, dlcr)


_backward_launch_cache = {}
_backward_launch_lock = threading.Lock()
_prepare_stages = (_prepare_dde_state, _prepare_du_params)


def _get_autotuned_kernels(key, args, chunk_size):
    kernels = _backward_launch_cache.get(key)
    if kernels is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up warp-specialized backward autotuning before CUDA graph capture",
            )
        with _backward_launch_lock:
            kernels = _backward_launch_cache.get(key)
            if kernels is None:
                dde_state_args, du_params_args = args
                _tune_dde_state(*dde_state_args, chunk_size=chunk_size)
                dde_state_config = _tune_dde_state.best_config.kwargs["config"]
                dde_state_kernel = _prepare_dde_state(
                    *dde_state_args, chunk_size=chunk_size, config=dde_state_config,
                )
                # dU/parameter tuning must consume this invocation's complete D/dE.
                dde_state_kernel(*dde_state_args)
                _tune_du_params(*du_params_args, chunk_size=chunk_size)
                du_params_config = _tune_du_params.best_config.kwargs["config"]
                du_params_kernel = _prepare_du_params(
                    *du_params_args, chunk_size=chunk_size, config=du_params_config,
                )
                kernels = (dde_state_kernel, du_params_kernel)
                _backward_launch_cache[key] = kernels
    return kernels


def _frac_chunk_fused_readout_bwd(
    d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size,
    config: tuple[
        ChunkFusedReadoutBwdDdeStateConfig, ChunkFusedReadoutBwdDuParamsConfig,
    ] | None = None,
):
    """Run the two backward stages with optional explicit configs for both."""
    B, L, R = u.shape
    M = c.shape[-1]
    du, dc, dk, dstates, dlcr = [
        torch.empty_like(x) for x in (u, c, kappa, states_acc, log_cum_rho)
    ]
    if B == 0 or L == 0:
        return du, dc, dk, dstates, dlcr
    assert u.dtype in (torch.float16, torch.bfloat16), f"u must be 16-bit, got {u.dtype}"
    assert d_h.dtype == c.dtype == kappa.dtype == u.dtype, "d_h, c, and kappa dtype must match u"
    assert log_cum_rho.dtype == torch.float32, "log_cum_rho must be float32"
    assert chunk_size in (32, 64, 128), f"unsupported chunk_size {chunk_size}"
    assert M in (8, 16), f"M must be 8 or 16, got {M}"

    dde = torch.empty(
        (B, math.ceil(L / chunk_size), chunk_size, chunk_size + M_PAD),
        device=u.device, dtype=torch.float32,
    )
    args = (
        (d_h, u, c, states_acc, log_cum_rho, dde, dstates),
        (d_h, c, kappa, log_cum_rho, dde, du, dc, dk, dlcr),
    )
    if config is None:
        key = (u.device.index, u.dtype, states_acc.dtype, R, M, chunk_size)
        kernels = _get_autotuned_kernels(key, args, chunk_size)
    else:
        assert len(config) == 2, "config must contain D/dE/state and dU/parameter stage configs"
        kernels = tuple(
            prepare(*inputs, chunk_size=chunk_size, config=stage_config)
            for prepare, inputs, stage_config in zip(_prepare_stages, args, config)
        )
    for kernel, inputs in zip(kernels, args):
        kernel(*inputs)
    return du, dc, dk, dstates, dlcr


__all__ = ["_frac_chunk_fused_readout_bwd"]
