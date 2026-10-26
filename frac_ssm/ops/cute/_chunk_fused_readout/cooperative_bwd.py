"""Cooperative SM90 readout backward with three independently tuned stages.

    dU = W.T @ dH
    D = dH @ U.T
    dStates = dH.T @ cast(C * exp(LCR))
    dE = dH @ cast(states_acc)

dU cooperatively constructs causal W tiles while WGMMA consumes the preceding
tile. The parameter stage uses a bounded CTA loop. One warp loads D operands,
one warp group computes D, then all warps reduce the parameter gradients.
The state stage writes dStates tiles. Its first R-tile CTA also adds the state
contributions to dC and dLCR using dE.

W and parameter gradients evaluate exp(LCR[p] - LCR[q]) directly. W uses four
independent M sums. Parameter reductions use FP32, and intra-chunk dC rounds
to the input dtype before its state contribution is added. dK uses shared
FP32 atomics.
"""

import math
import operator
import threading
from dataclasses import dataclass, replace
from itertools import product

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Boolean, Float16, Float32
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import and_, dsl_user_op
from cutlass.tensor_utils import LayoutEnum

from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.utils import (
    cvt_copy_r2s, make_fake_tensor, predicate_axis, torch2cute_dtype_map,
)


MMA_N_GRANULARITY = 16  # Output columns per WGMMA N increment.
MMA_K = 16  # Reduction elements per WGMMA instruction.
SMEM_ALIGNMENT_BYTES = 128
PARAM_ALIGNMENT_BYTES = 16
VECTOR_COPY_BITS = 128
REGISTERS_PER_SM = 65536
REGISTER_GRANULARITY = 8
DU_REGISTER_OVERHEAD = 40
MAX_R_TILES = 8  # Bound the unrolled R reduction in the parameter and state stages.


@dataclass(frozen=True)
class ChunkFusedReadoutBwdDuConfig:
    R_TILE: int = 128  # Output channels per dU tile.
    num_warp_groups: int = 2
    min_blocks_per_mp: int = 3


@dataclass(frozen=True)
class ChunkFusedReadoutBwdParamsConfig:
    R_TILE: int = 128  # Reduction channels per D tile.
    num_producer_threads: int = 128
    ctas_per_sm: int = 2
    max_ctas: int | None = None  # Optional absolute cap, resolved before compilation.


@dataclass(frozen=True)
class ChunkFusedReadoutBwdStateConfig:
    R_TILE: int = 128  # Output rows per dStates tile and reduction channels per dE tile.
    num_producer_threads: int = 128
    min_blocks_per_mp: int = 1


class ChunkFusedReadoutBwdDu:
    """All CTA threads build W tiles while WGMMA computes dU = W.T @ dH."""

    T_TILE = 64  # Output time rows per WGMMA tile.
    K_TILE = 32  # Time positions reduced per W/dH pipeline stage.
    NUM_W_ACCUMULATORS = 4  # Independent M reduction chains before rounding W.

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutBwdDuConfig,
    ):
        self.dtype = dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        self.R_TILE = config.R_TILE
        self.num_t_tiles = math.ceil(chunk_size / self.T_TILE)
        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.num_warp_groups = config.num_warp_groups
        self.mma_n = self.R_TILE // self.num_warp_groups
        self.num_threads = cute.arch.THREADS_PER_WARPGROUP * self.num_warp_groups  # 128
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE
        self.tma_warp_idx = 0
        self.num_stages = 2

        self.num_m_blocks = math.ceil(self.M / self.NUM_W_ACCUMULATORS)
        self.w_rows_per_iter = self.num_threads // self.K_TILE
        self.num_w_row_iters = math.ceil(self.T_TILE / self.w_rows_per_iter)

        assert dtype.width == 16
        assert M in (8, 16)
        assert chunk_size in (32, 64, 128)
        assert self.R_TILE in (64, 128, 192)
        assert self.mma_n % MMA_N_GRANULARITY == 0
        assert self.num_threads % self.K_TILE == 0
        assert self.M % self.NUM_W_ACCUMULATORS == 0

    @cute.jit
    def __call__(
        self,
        mdH: cute.Tensor,         # (B, L, R)
        mC: cute.Tensor,          # (B, L, M)
        mK: cute.Tensor,          # (B, L, M)
        mLCR: cute.Tensor,        # (B, L, M), FP32
        mdU: cute.Tensor,         # (B, L, R)
        stream: cuda.CUstream,
    ):
        B, L = mdH.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM layouts ##################
        mma_tiler = (self.T_TILE, self.mma_n, self.K_TILE)
        w_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, mma_tiler, self.dtype, num_stages=self.num_stages
        )
        dh_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR,
            (self.T_TILE, self.R_TILE, self.K_TILE),
            self.dtype,
            num_stages=self.num_stages,
        )
        du_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.T_TILE, self.R_TILE), 1
        )
        du_smem_layout = cute.slice_(du_smem_layout_staged, (None, None, 0))
        param_smem_layout = cute.make_ordered_layout((self.chunk_size, self.M), order=(1, 0))

        ################## MMA setup ##################
        mma_op_du = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.mma_n, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_mma_du = cute.make_tiled_mma(mma_op_du, (1, self.num_warp_groups, 1))

        ################## TMA setup ##################
        mdH_rlb = cute.make_tensor(
            mdH.iterator, cute.select(mdH.layout, mode=(2, 1, 0)),
        )
        mdU_lrb = cute.make_tensor(
            mdU.iterator, cute.select(mdU.layout, mode=(1, 2, 0)),
        )
        tma_atom_dh, tma_tensor_dh = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdH_rlb,
            cute.slice_(dh_smem_layout_staged, (None, None, 0)),
            (self.R_TILE, self.K_TILE),
        )
        tma_atom_du, tma_tensor_du = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdU_lrb, du_smem_layout,
            (self.T_TILE, self.R_TILE),
        )

        ################## parameter copy setup ##################
        param_vecsize = cutlass.const_expr(min(self.M, VECTOR_COPY_BITS // self.dtype.width))
        lcr_vecsize = cutlass.const_expr(min(self.M, VECTOR_COPY_BITS // Float32.width))
        tiled_copy_g2s_params = make_vec_g2s_copy(
            mC.element_type, self.chunk_size, self.M, self.num_threads, param_vecsize,
        )
        tiled_copy_g2s_lcr = make_vec_g2s_copy(
            mLCR.element_type, self.chunk_size, self.M, self.num_threads, lcr_vecsize,
        )

        ################## shared storage ##################
        @cute.struct
        class SharedStorage:
            dh_barriers: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(w_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            # Two dH stages also hold the 64-row dU epilogue tile.
            sdH: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(dh_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## launch ##################
        self.kernel(
            mC, mK, mLCR, mdU, tma_tensor_dh, tma_tensor_du, tiled_mma_du,
            tiled_copy_g2s_params, tiled_copy_g2s_lcr,
            w_smem_layout_staged, dh_smem_layout_staged, du_smem_layout_staged, param_smem_layout,
            tma_atom_dh, tma_atom_du,
        ).launch(
            grid=[self.num_t_tiles * self.num_r_tiles, nchunks, B],
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self, mC, mK, mLCR, mdU, mdH_rlb, mdU_lrb, tiled_mma_du,
        tiled_copy_g2s_params, tiled_copy_g2s_lcr,
        w_smem_layout_staged, dh_smem_layout_staged, du_smem_layout_staged, param_smem_layout,
        tma_atom_dh, tma_atom_du,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, chunk, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        log2_e = math.log2(math.e)
        L = mC.shape[1]

        t_tile_idx = block_idx // self.num_r_tiles
        r_tile_idx = block_idx % self.num_r_tiles
        chunk_start = chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
        q_start = t_tile_idx * self.T_TILE

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        dh_barrier_ptr = storage.dh_barriers.data_ptr()
        sW = storage.sW.get_tensor(w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner)
        sdH = storage.sdH.get_tensor(
            dh_smem_layout_staged.outer, swizzle=dh_smem_layout_staged.inner,
        )
        # dU reuses the dH buffer after the final WGMMA finishes.
        sdU = cute.make_tensor(
            cute.recast_ptr(sdH.iterator, du_smem_layout_staged.inner, self.dtype),
            du_smem_layout_staged.outer,
        )
        sdUStore = cute.slice_(sdU, (None, None, 0))
        sC = storage.sC.get_tensor(param_smem_layout)
        sK = storage.sK.get_tensor(param_smem_layout)
        sLCR = storage.sLCR.get_tensor(param_smem_layout)

        if warp_idx == self.tma_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_du)

        ################## pipelines ##################
        dh_tma_bytes = cutlass.const_expr(
            cute.size_in_bytes(self.dtype, cute.slice_(dh_smem_layout_staged, (None, None, 0)))
        )
        dh_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=dh_barrier_ptr,
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_warps),
            tx_count=dh_tma_bytes,
            defer_sync=True,
        )
        du_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads),
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()

        ################## parameter loads ##################
        load_param(tiled_copy_g2s_params, mC, sC, batch, chunk_start, chunk_len, tidx)
        load_param(tiled_copy_g2s_params, mK, sK, batch, chunk_start, chunk_len, tidx)
        load_param(tiled_copy_g2s_lcr, mLCR, sLCR, batch, chunk_start, chunk_len, tidx)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        ################## MMA fragments ##################
        thr_mma_du = tiled_mma_du.get_slice(tidx)
        tCrdU = thr_mma_du.make_fragment_C(thr_mma_du.partition_shape_C((self.T_TILE, self.R_TILE)))
        tCrW = thr_mma_du.make_fragment_A(thr_mma_du.partition_A(sW))
        tCrdH = thr_mma_du.make_fragment_B(thr_mma_du.partition_B(sdH))
        cdU = cute.make_identity_tensor((self.T_TILE, self.R_TILE))
        tCcdU = thr_mma_du.partition_C(cdU)

        gdH = cute.local_tile(
            mdH_rlb[None, None, batch], (self.R_TILE, self.K_TILE), (r_tile_idx, None)
        )
        bGS_sdH, bGS_gdH = cpasync.tma_partition(
            tma_atom_dh, cta_coord=0, cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sdH, 0, 2), gmem_tensor=cute.group_modes(gdH, 0, 2),
        )

        # W.T is upper triangular, so reduction begins at this output tile's diagonal.
        first_k_stage = (q_start + self.K_TILE - 1) // self.K_TILE
        num_k_stages = cute.ceil_div(chunk_len, self.K_TILE)

        dh_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages,
        )
        dh_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages,
        )

        ################## first W and dH tile ##################
        if first_k_stage < num_k_stages:
            if warp_idx == self.tma_warp_idx:
                dh_pipeline.producer_acquire(dh_producer_state)
                global_k_stage = (chunk_start + first_k_stage * self.K_TILE) // self.K_TILE
                cute.copy(
                    tma_atom_dh,
                    bGS_gdH[None, global_k_stage],
                    bGS_sdH[None, dh_producer_state.index],
                    tma_bar_ptr=dh_pipeline.producer_get_barrier(dh_producer_state),
                )
                dh_pipeline.producer_commit(dh_producer_state)
            dh_producer_state.advance()
            self.compute_w_tile(
                sW, sC, sK, sLCR, first_k_stage % self.num_stages,
                first_k_stage, q_start, chunk_len, log2_e, tidx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

        ################## dU WGMMA loop ##################
        for k_stage_idx in cutlass.range(first_k_stage, num_k_stages, 1, unroll=1):
            w_slot = k_stage_idx % self.num_stages
            dh_pipeline.consumer_wait(dh_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.nvgpu.warpgroup.fence()
            gemm(
                tiled_mma_du,
                tCrW[None, None, None, w_slot],
                tCrdH[None, None, None, dh_consumer_state.index],
                tCrdU, k_stage_idx != first_k_stage,
            )
            cute.nvgpu.warpgroup.commit_group()

            if k_stage_idx + 1 < num_k_stages:
                next_w_slot = (k_stage_idx + 1) % self.num_stages
                if warp_idx == self.tma_warp_idx:
                    dh_pipeline.producer_acquire(dh_producer_state)
                    global_k_stage = (chunk_start + (k_stage_idx + 1) * self.K_TILE) // self.K_TILE
                    cute.copy(
                        tma_atom_dh,
                        bGS_gdH[None, global_k_stage],
                        bGS_sdH[None, dh_producer_state.index],
                        tma_bar_ptr=dh_pipeline.producer_get_barrier(dh_producer_state),
                    )
                    dh_pipeline.producer_commit(dh_producer_state)
                dh_producer_state.advance()
                self.compute_w_tile(
                    sW, sC, sK, sLCR, next_w_slot,
                    k_stage_idx + 1, q_start, chunk_len, log2_e, tidx,
                )

            cute.nvgpu.warpgroup.wait_group(0)
            cute.arch.fence_view_async_shared()
            dh_pipeline.consumer_release(dh_consumer_state)
            dh_consumer_state.advance()
            cute.arch.sync_threads()

        ################## dU stores ##################
        self.store_du_epilogue(
            tiled_mma_du, tma_atom_du, mdU_lrb, sdU, sdUStore, tCrdU, tCcdU,
            t_tile_idx, r_tile_idx, chunk_start, chunk_len, batch,
            du_store_pipeline, tidx, warp_idx, mdU, first_k_stage, num_k_stages,
        )

    @cute.jit
    def compute_w_tile(
        self, sW, sC, sK, sLCR, w_slot, k_stage_idx,
        q_start, chunk_len, log2_e, tidx,
    ):
        """Build W.T[q, p] with the causal mask p >= q."""
        sWTile = cute.slice_(sW, (None, None, w_slot))
        local_p = tidx % self.K_TILE
        first_local_q = tidx // self.K_TILE
        p = k_stage_idx * self.K_TILE + local_p

        rC = cute.make_rmem_tensor((self.num_m_blocks, self.NUM_W_ACCUMULATORS), Float32)
        rLCR = cute.make_rmem_tensor((self.num_m_blocks, self.NUM_W_ACCUMULATORS), Float32)
        for m_block_idx in cutlass.range_constexpr(self.num_m_blocks):
            m_start = m_block_idx * self.NUM_W_ACCUMULATORS
            for v in cutlass.range_constexpr(self.NUM_W_ACCUMULATORS):
                rC[m_block_idx, v] = sC[p, m_start + v].to(Float32)
                rLCR[m_block_idx, v] = sLCR[p, m_start + v]

        for i in cutlass.range_constexpr(self.num_w_row_iters):
            local_q = first_local_q + i * self.w_rows_per_iter
            if local_q < self.T_TILE:
                q = q_start + local_q
                # Accumulate M in independent sums, then add those sums in order.
                rW = cute.make_rmem_tensor((self.NUM_W_ACCUMULATORS,), Float32)
                rW.fill(0.0)
                # Mask before exponentiating to avoid overflow above the diagonal.
                if q < chunk_len and p < chunk_len and p >= q:
                    for m_block_idx in cutlass.range_constexpr(self.num_m_blocks):
                        m_start = m_block_idx * self.NUM_W_ACCUMULATORS
                        for v in cutlass.range_constexpr(self.NUM_W_ACCUMULATORS):
                            rW[v] += (
                                rC[m_block_idx, v] * sK[q, m_start + v].to(Float32)
                                * cute.math.exp2(
                                    (rLCR[m_block_idx, v] - sLCR[q, m_start + v]) * log2_e,
                                    fastmath=True,
                                )
                            )
                w_value = rW[0]
                for v in cutlass.range_constexpr(1, self.NUM_W_ACCUMULATORS):
                    w_value += rW[v]
                sWTile[local_q, local_p] = w_value.to(self.dtype)

    @cute.jit
    def store_du_epilogue(
        self, tiled_mma_du, tma_atom_du, mdU_lrb, sdU, sdUStore, tCrdU, tCcdU,
        t_tile_idx, r_tile_idx, chunk_start, chunk_len, batch,
        du_store_pipeline, tidx, warp_idx, mdU, first_k_stage, num_k_stages,
    ):
        r2s_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(LayoutEnum.ROW_MAJOR.is_m_major_c(), num_matrices=4),
            self.dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C(r2s_atom, tiled_mma_du)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sdU = thr_copy_r2s.partition_D(sdUStore)

        full_tile = and_(
            (t_tile_idx + 1) * self.T_TILE <= chunk_len,
            (r_tile_idx + 1) * self.R_TILE <= self.R,
        )
        if first_k_stage < num_k_stages:
            if full_tile:
                if warp_idx == self.tma_warp_idx:
                    du_store_pipeline.producer_acquire()
                cute.arch.sync_threads()
                cvt_copy_r2s(tiled_copy_r2s, tCrdU, tRS_sdU)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_threads()
                gdU = cute.local_tile(
                    mdU_lrb[None, None, batch], (self.T_TILE, self.R_TILE), (None, r_tile_idx)
                )
                bSG_sdU, bSG_gdU = cpasync.tma_partition(
                    tma_atom_du, cta_coord=0, cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sdU, 0, 2),
                    gmem_tensor=cute.group_modes(gdU, 0, 2),
                )
                du_global_tile = (chunk_start + t_tile_idx * self.T_TILE) // self.T_TILE
                if warp_idx == self.tma_warp_idx:
                    cute.copy(tma_atom_du, bSG_sdU[None, 0], bSG_gdU[None, du_global_tile])
                    du_store_pipeline.producer_commit()
                    du_store_pipeline.producer_tail()
            else:
                for i in cutlass.range(cute.size(tCrdU.shape), unroll_full=True):
                    coord = tCcdU[i]
                    q = t_tile_idx * self.T_TILE + coord[0]
                    r = r_tile_idx * self.R_TILE + coord[1]
                    if q < chunk_len and r < self.R:
                        mdU[batch, chunk_start + q, r] = tCrdU[i].to(mdU.element_type)


class ChunkFusedReadoutBwdParams:
    """Compute D = dH @ U.T and reduce the intra-chunk parameter gradients.

    The state stage later adds its contributions to dC and dLCR.
    """

    D_PADDING_ELEMENTS = 8  # Extra FP32 elements per D row.
    D_COPY_BITS = 64
    T_TILE = 64  # Time rows and columns per D tile.
    CONSUMER_REGS = 192
    PRODUCER_REGS = 64

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutBwdParamsConfig,
    ):
        self.dtype = dtype
        self.M = M
        self.chunk_size = chunk_size

        self.R_TILE = config.R_TILE
        self.num_t_tiles = math.ceil(chunk_size / self.T_TILE)
        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.D_TILE = self.num_t_tiles * self.T_TILE
        self.num_d_stages = 2
        self.max_ctas = config.max_ctas

        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        self.producer_load_warp_idx = self.num_consumer_warps
        self.num_producer_threads = config.num_producer_threads
        self.num_threads = self.num_consumer_threads + self.num_producer_threads
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

        self.num_threads_Q = cute.arch.WARP_SIZE // self.M
        self.num_param_iters = math.ceil(self.chunk_size * self.M / self.num_threads)

        assert dtype.width == 16, f"expected 16-bit input dtype, got {dtype}"
        assert M in (8, 16), f"M must be 8 or 16, got {M}"
        assert chunk_size in (32, 64, 128), f"unsupported chunk_size {chunk_size}"
        assert self.R_TILE in (32, 64, 128), (
            f"R_TILE must be 32, 64, or 128, got {self.R_TILE}"
        )
        assert self.num_producer_threads % cute.arch.WARP_SIZE == 0
        assert self.num_threads % cute.arch.WARP_SIZE == 0
        assert self.max_ctas is not None and self.max_ctas > 0
        assert self.num_r_tiles <= MAX_R_TILES, (
            f"num_r_tiles={self.num_r_tiles} is too large for an in-CTA loop; "
            "increase R_TILE or split the reduction across CTAs"
        )

    @cute.jit
    def __call__(
        self,
        mdH: cute.Tensor,         # (B, L, R)
        mU: cute.Tensor,          # (B, L, R)
        mC: cute.Tensor,          # (B, L, M)
        mK: cute.Tensor,          # (B, L, M)
        mLCR: cute.Tensor,        # (B, L, M), FP32
        mdC: cute.Tensor,         # (B, L, M)
        mdK: cute.Tensor,         # (B, L, M)
        mdLCR: cute.Tensor,       # (B, L, M), FP32
        stream: cuda.CUstream,
    ):
        B, L = mU.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)
        work_count = B * nchunks
        grid = cutlass.min(work_count, self.max_ctas)

        ################## SMEM layouts ##################
        d_mma_tiler = (self.T_TILE, self.T_TILE, self.R_TILE)
        # D reduces over R, which is contiguous in both dH and U.
        dh_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, d_mma_tiler, self.dtype, num_stages=self.num_d_stages
        )
        u_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, d_mma_tiler, self.dtype, num_stages=self.num_d_stages
        )
        # Row padding preserves copy alignment and changes the row-to-bank mapping.
        d_smem_layout = cute.make_layout(
            (self.D_TILE, self.D_TILE),
            stride=(self.D_TILE + self.D_PADDING_ELEMENTS, 1),
        )
        param_smem_layout = cute.make_ordered_layout(
            (self.chunk_size, self.M),
            order=(1, 0),
        )

        ################## MMA setup ##################
        mma_op_d = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.T_TILE, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma_d = cute.make_tiled_mma(mma_op_d)

        ################## TMA setup ##################
        mdH_lrb = cute.make_tensor(
            mdH.iterator, cute.select(mdH.layout, mode=(1, 2, 0)),
        )
        mU_lrb = cute.make_tensor(
            mU.iterator, cute.select(mU.layout, mode=(1, 2, 0)),
        )
        tma_atom_dh, tma_tensor_dh = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdH_lrb,
            cute.slice_(dh_smem_layout_staged, (None, None, 0)),
            (self.T_TILE, self.R_TILE),
        )
        tma_atom_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mU_lrb,
            cute.slice_(u_smem_layout_staged, (None, None, 0)),
            (self.T_TILE, self.R_TILE),
        )

        ################## parameter copy setup ##################
        param_vecsize = cutlass.const_expr(min(self.M, VECTOR_COPY_BITS // self.dtype.width))
        lcr_vecsize = cutlass.const_expr(min(self.M, VECTOR_COPY_BITS // Float32.width))
        tiled_copy_g2s_params = make_vec_g2s_copy(
            mC.element_type, self.chunk_size, self.M, self.num_threads, param_vecsize
        )
        tiled_copy_g2s_lcr = make_vec_g2s_copy(
            mLCR.element_type, self.chunk_size, self.M, self.num_threads, lcr_vecsize
        )

        ################## shared storage ##################
        @cute.struct
        class SharedStorage:
            d_barriers: cute.struct.MemRange[cutlass.Int64, self.num_d_stages * 2]
            sD: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(d_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sdC: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sdK: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sdH: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(dh_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(u_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_smem_layout)],
                PARAM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## launch ##################
        self.kernel(
            mU, mC, mK, mLCR, mdC, mdK, mdLCR, tma_tensor_dh, tma_tensor_u, tiled_mma_d,
            tiled_copy_g2s_params, tiled_copy_g2s_lcr, d_smem_layout,
            dh_smem_layout_staged, u_smem_layout_staged, param_smem_layout, tma_atom_dh, tma_atom_u,
        ).launch(
            grid=[grid, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mU, mC, mK, mLCR, mdC, mdK, mdLCR,
        mdH_lrb: cute.Tensor,
        mU_lrb: cute.Tensor,
        tiled_mma_d: cute.TiledMma,
        tiled_copy_g2s_params: cute.TiledCopy,
        tiled_copy_g2s_lcr: cute.TiledCopy,
        d_smem_layout, dh_smem_layout_staged, u_smem_layout_staged, param_smem_layout,
        tma_atom_dh: cute.CopyAtom,
        tma_atom_u: cute.CopyAtom,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        L = mU.shape[1]
        nchunks = cute.ceil_div(L, self.chunk_size)
        B = mU.shape[0]
        work_count = B * nchunks
        work_idx = block_idx

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        d_barrier_ptr = storage.d_barriers.data_ptr()
        sD = storage.sD.get_tensor(d_smem_layout)
        sdC = storage.sdC.get_tensor(param_smem_layout)
        sdK = storage.sdK.get_tensor(param_smem_layout)
        sdH = storage.sdH.get_tensor(
            dh_smem_layout_staged.outer,
            swizzle=dh_smem_layout_staged.inner,
        )
        sU = storage.sU.get_tensor(
            u_smem_layout_staged.outer,
            swizzle=u_smem_layout_staged.inner,
        )
        sC = storage.sC.get_tensor(param_smem_layout)
        sK = storage.sK.get_tensor(param_smem_layout)
        sLCR = storage.sLCR.get_tensor(param_smem_layout)

        if warp_idx == self.producer_load_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_u)

        ################## pipelines ##################
        d_tma_bytes = cutlass.const_expr(
            cute.size_in_bytes(
                self.dtype,
                cute.slice_(dh_smem_layout_staged, (None, None, 0)),
            )
            + cute.size_in_bytes(
                self.dtype,
                cute.slice_(u_smem_layout_staged, (None, None, 0)),
            )
        )
        d_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=d_barrier_ptr,
            num_stages=self.num_d_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=d_tma_bytes,
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()
        d_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_d_stages
        )
        d_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_d_stages
        )

        ################## persistent work loop ##################
        # Each CTA processes successive chunks, reusing its D and parameter buffers.
        while work_idx < work_count:
            batch = work_idx // nchunks
            chunk = work_idx - batch * nchunks
            chunk_start = chunk * self.chunk_size
            chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
            num_active_t_tiles = cute.ceil_div(chunk_len, self.T_TILE)

            if warp_idx >= self.num_consumer_warps:
                cute.arch.setmaxregister_decrease(self.PRODUCER_REGS)
            else:
                cute.arch.setmaxregister_increase(self.CONSUMER_REGS)

            ################## parameter loads ##################
            load_param(tiled_copy_g2s_params, mC, sC, batch, chunk_start, chunk_len, tidx)
            load_param(tiled_copy_g2s_params, mK, sK, batch, chunk_start, chunk_len, tidx)
            load_param(tiled_copy_g2s_lcr, mLCR, sLCR, batch, chunk_start, chunk_len, tidx)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            ################## D operand loads ##################
            if warp_idx == self.producer_load_warp_idx:
                mdHChunk = cute.domain_offset(
                    (chunk_start, 0),
                    mdH_lrb[None, None, batch],
                )
                mUChunk = cute.domain_offset(
                    (chunk_start, 0),
                    mU_lrb[None, None, batch],
                )
                for p_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                    for q_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                        if and_(
                            p_tile_idx < num_active_t_tiles,
                            q_tile_idx < num_active_t_tiles,
                            q_tile_idx <= p_tile_idx,
                        ):
                            gdH = cute.local_tile(
                                mdHChunk,
                                (self.T_TILE, self.R_TILE),
                                (p_tile_idx, None),
                            )
                            gU = cute.local_tile(
                                mUChunk,
                                (self.T_TILE, self.R_TILE),
                                (q_tile_idx, None),
                            )
                            bGS_sdH, bGS_gdH = cpasync.tma_partition(
                                tma_atom_dh,
                                cta_coord=0,
                                cta_layout=cute.make_layout(1),
                                smem_tensor=cute.group_modes(sdH, 0, 2),
                                gmem_tensor=cute.group_modes(gdH, 0, 2),
                            )
                            bGS_sU, bGS_gU = cpasync.tma_partition(
                                tma_atom_u,
                                cta_coord=0,
                                cta_layout=cute.make_layout(1),
                                smem_tensor=cute.group_modes(sU, 0, 2),
                                gmem_tensor=cute.group_modes(gU, 0, 2),
                            )
                            d_acquire_token = d_pipeline.producer_try_acquire(
                                d_producer_state
                            )
                            for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
                                d_pipeline.producer_acquire(
                                    d_producer_state, d_acquire_token,
                                )
                                barrier_ptr = d_pipeline.producer_get_barrier(
                                    d_producer_state
                                )
                                cute.copy(
                                    tma_atom_dh,
                                    bGS_gdH[None, r_tile_idx],
                                    bGS_sdH[None, d_producer_state.index],
                                    tma_bar_ptr=barrier_ptr,
                                )
                                cute.copy(
                                    tma_atom_u,
                                    bGS_gU[None, r_tile_idx],
                                    bGS_sU[None, d_producer_state.index],
                                    tma_bar_ptr=barrier_ptr,
                                )
                                d_pipeline.producer_commit(d_producer_state)
                                d_producer_state.advance()
                                if r_tile_idx + 1 < self.num_r_tiles:
                                    d_acquire_token = d_pipeline.producer_try_acquire(
                                        d_producer_state
                                    )

            ################## D WGMMA ##################
            if warp_idx < self.num_consumer_warps:
                thr_mma_d = tiled_mma_d.get_slice(tidx)
                tCrdH = thr_mma_d.make_fragment_A(thr_mma_d.partition_A(sdH))
                tCrU = thr_mma_d.make_fragment_B(thr_mma_d.partition_B(sU))
                r2s_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    Float32,
                    num_bits_per_copy=self.D_COPY_BITS,
                )
                tiled_copy_r2s = cute.make_tiled_copy_C(r2s_atom, tiled_mma_d)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                for p_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                    for q_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                        if and_(
                            p_tile_idx < num_active_t_tiles,
                            q_tile_idx < num_active_t_tiles,
                            q_tile_idx <= p_tile_idx,
                        ):
                            tCrD = thr_mma_d.make_fragment_C(
                                thr_mma_d.partition_shape_C((self.T_TILE, self.T_TILE))
                            )
                            tCrD.fill(0.0)
                            d_wait_token = d_pipeline.consumer_try_wait(d_consumer_state)
                            for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
                                d_pipeline.consumer_wait(d_consumer_state, d_wait_token)
                                cute.arch.fence_view_async_shared()
                                cute.nvgpu.warpgroup.fence()
                                gemm(
                                    tiled_mma_d,
                                    tCrdH[None, None, None, d_consumer_state.index],
                                    tCrU[None, None, None, d_consumer_state.index],
                                    tCrD,
                                    r_tile_idx != 0,
                                )
                                cute.nvgpu.warpgroup.commit_group()
                                cute.nvgpu.warpgroup.wait_group(0)
                                cute.arch.fence_view_async_shared()
                                d_pipeline.consumer_release(d_consumer_state)
                                d_consumer_state.advance()
                                if r_tile_idx + 1 < self.num_r_tiles:
                                    d_wait_token = d_pipeline.consumer_try_wait(
                                        d_consumer_state
                                    )

                            sDTile = cute.local_tile(
                                sD,
                                (self.T_TILE, self.T_TILE),
                                (p_tile_idx, q_tile_idx),
                            )
                            tRS_sD = thr_copy_r2s.partition_D(sDTile)
                            cvt_copy_r2s(tiled_copy_r2s, tCrD, tRS_sD)

            cute.arch.sync_threads()

            ################## parameter reductions ##################
            for i in cutlass.range_constexpr(self.num_param_iters):
                idx = tidx + i * self.num_threads
                if idx < self.chunk_size * self.M:
                    q = idx // self.M
                    m = idx - q * self.M
                    sdK[q, m] = Float32(0.0)
            cute.arch.sync_threads()

            self.reduce_params(
                sD, sC, sK, sLCR, sdC, sdK, chunk_len, warp_idx,
            )
            cute.arch.sync_threads()

            ################## parameter stores ##################
            self.store_params(
                sC, sK, sdC, sdK, mdC, mdK, mdLCR,
                batch, chunk_start, chunk_len, tidx,
            )
            cute.arch.sync_threads()
            work_idx += num_ctas

        ################## pipeline drain ##################
        if warp_idx == self.producer_load_warp_idx:
            d_pipeline.producer_tail(d_producer_state)

    @cute.jit
    def reduce_params(
        self, sD, sC, sK, sLCR, sdC, sdK, chunk_len, warp_idx,
    ):
        """Reduce dC across lanes and atomically accumulate dK in shared memory."""
        log2_e = math.log2(math.e)
        # Each causal (p, q, m) shares one exponential between dC and dK.
        lane_idx = cute.arch.lane_idx()
        m_lane = lane_idx // self.num_threads_Q
        q_lane = lane_idx % self.num_threads_Q

        for p_base in cutlass.range(0, self.chunk_size, self.num_warps, unroll=1):
            p = p_base + warp_idx
            dc_partial = Float32(0.0)
            lcr_p = Float32(0.0)
            c_p = Float32(0.0)
            if p < chunk_len:
                lcr_p = sLCR[p, m_lane]
                c_p = sC[p, m_lane].to(Float32)
            for q_base in cutlass.range(0, p + 1, self.num_threads_Q, unroll=1):
                q = q_base + q_lane
                if p < chunk_len and q <= p:
                    d_decay = sD[p, q] * cute.math.exp2(
                        (lcr_p - sLCR[q, m_lane]) * log2_e,
                        fastmath=True,
                    )
                    dc_partial += d_decay * sK[q, m_lane].to(Float32)
                    cute.arch.atomic_add(
                        (sdK.iterator + cute.crd2idx((q, m_lane), sdK.layout)).llvm_ptr,
                        d_decay * c_p,
                        sem="relaxed",
                        scope="cta",
                    )
            dc = cute.arch.warp_reduction(
                dc_partial, operator.add, threads_in_group=self.num_threads_Q
            )
            if q_lane == 0 and p < chunk_len:
                sdC[p, m_lane] = dc

    @cute.jit
    def store_params(
        self, sC, sK, sdC, sdK, mdC, mdK, mdLCR,
        batch, chunk_start, chunk_len, tidx,
    ):
        """Write intra-chunk gradients and derive dLCR from the FP32 dC and dK sums."""
        for i in cutlass.range_constexpr(self.num_param_iters):
            idx = tidx + i * self.num_threads
            if idx < self.chunk_size * self.M:
                p = idx // self.M
                m = idx - p * self.M
                if p < chunk_len:
                    global_t = chunk_start + p
                    dc = sdC[p, m]
                    dk = sdK[p, m]
                    c = sC[p, m].to(Float32)
                    k = sK[p, m].to(Float32)
                    mdC[batch, global_t, m] = dc.to(mdC.element_type)
                    mdK[batch, global_t, m] = dk.to(mdK.element_type)
                    mdLCR[batch, global_t, m] = c * dc - k * dk


class ChunkFusedReadoutBwdState:
    """Compute dStates and add the state contributions to dC and dLCR.

    Saved states are converted to the input dtype before WGMMA, matching forward.
    Accumulation stays in FP32. The additive parameter stores follow the
    intra-chunk stores from ChunkFusedReadoutBwdParams.
    """

    T_TILE = 64  # WGMMA instruction rows and time positions per state tile.
    M_PAD = 16  # Padded state channels in the matrix operands.
    CONSUMER_BARRIER_ID = 7

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutBwdStateConfig,
        states_dtype: type[cutlass.Numeric],
    ):
        self.dtype = dtype
        self.states_dtype = states_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        self.R_TILE = config.R_TILE
        self.num_t_tiles = math.ceil(chunk_size / self.T_TILE)
        self.num_r_tiles = math.ceil(R / self.R_TILE)

        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        self.producer_load_warp_idx = self.num_consumer_warps
        self.num_producer_threads = config.num_producer_threads
        self.num_threads = self.num_consumer_threads + self.num_producer_threads
        self.dstates_tma_warp_idx = 0
        self.consumer_barrier = pipeline.NamedBarrier(
            self.CONSUMER_BARRIER_ID, self.num_consumer_threads,
        )

        self.num_c_load_iters = math.ceil(self.T_TILE * self.M / self.num_producer_threads)
        self.num_state_load_iters = math.ceil(self.R_TILE * self.M / self.num_producer_threads)

        assert dtype.width == 16, f"expected 16-bit input dtype, got {dtype}"
        assert M in (8, 16), f"M must be 8 or 16, got {M}"
        assert chunk_size in (32, 64, 128), f"unsupported chunk_size {chunk_size}"
        assert self.R_TILE in (64, 128), f"unsupported R_TILE {self.R_TILE}"
        assert self.num_r_tiles <= MAX_R_TILES, (
            f"num_r_tiles={self.num_r_tiles} is too large for one inter-kernel CTA loop"
        )

    @cute.jit
    def __call__(
        self,
        mdH: cute.Tensor,         # (B, L, R)
        mC: cute.Tensor,          # (B, L, M)
        mStatesAcc: cute.Tensor,  # (B, nchunks, R, M)
        mLCR: cute.Tensor,        # (B, L, M), FP32
        mdC: cute.Tensor,         # (B, L, M), accumulator
        mdStates: cute.Tensor,    # (B, nchunks, R, M)
        mdLCR: cute.Tensor,       # (B, L, M), accumulator
        stream: cuda.CUstream,
    ):
        B, L = mdH.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM layouts ##################
        dstates_mma_tiler = (self.R_TILE, self.M_PAD, self.T_TILE)
        de_mma_tiler = (self.T_TILE, self.M_PAD, self.R_TILE)
        dh_dstates_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, dstates_mma_tiler, self.dtype, num_stages=1
        )
        c_scaled_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR, dstates_mma_tiler, self.dtype, num_stages=1
        )
        # dE reduces over contiguous R values in dH, so its A operand is K-major.
        dh_de_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, de_mma_tiler, self.dtype, num_stages=1
        )
        states_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR, de_mma_tiler, self.dtype, num_stages=1
        )
        # The MMA uses M_PAD columns. TMA discards stores beyond M.
        dstates_smem_layout = cute.make_ordered_layout(
            (self.R_TILE, self.M_PAD),
            order=(1, 0),
        )

        ################## MMA setup ##################
        mma_op_dstates = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.M_PAD, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_mma_dstates = cute.make_tiled_mma(mma_op_dstates)
        mma_op_de = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.M_PAD, MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_mma_de = cute.make_tiled_mma(mma_op_de)

        ################## TMA setup ##################
        mdH_rlb = cute.make_tensor(
            mdH.iterator, cute.select(mdH.layout, mode=(2, 1, 0)),
        )
        mdH_lrb = cute.make_tensor(
            mdH.iterator, cute.select(mdH.layout, mode=(1, 2, 0)),
        )
        mdStates_rmbn = cute.make_tensor(
            mdStates.iterator, cute.select(mdStates.layout, mode=(2, 3, 0, 1)),
        )
        tma_atom_dh_dstates, tma_tensor_dh_dstates = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdH_rlb,
            cute.slice_(dh_dstates_smem_layout_staged, (None, None, 0)),
            (self.R_TILE, self.T_TILE),
        )
        tma_atom_dh_de, tma_tensor_dh_de = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdH_lrb,
            cute.slice_(dh_de_smem_layout_staged, (None, None, 0)),
            (self.T_TILE, self.R_TILE),
        )
        tma_atom_dstates, tma_tensor_dstates = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdStates_rmbn, dstates_smem_layout,
            (self.R_TILE, self.M_PAD),
        )

        ################## shared storage ##################
        @cute.struct
        class SharedStorage:
            # Each load pipeline has a full and an empty barrier for its single stage.
            dstates_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            de_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sdHDStates: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(dh_dstates_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sCScaled: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(c_scaled_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sdHDE: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(dh_de_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sStatesAcc: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(states_smem_layout_staged)],
                SMEM_ALIGNMENT_BYTES,
            ]
            sdStates: cute.struct.Align[
                cute.struct.MemRange[self.states_dtype, cute.cosize(dstates_smem_layout)],
                SMEM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## launch ##################
        self.kernel(
            mdH, mC, mStatesAcc, mLCR, mdC, mdLCR,
            tma_tensor_dh_dstates, tma_tensor_dh_de, tma_tensor_dstates,
            tiled_mma_dstates, tiled_mma_de,
            dh_dstates_smem_layout_staged, c_scaled_smem_layout_staged,
            dh_de_smem_layout_staged, states_smem_layout_staged, dstates_smem_layout,
            tma_atom_dh_dstates, tma_atom_dh_de, tma_atom_dstates,
        ).launch(
            grid=[self.num_r_tiles, nchunks, B],
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self, mdH, mC, mStatesAcc, mLCR, mdC, mdLCR,
        mdH_rlb: cute.Tensor,
        mdH_lrb: cute.Tensor,
        mdStates_rmbn: cute.Tensor,
        tiled_mma_dstates: cute.TiledMma,
        tiled_mma_de: cute.TiledMma,
        dh_dstates_smem_layout_staged, c_scaled_smem_layout_staged, dh_de_smem_layout_staged,
        states_smem_layout_staged, dstates_smem_layout,
        tma_atom_dh_dstates: cute.CopyAtom,
        tma_atom_dh_de: cute.CopyAtom,
        tma_atom_dstates: cute.CopyAtom,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        r_tile_idx, chunk, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        L = mdH.shape[1]
        chunk_start = chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        dstates_barrier_ptr = storage.dstates_barriers.data_ptr()
        de_barrier_ptr = storage.de_barriers.data_ptr()
        sdHDStates = storage.sdHDStates.get_tensor(
            dh_dstates_smem_layout_staged.outer,
            swizzle=dh_dstates_smem_layout_staged.inner,
        )
        sCScaled = storage.sCScaled.get_tensor(
            c_scaled_smem_layout_staged.outer,
            swizzle=c_scaled_smem_layout_staged.inner,
        )
        sdHDE = storage.sdHDE.get_tensor(
            dh_de_smem_layout_staged.outer,
            swizzle=dh_de_smem_layout_staged.inner,
        )
        sStatesAcc = storage.sStatesAcc.get_tensor(
            states_smem_layout_staged.outer,
            swizzle=states_smem_layout_staged.inner,
        )
        sdStates = storage.sdStates.get_tensor(dstates_smem_layout)

        if warp_idx == self.producer_load_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_dh_dstates)
            cpasync.prefetch_descriptor(tma_atom_dh_de)
        if warp_idx == self.dstates_tma_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_dstates)

        ################## pipelines ##################
        dstates_tma_bytes = cutlass.const_expr(
            cute.size_in_bytes(
                self.dtype,
                cute.slice_(dh_dstates_smem_layout_staged, (None, None, 0)),
            )
        )
        de_tma_bytes = cutlass.const_expr(
            cute.size_in_bytes(
                self.dtype,
                cute.slice_(dh_de_smem_layout_staged, (None, None, 0)),
            )
        )
        dstates_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=dstates_barrier_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=dstates_tma_bytes,
            defer_sync=True,
        )
        de_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=de_barrier_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=de_tma_bytes,
            defer_sync=True,
        )
        dstates_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()
        dstates_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        dstates_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        de_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        de_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        ################## MMA fragments ##################
        thr_mma_dstates = tiled_mma_dstates.get_slice(tidx)
        tCrdHDStates = thr_mma_dstates.make_fragment_A(thr_mma_dstates.partition_A(sdHDStates))
        tCrCScaled = thr_mma_dstates.make_fragment_B(thr_mma_dstates.partition_B(sCScaled))
        tCrdStates = thr_mma_dstates.make_fragment_C(
            thr_mma_dstates.partition_shape_C((self.R_TILE, self.M_PAD))
        )

        ################## state gradient loop ##################
        # Producers stage C * exp(LCR) and dH for the consumer WGMMA.
        for t_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
            tile_t_start = t_tile_idx * self.T_TILE
            if warp_idx >= self.num_consumer_warps:
                self.load_scaled_c(
                    sCScaled, mC, mLCR, batch, chunk_start, tile_t_start, chunk_len, tidx,
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            if warp_idx == self.producer_load_warp_idx:
                gdHDStates = cute.local_tile(
                    cute.domain_offset(
                        (
                            r_tile_idx * self.R_TILE,
                            chunk_start + tile_t_start,
                        ),
                        mdH_rlb[None, None, batch],
                    ),
                    (self.R_TILE, self.T_TILE),
                    (0, 0),
                )
                bGS_sdHDStates, bGS_gdHDStates = cpasync.tma_partition(
                    tma_atom_dh_dstates,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sdHDStates, 0, 2),
                    gmem_tensor=cute.group_modes(gdHDStates, 0, 2),
                )
                dstates_pipeline.producer_acquire(dstates_producer_state)
                cute.copy(
                    tma_atom_dh_dstates,
                    bGS_gdHDStates[None],
                    bGS_sdHDStates[None, dstates_producer_state.index],
                    tma_bar_ptr=dstates_pipeline.producer_get_barrier(
                        dstates_producer_state
                    ),
                )
                dstates_pipeline.producer_commit(dstates_producer_state)
                dstates_producer_state.advance()

            if warp_idx < self.num_consumer_warps:
                dstates_pipeline.consumer_wait(dstates_consumer_state)
                cute.arch.fence_view_async_shared()
                cute.nvgpu.warpgroup.fence()
                gemm(
                    tiled_mma_dstates,
                    tCrdHDStates[None, None, None, dstates_consumer_state.index],
                    tCrCScaled[None, None, None, 0],
                    tCrdStates,
                    t_tile_idx != 0,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                cute.arch.fence_view_async_shared()
                dstates_pipeline.consumer_release(dstates_consumer_state)
                dstates_consumer_state.advance()

            cute.arch.sync_threads()

        if warp_idx == self.producer_load_warp_idx:
            dstates_pipeline.producer_tail(dstates_producer_state)

        ################## state gradient stores ##################
        if warp_idx < self.num_consumer_warps:
            if warp_idx == self.dstates_tma_warp_idx:
                dstates_store_pipeline.producer_acquire()
            self.consumer_barrier.arrive_and_wait()
            self.store_dstates_r2s(tiled_mma_dstates, tCrdStates, sdStates, tidx)
            cute.arch.fence_view_async_shared()
            self.consumer_barrier.arrive_and_wait()

        if warp_idx == self.dstates_tma_warp_idx:
            gdStates = cute.local_tile(
                mdStates_rmbn[None, None, batch, chunk],
                (self.R_TILE, self.M_PAD),
                (r_tile_idx, 0),
            )
            bSG_sdStates, bSG_gdStates = cpasync.tma_partition(
                tma_atom_dstates,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdStates, 0, 2),
                gmem_tensor=cute.group_modes(gdStates, 0, 2),
            )
            cute.copy(tma_atom_dstates, bSG_sdStates[None], bSG_gdStates[None])
            dstates_store_pipeline.producer_commit()
            dstates_store_pipeline.producer_tail()

        cute.arch.sync_threads()

        ################## dE and parameter gradients ##################
        # Only the first R-tile CTA adds the state term to dC and dLCR.
        if r_tile_idx == 0:
            thr_mma_de = tiled_mma_de.get_slice(tidx)
            tCrdHDE = thr_mma_de.make_fragment_A(thr_mma_de.partition_A(sdHDE))
            tCrStatesAcc = thr_mma_de.make_fragment_B(thr_mma_de.partition_B(sStatesAcc))
            cdE = cute.make_identity_tensor((self.T_TILE, self.M_PAD))
            tCcdE = thr_mma_de.partition_C(cdE)
            for t_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                tile_t_start = t_tile_idx * self.T_TILE
                tCrdE = thr_mma_de.make_fragment_C(
                    thr_mma_de.partition_shape_C((self.T_TILE, self.M_PAD))
                )
                tCrdE.fill(0.0)
                for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
                    if warp_idx >= self.num_consumer_warps:
                        self.load_state(sStatesAcc, mStatesAcc, batch, chunk, r_tile_idx, tidx)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()

                    if warp_idx == self.producer_load_warp_idx:
                        gdHDE = cute.local_tile(
                            cute.domain_offset(
                                (
                                    chunk_start + tile_t_start,
                                    r_tile_idx * self.R_TILE,
                                ),
                                mdH_lrb[None, None, batch],
                            ),
                            (self.T_TILE, self.R_TILE),
                            (0, 0),
                        )
                        bGS_sdHDE, bGS_gdHDE = cpasync.tma_partition(
                            tma_atom_dh_de,
                            cta_coord=0,
                            cta_layout=cute.make_layout(1),
                            smem_tensor=cute.group_modes(sdHDE, 0, 2),
                            gmem_tensor=cute.group_modes(gdHDE, 0, 2),
                        )
                        de_pipeline.producer_acquire(de_producer_state)
                        cute.copy(
                            tma_atom_dh_de,
                            bGS_gdHDE[None],
                            bGS_sdHDE[None, de_producer_state.index],
                            tma_bar_ptr=de_pipeline.producer_get_barrier(
                                de_producer_state
                            ),
                        )
                        de_pipeline.producer_commit(de_producer_state)
                        de_producer_state.advance()

                    if warp_idx < self.num_consumer_warps:
                        de_pipeline.consumer_wait(de_consumer_state)
                        cute.arch.fence_view_async_shared()
                        cute.nvgpu.warpgroup.fence()
                        gemm(
                            tiled_mma_de,
                            tCrdHDE[None, None, None, de_consumer_state.index],
                            tCrStatesAcc[None, None, None, 0],
                            tCrdE,
                            r_tile_idx != 0,
                        )
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        cute.arch.fence_view_async_shared()
                        de_pipeline.consumer_release(de_consumer_state)
                        de_consumer_state.advance()

                    cute.arch.sync_threads()

                ################## parameter gradient stores ##################
                if warp_idx < self.num_consumer_warps:
                    self.add_state_grads(
                        tCrdE, tCcdE, mC, mLCR, mdC, mdLCR,
                        batch, chunk_start, tile_t_start, chunk_len,
                    )
            if warp_idx == self.producer_load_warp_idx:
                # producer_tail mutates its state. Use a local clone so that mutation
                # stays inside this warp branch and does not create an invalid scf.yield.
                de_pipeline.producer_tail(de_producer_state.clone())

    @cute.jit
    def load_scaled_c(self, sCScaled, mC, mLCR, batch, chunk_start, tile_t_start, chunk_len, tidx):
        """Stage cast(C * exp(LCR)) for dStates and zero the partial time tile."""
        log2_e = math.log2(math.e)
        producer_tidx = tidx - self.num_consumer_threads
        sCScaledTile = cute.slice_(sCScaled, (None, None, 0))
        for i in cutlass.range_constexpr(self.num_c_load_iters):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.T_TILE * self.M:
                local_t = idx // self.M
                m = idx - local_t * self.M
                t = tile_t_start + local_t
                global_t = chunk_start + t
                if t < chunk_len:
                    c = mC[batch, global_t, m]
                    lcr = mLCR[batch, global_t, m]
                    c_scaled = c.to(Float32) * cute.math.exp2(
                        lcr * log2_e,
                        fastmath=True,
                    )
                    sCScaledTile[m, local_t] = c_scaled.to(self.dtype)
                else:
                    sCScaledTile[m, local_t] = self.dtype(0.0)

    @cute.jit
    def load_state(self, sStatesAcc, mStatesAcc, batch, chunk, r_tile_idx, tidx):
        """Stage saved state in the same input dtype consumed by forward WGMMA."""
        producer_tidx = tidx - self.num_consumer_threads
        sStatesAccTile = cute.slice_(sStatesAcc, (None, None, 0))
        for i in cutlass.range_constexpr(self.num_state_load_iters):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.R_TILE * self.M:
                local_r = idx // self.M
                m = idx - local_r * self.M
                r = r_tile_idx * self.R_TILE + local_r
                if r < self.R:
                    sStatesAccTile[m, local_r] = mStatesAcc[
                        batch, chunk, r, m
                    ].to(self.dtype)
                else:
                    sStatesAccTile[m, local_r] = self.dtype(0.0)

    @cute.jit
    def store_dstates_r2s(self, tiled_mma_dstates, tCrdStates, sdStates, tidx):
        """Convert the FP32 accumulator to the saved-state dtype for the TMA store."""
        r2s_atom = cutlass.utils.sm90_get_smem_store_op(
            LayoutEnum.from_tensor(sdStates),
            elem_ty_d=self.states_dtype,
            elem_ty_acc=Float32,
        )
        # The 16-bit atom supplies the partition layout for either output dtype.
        r2s_layout_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                LayoutEnum.from_tensor(sdStates).is_m_major_c(),
                num_matrices=4,
            ),
            Float16,
        )
        tiled_copy_r2s_layout = cute.make_tiled_copy_C_atom(r2s_layout_atom, tiled_mma_dstates)
        tiled_copy_r2s = cute.make_tiled_copy_S(
            r2s_atom, tiled_copy_r2s_layout,
        )
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sdStates = thr_copy_r2s.partition_D(sdStates)
        cvt_copy_r2s(tiled_copy_r2s, tCrdStates, tRS_sdStates)

    @cute.jit
    def add_state_grads(
        self, tCrdE, tCcdE, mC, mLCR, mdC, mdLCR,
        batch, chunk_start, tile_t_start, chunk_len,
    ):
        """Add the state term after intra-chunk dC has rounded to the input dtype."""
        log2_e = math.log2(math.e)
        for i in cutlass.range(cute.size(tCrdE.shape), unroll_full=True):
            coord = tCcdE[i]
            local_t = coord[0]
            m = coord[1]
            t = tile_t_start + local_t
            if t < chunk_len and m < self.M:
                global_t = chunk_start + t
                lcr = mLCR[batch, global_t, m]
                c = mC[batch, global_t, m].to(Float32)
                decay = cute.math.exp2(lcr * log2_e, fastmath=True)
                dc_inter = tCrdE[i] * decay
                dlcr_inter = tCrdE[i] * c * decay
                mdC[batch, global_t, m] = (
                    mdC[batch, global_t, m].to(Float32) + dc_inter
                ).to(mdC.element_type)
                mdLCR[batch, global_t, m] = (
                    mdLCR[batch, global_t, m] + dlcr_inter
                )


def cooperative_du_configs():
    configs = []
    for R_TILE, num_warp_groups in ((64, 1), (128, 1), (128, 2), (192, 1)):
        num_threads = cute.arch.THREADS_PER_WARPGROUP * num_warp_groups  # 128
        accumulator_registers = ChunkFusedReadoutBwdDu.T_TILE * R_TILE // num_threads
        for min_blocks_per_mp in (1, 2, 3, 4):
            register_budget = (
                REGISTERS_PER_SM // (num_threads * min_blocks_per_mp)
                // REGISTER_GRANULARITY * REGISTER_GRANULARITY
            )
            if register_budget >= accumulator_registers + DU_REGISTER_OVERHEAD:
                configs.append(AutotuneConfig(config=ChunkFusedReadoutBwdDuConfig(
                    R_TILE=R_TILE,
                    num_warp_groups=num_warp_groups,
                    min_blocks_per_mp=min_blocks_per_mp,
                )))
    return configs


def cooperative_param_configs():
    return [
        AutotuneConfig(config=ChunkFusedReadoutBwdParamsConfig(
            R_TILE=R_TILE, num_producer_threads=num_producer_threads, ctas_per_sm=ctas_per_sm,
        ))
        for R_TILE, num_producer_threads, ctas_per_sm in product(
            (32, 64, 128), (32, 128, 384), (1, 2, 4),
        )
    ]


def cooperative_state_configs():
    return [
        AutotuneConfig(config=ChunkFusedReadoutBwdStateConfig(
            R_TILE=R_TILE,
            num_producer_threads=num_producer_threads,
            min_blocks_per_mp=min_blocks_per_mp,
        ))
        for R_TILE, num_producer_threads, min_blocks_per_mp in product(
            (64, 128), (32, 128, 384), (1, 2),
        )
    ]


@dsl_user_op
def make_vec_g2s_copy(
    dtype: cutlass.Constexpr,
    rows: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    vecsize: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> cute.TiledCopy:
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(),
        dtype,
        num_bits_per_copy=cutlass.const_expr(vecsize * dtype.width),
        loc=loc,
        ip=ip,
    )
    num_threads_M = cutlass.const_expr(cols // vecsize)
    # Round the copy's row count down to a power of two that divides the tile.
    # With 160 threads, a 40-row copy would overrun a 64-row buffer because
    # predicated cp.async still zero-fills its destination.
    num_threads_T = cutlass.const_expr(1 << (
        min(rows, max(1, num_threads // num_threads_M)).bit_length() - 1
    ))
    thr_layout_g2s = cute.make_ordered_layout(
        (num_threads_T, num_threads_M),
        order=(1, 0),
    )
    val_layout_g2s = cute.make_ordered_layout((1, vecsize), order=(1, 0))
    return cute.make_tiled_copy_tv(g2s_atom, thr_layout_g2s, val_layout_g2s, loc=loc, ip=ip)


@cute.jit
def load_param(tiled_copy_g2s, mX, sX, batch, chunk_start, chunk_len, tidx):
    """Issue a predicated parameter load for the caller's cp.async group."""
    gX = cute.local_tile(
        cute.domain_offset((chunk_start, 0), mX[batch, None, None]),
        sX.shape, (0, 0),
    )
    if tidx < cute.size(tiled_copy_g2s):
        thr_copy_g2s = tiled_copy_g2s.get_slice(tidx)
        cX = cute.make_identity_tensor(sX.shape)
        tGS_cX = thr_copy_g2s.partition_S(cX)
        tGS_gX = thr_copy_g2s.partition_S(gX)
        tGS_sX = thr_copy_g2s.partition_D(sX)
        tGS_pX = predicate_axis(tGS_cX, chunk_len, axis=0)
        cute.copy(tiled_copy_g2s, tGS_gX, tGS_sX, pred=tGS_pX)


@cute.jit
def gemm(tiled_mma, tCrA, tCrB, tCrAcc, accumulate_initial: Boolean):
    """Accumulate K blocks in order with optional accumulation on the first MMA."""
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k_block_idx in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
        mma_atom.set(
            cute.nvgpu.warpgroup.Field.ACCUMULATE,
            accumulate_initial or k_block_idx != 0,
        )
        cute.gemm(
            mma_atom, tCrAcc,
            tCrA[None, None, k_block_idx], tCrB[None, None, k_block_idx], tCrAcc,
        )


@jit_cache
def _compile_chunk_fused_readout_bwd_du(dtype, R, M, chunk_size, config):
    param_vecsize = VECTOR_COPY_BITS // dtype.width
    lcr_vecsize = VECTOR_COPY_BITS // Float32.width
    B = cute.sym_int()
    L = cute.sym_int()
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B, L, M), divisibility=param_vecsize)
    fake_kappa = make_fake_tensor(dtype, (B, L, M), divisibility=param_vecsize)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=lcr_vecsize)
    fake_du = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwdDu(dtype, R=R, M=M, chunk_size=chunk_size, config=config),
        fake_dh, fake_c, fake_kappa, fake_lcr, fake_du,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@jit_cache
def _compile_chunk_fused_readout_bwd_params(dtype, R, M, chunk_size, config):
    B = cute.sym_int()
    L = cute.sym_int()
    param_vecsize = VECTOR_COPY_BITS // dtype.width
    lcr_vecsize = VECTOR_COPY_BITS // Float32.width
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_u = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B, L, M), divisibility=param_vecsize)
    fake_kappa = make_fake_tensor(dtype, (B, L, M), divisibility=param_vecsize)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=lcr_vecsize)
    fake_dc = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_dk = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_dlcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwdParams(dtype, R=R, M=M, chunk_size=chunk_size, config=config),
        fake_dh, fake_u, fake_c, fake_kappa, fake_lcr, fake_dc, fake_dk, fake_dlcr,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@jit_cache
def _compile_chunk_fused_readout_bwd_state(dtype, states_dtype, R, M, chunk_size, config):
    B = cute.sym_int()
    L = cute.sym_int()
    nchunks = cute.sym_int()
    param_vecsize = VECTOR_COPY_BITS // dtype.width
    lcr_vecsize = VECTOR_COPY_BITS // Float32.width
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B, L, M), divisibility=param_vecsize)
    fake_states_acc = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=lcr_vecsize)
    fake_dc = make_fake_tensor(dtype, (B, L, M), divisibility=1)
    fake_dstates = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    fake_dlcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwdState(
            dtype, states_dtype=states_dtype,
            R=R,
            M=M,
            chunk_size=chunk_size,
            config=config,
        ),
        fake_dh, fake_c, fake_states_acc, fake_lcr, fake_dc, fake_dstates, fake_dlcr,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _resolve_param_config(config, device):
    max_ctas = config.max_ctas
    if max_ctas is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_ctas = num_sms * config.ctas_per_sm
    # The kernel clamps the grid to its current work count at launch.
    return replace(config, max_ctas=max_ctas)


def _prune_coop_params(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    d_h, chunk_size = args["d_h"], args["chunk_size"]
    work = d_h.shape[0] * math.ceil(d_h.shape[1] / chunk_size)
    seen, result = set(), []
    for candidate in configs:
        config = _resolve_param_config(candidate.kwargs["config"], d_h.device)
        if math.ceil(d_h.shape[-1] / config.R_TILE) > MAX_R_TILES:
            continue
        key = (config.R_TILE, config.num_producer_threads, min(work, config.max_ctas))
        if key not in seen:
            seen.add(key)
            result.append(AutotuneConfig(config=config))
    return result


def _prune_coop_state(configs, named_args, **kwargs):
    R = named_args["d_h"].shape[-1]
    return [
        candidate for candidate in configs
        if math.ceil(R / candidate.kwargs["config"].R_TILE) <= MAX_R_TILES
    ]


def _prepare_du(d_h, c, kappa, lcr, du, *, chunk_size, config):
    return _compile_chunk_fused_readout_bwd_du(
        torch2cute_dtype_map[d_h.dtype], d_h.shape[-1], c.shape[-1], chunk_size, config,
    )


def _prepare_params(d_h, u, c, kappa, lcr, dc, dk, dlcr, *, chunk_size, config):
    config = _resolve_param_config(config, u.device)
    return _compile_chunk_fused_readout_bwd_params(
        torch2cute_dtype_map[d_h.dtype], u.shape[-1], c.shape[-1], chunk_size, config,
    )


def _prepare_state(d_h, c, states_acc, lcr, dc, dstates, dlcr, *, chunk_size, config):
    return _compile_chunk_fused_readout_bwd_state(
        torch2cute_dtype_map[d_h.dtype], torch2cute_dtype_map[states_acc.dtype],
        d_h.shape[-1], c.shape[-1], chunk_size, config,
    )


@autotune(
    configs=cooperative_du_configs(), key=["chunk_size", ("d_h", 2), ("c", 2)],
    cache_results=False,
)
def _tune_du(d_h, c, kappa, lcr, du, *, chunk_size, config):
    kernel = _prepare_du(d_h, c, kappa, lcr, du, chunk_size=chunk_size, config=config)
    kernel(d_h, c, kappa, lcr, du)


@autotune(
    configs=cooperative_param_configs(), key=["chunk_size", ("d_h", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": _prune_coop_params}, cache_results=False,
)
def _tune_params(d_h, u, c, kappa, lcr, dc, dk, dlcr, *, chunk_size, config):
    kernel = _prepare_params(
        d_h, u, c, kappa, lcr, dc, dk, dlcr, chunk_size=chunk_size, config=config,
    )
    kernel(d_h, u, c, kappa, lcr, dc, dk, dlcr)


@autotune(
    configs=cooperative_state_configs(), key=["chunk_size", ("d_h", 2), ("c", 2)],
    restore_value=["dc", "dlcr"],
    prune_configs_by={"early_config_prune": _prune_coop_state}, cache_results=False,
)
def _tune_state(d_h, c, states_acc, lcr, dc, dstates, dlcr, *, chunk_size, config):
    kernel = _prepare_state(
        d_h, c, states_acc, lcr, dc, dstates, dlcr, chunk_size=chunk_size, config=config,
    )
    kernel(d_h, c, states_acc, lcr, dc, dstates, dlcr)


_backward_launch_cache = {}
_backward_launch_lock = threading.Lock()
_prepare_stages = (_prepare_du, _prepare_params, _prepare_state)


def _get_autotuned_kernels(key, args, chunk_size):
    kernels = _backward_launch_cache.get(key)
    if kernels is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up cooperative backward autotuning before CUDA graph capture")
        with _backward_launch_lock:
            kernels = _backward_launch_cache.get(key)
            if kernels is None:
                kernels = []
                for tuner, prepare, inputs in zip(
                    (_tune_du, _tune_params, _tune_state), _prepare_stages, args,
                ):
                    tuner(*inputs, chunk_size=chunk_size)
                    config = tuner.best_config.kwargs["config"]
                    kernels.append(prepare(*inputs, chunk_size=chunk_size, config=config))
                kernels = tuple(kernels)
                _backward_launch_cache[key] = kernels
    return kernels


def _frac_chunk_fused_readout_bwd(
    d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size,
    config: tuple[
        ChunkFusedReadoutBwdDuConfig, ChunkFusedReadoutBwdParamsConfig,
        ChunkFusedReadoutBwdStateConfig,
    ] | None = None,
):
    """Run dU, parameter, and state gradients with optional configs for all three stages."""
    B, L, R = u.shape
    M = c.shape[-1]
    du = torch.empty((B, L, R), device=u.device, dtype=u.dtype)
    dc = torch.empty((B, L, M), device=c.device, dtype=c.dtype)
    dk = torch.empty((B, L, M), device=kappa.device, dtype=kappa.dtype)
    dstates = torch.empty_like(states_acc)
    dlcr = torch.empty((B, L, M), device=log_cum_rho.device, dtype=torch.float32)

    if B == 0 or L == 0:
        return du, dc, dk, dstates, dlcr
    assert u.dtype in (torch.float16, torch.bfloat16), f"u must be 16-bit, got {u.dtype}"
    assert d_h.dtype == c.dtype == kappa.dtype == u.dtype, "d_h, c, and kappa dtype must match u"
    assert log_cum_rho.dtype == torch.float32, "log_cum_rho must be float32"
    assert chunk_size in (32, 64, 128), f"unsupported chunk_size {chunk_size}"
    assert M in (8, 16), f"M must be 8 or 16, got {M}"

    args = (
        (d_h, c, kappa, log_cum_rho, du),
        (d_h, u, c, kappa, log_cum_rho, dc, dk, dlcr),
        (d_h, c, states_acc, log_cum_rho, dc, dstates, dlcr),
    )
    if config is None:
        key = (u.device.index, u.dtype, states_acc.dtype, R, M, chunk_size)
        kernels = _get_autotuned_kernels(key, args, chunk_size)
    else:
        assert len(config) == 3, "config must contain dU, parameter, and state configs"
        kernels = tuple(
            prepare(*inputs, chunk_size=chunk_size, config=stage_config)
            for prepare, inputs, stage_config in zip(_prepare_stages, args, config)
        )

    # Replay all stages after tuning so the state term is added to dC and dLCR once.
    for kernel, inputs in zip(kernels, args):
        kernel(*inputs)
    return du, dc, dk, dstates, dlcr


__all__ = ["_frac_chunk_fused_readout_bwd"]
