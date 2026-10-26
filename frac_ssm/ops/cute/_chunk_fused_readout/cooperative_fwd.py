"""Cooperative SM90 forward for fused chunk readout.

Each CTA computes a (T_TILE, R_TILE) output tile. Its warp groups divide
the R columns.

    W[t, k] = sum_m C[t, m] * K[k, m] * exp(LCR[t, m] - LCR[k, m]), k <= t
    H = W @ U + (C * exp(LCR)) @ states_acc.T

W is zero above the diagonal. U is loaded with TMA and W is computed in
shared memory. Computing the next W tile overlaps the current WGMMA. The
state term uses the same accumulator, and H is stored with TMA.
"""

from dataclasses import dataclass
from itertools import product
import math
import threading

import torch
import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32
from cutlass.tensor_utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.utils import make_fake_tensor, torch2cute_dtype_map


R_TILES = (64, 128, 192, 256)
WARP_GROUP_COUNTS = (1, 2, 4)
REGISTERS_PER_SM = 65536
MAX_THREADS_PER_SM = 2048
REGISTER_GRANULARITY = 8
ESTIMATED_REGISTER_OVERHEAD = 32


@dataclass(frozen=True)
class ChunkFusedReadoutFwdConfig:
    R_TILE: int = 128
    num_warp_groups: int = 2
    min_blocks_per_mp: int = 3


def cooperative_configs():
    # All shapes use the same resource search space. Reduction tiles are fixed.
    configs = []
    for R_TILE, num_warp_groups, min_blocks_per_mp in product(
        R_TILES, WARP_GROUP_COUNTS, (1, 2, 3, 4, 5, 6),
    ):
        # Each warp group's column count must be a multiple or divisor of
        # the 64-element SMEM atom width.
        mma_n = R_TILE // num_warp_groups
        if not ChunkFusedReadoutFwd.supports_mma_n(mma_n):
            continue
        # Round the register budget per thread down to a multiple of eight.
        # Estimate 32 registers of overhead beyond the WGMMA accumulator.
        num_threads = cute.arch.THREADS_PER_WARPGROUP * num_warp_groups  # 128
        register_budget = (
            REGISTERS_PER_SM // (num_threads * min_blocks_per_mp)
            // REGISTER_GRANULARITY * REGISTER_GRANULARITY
        )
        accumulator_registers = (
            ChunkFusedReadoutFwd.T_TILE * mma_n // cute.arch.THREADS_PER_WARPGROUP  # 128
        )
        estimated_registers = accumulator_registers + ESTIMATED_REGISTER_OVERHEAD
        if register_budget < estimated_registers:
            continue
        configs.append(AutotuneConfig(
            config=ChunkFusedReadoutFwdConfig(
                R_TILE=R_TILE,
                num_warp_groups=num_warp_groups,
                min_blocks_per_mp=min_blocks_per_mp,
            ),
        ))
    return configs


class ChunkFusedReadoutFwd:
    T_TILE = 64  # WGMMA output rows
    K_TILE = 32  # Reduction width of each W @ U tile
    M_PAD = 16  # WGMMA reduction width for the state term
    SMEM_ATOM_ELEMENTS = 64  # Contiguous elements in the 128-byte SMEM atom
    SMEM_ALIGNMENT_BYTES = 128
    PARAM_ALIGNMENT_BYTES = 16

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutFwdConfig,
    ):
        self.dtype = dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        assert dtype.width == 16, f"dtype must be 16-bit, got {dtype}"
        assert M in (8, 16), f"M must be 8 or 16, got {M}"
        assert chunk_size in (32, 64, 128), (
            f"chunk_size must be 32, 64, or 128, got {chunk_size}"
        )

        # A 32-row chunk stores only half of the 64-row WGMMA output tile.
        self.chunk_tile = min(self.T_TILE, chunk_size)
        self.R_TILE = config.R_TILE
        assert self.R_TILE in R_TILES
        self.num_k_stages = math.ceil(chunk_size / self.K_TILE)
        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.num_t_tiles = math.ceil(chunk_size / self.T_TILE)

        # Warp groups share the W/U buffers and split the R output columns.
        self.num_warp_groups = config.num_warp_groups
        assert self.num_warp_groups in WARP_GROUP_COUNTS
        self.mma_n = self.R_TILE // self.num_warp_groups
        assert self.supports_mma_n(self.mma_n)
        self.num_threads = cute.arch.THREADS_PER_WARPGROUP * self.num_warp_groups  # 128
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE
        assert 1 <= config.min_blocks_per_mp <= MAX_THREADS_PER_SM // self.num_threads

        # Scalar parameter loads are distributed across all CTA threads.
        self.num_c_tile_iters = self.T_TILE * self.M_PAD // self.num_threads
        self.num_param_load_iters = self.chunk_size * self.M_PAD // self.num_threads
        self.num_states_load_iters = self.R_TILE * self.M_PAD // self.num_threads
        # Each thread computes W for one K column and a strided set of T rows.
        self.w_rows_per_iter = self.num_threads // self.K_TILE
        self.num_w_row_iters = self.T_TILE // self.w_rows_per_iter

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,          # (B, L, R)
        mC: cute.Tensor,          # (B, L, M)
        mK: cute.Tensor,          # (B, L, M)
        mStatesAcc: cute.Tensor,  # (B, nchunks, R, M)
        mLCR: cute.Tensor,        # (B, L, M), fp32
        mH: cute.Tensor,          # (B, L, R)
        stream: cuda.CUstream,
    ):
        B, L = mU.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM allocations ##################
        intra_mma_tiler_mnk = (self.T_TILE, self.R_TILE, self.K_TILE)
        state_mma_tiler_mnk = (self.T_TILE, self.R_TILE, self.M_PAD)
        w_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, intra_mma_tiler_mnk, self.dtype, num_stages=self.num_k_stages
        )
        u_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR, intra_mma_tiler_mnk, self.dtype, num_stages=self.num_k_stages
        )
        u_smem_layout = cute.slice_(u_smem_layout_staged, (None, None, 0))
        c_scaled_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, state_mma_tiler_mnk, self.dtype, num_stages=1
        )
        states_acc_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR, state_mma_tiler_mnk, self.dtype, num_stages=1
        )
        h_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.T_TILE, self.R_TILE),
            epi_stage=1,
        )
        h_smem_layout = cute.slice_(h_smem_layout_staged, (None, None, 0))
        h_store_smem_layout = cute.composition(
            h_smem_layout,
            cute.make_layout((self.chunk_tile, self.R_TILE), stride=(1, self.T_TILE)),
        )
        c_smem_layout = cute.make_ordered_layout(
            (self.T_TILE, self.M_PAD), order=(1, 0)
        )
        chunk_param_smem_layout = cute.make_ordered_layout(
            (self.chunk_size, self.M_PAD), order=(1, 0)
        )

        @cute.struct
        class SharedStorage:
            u_tma_mbarriers: cute.struct.MemRange[cutlass.Int64, self.num_k_stages * 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(w_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sU: cute.struct.Align[
                # H reuses U's shared-memory buffer.
                # For 32-row chunks, the 64-row H tile requires more space than U.
                cute.struct.MemRange[
                    self.dtype,
                    max(cute.cosize(u_smem_layout_staged), cute.cosize(h_smem_layout_staged)),
                ],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sCScaled: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(c_scaled_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sStatesAcc: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(states_acc_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(c_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(chunk_param_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(chunk_param_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## Tensor views ##################
        mU_rlb = cute.make_tensor(
            mU.iterator,
            cute.make_layout(
                (self.R, L, B),
                stride=(mU.stride[2], mU.stride[1], mU.stride[0]),
            ),
        )
        mH_lrb = cute.make_tensor(
            mH.iterator,
            cute.make_layout(
                (L, self.R, B),
                stride=(mH.stride[1], mH.stride[2], mH.stride[0]),
            ),
        )

        ################## TMA setup ##################
        tma_atom_u, tma_tensor_u = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mU_rlb,
            u_smem_layout,
            (self.R_TILE, self.K_TILE),
        )
        tma_atom_h, tma_tensor_h = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mH_lrb,
            h_store_smem_layout,
            (self.chunk_tile, self.R_TILE),
        )

        ################## WGMMA ##################
        mma_op = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.mma_n, self.M_PAD),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_mma = cute.make_tiled_mma(mma_op, (1, self.num_warp_groups, 1))

        self.kernel(
            tma_tensor_u,
            tma_tensor_h,
            mC,
            mK,
            mStatesAcc,
            mLCR,
            tiled_mma,
            w_smem_layout_staged,
            u_smem_layout_staged,
            c_scaled_smem_layout_staged,
            states_acc_smem_layout_staged,
            h_smem_layout_staged,
            c_smem_layout,
            chunk_param_smem_layout,
            tma_atom_u,
            tma_atom_h,
        ).launch(
            grid=[self.num_t_tiles * self.num_r_tiles, nchunks, B],
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        mU_rlb: cute.Tensor,      # (R, L, B), TMA view
        mH_lrb: cute.Tensor,      # (L, R, B), TMA view
        mC: cute.Tensor,          # (B, L, M)
        mK: cute.Tensor,          # (B, L, M)
        mStatesAcc: cute.Tensor,  # (B, nchunks, R, M)
        mLCR: cute.Tensor,        # (B, L, M), fp32
        tiled_mma: cute.TiledMma,
        w_smem_layout_staged,           # (T_TILE, K_TILE, PIPE)
        u_smem_layout_staged,           # (R_TILE, K_TILE, PIPE)
        c_scaled_smem_layout_staged,    # (T_TILE, M_PAD, 1)
        states_acc_smem_layout_staged,  # (R_TILE, M_PAD, 1)
        h_smem_layout_staged,           # (T_TILE, R_TILE, 1)
        c_smem_layout,                  # (T_TILE, M_PAD)
        chunk_param_smem_layout,                  # (chunk_size, M_PAD)
        tma_atom_u: cute.CopyAtom,
        tma_atom_h: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_tile, bidx_chunk, bidx_batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        log2_e = math.log2(math.e)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_u)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h)

        bidx_t_tile = bidx_tile // self.num_r_tiles
        bidx_r_tile = bidx_tile % self.num_r_tiles
        chunk_start = bidx_chunk * self.chunk_size
        t_tile_start = bidx_t_tile * self.T_TILE
        L = mC.shape[1]
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
        num_k_tiles = cute.ceil_div(
            cutlass.min(t_tile_start + self.T_TILE, chunk_len), self.K_TILE
        )

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        u_load_mbar_ptr = storage.u_tma_mbarriers.data_ptr()
        sW = storage.sW.get_tensor(
            w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner
        )
        sU = storage.sU.get_tensor(
            u_smem_layout_staged.outer, swizzle=u_smem_layout_staged.inner
        )
        sCScaled = storage.sCScaled.get_tensor(
            c_scaled_smem_layout_staged.outer, swizzle=c_scaled_smem_layout_staged.inner
        )
        sStatesAcc = storage.sStatesAcc.get_tensor(
            states_acc_smem_layout_staged.outer, swizzle=states_acc_smem_layout_staged.inner
        )

        # sH reuses sU's storage because U is no longer needed in the epilogue.
        sH = cute.make_tensor(
            cute.recast_ptr(sU.iterator, h_smem_layout_staged.inner, self.dtype),
            h_smem_layout_staged.outer,
        )
        sC = storage.sC.get_tensor(c_smem_layout)
        sK = storage.sK.get_tensor(chunk_param_smem_layout)
        sLCR = storage.sLCR.get_tensor(chunk_param_smem_layout)
        sCScaled_stage = cute.slice_(sCScaled, (None, None, 0))
        sStatesAcc_stage = cute.slice_(sStatesAcc, (None, None, 0))

        ################## copy partitions ##################
        gU = cute.local_tile(
            mU_rlb[None, None, bidx_batch],
            (self.R_TILE, self.K_TILE),
            (bidx_r_tile, None),
        )
        bGS_sU, bGS_gU = cute.nvgpu.cpasync.tma_partition(
            tma_atom_u,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sU, 0, 2),
            gmem_tensor=cute.group_modes(gU, 0, 2),
        )
        u_tma_copy_bytes = cutlass.const_expr(
            cute.size_in_bytes(
                self.dtype,
                cute.slice_(u_smem_layout_staged, (None, None, 0)),
            )
        )
        gH = cute.local_tile(
            mH_lrb[None, None, bidx_batch],
            (self.chunk_tile, self.R_TILE),
            (None, bidx_r_tile),
        )
        bSG_sH, bSG_gH = cute.nvgpu.cpasync.tma_partition(
            tma_atom_h,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(
                cute.local_tile(sH, (self.chunk_tile, self.R_TILE, 1), (0, 0, 0)),
                0, 2,
            ),
            gmem_tensor=cute.group_modes(gH, 0, 2),
        )

        ################## pipelines ##################
        u_load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=u_load_mbar_ptr,
            num_stages=self.num_k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_warps),
            tx_count=u_tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        h_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads),
        )

        ################## prologue ##################
        # Load C for this output-row tile.
        for i in cutlass.range_constexpr(self.num_c_tile_iters):
            idx = tidx + i * self.num_threads
            local_t = idx // self.M_PAD
            local_m = idx % self.M_PAD
            global_t = chunk_start + t_tile_start + local_t
            if local_m < self.M and t_tile_start + local_t < chunk_len:
                sC[local_t, local_m] = mC[bidx_batch, global_t, local_m]
            else:
                sC[local_t, local_m] = self.dtype(0.0)

        # Load K and LCR for the whole chunk. LCR is needed at both t and k.
        for i in cutlass.range_constexpr(self.num_param_load_iters):
            idx = tidx + i * self.num_threads
            chunk_k = idx // self.M_PAD
            local_m = idx % self.M_PAD
            global_k = chunk_start + chunk_k
            if local_m < self.M and chunk_k < chunk_len:
                sK[chunk_k, local_m] = mK[bidx_batch, global_k, local_m]
                sLCR[chunk_k, local_m] = mLCR[bidx_batch, global_k, local_m]
            else:
                sK[chunk_k, local_m] = self.dtype(0.0)
                sLCR[chunk_k, local_m] = Float32(0.0)

        # Load accumulated states for this output-column tile.
        for i in cutlass.range_constexpr(self.num_states_load_iters):
            idx = tidx + i * self.num_threads
            local_r = idx // self.M_PAD
            local_m = idx % self.M_PAD
            r = bidx_r_tile * self.R_TILE + local_r
            if local_m < self.M and r < self.R:
                sStatesAcc_stage[local_r, local_m] = mStatesAcc[bidx_batch, bidx_chunk, r, local_m]
            else:
                sStatesAcc_stage[local_r, local_m] = self.dtype(0.0)

        cute.arch.sync_threads()

        # Compute C_scaled = exp(LCR) * C for the state term.
        for i in cutlass.range_constexpr(self.num_c_tile_iters):
            idx = tidx + i * self.num_threads
            local_t = idx // self.M_PAD
            local_m = idx % self.M_PAD
            if local_m < self.M and t_tile_start + local_t < chunk_len:
                scaled = cute.math.exp2(
                    sLCR[t_tile_start + local_t, local_m] * log2_e, fastmath=True
                ) * sC[local_t, local_m].to(Float32)
                sCScaled_stage[local_t, local_m] = scaled.to(self.dtype)
            else:
                sCScaled_stage[local_t, local_m] = self.dtype(0.0)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        ################## MMA fragments and pipeline states ##################
        thr_mma = tiled_mma.get_slice(tidx)
        acc = thr_mma.make_fragment_C(
            thr_mma.partition_shape_C((self.T_TILE, self.R_TILE))
        )
        tCrW = thr_mma.make_fragment_A(thr_mma.partition_A(sW))
        tCrU = thr_mma.make_fragment_B(thr_mma.partition_B(sU))
        tCrCScaled = thr_mma.make_fragment_A(thr_mma.partition_A(sCScaled))
        tCrStatesAcc = thr_mma.make_fragment_B(thr_mma.partition_B(sStatesAcc))

        u_load_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_k_stages
        )
        u_load_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_k_stages
        )

        ################## load U and compute the first W tile ##################
        # Start the U loads so they can overlap with computation of the first W tile.
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            if warp_idx == 0:
                u_load_pipeline.producer_acquire(u_load_producer_state)
                u_tile = (chunk_start + k_tile * self.K_TILE) // self.K_TILE
                cute.copy(
                    tma_atom_u,
                    bGS_gU[None, u_tile],
                    bGS_sU[None, u_load_producer_state.index],
                    tma_bar_ptr=u_load_pipeline.producer_get_barrier(u_load_producer_state),
                )
                u_load_pipeline.producer_commit(u_load_producer_state)
            u_load_producer_state.advance()

        self.compute_w_stage(sW, sC, sK, sLCR, 0, tidx, t_tile_start, chunk_len, log2_e)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        ################## mainloop ##################
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            u_load_pipeline.consumer_wait(u_load_consumer_state)
            cute.arch.fence_view_async_shared()

            cute.nvgpu.warpgroup.fence()
            self.gemm(
                tiled_mma,
                tCrW[None, None, None, k_tile],
                tCrU[None, None, None, u_load_consumer_state.index],
                acc,
                k_tile != 0,
            )
            cute.nvgpu.warpgroup.commit_group()

            # Compute the next W tile while WGMMA consumes the current tile.
            if k_tile + 1 < num_k_tiles:
                self.compute_w_stage(
                    sW, sC, sK, sLCR, k_tile + 1, tidx, t_tile_start, chunk_len, log2_e,
                )

            cute.nvgpu.warpgroup.wait_group(0)
            cute.arch.fence_view_async_shared()
            u_load_pipeline.consumer_release(u_load_consumer_state)
            u_load_consumer_state.advance()
            # Make the next W stage visible to all MMA threads.
            cute.arch.sync_threads()

        ################## state WGMMA ##################
        # Accumulate C_scaled @ states_acc.T into the same output fragment.
        cute.nvgpu.warpgroup.fence()
        self.gemm(
            tiled_mma, tCrCScaled[None, None, None, 0], tCrStatesAcc[None, None, None, 0],
            acc, True,
        )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        ################## epilogue ##################
        h_tile = (chunk_start + t_tile_start) // self.chunk_tile
        self.store_h_epilogue(
            tiled_mma, tma_atom_h, bSG_sH, bSG_gH, h_store_pipeline, sH, acc,
            h_tile, tidx, warp_idx,
        )

    @classmethod
    def supports_mma_n(cls, mma_n):
        return mma_n % cls.SMEM_ATOM_ELEMENTS == 0 or cls.SMEM_ATOM_ELEMENTS % mma_n == 0

    @cute.jit
    def store_h_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_h: cute.CopyAtom,
        bSG_sH: cute.Tensor,
        bSG_gH: cute.Tensor,
        h_store_pipeline,
        sH: cute.Tensor,   # (T_TILE, R_TILE, 1)
        acc: cute.Tensor,
        h_tile,
        tidx,
        warp_idx,
    ):
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                LayoutEnum.ROW_MAJOR.is_m_major_c(),
                num_matrices=4,
            ),
            self.dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C(copy_atom_C, tiled_mma)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sH = thr_copy_r2s.partition_D(sH)

        if warp_idx == 0:
            h_store_pipeline.producer_acquire()
        cute.arch.sync_threads()

        tRS_rAcc = tiled_copy_r2s.retile(acc)
        tRS_rH = cute.make_rmem_tensor(tRS_rAcc.shape, self.dtype)
        tRS_rH.store(tRS_rAcc.load().to(self.dtype))
        cute.copy(tiled_copy_r2s, tRS_rH, tRS_sH[None, None, None, 0])

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.copy(tma_atom_h, bSG_sH[None, 0], bSG_gH[None, h_tile])
            h_store_pipeline.producer_commit()
            h_store_pipeline.producer_tail()

    @cute.jit
    def compute_w_stage(
        self,
        sW: cute.Tensor,    # (T_TILE, K_TILE, PIPE)
        sC: cute.Tensor,    # (T_TILE, M_PAD)
        sK: cute.Tensor,    # (chunk_size, M_PAD)
        sLCR: cute.Tensor,  # (chunk_size, M_PAD), fp32
        k_tile,
        tidx,
        t_tile_start,
        chunk_len,
        log2_e,
    ):
        """Compute a causal W tile, reusing each column's K/LCR across T rows."""
        sW_stage = cute.slice_(sW, (None, None, k_tile))
        k_base = k_tile * self.K_TILE
        local_k = tidx % self.K_TILE
        local_t_base = tidx // self.K_TILE
        k = k_base + local_k

        rK = cute.make_rmem_tensor((self.M,), Float32)
        rLCR = cute.make_rmem_tensor((self.M,), Float32)
        for m in cutlass.range_constexpr(self.M):
            rK[m] = sK[k, m].to(Float32)
            rLCR[m] = sLCR[k, m]

        for i in cutlass.range_constexpr(self.num_w_row_iters):
            local_t = local_t_base + i * self.w_rows_per_iter
            t = t_tile_start + local_t
            w_val = Float32(0.0)
            # Check causality before exp because noncausal entries can overflow.
            if t < chunk_len and k <= t:
                for m in cutlass.range_constexpr(self.M):
                    w_val += (
                        sC[local_t, m].to(Float32)
                        * rK[m]
                        * cute.math.exp2(
                            (sLCR[t, m] - rLCR[m]) * log2_e,
                            fastmath=True,
                        )
                    )
            sW_stage[local_t, local_k] = w_val.to(self.dtype)

    @staticmethod
    @cute.jit
    def gemm(tiled_mma, A, B, C, accumulate_initial: Boolean):
        """Accumulate a reduction tile, optionally overwriting C on its first MMA."""
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
            mma_atom.set(
                cute.nvgpu.warpgroup.Field.ACCUMULATE,
                accumulate_initial or k_block_idx != 0,
            )
            cute.gemm(mma_atom, C, A[None, None, k_block_idx], B[None, None, k_block_idx], C)


@jit_cache
def _compile_chunk_fused_readout_fwd(dtype, R, M, chunk_size, config):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_u = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    fake_c = make_fake_tensor(dtype, (B_sym, L_sym, M), divisibility=1)
    fake_kappa = make_fake_tensor(dtype, (B_sym, L_sym, M), divisibility=1)
    fake_states_acc = make_fake_tensor(dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=1)
    fake_h = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutFwd(dtype, R=R, M=M, chunk_size=chunk_size, config=config),
        fake_u,
        fake_c,
        fake_kappa,
        fake_states_acc,
        fake_lcr,
        fake_h,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _prepare(u, c, kappa, states_acc, log_cum_rho, h, *, chunk_size, config):
    return _compile_chunk_fused_readout_fwd(
        torch2cute_dtype_map[u.dtype], u.shape[-1], c.shape[-1], chunk_size, config,
    )


@autotune(
    configs=cooperative_configs(),
    key=["chunk_size", ("u", 2), ("c", 2)],
    cache_results=False,
)
def _launch(u, c, kappa, states_acc, log_cum_rho, h, *, chunk_size, config):
    kernel = _prepare(
        u, c, kappa, states_acc, log_cum_rho, h, chunk_size=chunk_size, config=config,
    )
    kernel(u, c, kappa, states_acc, log_cum_rho, h)


_kernel_cache = {}
_launch_lock = threading.Lock()


def _frac_chunk_fused_readout_fwd(
    u, c, kappa, states_acc, log_cum_rho, chunk_size, config=None,
):
    B, L, R = u.shape
    M = c.shape[-1]
    h = torch.empty((B, L, R), device=u.device, dtype=u.dtype)

    if B == 0 or L == 0:
        return h
    assert u.dtype in (torch.float16, torch.bfloat16), f"u must be 16-bit, got {u.dtype}"
    assert c.dtype == u.dtype and kappa.dtype == u.dtype, "c and kappa dtype must match u dtype"
    assert states_acc.dtype == u.dtype, (
        f"states_acc dtype must match u dtype, got {states_acc.dtype}"
    )
    assert log_cum_rho.dtype == torch.float32, "log_cum_rho must be float32"

    if config is not None:
        kernel = _prepare(
            u, c, kappa, states_acc, log_cum_rho, h,
            chunk_size=chunk_size, config=config,
        )
    else:
        key = (u.device.index, u.dtype, R, M, chunk_size)
        kernel = _kernel_cache.get(key)
        if kernel is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up cooperative autotuning before CUDA graph capture")
            with _launch_lock:
                kernel = _kernel_cache.get(key)
                if kernel is None:
                    _launch(u, c, kappa, states_acc, log_cum_rho, h, chunk_size=chunk_size)
                    selected = _launch.best_config.kwargs["config"]
                    _kernel_cache[key] = _prepare(
                        u, c, kappa, states_acc, log_cum_rho, h,
                        chunk_size=chunk_size, config=selected,
                    )
                    return h
    kernel(u, c, kappa, states_acc, log_cum_rho, h)
    return h


__all__ = ["_frac_chunk_fused_readout_fwd"]
