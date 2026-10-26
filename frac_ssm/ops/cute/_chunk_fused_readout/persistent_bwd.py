"""Persistent SM90 backward for fused chunk readout.

Each CTA processes a sequence of chunks and computes all five gradients.
The first warp group computes D = dH @ U.T and dE = dH @ cast(states), accumulates
dStates, and consumes W to compute dU = W.T @ dH. The next warp loads parameters
and matrix operands with TMA. The remaining warps consume D and dE, reconstruct W,
and reduce dC, dK, and dLCR.

D/dE and W use separate shared memory pipelines. The dStates accumulators remain
in FP32 across time tiles. Parameter reductions also use FP32, with intra-chunk
dC rounded to the input dtype before adding the state contribution.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from itertools import product

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Float32, TFloat32
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import select_
from cutlass.tensor_utils import LayoutEnum

from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.utils import (
    MmaTF32Op, cvt_copy_r2s, make_fake_tensor, tf32_rmem_view, tf32_smem_view, torch2cute_dtype_map,
)


REGISTERS_PER_SM = 65536
REGISTER_GRANULARITY = 8
MAX_REGISTERS_PER_THREAD = 255
MIN_REGISTER_ESTIMATE = 64
DU_REGISTER_OVERHEAD = 32
PARAM_REGISTER_OVERHEAD = 16
TENSOR_REGISTER_OVERHEAD = 24
SMEM_ESTIMATE_MARGIN_BYTES = 2048


@dataclass(frozen=True)
class ChunkFusedReadoutBwdConfig:
    """Resource choices for each CTA. A zero max_ctas leaves the grid uncapped."""

    R_TILE: int = 64  # Output channels per dU tile.
    STATE_R_TILE: int = 64  # Reduction channels per D/dE tile and dStates output tile.
    num_producer_threads: int = 192
    min_blocks_per_mp: int = 2
    max_ctas: int = 0
    num_dde_stages: int = 2
    num_w_stages: int = 2
    num_threads_Q: int = 32  # Lanes assigned to input positions for each output row.


class ChunkFusedReadoutBwd:
    T_TILE = 64  # WGMMA output rows per instruction.
    T_SUBTILE = 32  # Time rows produced per D/dE pipeline stage.
    MMA_K = 16  # Reduction elements per 16-bit WGMMA instruction (D/dE).
    TF32_MMA_K = 8  # Reduction elements per TF32 WGMMA instruction (dU, dStates).
    # Chunk rows of dH widened to FP32 per dU WGMMA slice. A small slice keeps
    # 64-element chunks within the shared memory of two CTAs per SM.
    DU_SLICE_ROWS = 32
    M_PAD = 16  # Padded state channels in the matrix operands.
    SMEM_ALIGNMENT_BYTES = 128
    MMA_ALIGNMENT_BYTES = 1024  # One full 128-byte swizzle pattern for TF32 WGMMA operands
    PARAM_ALIGNMENT_BYTES = 16
    PARAM_PAIR_SIZE = 2  # A parameter value and its LCR value.
    PARAM_GRAD_PADDING = 1  # Extra FP32 element per dC/dE/dK row.
    PARAM_PAIR_PADDING = 2  # Extra pairs per row keep vector loads aligned.
    PRODUCER_BARRIER_ID = 1
    CONSUMER_BARRIER_ID = 2

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutBwdConfig,
        states_dtype: type[cutlass.Numeric],
        param_dtype: type[cutlass.Numeric] | None = None,
    ):
        self.dtype = dtype
        self.states_dtype = states_dtype
        self.param_dtype = dtype if param_dtype is None else param_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        self.DU_T_TILE = max(self.T_TILE, chunk_size)
        self.R_TILE = config.R_TILE
        self.STATE_R_TILE = config.STATE_R_TILE
        self.DDE_N_TILE = chunk_size + self.M_PAD
        self.num_t_tiles = chunk_size // self.T_SUBTILE
        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.num_state_r_tiles = math.ceil(R / self.STATE_R_TILE)

        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        self.num_producer_threads = config.num_producer_threads
        self.num_producer_warps = self.num_producer_threads // cute.arch.WARP_SIZE
        self.producer_load_warp_idx = self.num_consumer_warps
        self.first_producer_tidx = self.num_consumer_threads + cute.arch.WARP_SIZE
        self.num_threads = self.first_producer_tidx + self.num_producer_threads
        self.du_tma_warp_idx = 0

        self.producer_barrier = pipeline.NamedBarrier(
            self.PRODUCER_BARRIER_ID, self.num_producer_threads,
        )
        self.consumer_barrier = pipeline.NamedBarrier(
            self.CONSUMER_BARRIER_ID, self.num_consumer_threads,
        )

        self.vecsize = 128 // Float32.width
        self.param_pairs_per_load = self.vecsize // self.PARAM_PAIR_SIZE
        self.param_grad_stride = self.M + self.PARAM_GRAD_PADDING
        self.param_pair_stride = self.PARAM_PAIR_SIZE * (self.M + self.PARAM_PAIR_PADDING)
        self.param_grad_elements = self.chunk_size * self.param_grad_stride

        self.num_param_iters = math.ceil(chunk_size * M / self.num_producer_threads)
        self.num_de_copy_iters = math.ceil(self.T_SUBTILE * M / self.num_producer_threads)
        self.num_q_tiles = chunk_size // config.num_threads_Q
        self.num_threads_P = cute.arch.WARP_SIZE // config.num_threads_Q

        self.mma_k_blocks_per_t_tile = self.T_TILE // self.TF32_MMA_K
        self.DU_P_SLICE = min(chunk_size, self.DU_SLICE_ROWS)
        self.num_du_slices = chunk_size // self.DU_P_SLICE
        self.du_k_blocks_per_slice = self.DU_P_SLICE // self.TF32_MMA_K
        self.num_state_k_blocks = self.T_SUBTILE // self.TF32_MMA_K

    @cute.jit
    def __call__(
        self, mdH: cute.Tensor, mU: cute.Tensor, mC: cute.Tensor, mK: cute.Tensor,
        mStatesAcc: cute.Tensor, mLCR: cute.Tensor, mdU: cute.Tensor, mdC: cute.Tensor,
        mdK: cute.Tensor, mdStates: cute.Tensor, mdLCR: cute.Tensor,
        max_ctas: cutlass.Int32, stream: cuda.CUstream,
    ):
        B, L, _ = mdH.shape
        nchunks = cute.ceil_div(L, self.chunk_size)
        # Reuse precomputed division parameters when decoding each batch and chunk.
        chunk_divisor = cute.fast_divmod_create_divisor_v2(nchunks)

        ################## SMEM layouts ##################
        self.make_shared_storage()

        ################## TMA setup ##################
        tma_atom_dh_du, tma_tensor_dh_du = self.make_tma_load(
            mdH, self.dh_du_smem_layout, (self.R_TILE, self.chunk_size), (2, 1, 0),
        )
        mdU_lrb = cute.make_tensor(mdU.iterator, cute.select(mdU.layout, mode=(1, 2, 0)))
        tma_atom_du, tma_tensor_du = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mdU_lrb, self.du_store_smem_layout,
            (self.chunk_size, self.R_TILE),
        )
        param_tile = (self.chunk_size, self.M)
        tma_atom_c, tma_tensor_c = self.make_tma_load(
            mC, self.param_raw_smem_layout, param_tile, (1, 2, 0),
        )
        tma_atom_k, tma_tensor_k = self.make_tma_load(
            mK, self.param_raw_smem_layout, param_tile, (1, 2, 0),
        )
        tma_atom_lcr, tma_tensor_lcr = self.make_tma_load(
            mLCR, self.param_raw_smem_layout, param_tile, (1, 2, 0),
        )
        tma_atom_dh, tma_tensor_dh = self.make_tma_load(
            mdH, self.dh_load_smem_layout, (self.T_SUBTILE, self.STATE_R_TILE), (1, 2, 0),
        )
        tma_atom_u, tma_tensor_u = self.make_tma_load(
            mU, self.u_smem_layout, (self.chunk_size, self.STATE_R_TILE), (1, 2, 0),
        )
        tma_atom_states, tma_tensor_states = self.make_tma_load(
            mStatesAcc, self.states_raw_smem_layout, (self.STATE_R_TILE, self.M), (2, 3, 1, 0),
        )
        ################## persistent launch ##################
        work_count = B * nchunks
        grid = work_count
        if cutlass.const_expr(self.config.max_ctas > 0):
            grid = cutlass.min(work_count, max_ctas)
        self.kernel(
            mdC, mdK, mdStates, mdLCR, L, work_count, chunk_divisor,
            tma_tensor_dh_du, tma_tensor_du, tma_tensor_c, tma_tensor_k, tma_tensor_lcr,
            tma_tensor_dh, tma_tensor_u, tma_tensor_states,
            tma_atom_dh_du, tma_atom_du, tma_atom_c, tma_atom_k, tma_atom_lcr,
            tma_atom_dh, tma_atom_u, tma_atom_states,
        ).launch(
            grid=(grid, 1, 1), block=(self.num_threads, 1, 1), stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self, mdC: cute.Tensor, mdK: cute.Tensor, mdStates: cute.Tensor, mdLCR: cute.Tensor,
        L: cutlass.Int32, work_count: cutlass.Int32,
        chunk_divisor: cute.FastDivmodDivisorV2,
        mdH_rlb: cute.Tensor, mdU_lrb: cute.Tensor,
        mC_lmb: cute.Tensor, mK_lmb: cute.Tensor, mLCR_lmb: cute.Tensor,
        mdH_lrb: cute.Tensor, mU_lrb: cute.Tensor, mStatesAcc_rmnb: cute.Tensor,
        tma_atom_dh_du: cute.CopyAtom, tma_atom_du: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom, tma_atom_k: cute.CopyAtom, tma_atom_lcr: cute.CopyAtom,
        tma_atom_dh: cute.CopyAtom, tma_atom_u: cute.CopyAtom, tma_atom_states: cute.CopyAtom,
    ):
        ################## prologue ##################
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        ################## SMEM allocations ##################
        # Build the layouts in the kernel's IR region as well as the launch region.
        self.make_shared_storage()
        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sW = storage.sW.get_tensor(
            self.w_smem_layout_staged.outer, swizzle=self.w_smem_layout_staged.inner,
        )
        sdHDU = storage.sdHDUBuffer.get_tensor(
            self.dh_du_smem_layout.outer, swizzle=self.dh_du_smem_layout.inner,
        )
        sdHDUMma = storage.sdHDUMma.get_tensor(
            self.dh_du_mma_smem_layout.outer, swizzle=self.dh_du_mma_smem_layout.inner,
        )
        # dH and dU share storage. Complete the dU store before loading the next dH tile.
        sdU = cute.make_tensor(
            cute.recast_ptr(sdHDU.iterator, self.du_smem_layout.inner, self.dtype),
            self.du_smem_layout.outer,
        )
        sdUStore = cute.make_tensor(sdU.iterator, self.du_store_smem_layout.outer)
        sCPair = storage.sCPair.get_tensor(self.param_pair_smem_layout)
        sKPair = storage.sKPair.get_tensor(self.param_pair_smem_layout)
        sDDE = storage.sDDE.get_tensor(self.dde_smem_layout_staged)
        sdParams = storage.sdParams.get_tensor(
            cute.make_layout(2 * self.param_grad_elements),
        )
        sdC = cute.make_tensor(sdParams.iterator, self.param_smem_layout)
        # Keep dE until the parameter epilogue, after its D/dE stage has been released.
        sdE = cute.make_tensor(
            sdParams.iterator + self.param_grad_elements, self.param_smem_layout,
        )
        sdK = storage.sdK.get_tensor(self.param_smem_layout)
        sCRaw = storage.sCRaw.get_tensor(self.param_raw_smem_layout)
        sKRaw = storage.sKRaw.get_tensor(self.param_raw_smem_layout)
        sLCRRaw = storage.sLCRRaw.get_tensor(self.param_raw_smem_layout)
        sStatesRaw = storage.sStatesRaw.get_tensor(self.states_raw_smem_layout)
        sdH = storage.sdH.get_tensor(
            self.dh_smem_layout.outer, swizzle=self.dh_smem_layout.inner,
        )
        sdHLoad = cute.make_tensor(sdH.iterator, self.dh_load_smem_layout.outer)
        sU = storage.sU.get_tensor(self.u_smem_layout.outer, swizzle=self.u_smem_layout.inner)
        sCScaled = storage.sCScaled.get_tensor(
            self.c_scaled_smem_layout.outer, swizzle=self.c_scaled_smem_layout.inner,
        )
        ################## pipelines ##################
        w_pipeline, dh_du_pipeline, param_pipeline, dde_pipeline, operand_pipeline = (
            self.make_pipelines(storage)
        )
        du_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
        )
        ################## initialize padding and barriers ##################
        self.initialize_padding(sdH, sW, tidx)
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()
        ################## TMA producer ##################
        if warp_idx == self.producer_load_warp_idx:
            param_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            operand_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            work_idx = block_idx
            while work_idx < work_count:
                bidx_batch, bidx_chunk = divmod(work_idx, chunk_divisor)
                chunk_start = bidx_chunk * self.chunk_size

                ################## load parameters ##################
                param_pipeline.producer_acquire(param_producer_state)
                barrier_ptr = param_pipeline.producer_get_barrier(param_producer_state)
                param_tile = (self.chunk_size, self.M)
                self.load_tile(
                    tma_atom_c, mC_lmb[None, None, bidx_batch], sCRaw,
                    (chunk_start, 0), param_tile, barrier_ptr,
                )
                self.load_tile(
                    tma_atom_k, mK_lmb[None, None, bidx_batch], sKRaw,
                    (chunk_start, 0), param_tile, barrier_ptr,
                )
                self.load_tile(
                    tma_atom_lcr, mLCR_lmb[None, None, bidx_batch], sLCRRaw,
                    (chunk_start, 0), param_tile, barrier_ptr,
                )
                param_pipeline.producer_commit(param_producer_state)
                param_producer_state.advance()

                ################## load D/dE and dStates operands ##################
                for t_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                    for r_tile_idx in cutlass.range_constexpr(self.num_state_r_tiles):
                        operand_pipeline.producer_acquire(operand_producer_state)
                        barrier_ptr = operand_pipeline.producer_get_barrier(operand_producer_state)
                        self.load_tile(
                            tma_atom_dh, mdH_lrb[None, None, bidx_batch], sdHLoad,
                            (chunk_start + t_tile_idx * self.T_SUBTILE, r_tile_idx * self.STATE_R_TILE),
                            (self.T_SUBTILE, self.STATE_R_TILE), barrier_ptr,
                        )
                        self.load_tile(
                            tma_atom_u, mU_lrb[None, None, bidx_batch], sU,
                            (chunk_start, r_tile_idx * self.STATE_R_TILE),
                            (self.chunk_size, self.STATE_R_TILE), barrier_ptr,
                        )
                        self.load_tile(
                            tma_atom_states, mStatesAcc_rmnb[None, None, bidx_chunk, bidx_batch],
                            sStatesRaw,
                            (r_tile_idx * self.STATE_R_TILE, 0), (self.STATE_R_TILE, self.M), barrier_ptr,
                        )
                        operand_pipeline.producer_commit(operand_producer_state)
                        operand_producer_state.advance()
                work_idx += num_ctas

            ################## drain TMA loads ##################
            param_pipeline.producer_tail(param_producer_state)
            operand_pipeline.producer_tail(operand_producer_state)

        ################## W and parameter gradients ##################
        elif warp_idx > self.producer_load_warp_idx:
            producer_tidx = tidx - self.first_producer_tidx
            w_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.config.num_w_stages,
            )
            param_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            dde_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.config.num_dde_stages,
            )
            work_idx = block_idx
            while work_idx < work_count:
                bidx_batch, bidx_chunk = divmod(work_idx, chunk_divisor)
                chunk_start = bidx_chunk * self.chunk_size
                chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

                ################## stage parameter pairs ##################
                w_pipeline.producer_acquire(w_producer_state)
                param_pipeline.consumer_wait(param_consumer_state)
                self.stage_parameter_pairs(sCRaw, sKRaw, sLCRRaw, sCPair, sKPair, producer_tidx)
                # The W producers have finished reading raw parameters. Tensor
                # consumers still hold them until the final time tile.
                self.producer_barrier.arrive_and_wait()
                param_pipeline.consumer_release(param_consumer_state)
                param_consumer_state.advance()
                rdK = cute.make_rmem_tensor(
                    (self.num_q_tiles, self.M), Float32,
                )
                rdK.fill(0.0)

                ################## recompute W and reduce parameter partials ##################
                for t_tile_idx in cutlass.range_constexpr(self.num_t_tiles):
                    dde_pipeline.consumer_wait(dde_consumer_state)
                    self.compute_w_and_param_partials(
                        sW[None, None, w_producer_state.index], sCPair, sKPair,
                        sDDE[None, None, dde_consumer_state.index], sdC,
                        chunk_len, producer_tidx, rdK, t_tile_idx * self.T_SUBTILE,
                    )
                    # Preserve dE for the epilogue before the tensor consumers reuse this stage.
                    for i in cutlass.range_constexpr(self.num_de_copy_iters):
                        idx = producer_tidx + i * self.num_producer_threads
                        if idx < self.T_SUBTILE * self.M:
                            t, m = idx // self.M, idx % self.M
                            sdE[t_tile_idx * self.T_SUBTILE + t, m] = sDDE[
                                t, self.chunk_size + m, dde_consumer_state.index
                            ]
                    dde_pipeline.consumer_release(dde_consumer_state)
                    dde_consumer_state.advance()

                ################## publish W and store parameter gradients ##################
                self.store_dk(rdK, sdK, producer_tidx)
                cute.arch.fence_view_async_shared()
                w_pipeline.producer_commit(w_producer_state)
                w_producer_state.advance()
                self.producer_barrier.arrive_and_wait()
                self.store_parameter_grads(
                    sCPair, sKPair, sdE, sdC, sdK,
                    mdC[bidx_batch, None, None], mdK[bidx_batch, None, None],
                    mdLCR[bidx_batch, None, None],
                    chunk_start, chunk_len, producer_tidx,
                )
                self.producer_barrier.arrive_and_wait()
                work_idx += num_ctas

            ################## drain W pipeline ##################
            w_pipeline.producer_tail(w_producer_state)

        else:
            ################## tensor consumer prologue ##################
            tiled_mma_du = self.make_du_mma()
            tiled_mma_dstates = self.make_dstates_mma()
            w_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.config.num_w_stages,
            )
            dh_du_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            dh_du_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            dde_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.config.num_dde_stages,
            )
            operand_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            param_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            ################## MMA fragments and output copy ##################
            # dStates remains in FP32 registers across the chunk's time tiles.
            thr_mma_dstates = tiled_mma_dstates.get_slice(tidx)
            dstates_acc_shape = thr_mma_dstates.partition_shape_C((self.STATE_R_TILE, self.M_PAD))
            tCrdStates = cute.make_rmem_tensor((dstates_acc_shape, self.num_state_r_tiles), Float32)
            thr_mma_du = tiled_mma_du.get_slice(tidx)
            tCrW = thr_mma_du.make_fragment_A(thr_mma_du.partition_A(
                tf32_smem_view(storage.sW, self.w_smem_layout_staged)))
            tCrdHDU = thr_mma_du.make_fragment_B(thr_mma_du.partition_B(
                tf32_smem_view(storage.sdHDUMma, self.dh_du_mma_smem_layout)))
            r2s_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    LayoutEnum.ROW_MAJOR.is_m_major_c(), num_matrices=4,
                ), self.dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_C(r2s_atom, tiled_mma_du)
            tRS_sdU = tiled_copy_r2s.get_slice(tidx).partition_D(sdU)
            # Carry the updated states out of each conditional compute_tensor_tile call.
            tensor_pipeline_states = (
                dde_producer_state, operand_consumer_state, param_consumer_state,
            )

            ################## persistent mainloop ##################
            work_idx = block_idx
            while work_idx < work_count:
                bidx_batch, bidx_chunk = divmod(work_idx, chunk_divisor)
                chunk_start = bidx_chunk * self.chunk_size
                chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
                ################## D/dE and dStates tiles ##################
                # Compute the first tile here only on the first chunk. Later chunks
                # inherit it from the previous iteration. Produce just one tile of
                # the next chunk before dU so a single W stage cannot stall progress.
                for t_tile_idx in cutlass.range_constexpr(self.num_t_tiles + 1):
                    if cutlass.const_expr(t_tile_idx == 0):
                        if work_idx == block_idx:
                            tensor_pipeline_states = self.compute_tensor_tile(
                                mdStates[bidx_batch, bidx_chunk, None, None],
                                sDDE, sCRaw, sLCRRaw, sStatesRaw,
                                sdH, sU, sCScaled, tCrdStates,
                                operand_pipeline, dde_pipeline, param_pipeline,
                                tensor_pipeline_states, chunk_len, 0,
                            )
                    elif cutlass.const_expr(t_tile_idx == self.num_t_tiles):
                        next_work_idx = work_idx + num_ctas
                        if next_work_idx < work_count:
                            next_bidx_batch, next_bidx_chunk = divmod(next_work_idx, chunk_divisor)
                            next_chunk_start = next_bidx_chunk * self.chunk_size
                            next_chunk_len = cutlass.min(self.chunk_size, L - next_chunk_start)
                            tensor_pipeline_states = self.compute_tensor_tile(
                                mdStates[next_bidx_batch, next_bidx_chunk, None, None],
                                sDDE, sCRaw, sLCRRaw, sStatesRaw,
                                sdH, sU, sCScaled, tCrdStates,
                                operand_pipeline, dde_pipeline, param_pipeline,
                                tensor_pipeline_states, next_chunk_len, 0,
                            )
                    else:
                        tensor_pipeline_states = self.compute_tensor_tile(
                            mdStates[bidx_batch, bidx_chunk, None, None],
                            sDDE, sCRaw, sLCRRaw, sStatesRaw,
                            sdH, sU, sCScaled, tCrdStates,
                            operand_pipeline, dde_pipeline, param_pipeline,
                            tensor_pipeline_states, chunk_len, t_tile_idx,
                        )
                ################## dU tiles ##################
                for r_tile_idx in cutlass.range_constexpr(self.num_r_tiles):
                    if warp_idx == self.du_tma_warp_idx:
                        dh_du_pipeline.producer_acquire(dh_du_producer_state)
                        self.load_tile(
                            tma_atom_dh_du, mdH_rlb[None, None, bidx_batch], sdHDU,
                            (r_tile_idx * self.R_TILE, chunk_start), (self.R_TILE, self.chunk_size),
                            dh_du_pipeline.producer_get_barrier(dh_du_producer_state),
                        )
                        dh_du_pipeline.producer_commit(dh_du_producer_state)
                    dh_du_producer_state.advance()
                    if cutlass.const_expr(r_tile_idx == 0):
                        w_pipeline.consumer_wait(w_consumer_state)
                    dh_du_pipeline.consumer_wait(dh_du_consumer_state)
                    tCrdU = self.compute_du(
                        tiled_mma_du, thr_mma_du, tCrW, tCrdHDU, w_consumer_state.index,
                        sdHDU, sdHDUMma, tidx,
                    )
                    dh_du_pipeline.consumer_release(dh_du_consumer_state)
                    dh_du_consumer_state.advance()
                    self.store_du(
                        tma_atom_du, mdU_lrb, sdUStore, tiled_copy_r2s, tRS_sdU, tCrdU,
                        du_store_pipeline, bidx_batch, chunk_start, r_tile_idx, warp_idx,
                    )

                # Every channel tile must finish before a producer can overwrite W.
                w_pipeline.consumer_release(w_consumer_state)
                w_consumer_state.advance()
                work_idx += num_ctas

            ################## drain ##################
            dde_producer_state, _, _ = tensor_pipeline_states
            dde_pipeline.producer_tail(dde_producer_state)
            if warp_idx == self.du_tma_warp_idx:
                dh_du_pipeline.producer_tail(dh_du_producer_state)

    @cute.jit
    def make_tma_load(self, mX: cute.Tensor, smem_layout, tile_shape, modes: cutlass.Constexpr):
        mView = cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=modes))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mView, smem_layout, tile_shape,
        )

    @cute.jit
    def load_tile(
        self, tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, sX: cute.Tensor,
        offset, tile_shape, barrier_ptr,
    ):
        gX = cute.local_tile(cute.domain_offset(offset, tma_tensor), tile_shape, (0, 0))
        bGS_sX, bGS_gX = cpasync.tma_partition(
            tma_atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX, 0, 2),
            gmem_tensor=cute.group_modes(gX, 0, 2),
        )
        cute.copy(tma_atom, bGS_gX, bGS_sX, tma_bar_ptr=barrier_ptr)

    @cute.jit
    def make_shared_storage(self):
        """Static layouts are shared by TMA construction and kernel allocation."""
        du_mma_tiler = (self.DU_T_TILE, self.R_TILE, self.chunk_size)
        self.w_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, du_mma_tiler, TFloat32, num_stages=self.config.num_w_stages,
        )
        dh_du_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR, du_mma_tiler, self.dtype, num_stages=1,
        )
        self.dh_du_smem_layout = cute.slice_(dh_du_smem_layout_staged, (None, None, 0))
        # One chunk slice of the dU tile of dH, widened to FP32.
        self.dh_du_mma_smem_layout = cute.slice_(sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (self.DU_T_TILE, self.R_TILE, self.DU_P_SLICE), TFloat32, num_stages=1,
        ), (None, None, 0))
        self.du_smem_layout = cute.slice_(sm90_utils.make_smem_layout_epi(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.DU_T_TILE, self.R_TILE), epi_stage=1,
        ), (None, None, 0))
        self.du_store_smem_layout = cute.composition(
            self.du_smem_layout,
            cute.make_layout((self.chunk_size, self.R_TILE), stride=(1, self.DU_T_TILE)),
        )
        self.param_raw_smem_layout = cute.make_layout(
            (self.chunk_size, self.M), stride=(self.M, 1),
        )
        # Row padding avoids bank conflicts. Paired parameters also retain the
        # alignment needed to load two adjacent value/LCR pairs at once.
        self.param_smem_layout = cute.make_layout(
            (self.chunk_size, self.M), stride=(self.param_grad_stride, 1),
        )
        self.param_pair_smem_layout = cute.make_layout(
            (self.chunk_size, self.M, self.PARAM_PAIR_SIZE),
            stride=(self.param_pair_stride, self.PARAM_PAIR_SIZE, 1),
        )
        self.dde_smem_layout_staged = cute.make_layout(
            (self.T_SUBTILE, self.DDE_N_TILE, self.config.num_dde_stages),
            stride=(self.DDE_N_TILE, 1, self.T_SUBTILE * self.DDE_N_TILE),
        )
        d_mma_tiler = (self.T_TILE, self.chunk_size, self.STATE_R_TILE)
        self.dh_smem_layout = cute.slice_(sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, d_mma_tiler, self.dtype, num_stages=1,
        ), (None, None, 0))
        self.u_smem_layout = cute.slice_(sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, d_mma_tiler, self.dtype, num_stages=1,
        ), (None, None, 0))
        self.c_scaled_smem_layout = cute.slice_(sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (self.STATE_R_TILE, self.M_PAD, self.T_TILE), TFloat32, num_stages=1,
        ), (None, None, 0))
        self.dh_load_smem_layout = cute.composition(
            self.dh_smem_layout,
            cute.make_layout((self.T_SUBTILE, self.STATE_R_TILE), stride=(1, self.T_TILE)),
        )
        self.states_raw_smem_layout = cute.make_layout((self.STATE_R_TILE, self.M), stride=(self.M, 1))

        @cute.struct
        class SharedStorage:
            # Each pipeline stage has a full barrier and an empty barrier.
            w_barriers: cute.struct.MemRange[cutlass.Int64, self.config.num_w_stages * 2]
            dh_du_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            param_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            dde_barriers: cute.struct.MemRange[cutlass.Int64, self.config.num_dde_stages * 2]
            operand_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sCRaw: cute.struct.Align[
                cute.struct.MemRange[self.param_dtype, self.chunk_size * self.M],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sKRaw: cute.struct.Align[
                cute.struct.MemRange[self.param_dtype, self.chunk_size * self.M],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sLCRRaw: cute.struct.Align[
                cute.struct.MemRange[Float32, self.chunk_size * self.M],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sStatesRaw: cute.struct.Align[
                cute.struct.MemRange[self.states_dtype, self.STATE_R_TILE * self.M],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sdH: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.dh_smem_layout)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.u_smem_layout)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sCScaled: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.c_scaled_smem_layout)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.w_smem_layout_staged)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sdHDUMma: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.dh_du_mma_smem_layout)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sdHDUBuffer: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype,
                    max(cute.cosize(dh_du_smem_layout_staged), cute.cosize(self.du_smem_layout)),
                ],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sCPair: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.param_pair_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sKPair: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.param_pair_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sDDE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.dde_smem_layout_staged)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sdParams: cute.struct.Align[
                cute.struct.MemRange[Float32, 2 * self.param_grad_elements],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sdK: cute.struct.Align[
                cute.struct.MemRange[Float32, self.param_grad_elements],
                self.PARAM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

    @cute.jit
    def make_du_mma(self):
        return cute.make_tiled_mma(MmaTF32Op(
            instruction_shape=(self.T_TILE, self.R_TILE, self.TF32_MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))

    @cute.jit
    def make_dstates_mma(self):
        return cute.make_tiled_mma(MmaTF32Op(
            instruction_shape=(self.T_TILE, self.M_PAD, self.TF32_MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.RMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))

    @cute.jit
    def make_pipelines(self, storage):
        # PipelineAsync counts participating threads. PipelineTmaAsync counts
        # consumer warps and lets TMA completion signal that a stage is full.
        # W passes from the scalar producers to the tensor consumers. D/dE flows
        # in the opposite direction. Both roles read the raw parameters.
        w_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.w_barriers.data_ptr(), num_stages=self.config.num_w_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
            defer_sync=True,
        )
        dh_du_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dh_du_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=self.R_TILE * self.chunk_size * self.dtype.width // 8, defer_sync=True,
        )
        param_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.param_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_warps + self.num_consumer_warps,
            ),
            tx_count=self.chunk_size * self.M * (2 * self.param_dtype.width + Float32.width) // 8,
            defer_sync=True,
        )
        dde_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.dde_barriers.data_ptr(), num_stages=self.config.num_dde_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads,
            ),
            defer_sync=True,
        )
        operand_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.operand_barriers.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=((self.T_SUBTILE + self.chunk_size) * self.STATE_R_TILE * self.dtype.width // 8
                      + self.STATE_R_TILE * self.M * self.states_dtype.width // 8),
            defer_sync=True,
        )
        return w_pipeline, dh_du_pipeline, param_pipeline, dde_pipeline, operand_pipeline

    @cute.jit
    def initialize_padding(self, sdH, sW, tidx):
        # TMA fills T_SUBTILE rows of dH. Zero the remaining WGMMA rows once.
        for i in cutlass.range_constexpr(
            cute.ceil_div(self.T_SUBTILE * self.STATE_R_TILE, self.num_threads),
        ):
            idx = tidx + i * self.num_threads
            if idx < self.T_SUBTILE * self.STATE_R_TILE:
                sdH[self.T_SUBTILE + idx // self.STATE_R_TILE, idx % self.STATE_R_TILE] = self.dtype(0.0)
        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.DU_T_TILE > self.chunk_size):
            # A 32-element chunk also needs padded W rows for the dU WGMMA.
            padded_rows = cutlass.const_expr(self.DU_T_TILE - self.chunk_size)
            padded_elements_per_stage = cutlass.const_expr(padded_rows * self.chunk_size)
            num_padded_elements = cutlass.const_expr(
                padded_elements_per_stage * self.config.num_w_stages,
            )
            for i in cutlass.range_constexpr(cute.ceil_div(num_padded_elements, self.num_threads)):
                pad_idx = tidx + i * self.num_threads
                if pad_idx < num_padded_elements:
                    pad_q = self.chunk_size + pad_idx % padded_rows
                    pad_p = pad_idx // padded_rows % self.chunk_size
                    pad_slot = pad_idx // padded_elements_per_stage
                    sW[pad_q, pad_p, pad_slot] = Float32(0.0)
            cute.arch.fence_view_async_shared()

    @cute.jit
    def compute_tensor_tile(
        self, mdStates, sDDE, sCRaw, sLCRRaw, sStatesRaw,
        sdH, sU, sCScaled, tCrdStates,
        operand_pipeline, dde_pipeline, param_pipeline, pipeline_states,
        chunk_len, t_tile_idx: cutlass.Constexpr,
    ):
        ################## MMA fragments ##################
        tidx, _, _ = cute.arch.thread_idx()
        dde_producer_state, operand_consumer_state, param_consumer_state = pipeline_states
        sdHTransposed = cute.make_tensor(
            sdH.iterator, cute.select(self.dh_smem_layout.outer, mode=[1, 0]),
        )
        # Create MMA objects within this call so their accumulator flags stay
        # in the same IR region as the conditional tile computation.
        tiled_mma_d = cute.make_tiled_mma(cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype, acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.chunk_size, self.MMA_K),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))
        # dE covers only T_SUBTILE x M_PAD, so a warp-level TF32 MMA computes it with
        # its operands in a few registers.
        tiled_mma_de = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaTF32Op((16, 8, self.TF32_MMA_K)), (2, 2, 1),
        )
        tiled_mma_dstates = self.make_dstates_mma()
        thr_mma_d = tiled_mma_d.get_slice(tidx)
        tCrdH = thr_mma_d.make_fragment_A(thr_mma_d.partition_A(sdH))
        tCrU = thr_mma_d.make_fragment_B(thr_mma_d.partition_B(sU))
        tCcD = thr_mma_d.partition_C(cute.make_identity_tensor((self.T_TILE, self.chunk_size)))
        thr_mma_de = tiled_mma_de.get_slice(tidx)
        tCsdHDE = thr_mma_de.partition_A(
            cute.local_tile(sdH, (self.T_SUBTILE, self.STATE_R_TILE), (0, 0)),
        )
        # The raw (STATE_R_TILE, M) states tile, viewed as the (M_PAD, STATE_R_TILE) B operand.
        tCsStates = thr_mma_de.partition_B(cute.make_tensor(
            sStatesRaw.iterator, cute.make_layout((self.M_PAD, self.STATE_R_TILE), stride=(1, self.M)),
        ))
        tCcStates = thr_mma_de.partition_B(cute.make_identity_tensor((self.M_PAD, self.STATE_R_TILE)))
        tCcdE = thr_mma_de.partition_C(cute.make_identity_tensor((self.T_SUBTILE, self.M_PAD)))
        thr_mma_dstates = tiled_mma_dstates.get_slice(tidx)
        tCsdHTransposed = thr_mma_dstates.partition_A(sdHTransposed)
        # Compact, so the register order is the MMA fragment order rather than the smem order.
        tCrdHTransposed = cute.make_rmem_tensor(tCsdHTransposed.shape, Float32)
        sCScaledMma = cute.make_tensor(
            cute.recast_ptr(sCScaled.iterator, self.c_scaled_smem_layout.inner, TFloat32),
            self.c_scaled_smem_layout.outer,
        )
        tCrCScaled = thr_mma_dstates.make_fragment_B(thr_mma_dstates.partition_B(sCScaledMma))
        tCcdStates = thr_mma_dstates.partition_C(
            cute.make_identity_tensor((self.STATE_R_TILE, self.M_PAD)),
        )

        ################## scale C ##################
        if cutlass.const_expr(t_tile_idx == 0):
            param_pipeline.consumer_wait(param_consumer_state)
        self.scale_c(sCRaw, sLCRRaw, sCScaled, t_tile_idx, chunk_len, tidx)
        tCrD = thr_mma_d.make_fragment_C(thr_mma_d.partition_shape_C((self.T_TILE, self.chunk_size)))
        tCrD.fill(0.0)
        tCrdE = thr_mma_de.make_fragment_C(thr_mma_de.partition_shape_C((self.T_SUBTILE, self.M_PAD)))
        tCrdE.fill(0.0)
        for r_tile_idx in cutlass.range_constexpr(self.num_state_r_tiles):
            ################## prepare state operands ##################
            operand_pipeline.consumer_wait(operand_consumer_state)
            cute.arch.fence_view_async_shared()
            self.consumer_barrier.arrive_and_wait()

            ################## dE MMA ##################
            for k_block_idx in cutlass.range_constexpr(cute.size(tCsdHDE, mode=[2])):
                rdH = cute.make_rmem_tensor(tCsdHDE[None, None, k_block_idx].shape, Float32)
                for i in cutlass.range_constexpr(cute.size(rdH)):
                    rdH[i] = tCsdHDE[None, None, k_block_idx][i].to(Float32)
                rStates = cute.make_rmem_tensor(tCsStates[None, None, k_block_idx].shape, Float32)
                for i in cutlass.range_constexpr(cute.size(rStates)):
                    if cutlass.const_expr(self.R % self.STATE_R_TILE == 0 and self.M == self.M_PAD):
                        rStates[i] = tCsStates[None, None, k_block_idx][i].to(Float32)
                    else:
                        m, r = tCcStates[None, None, k_block_idx][i]
                        rStates[i] = Float32(0.0)
                        if r_tile_idx * self.STATE_R_TILE + r < self.R and m < self.M:
                            rStates[i] = tCsStates[None, None, k_block_idx][i].to(Float32)
                cute.gemm(tiled_mma_de, tCrdE, tf32_rmem_view(rdH), tf32_rmem_view(rStates), tCrdE)

            # dH.T is the register A operand of the TF32 dStates WGMMA.
            for k_block_idx in cutlass.range_constexpr(self.num_state_k_blocks):
                src = tCsdHTransposed[None, None, k_block_idx]
                dst = tCrdHTransposed[None, None, k_block_idx]
                for i in cutlass.range_constexpr(cute.size(src)):
                    dst[i] = src[i].to(Float32)

            ################## D WGMMA ##################
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(cute.size(tCrdH, mode=[2])):
                tiled_mma_d.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, r_tile_idx != 0 or k_block_idx != 0,
                )
                cute.gemm(
                    tiled_mma_d, tCrD,
                    tCrdH[None, None, k_block_idx], tCrU[None, None, k_block_idx], tCrD,
                )
            ################## dStates WGMMA ##################
            # Continue the time reduction in order without rounding between tiles.
            tCrdStatesTile = cute.make_tensor(
                tCrdStates[None, r_tile_idx].iterator,
                cute.make_layout(thr_mma_dstates.partition_shape_C((self.STATE_R_TILE, self.M_PAD))),
            )
            # Each time tile accumulates only its T_SUBTILE rows.
            for k_block_idx in cutlass.range_constexpr(self.num_state_k_blocks):
                tiled_mma_dstates.set(
                    cute.nvgpu.warpgroup.Field.ACCUMULATE, t_tile_idx != 0 or k_block_idx != 0,
                )
                cute.gemm(
                    tiled_mma_dstates, tCrdStatesTile,
                    tf32_rmem_view(tCrdHTransposed)[None, None, k_block_idx],
                    tCrCScaled[None, None, k_block_idx], tCrdStatesTile,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            ################## store completed dStates ##################
            if cutlass.const_expr(t_tile_idx == self.num_t_tiles - 1):
                for i in cutlass.range_constexpr(cute.size(tCrdStatesTile)):
                    r, m = tCcdStates[i]
                    if r_tile_idx * self.STATE_R_TILE + r < self.R and m < self.M:
                        mdStates[r_tile_idx * self.STATE_R_TILE + r, m] = tCrdStatesTile[i].to(
                            mdStates.element_type,
                        )
            operand_pipeline.consumer_release(operand_consumer_state)
            operand_consumer_state.advance()

        ################## publish D/dE ##################
        dde_pipeline.producer_acquire(dde_producer_state)
        for i in cutlass.range_constexpr(cute.size(tCrD)):
            p, q = tCcD[i]
            if p < self.T_SUBTILE:
                # D = dH @ U.T is dW, which Triton stores in the input dtype.
                sDDE[p, q, dde_producer_state.index] = tCrD[i].to(self.dtype).to(Float32)
        for i in cutlass.range_constexpr(cute.size(tCrdE)):
            p, m = tCcdE[i]
            if m < self.M:
                sDDE[p, self.chunk_size + m, dde_producer_state.index] = tCrdE[i]
        cute.arch.fence_view_async_shared()
        dde_pipeline.producer_commit(dde_producer_state)
        dde_producer_state.advance()
        if cutlass.const_expr(t_tile_idx == self.num_t_tiles - 1):
            param_pipeline.consumer_release(param_consumer_state)
            param_consumer_state.advance()
        return dde_producer_state, operand_consumer_state, param_consumer_state

    @cute.jit
    def scale_c(self, sCRaw, sLCRRaw, sCScaled, t_tile_idx: cutlass.Constexpr, chunk_len, tidx):
        for i in cutlass.range_constexpr(
            cute.ceil_div(self.T_TILE * self.M_PAD, self.num_consumer_threads),
        ):
            idx = tidx + i * self.num_consumer_threads
            p, m = idx // self.M_PAD, idx % self.M_PAD
            value = Float32(0.0)
            if p < self.T_SUBTILE and t_tile_idx * self.T_SUBTILE + p < chunk_len and m < self.M:
                value = sCRaw[t_tile_idx * self.T_SUBTILE + p, m].to(Float32) * cute.math.exp2(
                    sLCRRaw[t_tile_idx * self.T_SUBTILE + p, m] * math.log2(math.e), fastmath=True)
            sCScaled[m, p] = value

    @cute.jit
    def widen_dh_du(self, sdHDU, sdHDUMma, p_slice: cutlass.Constexpr, tidx):
        """Copy one chunk slice of the dU tile of dH from its 16-bit TMA tile into the FP32 K-major WGMMA tile."""
        vecsize = 8
        for i in cutlass.range_constexpr(
            self.R_TILE * self.DU_P_SLICE // (vecsize * self.num_consumer_threads),
        ):
            idx = tidx + i * self.num_consumer_threads
            t = idx % self.DU_P_SLICE
            vec = idx // self.DU_P_SLICE
            rdH = cute.make_rmem_tensor((vecsize,), self.dtype)
            cute.autovec_copy(
                cute.local_tile(sdHDU[None, p_slice * self.DU_P_SLICE + t], (vecsize,), (vec,)), rdH,
            )
            for j in cutlass.range_constexpr(vecsize):
                sdHDUMma[vec * vecsize + j, t] = rdH[j].to(Float32)

    @cute.jit
    def compute_du(self, tiled_mma_du, thr_mma, tCrW, tCrdH, w_slot, sdHDU, sdHDUMma, tidx):
        tCrdU = thr_mma.make_fragment_C(thr_mma.partition_shape_C((self.DU_T_TILE, self.R_TILE)))
        # dH is widened one chunk slice at a time. W.T is upper triangular, so skip
        # reduction blocks before each output tile and accumulate the remaining
        # blocks in ascending time order.
        for p_slice in cutlass.range_constexpr(self.num_du_slices):
            # The previous slice's WGMMAs have retired, so its widened tile can be overwritten.
            self.widen_dh_du(sdHDU, sdHDUMma, p_slice, tidx)
            cute.arch.fence_view_async_shared()
            self.consumer_barrier.arrive_and_wait()
            cute.nvgpu.warpgroup.fence()
            for t_block_idx in cutlass.range_constexpr(self.DU_T_TILE // self.T_TILE):
                tCrdUTile = cute.local_tile(
                    tCrdU, (tCrdU.shape[0], 1, tCrdU.shape[2]), (0, t_block_idx, 0),
                )
                first_k_block = cutlass.const_expr(t_block_idx * self.mma_k_blocks_per_t_tile)
                for j in cutlass.range_constexpr(self.du_k_blocks_per_slice):
                    k_block_idx = cutlass.const_expr(p_slice * self.du_k_blocks_per_slice + j)
                    if cutlass.const_expr(k_block_idx >= first_k_block):
                        tCrWTile = cute.local_tile(
                            tCrW[None, None, k_block_idx, w_slot],
                            (tCrW.shape[0], 1), (0, t_block_idx),
                        )
                        tiled_mma_du.set(
                            cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != first_k_block,
                        )
                        cute.gemm(
                            tiled_mma_du, tCrdUTile, tCrWTile,
                            tCrdH[None, None, j], tCrdUTile,
                        )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
        return tCrdU

    @cute.jit
    def store_du(
        self, tma_atom_du, mdU_lrb, sdUStore, tiled_copy_r2s, tRS_sdU, tCrdU,
        du_store_pipeline,
        bidx_batch, chunk_start, r_tile_idx, warp_idx,
    ):
        ################## convert dU to shared memory ##################
        cvt_copy_r2s(tiled_copy_r2s, tCrdU, tRS_sdU)
        cute.arch.fence_view_async_shared()
        self.consumer_barrier.arrive_and_wait()

        ################## TMA store dU ##################
        if warp_idx == self.du_tma_warp_idx:
            gdU = cute.local_tile(
                cute.domain_offset((chunk_start, r_tile_idx * self.R_TILE), mdU_lrb[None, None, bidx_batch]),
                (self.chunk_size, self.R_TILE), (0, 0),
            )
            bSG_sdU, bSG_gdU = cpasync.tma_partition(
                tma_atom_du,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdUStore, 0, 2),
                gmem_tensor=cute.group_modes(gdU, 0, 2),
            )
            du_store_pipeline.producer_acquire()
            cute.copy(tma_atom_du, bSG_sdU, bSG_gdU)
            du_store_pipeline.producer_commit()
            du_store_pipeline.producer_tail()
        self.consumer_barrier.arrive_and_wait()

    @cute.jit
    def stage_parameter_pairs(self, sCRaw, sKRaw, sLCRRaw, sCPair, sKPair, producer_tidx):
        for i in cutlass.range_constexpr(self.num_param_iters):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.chunk_size * self.M:
                t, m = idx // self.M, idx % self.M
                sCPair[t, m, 0] = sCRaw[t, m].to(Float32)
                sKPair[t, m, 0] = sKRaw[t, m].to(Float32)
                sCPair[t, m, 1] = sLCRRaw[t, m]
                sKPair[t, m, 1] = sLCRRaw[t, m]

    @cute.jit
    def compute_w_and_param_partials(
        self, sW, sCPair, sKPair, sDDE, sdC, chunk_len, producer_tidx, rdK, t_tile_start,
    ):
        # p and q are output and input positions within the chunk. Each lane
        # keeps dK[q, m] partials across its warp's p rows. W, dC, and dK share
        # the same decay. Reduce dC across q lanes and dK across producer warps.
        lane_idx = cute.arch.lane_idx()
        producer_warp_idx = producer_tidx // cute.arch.WARP_SIZE
        q_lane = lane_idx % self.config.num_threads_Q
        p_lane = lane_idx // self.config.num_threads_Q
        for p_iter in cutlass.range(
            cute.ceil_div(self.T_SUBTILE, self.num_producer_warps * self.num_threads_P), unroll=1,
        ):
            p = (
                t_tile_start
                + (p_iter * self.num_producer_warps + producer_warp_idx) * self.num_threads_P
                + p_lane
            )
            rdC = cute.make_rmem_tensor((self.M,), Float32)
            rdC.fill(0.0)
            for q_tile_idx in cutlass.range_constexpr(self.num_q_tiles):
                q = q_lane + q_tile_idx * self.config.num_threads_Q
                rW = cute.make_rmem_tensor((self.vecsize,), Float32)
                rW.fill(0.0)
                if p < cutlass.min(chunk_len, t_tile_start + self.T_SUBTILE) and q <= p:
                    d_value = sDDE[p - t_tile_start, q]
                    for pair in cutlass.range_constexpr(self.M // self.param_pairs_per_load):
                        rCPairs = self.load_param_pairs(sCPair, p, pair * self.param_pairs_per_load)
                        rKPairs = self.load_param_pairs(sKPair, q, pair * self.param_pairs_per_load)
                        for offset in cutlass.range_constexpr(self.param_pairs_per_load):
                            m = cutlass.const_expr(pair * self.param_pairs_per_load + offset)
                            pair_offset = cutlass.const_expr(self.PARAM_PAIR_SIZE * offset)
                            c_p, lcr_p = rCPairs[pair_offset], rCPairs[pair_offset + 1]
                            k_q, lcr_q = rKPairs[pair_offset], rKPairs[pair_offset + 1]
                            decay = cute.math.exp2(
                                (lcr_p - lcr_q) * math.log2(math.e), fastmath=True,
                            )
                            rW[m % self.vecsize] += c_p * k_q * decay
                            weighted_d = d_value * decay
                            rdC[m] += weighted_d * k_q
                            rdK[q_tile_idx, m] += weighted_d * c_p
                if p < t_tile_start + self.T_SUBTILE:
                    w_value = rW[0]
                    for i in cutlass.range_constexpr(1, self.vecsize):
                        w_value += rW[i]
                    sW[q, p] = w_value
            self.reduce_dc(rdC, sdC, p, cutlass.min(chunk_len, t_tile_start + self.T_SUBTILE), q_lane)

    @cute.jit
    def reduce_dc(self, rdC, sdC, p, chunk_len, q_lane):
        # Each lane has accumulated its q tiles into rdC. Reduce across the
        # row's q lanes and distribute the M output channels among them.
        for stage in cutlass.range_constexpr(int(math.log2(self.M))):
            num_values = cutlass.const_expr(self.M >> (stage + 1))
            offset = cutlass.const_expr((self.config.num_threads_Q // 2) >> stage)
            is_upper_lane = (q_lane & offset) != 0
            for m in cutlass.range_constexpr(num_values):
                value_local = Float32(select_(is_upper_lane, rdC[m + num_values], rdC[m]))
                value_shuffle = Float32(select_(is_upper_lane, rdC[m], rdC[m + num_values]))
                rdC[m] = value_local + cute.arch.shuffle_sync_bfly(value_shuffle, offset=offset)
        num_threads_per_M = cutlass.const_expr(self.config.num_threads_Q // self.M)
        for stage in cutlass.range_constexpr(int(math.log2(num_threads_per_M))):
            offset = cutlass.const_expr(
                (self.config.num_threads_Q // 2) >> (int(math.log2(self.M)) + stage),
            )
            rdC[0] += cute.arch.shuffle_sync_bfly(rdC[0], offset=offset)
        if q_lane % num_threads_per_M == 0 and p < chunk_len:
            sdC[p, q_lane // num_threads_per_M] = rdC[0]

    @cute.jit
    def store_dk(self, rdK, sdK, producer_tidx):
        # Sum the producer warps' dK partials into one buffer, one warp at a time in
        # increasing warp order. This keeps the summation order fixed.
        lane_idx = cute.arch.lane_idx()
        p_lane = lane_idx // self.config.num_threads_Q
        producer_warp_idx = producer_tidx // cute.arch.WARP_SIZE
        # Reduce across each warp's p lanes in all warps at once. Only the
        # cross-warp adds below are serialized, so their loads can be batched.
        for q_tile_idx in cutlass.range_constexpr(self.num_q_tiles):
            for m in cutlass.range_constexpr(self.M):
                for stage in cutlass.range_constexpr(int(math.log2(self.num_threads_P))):
                    rdK[q_tile_idx, m] += cute.arch.shuffle_sync_bfly(
                        rdK[q_tile_idx, m], offset=(cute.arch.WARP_SIZE // 2) >> stage,
                    )
        for warp_idx in cutlass.range(self.num_producer_warps, unroll=1):
            if producer_warp_idx == warp_idx and p_lane == 0:
                for q_tile_idx in cutlass.range_constexpr(self.num_q_tiles):
                    q = lane_idx % self.config.num_threads_Q + q_tile_idx * self.config.num_threads_Q
                    for m in cutlass.range_constexpr(self.M):
                        if warp_idx == 0:
                            sdK[q, m] = rdK[q_tile_idx, m]
                        else:
                            sdK[q, m] = sdK[q, m] + rdK[q_tile_idx, m]
            self.producer_barrier.arrive_and_wait()

    @cute.jit
    def load_param_pairs(self, sTensor, p, m):
        # Read two adjacent (value, LCR) pairs in one aligned 128-bit load.
        sParams = cute.make_tensor(
            sTensor.iterator + cute.crd2idx((p, m, 0), sTensor.layout), cute.make_layout(self.vecsize),
        )
        rParams = cute.make_rmem_tensor((self.vecsize,), Float32)
        s2r_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32,
            num_bits_per_copy=self.vecsize * Float32.width,
        )
        cute.copy(s2r_atom, sParams, rParams)
        return rParams

    @cute.jit
    def load_param_pair(self, sTensor, p, m):
        sParams = cute.make_tensor(
            sTensor.iterator + cute.crd2idx((p, m, 0), sTensor.layout),
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
    def store_parameter_grads(
        self, sCPair, sKPair, sdE, sdC, sdK, mdC, mdK, mdLCR,
        chunk_start, chunk_len, producer_tidx,
    ):
        for i in cutlass.range_constexpr(self.num_param_iters):
            idx = producer_tidx + i * self.num_producer_threads
            if idx < self.chunk_size * self.M:
                p, m = idx // self.M, idx % self.M
                if p < chunk_len:
                    c_p, lcr_p = self.load_param_pair(sCPair, p, m)
                    k_p, _ = self.load_param_pair(sKPair, p, m)
                    dc_intra = sdC[p, m]
                    dk = sdK[p, m]
                    decay = cute.math.exp2(lcr_p * math.log2(math.e), fastmath=True)
                    de = sdE[p, m]
                    dlcr_intra = c_p * dc_intra - k_p * dk
                    # The intra-chunk and state dC terms each take C's dtype before they
                    # are added. dLCR uses the FP32 terms.
                    mdC[chunk_start + p, m] = (
                        dc_intra.to(mdC.element_type).to(Float32)
                        + (de * decay).to(mdC.element_type).to(Float32)
                    ).to(mdC.element_type)
                    mdK[chunk_start + p, m] = dk.to(mdK.element_type)
                    mdLCR[chunk_start + p, m] = dlcr_intra + de * c_p * decay


@jit_cache
def _compile_chunk_fused_readout_bwd(dtype, R, M, chunk_size, config, states_dtype, param_dtype):
    B, L, nchunks = cute.sym_int(), cute.sym_int(), cute.sym_int()
    fake_dh = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_u = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_c = make_fake_tensor(param_dtype, (B, L, M), divisibility=1)
    fake_kappa = make_fake_tensor(param_dtype, (B, L, M), divisibility=1)
    fake_states_acc = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    fake_du = make_fake_tensor(dtype, (B, L, R), divisibility=1)
    fake_dc = make_fake_tensor(param_dtype, (B, L, M), divisibility=1)
    fake_dk = make_fake_tensor(param_dtype, (B, L, M), divisibility=1)
    fake_dstates = make_fake_tensor(states_dtype, (B, nchunks, R, M), divisibility=1)
    fake_dlcr = make_fake_tensor(Float32, (B, L, M), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutBwd(
            dtype, R=R, M=M, chunk_size=chunk_size, config=config, states_dtype=states_dtype,
            param_dtype=param_dtype,
        ),
        fake_dh, fake_u, fake_c, fake_kappa, fake_states_acc, fake_lcr,
        fake_du, fake_dc, fake_dk, fake_dstates, fake_dlcr, cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


def _resource_config(config):
    # Positive CTA caps share one binary and pass the cap at launch.
    # Zero keeps a separate specialization with one CTA per chunk.
    return replace(config, max_ctas=int(config.max_ctas > 0))


def persistent_configs():
    # Pruning converts ctas_per_sm to an absolute CTA cap before compilation.
    return [
        AutotuneConfig(
            config=ChunkFusedReadoutBwdConfig(
                R_TILE=R_TILE,
                num_producer_threads=num_producer_threads,
                min_blocks_per_mp=min_blocks_per_mp,
                num_threads_Q=num_threads_Q,
                num_w_stages=num_w_stages,
                num_dde_stages=num_dde_stages,
            ),
            ctas_per_sm=ctas_per_sm,
        )
        for (
            R_TILE, num_producer_threads, min_blocks_per_mp, num_threads_Q, num_w_stages, num_dde_stages,
        ) in product(
            (64, 32), (96, 128, 192, 224), (1, 2, 3), (16, 32), (1, 2), (1, 2),
        )
        # Match the residency hint as well so each SM can receive a full wave.
        for ctas_per_sm in sorted({1, 2, 4, min_blocks_per_mp})
    ]


def prune_persistent_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    dh, c, states_acc, chunk_size = (
        args[name] for name in ("dh", "c", "states_acc", "chunk_size")
    )
    props = torch.cuda.get_device_properties(dh.device)
    B, L, R = dh.shape
    M = c.shape[-1]
    work_count = B * math.ceil(L / chunk_size)
    DU_T_TILE = max(ChunkFusedReadoutBwd.T_TILE, chunk_size)
    DDE_N_TILE = chunk_size + ChunkFusedReadoutBwd.M_PAD
    num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
    dtype_bytes = dh.element_size()
    param_dtype_bytes = c.element_size()
    acc_dtype_bytes = Float32.width // 8
    param_grad_elements = chunk_size * (M + ChunkFusedReadoutBwd.PARAM_GRAD_PADDING)
    candidates, seen = [], set()

    for candidate in configs:
        config = candidate.kwargs["config"]
        # Narrower dU tiles are only needed to fit the FP32 operands of 128-element chunks.
        if config.R_TILE < 64 and chunk_size < 128:
            continue
        num_threads = num_consumer_threads + cute.arch.WARP_SIZE + config.num_producer_threads
        registers_per_thread = REGISTERS_PER_SM // (num_threads * config.min_blocks_per_mp)
        register_limit = min(
            MAX_REGISTERS_PER_THREAD,
            registers_per_thread // REGISTER_GRANULARITY * REGISTER_GRANULARITY,
        )
        padded_r = math.ceil(R / config.STATE_R_TILE) * config.STATE_R_TILE
        du_acc_registers = DU_T_TILE * config.R_TILE // num_consumer_threads
        dc_dk_acc_registers = (chunk_size // config.num_threads_Q + 1) * M
        dde_acc_registers = ChunkFusedReadoutBwd.T_TILE * DDE_N_TILE // num_consumer_threads
        dstates_acc_registers = padded_r * ChunkFusedReadoutBwd.M_PAD // num_consumer_threads
        # Add estimates for operand descriptors, indices, and pipeline state.
        required_registers = max(
            MIN_REGISTER_ESTIMATE,
            du_acc_registers + DU_REGISTER_OVERHEAD,
            dc_dk_acc_registers + PARAM_REGISTER_OVERHEAD,
            dde_acc_registers + dstates_acc_registers + TENSOR_REGISTER_OVERHEAD,
        )
        # Keep single-CTA residency available for large R and let timing include spills.
        if config.min_blocks_per_mp > 1 and register_limit < required_registers:
            continue

        w_bytes = acc_dtype_bytes * DU_T_TILE * chunk_size * config.num_w_stages
        dh_du_bytes = (
            dtype_bytes * DU_T_TILE * config.R_TILE
            + acc_dtype_bytes * min(chunk_size, ChunkFusedReadoutBwd.DU_SLICE_ROWS) * config.R_TILE
        )
        raw_param_bytes = chunk_size * M * (2 * param_dtype_bytes + acc_dtype_bytes)
        raw_state_bytes = config.STATE_R_TILE * M * states_acc.element_size()
        operand_bytes = (
            dtype_bytes * (ChunkFusedReadoutBwd.T_TILE + chunk_size) * config.STATE_R_TILE
            + acc_dtype_bytes * ChunkFusedReadoutBwd.T_TILE * ChunkFusedReadoutBwd.M_PAD
        )
        param_pair_bytes = (
            2 * acc_dtype_bytes * chunk_size * ChunkFusedReadoutBwd.PARAM_PAIR_SIZE
            * (M + ChunkFusedReadoutBwd.PARAM_PAIR_PADDING)
        )
        dde_bytes = (
            acc_dtype_bytes * ChunkFusedReadoutBwd.T_SUBTILE * DDE_N_TILE * config.num_dde_stages
        )
        param_grad_bytes = 2 * acc_dtype_bytes * param_grad_elements
        dk_partial_bytes = acc_dtype_bytes * param_grad_elements
        shared_bytes = (
            w_bytes + dh_du_bytes + raw_param_bytes + raw_state_bytes + operand_bytes
            + param_pair_bytes + dde_bytes + param_grad_bytes + dk_partial_bytes
            + SMEM_ESTIMATE_MARGIN_BYTES
        )
        if shared_bytes > props.shared_memory_per_block_optin:
            continue
        if config.min_blocks_per_mp * shared_bytes > props.shared_memory_per_multiprocessor:
            continue

        # Keep the resource cap reusable when B or L changes. The launch clamps it.
        max_ctas = candidate.kwargs["ctas_per_sm"] * props.multi_processor_count
        resolved_config = replace(config, max_ctas=max_ctas)
        launch_config = replace(resolved_config, max_ctas=min(work_count, max_ctas))
        if launch_config not in seen:
            seen.add(launch_config)
            candidates.append(AutotuneConfig(config=resolved_config))
    return candidates


def _prepare(
    dh, u, c, kappa, states_acc, log_cum_rho, du, dc, dk, dstates, dlcr, *, chunk_size, config,
):
    return _compile_chunk_fused_readout_bwd(
        torch2cute_dtype_map[u.dtype], u.shape[-1], c.shape[-1],
        chunk_size, _resource_config(config), torch2cute_dtype_map[states_acc.dtype],
        torch2cute_dtype_map[c.dtype],
    )


@autotune(
    configs=persistent_configs(), key=["chunk_size", ("u", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": prune_persistent_configs}, cache_results=False,
)
def _launch(
    dh, u, c, kappa, states_acc, log_cum_rho, du, dc, dk, dstates, dlcr, *, chunk_size, config,
):
    args = (dh, u, c, kappa, states_acc, log_cum_rho, du, dc, dk, dstates, dlcr)
    kernel = _prepare(*args, chunk_size=chunk_size, config=config)
    kernel(*args, config.max_ctas)


_kernel_cache = {}
_launch_lock = threading.Lock()


def _get_autotuned_kernel(key, args, chunk_size):
    launch = _kernel_cache.get(key)
    if launch is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up persistent backward autotuning before CUDA graph capture")
        with _launch_lock:
            launch = _kernel_cache.get(key)
            if launch is None:
                _launch(*args, chunk_size=chunk_size)
                config = _launch.best_config.kwargs["config"]
                kernel = _prepare(*args, chunk_size=chunk_size, config=config)
                # Cache only code and the scalar cap, never invocation tensors.
                launch = (kernel, config.max_ctas)
                _kernel_cache[key] = launch
    return launch


def _frac_chunk_fused_readout_bwd(
    d_h, u, c, kappa, states_acc, log_cum_rho, chunk_size,
    config: ChunkFusedReadoutBwdConfig | None = None,
):
    """Compute all five gradients with an explicit config or an autotuned choice."""
    assert u.dtype in (torch.float16, torch.bfloat16), "Readout backward requires FP16/BF16 inputs"
    assert d_h.dtype == u.dtype, "dH must have the input dtype"
    assert c.dtype == kappa.dtype and c.dtype in (u.dtype, torch.float32), (
        "C and kappa must share the input dtype or float32"
    )
    assert states_acc.dtype in (u.dtype, torch.float32), (
        "states_acc dtype must match inputs or be float32"
    )
    assert log_cum_rho.dtype == torch.float32, "log_cum_rho must be float32"
    assert chunk_size in (32, 64, 128) and c.shape[-1] in (8, 16), "Unsupported chunk size or M"
    B, L, R = u.shape
    outputs = tuple(torch.empty_like(x) for x in (u, c, kappa, states_acc, log_cum_rho))
    if B == 0 or L == 0:
        return outputs
    args = (d_h, u, c, kappa, states_acc, log_cum_rho, *outputs)
    if config is None:
        key = (u.device.index, u.dtype, states_acc.dtype, c.dtype, R, c.shape[-1], chunk_size)
        kernel, max_ctas = _get_autotuned_kernel(key, args, chunk_size)
    else:
        kernel = _prepare(*args, chunk_size=chunk_size, config=config)
        max_ctas = config.max_ctas
    kernel(*args, max_ctas)
    return outputs


__all__ = ["_frac_chunk_fused_readout_bwd"]
