"""Warp-specialized SM90 forward for fused chunk readout.

Each CTA computes a (T_TILE, R_TILE) output tile. Producer warps build W
while one consumer warp group computes H in 64-column subtiles, reusing W.

    W[t, k] = sum_m C[t, m] * K[k, m] * exp(LCR[t, m] - LCR[k, m]), k <= t
    H = W @ U + (C * exp(LCR)) @ states_acc.T

W is initialized to zero before producers write its causal entries. TMA
loads U and the parameters. State tiles are loaded and converted ahead of
their WGMMA use, and each output subtile is stored with TMA.
"""

import math
import threading
from dataclasses import dataclass
from itertools import product

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Boolean, Float32
from cutlass.tensor_utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.utils import cvt_copy_r2s, make_fake_tensor, torch2cute_dtype_map


R_TILES = (64, 128, 192, 256)
PRODUCER_THREAD_COUNTS = (64, 128, 192, 256)
REGISTERS_PER_SM = 65536
REGISTER_GRANULARITY = 8
ESTIMATED_REGISTER_OVERHEAD = 32


@dataclass(frozen=True)
class ChunkFusedReadoutFwdConfig:
    R_TILE: int = 128
    num_producer_threads: int = 128
    min_blocks_per_mp: int = 4


def warp_specialized_configs():
    # Tune the output width and producer resources for one consumer warp group.
    configs = []
    for R_TILE, num_producer_threads, min_blocks_per_mp in product(
        R_TILES, PRODUCER_THREAD_COUNTS, (2, 3, 4, 5),
    ):
        num_threads = cute.arch.THREADS_PER_WARPGROUP + num_producer_threads  # 128
        # Round the register budget per thread down to a multiple of eight.
        register_budget = (
            REGISTERS_PER_SM // (num_threads * min_blocks_per_mp)
            // REGISTER_GRANULARITY * REGISTER_GRANULARITY
        )
        # Allow 32 accumulator registers and an estimated 32 for other work.
        accumulator_registers = (
            ChunkFusedReadoutFwd.T_TILE * ChunkFusedReadoutFwd.R_SUBTILE
            // cute.arch.THREADS_PER_WARPGROUP  # 128
        )
        if register_budget < accumulator_registers + ESTIMATED_REGISTER_OVERHEAD:
            continue
        configs.append(AutotuneConfig(
            config=ChunkFusedReadoutFwdConfig(
                R_TILE=R_TILE,
                num_producer_threads=num_producer_threads,
                min_blocks_per_mp=min_blocks_per_mp,
            ),
        ))
    return configs


def prune_warp_specialized_configs(configs, named_args, **kwargs):
    # Avoid wholly empty subtiles in a tile wider than the channel dimension.
    padded_r = ChunkFusedReadoutFwd.R_SUBTILE * math.ceil(
        named_args["u"].shape[-1] / ChunkFusedReadoutFwd.R_SUBTILE
    )
    return [c for c in configs if c.kwargs["config"].R_TILE <= padded_r]


class ChunkFusedReadoutFwd:
    T_TILE = 64  # WGMMA output rows
    K_TILE = 32  # Reduction width of each W @ U tile
    R_SUBTILE = 64  # Output columns computed by the consumer warp group at once
    M_PAD = 16  # WGMMA reduction width for the state term
    SMEM_ALIGNMENT_BYTES = 128
    PARAM_SWIZZLE_SHIFT = 3
    CONSUMER_BARRIER_ID = 1

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutFwdConfig,
        states_dtype: type[cutlass.Numeric],
    ):
        self.dtype = dtype
        self.states_dtype = states_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size
        self.config = config

        assert dtype.width == 16, f"dtype must be 16-bit, got {dtype}"
        assert states_dtype in (dtype, Float32), (
            f"states_dtype must match dtype or be float32, got {states_dtype}"
        )
        assert M in (8, 16), f"M must be 8 or 16, got {M}"
        assert chunk_size in (32, 64, 128), (
            f"chunk_size must be 32, 64, or 128, got {chunk_size}"
        )

        # Reuse W across 64-column subtiles and pad the final channel tile.
        self.R_TILE = config.R_TILE
        assert self.R_TILE in R_TILES
        self.num_r_subtiles = self.R_TILE // self.R_SUBTILE
        self.chunk_tile = min(self.T_TILE, chunk_size)
        self.raw_c_bytes = self.chunk_tile * self.M_PAD * dtype.width // 8
        self.raw_k_bytes = self.chunk_size * self.M_PAD * dtype.width // 8
        self.raw_lcr_bytes = self.chunk_size * self.M_PAD * Float32.width // 8
        self.raw_param_bytes = self.raw_c_bytes + self.raw_k_bytes + self.raw_lcr_bytes
        self.num_k_stages = math.ceil(self.chunk_size / self.K_TILE)
        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.num_t_tiles = math.ceil(self.chunk_size / self.T_TILE)
        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_producer_threads = config.num_producer_threads
        assert self.num_producer_threads in PRODUCER_THREAD_COUNTS
        self.num_threads = self.num_consumer_threads + self.num_producer_threads
        # Consumers occupy the first warp group, followed by producer warps.
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        # The first producer warp loads U and parameters. Consumer warp 1
        # loads states, and consumer warp 0 stores H.
        self.producer_load_warp_idx = self.num_consumer_warps
        self.state_load_warp_idx = 1
        self.h_store_warp_idx = 0

        self.num_c_scale_iters = self.T_TILE * self.M_PAD // self.num_consumer_threads
        self.num_lcr_copy_iters = math.ceil(self.chunk_size * self.M_PAD / self.num_threads)
        self.k_copy_vecsize = cutlass.Uint32.width // dtype.width
        self.num_k_copy_iters = math.ceil(
            self.chunk_size * self.M_PAD / (self.k_copy_vecsize * self.num_threads)
        )
        self.vecsize = 128 // Float32.width
        self.log_vecsize = int(math.log2(self.vecsize))
        self.input_vecsize = 128 // dtype.width
        # Each thread handles mirrored K columns and strides over T rows.
        self.w_k_columns_per_thread = 2
        self.num_w_threads_T = 4
        self.num_w_threads_K = self.K_TILE // self.w_k_columns_per_thread
        self.w_rows_per_iter = self.num_producer_threads // self.num_w_threads_K
        self.num_w_row_iters = math.ceil(self.chunk_tile / self.w_rows_per_iter)

        # Use the same state copy layout for input and FP32 state dtypes.
        self.states_vecsize = 4
        self.num_states_threads_M = self.M_PAD // self.states_vecsize
        self.num_states_threads_R = self.num_consumer_threads // self.num_states_threads_M

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
        # State tiles keep M contiguous for TMA loads.
        states_acc_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (self.T_TILE, self.R_SUBTILE, self.M_PAD), self.dtype,
            num_stages=1,
        )
        states_raw_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (self.T_TILE, self.R_SUBTILE, self.M_PAD), self.states_dtype,
            num_stages=1,
        )
        h_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.T_TILE, self.R_SUBTILE),
            epi_stage=1,
        )
        h_smem_layout = cute.slice_(h_smem_layout_staged, (None, None, 0))
        h_store_smem_layout = cute.composition(
            h_smem_layout,
            cute.make_layout((self.chunk_tile, self.R_SUBTILE), stride=(1, self.T_TILE)),
        )
        c_raw_smem_layout = cute.make_layout(
            (self.chunk_tile, self.M_PAD), stride=(self.M_PAD, 1),
        )
        param_smem_layout = cute.make_layout(
            (self.chunk_size, self.M_PAD), stride=(self.M_PAD, 1),
        )

        @cute.struct
        class SharedStorage:
            wu_mbarriers: cute.struct.MemRange[cutlass.Int64, self.num_k_stages]
            state_mbarrier: cute.struct.MemRange[cutlass.Int64, 1]
            param_mbarrier: cute.struct.MemRange[cutlass.Int64, 1]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(w_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(u_smem_layout_staged)],
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
            sStatesRaw: cute.struct.Align[
                cute.struct.MemRange[self.states_dtype, cute.cosize(states_raw_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sH: cute.struct.Align[
                # C, K and LCR initially share this buffer. H reuses it after
                # K/LCR have been copied out and all reads of C have finished.
                cute.struct.MemRange[
                    self.dtype,
                    max(
                        cute.cosize(h_smem_layout_staged),
                        self.raw_param_bytes * 8 // self.dtype.width,
                    ),
                ],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(param_smem_layout)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(param_smem_layout)],
                self.SMEM_ALIGNMENT_BYTES,
            ]

        self.shared_storage = SharedStorage

        ################## Tensor views ##################
        # Flatten batch and chunk into the last dimension for state TMA loads.
        mStatesAcc_rmn = cute.make_tensor(
            mStatesAcc.iterator,
            cute.make_layout(
                (self.R, self.M, B * mStatesAcc.shape[1]),
                stride=(mStatesAcc.stride[2], mStatesAcc.stride[3], mStatesAcc.stride[1]),
            ),
        )
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
        tma_atom_states, tma_tensor_states = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mStatesAcc_rmn,
            cute.slice_(states_raw_smem_layout_staged, (None, None, 0)),
            (self.R_SUBTILE, self.M_PAD),
        )
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
            (self.chunk_tile, self.R_SUBTILE),
        )
        tma_atom_c, tma_tensor_c = self.make_param_tma(mC, c_raw_smem_layout, self.chunk_tile)
        tma_atom_k, tma_tensor_k = self.make_param_tma(mK, param_smem_layout, self.chunk_size)
        tma_atom_lcr, tma_tensor_lcr = self.make_param_tma(mLCR, param_smem_layout, self.chunk_size)

        ################## WGMMA ##################
        mma_op = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.R_SUBTILE, self.M_PAD),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        tiled_mma_states = cute.make_tiled_mma(cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(self.T_TILE, self.R_SUBTILE, self.M_PAD),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))

        self.kernel(
            L,
            nchunks,
            tma_tensor_u,
            tma_tensor_c,
            tma_tensor_k,
            tma_tensor_states,
            tma_tensor_lcr,
            tma_tensor_h,
            tiled_mma,
            tiled_mma_states,
            w_smem_layout_staged,
            u_smem_layout_staged,
            c_scaled_smem_layout_staged,
            states_acc_smem_layout_staged,
            states_raw_smem_layout_staged,
            h_smem_layout_staged,
            c_raw_smem_layout,
            param_smem_layout,
            tma_atom_u,
            tma_atom_c,
            tma_atom_k,
            tma_atom_states,
            tma_atom_lcr,
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
        L: cutlass.Int32,
        nchunks: cutlass.Int32,
        mU_rlb: cute.Tensor,
        mC_lmb: cute.Tensor,
        mK_lmb: cute.Tensor,
        mStatesAcc_rmn: cute.Tensor,
        mLCR_lmb: cute.Tensor,
        mH_lrb: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_mma_states: cute.TiledMma,
        w_smem_layout_staged,           # (T_TILE, K_TILE, PIPE)
        u_smem_layout_staged,           # (R_TILE, K_TILE, PIPE)
        c_scaled_smem_layout_staged,    # (T_TILE, M_PAD, 1)
        states_acc_smem_layout_staged,  # (R_SUBTILE, M_PAD, 1)
        states_raw_smem_layout_staged,  # (R_SUBTILE, M_PAD, 1), states_dtype
        h_smem_layout_staged,           # (T_TILE, R_SUBTILE, 1)
        c_raw_smem_layout,              # (chunk_tile, M_PAD)
        param_smem_layout,              # (chunk_size, M_PAD)
        tma_atom_u: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        tma_atom_k: cute.CopyAtom,
        tma_atom_states: cute.CopyAtom,
        tma_atom_lcr: cute.CopyAtom,
        tma_atom_h: cute.CopyAtom,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_tile, bidx_chunk, bidx_batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        log2_e = math.log2(math.e)

        bidx_t_tile = bidx_tile // self.num_r_tiles
        bidx_r_tile = bidx_tile % self.num_r_tiles
        chunk_start = bidx_chunk * self.chunk_size
        t_tile_start = bidx_t_tile * self.T_TILE
        chunk_batch = bidx_batch * nchunks + bidx_chunk
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
        num_k_tiles = cute.ceil_div(
            cutlass.min(t_tile_start + self.T_TILE, chunk_len), self.K_TILE
        )

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # W/U stages are filled once per CTA, so they need only readiness barriers.
        # One arrival per W producer warp, plus one from the U TMA issuer.
        wu_barriers = pipeline.MbarrierArray(
            barrier_storage=storage.wu_mbarriers.data_ptr(),
            num_stages=self.num_k_stages,
            agent=(pipeline.PipelineOp.TmaLoad, pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads // cute.arch.WARP_SIZE + 1,
            )),
        )
        state_barrier = pipeline.MbarrierArray(
            barrier_storage=storage.state_mbarrier.data_ptr(),
            num_stages=1,
            agent=(pipeline.PipelineOp.TmaLoad, pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1,
            )),
        )
        param_barrier = pipeline.MbarrierArray(
            barrier_storage=storage.param_mbarrier.data_ptr(),
            num_stages=1,
            agent=(pipeline.PipelineOp.TmaLoad, pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1,
            )),
        )
        sW = storage.sW.get_tensor(
            w_smem_layout_staged.outer,
            swizzle=w_smem_layout_staged.inner,
        )
        sU = storage.sU.get_tensor(
            u_smem_layout_staged.outer,
            swizzle=u_smem_layout_staged.inner,
        )
        sCScaled = storage.sCScaled.get_tensor(
            c_scaled_smem_layout_staged.outer,
            swizzle=c_scaled_smem_layout_staged.inner,
        )
        sStatesAcc = storage.sStatesAcc.get_tensor(
            states_acc_smem_layout_staged.outer,
            swizzle=states_acc_smem_layout_staged.inner,
        )
        sStatesRaw = storage.sStatesRaw.get_tensor(
            states_raw_smem_layout_staged.outer, swizzle=states_raw_smem_layout_staged.inner,
        )
        # H has its own tile because U remains live for subsequent channel subtiles.
        sH = storage.sH.get_tensor(
            h_smem_layout_staged.outer, swizzle=h_smem_layout_staged.inner,
        )
        raw_param_ptr = cute.recast_ptr(storage.sH.data_ptr(), dtype=cutlass.Uint8)
        sC = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr, dtype=self.dtype),
            c_raw_smem_layout,
        )
        sKRaw = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr + self.raw_c_bytes, dtype=self.dtype),
            param_smem_layout,
        )
        sLCRRaw = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr + self.raw_c_bytes + self.raw_k_bytes, dtype=Float32),
            param_smem_layout,
        )
        # Swizzle 16-byte groups within each row to spread producer reads across banks.
        log_dtype_bytes = cutlass.const_expr(int(math.log2(Float32.width // 8)))
        log_copy_bytes = self.log_vecsize + log_dtype_bytes
        lcr_swizzle_bits = (
            cutlass.const_expr(int(math.log2(self.M_PAD * Float32.width // 8))) - log_copy_bytes
        )
        k_swizzle_bits = (
            cutlass.const_expr(int(math.log2(self.M_PAD * self.dtype.width // 8))) - log_copy_bytes
        )
        lcr_swizzle = cute.make_swizzle(lcr_swizzle_bits, log_copy_bytes, self.PARAM_SWIZZLE_SHIFT)
        k_swizzle = cute.make_swizzle(k_swizzle_bits, log_copy_bytes, self.PARAM_SWIZZLE_SHIFT)
        sLCR = storage.sLCR.get_tensor(param_smem_layout, swizzle=lcr_swizzle)
        sK = storage.sK.get_tensor(param_smem_layout, swizzle=k_swizzle)

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
            (self.chunk_tile, self.R_SUBTILE),
            (None, None),
        )
        bSG_sH, bSG_gH = cute.nvgpu.cpasync.tma_partition(
            tma_atom_h,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(
                cute.local_tile(sH, (self.chunk_tile, self.R_SUBTILE, 1), (0, 0, 0)),
                0, 2,
            ),
            gmem_tensor=cute.group_modes(gH, 0, 2),
        )
        if warp_idx == self.producer_load_warp_idx:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_u)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_states)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_lcr)
        if warp_idx == self.h_store_warp_idx:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_h)

        ################## pipelines ##################
        cute.arch.mbarrier_init_fence()
        h_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads
            ),
        )

        ################## prologue ##################
        cute.arch.sync_threads()
        if warp_idx == self.producer_load_warp_idx:
            param_barrier.arrive_and_expect_tx(0, self.raw_param_bytes)
            self.load_param(
                tma_atom_c, mC_lmb, sC, self.chunk_tile,
                (chunk_start + t_tile_start) // self.chunk_tile, bidx_batch,
                param_barrier.get_barrier(0),
            )
            self.load_param(
                tma_atom_k, mK_lmb, sKRaw, self.chunk_size,
                bidx_chunk, bidx_batch, param_barrier.get_barrier(0),
            )
            self.load_param(
                tma_atom_lcr, mLCR_lmb, sLCRRaw, self.chunk_size,
                bidx_chunk, bidx_batch, param_barrier.get_barrier(0),
            )

        # Each stage's barrier tracks both the U load and all W producer warps.
        if warp_idx == self.producer_load_warp_idx:
            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                wu_barriers.arrive_and_expect_tx(k_tile, u_tma_copy_bytes)
                u_tile = (chunk_start + k_tile * self.K_TILE) // self.K_TILE
                cute.copy(
                    tma_atom_u, bGS_gU[None, u_tile], bGS_sU[None, k_tile],
                    tma_bar_ptr=wu_barriers.get_barrier(k_tile),
                )

        # Zero W while parameters load. Producers later write only causal entries.
        sWLinear = cute.make_tensor(
            cute.recast_ptr(sW.iterator, dtype=self.dtype),
            cute.make_layout(cute.cosize(w_smem_layout_staged)),
        )
        rWZeros = cute.make_rmem_tensor((self.input_vecsize,), self.dtype)
        rWZeros.fill(self.dtype(0.0))
        for zero_i in cutlass.range_constexpr(cute.ceil_div(
            cute.cosize(w_smem_layout_staged), self.input_vecsize * self.num_threads
        )):
            zero_vec = tidx + zero_i * self.num_threads
            if zero_vec * self.input_vecsize < cute.cosize(w_smem_layout_staged):
                cute.autovec_copy(rWZeros, cute.local_tile(sWLinear, (self.input_vecsize,), (zero_vec,)))
        param_barrier.wait(0, 0)
        cute.arch.fence_view_async_shared()

        for i in cutlass.range_constexpr(self.num_lcr_copy_iters):
            idx = tidx + i * self.num_threads
            if idx < self.chunk_size * self.M_PAD:
                local_k = idx // self.M_PAD
                local_m = idx % self.M_PAD
                sLCR[local_k, local_m] = sLCRRaw[local_k, local_m]

        sKRawWords = cute.recast_tensor(sKRaw, cutlass.Uint32)
        # recast_tensor drops pointer swizzles. Reapply K's swizzle for the packed stores.
        sKWords = cute.make_tensor(
            cute.recast_ptr(sK.iterator, k_swizzle, dtype=cutlass.Uint32),
            sKRawWords.layout,
        )
        for i in cutlass.range_constexpr(self.num_k_copy_iters):
            vec_idx = tidx + i * self.num_threads
            if vec_idx < self.chunk_size * self.M_PAD // self.k_copy_vecsize:
                local_k = vec_idx // (self.M_PAD // self.k_copy_vecsize)
                local_vec = vec_idx % (self.M_PAD // self.k_copy_vecsize)
                sKWords[local_k, local_vec] = sKRawWords[local_k, local_vec]

        # Finish copying K/LCR out of the H buffer before producers read them.
        # C remains there until W and C_scaled are complete, before the first H store.
        cute.arch.sync_threads()

        ################## producer mainloop ##################
        if warp_idx >= self.num_consumer_warps:
            producer_tidx = tidx - self.num_consumer_threads

            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                self.compute_w_stage(
                    sW, sC, sK, sLCR, k_tile, producer_tidx,
                    t_tile_start, chunk_len, log2_e,
                )

                cute.arch.fence_view_async_shared()
                # Publish once per warp, after all lanes' stores are ordered.
                # No producer warp waits for another warp or for the consumer.
                cute.arch.sync_warp()
                if producer_tidx % cute.arch.WARP_SIZE == 0:
                    wu_barriers.arrive_mbarrier(k_tile)

        ################## consumer prologue ##################
        if warp_idx < self.num_consumer_warps:
            if warp_idx == self.state_load_warp_idx:
                self.load_state(
                    tma_atom_states, mStatesAcc_rmn, sStatesRaw, state_barrier,
                    chunk_batch, bidx_r_tile, 0,
                )
            thr_mma = tiled_mma.get_slice(tidx)
            tCrW = thr_mma.make_fragment_A(thr_mma.partition_A(sW))
            tCrU = thr_mma.make_fragment_B(thr_mma.partition_B(sU))
            thr_mma_states = tiled_mma_states.get_slice(tidx)
            tCrCScaled = thr_mma_states.make_fragment_A(thr_mma_states.partition_A(sCScaled))
            tCrStatesAcc = thr_mma_states.make_fragment_B(thr_mma_states.partition_B(sStatesAcc))

            tiled_copy_states = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.states_dtype,
                    num_bits_per_copy=self.states_vecsize * self.states_dtype.width,
                ),
                cute.make_layout(
                    (self.num_states_threads_R, self.num_states_threads_M),
                    stride=(self.num_states_threads_M, 1),
                ),
                cute.make_layout((1, self.states_vecsize), stride=(self.states_vecsize, 1)),
            )
            thr_copy_states = tiled_copy_states.get_slice(tidx)
            tSS_sStatesRaw = thr_copy_states.partition_S(sStatesRaw)
            tSS_sStatesAcc = thr_copy_states.partition_D(sStatesAcc)

            for i in cutlass.range_constexpr(self.num_c_scale_iters):
                idx = tidx + i * self.num_consumer_threads
                local_t = idx // self.M_PAD
                local_m = idx % self.M_PAD
                valid_state = local_m < self.M and t_tile_start + local_t < chunk_len
                if valid_state:
                    scaled = cute.math.exp2(
                        sLCR[t_tile_start + local_t, local_m] * log2_e,
                        fastmath=True,
                    ) * sC[local_t, local_m].to(Float32)
                    sCScaled[local_t, local_m, 0] = scaled.to(self.dtype)
                else:
                    sCScaled[local_t, local_m, 0] = self.dtype(0.0)

            cute.arch.fence_view_async_shared()
            self.consumer_sync()

            ################## consumer mainloop ##################
            # Computing 64 columns at a time keeps the accumulator at 32 FP32
            # registers per consumer thread for every R_TILE.
            for r_subtile in cutlass.range_constexpr(self.num_r_subtiles):
                state_barrier.wait(0, r_subtile % 2)
                cute.arch.fence_view_async_shared()
                rStatesRaw = cute.make_rmem_tensor_like(tSS_sStatesRaw[None, None, None, 0])
                rStatesAcc = cute.make_rmem_tensor_like(tSS_sStatesAcc[None, None, None, 0])
                cute.autovec_copy(tSS_sStatesRaw[None, None, None, 0], rStatesRaw)
                rStatesAcc.store(rStatesRaw.load().to(self.dtype))
                cute.autovec_copy(rStatesAcc, tSS_sStatesAcc[None, None, None, 0])

                cute.arch.fence_view_async_shared()
                self.consumer_sync()
                # All consumers have finished reading sStatesRaw, so load the next
                # state subtile while WGMMA uses the converted values in sStatesAcc.
                if r_subtile + 1 < self.num_r_subtiles:
                    if warp_idx == self.state_load_warp_idx:
                        self.load_state(
                            tma_atom_states, mStatesAcc_rmn, sStatesRaw,
                            state_barrier,
                            chunk_batch, bidx_r_tile, r_subtile + 1,
                        )

                ################## W @ U WGMMA ##################
                acc = thr_mma.make_fragment_C(
                    thr_mma.partition_shape_C((self.T_TILE, self.R_SUBTILE))
                )
                cute.nvgpu.warpgroup.fence()
                # W and U remain in shared memory for every R subtile, so only
                # the first subtile needs to wait for their producers.
                for k_tile in cutlass.range_constexpr(self.num_k_stages):
                    if k_tile < num_k_tiles:
                        if r_subtile == 0:
                            wu_barriers.wait(k_tile, 0)
                            cute.arch.fence_view_async_shared()

                        self.gemm(
                            tiled_mma,
                            tCrW[None, None, None, k_tile],
                            cute.local_tile(
                                tCrU[None, None, None, k_tile],
                                (cute.size(tCrU, mode=[0]), 1, cute.size(tCrU, mode=[2])),
                                (0, r_subtile, 0),
                            ),
                            acc,
                            k_tile != 0,
                        )
                        cute.nvgpu.warpgroup.commit_group()

                ################## state WGMMA ##################
                self.gemm(
                    tiled_mma_states,
                    tCrCScaled[None, None, None, 0],
                    tCrStatesAcc[None, None, None, 0],
                    acc,
                    True,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                ################## epilogue ##################
                h_tile = (chunk_start + t_tile_start) // self.chunk_tile
                self.store_h_epilogue(
                    tiled_mma,
                    tma_atom_h,
                    bSG_sH,
                    bSG_gH[None, None, bidx_r_tile * self.num_r_subtiles + r_subtile],
                    h_store_pipeline,
                    sH,
                    acc,
                    h_tile,
                    tidx,
                    warp_idx,
                )
                # Wait for the TMA store before reusing sH for the next subtile.
                self.consumer_sync()

    @cute.jit
    def make_param_tma(self, mX: cute.Tensor, smem_layout, tile_rows):
        mX_lmb = cute.make_tensor(
            mX.iterator,
            cute.make_layout(
                (mX.shape[1], self.M, mX.shape[0]),
                stride=(mX.stride[1], mX.stride[2], mX.stride[0]),
            ),
        )
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX_lmb, smem_layout,
            (tile_rows, self.M_PAD),
        )

    @cute.jit
    def load_param(
        self, tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, sX: cute.Tensor,
        tile_rows, tile_row, bidx_batch, barrier_ptr,
    ):
        gX = cute.local_tile(
            tma_tensor[None, None, bidx_batch], (tile_rows, self.M_PAD), (tile_row, 0),
        )
        bGS_sX, bGS_gX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX, 0, 2),
            gmem_tensor=cute.group_modes(gX, 0, 2),
        )
        cute.copy(tma_atom, bGS_gX[None], bGS_sX[None], tma_bar_ptr=barrier_ptr)

    @cute.jit
    def load_state(
        self, tma_atom_states: cute.CopyAtom, mStatesAcc_rmn: cute.Tensor,
        sStatesRaw: cute.Tensor,
        state_barrier: pipeline.MbarrierArray,
        chunk_batch, bidx_r_tile, r_subtile,
    ):
        state_barrier.arrive_and_expect_tx(
            0, cutlass.const_expr(self.R_SUBTILE * self.M_PAD * self.states_dtype.width // 8),
        )
        gStatesRaw = cute.local_tile(
            mStatesAcc_rmn[None, None, chunk_batch],
            (self.R_SUBTILE, self.M_PAD),
            (bidx_r_tile * self.num_r_subtiles + r_subtile, 0),
        )
        bGS_sStatesRaw, bGS_gStatesRaw = cute.nvgpu.cpasync.tma_partition(
            tma_atom_states,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sStatesRaw, 0, 2),
            gmem_tensor=cute.group_modes(gStatesRaw, 0, 2),
        )
        cute.copy(
            tma_atom_states, bGS_gStatesRaw[None], bGS_sStatesRaw[None, 0],
            tma_bar_ptr=state_barrier.get_barrier(0),
        )

    @cute.jit
    def consumer_sync(self):
        cute.arch.barrier(
            barrier_id=self.CONSUMER_BARRIER_ID,
            number_of_threads=self.num_consumer_threads,
        )

    @staticmethod
    @cute.jit
    def load_param_s2r(
        sX: cute.Tensor, row_idx, tile_idx, tile_size: cutlass.Constexpr,
    ):
        # Use 16-byte loads to reduce swizzled address calculations.
        sXTile = cute.local_tile(
            cute.slice_(sX, (row_idx, None)), (tile_size,), (tile_idx,),
        )
        rX = cute.make_rmem_tensor((tile_size,), sX.element_type)
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), sX.element_type, num_bits_per_copy=128,
            ),
            sXTile, rX,
        )
        return rX

    @cute.jit
    def compute_w_stage(
        self,
        sW: cute.Tensor,    # (T_TILE, K_TILE, PIPE)
        sC: cute.Tensor,    # (chunk_tile, M_PAD)
        sK: cute.Tensor,    # (chunk_size, M_PAD)
        sLCR: cute.Tensor,  # (chunk_size, M_PAD), fp32
        k_tile,
        producer_tidx,
        t_tile_start,
        chunk_len,
        log2_e,
    ):
        """Fill the causal entries of a W tile whose other entries are already zero."""
        sW_stage = cute.slice_(sW, (None, None, k_tile))
        # Groups of four threads share a K column and start on four different rows.
        num_threads_per_group = self.num_w_threads_T * self.num_w_threads_K
        local_k_pair = (producer_tidx % num_threads_per_group) // self.num_w_threads_T
        local_t_base = (
            (producer_tidx // num_threads_per_group) * self.num_w_threads_T
            + producer_tidx % self.num_w_threads_T
        )
        # Pair low and high K columns to balance work across the causal triangle.
        for k_side in cutlass.range(self.w_k_columns_per_thread, unroll=1):
            local_k = local_k_pair
            if k_side != 0:
                local_k = self.K_TILE - 1 - local_k_pair
            k = k_tile * self.K_TILE + local_k
            rK = cute.make_rmem_tensor((self.M,), Float32)
            rKRaw = self.load_param_s2r(sK, k, 0, self.M)
            rK.store(rKRaw.load().to(Float32))
            rLCRK = self.load_param_s2r(sLCR, k, 0, self.M)

            for i in cutlass.range_constexpr(self.num_w_row_iters):
                local_t = local_t_base + i * self.w_rows_per_iter
                if local_t < self.chunk_tile:
                    t = t_tile_start + local_t
                    w_val = Float32(0.0)
                    # Check causality before exp because noncausal entries can overflow.
                    if t < chunk_len and k <= t:
                        # Load four LCR values together and accumulate in increasing m order.
                        for m_group in cutlass.range_constexpr(self.M // self.vecsize):
                            rLCRT = self.load_param_s2r(sLCR, t, m_group, self.vecsize)
                            for m_offset in cutlass.range_constexpr(self.vecsize):
                                m = m_group * self.vecsize + m_offset
                                w_val += (
                                    sC[local_t, m].to(Float32)
                                    * rK[m]
                                    * cute.math.exp2(
                                        (rLCRT[m_offset] - rLCRK[m]) * log2_e,
                                        fastmath=True,
                                    )
                                )
                        sW_stage[local_t, local_k] = w_val.to(self.dtype)

    @cute.jit
    def store_h_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_h: cute.CopyAtom,
        bSG_sH: cute.Tensor,
        bSG_gH: cute.Tensor,
        h_store_pipeline,
        sH: cute.Tensor,  # (T_TILE, R_SUBTILE, 1)
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

        cvt_copy_r2s(tiled_copy_r2s, acc, tRS_sH[None, None, None, 0])

        cute.arch.fence_view_async_shared()
        self.consumer_sync()

        if warp_idx == self.h_store_warp_idx:
            cute.copy(tma_atom_h, bSG_sH[None, 0], bSG_gH[None, h_tile])
            h_store_pipeline.producer_commit()
            h_store_pipeline.producer_tail()

    @staticmethod
    @cute.jit
    def gemm(tiled_mma, A, B, C, accumulate_initial: Boolean):
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
            mma_atom.set(
                cute.nvgpu.warpgroup.Field.ACCUMULATE,
                accumulate_initial or k_block_idx != 0,
            )
            cute.gemm(mma_atom, C, A[None, None, k_block_idx], B[None, None, k_block_idx], C)


@jit_cache
def _compile_chunk_fused_readout_fwd(dtype, R, M, chunk_size, config, states_dtype):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_u = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    # C advertises vector alignment and K advertises word alignment.
    # The remaining tensors retain scalar stride promises.
    div_c = 128 // dtype.width
    div_k = cutlass.Uint32.width // dtype.width
    fake_c = make_fake_tensor(dtype, (B_sym, L_sym, M), divisibility=div_c)
    fake_kappa = make_fake_tensor(dtype, (B_sym, L_sym, M), divisibility=div_k)
    fake_states_acc = make_fake_tensor(states_dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=1)
    fake_h = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutFwd(
            dtype, R=R, M=M, chunk_size=chunk_size, config=config, states_dtype=states_dtype,
        ),
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
        torch2cute_dtype_map[u.dtype], u.shape[-1], c.shape[-1], chunk_size,
        config, torch2cute_dtype_map[states_acc.dtype],
    )


@autotune(
    configs=warp_specialized_configs(),
    key=["chunk_size", ("u", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": prune_warp_specialized_configs},
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
    u,
    c,
    kappa,
    states_acc,
    log_cum_rho,
    chunk_size,
    config=None,
):
    B, L, R = u.shape
    M = c.shape[-1]
    h = torch.empty((B, L, R), device=u.device, dtype=u.dtype)

    if B == 0 or L == 0:
        return h
    assert u.dtype in (torch.float16, torch.bfloat16), f"u must be 16-bit, got {u.dtype}"
    assert c.dtype == u.dtype and kappa.dtype == u.dtype, "c and kappa dtype must match u dtype"
    assert states_acc.dtype in (u.dtype, torch.float32), (
        f"states_acc must match u dtype or be float32, got {states_acc.dtype}"
    )
    assert log_cum_rho.dtype == torch.float32, "log_cum_rho must be float32"

    if config is not None:
        kernel = _prepare(
            u, c, kappa, states_acc, log_cum_rho, h,
            chunk_size=chunk_size, config=config,
        )
    else:
        key = (u.device.index, u.dtype, states_acc.dtype, R, M, chunk_size)
        kernel = _kernel_cache.get(key)
        if kernel is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up warp-specialized autotuning before CUDA graph capture")
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
