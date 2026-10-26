"""Autotuned persistent SM90 forward for fused chunk readout.

A bounded CTA grid reuses W/U slots across output tiles. Producers build W
while one consumer warp group computes H.T in 64-channel subtiles.

    W[t, k] = sum_m C[t, m] * K[k, m] * exp(LCR[t, m] - LCR[k, m]), k <= t
    H.T = U.T @ W.T + states_acc @ (C * exp(LCR)).T

U.T is widened from its 16-bit tile into the register A operand of the TF32
WGMMA, so W and the scaled C are the K-major shared-memory B operands. U.T is
read with ldmatrix, which needs K reordered within each WGMMA step, so W is
stored in the same K order.

State loads cover 64 channels or the full CTA R tile. Each raw-state tile is
reused after all of its channels have been converted for WGMMA. Each
output subtile is stored with TMA before its shared buffer is reused.
"""

import math
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Boolean, Float32, TFloat32
from cutlass.cute.nvgpu import cpasync
from cutlass.tensor_utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

from frac_ssm.ops.cute.cache_utils import jit_cache
from frac_ssm.ops.cute.autotuner import AutotuneConfig, autotune
from frac_ssm.ops.cute.utils import (
    MmaTF32Op,
    cvt_copy_r2s,
    make_fake_tensor,
    tf32_rmem_view,
    tf32_smem_view,
    torch2cute_dtype_map,
)


@dataclass(frozen=True)
class ChunkFusedReadoutFwdConfig:
    """Compile-time resources and runtime CTA limits for the persistent kernel."""

    num_producer_threads: int = 128
    min_blocks_per_mp: int = 4  # Compiler occupancy target.
    ctas_per_sm: int = 4  # Persistent grid cap per SM.
    max_ctas: int | None = None
    work_slots: int = 1
    producer_regs: int = 0
    consumer_regs: int = 0
    work_group: int = 1
    # None loads all R_TILE channels. 64 loads one output subtile at a time.
    STATE_R_TILE: int | None = None


PRODUCER_CONFIGS = (
    # Producer threads, minimum CTAs per SM, producer/consumer register limits per thread.
    # Zero leaves register allocation to the compiler without explicit redistribution.
    (64, 5, 0, 0),
    (128, 4, 0, 0),
    (128, 3, 0, 0),
    (128, 2, 0, 0),
    (128, 1, 0, 0),
    (192, 3, 0, 0),
    (256, 2, 0, 0),
    (128, 3, 96, 64),
)


REGISTERS_PER_SM = 65536
REGISTER_GRANULARITY = 8
MIN_REGISTER_LIMIT = 24
MAX_REGISTER_LIMIT = 256


def persistent_configs():
    # All shapes share the same resource and scheduling search space.
    configs = []
    for ctas_per_sm, work_slots, work_group, STATE_R_TILE, producer_config in product(
        (1, 2, 3, 4, 5), (1, 2), (1, 2),
        (None, ChunkFusedReadoutFwd.R_SUBTILE), PRODUCER_CONFIGS,
    ):
        num_producer_threads, min_blocks_per_mp, producer_regs, consumer_regs = producer_config
        configs.append(AutotuneConfig(config=ChunkFusedReadoutFwdConfig(
            num_producer_threads=num_producer_threads,
            min_blocks_per_mp=min_blocks_per_mp,
            producer_regs=producer_regs,
            consumer_regs=consumer_regs,
            ctas_per_sm=ctas_per_sm,
            work_slots=work_slots,
            work_group=work_group,
            STATE_R_TILE=STATE_R_TILE,
        )))
    return configs


def prune_persistent_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    u, c, states_acc, chunk_size = args["u"], args["c"], args["states_acc"], args["chunk_size"]
    B, L, R = u.shape
    props = torch.cuda.get_device_properties(u.device)
    sm_count = props.multi_processor_count
    R_TILE = ChunkFusedReadoutFwd.r_tile_for(R)
    work_count = (
        B * math.ceil(L / chunk_size)
        * math.ceil(chunk_size / ChunkFusedReadoutFwd.T_TILE) * math.ceil(R / R_TILE)
    )
    seen = set()
    candidates = []
    for candidate in configs:
        config = candidate.kwargs["config"]
        groups = math.ceil(work_count / config.work_group)
        cap = config.ctas_per_sm * sm_count
        grid = min(groups, cap)
        max_items = min(work_count, math.ceil(groups / grid) * config.work_group)
        if config.work_slots > max_items:
            continue
        shared_bytes = ChunkFusedReadoutFwd.shared_memory_bytes(
            R, chunk_size, config, u.element_size(), c.element_size(), states_acc.element_size(),
        )
        if shared_bytes > props.shared_memory_per_block_optin:
            continue
        launch = (
            config.num_producer_threads, config.min_blocks_per_mp,
            config.producer_regs, config.consumer_regs, config.work_slots,
            config.work_group,
            config.STATE_R_TILE or R_TILE, grid,
        )
        if launch in seen:
            continue
        seen.add(launch)
        candidates.append(AutotuneConfig(config=replace(config, max_ctas=cap)))
    return candidates


class ChunkFusedReadoutFwd:
    T_TILE = 64  # Output time steps per work tile
    K_TILE = 32  # Reduction width of each W @ U tile
    R_SUBTILE = 64  # Output columns computed by the consumer warp group at once
    M_PAD = 16  # WGMMA reduction width for the state term
    MAX_R_SUBTILES = 3
    SMEM_ALIGNMENT_BYTES = 128
    MMA_ALIGNMENT_BYTES = 1024  # One full 128-byte swizzle pattern for TF32 WGMMA operands
    PARAM_ALIGNMENT_BYTES = 16
    PARAM_PADDING_BYTES = 16  # Extra space per K/LCR row to avoid bank conflicts.
    PRODUCER_BARRIER_ID = 2
    CONSUMER_BARRIER_ID = 1

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        R: int,
        M: int,
        chunk_size: int,
        config: ChunkFusedReadoutFwdConfig,
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

        # The default width shares W across up to three 64-column output subtiles.
        self.R_TILE = self.r_tile_for(R)
        self.num_r_subtiles = self.R_TILE // self.R_SUBTILE

        assert self.config.STATE_R_TILE in (None, self.R_SUBTILE)
        self.STATE_R_TILE = self.config.STATE_R_TILE or self.R_TILE
        self.r_subtiles_per_state_tile = self.STATE_R_TILE // self.R_SUBTILE

        param_dtype_bytes = self.param_dtype.width // 8
        lcr_dtype_bytes = Float32.width // 8
        self.lcr_stride = self.M_PAD + self.PARAM_PADDING_BYTES // lcr_dtype_bytes  # Padding in FP32 elements.
        self.k_stride = self.M_PAD + self.PARAM_PADDING_BYTES // param_dtype_bytes  # Padding in K elements.

        self.chunk_tile = min(self.T_TILE, chunk_size)
        self.raw_c_bytes = self.chunk_tile * self.M_PAD * param_dtype_bytes  # C covers the output row tile.
        self.raw_k_bytes = self.chunk_size * self.M_PAD * param_dtype_bytes  # K covers the full chunk.
        self.raw_lcr_bytes = self.chunk_size * self.M_PAD * lcr_dtype_bytes  # LCR covers the full chunk.
        self.raw_param_bytes = self.raw_c_bytes + self.raw_k_bytes + self.raw_lcr_bytes

        self.num_k_stages = math.ceil(self.chunk_size / self.K_TILE)  # Maximum number of K reduction tiles per work slot.
        self.num_work_slots = self.config.work_slots
        self.work_group = self.config.work_group
        assert self.work_group in (1, 2)
        assert self.num_work_slots in (1, 2)
        self.num_wu_stages = self.num_work_slots * self.num_k_stages

        self.num_r_tiles = math.ceil(R / self.R_TILE)
        self.num_t_tiles = math.ceil(self.chunk_size / self.T_TILE)  # Number of output row tiles per chunk.

        self.num_consumer_threads = cute.arch.THREADS_PER_WARPGROUP  # 128
        self.num_producer_threads = self.config.num_producer_threads
        self.num_threads = self.num_consumer_threads + self.num_producer_threads

        if self.config.producer_regs or self.config.consumer_regs:
            assert self.num_producer_threads % cute.arch.THREADS_PER_WARPGROUP == 0  # 128
            assert (
                self.config.producer_regs % REGISTER_GRANULARITY
                == self.config.consumer_regs % REGISTER_GRANULARITY == 0
            )
            assert (
                MIN_REGISTER_LIMIT <= self.config.consumer_regs
                <= self.config.producer_regs <= MAX_REGISTER_LIMIT
            )
            assert (
                self.num_producer_threads * self.config.producer_regs
                + self.num_consumer_threads * self.config.consumer_regs
            ) * self.config.min_blocks_per_mp <= REGISTERS_PER_SM

        # Consumers occupy the first warp group, followed by producer warps.
        # The first producer warp loads U/parameters, consumer warp 1 loads
        # states, and consumer warp 0 stores H.
        self.num_consumer_warps = self.num_consumer_threads // cute.arch.WARP_SIZE
        self.producer_load_warp_idx = self.num_consumer_warps
        self.state_load_warp_idx = 1
        self.h_store_warp_idx = 0
        self.param_release_warp_idx = 0

        self.producer_barrier = pipeline.NamedBarrier(
            self.PRODUCER_BARRIER_ID, self.num_producer_threads
        )
        self.consumer_barrier = pipeline.NamedBarrier(
            self.CONSUMER_BARRIER_ID, self.num_consumer_threads
        )

        self.vecsize = 128 // Float32.width
        self.input_vecsize = 128 // dtype.width
        self.param_vecsize = 128 // self.param_dtype.width
        self.num_c_scale_iters = self.chunk_tile * self.M_PAD // self.num_consumer_threads
        self.num_lcr_copy_iters = math.ceil(
            self.chunk_size * self.M_PAD / (self.vecsize * self.num_producer_threads)
        )
        self.num_k_copy_iters = math.ceil(
            self.chunk_size * self.M_PAD / (self.param_vecsize * self.num_producer_threads)
        )
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

        assert dtype.width == 16, (
            f"chunk fused readout expects 16-bit inputs, got {dtype}"
        )
        assert self.states_dtype in (dtype, Float32), (
            f"states_dtype must match dtype or be float32, got {self.states_dtype}"
        )
        assert self.param_dtype in (dtype, Float32), (
            f"C and K dtype must match dtype or be float32, got {self.param_dtype}"
        )
        assert M in (8, 16), f"M must be 8 or 16, got {M}"
        assert chunk_size in (32, 64, 128), (
            f"chunk fused readout supports chunk_size 32/64/128, got {chunk_size}"
        )
        assert self.num_producer_threads in (64, 128, 192, 256)

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,                  # (B, L, R)
        mC: cute.Tensor,                  # (B, L, M)
        mK: cute.Tensor,                  # (B, L, M)
        mStatesAcc: cute.Tensor,          # (B, nchunks, R, M)
        mLCR: cute.Tensor,                # (B, L, M), fp32
        mH: cute.Tensor,                  # (B, L, R)
        max_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        B, L = mU.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)

        ################## SMEM layouts ##################
        intra_mma_tiler_mnk = (self.R_TILE, self.chunk_tile, self.K_TILE)
        state_mma_tiler_mnk = (self.R_SUBTILE, self.chunk_tile, self.M_PAD)
        w_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, intra_mma_tiler_mnk, TFloat32, num_stages=self.num_wu_stages
        )
        u_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, intra_mma_tiler_mnk, self.dtype, num_stages=self.num_wu_stages
        )
        u_smem_layout = cute.slice_(u_smem_layout_staged, (None, None, 0))  # (R_TILE, K_TILE)
        c_scaled_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, state_mma_tiler_mnk, TFloat32, num_stages=1
        )
        states_acc_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, state_mma_tiler_mnk, TFloat32, num_stages=1
        )
        states_raw_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (self.T_TILE, self.STATE_R_TILE, self.M_PAD), self.states_dtype,
            num_stages=1,
        )
        h_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.dtype,
            LayoutEnum.COL_MAJOR,
            (self.R_SUBTILE, self.chunk_tile),
            epi_stage=1,
        )
        h_smem_layout = cute.slice_(h_smem_layout_staged, (None, None, 0))
        lcr_smem_layout = cute.make_layout(
            (self.chunk_size, self.M_PAD), stride=(self.lcr_stride, 1)
        )
        k_smem_layout = cute.make_layout(
            (self.chunk_size, self.M_PAD), stride=(self.k_stride, 1)
        )
        c_raw_smem_layout = cute.make_layout(
            (self.chunk_tile, self.M_PAD), stride=(self.M_PAD, 1),
        )
        param_raw_smem_layout = cute.make_layout(
            (self.chunk_size, self.M_PAD), stride=(self.M_PAD, 1),
        )
        ################## SMEM allocations ##################
        @cute.struct
        class SharedStorage:
            wu_barriers: cute.struct.MemRange[cutlass.Int64, 2 * self.num_work_slots]
            u_barriers: cute.struct.MemRange[cutlass.Int64, self.num_work_slots]
            state_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            param_load_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(w_smem_layout_staged)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(u_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sCScaled: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(c_scaled_smem_layout_staged)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sStatesAcc: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(states_acc_smem_layout_staged)],
                self.MMA_ALIGNMENT_BYTES,
            ]
            sStatesRaw: cute.struct.Align[
                cute.struct.MemRange[self.states_dtype, cute.cosize(states_raw_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sH: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(h_smem_layout_staged)],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sParamRaw: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.raw_param_bytes],
                self.SMEM_ALIGNMENT_BYTES,
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(lcr_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.param_dtype, cute.cosize(k_smem_layout)],
                self.PARAM_ALIGNMENT_BYTES,
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
        mH_rlb = cute.make_tensor(
            mH.iterator,
            cute.make_layout(
                (self.R, L, B),
                stride=(mH.stride[2], mH.stride[1], mH.stride[0]),
            ),
        )

        ################## TMA setup ##################
        tma_atom_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mU_rlb,
            u_smem_layout,
            (self.R_TILE, self.K_TILE),
        )
        tma_atom_c, tma_tensor_c = self.make_param_tma(mC, c_raw_smem_layout, self.chunk_tile)
        tma_atom_k, tma_tensor_k = self.make_param_tma(mK, param_raw_smem_layout, self.chunk_size)
        tma_atom_states, tma_tensor_states = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mStatesAcc_rmn,
            cute.slice_(states_raw_smem_layout_staged, (None, None, 0)),
            (self.STATE_R_TILE, self.M_PAD),
        )
        tma_atom_lcr, tma_tensor_lcr = self.make_param_tma(
            mLCR, param_raw_smem_layout, self.chunk_size,
        )
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mH_rlb,
            h_smem_layout,
            (self.R_SUBTILE, self.chunk_tile),
        )

        ################## WGMMA ##################
        tiled_mma = cute.make_tiled_mma(MmaTF32Op(
            instruction_shape=(self.R_SUBTILE, self.chunk_tile, 8),
            a_src=cute.nvgpu.warpgroup.OperandSource.RMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))
        tiled_mma_states = cute.make_tiled_mma(MmaTF32Op(
            instruction_shape=(self.R_SUBTILE, self.chunk_tile, 8),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        ))

        ################## persistent launch ##################
        work_count = B * nchunks * self.num_t_tiles * self.num_r_tiles
        chunk_divisor = cute.fast_divmod_create_divisor_v2(nchunks)
        self.kernel(
            L,
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
            lcr_smem_layout,
            k_smem_layout,
            c_raw_smem_layout,
            param_raw_smem_layout,
            tma_atom_u,
            tma_atom_c,
            tma_atom_k,
            tma_atom_states,
            tma_atom_lcr,
            tma_atom_h,
            work_count,
            chunk_divisor,
        ).launch(
            grid=[cutlass.min(cute.ceil_div(work_count, self.work_group), max_ctas), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.config.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        L: cutlass.Int32,
        mU_rlb: cute.Tensor,
        mC_lmb: cute.Tensor,
        mK_lmb: cute.Tensor,
        mStatesAcc_rmn: cute.Tensor,
        mLCR_lmb: cute.Tensor,
        mH_rlb: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_mma_states: cute.TiledMma,
        w_smem_layout_staged,             # (chunk_tile, K_TILE, num_wu_stages)
        u_smem_layout_staged,             # (R_TILE, K_TILE, num_wu_stages)
        c_scaled_smem_layout_staged,      # (chunk_tile, M_PAD, 1)
        states_acc_smem_layout_staged,    # (R_SUBTILE, M_PAD, 1)
        states_raw_smem_layout_staged,    # (STATE_R_TILE, M_PAD, 1)
        h_smem_layout_staged,             # (R_SUBTILE, chunk_tile, 1)
        lcr_smem_layout,                  # (chunk_size, M_PAD), padded row stride
        k_smem_layout,                    # (chunk_size, M_PAD), padded row stride
        c_raw_smem_layout,                # (chunk_tile, M_PAD)
        param_raw_smem_layout,            # (chunk_size, M_PAD)
        tma_atom_u: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        tma_atom_k: cute.CopyAtom,
        tma_atom_states: cute.CopyAtom,
        tma_atom_lcr: cute.CopyAtom,
        tma_atom_h: cute.CopyAtom,
        work_count: cutlass.Int32,
        chunk_divisor: cute.FastDivmodDivisorV2,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        work_idx = block_idx * self.work_group
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        log2_e = math.log2(math.e)

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sW = storage.sW.get_tensor(w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner)
        sU = storage.sU.get_tensor(u_smem_layout_staged.outer, swizzle=u_smem_layout_staged.inner)
        sCScaled = storage.sCScaled.get_tensor(
            c_scaled_smem_layout_staged.outer, swizzle=c_scaled_smem_layout_staged.inner,
        )
        sStatesAcc = storage.sStatesAcc.get_tensor(
            states_acc_smem_layout_staged.outer, swizzle=states_acc_smem_layout_staged.inner,
        )
        sStatesRaw = storage.sStatesRaw.get_tensor(
            states_raw_smem_layout_staged.outer, swizzle=states_raw_smem_layout_staged.inner,
        )

        # H needs a separate tile because later channel subtiles still read U.
        sH = storage.sH.get_tensor(h_smem_layout_staged.outer, swizzle=h_smem_layout_staged.inner)
        raw_param_ptr = storage.sParamRaw.data_ptr()
        sC = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr, dtype=self.param_dtype),
            c_raw_smem_layout,
        )
        sKRaw = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr + self.raw_c_bytes, dtype=self.param_dtype),
            param_raw_smem_layout,
        )
        sLCRRaw = cute.make_tensor(
            cute.recast_ptr(raw_param_ptr + self.raw_c_bytes + self.raw_k_bytes, dtype=Float32),
            param_raw_smem_layout,
        )
        sLCR = storage.sLCR.get_tensor(lcr_smem_layout)
        sK = storage.sK.get_tensor(k_smem_layout)

        if warp_idx == self.producer_load_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_u)
            cpasync.prefetch_descriptor(tma_atom_states)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_lcr)
        if warp_idx == self.h_store_warp_idx:
            cpasync.prefetch_descriptor(tma_atom_h)

        u_tma_copy_bytes = cutlass.const_expr(
            cute.size_in_bytes(
                self.dtype,
                cute.slice_(u_smem_layout_staged, (None, None, 0)),
            )
        )
        w_slot_elements = cutlass.const_expr(
            cute.cosize(w_smem_layout_staged) // self.num_work_slots
        )

        ################## pipelines ##################
        # Initialize all barriers before the shared fence and CTA synchronization.
        # The pipeline publishes W and controls reuse of the entire W/U slot.
        wu_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_work_slots,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads // cute.arch.WARP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            barrier_storage=storage.wu_barriers.data_ptr(), defer_sync=True,
            name="readout_wu",
        )
        # U completion has its own phase for each reusable work slot.
        u_barriers = pipeline.MbarrierArray(
            barrier_storage=storage.u_barriers.data_ptr(),
            num_stages=self.num_work_slots,
            agent=(
                pipeline.PipelineOp.TmaLoad,
                pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            ),
            name="readout_u",
        )
        # Producers release parameters after computing W. One consumer warp releases
        # them after C scaling. Both groups must finish before the next parameter load.
        param_load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_producer_threads // cute.arch.WARP_SIZE + 1,
            ),
            tx_count=self.raw_param_bytes,
            barrier_storage=storage.param_load_barriers.data_ptr(), defer_sync=True,
            name="readout_param_load",
        )
        state_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_warps,
            ),
            tx_count=self.STATE_R_TILE * self.M_PAD * self.states_dtype.width // 8,
            barrier_storage=storage.state_barriers.data_ptr(), defer_sync=True,
            name="readout_state",
        )
        pipeline.pipeline_init_arrive()
        h_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_consumer_threads
            ),
        )

        ################## prologue ##################
        pipeline.pipeline_init_wait()
        if cutlass.const_expr(self.config.consumer_regs > 0):
            if warp_idx < self.num_consumer_warps:
                cute.arch.setmaxregister_decrease(self.config.consumer_regs)

        ################## producer mainloop ##################
        if warp_idx >= self.num_consumer_warps:
            if cutlass.const_expr(self.config.producer_regs > 0):
                cute.arch.setmaxregister_increase(self.config.producer_regs)
            producer_tidx = tidx - self.num_consumer_threads
            wu_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_work_slots,
            )
            param_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            param_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            while work_idx < work_count:
                work_iter = wu_producer_state.count
                next_work_idx = self.next_work(work_idx, work_iter, num_ctas)

                _, bidx_batch, bidx_chunk, bidx_t_tile, bidx_r_tile = self.decode_work(
                    work_idx, chunk_divisor,
                )

                t_tile_start = bidx_t_tile * self.T_TILE
                chunk_start = bidx_chunk * self.chunk_size
                chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
                num_k_tiles = cute.ceil_div(
                    cutlass.min(t_tile_start + self.T_TILE, chunk_len), self.K_TILE,
                )
                work_slot = wu_producer_state.index

                ################## copy partitions ##################
                gU = cute.local_tile(mU_rlb[None, None, bidx_batch],
                                    (self.R_TILE, self.K_TILE),
                                    (bidx_r_tile, None))

                bGS_sU, bGS_gU = cpasync.tma_partition(
                    tma_atom_u,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sU, 0, 2),
                    gmem_tensor=cute.group_modes(gU, 0, 2),
                )

                if warp_idx == self.producer_load_warp_idx:
                    # The parameter buffer has no readers on the first work item.
                    param_load_pipeline.producer_acquire(
                        param_producer_state, try_acquire_token=(work_iter == 0),
                    )
                    param_barrier_ptr = param_load_pipeline.producer_get_barrier(
                        param_producer_state
                    )
                    self.load_param(
                        tma_atom_c, mC_lmb, sC, self.chunk_tile,
                        (chunk_start + t_tile_start) // self.chunk_tile,
                        bidx_batch, param_barrier_ptr,
                    )
                    self.load_param(
                        tma_atom_k, mK_lmb, sKRaw, self.chunk_size,
                        bidx_chunk, bidx_batch, param_barrier_ptr,
                    )
                    self.load_param(
                        tma_atom_lcr, mLCR_lmb, sLCRRaw, self.chunk_size,
                        bidx_chunk, bidx_batch, param_barrier_ptr,
                    )
                # Each work slot holds all K stages until the consumer finishes
                # the final R subtile.
                wu_pipeline.producer_acquire(
                    wu_producer_state, try_acquire_token=(work_iter < self.num_work_slots),
                )
                if warp_idx == self.producer_load_warp_idx:
                    u_barriers.arrive_and_expect_tx(
                        work_slot, num_k_tiles * u_tma_copy_bytes,
                    )
                    stage_barrier = u_barriers.get_barrier(work_slot)
                    for k_tile in cutlass.range(0, num_k_tiles, 1, unroll=1):
                        u_tile = (chunk_start + k_tile * self.K_TILE) // self.K_TILE
                        cute.copy(
                            tma_atom_u, bGS_gU[None, u_tile],
                            bGS_sU[None, work_slot * self.num_k_stages + k_tile],
                            tma_bar_ptr=stage_barrier,
                        )

                ################## stage parameters and zero W ##################
                # Zero noncausal and padded entries before producers fill the causal triangle.
                sWLinear = cute.make_tensor(
                    storage.sW.data_ptr() + work_slot * w_slot_elements,
                    cute.make_layout(w_slot_elements),
                )
                rWZeros = cute.make_rmem_tensor((self.vecsize,), Float32)
                rWZeros.fill(Float32(0.0))
                for zero_i in cutlass.range_constexpr(cute.ceil_div(
                    w_slot_elements, self.vecsize * self.num_producer_threads,
                )):
                    zero_vec = producer_tidx + zero_i * self.num_producer_threads
                    if zero_vec * self.vecsize < w_slot_elements:
                        cute.autovec_copy(rWZeros, cute.local_tile(sWLinear, (self.vecsize,), (zero_vec,)))
                param_load_pipeline.consumer_wait(param_consumer_state)
                cute.arch.fence_view_async_shared()

                for i in cutlass.range_constexpr(self.num_lcr_copy_iters):
                    vec_idx = producer_tidx + i * self.num_producer_threads
                    if vec_idx < self.chunk_size * self.M_PAD // self.vecsize:
                        local_k = vec_idx // (self.M_PAD // self.vecsize)
                        local_vec = vec_idx % (self.M_PAD // self.vecsize)
                        sRawTile = cute.local_tile(sLCRRaw[local_k, None], (self.vecsize,), (local_vec,))
                        sTile = cute.local_tile(sLCR[local_k, None], (self.vecsize,), (local_vec,))
                        rLCR = cute.make_rmem_tensor((self.vecsize,), Float32)
                        cute.autovec_copy(sRawTile, rLCR)
                        cute.autovec_copy(rLCR, sTile)

                for i in cutlass.range_constexpr(self.num_k_copy_iters):
                    vec_idx = producer_tidx + i * self.num_producer_threads
                    if vec_idx < self.chunk_size * self.M_PAD // self.param_vecsize:
                        local_k = vec_idx // (self.M_PAD // self.param_vecsize)
                        local_vec = vec_idx % (self.M_PAD // self.param_vecsize)
                        sRawTile = cute.local_tile(sKRaw[local_k, None], (self.param_vecsize,), (local_vec,))
                        sTile = cute.local_tile(sK[local_k, None], (self.param_vecsize,), (local_vec,))
                        rKRaw = cute.make_rmem_tensor((self.param_vecsize,), self.param_dtype)
                        cute.autovec_copy(sRawTile, rKRaw)
                        cute.autovec_copy(rKRaw, sTile)

                cute.arch.fence_view_async_shared()
                self.producer_barrier.arrive_and_wait()

                ################## compute W ##################
                # Keep the K-stage loop rolled to limit generated code size.
                for k_tile in cutlass.range(self.num_k_stages, unroll=1):
                    if k_tile < num_k_tiles:
                        stage = work_slot * self.num_k_stages + k_tile
                        # W columns follow the K order in which ldmatrix delivers U.T.
                        sW_stage = self.k_permuted(cute.slice_(sW, (None, None, stage)), inverse=True)
                        self.compute_w_stage(
                            sW_stage, sC, sK, sLCR, k_tile, producer_tidx,
                            t_tile_start, chunk_len, log2_e,
                        )

                # Publish the whole item after each warp has ordered every W store.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if producer_tidx % cute.arch.WARP_SIZE == 0:
                    wu_pipeline.producer_commit(wu_producer_state)
                # No producer may overwrite parameters while another still
                # reads them for this work item. Consumer release is separate.
                if next_work_idx < work_count:
                    self.producer_barrier.arrive_and_wait()
                    param_load_pipeline.consumer_release(param_consumer_state)
                wu_producer_state.advance()
                param_producer_state.advance()
                param_consumer_state.advance()
                work_idx = next_work_idx

        ################## consumer prologue ##################
        if warp_idx < self.num_consumer_warps:
            wu_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_work_slots,
            )
            param_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            state_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1,
            )
            state_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1,
            )
            if warp_idx == self.state_load_warp_idx:
                self.load_state(
                    tma_atom_states, mStatesAcc_rmn, sStatesRaw,
                    state_pipeline, state_producer_state,
                    work_idx // (self.num_t_tiles * self.num_r_tiles),
                    work_idx % self.num_r_tiles, 0,
                )
            state_producer_state.advance()

            thr_mma = tiled_mma.get_slice(tidx)
            tCrW = thr_mma.make_fragment_B(thr_mma.partition_B(
                tf32_smem_view(storage.sW, w_smem_layout_staged)))
            # U.T of one R subtile, widened to FP32 for each K tile.
            tCrU = cute.make_rmem_tensor(
                (*thr_mma.partition_shape_A((self.R_SUBTILE, self.K_TILE)), self.num_k_stages),
                Float32,
            )
            tCrU16 = cute.make_rmem_tensor(
                thr_mma.partition_shape_A((self.R_SUBTILE, self.K_TILE)), self.dtype,
            )
            thr_copy_u = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2), self.dtype,
                ),
                tiled_mma,
            ).get_slice(tidx)
            thr_mma_states = tiled_mma_states.get_slice(tidx)
            tCrStatesAcc = thr_mma_states.make_fragment_A(thr_mma_states.partition_A(
                tf32_smem_view(storage.sStatesAcc, states_acc_smem_layout_staged)))
            tCrCScaled = thr_mma_states.make_fragment_B(thr_mma_states.partition_B(
                tf32_smem_view(storage.sCScaled, c_scaled_smem_layout_staged)))

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
            tSS_sStatesAcc = thr_copy_states.partition_D(sStatesAcc)

            ################## consumer mainloop ##################
            while work_idx < work_count:
                work_iter = wu_consumer_state.count
                next_work_idx = self.next_work(work_idx, work_iter, num_ctas)
                chunk_batch, bidx_batch, bidx_chunk, bidx_t_tile, bidx_r_tile = self.decode_work(
                    work_idx, chunk_divisor,
                )

                t_tile_start = bidx_t_tile * self.T_TILE
                chunk_start = bidx_chunk * self.chunk_size
                chunk_len = cutlass.min(self.chunk_size, L - chunk_start)
                num_k_tiles = cute.ceil_div(
                    cutlass.min(t_tile_start + self.T_TILE, chunk_len), self.K_TILE,
                )
                work_slot = wu_consumer_state.index

                ################## output TMA partition ##################
                gH = cute.local_tile(
                    mH_rlb[None, None, bidx_batch],
                    (self.R_SUBTILE, self.chunk_tile),
                    (None, None),
                )
                bSG_sH, bSG_gH = cpasync.tma_partition(
                    tma_atom_h,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sH, 0, 2),
                    gmem_tensor=cute.group_modes(gH, 0, 2),
                )

                ################## scale C for the state term ##################
                param_load_pipeline.consumer_wait(param_consumer_state)
                cute.arch.fence_view_async_shared()
                for i in cutlass.range_constexpr(self.num_c_scale_iters):
                    idx = tidx + i * self.num_consumer_threads
                    local_t = idx // self.M_PAD
                    local_m = idx % self.M_PAD
                    valid_state = local_m < self.M and t_tile_start + local_t < chunk_len
                    if valid_state:
                        scaled = cute.math.exp2(
                            sLCRRaw[t_tile_start + local_t, local_m] * log2_e,
                            fastmath=True,
                        ) * sC[local_t, local_m].to(Float32)
                        sCScaled[local_t, local_m, 0] = scaled
                    else:
                        sCScaled[local_t, local_m, 0] = Float32(0.0)

                cute.arch.fence_view_async_shared()
                # Release the parameters for reuse only if this CTA has another work tile.
                if next_work_idx < work_count:
                    self.consumer_barrier.arrive_and_wait()
                    if warp_idx == self.param_release_warp_idx:
                        param_load_pipeline.consumer_release(param_consumer_state)

                ################## compute 64-column R subtiles ##################
                # Each subtile needs 32 FP32 accumulator registers per consumer thread,
                # regardless of the full R tile width.
                for r_subtile in cutlass.range_constexpr(self.num_r_subtiles):
                    if r_subtile % self.r_subtiles_per_state_tile == 0:
                        state_pipeline.consumer_wait(state_consumer_state)
                        cute.arch.fence_view_async_shared()
                    sStatesRawTile = cute.local_tile(
                        sStatesRaw, (self.R_SUBTILE, self.M_PAD, 1),
                        (r_subtile % self.r_subtiles_per_state_tile, 0, 0),
                    )
                    tSS_sStatesRaw = thr_copy_states.partition_S(sStatesRawTile)
                    rStatesRaw = cute.make_rmem_tensor_like(tSS_sStatesRaw[None, None, None, 0])
                    rStatesAcc = cute.make_rmem_tensor_like(tSS_sStatesAcc[None, None, None, 0])
                    cute.autovec_copy(tSS_sStatesRaw[None, None, None, 0], rStatesRaw)
                    rStatesAcc.store(rStatesRaw.load().to(Float32))
                    cute.autovec_copy(rStatesAcc, tSS_sStatesAcc[None, None, None, 0])

                    # Fence and synchronize converted state writes before WGMMA.
                    # On this CTA's final work tile, this also synchronizes sCScaled writes.
                    cute.arch.fence_view_async_shared()
                    self.consumer_barrier.arrive_and_wait()

                    # Reuse sStatesRaw once its last subtile has been copied out.
                    # The next load can overlap WGMMA, which reads sStatesAcc.
                    if (r_subtile + 1) % self.r_subtiles_per_state_tile == 0:
                        state_consumer_state.advance()
                        if r_subtile + 1 < self.num_r_subtiles:
                            # Load the next state tile for this work item.
                            if warp_idx == self.state_load_warp_idx:
                                self.load_state(
                                    tma_atom_states, mStatesAcc_rmn, sStatesRaw,
                                    state_pipeline, state_producer_state,
                                    chunk_batch, bidx_r_tile, r_subtile + 1,
                                )
                        elif next_work_idx < work_count:
                            # Load the first state tile for this CTA's next work item.
                            if warp_idx == self.state_load_warp_idx:
                                self.load_state(
                                    tma_atom_states, mStatesAcc_rmn, sStatesRaw,
                                    state_pipeline, state_producer_state,
                                    next_work_idx // (self.num_t_tiles * self.num_r_tiles),
                                    next_work_idx % self.num_r_tiles, 0,
                                )
                        state_producer_state.advance()

                    ################## U.T @ W.T WGMMA ##################
                    acc = thr_mma.make_fragment_C(
                        thr_mma.partition_shape_C((self.R_SUBTILE, self.chunk_tile))
                    )
                    # Wait for W and U once per work tile. All R subtiles reuse this slot.
                    if r_subtile == 0:
                        wu_pipeline.consumer_wait(wu_consumer_state)
                        u_barriers.wait(wu_consumer_state.index, wu_consumer_state.phase)
                        cute.arch.fence_view_async_shared()
                    self.widen_u_subtile(thr_copy_u, sU, tCrU16, tCrU, work_slot, r_subtile, num_k_tiles)
                    cute.nvgpu.warpgroup.fence()
                    # K tiles past the work tile hold zeros, so every WGMMA is unconditional.
                    for k_tile in cutlass.range_constexpr(self.num_k_stages):
                        stage = work_slot * self.num_k_stages + k_tile
                        self.gemm(
                            tiled_mma,
                            tf32_rmem_view(tCrU)[None, None, None, k_tile],
                            tCrW[None, None, None, stage],
                            acc,
                            k_tile != 0,
                        )
                        cute.nvgpu.warpgroup.commit_group()

                    ################## accumulate the state term with WGMMA ##################
                    self.gemm(
                        tiled_mma_states,
                        tCrStatesAcc[None, None, None, 0],
                        tCrCScaled[None, None, None, 0],
                        acc,
                        True,
                    )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    ################## release W/U slot ##################
                    if r_subtile + 1 == self.num_r_subtiles:
                        # Release only if a later work tile on this CTA will reuse this slot.
                        crossed_groups = (
                            work_iter % self.work_group + self.num_work_slots
                        ) // self.work_group
                        future_slot_work = (
                            work_idx + self.num_work_slots
                            + crossed_groups * (num_ctas - 1) * self.work_group
                        )
                        if future_slot_work < work_count:
                            self.consumer_barrier.arrive_and_wait()
                            if tidx % cute.arch.WARP_SIZE == 0:
                                wu_pipeline.consumer_release(wu_consumer_state)

                    ################## store the output subtile ##################
                    h_tile = (chunk_start + t_tile_start) // self.chunk_tile
                    self.store_h_epilogue(
                        tiled_mma,
                        tma_atom_h,
                        bSG_sH,
                        bSG_gH[None, bidx_r_tile * self.num_r_subtiles + r_subtile, None],
                        h_store_pipeline,
                        sH,
                        acc,
                        h_tile,
                        tidx,
                        warp_idx,
                    )
                    # All consumer warps must wait for the store warp before reusing sH.
                    self.consumer_barrier.arrive_and_wait()

                wu_consumer_state.advance()
                param_consumer_state.advance()
                work_idx = next_work_idx

    @classmethod
    def r_tile_for(cls, R):
        return cls.R_SUBTILE * min(cls.MAX_R_SUBTILES, math.ceil(R / cls.R_SUBTILE))

    @classmethod
    def shared_memory_bytes(cls, R, chunk_size, config, dtype_bytes, param_dtype_bytes, states_dtype_bytes):
        """Estimate the SharedStorage size, including alignment padding, for config pruning."""
        fp32_bytes = Float32.width // 8
        R_TILE = cls.r_tile_for(R)
        STATE_R_TILE = config.STATE_R_TILE or R_TILE
        chunk_tile = min(cls.T_TILE, chunk_size)
        num_k_stages = math.ceil(chunk_size / cls.K_TILE)
        num_wu_stages = config.work_slots * num_k_stages
        return (
            fp32_bytes * chunk_tile * cls.K_TILE * num_wu_stages                    # sW
            + dtype_bytes * R_TILE * cls.K_TILE * num_wu_stages                     # sU
            + fp32_bytes * (chunk_tile + cls.R_SUBTILE) * cls.M_PAD                # sCScaled, sStatesAcc
            + states_dtype_bytes * STATE_R_TILE * cls.M_PAD                         # sStatesRaw
            + dtype_bytes * chunk_tile * cls.R_SUBTILE                              # sH
            + (chunk_tile + chunk_size) * cls.M_PAD * param_dtype_bytes             # raw C, K
            + chunk_size * cls.M_PAD * fp32_bytes                                   # raw LCR
            + chunk_size * (cls.M_PAD * fp32_bytes + cls.PARAM_PADDING_BYTES)       # sLCR
            + chunk_size * (cls.M_PAD * param_dtype_bytes + cls.PARAM_PADDING_BYTES)  # sK
            + 3 * cls.MMA_ALIGNMENT_BYTES + 6 * cls.SMEM_ALIGNMENT_BYTES            # alignment padding
        )

    @cute.jit
    def next_work(self, work_idx, work_iter, num_ctas):
        # At the end of each work group, skip the groups assigned to other CTAs.
        next_work_idx = work_idx + 1
        if (work_iter + 1) % self.work_group == 0:
            next_work_idx += (num_ctas - 1) * self.work_group
        return next_work_idx

    @cute.jit
    def decode_work(self, work_idx, chunk_divisor):
        chunk_batch = work_idx // (self.num_t_tiles * self.num_r_tiles)
        bidx_tile = work_idx % (self.num_t_tiles * self.num_r_tiles)
        bidx_t_tile = bidx_tile // self.num_r_tiles
        bidx_r_tile = bidx_tile % self.num_r_tiles
        # chunk_divisor overloads divmod to use precomputed division parameters
        # for nchunks, which is only known at runtime.
        bidx_batch, bidx_chunk = divmod(chunk_batch, chunk_divisor)
        return chunk_batch, bidx_batch, bidx_chunk, bidx_t_tile, bidx_r_tile

    @cute.jit
    def make_param_tma(self, mX: cute.Tensor, smem_layout, tile_rows):
        mX_lmb = cute.make_tensor(
            mX.iterator,
            cute.make_layout(
                (mX.shape[1], self.M, mX.shape[0]),
                stride=(mX.stride[1], mX.stride[2], mX.stride[0]),
            ),
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX_lmb, smem_layout,
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
        bGS_sX, bGS_gX = cpasync.tma_partition(
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
        sStatesRaw: cute.Tensor, state_pipeline, state_producer_state,
        chunk_batch, bidx_r_tile, r_subtile,
    ):
        # sStatesRaw is unused on the first load. Before reusing it, all
        # consumer threads synchronize after reading the previous tile.
        state_pipeline.producer_acquire(
            state_producer_state, try_acquire_token=Boolean(True),
        )
        state_barrier_ptr = state_pipeline.producer_get_barrier(state_producer_state)
        gStatesRaw = cute.local_tile(
            mStatesAcc_rmn[None, None, chunk_batch],
            (self.STATE_R_TILE, self.M_PAD),
            ((bidx_r_tile * self.num_r_subtiles + r_subtile) // self.r_subtiles_per_state_tile, 0),
        )
        bGS_sStatesRaw, bGS_gStatesRaw = cpasync.tma_partition(
            tma_atom_states,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sStatesRaw, 0, 2),
            gmem_tensor=cute.group_modes(gStatesRaw, 0, 2),
        )
        cute.copy(
            tma_atom_states, bGS_gStatesRaw[None], bGS_sStatesRaw[None, 0],
            tma_bar_ptr=state_barrier_ptr,
        )

    @cute.jit
    def compute_w_stage(
        self,
        sW_stage: cute.Tensor,  # (chunk_tile, K_TILE)
        sC: cute.Tensor,        # (chunk_tile, M_PAD)
        sK: cute.Tensor,        # (chunk_size, M_PAD)
        sLCR: cute.Tensor,      # (chunk_size, M_PAD), fp32
        k_tile,
        producer_tidx,
        t_tile_start,
        chunk_len,
        log2_e,
    ):
        """Fill a causal W tile whose other entries are already zero."""
        # Groups of four threads share a K column and start on different T rows.
        num_threads_per_group = self.num_w_threads_T * self.num_w_threads_K
        local_k_pair = (producer_tidx % num_threads_per_group) // self.num_w_threads_T
        local_t_base = (
            (producer_tidx // num_threads_per_group) * self.num_w_threads_T
            + producer_tidx % self.num_w_threads_T
        )
        # Pair low and high K columns to balance work across the causal triangle.
        for k_side in cutlass.range_constexpr(self.w_k_columns_per_thread):
            local_k = local_k_pair
            if k_side != 0:
                local_k = self.K_TILE - 1 - local_k_pair
            k = k_tile * self.K_TILE + local_k
            rK = cute.make_rmem_tensor((self.M,), Float32)
            rLCRK = cute.make_rmem_tensor((self.M,), Float32)
            rKRaw = cute.make_rmem_tensor((self.M,), self.param_dtype)
            cute.autovec_copy(
                cute.local_tile(sK[k, None], (self.M,), (0,)),
                rKRaw,
            )
            rK.store(rKRaw.load().to(Float32))
            cute.autovec_copy(
                cute.local_tile(sLCR[k, None], (self.M,), (0,)),
                rLCRK,
            )

            for i in cutlass.range_constexpr(self.num_w_row_iters):
                local_t = local_t_base + i * self.w_rows_per_iter
                if local_t < self.chunk_tile:
                    t = t_tile_start + local_t
                    w_val = Float32(0.0)
                    # Check causality before exp to avoid overflow in noncausal entries.
                    if t < chunk_len and k <= t:
                        # Load four values at a time and accumulate in increasing m order.
                        for m_group in cutlass.range_constexpr(self.M // self.vecsize):
                            rC = cute.make_rmem_tensor((self.vecsize,), self.param_dtype)
                            rLCRT = cute.make_rmem_tensor((self.vecsize,), Float32)
                            cute.autovec_copy(
                                cute.local_tile(sC[local_t, None], (self.vecsize,), (m_group,)), rC,
                            )
                            cute.autovec_copy(
                                cute.local_tile(sLCR[t, None], (self.vecsize,), (m_group,)),
                                rLCRT,
                            )
                            for m_offset in cutlass.range_constexpr(self.vecsize):
                                m = m_group * self.vecsize + m_offset
                                w_val += (
                                    rC[m_offset].to(Float32)
                                    * rK[m]
                                    * cute.math.exp2(
                                        (rLCRT[m_offset] - rLCRK[m]) * log2_e,
                                        fastmath=True,
                                    )
                                )
                        sW_stage[local_t, local_k] = w_val

    @cute.jit
    def widen_u_subtile(self, thr_copy_u, sU, tCrU16, tCrU, work_slot, r_subtile, num_k_tiles):
        """Widen one R subtile of U.T from its 16-bit TMA tile into the FP32 register A operand."""
        tCsU = thr_copy_u.partition_S(self.k_permuted(cute.local_tile(
            sU, (self.R_SUBTILE, self.K_TILE, self.num_wu_stages), (r_subtile, 0, 0),
        )))
        tCrU16_copy = thr_copy_u.retile(tCrU16)
        for k_tile in cutlass.range_constexpr(self.num_k_stages):
            cute.copy(
                thr_copy_u, tCsU[None, None, None, work_slot * self.num_k_stages + k_tile], tCrU16_copy,
            )
            dst = tCrU[None, None, None, k_tile]
            for i in cutlass.range_constexpr(cute.size(dst)):
                if cutlass.const_expr(k_tile == 0):
                    dst[i] = tCrU16[i].to(Float32)
                else:
                    # Stale stages past the work tile may hold any bits; zero them.
                    dst[i] = cutlass.select_(k_tile < num_k_tiles, tCrU16[i].to(Float32), Float32(0.0))

    def k_permuted(self, tensor, inverse=False):
        """Reorder K within each 8-wide WGMMA step so ldmatrix.trans yields TF32 A fragments.

        Position p0 + 4 * h holds k = 2 * p0 + h, so each thread's two TF32 K values are
        adjacent 16-bit elements of a transposed 8x8 matrix.
        """
        k_blocks = self.K_TILE // 8
        if inverse:  # k -> position, for writing W
            perm = cute.make_layout((2, 4, k_blocks), stride=(4, 1, 8))
        else:  # position -> k, for reading U
            perm = cute.make_layout((4, 2, k_blocks), stride=(2, 1, 8))
        return cute.composition(tensor, (tensor.shape[0], perm, *tensor.shape[2:]))

    @cute.jit
    def store_h_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_h: cute.CopyAtom,
        bSG_sH: cute.Tensor,
        bSG_gH: cute.Tensor,
        h_store_pipeline,
        sH: cute.Tensor,
        acc: cute.Tensor,
        h_tile,
        tidx,
        warp_idx,
    ):
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                LayoutEnum.COL_MAJOR.is_m_major_c(),
                num_matrices=4,
            ),
            self.dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C(copy_atom_C, tiled_mma)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sH = thr_copy_r2s.partition_D(sH)

        cvt_copy_r2s(tiled_copy_r2s, acc, tRS_sH[None, None, None, 0])

        cute.arch.fence_view_async_shared()
        self.consumer_barrier.arrive_and_wait()

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
def _compile_chunk_fused_readout_fwd(dtype, R, M, chunk_size, config, states_dtype, param_dtype):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_u = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    # C advertises vector alignment and K advertises word alignment.
    # The remaining tensors retain scalar stride promises.
    div_c = 128 // param_dtype.width
    div_k = cutlass.Uint32.width // param_dtype.width
    fake_c = make_fake_tensor(param_dtype, (B_sym, L_sym, M), divisibility=div_c)
    fake_kappa = make_fake_tensor(param_dtype, (B_sym, L_sym, M), divisibility=div_k)
    fake_states_acc = make_fake_tensor(states_dtype, (B_sym, nchunks_sym, R, M), divisibility=1)
    fake_lcr = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=1)
    fake_h = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=1)
    return cute.compile(
        ChunkFusedReadoutFwd(
            dtype, R=R, M=M, chunk_size=chunk_size, config=config, states_dtype=states_dtype,
            param_dtype=param_dtype,
        ),
        fake_u,
        fake_c,
        fake_kappa,
        fake_states_acc,
        fake_lcr,
        fake_h,
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _resource_config(config):
    """Normalize runtime CTA limits so they share a compiled kernel."""
    return replace(config, ctas_per_sm=1, max_ctas=1)


@lru_cache(maxsize=None)
def _resolve_config(config, device_index):
    """Resolve CTA limits once per device without retaining tensors."""
    if config.max_ctas is not None:
        return config
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return replace(config, max_ctas=sm_count * config.ctas_per_sm)


def _prepare(u, c, kappa, states_acc, log_cum_rho, h, *, chunk_size, config):
    return _compile_chunk_fused_readout_fwd(
        torch2cute_dtype_map[u.dtype], u.shape[-1], c.shape[-1], chunk_size,
        _resource_config(config), torch2cute_dtype_map[states_acc.dtype],
        torch2cute_dtype_map[c.dtype],
    )


@autotune(
    configs=persistent_configs(),
    key=["chunk_size", ("u", 2), ("c", 2)],
    prune_configs_by={"early_config_prune": prune_persistent_configs},
    cache_results=False,
)
def _launch(u, c, kappa, states_acc, log_cum_rho, h, *, chunk_size, config):
    kernel = _prepare(
        u, c, kappa, states_acc, log_cum_rho, h, chunk_size=chunk_size, config=config,
    )
    kernel(u, c, kappa, states_acc, log_cum_rho, h, config.max_ctas)


_kernel_cache = {}
_launch_lock = threading.Lock()


def _frac_chunk_fused_readout_fwd(
    u, c, kappa, states_acc, log_cum_rho, chunk_size, config=None,
):
    B, L, R = u.shape
    h = torch.empty((B, L, R), device=u.device, dtype=u.dtype)

    if B == 0 or L == 0:
        return h
    if u.dtype not in (torch.float16, torch.bfloat16):
        raise AssertionError(f"persistent readout expects 16-bit inputs, got {u.dtype}")
    if c.dtype != kappa.dtype or c.dtype not in (u.dtype, torch.float32):
        raise AssertionError("c and kappa must share the u dtype or float32")
    if states_acc.dtype not in (u.dtype, torch.float32):
        raise AssertionError(f"states_acc must be input dtype or float32, got {states_acc.dtype}")
    if log_cum_rho.dtype != torch.float32:
        raise AssertionError("log_cum_rho must be float32")
    if config is not None:
        config = _resolve_config(config, u.device.index)
        kernel = _prepare(
            u, c, kappa, states_acc, log_cum_rho, h,
            chunk_size=chunk_size, config=config,
        )
        kernel(u, c, kappa, states_acc, log_cum_rho, h, config.max_ctas)
        return h

    # The public boundary validates shapes and aligns all five inputs.
    key = (u.device.index, u.dtype, states_acc.dtype, c.dtype, R, c.shape[-1], chunk_size)
    cached = _kernel_cache.get(key)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up persistent autotuning before CUDA graph capture")
        with _launch_lock:
            cached = _kernel_cache.get(key)
            if cached is None:
                _launch(u, c, kappa, states_acc, log_cum_rho, h, chunk_size=chunk_size)
                selected = _launch.best_config.kwargs["config"]
                kernel = _prepare(
                    u, c, kappa, states_acc, log_cum_rho, h,
                    chunk_size=chunk_size, config=selected,
                )
                _kernel_cache[key] = (kernel, selected.max_ctas)
                return h
    kernel, max_ctas = cached
    kernel(u, c, kappa, states_acc, log_cum_rho, h, max_ctas)
    return h


__all__ = ["_frac_chunk_fused_readout_fwd"]
