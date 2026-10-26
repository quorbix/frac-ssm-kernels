import math
from dataclasses import dataclass

import torch
import cutlass
from cutlass import cute
import cutlass.pipeline as pipeline

from cutlass import Float16, Float32, TFloat32
import cuda.bindings.driver as cuda
from cutlass.tensor_utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils


from frac_ssm.ops.cute.utils import (
    MmaTF32Op, make_fake_tensor, predicate_axis, tf32_rmem_view, tf32_smem_view, torch2cute_dtype_map,
)
from frac_ssm.ops.cute.cache_utils import jit_cache


@dataclass(frozen=True)
class CpAsyncParams:
    vecsize: int
    num_threads_M: int
    num_threads_L: int
    num_copy_threads: int


class ChunkStateFwd:
    def __init__(self, dtype: type[cutlass.Numeric], R: int, M: int, chunk_size: int, states_dtype: type[cutlass.Numeric],
                 kappa_dtype: type[cutlass.Numeric] | None = None):
        self.dtype = dtype
        self.kappa_dtype = dtype if kappa_dtype is None else kappa_dtype
        self.states_dtype = states_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size

        self.chunk_tile = min(chunk_size, 128)
        self.num_chunk_tiles = chunk_size // self.chunk_tile

        self.vecsize = math.gcd(128 // dtype.width, M)
        self.num_warps_M = M // self.vecsize
        self.rows_per_lane = self.chunk_tile // cute.arch.WARP_SIZE

        self.num_warp_groups = max(1, math.ceil(self.num_warps_M / 4))
        self.num_threads = 128 * self.num_warp_groups
        self.num_load_threads = cute.arch.WARP_SIZE * self.num_warps_M

        assert chunk_size in (32, 64) or chunk_size % 128 == 0, (
            f"chunk_size must be 32, 64, or a multiple of 128, got {chunk_size}"
        )
        assert dtype.width == 16, (
            f"dtype must be 16-bit (F16/BF16), got {dtype} (width {dtype.width})"
        )
        assert self.kappa_dtype in (dtype, Float32), (
            f"kappa dtype must be {dtype} or Float32, got {self.kappa_dtype}"
        )
        assert M == 16, f"M=16 is the only supported configuration, got {M}"
        assert R % 64 == 0, f"R must be a multiple of 64, got {R}"

        # U is widened into TF32 register fragments in groups of K-blocks.
        self.num_k_blocks = self.chunk_tile // 8
        self.k_blocks_per_group = 2
        assert self.num_k_blocks % self.k_blocks_per_group == 0

        # The 128-bit vectorized load has different number of elements for LCR (fp32) and K.
        self.cp_lcr = self._cp_params(Float32.width)
        self.cp_k = self._cp_params(self.kappa_dtype.width)


    def _cp_params(self, element_bits: int) -> CpAsyncParams:
        """
        Parameters for cp.async based on the number of bits of each element.
        """
        elements_per_copy = 128 // element_bits
        threads_per_row = self.M // elements_per_copy
        rows_per_warp = cute.arch.WARP_SIZE // threads_per_row
        num_cta_warps = self.num_threads // cute.arch.WARP_SIZE
        num_copy_warps = max(
            num_warps
            for num_warps in range(1, num_cta_warps + 1)
            if self.chunk_tile % (rows_per_warp * num_warps) == 0
        )
        rows_per_copy = rows_per_warp * num_copy_warps
        num_copy_threads = cute.arch.WARP_SIZE * num_copy_warps
        return CpAsyncParams(elements_per_copy, threads_per_row, rows_per_copy, num_copy_threads)


    @cute.jit
    def __call__(self,
        mX: cute.Tensor,    # (B, L, R)
        mK: cute.Tensor,    # (B, L, M)
        mLCR: cute.Tensor,  # (B, L, M) # fp32
        mY: cute.Tensor,    # (B, nchunks, R, M)
        stream: cuda.CUstream,
    ):

        B, L = mX.shape[:2]
        nchunks = mY.shape[1]

        ################## gmem to smem copy ##################
        copy_atom_g2s_LCR = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mLCR.element_type, num_bits_per_copy=128
        )
        copy_atom_g2s_K = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mK.element_type, num_bits_per_copy=128
        )

        thr_layout_g2s_LCR = cute.make_ordered_layout((self.cp_lcr.num_threads_M, self.cp_lcr.num_threads_L), order=(0, 1))
        val_layout_g2s_LCR = cute.make_ordered_layout((self.cp_lcr.vecsize, 1), order=(0, 1))
        tiled_copy_g2s_LCR = cute.make_tiled_copy_tv(copy_atom_g2s_LCR, thr_layout_g2s_LCR, val_layout_g2s_LCR)

        thr_layout_g2s_K = cute.make_ordered_layout((self.cp_k.num_threads_M, self.cp_k.num_threads_L), order=(0, 1))
        val_layout_g2s_K = cute.make_ordered_layout((self.cp_k.vecsize, 1), order=(0, 1))
        tiled_copy_g2s_K = cute.make_tiled_copy_tv(copy_atom_g2s_K, thr_layout_g2s_K, val_layout_g2s_K)

        ################## smem to rmem ##################
        num_copy_bits_LCR = min(128, self.vecsize * mLCR.element_type.width)
        num_copy_bits_K = min(128, self.vecsize * mK.element_type.width)

        copy_atom_s2r_LCR = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mLCR.element_type, num_bits_per_copy=num_copy_bits_LCR,
        )
        copy_atom_s2r_K = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mK.element_type, num_bits_per_copy=num_copy_bits_K,
        )
        thr_layout_s2r = cute.make_ordered_layout((self.num_warps_M, cute.arch.WARP_SIZE), order=(1, 0))
        val_layout_s2r = cute.make_ordered_layout((self.vecsize, self.rows_per_lane), order=(0, 1))
        tiled_copy_s2r_LCR = cute.make_tiled_copy_tv(
            copy_atom_s2r_LCR, thr_layout_s2r, val_layout_s2r
        )
        tiled_copy_s2r_K = cute.make_tiled_copy_tv(copy_atom_s2r_K, thr_layout_s2r, val_layout_s2r)



        ################## SMEM allocations ##################
        mma_tiler_mnk = (self.R, self.M, self.chunk_tile)

        x_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, mma_tiler_mnk, self.dtype, num_stages=1
        )
        x_smem_layout = cute.slice_(x_smem_layout_staged, (None, None, 0))

        lcr_smem_layout = cute.make_ordered_layout((self.M, self.chunk_tile), order=(0, 1))

        scale_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, mma_tiler_mnk, TFloat32, num_stages=1
        )

        y_smem_layout = cute.make_ordered_layout((self.R, self.M), order=(1, 0))


        @cute.struct
        class SharedStorage:
            x_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sLCREnd: cute.struct.Align[
                cute.struct.MemRange[mLCR.element_type, self.M],
                16
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[mLCR.element_type, cute.cosize(lcr_smem_layout)],
                16
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(lcr_smem_layout)],
                16
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(scale_smem_layout_staged)],
                1024
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[mX.element_type, cute.cosize(x_smem_layout_staged)],
                128
            ]
            sY: cute.struct.Align[
                cute.struct.MemRange[mY.element_type, cute.cosize(y_smem_layout)],
                128
            ]
        self.shared_storage = SharedStorage


        ################## Tensor views ##################
        # we have to create a new view for the inputs as tma expects the b and n
        # dimensions to be last ones.
        mX_rlb = cute.make_tensor(
            mX.iterator,
            cute.make_layout(
                (self.R, L, B),
                stride=(mX.stride[2], mX.stride[1], mX.stride[0]),
            ),
        )

        mY_rmbn = cute.make_tensor(
            mY.iterator,
            cute.make_layout(
                (self.R, self.M, B, nchunks),
                stride=(mY.stride[2], mY.stride[3], mY.stride[0], mY.stride[1]),
            ),
        )

        # We don't load this using TMA. However, producing the WGMMA B operand
        # needs M (N in GEMM) and L (K in GEMM) in this order.
        mK_bml = cute.make_tensor(
            mK.iterator,
            cute.make_layout((B, self.M, L), stride=(mK.stride[0],  mK.stride[2], mK.stride[1])),
        )
        # LCR isn't fed to WGMMA, but it uses the same order so it can share the
        # K's partitioning in the scale's computation.
        mLCR_bml = cute.make_tensor(
            mLCR.iterator,
            cute.make_layout((B, self.M, L), stride=(mLCR.stride[0], mLCR.stride[2], mLCR.stride[1])),
        )

        ################## TMA setup ##################
        x_cta_tiler = (self.R, self.chunk_tile)
        y_cta_tiler = (self.R, self.M)

        tma_atom_x, tma_tensor_x = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX_rlb, x_smem_layout, x_cta_tiler
        )

        tma_atom_y, tma_tensor_y = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mY_rmbn, y_smem_layout, y_cta_tiler
        )


        ################## WGMMA ##################
        mma_op = MmaTF32Op(
            instruction_shape=(64, self.M, 8),
            a_src=cute.nvgpu.warpgroup.OperandSource.RMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        self.kernel(tma_atom_x, tma_tensor_x, mK_bml, mLCR_bml,
                    tma_atom_y, tma_tensor_y,
                    tiled_copy_g2s_LCR, tiled_copy_g2s_K,
                    tiled_copy_s2r_LCR, tiled_copy_s2r_K,
                    tiled_mma,
                    x_smem_layout_staged, lcr_smem_layout,
                    scale_smem_layout_staged, y_smem_layout).launch(
            grid=[B, nchunks, 1],
            block=[self.num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_rlb: cute.Tensor, # (R, L, B)
        mK_bml: cute.Tensor, # (B, M, L)
        mLCR_bml: cute.Tensor, # (B, M, L)
        tma_atom_y: cute.CopyAtom,
        mY_rmbn: cute.Tensor, # (R, M, B, nchunks)
        tiled_copy_g2s_LCR: cute.TiledCopy,
        tiled_copy_g2s_K: cute.TiledCopy,
        tiled_copy_s2r_LCR: cute.TiledCopy,
        tiled_copy_s2r_K: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged, # (R, chunk_tile, PIPE)
        lcr_smem_layout, # (M, chunk_tile)
        scale_smem_layout_staged, # (M, chunk_tile, PIPE) WGMMA B K-major, FP32
        y_smem_layout, # (R, M)
    ):
        """
        X is loaded with TMA while K and LCR use cp.async, allowing their loads to
        overlap with the X TMA load. Y is stored with TMA.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_chunk, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_y)

        L = mK_bml.shape[2]
        chunk_start = bidx_chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        mK_ml   = mK_bml[bidx_batch, None, None]   # (M, L)
        mLCR_ml = mLCR_bml[bidx_batch, None, None] # (M, L)
        idML = cute.make_identity_tensor(mK_ml.shape)

        gX = cute.local_tile(mX_rlb[None, None, bidx_batch], (self.R, self.chunk_tile), (None, None)) # (R, chunk_tile, RestR, RestL/chunk_tile)
        gY = cute.local_tile(mY_rmbn[None, None, bidx_batch, bidx_chunk], (self.R, self.M), (None, None)) # (R, M, RestR, RestM)


        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        x_load_mbar_ptr = storage.x_load_mbar_ptr.data_ptr()
        sLCREnd = storage.sLCREnd.get_tensor(cute.make_layout(self.M))
        sLCR = storage.sLCR.get_tensor(lcr_smem_layout, swizzle=cute.make_swizzle(3, 4, 3))
        sK = storage.sK.get_tensor(lcr_smem_layout, swizzle=cute.make_swizzle(3, 4, 3))
        sScale = storage.sScale.get_tensor(scale_smem_layout_staged.outer, swizzle=scale_smem_layout_staged.inner)
        sScale_mma = tf32_smem_view(storage.sScale, scale_smem_layout_staged)
        sX = storage.sX.get_tensor(x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner)
        sY = storage.sY.get_tensor(y_smem_layout)


        # we don't multicast here so cta_coord and cta_layout are simple
        bGS_sX, bGS_gX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX, 0, 2),
            gmem_tensor=cute.group_modes(gX, 0, 2),
        )
        bGS_sX_pipe = bGS_sX[None, 0]
        x_tma_copy_bytes = cutlass.const_expr(cute.size_in_bytes(self.dtype, cute.slice_(x_smem_layout_staged, (None, None, 0))))


        thr_mma = tiled_mma.get_slice(tidx)
        acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((self.R, self.M)))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sScale_mma))
        tCsX = thr_mma.partition_A(sX)

        thr_copy_g2s_LCR   = tiled_copy_g2s_LCR.get_slice(tidx)
        thr_copy_g2s_K     = tiled_copy_g2s_K.get_slice(tidx)

        # LCR
        thr_copy_s2r_LCR = tiled_copy_s2r_LCR.get_slice(tidx)
        tLCRsLCR = thr_copy_s2r_LCR.partition_S(sLCR)

        # K is loaded from s to r, and the FP32 scale is written from r to the WGMMA B tile
        thr_copy_s2r_K   = tiled_copy_s2r_K.get_slice(tidx)
        tKsK_s2r = thr_copy_s2r_K.partition_S(sK)
        tSsS_r2s = thr_copy_s2r_K.partition_D(cute.slice_(sScale, (None, None, 0)))

        # Load the end of chunk's decay once
        if tidx < self.M:
            sLCREnd[tidx] = mLCR_ml[tidx, chunk_start + chunk_len - 1]

        for k_tile in cutlass.range_constexpr(self.num_chunk_tiles):
            tile_idx = bidx_chunk * self.num_chunk_tiles + k_tile

            gK   = cute.local_tile(mK_ml,   (self.M, self.chunk_tile), (0, tile_idx))
            gLCR = cute.local_tile(mLCR_ml, (self.M, self.chunk_tile), (0, tile_idx))
            cML = cute.local_tile(idML,   (self.M, self.chunk_tile), (0, tile_idx))


            # TMA load for X. Only one thread from warp 0 issues
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(x_load_mbar_ptr, 1)
                    cute.arch.mbarrier_expect_tx(x_load_mbar_ptr, x_tma_copy_bytes)

            # All threads should run this
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            if warp_idx == 0:
                cute.copy(tma_atom_x, bGS_gX[None, 0, tile_idx], bGS_sX_pipe, tma_bar_ptr=x_load_mbar_ptr)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(x_load_mbar_ptr)

            # Loading LCR and Kappa using G2S. Only the active threads participate
            if tidx < self.cp_lcr.num_copy_threads:
                tLCRgLCR    = thr_copy_g2s_LCR.partition_S(gLCR)
                tLCRsLCR_copy_view = thr_copy_g2s_LCR.partition_D(sLCR)
                tLCRpLCR    = predicate_axis(thr_copy_g2s_LCR.partition_S(cML), L, axis=1)
                cute.copy(tiled_copy_g2s_LCR, tLCRgLCR, tLCRsLCR_copy_view, pred=tLCRpLCR)

            if tidx < self.cp_k.num_copy_threads:
                tKgK    = thr_copy_g2s_K.partition_S(gK)
                tKsK_copy_view = thr_copy_g2s_K.partition_D(sK)
                tKpK    = predicate_axis(thr_copy_g2s_K.partition_S(cML), L, axis=1)
                cute.copy(tiled_copy_g2s_K, tKgK, tKsK_copy_view, pred=tKpK)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # flow is smem -> rmem (compute scale) -> smem
            if tidx < self.num_load_threads:
                tKcML = thr_copy_s2r_K.partition_S(cML)

                tLCRrLCR = cute.make_rmem_tensor_like(tLCRsLCR)
                tKrK = cute.make_rmem_tensor_like(tKsK_s2r)
                tSrS = cute.make_rmem_tensor_like(tKsK_s2r, Float32)

                cute.autovec_copy(tLCRsLCR, tLCRrLCR)
                cute.autovec_copy(tKsK_s2r, tKrK)

                # The formula is scale = exp(lcr_end - log_cum_rho) * kappa
                num_elems = cute.size(tKcML, mode=[0])
                for i in cutlass.range_constexpr(num_elems):
                    coord = tKcML[i, 0, 0]
                    m_idx = coord[0]
                    l_idx = coord[1]
                    if cute.elem_less(l_idx, L):
                        tSrS[i, 0, 0] = cute.exp(sLCREnd[m_idx] - tLCRrLCR[i, 0, 0]) * tKrK[i, 0, 0].to(Float32)
                    else:
                        tSrS[i, 0, 0] = Float32(0.0)

                cute.autovec_copy(tSrS, tSsS_r2s)

            # Wait for TMA load to finish
            cute.arch.mbarrier_wait(x_load_mbar_ptr, phase=0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # Accumulate this reduction tile into the running acc
            self.gemm_widen_a(
                tiled_mma,
                tCsX[None, None, None, 0],
                tCrB[None, None, None, 0],
                acc,
                k_tile != 0,  # accumulate across iters (False on first iter so acc=zero+gemm)
            )


        # Writing the results
        # Select the actual store atom from sY's layout and dtype
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(LayoutEnum.from_tensor(sY), elem_ty_d=self.states_dtype, elem_ty_acc=Float32)
        # stmatrix.x4 matches the eight values per thread in each WGMMA C tile
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                LayoutEnum.from_tensor(sY).is_m_major_c(),
                num_matrices=4,
            ),
            Float16, # Used for Layout only. Not related the output dtype
        )

        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sY       = thr_copy_r2s.partition_D(sY)
        tRS_rAcc     = tiled_copy_r2s.retile(acc)

        # convert to states_dtype
        tRS_rY_out = cute.make_rmem_tensor(tRS_rAcc.shape, self.states_dtype)
        tRS_rY_out.store(tRS_rAcc.load().to(self.states_dtype))
        cute.copy(tiled_copy_r2s, tRS_rY_out, tRS_sY)

        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()


        # TMA write from smem to gmem
        bSG_sY, bSG_gY = cute.nvgpu.cpasync.tma_partition(
            tma_atom_y,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sY, 0, 2),
            gmem_tensor=cute.group_modes(gY, 0, 2),
        )
        if warp_idx == 0:
            # TMA stores use bulk async groups instead of an mbarrier
            cute.copy(tma_atom_y, bSG_sY, bSG_gY[None, 0, 0])
            cute.arch.cp_async_bulk_commit_group()

        cute.arch.cp_async_bulk_wait_group(0, read=True)


    @cute.jit
    def gemm_widen_a(self, tiled_mma, sA, B, C, accumulate_initial: cutlass.Constexpr):
        """One pass of K-tile TF32 WGMMA with A widened from 16-bit smem into FP32 registers.

        K-blocks are converted one group at a time. `accumulate_initial` selects whether the
        first WGMMA accumulates into C (True after the first chunk tile) or overwrites it
        (False on the very first K-tile).
        """
        group = self.k_blocks_per_group
        buffer = []
        for _ in cutlass.range_constexpr(group):
            # Compact, so the register order is the MMA fragment order rather than the smem order.
            buffer.append(cute.make_rmem_tensor(sA[None, None, 0].shape, Float32))

        for group_idx in cutlass.range_constexpr(self.num_k_blocks // group):
            for j in cutlass.range_constexpr(group):
                src = sA[None, None, group_idx * group + j]
                for i in cutlass.range_constexpr(cute.size(src)):
                    buffer[j][i] = src[i].to(Float32)

            cute.nvgpu.warpgroup.fence()
            for j in cutlass.range_constexpr(group):
                k_block_idx = group_idx * group + j
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, accumulate_initial or k_block_idx != 0)
                cute.gemm(tiled_mma, C, tf32_rmem_view(buffer[j]), B[None, None, k_block_idx], C)
            cute.nvgpu.warpgroup.commit_group()
            # The buffer is rewritten next, so its WGMMAs must have retired.
            cute.nvgpu.warpgroup.wait_group(0)

class ChunkStateBwd:
    """Backward pass of the per-chunk state GEMM.

    Per (batch, chunk) CTA, with ``decay[t, m] = exp(lcr_end[m] - lcr[t, m])``:

        scale[t, m]              = decay[t, m] * K[t, m]
        dX[t, r]                 = sum_m dY[r, m] * scale[t, m]  # WGMMA as dX.T (R_TILE, T_TILE, K=M)
        dScale[t, m]             = sum_r X[t, r] * dY[r, m]      # WGMMA (T_TILE, M, K=R_TILE)
        dK[t, m]                 = dScale[t, m] * decay[t, m]
        dLCR[t, m]               = -dScale[t, m] * scale[t, m]   # t < chunk_len - 1
        dLCR[chunk_len - 1, m]   = sum_{t < chunk_len - 1} dScale[t, m] * scale[t, m]
    """

    T_TILE = 64  # WGMMA N for dX.T, WGMMA M for dScale; rows of the chunk per t-tile
    R_TILE = 64  # WGMMA M for dX.T, WGMMA K for dScale

    def __init__(self, dtype: type[cutlass.Numeric], R: int, M: int, chunk_size: int,
                 states_dtype: type[cutlass.Numeric], kappa_dtype: type[cutlass.Numeric] | None = None):
        self.dtype = dtype
        self.kappa_dtype = dtype if kappa_dtype is None else kappa_dtype
        self.states_dtype = states_dtype
        self.R = R
        self.M = M
        self.chunk_size = chunk_size

        # Rows of the chunk per t_tile iteration and the X/dX TMA tile height. It is smaller than
        # the T_TILE-row WGMMA tile only for chunk_size=32, where the rest is padded.
        self.chunk_tile = min(chunk_size, self.T_TILE)
        self.num_chunk_tiles = chunk_size // self.chunk_tile

        self.num_r_tiles = R // self.R_TILE
        self.num_x_tiles = self.num_chunk_tiles * self.num_r_tiles

        self.num_x_stages = min(2, self.num_r_tiles)

        self.num_threads = 128

        self.rows_per_pass = self.num_threads // M
        self.rows_per_lane = self.T_TILE // self.rows_per_pass

        # 128-bit cp.async thread layouts based on the dtype.
        self.dY_vecsize = math.gcd(128 // states_dtype.width, M)
        self.num_dY_threads_M = M // self.dY_vecsize
        self.num_dY_threads_R = self.num_threads // self.num_dY_threads_M

        self.lcr_vecsize = math.gcd(128 // Float32.width, M)
        self.num_lcr_threads_M = M // self.lcr_vecsize
        self.num_lcr_threads_T = self.num_threads // self.num_lcr_threads_M
        self.k_vecsize = math.gcd(128 // self.kappa_dtype.width, M)
        self.num_k_threads_M = M // self.k_vecsize
        self.num_k_threads_T = self.num_threads // self.num_k_threads_M

        assert chunk_size in (32, 64) or chunk_size % 128 == 0, (
            f"chunk_size must be 32, 64, or a multiple of 128, got {chunk_size}"
        )
        assert dtype.width == 16, f"dtype must be 16-bit (F16/BF16), got {dtype} (width {dtype.width})"
        assert M == 16, f"M=16 is the only supported configuration, got {M}"
        assert R % self.R_TILE == 0, f"R must be a multiple of {self.R_TILE}, got {R}"
        assert self.R_TILE % self.num_dY_threads_R == 0
        assert self.T_TILE % self.num_lcr_threads_T == 0
        assert self.T_TILE % self.num_k_threads_T == 0

    @cute.jit
    def __call__(self,
        mdY: cute.Tensor,   # (B, nchunks, R, M)
        mX: cute.Tensor,    # (B, L, R)
        mK: cute.Tensor,    # (B, L, M)
        mLCR: cute.Tensor,  # (B, L, M) # fp32
        mdX: cute.Tensor,   # (B, L, R)
        mdK: cute.Tensor,   # (B, L, M)
        mdLCR: cute.Tensor, # (B, L, M) # fp32
        stream: cuda.CUstream,
    ):

        B, L = mX.shape[:2]
        nchunks = mdY.shape[1]

        ################## gmem to smem copy ##################
        copy_atom_g2s_LCR = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), mLCR.element_type, num_bits_per_copy=128)
        copy_atom_g2s_K = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), mK.element_type, num_bits_per_copy=128)
        copy_atom_g2s_dY = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=128)

        thr_layout_g2s_LCR = cute.make_ordered_layout((self.num_lcr_threads_M, self.num_lcr_threads_T), order=(0, 1))
        val_layout_g2s_LCR = cute.make_ordered_layout((self.lcr_vecsize, 1), order=(0, 1))
        tiled_copy_g2s_LCR = cute.make_tiled_copy_tv(copy_atom_g2s_LCR, thr_layout_g2s_LCR, val_layout_g2s_LCR)
        # lcr_end is a single M-vector, loaded by the first num_lcr_threads_M threads.
        tiled_copy_g2s_LCREnd = cute.make_tiled_copy_tv(copy_atom_g2s_LCR,
                                                        cute.make_layout(self.num_lcr_threads_M),
                                                        cute.make_layout(self.lcr_vecsize))

        thr_layout_g2s_K = cute.make_ordered_layout((self.num_k_threads_M, self.num_k_threads_T), order=(0, 1))
        val_layout_g2s_K = cute.make_ordered_layout((self.k_vecsize, 1), order=(0, 1))
        tiled_copy_g2s_K = cute.make_tiled_copy_tv(copy_atom_g2s_K, thr_layout_g2s_K, val_layout_g2s_K)

        thr_layout_dY = cute.make_ordered_layout((self.num_dY_threads_M, self.num_dY_threads_R), order=(0, 1))
        val_layout_dY = cute.make_ordered_layout((self.dY_vecsize, 1), order=(0, 1))
        tiled_copy_g2s_dY = cute.make_tiled_copy_tv(copy_atom_g2s_dY, thr_layout_dY, val_layout_dY)

        ################## smem to rmem ##################
        # dY is read back with the same thread/value layout it was loaded with, converted in
        # registers, and written into the WGMMA B layout.
        copy_atom_s2r_dY = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mdY.element_type, num_bits_per_copy=128)
        tiled_copy_s2r_dY = cute.make_tiled_copy_tv(copy_atom_s2r_dY, thr_layout_dY, val_layout_dY)

        ################## SMEM allocations ##################
        dX_mma_tiler_mnk = (self.R_TILE, self.T_TILE, self.M)
        dScale_mma_tiler_mnk = (self.T_TILE, self.M, self.R_TILE)

        # X is the A operand of dScale (K-major). The staged layout is (T_TILE, R_TILE, num_x_stages).
        x_smem_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, dScale_mma_tiler_mnk, self.dtype, num_stages=self.num_x_stages
        )
        # The first chunk_tile rows of each stage, which is the part TMA moves. Shape (chunk_tile, R_TILE, num_x_stages).
        x_tma_smem_layout_staged = cute.slice_(
            cute.zipped_divide(x_smem_layout_staged, (self.chunk_tile, self.R_TILE, 1)),
            ((None, None, 0), (0, 0, None)),
        )
        x_tma_smem_layout = cute.slice_(x_tma_smem_layout_staged, (None, None, 0))

        lcr_end_smem_layout = cute.make_layout(self.M)
        d_lcr_end_partial_smem_layout = cute.make_ordered_layout((self.rows_per_pass, self.M), order=(1, 0))

        # Two views of one M-contiguous tile,
        # (M, T_TILE) for the cp.async copies.
        # (T_TILE, M) for the elementwise passes.
        tm_smem_layout = cute.make_ordered_layout((self.T_TILE, self.M), order=(1, 0))
        mt_smem_layout = cute.make_ordered_layout((self.M, self.T_TILE), order=(0, 1))

        # scale is the B operand of dX.T (FP32 K-major, single stage).
        scale_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, dX_mma_tiler_mnk, TFloat32, num_stages=1
        )

        # dY is loaded as (M, R_TILE, num_r_tiles) in states_dtype. dX.T widens it into FP32 A
        # registers, and it is converted into the FP32 K-major (M, R_TILE) B operand of dScale.
        dY_mr_smem_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, dScale_mma_tiler_mnk, TFloat32, num_stages=self.num_r_tiles
        )
        dY_load_smem_layout = cute.make_ordered_layout(
            (self.M, self.R_TILE, self.num_r_tiles), order=(0, 1, 2)
        )

        @cute.struct
        class SharedStorage:
            x_load_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_x_stages * 2
            ]
            sLCREnd: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(lcr_end_smem_layout)],
                16
            ]
            sdLCREnd: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(lcr_end_smem_layout)],
                16
            ]
            sdLCREndPartial: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(d_lcr_end_partial_smem_layout)],
                16
            ]
            # sLCR is overwritten in place with decay once scale has been formed.
            sLCR: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(mt_smem_layout)],
                16
            ]
            sdScale: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(tm_smem_layout)],
                16
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.kappa_dtype, cute.cosize(tm_smem_layout)],
                16
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(scale_smem_layout_staged)],
                1024
            ]
            sdY: cute.struct.Align[
                cute.struct.MemRange[
                    self.states_dtype, cute.cosize(dY_load_smem_layout)
                ],
                128
            ]
            sdY_mr: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(dY_mr_smem_layout_staged)],
                1024
            ]
            # X stages, reused as the dX store buffer (see class docstring).
            sX: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(x_smem_layout_staged)],
                128
            ]
        self.shared_storage = SharedStorage

        ################## Tensor views ##################
        # we have to create a new view for the inputs as tma expects the b and n
        # dimensions to be last ones.
        mX_lrb = cute.make_tensor(
            mX.iterator,
            cute.make_layout(
                (L, self.R, B), stride=(mX.stride[1], mX.stride[2], mX.stride[0])
            ),
        )
        mdX_lrb = cute.make_tensor(
            mdX.iterator,
            cute.make_layout(
                (L, self.R, B), stride=(mdX.stride[1], mdX.stride[2], mdX.stride[0])
            ),
        )

        # K and LCR are not loaded with TMA. Unlike the forward, the SMEM tile (T_TILE rows) can be larger
        # than the chunk, so we reshape L into (chunk_size, nchunks) so we can index the chunk first and
        # predicate the tile on chunk_len.
        mK_mtnb = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (self.M, self.chunk_size, nchunks, B),
                stride=(mK.stride[2], mK.stride[1], self.chunk_size * mK.stride[1], mK.stride[0]),
            ),
        )
        mLCR_mtnb = cute.make_tensor(
            mLCR.iterator,
            cute.make_layout(
                (self.M, self.chunk_size, nchunks, B),
                stride=(mLCR.stride[2], mLCR.stride[1], self.chunk_size * mLCR.stride[1], mLCR.stride[0]),
            ),
        )
        mdY_bmrn = cute.make_tensor(
            mdY.iterator,
            cute.make_layout(
                (B, self.M, self.R, nchunks),
                stride=(mdY.stride[0], mdY.stride[3], mdY.stride[2], mdY.stride[1]),
            ),
        )

        ################## TMA setup ##################
        x_cta_tiler = (self.chunk_tile, self.R_TILE)

        tma_atom_x, tma_tensor_x = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX_lrb, x_tma_smem_layout, x_cta_tiler,
        )
        tma_atom_dx, tma_tensor_dx = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mdX_lrb, x_tma_smem_layout, x_cta_tiler,
        )

        ################## WGMMA ##################
        dX_mma_op = MmaTF32Op(
            instruction_shape=(64, self.T_TILE, 8),
            a_src=cute.nvgpu.warpgroup.OperandSource.RMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        )
        dScale_mma_op = MmaTF32Op(
            instruction_shape=(64, self.M, 8),
            a_src=cute.nvgpu.warpgroup.OperandSource.RMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.K,
            b_major_mode=cute.nvgpu.OperandMajorMode.K,
        )
        dX_tiled_mma = cute.make_tiled_mma(dX_mma_op)
        dScale_tiled_mma = cute.make_tiled_mma(dScale_mma_op)

        self.kernel(tma_atom_x, tma_tensor_x, mK_mtnb, mLCR_mtnb, mdY_bmrn,
                    tma_atom_dx, tma_tensor_dx, mdK, mdLCR,
                    tiled_copy_g2s_LCR, tiled_copy_g2s_LCREnd, tiled_copy_g2s_K,
                    tiled_copy_g2s_dY, tiled_copy_s2r_dY,
                    dX_tiled_mma, dScale_tiled_mma,
                    x_smem_layout_staged, x_tma_smem_layout_staged,
                    lcr_end_smem_layout, d_lcr_end_partial_smem_layout,
                    tm_smem_layout, mt_smem_layout,
                    scale_smem_layout_staged,
                    dY_load_smem_layout, dY_mr_smem_layout_staged).launch(
            grid=[B, nchunks, 1],
            block=[self.num_threads, 1, 1],
            stream=stream
        )


    @cute.kernel
    def kernel(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_lrb: cute.Tensor,   # (L, R, B)
        mK_mtnb: cute.Tensor,  # (M, chunk_size, nchunks, B)
        mLCR_mtnb: cute.Tensor,  # (M, chunk_size, nchunks, B)
        mdY_bmrn: cute.Tensor,  # (B, M, R, nchunks)
        tma_atom_dx: cute.CopyAtom,
        mdX_lrb: cute.Tensor,  # (L, R, B)
        mdK: cute.Tensor,      # (B, L, M)
        mdLCR: cute.Tensor,    # (B, L, M)
        tiled_copy_g2s_LCR: cute.TiledCopy,
        tiled_copy_g2s_LCREnd: cute.TiledCopy,
        tiled_copy_g2s_K: cute.TiledCopy,
        tiled_copy_g2s_dY: cute.TiledCopy,
        tiled_copy_s2r_dY: cute.TiledCopy,
        dX_tiled_mma: cute.TiledMma,
        dScale_tiled_mma: cute.TiledMma,
        x_smem_layout_staged,  # (T_TILE, R_TILE, PIPE) WGMMA A K-major
        x_tma_smem_layout_staged,  # (chunk_tile, R_TILE, PIPE)
        lcr_end_smem_layout,  # (M)
        d_lcr_end_partial_smem_layout,  # (rows_per_pass, M)
        tm_smem_layout,  # (T_TILE, M)
        mt_smem_layout,  # (M, T_TILE)
        scale_smem_layout_staged,  # (T_TILE, M, 1) dX.T WGMMA B K-major, FP32
        dY_load_smem_layout,  # (M, R_TILE, num_r_tiles) states_dtype
        dY_mr_smem_layout_staged,  # (M, R_TILE, num_r_tiles) dScale WGMMA B K-major, FP32
    ):
        """Use TMA for the pipelined X loads and asynchronous dX stores.

        K and LCR use cp.async to load the next tile while the current tile finishes
        its dLCR reduction. dY is preloaded once with cp.async while the X pipeline
        is primed, then converted to FP32 and written back in the layout required by
        the dScale WGMMA.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_chunk, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_dx)

        L = mX_lrb.shape[0]
        chunk_start = bidx_chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        ################## gmem views ##################
        gX  = cute.local_tile(mX_lrb[None, None, bidx_batch],  (self.chunk_tile, self.R_TILE), (None, None))  # (chunk_tile, R_TILE, RestL/chunk_tile, RestR)
        gdX = cute.local_tile(mdX_lrb[None, None, bidx_batch], (self.chunk_tile, self.R_TILE), (None, None))  # (chunk_tile, R_TILE, RestL/chunk_tile, RestR)
        gdY = cute.local_tile(mdY_bmrn[bidx_batch, None, None, bidx_chunk], (self.M, self.R_TILE), (0, None))  # (M, R_TILE, RestR)
        mK_mt   = mK_mtnb[None, None, bidx_chunk, bidx_batch]    # (M, chunk_size)
        mLCR_mt = mLCR_mtnb[None, None, bidx_chunk, bidx_batch]  # (M, chunk_size)
        idMT = cute.make_identity_tensor((self.M, self.chunk_size))

        ################## SMEM allocations ##################
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        x_load_pipeline_array_ptr = storage.x_load_pipeline_array_ptr.data_ptr()
        sLCREnd = storage.sLCREnd.get_tensor(lcr_end_smem_layout)
        sdLCREnd = storage.sdLCREnd.get_tensor(lcr_end_smem_layout)
        sdLCREndPartial = storage.sdLCREndPartial.get_tensor(d_lcr_end_partial_smem_layout)
        # (M, T) view for the cp.async copy, (T, M) view for the elementwise passes. Decay
        # replaces LCR element by element once it has been consumed.
        sLCR = storage.sLCR.get_tensor(mt_smem_layout)
        sDecay = storage.sLCR.get_tensor(tm_smem_layout)
        sK_mt = storage.sK.get_tensor(mt_smem_layout)
        sK = storage.sK.get_tensor(tm_smem_layout)
        sdScale = storage.sdScale.get_tensor(tm_smem_layout)
        sScale = storage.sScale.get_tensor(scale_smem_layout_staged.outer, swizzle=scale_smem_layout_staged.inner)
        sScale_stage = cute.slice_(sScale, (None, None, 0))
        sX = storage.sX.get_tensor(x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner)
        sX_tma = storage.sX.get_tensor(x_tma_smem_layout_staged.outer, swizzle=x_tma_smem_layout_staged.inner)

        # dY has a states_dtype load view, its (R, M) view that the dX.T A operand is read
        # from, and the FP32 (M, R) WGMMA B operand of dScale.
        sdY_load = storage.sdY.get_tensor(dY_load_smem_layout)
        sdY_load_rm = storage.sdY.get_tensor(
            cute.make_ordered_layout((self.R_TILE, self.M, self.num_r_tiles), order=(1, 0, 2))
        )
        sdY_mr = storage.sdY_mr.get_tensor(dY_mr_smem_layout_staged.outer, swizzle=dY_mr_smem_layout_staged.inner)

        ################## copy partitions ##################
        thr_copy_g2s_dY = tiled_copy_g2s_dY.get_slice(tidx)
        tDgDY = thr_copy_g2s_dY.partition_S(gdY)
        tDsdY_load = thr_copy_g2s_dY.partition_D(sdY_load)

        thr_copy_s2r_dY = tiled_copy_s2r_dY.get_slice(tidx)
        tDsdY_s2r = thr_copy_s2r_dY.partition_S(sdY_load)
        tDrdY = cute.make_rmem_tensor_like(tDsdY_s2r)
        tDsdY_r2s = thr_copy_s2r_dY.partition_D(sdY_mr)

        gLCREnd = mLCR_mt[None, chunk_len - 1]
        thr_copy_g2s_LCREnd = tiled_copy_g2s_LCREnd.get_slice(tidx)
        tLCRgLCREnd = thr_copy_g2s_LCREnd.partition_S(gLCREnd)
        tLCRsLCREnd = thr_copy_g2s_LCREnd.partition_D(sLCREnd)

        # we don't multicast here so cta_coord and cta_layout are simple
        bGS_sX, bGS_gX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX_tma, 0, 2),
            gmem_tensor=cute.group_modes(gX, 0, 2),
        )
        # TMA store partition for dX. It reads the same SMEM the X load wrote.
        bSG_sX, bSG_gdX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dx,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX_tma, 0, 2),
            gmem_tensor=cute.group_modes(gdX, 0, 2),
        )

        ################## pipelines ##################
        x_tma_copy_bytes = cutlass.const_expr(
            cute.size_in_bytes(self.dtype, cute.slice_(x_tma_smem_layout_staged, (None, None, 0)))
        )
        # One elected thread of warp 0 produces through TMA and every consumer warp arrives once.
        x_load_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        x_load_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads // cute.arch.WARP_SIZE
        )
        x_load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=x_load_pipeline_array_ptr,
            num_stages=self.num_x_stages,
            producer_group=x_load_pipeline_producer_group,
            consumer_group=x_load_pipeline_consumer_group,
            tx_count=x_tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        # TMA stores are tracked with bulk async-groups, so this pipeline has no mbarriers.
        # With one stage, producer_acquire waits until the previous store has finished reading SMEM.
        dx_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads),  # unused by PipelineTmaStore
        )

        x_load_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_x_stages)
        x_load_consumer_read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_x_stages)
        x_load_consumer_release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_x_stages)

        ################## MMA fragments ##################
        dX_thr_mma = dX_tiled_mma.get_slice(tidx)
        # dY.T (A of dX.T) is widened from its states_dtype load into FP32 registers.
        tCsdY_A = dX_thr_mma.partition_A(sdY_load_rm)
        # Compact, so the register order is the MMA fragment order rather than the smem order.
        tCrdY = cute.make_rmem_tensor(tCsdY_A[None, None, None, 0].shape, Float32)
        tCrScale = dX_thr_mma.make_fragment_B(dX_thr_mma.partition_B(
            tf32_smem_view(storage.sScale, scale_smem_layout_staged)))
        acc_dX = dX_thr_mma.make_fragment_C(dX_thr_mma.partition_shape_C((self.R_TILE, self.T_TILE)))
        # dX leaves through the X stage it was computed against, as a converted accumulator in sX that TMA stores.
        rdX = cute.make_rmem_tensor(acc_dX.shape, self.dtype)
        # (R_TILE, T_TILE, PIPE) view of sX, matching the dX.T accumulator. sX is R-contiguous,
        # so the accumulator is written with transposed stmatrix.
        sX_rt = cute.composition(
            sX,
            cute.make_ordered_layout((self.R_TILE, self.T_TILE, self.num_x_stages), order=(1, 0, 2)),
        )
        copy_atom_r2s_dX = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(LayoutEnum.COL_MAJOR.is_m_major_c(), num_matrices=4),
            self.dtype,
        )
        tiled_copy_r2s_dX = cute.make_tiled_copy_C(copy_atom_r2s_dX, dX_tiled_mma)
        tRSrdX = tiled_copy_r2s_dX.retile(rdX)
        tRSsX = tiled_copy_r2s_dX.get_slice(tidx).partition_D(sX_rt)

        dScale_thr_mma = dScale_tiled_mma.get_slice(tidx)
        # X (A of dScale) is widened from its 16-bit stage into FP32 registers.
        tCsX_A = dScale_thr_mma.partition_A(sX)
        # Compact, so the register order is the MMA fragment order rather than the smem order.
        tCrX = cute.make_rmem_tensor(tCsX_A[None, None, None, 0].shape, Float32)
        tCrdY_mr = dScale_thr_mma.make_fragment_B(dScale_thr_mma.partition_B(
            tf32_smem_view(storage.sdY_mr, dY_mr_smem_layout_staged)))
        acc_dScale = dScale_thr_mma.make_fragment_C(dScale_thr_mma.partition_shape_C((self.T_TILE, self.M)))
        tCsdScale = dScale_thr_mma.partition_C(sdScale)

        ################## prologue ##################

        # Load dY, lcr_end and the first K/LCR tile with cp.async.
        cute.copy(tiled_copy_g2s_dY, tDgDY, tDsdY_load)
        if tidx < self.num_lcr_threads_M:
            cute.copy(tiled_copy_g2s_LCREnd, tLCRgLCREnd, tLCRsLCREnd)
        self._load_k_lcr(
            0, chunk_len, mK_mt, mLCR_mt, idMT, sK_mt, sLCR,
            tiled_copy_g2s_K, tiled_copy_g2s_LCR, tidx,
        )
        cute.arch.cp_async_commit_group()

        if tidx < self.M:
            sdLCREnd[tidx] = Float32(0.0)

        # Fill every X stage with TMA. The loads overlap the cp.async wait and the dY conversion.
        for _ in cutlass.range_constexpr(self.num_x_stages):
            if warp_idx == 0:
                self._load_x(x_load_pipeline, x_load_producer_state, tma_atom_x, bGS_gX, bGS_sX, bidx_chunk)
            x_load_producer_state.advance()

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Convert dY to the FP32 dScale B operand.
        cute.autovec_copy(tDsdY_s2r, tDrdY)
        tDrdY_out = cute.make_rmem_tensor(tDrdY.layout, Float32)
        tDrdY_out.store(tDrdY.load().to(Float32))
        cute.autovec_copy(tDrdY_out, tDsdY_r2s)

        ################## mainloop ##################
        m_idx = tidx % self.M
        t_row = tidx // self.M
        for t_tile in cutlass.range(self.num_chunk_tiles, unroll=1):
            t_tile_start = t_tile * self.chunk_tile

            ################## recompute scale ##################
            # scale = exp(lcr_end - log_cum_rho) * kappa. rows past chunk_len are zeroed.
            for i in cutlass.range_constexpr(self.rows_per_lane):
                t_offset = t_row + i * self.rows_per_pass
                t_idx = t_tile_start + t_offset
                if t_idx < chunk_len:
                    decay = cute.exp(sLCREnd[m_idx] - sLCR[m_idx, t_offset])
                    kappa = sK[t_offset, m_idx]
                    sDecay[t_offset, m_idx] = decay
                    sScale_stage[t_offset, m_idx] = decay * kappa.to(Float32)
                else:
                    sDecay[t_offset, m_idx] = Float32(0.0)
                    sK[t_offset, m_idx] = self.kappa_dtype(0.0)
                    sScale_stage[t_offset, m_idx] = Float32(0.0)

            # Make the generic-proxy writes to sScale/sdY visible to WGMMA (async proxy).
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            for r_tile in cutlass.range_constexpr(self.num_r_tiles):
                ################## dX WGMMA ##################
                # It doesn't need X, so it is issued before waiting on the X stage.
                # dY.T is the same for every t_tile, but it is widened again each time
                # so its registers are held only during this WGMMA.
                self.widen(tCsdY_A[None, None, None, r_tile], tCrdY)
                cute.nvgpu.warpgroup.fence()
                self.gemm(
                    dX_tiled_mma,
                    tf32_rmem_view(tCrdY),
                    tCrScale[None, None, None, 0],
                    acc_dX,
                    False,
                )
                cute.nvgpu.warpgroup.commit_group()

                ################## release the previous X stage and refill it ##################
                if x_load_consumer_read_state.count > 0:  # the first tile has no previous stage
                    # Wait for the previous dX TMA store to finish reading the stage
                    if warp_idx == 0:
                        dx_store_pipeline.producer_acquire()
                    cute.arch.sync_threads()
                    x_load_pipeline.consumer_release(x_load_consumer_release_state)
                    x_load_consumer_release_state.advance()
                    if x_load_producer_state.count < self.num_x_tiles:
                        if warp_idx == 0:
                            self._load_x(x_load_pipeline, x_load_producer_state, tma_atom_x, bGS_gX, bGS_sX, bidx_chunk)
                        x_load_producer_state.advance()

                ################## dScale WGMMA ##################
                # We need X now.

                x_load_pipeline.consumer_wait(x_load_consumer_read_state)
                self.widen(tCsX_A[None, None, None, x_load_consumer_read_state.index], tCrX)
                cute.nvgpu.warpgroup.fence()
                self.gemm(
                    dScale_tiled_mma,
                    tf32_rmem_view(tCrX),
                    tCrdY_mr[None, None, None, r_tile],
                    acc_dScale,
                    r_tile != 0,  # accumulate across r_tiles (False on first r_tile so acc=zero+gemm)
                )
                cute.nvgpu.warpgroup.commit_group()

                ################## convert dX ##################
                # dX is the older WGMMA group, so it can be converted while dScale runs.
                cute.nvgpu.warpgroup.wait_group(1)
                rdX.store(acc_dX.load().to(self.dtype))

                ################## TMA store dX ##################
                # Once dScale has finished reading the stage, overwrite it with dX and store it.
                cute.nvgpu.warpgroup.wait_group(0)
                cute.copy(tiled_copy_r2s_dX, tRSrdX, tRSsX[None, None, None, x_load_consumer_read_state.index])
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                if warp_idx == 0:
                    dx_tma_tile = bidx_chunk * self.num_chunk_tiles + t_tile
                    cute.copy(tma_atom_dx, bSG_sX[None, x_load_consumer_read_state.index], bSG_gdX[None, dx_tma_tile, r_tile])
                    dx_store_pipeline.producer_commit()

                x_load_consumer_read_state.advance()

            ################## dK / dLCR epilogue for this t_tile ##################
            # Convert dScale to a linear (T, M) layout for the elementwise pass.
            cute.autovec_copy(acc_dScale, tCsdScale)
            cute.arch.sync_threads()

            d_lcr_end = Float32(0.0)
            for i in cutlass.range_constexpr(self.rows_per_lane):
                t_offset = t_row + i * self.rows_per_pass
                t_idx = t_tile_start + t_offset
                if t_idx < chunk_len:
                    l_idx = chunk_start + t_idx
                    d_kappa = sdScale[t_offset, m_idx] * sDecay[t_offset, m_idx]
                    d_exp = d_kappa * sK[t_offset, m_idx].to(Float32)
                    mdK[bidx_batch, l_idx, m_idx] = d_kappa.to(mdK.element_type)
                    if t_idx != chunk_len - 1:
                        mdLCR[bidx_batch, l_idx, m_idx] = -d_exp
                        d_lcr_end += d_exp

            sdLCREndPartial[t_row, m_idx] = d_lcr_end
            cute.arch.sync_threads()

            ################## prefetch the next K/LCR tile ##################
            # sK and sLCR are free after the barrier above. The load overlaps the reduction below.
            if t_tile + 1 < self.num_chunk_tiles:
                self._load_k_lcr(
                    t_tile + 1, chunk_len, mK_mt, mLCR_mt, idMT, sK_mt, sLCR,
                    tiled_copy_g2s_K, tiled_copy_g2s_LCR, tidx,
                )
                cute.arch.cp_async_commit_group()

            ################## reduce the dLCR partials of the chunk end ##################
            if tidx < self.M:
                d_lcr_end = sdLCREnd[tidx]
                for row in cutlass.range_constexpr(self.rows_per_pass):
                    d_lcr_end += sdLCREndPartial[row, tidx]
                sdLCREnd[tidx] = d_lcr_end
            if t_tile + 1 < self.num_chunk_tiles:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        if tidx < self.M:
            mdLCR[bidx_batch, chunk_start + chunk_len - 1, tidx] = sdLCREnd[tidx]

        ################## drain ##################
        # Wait for the last dX store to finish reading SMEM, then release the last X stage.
        if warp_idx == 0:
            dx_store_pipeline.producer_tail()
        cute.arch.sync_threads()
        x_load_pipeline.consumer_release(x_load_consumer_release_state)
        x_load_consumer_release_state.advance()
        # Waits for every release before the CTA exits. It only matters with clusters, where a
        # release arrives on another CTA's barrier, and it is kept to follow the pipeline contract.
        if warp_idx == 0:
            x_load_pipeline.producer_tail(x_load_producer_state)


    @cute.jit
    def _load_x(self, x_load_pipeline, x_load_producer_state, tma_atom_x, bGS_gX, bGS_sX, bidx_chunk):
        """Wait for stage ``state.index`` to be empty, then TMA load X tile ``state.count`` into it."""
        x_load_pipeline.producer_acquire(x_load_producer_state)
        x_t_tile = x_load_producer_state.count // self.num_r_tiles
        x_r_tile = x_load_producer_state.count % self.num_r_tiles
        x_tma_tile = bidx_chunk * self.num_chunk_tiles + x_t_tile
        cute.copy(
            tma_atom_x,
            bGS_gX[None, x_tma_tile, x_r_tile],
            bGS_sX[None, x_load_producer_state.index],
            tma_bar_ptr=x_load_pipeline.producer_get_barrier(x_load_producer_state),
        )
        # No-op for TMA since the hardware completes the barrier. Kept to follow the pipeline contract.
        x_load_pipeline.producer_commit(x_load_producer_state)

    @staticmethod
    @cute.jit
    def _load_k_lcr(
        t_tile, chunk_len, mK_mt, mLCR_mt, idMT, sK_mt, sLCR, tiled_copy_g2s_K, tiled_copy_g2s_LCR, tidx
    ):
        """cp.async the (M, T_TILE) K and LCR tiles of ``t_tile`` into SMEM, predicated on chunk_len."""
        tile_shape = cute.shape(sK_mt)
        gK = cute.local_tile(mK_mt, tile_shape, (0, t_tile))
        gLCR = cute.local_tile(mLCR_mt, tile_shape, (0, t_tile))
        cMT = cute.local_tile(idMT, tile_shape, (0, t_tile))

        thr_copy_g2s_K = tiled_copy_g2s_K.get_slice(tidx)
        tKgK = thr_copy_g2s_K.partition_S(gK)
        tKsK = thr_copy_g2s_K.partition_D(sK_mt)
        tKpK = predicate_axis(thr_copy_g2s_K.partition_S(cMT), chunk_len, axis=1)
        cute.copy(tiled_copy_g2s_K, tKgK, tKsK, pred=tKpK)

        thr_copy_g2s_LCR = tiled_copy_g2s_LCR.get_slice(tidx)
        tLCRgLCR = thr_copy_g2s_LCR.partition_S(gLCR)
        tLCRsLCR = thr_copy_g2s_LCR.partition_D(sLCR)
        tLCRpLCR = predicate_axis(thr_copy_g2s_LCR.partition_S(cMT), chunk_len, axis=1)
        cute.copy(tiled_copy_g2s_LCR, tLCRgLCR, tLCRsLCR, pred=tLCRpLCR)


    @staticmethod
    @cute.jit
    def widen(src, dst):
        """Convert a thread's smem A partition element by element into its compact FP32 register fragment."""
        for i in cutlass.range_constexpr(cute.size(dst)):
            dst[i] = src[i].to(Float32)

    @staticmethod
    @cute.jit
    def gemm(tiled_mma, A, B, C, accumulate_initial):
        """One pass of K-tile WGMMA. `accumulate_initial` selects whether the first WGMMA atom
        in this call accumulates into C (True after the first reduction tile) or overwrites
        the prior C (False on the very first K-tile)."""
        # Static K-blocks, so a register A operand is indexed at compile time.
        for k_block_idx in cutlass.range_constexpr(cute.size(A, mode=[2])):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, accumulate_initial or k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, None, k_block_idx],
                B[None, None, k_block_idx],
                C,
            )



@jit_cache
def _compile_chunk_state_fwd(dtype, kappa_dtype, states_dtype, R, M, chunk_size):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    # elems-per-128b for each dtype; promises 16B-aligned base ptr for cp.async at 128b.
    div_input = 128 // dtype.width
    div_kappa = 128 // kappa_dtype.width
    div_state = 128 // states_dtype.width
    fake_u = make_fake_tensor(dtype,   (B_sym, L_sym, R), divisibility=div_input)
    fake_kappa = make_fake_tensor(kappa_dtype, (B_sym, L_sym, M),   divisibility=div_kappa)
    fake_log_cum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M),   divisibility=4)
    fake_states = make_fake_tensor(states_dtype, (B_sym, nchunks_sym, R, M), divisibility=div_state)
    return cute.compile(
        ChunkStateFwd(dtype, R=R, M=M, chunk_size=chunk_size, states_dtype=states_dtype, kappa_dtype=kappa_dtype),
        fake_u, fake_kappa, fake_log_cum_rho, fake_states,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32=True):
    B, L, R = u.shape
    M = log_cum_rho.shape[-1]
    assert 0 < M <= 16, f"M must be between 1 and 16, got {M}"

    # The kernel needs M==16, but we can pad smaller M tensors. 
    kernel_M = 16
    nchunks = math.ceil(L / chunk_size)

    torch_states_dtype = torch.float32 if state_fp32 else u.dtype
    states = torch.empty((B, nchunks, R, kernel_M), device=u.device, dtype=torch_states_dtype)

    if B == 0 or L == 0:
        return states[..., :M].contiguous()

    if M != kernel_M:
        kappa = torch.nn.functional.pad(kappa, (0, kernel_M - M))
        log_cum_rho = torch.nn.functional.pad(log_cum_rho, (0, kernel_M - M))

    dtype = torch2cute_dtype_map[u.dtype]
    kappa_dtype = torch2cute_dtype_map[kappa.dtype]
    states_dtype = torch2cute_dtype_map[torch_states_dtype]
    _compile_chunk_state_fwd(dtype, kappa_dtype, states_dtype, R, kernel_M, chunk_size)(
        u, kappa, log_cum_rho, states
    )

    return states[..., :M].contiguous()


@jit_cache
def _compile_chunk_state_bwd(dtype, kappa_dtype, states_dtype, R, M, chunk_size):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    div_input = 128 // dtype.width
    div_kappa = 128 // kappa_dtype.width
    div_state = 128 // states_dtype.width
    fake_d_states = make_fake_tensor(states_dtype, (B_sym, nchunks_sym, R, M), divisibility=div_state)
    fake_u = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=div_input)
    fake_kappa = make_fake_tensor(kappa_dtype, (B_sym, L_sym, M), divisibility=div_kappa)
    fake_log_cum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=4)
    fake_d_u = make_fake_tensor(dtype, (B_sym, L_sym, R), divisibility=div_input)
    fake_d_kappa = make_fake_tensor(kappa_dtype, (B_sym, L_sym, M), divisibility=div_kappa)
    fake_d_log_cum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=4)
    return cute.compile(
        ChunkStateBwd(dtype, R=R, M=M, chunk_size=chunk_size, states_dtype=states_dtype, kappa_dtype=kappa_dtype),
        fake_d_states, fake_u, fake_kappa, fake_log_cum_rho, fake_d_u, fake_d_kappa, fake_d_log_cum_rho,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _frac_chunk_state_bwd(d_states, u, kappa, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = kappa.shape[-1]
    assert 0 < M <= 16, f"M must be between 1 and 16, got {M}"

    # The kernel needs M==16, but we can pad smaller M tensors.
    kernel_M = 16

    d_u = torch.empty((B, L, R), device=u.device, dtype=u.dtype)
    d_kappa = torch.empty((B, L, kernel_M), device=kappa.device, dtype=kappa.dtype)
    d_log_cum_rho = torch.empty((B, L, kernel_M), device=log_cum_rho.device, dtype=torch.float32)

    if B == 0 or L == 0:
        return d_u, d_kappa[..., :M].contiguous(), d_log_cum_rho[..., :M].contiguous()

    if M != kernel_M:
        pad = (0, kernel_M - M)
        d_states = torch.nn.functional.pad(d_states, pad)
        kappa = torch.nn.functional.pad(kappa, pad)
        log_cum_rho = torch.nn.functional.pad(log_cum_rho, pad)

    dtype = torch2cute_dtype_map[u.dtype]
    kappa_dtype = torch2cute_dtype_map[kappa.dtype]
    states_dtype = torch2cute_dtype_map[d_states.dtype]
    _compile_chunk_state_bwd(dtype, kappa_dtype, states_dtype, R, kernel_M, chunk_size)(
        d_states, u, kappa, log_cum_rho, d_u, d_kappa, d_log_cum_rho
    )

    return d_u, d_kappa[..., :M].contiguous(), d_log_cum_rho[..., :M].contiguous()


# ============================================================================
# Public API (autograd-wrapped) and reference
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_state_fwd_cute", mutates_args=())
def _chunk_state_fwd_op(u: torch.Tensor, kappa: torch.Tensor, log_cum_rho: torch.Tensor, chunk_size: int, state_fp32: bool) -> torch.Tensor:
    return _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32)


@_chunk_state_fwd_op.register_fake
def _(u, kappa, log_cum_rho, chunk_size, state_fp32):
    B, L, R = u.shape
    M = kappa.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    states_dtype = torch.float32 if state_fp32 else u.dtype
    return torch.empty(B, nchunks, R, M, device=u.device, dtype=states_dtype)


@torch.library.custom_op("frac_ssm::chunk_state_bwd_cute", mutates_args=())
def _chunk_state_bwd_op(
    d_states: torch.Tensor,
    u: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _frac_chunk_state_bwd(d_states, u, kappa, log_cum_rho, chunk_size)


@_chunk_state_bwd_op.register_fake
def _(d_states, u, kappa, log_cum_rho, chunk_size):
    B, L, R = u.shape
    M = kappa.shape[-1]
    return (
        torch.empty(B, L, R, device=u.device, dtype=u.dtype),
        torch.empty(B, L, M, device=kappa.device, dtype=kappa.dtype),
        torch.empty(B, L, M, device=log_cum_rho.device, dtype=log_cum_rho.dtype),
    )


def _chunk_state_setup_ctx(ctx, inputs, output):
    u, kappa, log_cum_rho, chunk_size, state_fp32 = inputs
    ctx.save_for_backward(u, kappa, log_cum_rho)
    ctx.chunk_size = chunk_size


def _chunk_state_backward(ctx, d_states):
    u, kappa, log_cum_rho = ctx.saved_tensors
    return (
        *_chunk_state_bwd_op(
            d_states.contiguous(), u, kappa, log_cum_rho, ctx.chunk_size
        ),
        None,
        None,
    )


_chunk_state_fwd_op.register_autograd(_chunk_state_backward, setup_context=_chunk_state_setup_ctx)


def chunk_state(u, kappa, log_cum_rho, chunk_size, state_fp32=True):
    """Compute per-chunk states for scan."""
    assert u.dim() == 3, f"u must be 3D (B, L, R), got {u.dim()}D"
    assert kappa.dim() == 3, f"kappa must be 3D (B, L, M), got {kappa.dim()}D"
    assert log_cum_rho.dim() == 3, f"log_cum_rho must be 3D (B, L, M), got {log_cum_rho.dim()}D"
    assert u.is_cuda and kappa.is_cuda and log_cum_rho.is_cuda, "inputs must be on CUDA"
    assert log_cum_rho.dtype == torch.float32, (
        f"log_cum_rho dtype must be float32, got {log_cum_rho.dtype}"
    )
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    return _chunk_state_fwd_op(u, kappa, log_cum_rho, chunk_size, state_fp32)


if __name__ == "__main__":
    from frac_ssm.ops.cute.chunk_cumsum import chunk_cumsum

    chunk_size = 32
    B, L, R, M = 32, 4096, 64, 16
    device = "cuda"
    u       = torch.randn(B, L, R, device=device, dtype=torch.bfloat16)
    kappa   = torch.randn(B, L, M, device=device, dtype=torch.bfloat16)
    log_rho = torch.randn(B, L, M, device=device, dtype=torch.float32)

    chunk_sizes = [32, 64, 128, 256]

    lcr_precomp = {}
    for chunk_size in chunk_sizes:
        lcr_precomp[chunk_size] = chunk_cumsum(log_rho, chunk_size)
        log_cum_rho, _ = lcr_precomp[chunk_size]
        for _ in range(10):
            _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, "tf32")
            import sys
            sys.exit()

    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    for chunk_size in chunk_sizes:
        log_cum_rho, _ = lcr_precomp[chunk_size]
        torch.cuda.nvtx.range_push(f"cs_{chunk_size}")
        _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, "tf32")
        torch.cuda.nvtx.range_pop()

    torch.cuda.profiler.stop()
