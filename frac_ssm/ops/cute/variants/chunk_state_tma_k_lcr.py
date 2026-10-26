"""
X, K, and LCR are loaded with TMA, and Y is stored with TMA. 
Putting all input loads on TMA can reduce their overlap and increase scoreboard stalls.
"""

import math

import torch
import cutlass
from cutlass import cute

from cutlass import Float16, Float32
import cuda.bindings.driver as cuda
from cutlass.utils import LayoutEnum, sm90_make_smem_layout_a, sm90_make_smem_layout_b


from frac_ssm.ops.cute.utils import make_fake_tensor, torch2cute_dtype_map
from frac_ssm.ops.cute.cache_utils import jit_cache


class ChunkStateFwd:
    def __init__(self, dtype: type[cutlass.Numeric], R: int, M: int, chunk_size: int, states_dtype: type[cutlass.Numeric]):
        self.dtype = dtype
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

        assert self.chunk_size % cute.arch.WARP_SIZE == 0, (
            f"chunk_size must be a multiple of {cute.arch.WARP_SIZE}, got {chunk_size}"
        )
        assert self.chunk_size % self.chunk_tile == 0, (
            f"chunk_size {chunk_size} not divisible by chunk_tile {self.chunk_tile}"
        )
        assert dtype.width == 16, (
            f"dtype must be 16-bit (F16/BF16), got {dtype} (width {dtype.width})"
        )
        assert M == 16, f"M=16 is the only supported configuration, got {M}"
        assert R % 64 == 0, f"R must be a multiple of 64, got {R}"


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

        x_smem_layout_staged = sm90_make_smem_layout_a(
            LayoutEnum.COL_MAJOR, mma_tiler_mnk, self.dtype, num_stages=1
        )
        x_smem_layout = cute.slice_(x_smem_layout_staged, (None, None, 0))

        lcr_smem_layout = cute.make_ordered_layout((self.M, self.chunk_tile), order=(0, 1))

        scale_smem_layout_staged = sm90_make_smem_layout_b(
            LayoutEnum.COL_MAJOR, mma_tiler_mnk, self.dtype, num_stages=1
        )
        scale_smem_layout = cute.slice_(scale_smem_layout_staged, (None, None, 0))

        y_smem_layout = cute.make_ordered_layout((self.R, self.M), order=(1, 0))


        @cute.struct
        class SharedStorage:
            x_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            k_lcr_load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sLCREnd: cute.struct.Align[
                cute.struct.MemRange[mLCR.element_type, self.M],
                16
            ]
            sLCR: cute.struct.Align[
                cute.struct.MemRange[mLCR.element_type, cute.cosize(lcr_smem_layout)],
                128
            ]
            # we use sK to store scale and to feed wgmma so we use cosize of scale_smem_layout_staged
            sK: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(scale_smem_layout_staged)],
                128
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

        mK_mlb = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (self.M, L, B),
                stride=(mK.stride[2], mK.stride[1], mK.stride[0]),
            ),
        )

        mLCR_mlb = cute.make_tensor(
            mLCR.iterator,
            cute.make_layout(
                (self.M, L, B),
                stride=(mLCR.stride[2], mLCR.stride[1], mLCR.stride[0]),
            ),
        )

        mY_rmbn = cute.make_tensor(
            mY.iterator,
            cute.make_layout(
                (self.R, self.M, B, nchunks),
                stride=(mY.stride[2], mY.stride[3], mY.stride[0], mY.stride[1]),
            ),
        )

        ################## TMA setup ##################
        x_cta_tiler = (self.R, self.chunk_tile)
        k_lcr_cta_tiler = (self.M, self.chunk_tile)
        y_cta_tiler = (self.R, self.M)

        tma_atom_x, tma_tensor_x = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mX_rlb, x_smem_layout, x_cta_tiler
        )

        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mK_mlb, scale_smem_layout, k_lcr_cta_tiler
        )
        tma_atom_lcr, tma_tensor_lcr = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mLCR_mlb, lcr_smem_layout, k_lcr_cta_tiler
        )

        tma_atom_y, tma_tensor_y = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mY_rmbn, y_smem_layout, y_cta_tiler
        )


        ################## WGMMA ##################
        mma_op = cute.nvgpu.warpgroup.MmaF16BF16Op(
            ab_dtype=self.dtype,
            acc_dtype=Float32,
            instruction_shape=(64, self.M, 16),
            a_src=cute.nvgpu.warpgroup.OperandSource.SMEM,
            a_major_mode=cute.nvgpu.OperandMajorMode.MN,
            b_major_mode=cute.nvgpu.OperandMajorMode.MN
        )
        tiled_mma = cute.make_tiled_mma(mma_op)
        self.kernel(tma_atom_x, tma_tensor_x,
                    tma_atom_k, tma_tensor_k,
                    tma_atom_lcr, tma_tensor_lcr, mLCR,
                    tma_atom_y, tma_tensor_y,
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
        mX_rlb: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_mlb: cute.Tensor,
        tma_atom_lcr: cute.CopyAtom,
        mLCR_mlb: cute.Tensor,
        mLCR_blm: cute.Tensor,
        tma_atom_y: cute.CopyAtom,
        mY_rmbn: cute.Tensor,
        tiled_copy_s2r_LCR: cute.TiledCopy,
        tiled_copy_s2r_K: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged, # (R, chunk_tile, PIPE)
        lcr_smem_layout,
        scale_smem_layout_staged, # (M, chunk_tile, PIPE) WGMMA B MN-major
        y_smem_layout, # (R, M)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_chunk, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        L = mLCR_blm.shape[1]
        chunk_start = bidx_chunk * self.chunk_size
        chunk_len = cutlass.min(self.chunk_size, L - chunk_start)

        idML = cute.make_identity_tensor((self.M, L))

        gX = cute.local_tile(mX_rlb[None, None, bidx_batch], (self.R, self.chunk_tile), (None, None)) # (R, chunk_tile, RestR, RestL/chunk_tile)
        gK = cute.local_tile(mK_mlb[None, None, bidx_batch], (self.M, self.chunk_tile), (None, None)) # (M, chunk_tile, RestM, RestL/chunk_tile)
        gLCR = cute.local_tile(mLCR_mlb[None, None, bidx_batch], (self.M, self.chunk_tile), (None, None)) # (M, chunk_tile, RestM, RestL/chunk_tile)
        gY = cute.local_tile(mY_rmbn[None, None, bidx_batch, bidx_chunk], (self.R, self.M), (None, None)) # (R, M, RestR, RestM)


        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        x_load_mbar_ptr = storage.x_load_mbar_ptr.data_ptr()
        k_lcr_load_mbar_ptr = storage.k_lcr_load_mbar_ptr.data_ptr()
        sLCREnd = storage.sLCREnd.get_tensor(cute.make_layout(self.M))
        sLCR = storage.sLCR.get_tensor(lcr_smem_layout)
        sK = storage.sK.get_tensor(scale_smem_layout_staged.outer, swizzle=scale_smem_layout_staged.inner)
        sX = storage.sX.get_tensor(x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner)
        sY = storage.sY.get_tensor(y_smem_layout)


        # we don't multicast here so cta_coord and cta_layout are simple
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sX, 0, 2),
            gmem_tensor=cute.group_modes(gX, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sK, 0, 2),
            gmem_tensor=cute.group_modes(gK, 0, 2),
        )
        tLCRsLCR, tLCRgLCR = cute.nvgpu.cpasync.tma_partition(
            tma_atom_lcr,
            cta_coord=0,
            cta_layout=cute.make_layout(1),
            smem_tensor=cute.group_modes(sLCR, 0, 2),
            gmem_tensor=cute.group_modes(gLCR, 0, 2),
        )
        tXsX_pipe = tXsX[None, 0]
        tKsK_pipe = tKsK[None, 0]
        tLCRsLCR_tma = tLCRsLCR[None]

        x_tma_copy_bytes = cutlass.const_expr(cute.size_in_bytes(self.dtype, cute.slice_(x_smem_layout_staged, (None, None, 0))))
        k_lcr_tma_copy_bytes = cutlass.const_expr(
            cute.size_in_bytes(self.dtype, cute.slice_(scale_smem_layout_staged, (None, None, 0)))
            + cute.size_in_bytes(Float32, lcr_smem_layout)
        )


        thr_mma = tiled_mma.get_slice(tidx)
        acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((self.R, self.M)))
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sX))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sK))

        # LCR
        thr_copy_s2r_LCR = tiled_copy_s2r_LCR.get_slice(tidx)
        tLCRsLCR_s2r = thr_copy_s2r_LCR.partition_S(sLCR)

        # K is loaded from s to r and also used to write output of scale from r to s
        thr_copy_s2r_K   = tiled_copy_s2r_K.get_slice(tidx)
        sK_stage = cute.slice_(sK, (None, None, 0))
        tKsK_s2r = thr_copy_s2r_K.partition_S(sK_stage)
        tKsK_r2s = thr_copy_s2r_K.partition_D(sK_stage)

        # Load the end of chunk's decay once
        if tidx < self.M:
            sLCREnd[tidx] = mLCR_blm[bidx_batch, chunk_start + chunk_len - 1, tidx]

        for k_tile in cutlass.range_constexpr(self.num_chunk_tiles):
            tile_idx = bidx_chunk * self.num_chunk_tiles + k_tile

            cML = cute.local_tile(idML, (self.M, self.chunk_tile), (0, tile_idx))

            # TMA loads for K and LCR. Only one thread from warp 0 issues
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(k_lcr_load_mbar_ptr, 1)
                    cute.arch.mbarrier_expect_tx(k_lcr_load_mbar_ptr, k_lcr_tma_copy_bytes)

            # TMA load for X. Only one thread from warp 0 issues
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(x_load_mbar_ptr, 1)
                    cute.arch.mbarrier_expect_tx(x_load_mbar_ptr, x_tma_copy_bytes)

            # All threads should run this
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()

            if warp_idx == 0:
                cute.copy(tma_atom_k, tKgK[None, 0, tile_idx], tKsK_pipe, tma_bar_ptr=k_lcr_load_mbar_ptr)
                cute.copy(tma_atom_lcr, tLCRgLCR[None, 0, tile_idx], tLCRsLCR_tma, tma_bar_ptr=k_lcr_load_mbar_ptr)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(k_lcr_load_mbar_ptr)

            if warp_idx == 0:
                cute.copy(tma_atom_x, tXgX[None, 0, tile_idx], tXsX_pipe, tma_bar_ptr=x_load_mbar_ptr)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(x_load_mbar_ptr)

            # Wait for K and LCR TMA loads to finish
            cute.arch.mbarrier_wait(k_lcr_load_mbar_ptr, phase=0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            # flow is smem -> rmem (compute scale) -> smem
            if tidx < self.num_load_threads:
                tKcML = thr_copy_s2r_K.partition_S(cML)

                tLCRrLCR = cute.make_rmem_tensor_like(tLCRsLCR_s2r)
                tKrK = cute.make_rmem_tensor_like(tKsK_s2r)

                cute.autovec_copy(tLCRsLCR_s2r, tLCRrLCR)
                cute.autovec_copy(tKsK_s2r, tKrK)

                # The formula is scale = exp(lcr_end - log_cum_rho) * kappa
                # Reuse sK for scale which is feed as the WGMMA B operand
                num_elems = cute.size(tKcML, mode=[0])
                for i in cutlass.range_constexpr(num_elems):
                    coord = tKcML[i, 0, 0]
                    m_idx = coord[0]
                    l_idx = coord[1]
                    if cute.elem_less(l_idx, L):
                        scale_fp32 = cute.exp(sLCREnd[m_idx] - tLCRrLCR[i, 0, 0]) * tKrK[i, 0, 0]
                        tKrK[i, 0, 0] = scale_fp32.to(self.dtype)
                    else:
                        tKrK[i, 0, 0] = self.dtype(0.0)

                cute.autovec_copy(tKrK, tKsK_r2s)

            # Wait for X TMA load to finish
            cute.arch.mbarrier_wait(x_load_mbar_ptr, phase=0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            # Accumulate this reduction tile into the running acc
            cute.nvgpu.warpgroup.fence()
            self.gemm(
                tiled_mma,
                tCrA[None, None, None, 0],
                tCrB[None, None, None, 0],
                acc,
                k_tile != 0,  # accumulate across iters (False on first iter so acc=zero+gemm)
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)


        # Writing the results
        # Select the actual store atom from sY's layout and dtype
        copy_atom_r2s = cutlass.utils.sm90_get_smem_store_op(LayoutEnum.from_tensor(sY), elem_ty_d=self.states_dtype, elem_ty_acc=Float32)
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
        cute.arch.barrier()


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


    @staticmethod
    @cute.jit
    def gemm(tiled_mma, A, B, C, accumulate_initial):
        """One pass of K-tile WGMMA. `accumulate_initial` selects whether the first WGMMA atom
        in this call accumulates into C (True after the first chunk tile) or overwrites
        the prior C (False on the very first K-tile)."""
        for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, accumulate_initial or k_block_idx != 0)
            cute.gemm(
                tiled_mma,
                C,
                A[None, None, k_block_idx],
                B[None, None, k_block_idx],
                C,
            )

@jit_cache
def _compile_chunk_state_fwd(dtype, states_dtype, R, M, chunk_size):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    # elems-per-128b for the user dtype; promises 16B-aligned base ptr for cp.async at 128b.
    div_input = 128 // dtype.width
    div_state = 128 // states_dtype.width
    fake_u = make_fake_tensor(dtype,   (B_sym, L_sym, R), divisibility=div_input)
    fake_kappa = make_fake_tensor(dtype, (B_sym, L_sym, M),   divisibility=div_input)
    fake_log_cum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M),   divisibility=4)
    fake_states = make_fake_tensor(states_dtype, (B_sym, nchunks_sym, R, M), divisibility=div_state)
    return cute.compile(
        ChunkStateFwd(dtype, R=R, M=M, chunk_size=chunk_size, states_dtype=states_dtype),
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
    states_dtype = torch2cute_dtype_map[torch_states_dtype]
    _compile_chunk_state_fwd(dtype, states_dtype, R, kernel_M, chunk_size)(
        u, kappa, log_cum_rho, states
    )

    return states[..., :M].contiguous()


# We don't have a separate backward for this variant.
from frac_ssm.ops.cute.chunk_state import _frac_chunk_state_bwd

# ============================================================================
# Public API (autograd-wrapped) and reference
# ============================================================================


@torch.library.custom_op("frac_ssm::chunk_state_fwd_cute_tma", mutates_args=())
def _chunk_state_fwd_op(u: torch.Tensor, kappa: torch.Tensor, log_cum_rho: torch.Tensor, chunk_size: int, state_fp32: bool) -> torch.Tensor:
    return _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size, state_fp32)


@_chunk_state_fwd_op.register_fake
def _(u, kappa, log_cum_rho, chunk_size, state_fp32):
    B, L, R = u.shape
    M = kappa.shape[-1]
    nchunks = math.ceil(L / chunk_size)
    states_dtype = torch.float32 if state_fp32 else u.dtype
    return torch.empty(B, nchunks, R, M, device=u.device, dtype=states_dtype)


@torch.library.custom_op("frac_ssm::chunk_state_bwd_cute_tma", mutates_args=())
def _chunk_state_bwd_op(
    d_states: torch.Tensor,
    u: torch.Tensor,
    kappa: torch.Tensor,
    log_cum_rho: torch.Tensor,
    chunk_size: int,
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
            _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size)
            import sys
            sys.exit()

    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    for chunk_size in chunk_sizes:
        log_cum_rho, _ = lcr_precomp[chunk_size]
        torch.cuda.nvtx.range_push(f"cs_{chunk_size}")
        _frac_chunk_state_fwd(u, kappa, log_cum_rho, chunk_size)
        torch.cuda.nvtx.range_pop()
        
    torch.cuda.profiler.stop()
