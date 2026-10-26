"""
CuTe chunk_cumsum.

Each warp initially loads a set of full rows contiguously from gmem to a swizzled smem. 
One CTA loads an entire (chunk_size, M). 
Then, each warp computes prefix (or suffix) sum on a set of full columns and stores the results. 
"""

import math

import torch

import cutlass
from cutlass import cute
from cutlass import Float32
import cuda.bindings.driver as cuda

from frac_ssm.ops.cute.utils import make_fake_tensor, predicate_axis
from frac_ssm.ops.cute.cache_utils import jit_cache


@cute.jit
def warp_prefix_sum(val: Float32, reverse: cutlass.Constexpr[bool] = False) -> Float32:
    lane = cute.arch.lane_idx()

    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        if cutlass.const_expr(reverse):
            partial = cute.arch.shuffle_sync_down(val, offset=offset, mask_and_clamp=31)
            if lane + offset < cute.arch.WARP_SIZE:
                val += partial
        else:
            partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
            if lane >= offset:
                val += partial

    return val


class ChunkCumsumFwd:
    def __init__(self, dtype: type[cutlass.Numeric], M: int, chunk_size: int):
        self.M = M
        self.chunk_size = chunk_size

        self.vecsize = math.gcd(128 // dtype.width, M)
        self.num_warps_M = M // self.vecsize
        self.rows_per_lane = self.chunk_size // cute.arch.WARP_SIZE
        self.num_threads = cute.arch.WARP_SIZE * self.num_warps_M
        self.log_vecsize = int(math.log2(self.vecsize))

        assert self.chunk_size % cute.arch.WARP_SIZE == 0, \
            f"chunk_size must be a multiple of {cute.arch.WARP_SIZE}, got {chunk_size}"
        assert self.vecsize in (1, 2, 4), f"vecsize must be 1/2/4, got {self.vecsize}"

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,    # (B, L, M)
        mY: cute.Tensor,    # (B, L, M) fp32
        mD: cute.Tensor,    # (B, nchunks, M) fp32
        tiled_copy_g2s: cute.TiledCopy,
        tiled_copy_r2g: cute.TiledCopy
    ):

        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_chunk, _ = cute.arch.block_idx()

        L = mX.shape[1]

        idX = cute.make_identity_tensor(mX[bidx_batch, None, None].shape)
        gX = cute.local_tile(mX[bidx_batch, None, None], (self.chunk_size, self.M), (bidx_chunk, 0))
        gY = cute.local_tile(mY[bidx_batch, None, None], (self.chunk_size, self.M), (bidx_chunk, 0))
        cX = cute.local_tile(idX, (self.chunk_size, self.M), (bidx_chunk, 0))

        gD = cute.local_tile(mD[bidx_batch, None, None], (1, self.M), (bidx_chunk, 0))

        smem = cutlass.memory.SmemAllocator()
        log_dtype_bytes = cutlass.const_expr(int(math.log2(mX.element_type.width // 8)))
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout((self.chunk_size, self.M), order=(1, 0)),
            byte_alignment=16,
            swizzle=cute.make_swizzle(3, self.log_vecsize + log_dtype_bytes, 3)
        )

        # Copy from gmem to smem. Note that CopyG2SOp zeros the predicate=false positions, so no need to manually zero them 
        thr_copy_X = tiled_copy_g2s.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXpX = predicate_axis(tXcX, L, axis=0)
        tXsX = thr_copy_X.partition_D(sX)

        cute.copy(tiled_copy_g2s, tXgX, tXsX, pred=tXpX)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()


        # copy from smem to rmem.
        thr_copy_C = tiled_copy_r2g.get_slice(tidx)
        tCsX = thr_copy_C.partition_S(sX)
        tCrX = cute.make_rmem_tensor_like(tCsX)
        cute.autovec_copy(tCsX, tCrX)
        tCcX = thr_copy_C.partition_S(cX)

        # chunk level prefix sum
        chunk_total = cute.make_rmem_tensor((self.vecsize,), Float32)
        for v in cutlass.range_constexpr(self.vecsize):
            prefix_carry = 0.0
            # each warp level prefix sum can calculate for 32, so we need to iterate
            for row_step in cutlass.range_constexpr(self.rows_per_lane):
                x_val = tCrX[(v, row_step), 0, 0]
                prefix_val = warp_prefix_sum(x_val) + prefix_carry
                tCrX[(v, row_step), 0, 0] = prefix_val
                prefix_carry = cute.arch.shuffle_sync(prefix_val, offset=31)

            chunk_total[v] = prefix_carry

        # copying for rmem to gmem
        tCgY = thr_copy_C.partition_D(gY)
        tCpY = predicate_axis(tCcX, L, axis=0)
        cute.copy(tiled_copy_r2g, tCrX, tCgY, pred=tCpY)

        # store chunk decay
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if lane == 0:
            for v in cutlass.range_constexpr(self.vecsize):
                gD[0, self.vecsize * warp_idx + v] = chunk_total[v]

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, mD: cute.Tensor, stream: cuda.CUstream):
        B, L = mY.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)
        num_copy_bits = self.vecsize * mX.element_type.width
        copy_atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=num_copy_bits
        )
        copy_atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=num_copy_bits
        )

        thr_layout_g2s = cute.make_ordered_layout((cute.arch.WARP_SIZE, self.num_warps_M), order=(1, 0))
        val_layout_g2s = cute.make_ordered_layout((self.rows_per_lane, self.vecsize), order=(1, 0))
        tiled_copy_g2s = cute.make_tiled_copy_tv(copy_atom_g2s, thr_layout_g2s, val_layout_g2s)
        thr_layout_r2g = cute.make_layout(
            (cute.arch.WARP_SIZE, self.num_warps_M),
            stride=(1, self.vecsize * self.chunk_size),
        )
        val_layout_r2g = cute.make_layout(
            (self.vecsize, self.rows_per_lane),
            stride=(self.chunk_size, cute.arch.WARP_SIZE),
        )
        layout_tv_r2g = cute.make_layout(
            (thr_layout_r2g, val_layout_r2g)
        )
        tiled_copy_r2g = cute.make_tiled_copy(
            copy_atom_r2g, layout_tv_r2g, (self.chunk_size, self.M)
        )

        self.kernel(mX, mY, mD, tiled_copy_g2s, tiled_copy_r2g).launch(
            grid=[B, nchunks, 1],
            block=[self.num_threads, 1, 1],
            stream=stream
        )


class ChunkCumsumBwd:
    def __init__(self, dtype: type[cutlass.Numeric], M: int, chunk_size: int):
        self.M = M
        self.chunk_size = chunk_size

        self.vecsize = math.gcd(128 // dtype.width, M)
        self.num_warps_M = M // self.vecsize
        self.rows_per_lane = self.chunk_size // cute.arch.WARP_SIZE
        self.num_threads = cute.arch.WARP_SIZE * self.num_warps_M
        self.log_vecsize = int(math.log2(self.vecsize))

        assert self.chunk_size % cute.arch.WARP_SIZE == 0, \
            f"chunk_size must be a multiple of {cute.arch.WARP_SIZE}, got {chunk_size}"
        assert self.vecsize in (1, 2, 4), f"vecsize must be 1/2/4, got {self.vecsize}"

    @cute.kernel
    def kernel(
        self,
        mdY: cute.Tensor,    # (B, L, M) fp32
        mdD: cute.Tensor,    # (B, nchunks, M) fp32
        mdX: cute.Tensor,    # (B, L, M)
        tiled_copy_g2s: cute.TiledCopy,
        tiled_copy_r2g: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_batch, bidx_chunk, _ = cute.arch.block_idx()

        L = mdY.shape[1]

        idX = cute.make_identity_tensor(mdY[bidx_batch, None, None].shape)
        gdY = cute.local_tile(mdY[bidx_batch, None, None], (self.chunk_size, self.M), (bidx_chunk, 0))
        gdX = cute.local_tile(mdX[bidx_batch, None, None], (self.chunk_size, self.M), (bidx_chunk, 0))
        cX = cute.local_tile(idX, (self.chunk_size, self.M), (bidx_chunk, 0))

        gdD = cute.local_tile(mdD[bidx_batch, None, None], (1, self.M), (bidx_chunk, 0))

        smem = cutlass.memory.SmemAllocator()
        log_dtype_bytes = cutlass.const_expr(int(math.log2(mdY.element_type.width // 8)))
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout((self.chunk_size, self.M), order=(1, 0)),
            byte_alignment=16,
            swizzle=cute.make_swizzle(3, self.log_vecsize + log_dtype_bytes, 3),
        )

        # Copy from gmem to smem. Note that CopyG2SOp zeros the predicate=false positions, so no need to manually zero them
        thr_copy_dY = tiled_copy_g2s.get_slice(tidx)
        tdYgdY = thr_copy_dY.partition_S(gdY)
        tdYcdY = thr_copy_dY.partition_S(cX)
        tdYpdY = predicate_axis(tdYcdY, L, axis=0)
        tdYsdY = thr_copy_dY.partition_D(sdY)

        cute.copy(tiled_copy_g2s, tdYgdY, tdYsdY, pred=tdYpdY)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Copy from smem to rmem using the compute/output ownership mapping.
        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)
        tCsdY = thr_copy_r2g.partition_S(sdY)
        tCcX = thr_copy_r2g.partition_S(cX)
        tCrdX = cute.make_rmem_tensor_like(tCsdY)
        cute.autovec_copy(tCsdY, tCrdX)

        # Chunk-level suffix sum.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        for v in cutlass.range_constexpr(self.vecsize):
            suffix_carry = gdD[0, self.vecsize * warp_idx + v]

            for row_step_rev in cutlass.range_constexpr(self.rows_per_lane):
                row_step = self.rows_per_lane - 1 - row_step_rev
                dy_val = tCrdX[(v, row_step), 0, 0]
                suffix_val = warp_prefix_sum(dy_val, reverse=True) + suffix_carry
                tCrdX[(v, row_step), 0, 0] = suffix_val
                suffix_carry = cute.arch.shuffle_sync(suffix_val, offset=0)

        # copy from rmem to gmem
        tCpX = predicate_axis(tCcX, L, axis=0)
        tCgdX = thr_copy_r2g.partition_D(gdX)
        cute.copy(tiled_copy_r2g, tCrdX, tCgdX, pred=tCpX)

    @cute.jit
    def __call__(self, mdY: cute.Tensor, mdD: cute.Tensor, mdX: cute.Tensor, stream: cuda.CUstream):
        B, L = mdY.shape[:2]
        nchunks = cute.ceil_div(L, self.chunk_size)
        num_copy_bits = self.vecsize * mdY.element_type.width
        copy_atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=num_copy_bits
        )
        copy_atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=num_copy_bits
        )

        thr_layout_g2s = cute.make_ordered_layout((cute.arch.WARP_SIZE, self.num_warps_M), order=(1, 0))
        val_layout_g2s = cute.make_ordered_layout((self.rows_per_lane, self.vecsize), order=(1, 0))
        tiled_copy_g2s = cute.make_tiled_copy_tv(copy_atom_g2s, thr_layout_g2s, val_layout_g2s)
        thr_layout_r2g = cute.make_layout(
            (cute.arch.WARP_SIZE, self.num_warps_M),
            stride=(1, self.vecsize * self.chunk_size),
        )
        val_layout_r2g = cute.make_layout(
            (self.vecsize, self.rows_per_lane),
            stride=(self.chunk_size, cute.arch.WARP_SIZE),
        )
        layout_tv_r2g = cute.make_layout(
            (thr_layout_r2g, val_layout_r2g)
        )
        tiled_copy_r2g = cute.make_tiled_copy(
            copy_atom_r2g, layout_tv_r2g, (self.chunk_size, self.M)
        )

        self.kernel(mdY, mdD, mdX, tiled_copy_g2s, tiled_copy_r2g).launch(
            grid=[B, nchunks, 1],
            block=[self.num_threads, 1, 1],
            stream=stream
        )


@jit_cache
def _compile_chunk_cumsum_fwd(dtype, M, chunk_size):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()

    fake_log_rho = make_fake_tensor(dtype, (B_sym, L_sym, M), divisibility=4)
    fake_log_cumsum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=4)
    fake_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks_sym, M), divisibility=4)
    return cute.compile(
        ChunkCumsumFwd(dtype, M=M, chunk_size=chunk_size),
        fake_log_rho, fake_log_cumsum_rho, fake_chunk_decay,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _frac_chunk_cumsum_fwd(log_rho, chunk_size):
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)

    log_cumsum_rho = torch.empty((B, L, M), dtype=torch.float32, device=log_rho.device)
    chunk_decay = torch.empty((B, nchunks, M), dtype=torch.float32, device=log_rho.device)

    if B == 0 or L == 0:
        return log_cumsum_rho, chunk_decay

    dtype = Float32#torch2cute_dtype_map[log_rho.dtype]
    _compile_chunk_cumsum_fwd(dtype, M, chunk_size)(
        log_rho, log_cumsum_rho, chunk_decay,
    )

    return log_cumsum_rho, chunk_decay
@jit_cache
def _compile_chunk_cumsum_bwd(dtype, M, chunk_size):
    B_sym = cute.sym_int()
    L_sym = cute.sym_int()
    nchunks_sym = cute.sym_int()
    fake_d_log_cumsum_rho = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=4)
    fake_d_chunk_decay = make_fake_tensor(Float32, (B_sym, nchunks_sym, M), divisibility=4)
    fake_d_log_rho = make_fake_tensor(Float32, (B_sym, L_sym, M), divisibility=4)
    return cute.compile(
        ChunkCumsumBwd(dtype, M=M, chunk_size=chunk_size),
        fake_d_log_cumsum_rho, fake_d_chunk_decay, fake_d_log_rho,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _frac_chunk_cumsum_bwd(d_log_cumsum_rho, d_chunk_decay, chunk_size):
    B, L, M = d_log_cumsum_rho.shape

    d_log_rho = torch.empty((B, L, M), dtype=torch.float32, device=d_log_cumsum_rho.device)

    if B == 0 or L == 0:
        return d_log_rho

    
    dtype = Float32#torch2cute_dtype_map[d_log_cum_rho.dtype]

    _compile_chunk_cumsum_bwd(dtype, M, chunk_size)(
        d_log_cumsum_rho, d_chunk_decay, d_log_rho
    )

    return d_log_rho


@torch.library.custom_op("frac_ssm::chunk_cumsum_fwd_cute", mutates_args=())
def _chunk_cumsum_fwd_op(
    log_rho: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return _frac_chunk_cumsum_fwd(log_rho, chunk_size)


@_chunk_cumsum_fwd_op.register_fake
def _(log_rho, chunk_size):
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)
    return (
        torch.empty(B, L, M, dtype=torch.float32, device=log_rho.device),
        torch.empty(B, nchunks, M, dtype=torch.float32, device=log_rho.device),
    )


@torch.library.custom_op("frac_ssm::chunk_cumsum_bwd_cute", mutates_args=())
def _chunk_cumsum_bwd_op(
    d_log_cumsum_rho: torch.Tensor, d_chunk_decay: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _frac_chunk_cumsum_bwd(d_log_cumsum_rho, d_chunk_decay, chunk_size)


@_chunk_cumsum_bwd_op.register_fake
def _(d_log_cumsum_rho, d_chunk_decay, chunk_size):
    B, L, M = d_log_cumsum_rho.shape
    return torch.empty(B, L, M, dtype=torch.float32, device=d_log_cumsum_rho.device)


def _chunk_cumsum_setup_ctx(ctx, inputs, output):
    log_rho, chunk_size = inputs
    ctx.chunk_size = chunk_size
    ctx.input_dtype = log_rho.dtype


def _chunk_cumsum_backward(ctx, d_log_cumsum_rho, d_chunk_decay):
    d_log_rho = _chunk_cumsum_bwd_op(
        d_log_cumsum_rho.contiguous(), d_chunk_decay.contiguous(),
        ctx.chunk_size,
    )
    return d_log_rho.to(ctx.input_dtype), None


_chunk_cumsum_fwd_op.register_autograd(_chunk_cumsum_backward, setup_context=_chunk_cumsum_setup_ctx)


def chunk_cumsum(log_rho, chunk_size):
    """Per-chunk log-cumsum (swizzled-smem variant)."""
    assert log_rho.dim() == 3, f"log_rho must be 3D (B, L, M), got {log_rho.dim()}D"
    assert log_rho.is_cuda, "log_rho must be on CUDA"
    assert log_rho.dtype == torch.float32, f"log_rho dtype must be float32, got {log_rho.dtype}"
    assert isinstance(chunk_size, int) and (chunk_size > 0), f"chunk_size must be a positive integer, got {chunk_size}"
    return _chunk_cumsum_fwd_op(log_rho, chunk_size)


if __name__ == "__main__":
    chunk_size = 64
    B, L, M = 2, 4096, 16
    device = "cuda"
    log_rho = torch.randn(B, L, M, device=device, dtype=torch.float32)

    for _ in range(10):
        chunk_cumsum(log_rho, chunk_size)
        break
    # torch.cuda.profiler.start()
    # chunk_cumsum(log_rho, chunk_size)
    # torch.cuda.profiler.stop()
