import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float16, BFloat16, Float32, TFloat32, Boolean
from cutlass import const_expr
from cutlass.cutlass_dsl import dsl_user_op

torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@dsl_user_op
def cvt_copy_r2s(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    loc=None,
    ip=None,
) -> None:
    if cutlass.const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type, loc=loc, ip=ip)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    cute.copy(tiled_copy, tiled_copy.retile(src), dst, loc=loc, ip=ip)


def make_fake_tensor(dtype, shape, *, divisibility=1, leading_dim=-1):
    """Build a CuTe fake tensor as a compile-time template.

    The tensor is treated as contiguous on `leading_dim` (stride = 1 there);
    every other dim is a dynamic int64 stride asserted to be a multiple of
    `divisibility` elements. `assumed_align` is set so the base-pointer byte
    alignment matches the same vector width.

    Args:
        dtype: A `cutlass.Numeric` subclass (e.g. `Float32`).
        shape: Tuple of ints / `cute.sym_int()` describing the logical shape.
        divisibility: Vector width in *elements* (not bytes). For 128-bit
            loads use 4 for fp32, 8 for fp16/bf16, 2 for fp64. Defaults to 1
            (no vectorization promise, scalar loads only).
        leading_dim: Index of the contiguous dim. Defaults to last.
    """
    rank = len(shape)
    if leading_dim < 0:
        leading_dim += rank
    stride = tuple(
        1 if i == leading_dim else cute.sym_int64(divisibility=divisibility)
        for i in range(rank)
    )
    align_bytes = max(1, divisibility * dtype.width // 8)
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=align_bytes,
    )


@cute.jit
def predicate_axis(
    tAcA: cute.Tensor,                  # ((V_inner, V_outer), RestM, RestN) of (row, col) coords
    limit: Int32,
    axis: cutlass.Constexpr[int],       # 0 → predicate row coord, 1 → col coord
) -> cute.Tensor:
    # One bool per atom call (V_inner shares one predicate — broadcast away).
    # Of the two Rest modes, the one orthogonal to `axis` is broadcast via stride 0,
    # since the predicated coord is constant when stepping along that mode.
    V_outer = cute.size(tAcA, mode=[0, 1])
    RestM   = cute.size(tAcA, mode=[1])
    RestN   = cute.size(tAcA, mode=[2])
    if const_expr(axis == 0):
        # row predicate: varies along RestM (rows), constant along RestN (cols)
        stride = (RestM, 1, 0)
    else:
        # col predicate: varies along RestN (cols), constant along RestM (rows)
        stride = (RestN, 0, 1)
    tApA = cute.make_rmem_tensor(
        cute.make_layout((V_outer, RestM, RestN), stride=stride),
        Boolean,
    )                                                  # (V_outer, RestM, RestN) bool
    for v_outer in cutlass.range_constexpr(V_outer):
        if const_expr(axis == 0):
            for rest_m in cutlass.range_constexpr(RestM):
                # (0, v_outer) → V_inner=0, V_outer=v_outer; rest_m → RestM; 0 → RestN; [0] → row
                tApA[v_outer, rest_m, 0] = cute.elem_less(tAcA[(0, v_outer), rest_m, 0][0], limit)
        else:
            for rest_n in cutlass.range_constexpr(RestN):
                # (0, v_outer) → V_inner=0, V_outer=v_outer; 0 → RestM; rest_n → RestN; [1] → col
                tApA[v_outer, 0, rest_n] = cute.elem_less(tAcA[(0, v_outer), 0, rest_n][1], limit)
    return tApA


class MmaTF32Op(cute.nvgpu.warpgroup.MmaF16BF16Op):
    """SM90 warpgroup MMA with TF32 inputs and FP32 accumulation.

    CuTe DSL has no public TF32 warpgroup op; the MLIR atom supports it, and only
    MmaF16BF16Op's Python-side dtype check rejects it. This lowers to
    ``wgmma.mma_async.*.f32.tf32.tf32`` (Mma-K = 8). Both operands must be K-major
    when read from shared memory.
    """

    descriptive_name = "warpgroup TF32 MMA Operation"

    def __init__(self, instruction_shape, a_src, a_major_mode, b_major_mode):
        super().__init__(TFloat32, Float32, instruction_shape, a_src, a_major_mode, b_major_mode)

    def _verify(self) -> None:
        shape_mnk = tuple(self.shape_mnk)
        if len(shape_mnk) == 2:
            object.__setattr__(self, "shape_mnk", (*shape_mnk, 8))
        assert self.shape_mnk[2] == 8, f"TF32 WGMMA needs Mma-K = 8, got {self.shape_mnk[2]}"


def tf32_smem_view(storage_field, smem_layout: cute.ComposedLayout) -> cute.Tensor:
    """View FP32 shared-memory storage as a swizzled TF32 WGMMA operand.

    Recasting a swizzled tensor's iterator drops the swizzle, so this rebuilds the
    view from the storage base pointer, the swizzle, and the outer layout.
    """
    ptr = cute.recast_ptr(storage_field.data_ptr(), smem_layout.inner, dtype=TFloat32)
    return cute.make_tensor(ptr, smem_layout.outer)


def tf32_rmem_view(fragment: cute.Tensor) -> cute.Tensor:
    """View an FP32 register fragment as a TF32 WGMMA operand without converting it."""
    return cute.make_tensor(cute.recast_ptr(fragment.iterator, dtype=TFloat32), fragment.layout)


## Used for personal debugging
import contextlib
import io
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from cutlass.utils import print_latex, print_latex_tv
import os 

PLOTS_DIR =  Path(os.getcwd()) / "plots" #Path(__file__).parent / "plots"


def plot_layout(layout, name=None):
    """Render a layout to ``ops/cute/plots/{name}.jpg``.

    ``print_latex`` only emits the LaTeX source for a color-coded TiKZ table;
    this captures that source, compiles it with ``pdflatex``, and rasterizes
    the resulting PDF to a JPG. ``name`` defaults to the current timestamp.
    """
    # Capture the LaTeX document print_latex writes to stdout.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_latex(layout)
    tex = buf.getvalue()

    # The [convert] option asks standalone to auto-rasterize via ImageMagick
    # (needs --shell-escape); we rasterize ourselves, so drop it.
    tex = tex.replace("[convert]{standalone}", "{standalone}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = PLOTS_DIR / f"{name}.jpg"

    # Call pdflatex directly so we can surface its log on failure — the
    # PDFLaTeX library raises FileNotFoundError when compilation fails,
    # which hides the actual LaTeX error.
    with tempfile.TemporaryDirectory() as td:
        tex_path = Path(td) / "layout.tex"
        tex_path.write_text(tex)
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", td, str(tex_path)],
            capture_output=True,
        )
        pdf_path = Path(td) / "layout.pdf"
        if not pdf_path.exists():
            log = (proc.stdout + proc.stderr).decode(errors="replace")
            tex_dump = Path("/tmp") / f"plot_layout_{name}.tex"
            tex_dump.write_text(tex)
            raise RuntimeError(
                f"pdflatex failed for layout '{name}'. TeX dumped to {tex_dump}.\n"
                f"--- pdflatex log (tail) ---\n{log[-2000:]}"
            )

        # pdftoppm appends the .jpg extension itself, so pass the stem.
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", "300", "-singlefile",
             str(pdf_path), str(out_path.with_suffix(""))],
            check=True,
            capture_output=True,
        )

    # print(f"saved {out_path}")
    return out_path


def _layout_rank(layout):
    try:
        return cute.rank(layout)
    except Exception:
        return len(cute.shape(layout))


def _layout_size(layout):
    try:
        return cute.size(layout)
    except Exception:
        size = 1
        for dim in cute.shape(layout):
            size *= dim
        return size


def debug_layouts(layouts):
    for name, layout in layouts.items():
        try:
            cute.printf(f"{name}"+": {}", layout)
        except TypeError:
            # cute.printf only formats layouts/scalars (e.g. TiledCopy is
            # rejected on cutlass-dsl 4.6.1); fall back to the host repr
            print(f"{name}: {layout}")
        try:
            if _layout_rank(layout) <= 2 and _layout_size(layout) <= 4096:
                if (PLOTS_DIR / f"{name}.jpg").exists():
                    continue
                plot_layout(layout, name)
        except Exception:
            pass
