import importlib
import math

import pytest
import torch

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref
from frac_ssm.ops.refs.chunk_readout import chunk_readout_ref
from frac_ssm.ops.refs.chunk_state import chunk_state_ref
from frac_ssm.ops.refs.chunk_w import chunk_w_ref
from frac_ssm.ops.refs.state_passing import state_passing_ref
from frac_ssm.ops.registry import available
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "cute_sm90": "frac_ssm.ops.cute.chunk_fused_readout",
    "cute_sm90_warp_specialized": "frac_ssm.ops.cute.variants.chunk_fused_readout_warp_specialized",
    "cute_sm90_cooperative": "frac_ssm.ops.cute.variants.chunk_fused_readout_cooperative",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def fused_readout_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    backend = request.param
    ok, reason = available("cute_sm90")
    if not ok:
        pytest.skip(reason or f"Backend {backend!r} is unavailable")

    return importlib.import_module(_IMPL_MODULES[backend])


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 1e-5


def _make_inputs(
    B, L, R, M, chunk_size, dtype, input_gen, *, seed=0, device="cuda",
):
    g = torch.Generator(device=device).manual_seed(seed)
    u = 0.25 * torch.randn(B, L, R, device=device, dtype=dtype, generator=g)
    c = 0.25 * torch.randn(B, L, M, device=device, dtype=dtype, generator=g)
    kappa = 0.25 * torch.randn(
        B, L, M, device=device, dtype=dtype, generator=g,
    )
    log_rho = LOG_RHO_GENS[input_gen](
        B, L, M, seed=seed + 1, device=device,
    )
    log_cum_rho, chunk_decay = chunk_cumsum_ref(log_rho, chunk_size)

    if input_gen == "random":
        states_acc = 0.25 * torch.randn(
            B,
            math.ceil(L / chunk_size),
            R,
            M,
            device=device,
            dtype=torch.float32,
            generator=g,
        )
    else:
        states = chunk_state_ref(
            u, kappa, log_cum_rho, chunk_size, upcast=True,
        )
        states_acc, _ = state_passing_ref(states, chunk_decay)
    return u, c, kappa, states_acc, log_cum_rho


def _check_fwd_bwd(
    impl, inputs, chunk_size, *, seed=0, use_compile=False,
):
    u, c, kappa, states_acc, log_cum_rho = (
        x.detach().requires_grad_() for x in inputs
    )

    u_r = u.detach().float().requires_grad_()
    c_r = c.detach().float().requires_grad_()
    kappa_r = kappa.detach().float().requires_grad_()
    states_acc_r = states_acc.detach().float().requires_grad_()
    lcr_r = log_cum_rho.detach().float().requires_grad_()

    u_pt = u.detach().clone().requires_grad_()
    c_pt = c.detach().clone().requires_grad_()
    kappa_pt = kappa.detach().clone().requires_grad_()
    states_acc_pt = states_acc.detach().clone().requires_grad_()
    lcr_pt = log_cum_rho.detach().clone().requires_grad_()

    fn = impl.chunk_fused_readout
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    h = fn(u, c, kappa, states_acc, log_cum_rho, chunk_size)
    W_ref = chunk_w_ref(c_r, kappa_r, lcr_r, chunk_size, upcast=True)
    h_ref = chunk_readout_ref(
        u_r, c_r, states_acc_r, W_ref, lcr_r, chunk_size, upcast=True,
    )
    W_pt = chunk_w_ref(c_pt, kappa_pt, lcr_pt, chunk_size, upcast=False)
    h_pt = chunk_readout_ref(
        u_pt, c_pt, states_acc_pt, W_pt, lcr_pt, chunk_size, upcast=False,
    )

    assert h.dtype == u.dtype
    _assert_ref_error(h, h_pt, h_ref)

    g = torch.Generator(device=h.device).manual_seed(seed + 17)
    d_h = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g)
    inputs = (u, c, kappa, states_acc, log_cum_rho)
    inputs_r = (u_r, c_r, kappa_r, states_acc_r, lcr_r)
    inputs_pt = (u_pt, c_pt, kappa_pt, states_acc_pt, lcr_pt)

    grads = torch.autograd.grad(h, inputs, grad_outputs=d_h)
    grads_ref = torch.autograd.grad(
        h_ref, inputs_r, grad_outputs=d_h.float(),
    )
    grads_pt = torch.autograd.grad(h_pt, inputs_pt, grad_outputs=d_h)

    for grad, grad_pt, grad_ref, x in zip(
        grads, grads_pt, grads_ref, inputs,
    ):
        assert grad.dtype == x.dtype
        _assert_ref_error(grad, grad_pt, grad_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen", list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("L", [32, 4001, 4096, 32768])
@pytest.mark.parametrize("M", [16])
@pytest.mark.parametrize("R", [64, 128, 192])
@pytest.mark.parametrize("B", [1, 17, 256])
def test_chunk_fused_readout(
    fused_readout_impl, B, R, M, L, chunk_size, dtype, input_gen,
):
    try:
        inputs = _make_inputs(
            B, L, R, M, chunk_size, dtype, input_gen,
        )
        _check_fwd_bwd(fused_readout_impl, inputs, chunk_size)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping chunk_fused_readout case after CUDA OOM")


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_chunk_fused_readout_compile(fused_readout_impl, chunk_size):
    inputs = _make_inputs(
        2, 4001, 64, 16, chunk_size, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(fused_readout_impl, inputs, chunk_size, use_compile=True)


@pytest.mark.parametrize("chunk_size", [32, 64])
def test_M_eight(fused_readout_impl, chunk_size):
    # M=8 is supported through the kernel's padded M=16 path.
    inputs = _make_inputs(
        2, 257, 64, 8, chunk_size, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(fused_readout_impl, inputs, chunk_size)


@pytest.mark.parametrize("chunk_size", [32, 64])
def test_large_R(fused_readout_impl, chunk_size):
    inputs = _make_inputs(
        2, 257, 512, 16, chunk_size, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(fused_readout_impl, inputs, chunk_size)


# ============================================================================
# Supported intermediate dtypes
# ============================================================================

# Inference may supply BF16 accumulated states even though backward uses FP32.
def test_bf16_states(fused_readout_impl):
    inputs = list(_make_inputs(
        2, 257, 64, 16, 64, torch.bfloat16, "model_like", seed=15,
    ))
    inputs[3] = inputs[3].to(torch.bfloat16)
    _check_fwd_bwd(fused_readout_impl, tuple(inputs), 64, seed=15)


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(fused_readout_impl):
    inputs = _make_inputs(
        2, 1, 64, 16, 64, torch.bfloat16, "model_like", seed=7,
    )
    _check_fwd_bwd(fused_readout_impl, inputs, 64, seed=7)


def test_zero_batch(fused_readout_impl):
    L, R, M, chunk_size = 128, 64, 16, 64
    nchunks = math.ceil(L / chunk_size)
    inputs = (
        torch.empty(
            0, L, R, device="cuda", dtype=torch.bfloat16, requires_grad=True,
        ),
        torch.empty(
            0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
        ),
        torch.empty(
            0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
        ),
        torch.empty(
            0, nchunks, R, M,
            device="cuda", dtype=torch.float32, requires_grad=True,
        ),
        torch.empty(
            0, L, M, device="cuda", dtype=torch.float32, requires_grad=True,
        ),
    )

    h = fused_readout_impl.chunk_fused_readout(*inputs, chunk_size)
    assert h.shape == (0, L, R)
    assert h.dtype == torch.bfloat16

    torch.autograd.backward(h, torch.empty_like(h))
    for x in inputs:
        assert x.grad.shape == x.shape


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_strided_inputs(fused_readout_impl, chunk_size):
    B, L, R, M = 2, 4001, 64, 16
    inputs = _make_inputs(
        B, L, R, M, chunk_size, torch.bfloat16, "model_like", seed=13,
    )
    u, c, kappa, states_acc, log_cum_rho = inputs
    nchunks = math.ceil(L / chunk_size)

    u_storage = torch.empty(B, L, R + 4, device="cuda", dtype=u.dtype)
    c_storage = torch.empty(B, L, M + 5, device="cuda", dtype=c.dtype)
    kappa_storage = torch.empty(B, L, M + 5, device="cuda", dtype=kappa.dtype)
    states_storage = torch.empty(
        B, nchunks, R, M + 5, device="cuda", dtype=states_acc.dtype,
    )
    lcr_storage = torch.empty(
        B, L, M + 5, device="cuda", dtype=log_cum_rho.dtype,
    )
    u_storage[..., :R].copy_(u)
    c_storage[..., :M].copy_(c)
    kappa_storage[..., :M].copy_(kappa)
    states_storage[..., :M].copy_(states_acc)
    lcr_storage[..., :M].copy_(log_cum_rho)

    inputs = (
        u_storage[..., :R],
        c_storage[..., :M],
        kappa_storage[..., :M],
        states_storage[..., :M],
        lcr_storage[..., :M],
    )
    for x in inputs:
        assert not x.is_contiguous()
    _check_fwd_bwd(fused_readout_impl, inputs, chunk_size, seed=13)


def test_input_validation(fused_readout_impl):
    inputs = _make_inputs(
        2, 32, 64, 16, 64, torch.bfloat16, "random",
    )
    u, c, kappa, states_acc, log_cum_rho = inputs

    with pytest.raises(AssertionError, match="3D"):
        fused_readout_impl.chunk_fused_readout(
            u[0], c, kappa, states_acc, log_cum_rho, 64,
        )
    with pytest.raises(AssertionError, match="CUDA"):
        fused_readout_impl.chunk_fused_readout(
            u.cpu(), c.cpu(), kappa.cpu(), states_acc.cpu(),
            log_cum_rho.cpu(), 64,
        )
    with pytest.raises(AssertionError, match="states_acc dtype"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc.to(torch.float16), log_cum_rho, 64,
        )
    with pytest.raises(AssertionError, match="chunk_size"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc, log_cum_rho, 16,
        )
    with pytest.raises(AssertionError, match="log_cum_rho dtype"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc, log_cum_rho.to(torch.bfloat16), 64,
        )
    with pytest.raises(AssertionError, match="chunk_size"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc, log_cum_rho, 0,
        )
    with pytest.raises(AssertionError, match="c and kappa shapes"):
        fused_readout_impl.chunk_fused_readout(
            u, c[:, :-1], kappa, states_acc, log_cum_rho, 64,
        )
    with pytest.raises(AssertionError, match="log_cum_rho shape"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc, log_cum_rho[:, :-1], 64,
        )
    with pytest.raises(AssertionError, match="states_acc shape"):
        fused_readout_impl.chunk_fused_readout(
            u, c, kappa, states_acc[:, :, :-1], log_cum_rho, 64,
        )


# Explicit configurations cover tile layouts and buffer reuse without autotuning.
@pytest.fixture
def direct_kernel_checks(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ok, reason = available("cute_sm90")
    if not ok:
        pytest.skip(reason or "SM90 CUDA device required")

    from frac_ssm.ops.cute.autotuner import Autotuner

    def unexpected_autotune(*args, **kwargs):
        pytest.fail("Direct kernel checks must supply an explicit configuration")

    monkeypatch.setattr(Autotuner, "__call__", unexpected_autotune)


def _config_inputs(chunk_size, dtype, R, M, *, native_states=False, input_gen="random"):
    # Two batches exercise batch indexing. L=257 leaves one token in the final
    # chunk for each tested chunk size of 32, 64, or 128.
    inputs = list(_make_inputs(2, 257, R, M, chunk_size, dtype, input_gen, seed=257))
    # Cooperative forward requires native states. Other kernels also cover FP32 states.
    if native_states or dtype == torch.float16:
        inputs[3] = inputs[3].to(dtype)
    return tuple(inputs)


def _reference_readout(inputs, chunk_size, *, upcast):
    u, c, kappa, states_acc, lcr = inputs
    w = chunk_w_ref(c, kappa, lcr, chunk_size, upcast=upcast)
    return chunk_readout_ref(u, c, states_acc, w, lcr, chunk_size, upcast=upcast)


def _check_fwd_config(kernel, inputs, chunk_size, config):
    out = kernel(*inputs, chunk_size, config=config)
    out_ref = _reference_readout(tuple(x.float() for x in inputs), chunk_size, upcast=True)
    out_pt = _reference_readout(inputs, chunk_size, upcast=False)
    assert out.dtype == inputs[0].dtype
    _assert_ref_error(out, out_pt, out_ref)


def _check_bwd_config(kernel, inputs, chunk_size, config):
    generator = torch.Generator(device=inputs[0].device).manual_seed(17)
    dh = torch.randn(
        inputs[0].shape, device=inputs[0].device, dtype=inputs[0].dtype, generator=generator,
    )
    grads = kernel(dh, *inputs, chunk_size, config=config)
    references = []
    for upcast in (True, False):
        xs = tuple((x.float() if upcast else x).detach().requires_grad_() for x in inputs)
        out = _reference_readout(xs, chunk_size, upcast=upcast)
        references.append(torch.autograd.grad(out, xs, dh.float() if upcast else dh))
    grads_ref, grads_pt = references
    for name, grad, grad_pt, grad_ref, x in zip(
        ("du", "dc", "dk", "dstates", "dlcr"), grads, grads_pt, grads_ref, inputs, strict=True,
    ):
        assert grad.dtype == x.dtype, name
        try:
            _assert_ref_error(grad, grad_pt, grad_ref)
        except AssertionError as exc:
            raise AssertionError(f"{name} failed for {config}") from exc


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,num_producer_threads,work_slots,work_group,STATE_R_TILE,producer_regs",
    [
        pytest.param(32, torch.bfloat16, 64, 1, 1, None, 0, id="32-bf16-p64-w1-g1-full-state"),
        pytest.param(32, torch.float16, 128, 2, 1, 64, 0, id="32-fp16-p128-w2-g1-state64"),
        pytest.param(64, torch.bfloat16, 192, 1, 2, 64, 0, id="64-bf16-p192-w1-g2-state64"),
        pytest.param(64, torch.float16, 256, 2, 2, None, 0, id="64-fp16-p256-w2-g2-full-state"),
        pytest.param(128, torch.bfloat16, 128, 2, 2, 64, 96, id="128-bf16-w2-g2-registers"),
        pytest.param(128, torch.float16, 128, 1, 1, None, 0, id="128-fp16-w1-g1-full-state"),
    ],
)
def test_persistent_fwd_buffer_reuse(
    chunk_size, dtype, num_producer_threads, work_slots, work_group, STATE_R_TILE, producer_regs,
):
    from frac_ssm.ops.cute._chunk_fused_readout import persistent_fwd

    config = persistent_fwd.ChunkFusedReadoutFwdConfig(
        num_producer_threads=num_producer_threads, min_blocks_per_mp=1, max_ctas=3,
        work_slots=work_slots, work_group=work_group, STATE_R_TILE=STATE_R_TILE,
        producer_regs=producer_regs, consumer_regs=64 if producer_regs else 0,
    )
    # Three CTAs repeatedly reuse slots across batches, R tiles, and the final partial chunk.
    # R=512 needs three 192-channel tiles, with 128 channels in the last tile.
    # M=8 exercises the padded 16-column state operands.
    inputs = _config_inputs(chunk_size, dtype, R=512, M=8)
    _check_fwd_config(persistent_fwd._frac_chunk_fused_readout_fwd, inputs, chunk_size, config)


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,num_producer_threads,num_w_stages,num_dde_stages,"
    "num_threads_Q,R_TILE,STATE_R_TILE",
    [
        pytest.param(32, torch.bfloat16, 96, 1, 1, 16, 64, 64, id="32-bf16-p96-w1-d1-q16"),
        pytest.param(32, torch.float16, 128, 2, 2, 32, 128, 128, id="32-fp16-p128-w2-d2-q32"),
        pytest.param(64, torch.bfloat16, 192, 1, 2, 16, 64, 128, id="64-bf16-p192-w1-d2-q16"),
        pytest.param(64, torch.float16, 224, 2, 1, 32, 128, 64, id="64-fp16-p224-w2-d1-q32"),
        pytest.param(128, torch.bfloat16, 128, 1, 2, 16, 64, 64, id="128-bf16-p128-w1-d2-q16"),
        pytest.param(128, torch.float16, 96, 1, 1, 32, 128, 128, id="128-fp16-p96-w1-d1-q32"),
    ],
)
def test_persistent_bwd_buffer_reuse(
    chunk_size, dtype, num_producer_threads, num_w_stages, num_dde_stages,
    num_threads_Q, R_TILE, STATE_R_TILE,
):
    from frac_ssm.ops.cute._chunk_fused_readout import persistent_bwd

    config = persistent_bwd.ChunkFusedReadoutBwdConfig(
        R_TILE=R_TILE, STATE_R_TILE=STATE_R_TILE,
        num_producer_threads=num_producer_threads, min_blocks_per_mp=1, max_ctas=3,
        num_w_stages=num_w_stages, num_dde_stages=num_dde_stages, num_threads_Q=num_threads_Q,
    )
    # Three CTAs repeatedly reuse both pipelines. R=96 leaves a partial tile
    # for either R width of 64 or 128, and M=8 exercises padded state columns.
    inputs = _config_inputs(chunk_size, dtype, R=96, M=8, input_gen="model_like")
    _check_bwd_config(persistent_bwd._frac_chunk_fused_readout_bwd, inputs, chunk_size, config)


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,R_TILE,num_warp_groups",
    [
        pytest.param(32, torch.bfloat16, 64, 1, id="32-bf16-r64-wg1"),
        pytest.param(32, torch.float16, 128, 2, id="32-fp16-r128-wg2"),
        pytest.param(64, torch.bfloat16, 192, 1, id="64-bf16-r192-wg1"),
        pytest.param(64, torch.float16, 256, 4, id="64-fp16-r256-wg4"),
        pytest.param(128, torch.bfloat16, 128, 2, id="128-bf16-r128-wg2"),
        pytest.param(128, torch.float16, 256, 4, id="128-fp16-r256-wg4"),
    ],
)
def test_cooperative_fwd_configs(chunk_size, dtype, R_TILE, num_warp_groups):
    from frac_ssm.ops.cute._chunk_fused_readout import cooperative_fwd

    config = cooperative_fwd.ChunkFusedReadoutFwdConfig(
        R_TILE=R_TILE, num_warp_groups=num_warp_groups, min_blocks_per_mp=1,
    )
    # R=192 fills 64- and 192-channel tiles and leaves a partial tile at widths 128 and 256.
    inputs = _config_inputs(chunk_size, dtype, R=192, M=16, native_states=True)
    _check_fwd_config(cooperative_fwd._frac_chunk_fused_readout_fwd, inputs, chunk_size, config)


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,R_TILE,num_warp_groups,PARAM_R_TILE,STATE_R_TILE,num_producer_threads",
    [
        pytest.param(32, torch.bfloat16, 64, 1, 32, 64, 32, id="32-bf16-r64-wg1-p32"),
        pytest.param(32, torch.float16, 128, 2, 64, 128, 128, id="32-fp16-r128-wg2-p128"),
        pytest.param(64, torch.bfloat16, 192, 1, 128, 64, 384, id="64-bf16-r192-wg1-p384"),
        pytest.param(64, torch.float16, 128, 1, 32, 128, 32, id="64-fp16-r128-wg1-p32"),
        pytest.param(128, torch.bfloat16, 128, 2, 64, 64, 128, id="128-bf16-r128-wg2-p128"),
        pytest.param(128, torch.float16, 192, 1, 128, 128, 384, id="128-fp16-r192-wg1-p384"),
    ],
)
def test_cooperative_bwd_configs(
    chunk_size, dtype, R_TILE, num_warp_groups, PARAM_R_TILE, STATE_R_TILE, num_producer_threads,
):
    from frac_ssm.ops.cute._chunk_fused_readout import cooperative_bwd

    config = (
        cooperative_bwd.ChunkFusedReadoutBwdDuConfig(
            R_TILE=R_TILE, num_warp_groups=num_warp_groups, min_blocks_per_mp=1,
        ),
        cooperative_bwd.ChunkFusedReadoutBwdParamsConfig(
            R_TILE=PARAM_R_TILE, num_producer_threads=num_producer_threads, max_ctas=3,
        ),
        cooperative_bwd.ChunkFusedReadoutBwdStateConfig(
            R_TILE=STATE_R_TILE, num_producer_threads=num_producer_threads, min_blocks_per_mp=1,
        ),
    )
    # R=192 fills the 32-, 64-, and 192-channel tiles but leaves a partial 128-channel tile.
    # The three-CTA cap also forces the parameter stage to reuse its buffers across chunks.
    inputs = _config_inputs(chunk_size, dtype, R=192, M=16)
    _check_bwd_config(cooperative_bwd._frac_chunk_fused_readout_bwd, inputs, chunk_size, config)


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,R_TILE,num_producer_threads",
    [
        pytest.param(32, torch.bfloat16, 64, 64, id="32-bf16-r64-p64"),
        pytest.param(32, torch.float16, 128, 128, id="32-fp16-r128-p128"),
        pytest.param(64, torch.bfloat16, 192, 192, id="64-bf16-r192-p192"),
        pytest.param(64, torch.float16, 256, 256, id="64-fp16-r256-p256"),
        pytest.param(128, torch.bfloat16, 128, 64, id="128-bf16-r128-p64"),
        pytest.param(128, torch.float16, 192, 256, id="128-fp16-r192-p256"),
    ],
)
def test_warp_specialized_fwd_configs(chunk_size, dtype, R_TILE, num_producer_threads):
    from frac_ssm.ops.cute._chunk_fused_readout import warp_specialized_fwd

    config = warp_specialized_fwd.ChunkFusedReadoutFwdConfig(
        R_TILE=R_TILE, num_producer_threads=num_producer_threads, min_blocks_per_mp=1,
    )
    # R=320 is five 64-channel subtiles. All four tile widths are eligible,
    # with partial final tiles at widths 128, 192, and 256.
    inputs = _config_inputs(chunk_size, dtype, R=320, M=16)
    _check_fwd_config(
        warp_specialized_fwd._frac_chunk_fused_readout_fwd, inputs, chunk_size, config,
    )


@pytest.mark.usefixtures("direct_kernel_checks")
@pytest.mark.parametrize(
    "chunk_size,dtype,R_TILE,STATE_R_TILE,num_producer_threads,num_threads_Q",
    [
        pytest.param(32, torch.bfloat16, 64, 64, 64, 16, id="32-bf16-r64-p64-q16"),
        pytest.param(32, torch.float16, 128, 128, 128, 32, id="32-fp16-r128-p128-q32"),
        pytest.param(64, torch.bfloat16, 64, 128, 192, 16, id="64-bf16-r64-p192-q16"),
        pytest.param(64, torch.float16, 128, 64, 256, 32, id="64-fp16-r128-p256-q32"),
        pytest.param(128, torch.bfloat16, 64, 64, 64, 16, id="128-bf16-r64-p64-q16"),
        pytest.param(128, torch.float16, 128, 128, 128, 32, id="128-fp16-r128-p128-q32"),
    ],
)
def test_warp_specialized_bwd_configs(
    chunk_size, dtype, R_TILE, STATE_R_TILE, num_producer_threads, num_threads_Q,
):
    from frac_ssm.ops.cute._chunk_fused_readout import warp_specialized_bwd

    config = (
        warp_specialized_bwd.ChunkFusedReadoutBwdDdeStateConfig(
            R_TILE=STATE_R_TILE, min_blocks_per_mp=1,
        ),
        warp_specialized_bwd.ChunkFusedReadoutBwdDuParamsConfig(
            R_TILE=R_TILE, num_producer_threads=num_producer_threads,
            num_threads_Q=num_threads_Q, min_blocks_per_mp=1,
        ),
    )
    # R=192 fills three 64-channel tiles or leaves half of the second 128-channel tile unused.
    inputs = _config_inputs(chunk_size, dtype, R=192, M=16)
    _check_bwd_config(
        warp_specialized_bwd._frac_chunk_fused_readout_bwd, inputs, chunk_size, config,
    )
