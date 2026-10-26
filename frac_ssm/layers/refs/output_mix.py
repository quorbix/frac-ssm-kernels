def frac_skip_mix_ref(
    model,
    mixed_flat,
    x_conv,
    out_dtype=None,
    *,
    upcast=True,
):
    """PyTorch reference for the output skip connection and reshape."""
    if out_dtype is None:
        out_dtype = x_conv.dtype

    bsz, seqlen, _ = x_conv.shape
    if upcast:
        mixed_flat = mixed_flat.double()
        x_conv = x_conv.double()

    u = x_conv.view(bsz, seqlen, model.num_heads, model.head_dim)
    mixed = mixed_flat.view(
        bsz, model.num_heads, seqlen, model.head_dim,
    ).permute(0, 2, 1, 3)
    mixed = mixed + u.to(mixed.dtype) * model.D[None, None].to(mixed.dtype)
    mixed = mixed.reshape(bsz, seqlen, model.inner_dim)
    if not upcast:
        mixed = mixed.to(out_dtype)
    return mixed
