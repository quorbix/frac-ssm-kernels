import torch


def state_passing_ref(
    states, chunk_decay, initial_state=None, out_dtype=None, upcast=False,
):
    """PyTorch reference for state_passing.

    If upcast, compute the recurrence in float64 and cast the outputs back to
    their kernel dtypes. 
    """
    B, nchunks, R, M = states.shape
    out_dtype = states.dtype if out_dtype is None else out_dtype
    exp_dtype = torch.float64 if upcast else torch.float32
    if upcast:
        compute_dtype = torch.float64
    else:
        compute_dtype = torch.bfloat16 if states.dtype == torch.bfloat16 else torch.float32

    if initial_state is None:
        running = torch.zeros(B, R, M, device=states.device, dtype=compute_dtype)
    else:
        running = initial_state.to(compute_dtype)

    states_acc = torch.empty(B, nchunks, R, M, device=states.device, dtype=out_dtype)
    for n in range(nchunks):
        states_acc[:, n] = running.to(out_dtype)
        decay = torch.exp(chunk_decay[:, n].to(exp_dtype)).to(compute_dtype).unsqueeze(1)
        running = torch.addcmul(states[:, n].to(compute_dtype), decay, running)

    return states_acc, running.to(states.dtype)


def state_passing_bwd_ref(d_states_acc, d_final_state, states_acc, chunk_decay, has_initial_state=False):
    """PyTorch reference for state_passing backward, matching the Triton output shapes."""
    B, nchunks, R, M = states_acc.shape
    d_states = torch.empty(B, nchunks, R, M, device=states_acc.device, dtype=torch.float32)
    d_chunk_decay = torch.empty(B, nchunks, M, device=chunk_decay.device, dtype=torch.float32)
    d_running = d_final_state.to(torch.float32)

    for n in reversed(range(nchunks)):
        running_before = states_acc[:, n].to(torch.float32)
        decay = torch.exp(chunk_decay[:, n].to(torch.float32)).unsqueeze(1)

        d_states[:, n] = d_running
        d_chunk_decay[:, n] = ((d_running * running_before) * decay).sum(dim=1)
        d_running = d_running * decay + d_states_acc[:, n].to(torch.float32)

    d_initial_state = d_running if has_initial_state else None
    return d_states, d_chunk_decay, d_initial_state
