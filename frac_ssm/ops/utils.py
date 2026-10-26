import os
import torch
from contextlib import contextmanager, nullcontext
from torch.profiler import record_function

_PROFILE_MODE = os.environ.get("PROFILE_MODE", "")  # "nvtx", "record_function", or ""

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def profile_range(name):
    if _PROFILE_MODE == "nvtx":
        return nvtx_range(name)
    elif _PROFILE_MODE == "record_function":
        return record_function(name)
    else:
        return nullcontext()
