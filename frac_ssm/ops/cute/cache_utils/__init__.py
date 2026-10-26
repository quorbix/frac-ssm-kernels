# Adapted from Dao-AILab/quack at 14b0b7122e252a4df40a0b762c7ffd0f8b67a015.
# Licensed under Apache-2.0. See LICENSE and AUTOTUNER_LICENSE.
# Copyright (c) 2025-2026, Tri Dao.
"""Persistent kernel-cache utilities adapted from QuACK.

Public API
----------

Persistent ``.o`` cache:
* :func:`jit_cache` — decorator that wraps a kernel-compile function with
  in-memory + persistent ``.o`` caching (see :mod:`frac_ssm.ops.cute.cache_utils.jit`).
* :data:`CACHE_ENABLED`, :data:`CACHE_DIR`, :data:`EXTRA_SOURCE_DIRS` —
  static-config flags.
* :class:`FileLock`, :func:`get_cache_path`, :class:`CacheInfo` —
  supporting types.

Async compilation (see :mod:`frac_ssm.ops.cute.cache_utils.async_compile`):
* :class:`CompilePending` — raised by ``jit_cache`` on a cold miss while a
  compile pool is active; the caller defers and retries once the ``.o``
  lands.
* :func:`pool_scope` — activate a compile pool for a scoped block (used by
  the autotuner's bench loop).

CRITICAL ORDERING: the static-config flags below MUST be defined before the
``from frac_ssm.ops.cute.cache_utils.jit import ...`` block. ``jit.py`` does
``import frac_ssm.ops.cute.cache_utils as _state`` at its module top; Python returns the
partially-initialized package object, and lookups inside ``jit_cache``'s
wrapper rely on these names already existing at that checkpoint. Reordering
the imports here, even via an auto-formatter, will break the first kernel
compile with ``AttributeError``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: Optional[str] = os.getenv("QUACK_CACHE_DIR", None)

#: Downstream projects can append directories here to include their sources
#: in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: List[Path] = []


# ---------------------------------------------------------------------------
# Public API surface. Imported AFTER the flags are defined.
# ---------------------------------------------------------------------------

from frac_ssm.ops.cute.cache_utils.jit import (  # noqa: E402
    EXPORT_FUNC_NAME,
    LOCK_TIMEOUT,
    CacheInfo,
    FileLock,
    get_cache_path,
    jit_cache,
)
from frac_ssm.ops.cute.cache_utils.async_compile import (  # noqa: E402
    CompilePending,
    pool_scope,
)

__all__ = [
    # Persistent .o cache.
    "jit_cache",
    "CacheInfo",
    "EXPORT_FUNC_NAME",
    "LOCK_TIMEOUT",
    "FileLock",
    "get_cache_path",
    # Async compilation.
    "CompilePending",
    "pool_scope",
]
