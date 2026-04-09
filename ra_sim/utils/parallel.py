"""Shared helpers for sizing worker pools and masking Numba thread use."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Callable, Iterator, Optional


AUTO_PARALLEL_WORKER_RESERVE = 2

try:
    from numba import get_num_threads as _numba_get_num_threads
    from numba import set_num_threads as _numba_set_num_threads
except Exception:  # pragma: no cover - numba is optional in some test/runtime paths
    _numba_get_num_threads = None
    _numba_set_num_threads = None


def system_cpu_worker_count() -> int:
    """Return the host CPU worker capacity with a minimum of one thread."""

    cpu_count = os.cpu_count() or 1
    return max(int(cpu_count), 1)


def current_numba_thread_count() -> Optional[int]:
    """Return the active Numba thread count, or ``None`` when unavailable."""

    if not callable(_numba_get_num_threads):
        return None
    try:
        count = int(_numba_get_num_threads())
    except Exception:
        return None
    return count if count > 0 else None


def current_parallel_thread_budget() -> int:
    """Return the current parallel thread budget for nested worker planning."""

    current_numba_threads = current_numba_thread_count()
    if current_numba_threads is not None:
        return int(current_numba_threads)
    return system_cpu_worker_count()


def reserved_worker_count(
    *,
    thread_budget: int | None = None,
    reserve: int = AUTO_PARALLEL_WORKER_RESERVE,
) -> int:
    """Return one default worker count after reserving a small CPU margin."""

    budget = system_cpu_worker_count() if thread_budget is None else max(int(thread_budget), 1)
    return max(int(budget) - int(reserve), 1)


def default_reserved_cpu_worker_count() -> int:
    """Return the default app-wide worker count: available CPUs minus two."""

    return reserved_worker_count(thread_budget=system_cpu_worker_count())


def numba_threads_per_worker(
    worker_count: int,
    *,
    thread_budget: int | None = None,
) -> int:
    """Split one thread budget conservatively across multiple outer workers."""

    workers = max(int(worker_count), 1)
    budget = system_cpu_worker_count() if thread_budget is None else max(int(thread_budget), 1)
    if workers <= 1:
        return int(budget)
    return max(int(budget // workers), 1)


@contextmanager
def temporary_numba_thread_limit(num_threads: Optional[int]) -> Iterator[None]:
    """Temporarily mask Numba's worker thread count around one operation."""

    if num_threads is None or not callable(_numba_set_num_threads):
        yield
        return

    original_threads = current_numba_thread_count()
    try:
        _numba_set_num_threads(max(int(num_threads), 1))
        yield
    finally:
        if original_threads is not None and callable(_numba_set_num_threads):
            try:
                _numba_set_num_threads(max(int(original_threads), 1))
            except Exception:
                pass


def call_with_numba_thread_limit(
    fn: Callable[..., object],
    *args,
    numba_threads: Optional[int] = None,
    **kwargs,
):
    """Invoke ``fn`` while temporarily applying one Numba thread mask."""

    with temporary_numba_thread_limit(numba_threads):
        return fn(*args, **kwargs)
