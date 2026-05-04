"""Shared helpers for sizing worker pools and masking Numba thread use."""

from __future__ import annotations

import concurrent.futures
import concurrent.futures.thread as futures_thread
from contextlib import contextmanager
import os
import threading
from typing import Callable, Iterator, Optional
import weakref

from ra_sim.utils.numba_compat import get_num_threads as _numba_get_num_threads
from ra_sim.utils.numba_compat import set_num_threads as _numba_set_num_threads

AUTO_PARALLEL_WORKER_RESERVE = 2
WEIGHTED_EVENT_WORKERS_ENV = "RA_SIM_WEIGHTED_EVENT_WORKERS"


class _DetachedThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Thread pool whose workers do not register for interpreter-exit joins."""

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            worker_thread = threading.Thread(
                name=thread_name,
                target=futures_thread._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            worker_thread.start()
            self._threads.add(worker_thread)


def make_detached_thread_pool_executor(
    *,
    max_workers: int,
    thread_name_prefix: str | None = None,
) -> concurrent.futures.ThreadPoolExecutor:
    """Return one thread pool whose workers will not keep the process alive."""

    return _DetachedThreadPoolExecutor(
        max_workers=max(int(max_workers), 1),
        thread_name_prefix=thread_name_prefix,
    )


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


def _positive_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "auto":
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _auto_weighted_event_worker_count(n_samp: int) -> int:
    n_samp_i = max(int(n_samp), 0)
    if n_samp_i < 96:
        return 1
    if n_samp_i < 512:
        return 2
    if n_samp_i < 2048:
        return 4
    return 8


def resolve_weighted_event_worker_count(
    requested: int | str | None,
    *,
    n_samp: int,
    outer_workers: int = 1,
    auto: bool = True,
) -> tuple[int, str]:
    """Resolve weighted-event inner workers from explicit/env/auto policy."""

    source = "auto"
    requested_count = _positive_int_or_none(requested)
    if requested_count is not None:
        desired = requested_count
        source = "explicit"
    else:
        env_value = os.environ.get(WEIGHTED_EVENT_WORKERS_ENV)
        env_count = _positive_int_or_none(env_value)
        if env_count is not None:
            desired = env_count
            source = "env"
        else:
            desired = _auto_weighted_event_worker_count(n_samp) if bool(auto) else 1
            if env_value is not None and str(env_value).strip().lower() not in {"", "auto"}:
                source = "auto_invalid_env"

    try:
        outer_workers_i = max(int(outer_workers), 1)
    except (TypeError, ValueError):
        outer_workers_i = 1

    max_inner = max(system_cpu_worker_count() // outer_workers_i, 1)
    effective = max(min(int(desired), int(max_inner)), 1)
    n_samp_i = max(int(n_samp), 0)
    if n_samp_i > 0:
        effective = min(effective, n_samp_i)
    return int(effective), source


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
