from ra_sim.utils import parallel as parallel_utils


def test_default_reserved_cpu_worker_count_reserves_two_threads(monkeypatch) -> None:
    monkeypatch.setattr(parallel_utils.os, "cpu_count", lambda: 12)

    assert parallel_utils.default_reserved_cpu_worker_count() == 10


def test_default_reserved_cpu_worker_count_keeps_one_worker_minimum(monkeypatch) -> None:
    monkeypatch.setattr(parallel_utils.os, "cpu_count", lambda: 2)

    assert parallel_utils.default_reserved_cpu_worker_count() == 1


def test_numba_threads_per_worker_splits_system_cpu_budget(monkeypatch) -> None:
    monkeypatch.setattr(parallel_utils.os, "cpu_count", lambda: 12)

    assert parallel_utils.numba_threads_per_worker(1) == 12
    assert parallel_utils.numba_threads_per_worker(3) == 4
