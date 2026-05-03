from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.test_disordered_phase_hit_tables import (
    _fake_run_one_factory,
    _job,
    _runtime_session,
)
from tests.test_disordered_phase_live_runtime_regression import (
    _hit_table,
    _prepare_live_runtime,
)


def _scheduled_disordered_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mutate_state=None,
) -> dict[str, object]:
    runtime_session, state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    if mutate_state is not None:
        mutate_state(state)
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()

    assert requested_jobs
    return requested_jobs[-1]


def test_enabled_disordered_qr_with_missing_rows_schedules_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _scheduled_disordered_job(monkeypatch, tmp_path)

    assert job["collect_hit_tables"] is True
    assert job["collect_disordered_phase_hit_tables"] is True


def test_disordered_collection_runs_when_primary_job_is_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _scheduled_disordered_job(monkeypatch, tmp_path)

    assert job["collect_disordered_phase_hit_tables"] is True
    assert np.asarray(job["disordered_phase_data"]).shape == (1, 3)


def test_disordered_collection_runs_when_run_primary_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def clear_primary_rows(state) -> None:
        state.sim_miller1 = np.empty((0, 3), dtype=np.float64)
        state.sim_intens1 = np.empty((0,), dtype=np.float64)
        state.sim_miller1_all = state.sim_miller1.copy()
        state.sim_intens1_all = state.sim_intens1.copy()
        state.sim_primary_qr = {}
        state.sim_primary_qr_all = {}

    job = _scheduled_disordered_job(monkeypatch, tmp_path, mutate_state=clear_primary_rows)

    assert job["run_primary"] is False
    assert job["collect_disordered_phase_hit_tables"] is True


def test_disordered_collection_request_schedules_hit_table_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def request_disordered_collection_without_primary(state) -> None:
        state.sim_miller1 = np.empty((0, 3), dtype=np.float64)
        state.sim_intens1 = np.empty((0,), dtype=np.float64)
        state.sim_miller1_all = state.sim_miller1.copy()
        state.sim_intens1_all = state.sim_intens1.copy()
        state.sim_primary_qr = {}
        state.sim_primary_qr_all = {}
        state.stored_disordered_phase_max_positions = []
        state.disordered_phase_hit_table_collection_requested = True
        state.disordered_phase_hit_table_collection_request_signature = ("pending",)

    job = _scheduled_disordered_job(
        monkeypatch,
        tmp_path,
        mutate_state=request_disordered_collection_without_primary,
    )

    assert job["run_primary"] is False
    assert job["collect_hit_tables"] is True
    assert job["collect_disordered_phase_hit_tables"] is True


def test_disordered_collection_runs_when_signature_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def stale_disordered_rows(state) -> None:
        state.stored_disordered_phase_max_positions = [_hit_table()]
        state.stored_disordered_phase_source_reflection_indices = [0]
        state.stored_disordered_phase_peak_table_lattice = [
            (4.557, 20.937, "disordered_phase", "Disordered phase", "disordered")
        ]
        state.stored_hit_table_signature = ("stale",)

    job = _scheduled_disordered_job(monkeypatch, tmp_path, mutate_state=stale_disordered_rows)

    assert job["collect_hit_tables"] is True
    assert job["collect_disordered_phase_hit_tables"] is True


def test_disordered_collection_still_uses_accumulate_image_false() -> None:
    runtime_session = _runtime_session()
    calls = []

    runtime_session._run_disordered_phase_hit_table_collection(
        _job(),
        _fake_run_one_factory(calls),
    )

    assert calls
    assert calls[0][1]["accumulate_image"] is False
