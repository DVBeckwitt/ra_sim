from __future__ import annotations

import importlib

import numpy as np

from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


def _runtime_session():
    return importlib.import_module("ra_sim.gui._runtime.runtime_session")


def _job(**overrides):
    job = {
        "collect_disordered_phase_hit_tables": True,
        "disordered_phase_data": np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        "disordered_phase_intensities": np.array([12.0], dtype=np.float64),
        "disordered_phase_a": 4.557,
        "disordered_phase_c": 20.937,
    }
    job.update(overrides)
    return job


def _fake_run_one_factory(calls):
    def fake_run_one(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            np.ones((2, 2), dtype=np.float64),
            [np.array([[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 1.0]], dtype=np.float64)],
            [],
            np.empty((0,), dtype=np.int64),
            False,
            True,
            False,
        )

    return fake_run_one


def test_disordered_phase_run_one_uses_accumulate_image_false():
    runtime_session = _runtime_session()
    calls = []

    runtime_session._run_disordered_phase_hit_table_collection(
        _job(),
        _fake_run_one_factory(calls),
    )

    assert calls
    assert calls[0][1]["accumulate_image"] is False


def test_disordered_phase_run_one_collects_hit_tables():
    runtime_session = _runtime_session()
    calls = []

    result = runtime_session._run_disordered_phase_hit_table_collection(
        _job(),
        _fake_run_one_factory(calls),
    )

    assert calls
    kwargs = calls[0][1]
    assert kwargs["collect_hit_tables"] is True
    assert kwargs["build_intersection_cache"] is False
    assert kwargs["capture_raw_hit_tables"] is False
    assert result["disordered_phase_hit_table_state_refreshed"] is True
    assert len(result["disordered_phase_max_positions"]) == 1


def test_disordered_phase_result_contains_source_label():
    runtime_session = _runtime_session()
    calls = []

    result = runtime_session._run_disordered_phase_hit_table_collection(
        _job(),
        _fake_run_one_factory(calls),
    )

    assert result["disordered_phase_peak_table_lattice"] == [
        (4.557, 20.937, DISORDERED_PHASE_SOURCE_LABEL, "Disordered phase", "disordered")
    ]


def test_disordered_phase_not_run_when_inventory_missing():
    runtime_session = _runtime_session()
    calls = []

    result = runtime_session._run_disordered_phase_hit_table_collection(
        _job(disordered_phase_data=np.empty((0, 3), dtype=np.float64)),
        _fake_run_one_factory(calls),
    )

    assert calls == []
    assert result == {
        "disordered_phase_max_positions": [],
        "disordered_phase_peak_table_lattice": [],
        "disordered_phase_hit_table_state_refreshed": False,
    }
