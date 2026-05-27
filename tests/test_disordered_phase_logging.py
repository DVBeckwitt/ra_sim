from __future__ import annotations

import importlib

import numpy as np
import pytest

from ra_sim.gui.state import SimulationRuntimeState
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


class _Var:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def get(self) -> float:
        return float(self.value)


def _runtime_session():
    return importlib.import_module("ra_sim.gui._runtime.runtime_session")


def _hit_table(
    *,
    intensity: float = 5.0,
    h: int = 1,
    k: int = 0,
    l_val: int = 1,
) -> np.ndarray:
    return np.asarray(
        [[float(intensity), 10.0, 11.0, 0.0, float(h), float(k), float(l_val)]],
        dtype=np.float64,
    )


def test_q_group_update_log_includes_source_counts() -> None:
    runtime_session = _runtime_session()
    message = runtime_session._format_qr_qz_group_update_log_message(
        [
            {"source_label": "primary", "peak_count": 2},
            {"source_label": DISORDERED_PHASE_SOURCE_LABEL, "peak_count": 3},
        ]
    )

    assert message == (
        "Updated listed Qr/Qz peaks: 2 groups, 5 peaks; sources: primary=1, disordered_phase=1"
    )


def test_disordered_skip_reason_logged_when_checkbox_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append({"event": event, **fields}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimulationRuntimeState(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_include_disordered_phase_qr_reference_var",
        _Var(False),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "defaults", {"w0": 0.0, "w1": 1.0, "w2": 0.0})
    monkeypatch.setattr(runtime_session, "w0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", _Var(100.0), raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", _Var(0.5), raising=False)

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["signature"] == ("disordered_phase_qr_reference", False)
    assert any(
        event["event"] == "disordered_qr_refs"
        and event["message"] == "Disordered Qr refs skipped: checkbox disabled"
        for event in traces
    )


def test_disordered_enable_status_logger_emits_enabled_and_skip_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimulationRuntimeState(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session._log_disordered_qr_enable_status(
        runtime_session.DisorderedPhaseQrEnableStatus(
            enabled=True,
            reason="enabled",
            stacking_disorder_enabled=True,
            nonzero_disordered_weight=True,
            active_primary_cif_path="active.cif",
        )
    )
    runtime_session._log_disordered_qr_enable_status(
        runtime_session.DisorderedPhaseQrEnableStatus(
            enabled=False,
            reason="zero disordered weight",
            stacking_disorder_enabled=True,
            nonzero_disordered_weight=False,
            active_primary_cif_path="active.cif",
        )
    )

    assert [event["message"] for event in traces] == [
        "Disordered Qr refs enabled: true",
        "Disordered Qr refs skipped: zero disordered weight",
    ]
    assert traces[0]["enabled"] is True
    assert traces[1]["enabled"] is False
    assert traces[1]["skip_reason"] == "zero disordered weight"


def test_disordered_collect_log_includes_hit_table_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append({"event": event, **fields}),
        raising=False,
    )

    def fake_run_one(*_args, **_kwargs):
        return (
            np.zeros((2, 2), dtype=np.float64),
            [_hit_table()],
            [],
            [],
            False,
            True,
            False,
        )

    result = runtime_session._run_disordered_phase_hit_table_collection(
        {
            "job_kind": "full",
            "timing_update_id": 17,
            "collect_disordered_phase_hit_tables": True,
            "disordered_phase_data": np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
            "disordered_phase_intensities": np.asarray([8.0], dtype=np.float64),
            "disordered_phase_a": 4.557,
            "disordered_phase_c": 20.937,
        },
        fake_run_one,
    )

    assert result["disordered_phase_hit_table_state_refreshed"] is True
    assert any(
        event["event"] == "disordered_qr_refs"
        and event["message"] == "Disordered Qr refs collected: hit_tables=1"
        and event["hit_tables"] == 1
        for event in traces
    )


def test_disordered_publish_log_includes_group_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    runtime_state = SimulationRuntimeState()
    runtime_state.stored_disordered_phase_max_positions = [_hit_table()]
    runtime_state.stored_disordered_phase_source_reflection_indices = [0]
    runtime_state.stored_disordered_phase_peak_table_lattice = [
        (
            4.557,
            20.937,
            DISORDERED_PHASE_SOURCE_LABEL,
            DISORDERED_PHASE_DISPLAY_LABEL,
            "disordered",
        )
    ]
    traces: list[dict[str, object]] = []
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "_geometry_disordered_phase_qr_enabled", lambda: True)
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append({"event": event, **fields}),
        raising=False,
    )

    runtime_session._publish_combined_simulation_state(
        image_size_value=4,
        primary_a_value=4.557,
        primary_c_value=6.979,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=(),
    )

    assert any(
        event["event"] == "disordered_qr_refs"
        and event["message"] == "Disordered Qr refs published: groups=1 peaks=1"
        and event["groups"] == 1
        and event["peaks"] == 1
        for event in traces
    )
