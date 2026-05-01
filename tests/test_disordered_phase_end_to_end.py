from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui import manual_geometry
from ra_sim.gui.state import SimulationRuntimeState
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


SOURCE_2H = Path(__file__).resolve().parents[1] / "tests" / "Diffuse" / "PbI2_2H.cif"


class _Var:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def get(self) -> float:
        return float(self.value)


def _patch_runtime(monkeypatch, tmp_path):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimulationRuntimeState()
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "defaults",
        {"p0": 0.0, "p1": 1.0, "p2": 0.5, "w0": 0.0, "w1": 1.0, "w2": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "p0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "w0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", _Var(100.0), raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _Var(4.557), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _Var(6.979), raising=False)
    monkeypatch.setattr(runtime_session, "mx", 4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.5406, raising=False)
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 70.0), raising=False)
    monkeypatch.setattr(runtime_session, "intensity_threshold", 0.25, raising=False)
    monkeypatch.setattr(runtime_session, "_active_primary_cif_path", lambda: str(SOURCE_2H))
    monkeypatch.setattr(runtime_session, "_occupancy_control_vars", lambda: [_Var(1.0)])
    monkeypatch.setattr(
        runtime_session,
        "_current_atom_site_fractional_values",
        lambda: [{"x": 0.0, "y": 0.0, "z": 0.0}],
    )
    monkeypatch.setattr(
        runtime_session,
        "_atom_site_fractional_signature",
        lambda values: tuple(
            (round(float(row["x"]), 9), round(float(row["y"]), 9), round(float(row["z"]), 9))
            for row in values
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_disordered_phase_cif_cache_dir",
        lambda: tmp_path / "generated-cifs",
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (np.array([[1.0, 0.0, 0.0]]), np.array([10.0]), None, None),
    )
    return runtime_session, runtime_state


def _hit_table(intensity: float = 10.0) -> np.ndarray:
    return np.array(
        [[float(intensity), 120.0, 130.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )


def _has_adjacent_primary_disordered_overlap(entries: list[dict[str, object]]) -> bool:
    for left, right in zip(entries, entries[1:]):
        labels = {left.get("source_label"), right.get("source_label")}
        if labels != {"primary", DISORDERED_PHASE_SOURCE_LABEL}:
            continue
        if (
            abs(float(left.get("qr", np.nan)) - float(right.get("qr", np.nan))) <= 1.0e-6
            and abs(float(left.get("qz", np.nan)) - float(right.get("qz", np.nan))) <= 1.0e-6
        ):
            return True
    return False


def test_disordered_phase_non_gui_path_builds_labeled_overlapping_picker_groups(
    monkeypatch,
    tmp_path,
):
    runtime_session, runtime_state = _patch_runtime(monkeypatch, tmp_path)

    disordered_payload = runtime_session._geometry_disordered_phase_inventory_payload()
    assert Path(str(disordered_payload["cif_path"])).is_file()
    assert disordered_payload["source_label"] == DISORDERED_PHASE_SOURCE_LABEL

    def fake_run_one(*_args, **kwargs):
        assert kwargs["collect_hit_tables"] is True
        assert kwargs["accumulate_image"] is False
        return (
            np.zeros((256, 256), dtype=np.float64),
            [_hit_table(5.0)],
            [],
            [],
            False,
            True,
            False,
        )

    disordered_hit_tables = runtime_session._run_disordered_phase_hit_table_collection(
        {
            "collect_disordered_phase_hit_tables": True,
            "disordered_phase_data": disordered_payload["miller"],
            "disordered_phase_intensities": disordered_payload["intensities"],
            "disordered_phase_a": disordered_payload["a"],
            "disordered_phase_c": disordered_payload["c"],
        },
        fake_run_one,
    )

    primary_a = float(disordered_payload["a"])
    primary_c = float(disordered_payload["c"]) / 3.0
    runtime_state.stored_primary_sim_image = np.zeros((256, 256), dtype=np.float64)
    runtime_state.stored_primary_max_positions = [_hit_table(10.0)]
    runtime_state.stored_primary_source_reflection_indices = [0]
    runtime_state.stored_primary_peak_table_lattice = [
        (primary_a, primary_c, "primary", "Ordered phase", "ordered")
    ]
    runtime_state.stored_disordered_phase_max_positions = list(
        disordered_hit_tables["disordered_phase_max_positions"]
    )
    runtime_state.stored_disordered_phase_source_reflection_indices = [1]
    runtime_state.stored_disordered_phase_peak_table_lattice = list(
        disordered_hit_tables["disordered_phase_peak_table_lattice"]
    )

    runtime_session._publish_combined_simulation_state(
        image_size_value=256,
        primary_a_value=primary_a,
        primary_c_value=primary_c,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=("primary",),
    )

    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        runtime_state.stored_max_positions_local,
        peak_table_lattice=runtime_state.stored_peak_table_lattice,
        primary_a=primary_a,
        primary_c=primary_c,
        allow_nominal_hkl_indices=True,
    )
    ordered = [entry for entry in entries if entry["source_label"] == "primary"]
    disordered = [
        entry for entry in entries if entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    ]

    assert ordered
    assert disordered
    assert all(entry["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL for entry in disordered)
    assert _has_adjacent_primary_disordered_overlap(entries)

    picker_rows = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        runtime_state.stored_max_positions_local,
        image_shape=(256, 256),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        peak_table_lattice=runtime_state.stored_peak_table_lattice,
        source_reflection_indices=runtime_state.stored_source_reflection_indices_local,
        primary_a=primary_a,
        primary_c=primary_c,
        allow_nominal_hkl_indices=True,
    )
    grouped_picker = manual_geometry.geometry_manual_detector_picker_grouped_candidates_from_cache(
        {"detector_picker_rows": picker_rows},
        display_background=np.zeros((256, 256), dtype=np.float64),
        native_background=np.zeros((256, 256), dtype=np.float64),
        profile_cache={},
    )

    assert ("q_group", "primary", 1, 0) in grouped_picker
    assert ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 0) in grouped_picker
    flattened_picker = manual_geometry._geometry_manual_flatten_grouped_candidates(grouped_picker)
    assert [entry["source_label"] for entry in flattened_picker[:2]] == [
        "primary",
        DISORDERED_PHASE_SOURCE_LABEL,
    ]
