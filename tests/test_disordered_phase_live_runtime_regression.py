from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui import manual_geometry
from ra_sim.gui.state import SimulationRuntimeState
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)
from tests.test_gui_runtime_import_safe import (
    _RuntimeVar,
    _install_matching_hidden_analysis_payload_state,
    _patch_do_update_detector_cache_prereqs,
    _patch_do_update_first_visible_simulation_finish_prereqs,
    _patch_rich_mosaic_params,
)


def _runtime_session():
    return importlib.import_module("ra_sim.gui._runtime.runtime_session")


def _hit_table(
    *,
    intensity: float = 10.0,
    col: float = 100.0,
    row: float = 101.0,
    h: int = 1,
    k: int = 0,
    l_val: int = 1,
) -> np.ndarray:
    return np.asarray(
        [[float(intensity), float(col), float(row), 0.0, float(h), float(k), float(l_val)]],
        dtype=np.float64,
    )


def _write_source_cif(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data_pbi2_primary",
                "_cell_length_a 4.557",
                "_cell_length_b 4.557",
                "_cell_length_c 6.979",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 120",
                "loop_",
                "_atom_site_label",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_type_symbol",
                "Pb1 1.0 0 0 0 Pb",
                "I1 1.0 0.333333 0.666667 0.2675 I",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _patch_disordered_controls(
    monkeypatch: pytest.MonkeyPatch,
    runtime_session,
    tmp_path: Path,
    *,
    enabled: bool,
) -> None:
    source_cif = _write_source_cif(tmp_path / "active-pbi2.cif")
    monkeypatch.setattr(
        runtime_session,
        "defaults",
        {"p0": 0.0, "p1": 1.0, "p2": 0.5, "w0": 1.0, "w1": 0.0, "w2": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "p0_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", _RuntimeVar(0.5), raising=False)
    monkeypatch.setattr(
        runtime_session, "w0_var", _RuntimeVar(0.0 if enabled else 1.0), raising=False
    )
    monkeypatch.setattr(
        runtime_session, "w1_var", _RuntimeVar(100.0 if enabled else 0.0), raising=False
    )
    monkeypatch.setattr(runtime_session, "w2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "mx", 4, raising=False)
    monkeypatch.setattr(runtime_session, "intensity_threshold", 0.0, raising=False)
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 2.0), raising=False)
    monkeypatch.setattr(runtime_session, "_active_primary_cif_path", lambda: str(source_cif))
    monkeypatch.setattr(runtime_session, "_occupancy_control_vars", lambda: [_RuntimeVar(1.0)])
    monkeypatch.setattr(
        runtime_session,
        "_current_atom_site_fractional_values",
        lambda: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_atom_site_fractional_signature",
        lambda values: tuple(values),
    )
    monkeypatch.setattr(
        runtime_session,
        "_disordered_phase_cif_cache_dir",
        lambda: tmp_path / "generated-cifs",
    )
    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda *, source_cif, output_dir, mode: SimpleNamespace(
            cif_path=tmp_path / "generated-disordered.cif",
            a=4.557,
            c=20.937,
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (
            np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
            np.asarray([5.0], dtype=np.float64),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_6h_reference_inventory_payload",
        lambda: {
            "signature": ("pbii_6h_qr_reference", False),
            "miller": np.empty((0, 3), dtype=np.float64),
            "intensities": np.empty((0,), dtype=np.float64),
            "a": float("nan"),
            "c": float("nan"),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_include_6h_qr_reference_var",
        _RuntimeVar(False),
        raising=False,
    )


def _prepare_live_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    disordered_enabled: bool,
):
    runtime_session = _runtime_session()
    cached_hit_tables_reusable = runtime_session._cached_hit_tables_reusable
    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture)
    _patch_do_update_first_visible_simulation_finish_prereqs(
        monkeypatch,
        runtime_session,
        scheduled_post_idle_redraw_calls=[],
        scheduled_settle_calls=[],
        apply_scale_factor_calls=[],
    )
    _patch_rich_mosaic_params(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_cached_hit_tables_reusable",
        cached_hit_tables_reusable,
        raising=False,
    )
    _patch_disordered_controls(
        monkeypatch,
        runtime_session,
        tmp_path,
        enabled=disordered_enabled,
    )
    state = runtime_session.simulation_runtime_state
    state.sim_miller1 = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64)
    state.sim_intens1 = np.asarray([10.0], dtype=np.float64)
    state.sim_miller1_all = state.sim_miller1.copy()
    state.sim_intens1_all = state.sim_intens1.copy()
    state.sim_primary_qr = {}
    state.sim_primary_qr_all = {}
    state.sim_miller2 = np.empty((0, 3), dtype=np.float64)
    state.sim_intens2 = np.empty((0,), dtype=np.float64)
    state.sim_miller2_all = state.sim_miller2.copy()
    state.sim_intens2_all = state.sim_intens2.copy()
    state.stored_primary_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_secondary_sim_image = None
    state.stored_sim_image = np.ones((2, 2), dtype=np.float64)
    state.stored_primary_max_positions = [_hit_table()]
    state.stored_primary_source_reflection_indices = [0]
    state.stored_primary_peak_table_lattice = [(4.557, 6.979, "primary")]
    state.stored_sixh_reference_max_positions = None
    state.stored_disordered_phase_max_positions = None
    state.stored_disordered_phase_source_reflection_indices = None
    state.stored_disordered_phase_peak_table_lattice = None
    state.disordered_phase_inventory_cache = None
    state.generated_disordered_phase_cif_path = None
    state.stored_hit_table_signature = fixture["sim_signature"]
    state.last_sim_signature = fixture["sim_signature"]
    state.last_simulation_signature = fixture["sim_signature"] + (0, 0)
    state.worker_ready_result = None
    state.worker_future = None
    state.worker_poll_token = None
    state.worker_active_job = None
    state.worker_queued_job = None
    state.simulation_epoch = 0
    state.analysis_future = None
    state.analysis_poll_token = None
    state.peak_positions = []
    state.peak_millers = []
    state.peak_intensities = []
    state.peak_records = []
    state.selected_peak_record = None
    return runtime_session, state


def test_nonzero_disordered_weight_changes_runtime_dependency_source_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, _state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
    )
    dependency_updates = []
    original_classify_update = runtime_session.classify_update
    monkeypatch.setattr(
        runtime_session,
        "classify_update",
        lambda previous, current, cache_state: (
            dependency_updates.append(current)
            or original_classify_update(previous, current, cache_state)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda _job: "submitted",
        raising=False,
    )

    runtime_session.do_update()
    primary_only_source_sig = dependency_updates[-1].source_sig

    runtime_session.w0_var.set(0.0)
    runtime_session.w1_var.set(100.0)
    runtime_session.do_update()
    disordered_source_sig = dependency_updates[-1].source_sig

    assert disordered_source_sig != primary_only_source_sig
    assert any(
        isinstance(part, tuple)
        and len(part) >= 2
        and part[0] == "disordered_phase_qr_reference"
        and part[1] is True
        for part in disordered_source_sig
    )


def test_nonzero_disordered_weight_schedules_hit_tables_when_primary_cache_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
    )
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_should_collect_hit_tables_for_update",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )

    runtime_session.do_update()
    state.stored_hit_table_signature = state.last_simulation_signature[:-2]
    state.sim_miller1 = np.empty((0, 3), dtype=np.float64)
    state.sim_intens1 = np.empty((0,), dtype=np.float64)
    state.sim_miller1_all = state.sim_miller1.copy()
    state.sim_intens1_all = state.sim_intens1.copy()
    requested_jobs.clear()

    runtime_session.w0_var.set(0.0)
    runtime_session.w1_var.set(100.0)
    runtime_session.do_update()

    assert requested_jobs
    assert requested_jobs[-1]["collect_hit_tables"] is True
    assert requested_jobs[-1]["collect_disordered_phase_hit_tables"] is True
    assert requested_jobs[-1]["run_primary"] is False


def test_nonzero_disordered_weight_publishes_disordered_q_groups_after_primary_only_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    runtime_state = SimulationRuntimeState()
    runtime_state.stored_primary_sim_image = np.ones((4, 4), dtype=np.float64)
    runtime_state.stored_primary_max_positions = [_hit_table()]
    runtime_state.stored_primary_source_reflection_indices = [0]
    runtime_state.stored_primary_peak_table_lattice = [(4.557, 6.979, "primary")]
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "_geometry_disordered_phase_qr_enabled", lambda: True)

    runtime_session._publish_combined_simulation_state(
        image_size_value=4,
        primary_a_value=4.557,
        primary_c_value=6.979,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=("primary",),
    )
    runtime_state.stored_disordered_phase_max_positions = [
        _hit_table(intensity=5.0, col=110.0, row=111.0)
    ]
    runtime_state.stored_disordered_phase_source_reflection_indices = [1]
    runtime_state.stored_disordered_phase_peak_table_lattice = [
        (
            4.557,
            20.937,
            DISORDERED_PHASE_SOURCE_LABEL,
            DISORDERED_PHASE_DISPLAY_LABEL,
            "disordered",
        )
    ]

    runtime_session._publish_combined_simulation_state(
        image_size_value=4,
        primary_a_value=4.557,
        primary_c_value=6.979,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=(),
    )

    assert any(
        len(tuple(lattice)) >= 3 and tuple(lattice)[2] == DISORDERED_PHASE_SOURCE_LABEL
        for lattice in runtime_state.stored_peak_table_lattice
    )


def test_live_regression_disordered_q_groups_are_clickable_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_session = _runtime_session()
    runtime_state = SimulationRuntimeState()
    primary_rows = _hit_table(intensity=10.0, h=1, k=0, l_val=1)
    disordered_rows = _hit_table(intensity=5.0, h=1, k=0, l_val=1)
    runtime_state.stored_primary_sim_image = np.ones((128, 128), dtype=np.float64)
    runtime_state.stored_primary_max_positions = [primary_rows]
    runtime_state.stored_primary_source_reflection_indices = [0]
    runtime_state.stored_primary_peak_table_lattice = [(4.557, 6.979, "primary")]
    runtime_state.stored_disordered_phase_max_positions = [disordered_rows]
    runtime_state.stored_disordered_phase_source_reflection_indices = [1]
    runtime_state.stored_disordered_phase_peak_table_lattice = [
        (
            4.557,
            20.937,
            DISORDERED_PHASE_SOURCE_LABEL,
            DISORDERED_PHASE_DISPLAY_LABEL,
            "disordered",
        )
    ]
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "_geometry_disordered_phase_qr_enabled", lambda: True)

    runtime_session._publish_combined_simulation_state(
        image_size_value=128,
        primary_a_value=4.557,
        primary_c_value=6.979,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=("primary",),
    )

    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        runtime_state.stored_max_positions_local,
        peak_table_lattice=runtime_state.stored_peak_table_lattice,
        primary_a=4.557,
        primary_c=6.979,
        allow_nominal_hkl_indices=True,
    )
    assert any(entry["source_label"] == "primary" for entry in entries)
    assert any(entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL for entry in entries)

    picker_rows = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        runtime_state.stored_max_positions_local,
        image_shape=(128, 128),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        peak_table_lattice=runtime_state.stored_peak_table_lattice,
        source_reflection_indices=runtime_state.stored_source_reflection_indices_local,
        primary_a=4.557,
        primary_c=6.979,
        allow_nominal_hkl_indices=True,
    )
    grouped = manual_geometry.geometry_manual_detector_picker_grouped_candidates_from_cache(
        {"detector_picker_rows": picker_rows},
        display_background=np.zeros((128, 128), dtype=np.float64),
        native_background=np.zeros((128, 128), dtype=np.float64),
        profile_cache={},
    )

    assert ("q_group", "primary", 1, 1) in grouped
    assert ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 1) in grouped
