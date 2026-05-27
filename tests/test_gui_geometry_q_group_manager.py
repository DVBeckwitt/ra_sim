from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager, state
from ra_sim.simulation import diffraction, intersection_cache_schema as cache_schema
from tests.helpers.gui_fakes import RuntimeVar as _FakeVar


def _entry(group_key, *, peak_count, total_intensity, source="primary"):
    return {
        "key": group_key,
        "source_label": source,
        "qr": 1.25,
        "qz": 0.5,
        "gz_index": group_key[3],
        "total_intensity": total_intensity,
        "peak_count": peak_count,
        "hkl_preview": [(1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)],
    }


def test_q_group_signature_value_handles_recursive_or_bad_sequence_values() -> None:
    from collections.abc import Sequence as AbstractSequence

    recursive = []
    recursive.append(recursive)

    class BadSequence(AbstractSequence):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise RuntimeError(f"bad sequence index {index}")

    recursive_signature = geometry_q_group_manager._geometry_q_group_signature_value(recursive)
    bad_signature = geometry_q_group_manager._geometry_q_group_signature_value(BadSequence())

    assert recursive_signature == (("cycle", "list"),)
    assert bad_signature[0] == "sequence_iter_error"
    assert bad_signature[1] == "BadSequence"


def test_geometry_q_group_ml_helpers_map_hexagonal_equivalents() -> None:
    assert geometry_q_group_manager.geometry_q_group_m_from_hk(-1, 0) == 1
    assert geometry_q_group_manager.geometry_q_group_m_from_hk(1, 1) == 3

    equivalent_hkls = [(-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10)]
    assert [
        geometry_q_group_manager.geometry_q_group_ml_from_hkl(hkl)
        for hkl in equivalent_hkls
    ] == [(1, 10), (1, 10), (1, 10), (1, 10)]

    key, _qr_val, _qz_val = geometry_q_group_manager.reflection_q_group_metadata(
        (-1, 0, 10),
        source_label="primary",
        a_value=3.0,
        c_value=5.0,
    )
    assert key == ("q_group", "primary", 1, 10)
    assert geometry_q_group_manager.geometry_q_group_ml_from_key(key) == (1, 10)
    assert geometry_q_group_manager.geometry_q_group_ml_from_key(("primary", 1, 10)) == (1, 10)
    assert geometry_q_group_manager.geometry_q_group_ml_from_key((1, 0, 10)) is None


def _make_runtime_q_group_bundle(
    runtime_state,
    *,
    primary_a_factory=3.0,
    primary_c_factory=5.0,
    caked_view_enabled_factory=False,
    project_peaks_to_current_view=None,
):
    return geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=primary_a_factory,
        primary_c_factory=primary_c_factory,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        caked_view_enabled_factory=caked_view_enabled_factory,
        project_peaks_to_current_view=project_peaks_to_current_view,
    )


def test_geometry_q_group_manager_geometry_metadata_helpers() -> None:
    rows = geometry_q_group_manager.geometry_reference_hit_rows(
        [
            [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [5.0, np.nan, 2.0, 0.0, 1.0, 0.0, 0.0],
            [7.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        ]
    )
    key, qr_val, qz_val = geometry_q_group_manager.reflection_q_group_metadata(
        (1.0, 0.0, 2.0),
        source_label="SECONDARY",
        a_value=2.5,
        c_value=5.0,
    )

    assert len(rows) == 1
    np.testing.assert_allclose(rows[0], [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    assert key == ("q_group", "secondary", 1, 2)
    assert np.isclose(qr_val, (2.0 * np.pi / 2.5) * np.sqrt(4.0 / 3.0))
    assert np.isclose(qz_val, (2.0 * np.pi / 5.0) * 2.0)
    assert (
        geometry_q_group_manager.geometry_q_group_key_from_entry(
            {
                "hkl_raw": (1.0, 0.0, 2.0),
                "source_label": "secondary",
                "av": 2.5,
                "cv": 5.0,
            }
        )
        == key
    )
    missing_key, missing_qr, missing_qz = geometry_q_group_manager.reflection_q_group_metadata(
        (1.0, 0.0, 1.25),
        source_label="primary",
        a_value=3.0,
        c_value=6.0,
    )
    assert missing_key is None
    assert np.isnan(missing_qr)
    assert np.isnan(missing_qz)

    nominal_key, nominal_qr, nominal_qz = geometry_q_group_manager.reflection_q_group_metadata(
        (1.0, 0.0, 1.25),
        source_label="primary",
        a_value=3.0,
        c_value=6.0,
        allow_nominal_hkl_indices=True,
    )
    assert nominal_key == ("q_group", "primary", 1, 1)
    assert np.isclose(nominal_qr, (2.0 * np.pi / 3.0) * np.sqrt(4.0 / 3.0))
    assert np.isclose(nominal_qz, 2.0 * np.pi / 6.0)


def test_geometry_q_group_manager_nominal_hkl_grouping_supports_cache_rows() -> None:
    cache_like_hit_tables = [
        np.asarray(
            [
                [12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.29],
            ],
            dtype=float,
        )
    ]

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        cache_like_hit_tables,
        image_shape=(32, 32),
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        allow_nominal_hkl_indices=True,
    )
    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        cache_like_hit_tables,
        peak_table_lattice=[(3.0, 5.0, "primary")],
        allow_nominal_hkl_indices=True,
    )

    assert len(peaks) == 1
    assert peaks[0]["hkl"] == (1, 0, 1)
    assert peaks[0]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert peaks[0]["q_group_nominal_hkl"] is True
    assert "source_reflection_index" not in peaks[0]
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)
    assert entries[0]["hkl_preview"] == [(1, 0, 1)]


def test_build_geometry_fit_simulated_peaks_recovers_mirrored_live_source_branches() -> None:
    hit_tables = [
        np.asarray(
            [
                [10.0, 100.0, 200.0, -0.44, -1.0, 0.0, 13.0],
            ],
            dtype=float,
        ),
        np.asarray(
            [
                [11.0, 140.0, 150.0, -0.41, -1.0, 0.0, 13.0],
            ],
            dtype=float,
        ),
    ]

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=(256, 256),
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice=[(3.0, 5.0, "primary"), (3.0, 5.0, "primary")],
        source_reflection_indices=[220, 221],
    )

    assert len(peaks) == 2
    assert [peak["source_reflection_index"] for peak in peaks] == [220, 221]
    assert [peak["source_reflection_namespace"] for peak in peaks] == [
        "full_reflection",
        "full_reflection",
    ]
    assert [peak["source_reflection_is_full"] for peak in peaks] == [True, True]
    assert [peak["source_branch_index"] for peak in peaks] == [0, 1]
    assert [peak["source_peak_index"] for peak in peaks] == [0, 1]


def test_build_geometry_fit_simulated_peaks_restores_phi_branch_after_canonicalization(
    monkeypatch,
) -> None:
    original_canonicalize = (
        geometry_q_group_manager.gui_manual_geometry.geometry_manual_canonicalize_live_source_entry
    )

    def _strip_branch_fields(*args, **kwargs):
        entry = original_canonicalize(*args, **kwargs)
        if isinstance(entry, dict):
            entry.pop("source_branch_index", None)
            entry.pop("source_peak_index", None)
        return entry

    monkeypatch.setattr(
        geometry_q_group_manager.gui_manual_geometry,
        "geometry_manual_canonicalize_live_source_entry",
        _strip_branch_fields,
    )

    hit_tables = [
        np.asarray(
            [
                [10.0, 100.0, 200.0, -12.0, -1.0, 0.0, 10.0],
                [11.0, 100.0, 200.0, 12.0, -1.0, 0.0, 10.0],
            ],
            dtype=float,
        )
    ]

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=(256, 256),
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice=[(3.0, 5.0, "primary")],
    )

    assert [peak["source_branch_index"] for peak in peaks] == [0, 1]
    assert [peak["source_peak_index"] for peak in peaks] == [0, 1]


def test_geometry_q_group_manager_builds_entries_from_hit_tables() -> None:
    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        [
            np.asarray(
                [
                    [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                    [5.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0],
                    [8.0, 5.0, 6.0, 0.0, 1.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    [6.0, 7.0, 8.0, 0.0, 1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        ],
        peak_table_lattice=[
            (3.0, 5.0, "primary"),
            (4.0, 6.0, "secondary"),
        ],
    )

    assert [entry["key"] for entry in entries] == [
        ("q_group", "primary", 1, 0),
        ("q_group", "primary", 1, 1),
        ("q_group", "secondary", 3, 0),
    ]
    assert entries[0]["total_intensity"] == 15.0
    assert entries[0]["peak_count"] == 2
    assert entries[0]["hkl_preview"] == [(1, 0, 0)]
    assert np.isclose(entries[1]["qz"], 2.0 * np.pi / 5.0)
    assert np.isclose(
        entries[2]["qr"],
        (2.0 * np.pi / 4.0) * np.sqrt((4.0 / 3.0) * 3.0),
    )

    fallback_entries = geometry_q_group_manager.build_geometry_q_group_entries(
        [
            np.asarray(
                [
                    [4.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        ],
        peak_table_lattice=None,
        primary_a=7.0,
        primary_c=9.0,
    )
    assert fallback_entries[0]["source_label"] == "primary"
    assert np.isclose(
        fallback_entries[0]["qr"],
        (2.0 * np.pi / 7.0) * np.sqrt(4.0 / 3.0),
    )


def test_geometry_q_group_manager_builds_entries_from_cached_peak_records() -> None:
    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        None,
        peak_records=[
            {
                "hkl": (1, 0, 0),
                "hkl_raw": (1.0, 0.0, 0.0),
                "source_label": "primary",
                "av": 3.0,
                "cv": 5.0,
                "qr": 1.234,
                "qz": 0.0,
                "intensity": 10.0,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
            {
                "hkl": (0, 1, 0),
                "hkl_raw": (0.0, 1.0, 0.0),
                "source_label": "primary",
                "av": 3.0,
                "cv": 5.0,
                "qr": 1.234,
                "qz": 0.0,
                "intensity": 5.0,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
            {
                "hkl": (1, 0, 1),
                "hkl_raw": (1.0, 0.0, 1.29),
                "source_label": "primary",
                "av": 3.0,
                "cv": 5.0,
                "intensity": 7.0,
                "q_group_nominal_hkl": True,
            },
        ],
        allow_nominal_hkl_indices=True,
    )

    assert [entry["key"] for entry in entries] == [
        ("q_group", "primary", 1, 0),
        ("q_group", "primary", 1, 1),
    ]
    assert entries[0]["total_intensity"] == 15.0
    assert entries[0]["peak_count"] == 2
    assert entries[0]["hkl_preview"] == [(1, 0, 0), (0, 1, 0)]
    assert np.isclose(entries[0]["qr"], 1.234)
    assert np.isclose(entries[1]["qz"], 2.0 * np.pi / 5.0)


def test_geometry_q_group_entries_keep_explicit_group_without_hkl() -> None:
    group_key = ("q_group", "primary", 5, 2)

    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        None,
        peak_records=[
            {
                "q_group_key": group_key,
                "source_label": "primary",
                "qr": 1.25,
                "qz": 2.5,
                "weight": 3.0,
            }
        ],
    )

    assert len(entries) == 1
    assert entries[0]["key"] == group_key
    assert entries[0]["qr"] == 1.25
    assert entries[0]["qz"] == 2.5
    assert entries[0]["m_index"] == 5
    assert entries[0]["l_index"] == 2
    assert entries[0]["gz_index"] == 2
    assert entries[0]["peak_count"] == 1
    assert entries[0]["hkl_preview"] == []


def test_geometry_q_group_manager_builds_simulated_peaks_from_hit_tables() -> None:
    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [
            np.asarray(
                [
                    [10.0, 1.2, 2.8, -15.0, 1.0, 0.0, 0.0],
                    [8.0, 3.0, 4.0, 15.0, 1.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        ],
        image_shape=(20, 30),
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        primary_a=7.0,
        primary_c=9.0,
        default_source_label=None,
        round_pixel_centers=False,
    )

    assert len(peaks) == 2
    assert peaks[0]["hkl"] == (1, 0, 0)
    assert peaks[0]["sim_col"] == 31.2
    assert peaks[0]["sim_row"] == 22.8
    assert peaks[0]["source_label"] == "primary"
    assert peaks[0]["source_branch_index"] == 0
    assert peaks[0]["source_peak_index"] == 0
    assert "source_reflection_index" not in peaks[0]
    assert peaks[1]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert peaks[1]["source_row_index"] == 1
    assert peaks[1]["source_branch_index"] == 1
    assert peaks[1]["source_peak_index"] == 1

    fallback_peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [
            np.asarray(
                [
                    [4.0, 1.2, 2.8, 0.0, 1.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        ],
        image_shape=(20, 30),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        peak_table_lattice=None,
        primary_a=4.0,
        primary_c=6.0,
        default_source_label=None,
        round_pixel_centers=True,
    )
    assert fallback_peaks[0]["source_label"] == "table_0"
    assert fallback_peaks[0]["sim_col"] == 1.0
    assert fallback_peaks[0]["sim_row"] == 3.0
    assert fallback_peaks[0]["q_group_key"] == ("q_group", "primary", 3, 0)


def test_geometry_q_group_manager_builds_simulated_peaks_from_provenance_hit_rows() -> None:
    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [
            np.asarray(
                [
                    [10.0, 1.2, 2.8, -15.0, 1.0, 0.0, 0.0, 5.0, 6.0, 7.0],
                ],
                dtype=float,
            )
        ],
        image_shape=(20, 30),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        round_pixel_centers=False,
    )

    assert len(peaks) == 1
    assert peaks[0]["source_table_index"] == 5
    assert peaks[0]["source_row_index"] == 6
    assert peaks[0]["best_sample_index"] == 7


def test_geometry_q_group_manager_aggregates_peak_centers_from_max_positions() -> None:
    assert geometry_q_group_manager.geometry_fit_peak_center_from_max_position(
        [9.0, 1.0, 2.0, 4.0, 8.0, 9.0]
    ) == (1.0, 2.0)
    assert geometry_q_group_manager.geometry_fit_peak_center_from_max_position(
        [3.0, 1.0, 2.0, 7.0, 5.0, 6.0]
    ) == (5.0, 6.0)

    peaks = geometry_q_group_manager.aggregate_geometry_fit_peak_centers_from_max_positions(
        [
            [9.0, 1.0, 2.0, 4.0, 8.0, 9.0],
            [3.0, 1.0, 2.0, 7.0, 5.0, 6.0],
            [6.0, np.nan, np.nan, 2.0, 10.0, 12.0],
        ],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        np.asarray([4.0, -6.0, 7.0], dtype=float),
    )

    assert peaks == [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "sim_col": 3.0,
            "sim_row": 4.0,
            "weight": 10.0,
        },
        {
            "hkl": (2, 0, 1),
            "label": "2,0,1",
            "sim_col": 10.0,
            "sim_row": 12.0,
            "weight": 7.0,
        },
    ]


def test_geometry_q_group_manager_simulate_geometry_fit_helpers() -> None:
    captured = {}
    n2_override = np.asarray([0.8 + 0.01j, 0.7 + 0.02j], dtype=np.complex128)

    def _build_mosaic(params):
        captured["params"] = dict(params)
        return {
            "beam_x_array": np.asarray([1.0, 2.0], dtype=float),
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "n2_sample_array": n2_override,
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    param_set = {
        "a": 3.0,
        "c": 5.0,
        "lambda": 1.54,
        "corto_detector": 100.0,
        "gamma": 1.0,
        "Gamma": 2.0,
        "chi": 3.0,
        "psi": 4.0,
        "psi_z": 5.0,
        "zs": 6.0,
        "zb": 7.0,
        "n2": "n2",
        "debye_x": 0.1,
        "debye_y": 0.2,
        "center": (11.0, 12.0),
        "theta_initial": 8.0,
        "cor_angle": 9.0,
        "optics_mode": diffraction.OPTICS_MODE_EXACT,
    }
    miller_array = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    intensity_array = np.asarray([5.0], dtype=float)

    hit_tables = geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert len(hit_tables) == 1
    np.testing.assert_allclose(captured["args"][5], [1.54, 1.54])
    np.testing.assert_allclose(captured["args"][16], [1.0, 2.0])
    np.testing.assert_allclose(captured["args"][23], [1.54, 1.54])
    assert captured["args"][15] == "n2"
    assert captured["kwargs"]["optics_mode"] == diffraction.OPTICS_MODE_EXACT
    assert captured["kwargs"]["solve_q_steps"] == 123
    assert captured["kwargs"]["solve_q_mode"] == 1
    np.testing.assert_array_equal(
        captured["kwargs"]["n2_sample_array_override"],
        n2_override,
    )

    centers = geometry_q_group_manager.simulate_geometry_fit_peak_centers(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [[9.0, 1.0, 2.0, 4.0, 6.0, 7.0]],
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )
    assert centers == [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "sim_col": 1.0,
            "sim_row": 2.0,
            "weight": 5.0,
        }
    ]

    preview_peaks = geometry_q_group_manager.simulate_geometry_fit_preview_style_peaks(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        primary_a=7.0,
        primary_c=9.0,
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert preview_peaks[0]["sim_col"] == 33.2
    assert preview_peaks[0]["sim_row"] == 34.8
    assert preview_peaks[0]["source_label"] == "primary"
    assert preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)


def test_simulate_geometry_fit_hit_tables_drops_wrong_length_n2_override() -> None:
    captured = {}

    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0, 2.0], dtype=float),
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "n2_sample_array": np.asarray([1.0 + 0.0j], dtype=np.complex128),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([5.0], dtype=float),
        32,
        {
            "a": 3.0,
            "c": 5.0,
            "lambda": 1.54,
            "corto_detector": 100.0,
            "gamma": 1.0,
            "Gamma": 2.0,
            "chi": 3.0,
            "psi": 4.0,
            "psi_z": 5.0,
            "zs": 6.0,
            "zb": 7.0,
            "n2": "n2",
            "debye_x": 0.1,
            "debye_y": 0.2,
            "center": (11.0, 12.0),
            "theta_initial": 8.0,
            "cor_angle": 9.0,
            "optics_mode": diffraction.OPTICS_MODE_EXACT,
        },
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert captured["kwargs"].get("n2_sample_array_override") is None


def test_simulate_geometry_fit_hit_tables_drops_malformed_n2_override() -> None:
    captured = {}

    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0, 2.0], dtype=float),
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "n2_sample_array": ["bad", "override"],
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([5.0], dtype=float),
        32,
        {
            "a": 3.0,
            "c": 5.0,
            "lambda": 1.54,
            "corto_detector": 100.0,
            "gamma": 1.0,
            "Gamma": 2.0,
            "chi": 3.0,
            "psi": 4.0,
            "psi_z": 5.0,
            "zs": 6.0,
            "zb": 7.0,
            "n2": "n2",
            "debye_x": 0.1,
            "debye_y": 0.2,
            "center": (11.0, 12.0),
            "theta_initial": 8.0,
            "cor_angle": 9.0,
            "optics_mode": diffraction.OPTICS_MODE_EXACT,
        },
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert captured["kwargs"].get("n2_sample_array_override") is None


def test_simulate_geometry_fit_hit_tables_records_diagnostics_for_missing_beam_x_array() -> None:
    called = False

    def _build_mosaic(_params):
        return {
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("process_peaks_parallel should not be reached")

    with pytest.raises(KeyError):
        geometry_q_group_manager.simulate_geometry_fit_hit_tables(
            np.asarray([[1.0, 0.0, 0.0]], dtype=float),
            np.asarray([5.0], dtype=float),
            32,
            {
                "a": 3.0,
                "c": 5.0,
                "lambda": 1.54,
                "corto_detector": 100.0,
                "gamma": 1.0,
                "Gamma": 2.0,
                "chi": 3.0,
                "psi": 4.0,
                "psi_z": 5.0,
                "zs": 6.0,
                "zb": 7.0,
                "n2": "n2",
                "debye_x": 0.1,
                "debye_y": 0.2,
                "center": (11.0, 12.0),
                "theta_initial": 8.0,
                "cor_angle": 9.0,
                "optics_mode": diffraction.OPTICS_MODE_EXACT,
            },
            build_geometry_fit_central_mosaic_params=_build_mosaic,
            process_peaks_parallel=_process_peaks_parallel,
            default_solve_q_steps=123,
            default_solve_q_rel_tol=2.5e-4,
            default_solve_q_mode=1,
        )

    diagnostics = geometry_q_group_manager._function_last_diagnostics(
        geometry_q_group_manager.simulate_geometry_fit_hit_tables
    )

    assert called is False
    assert diagnostics["stage"] == "simulate_hit_tables"
    assert diagnostics["status"] == "process_peaks_parallel_exception"
    assert diagnostics["exception_type"] == "KeyError"


def test_simulate_geometry_fit_hit_tables_marks_empty_target_filter_unused() -> None:
    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0], dtype=float),
            "beam_y_array": np.asarray([2.0], dtype=float),
            "theta_array": np.asarray([3.0], dtype=float),
            "phi_array": np.asarray([4.0], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*_args, **_kwargs):
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 2.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    hit_tables = geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        np.asarray([[2.0, 0.0, 0.0]], dtype=float),
        np.asarray([5.0], dtype=float),
        32,
        {
            "a": 3.0,
            "c": 5.0,
            "lambda": 1.54,
            "corto_detector": 100.0,
            "gamma": 1.0,
            "Gamma": 2.0,
            "chi": 3.0,
            "psi": 4.0,
            "psi_z": 5.0,
            "zs": 6.0,
            "zb": 7.0,
            "n2": "n2",
            "debye_x": 0.1,
            "debye_y": 0.2,
            "center": (11.0, 12.0),
            "theta_initial": 8.0,
            "cor_angle": 9.0,
            "optics_mode": diffraction.OPTICS_MODE_EXACT,
        },
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        required_branch_group_keys=[((1, 0, 0), 1, ("q", 1))],
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    diagnostics = geometry_q_group_manager._function_last_diagnostics(
        geometry_q_group_manager.simulate_geometry_fit_hit_tables
    )

    assert hit_tables == []
    assert diagnostics["targeted_simulation_supported"] is True
    assert diagnostics["targeted_simulation_used"] is False
    assert diagnostics["targeted_simulation_fallback_reason"] == ("targeted_hkl_filter_empty")


def test_simulate_geometry_fit_preview_style_peaks_respects_lattice_and_provenance() -> None:
    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0], dtype=float),
            "beam_y_array": np.asarray([2.0], dtype=float),
            "theta_array": np.asarray([3.0], dtype=float),
            "phi_array": np.asarray([4.0], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*_args, **_kwargs):
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    preview_peaks = geometry_q_group_manager.simulate_geometry_fit_preview_style_peaks(
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([5.0], dtype=float),
        32,
        {
            "a": 3.0,
            "c": 5.0,
            "lambda": 1.54,
            "corto_detector": 100.0,
            "gamma": 1.0,
            "Gamma": 2.0,
            "chi": 3.0,
            "psi": 4.0,
            "psi_z": 5.0,
            "zs": 6.0,
            "zb": 7.0,
            "n2": "n2",
            "debye_x": 0.1,
            "debye_y": 0.2,
            "center": (11.0, 12.0),
            "theta_initial": 8.0,
            "cor_angle": 9.0,
            "optics_mode": diffraction.OPTICS_MODE_EXACT,
        },
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        primary_a=7.0,
        primary_c=9.0,
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert len(preview_peaks) == 1
    assert preview_peaks[0]["source_label"] == "primary"
    assert preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert preview_peaks[0]["source_reflection_index"] == 0
    assert preview_peaks[0]["source_reflection_namespace"] == "full_reflection"
    assert preview_peaks[0]["source_reflection_is_full"] is True


def test_geometry_q_group_manager_runtime_simulation_callback_bundle_uses_live_values(
    monkeypatch,
) -> None:
    calls = []
    live = {
        "peak_table_lattice": [(3.0, 5.0, "primary")],
        "primary_a": 7.0,
        "primary_c": 9.0,
    }

    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_hit_tables",
        lambda *args, **kwargs: calls.append(("hit_tables", args, kwargs)) or ["hit"],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_peak_centers",
        lambda *args, **kwargs: (
            calls.append(("peak_centers", args, kwargs)) or [{"hkl": (1, 0, 0)}]
        ),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_preview_style_peaks",
        lambda *args, **kwargs: (
            calls.append(("preview_style", args, kwargs)) or [{"hkl": (1, 0, 1)}]
        ),
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        build_geometry_fit_central_mosaic_params="build-mosaic",
        process_peaks_parallel="process-parallel",
        hit_tables_to_max_positions="to-maxpos",
        native_sim_to_display_coords="native-to-display",
        peak_table_lattice_factory=lambda: live["peak_table_lattice"],
        primary_a_factory=lambda: live["primary_a"],
        primary_c_factory=lambda: live["primary_c"],
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert bundle.simulate_hit_tables("miller", "intensity", 32, {"a": 1.0}) == ["hit"]
    assert bundle.simulate_peak_centers("miller", "intensity", 32, {"a": 1.0}) == [
        {"hkl": (1, 0, 0)}
    ]
    assert bundle.simulate_preview_style_peaks(
        "miller",
        "intensity",
        32,
        {"a": 1.0},
    ) == [{"hkl": (1, 0, 1)}]

    assert calls[0] == (
        "hit_tables",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )
    assert calls[1] == (
        "peak_centers",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "hit_tables_to_max_positions": "to-maxpos",
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )
    assert calls[2] == (
        "preview_style",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "native_sim_to_display_coords": "native-to-display",
            "peak_table_lattice": [(3.0, 5.0, "primary")],
            "primary_a": 7.0,
            "primary_c": 9.0,
            "default_source_label": None,
            "round_pixel_centers": False,
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )

    live["peak_table_lattice"] = None
    live["primary_a"] = 11.0
    live["primary_c"] = 13.0
    bundle.simulate_preview_style_peaks("miller", "intensity", 64, {"c": 2.0})
    assert calls[3] == (
        "preview_style",
        ("miller", "intensity", 64, {"c": 2.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "native_sim_to_display_coords": "native-to-display",
            "peak_table_lattice": None,
            "primary_a": 11.0,
            "primary_c": 13.0,
            "default_source_label": None,
            "round_pixel_centers": False,
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )


def test_geometry_q_group_manager_runtime_simulation_callback_bundle_captures_diagnostics() -> None:
    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0, 2.0], dtype=float),
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "wavelength_array": np.asarray([1.54, 1.55], dtype=float),
            "sample_weights": np.asarray([0.25, 0.75], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*_args, **_kwargs):
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        [7.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0],
                    ],
                    dtype=float,
                ),
            ],
        )

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [
            [9.0, 1.0, 2.0, 4.0, 6.0, 7.0],
            [8.0, 3.0, 4.0, 2.0, 7.0, 8.0],
        ],
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice_factory=lambda: [(3.0, 5.0, "primary"), (3.0, 5.0, "primary")],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        default_source_label="primary",
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    miller_array = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    intensity_array = np.asarray([5.0, 7.0], dtype=float)
    param_set = {
        "a": 3.0,
        "c": 5.0,
        "lambda": 1.54,
        "corto_detector": 100.0,
        "gamma": 1.0,
        "Gamma": 2.0,
        "chi": 3.0,
        "psi": 4.0,
        "psi_z": 5.0,
        "zs": 6.0,
        "zb": 7.0,
        "n2": "n2",
        "debye_x": 0.1,
        "debye_y": 0.2,
        "center": (11.0, 12.0),
        "theta_initial": 8.0,
        "cor_angle": 9.0,
        "optics_mode": diffraction.OPTICS_MODE_EXACT,
    }

    bundle.simulate_hit_tables(miller_array, intensity_array, 32, param_set)
    diagnostics = bundle.last_simulation_diagnostics()
    assert diagnostics["stage"] == "simulate_hit_tables"
    assert diagnostics["miller_shape"] == [2, 3]
    assert diagnostics["miller_count"] == 2
    assert diagnostics["intensity_shape"] == [2]
    assert diagnostics["intensity_count"] == 2
    assert diagnostics["image_size"] == 32
    assert diagnostics["parameter_summary"] == diagnostics["param_summary"]
    assert diagnostics["param_summary"]["optics_mode"] == diffraction.OPTICS_MODE_EXACT
    assert diagnostics["mosaic_array_sizes"] == {
        "beam_x_array": 2,
        "beam_y_array": 2,
        "theta_array": 2,
        "phi_array": 2,
        "wavelength_array": 2,
        "sample_weights": 2,
    }
    assert diagnostics["hit_table_count"] == 2
    assert diagnostics["nonempty_hit_table_count"] == 2
    assert diagnostics["finite_hit_row_total"] == 2
    assert diagnostics["row_count_preview_per_table"] == [1, 1]
    assert diagnostics["projected_peak_count"] == 2

    bundle.simulate_peak_centers(miller_array, intensity_array, 32, param_set)
    diagnostics = bundle.last_simulation_diagnostics()
    assert diagnostics["stage"] == "simulate_peak_centers"
    assert diagnostics["peak_center_count"] == 2
    assert diagnostics["projected_peak_count"] == 2

    bundle.simulate_preview_style_peaks(miller_array, intensity_array, 32, param_set)
    diagnostics = bundle.last_simulation_diagnostics()
    assert diagnostics["stage"] == "simulate_preview_style_peaks"
    assert diagnostics["peak_count"] == 2
    assert diagnostics["projected_peak_count"] == 2
    assert diagnostics["row_count_preview_per_table"] == [1, 1]


def test_geometry_q_group_manager_runtime_simulation_callback_bundle_captures_exceptions() -> None:
    def _build_mosaic(_params):
        return {
            "beam_x_array": np.asarray([1.0], dtype=float),
            "beam_y_array": np.asarray([2.0], dtype=float),
            "theta_array": np.asarray([3.0], dtype=float),
            "phi_array": np.asarray([4.0], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*_args, **_kwargs):
        raise RuntimeError("boom")

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [],
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice_factory=lambda: None,
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        default_source_label="primary",
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    miller_array = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    intensity_array = np.asarray([5.0], dtype=float)
    param_set = {
        "a": 3.0,
        "c": 5.0,
        "lambda": 1.54,
        "corto_detector": 100.0,
        "gamma": 1.0,
        "Gamma": 2.0,
        "chi": 3.0,
        "psi": 4.0,
        "psi_z": 5.0,
        "zs": 6.0,
        "zb": 7.0,
        "n2": "n2",
        "debye_x": 0.1,
        "debye_y": 0.2,
        "center": (11.0, 12.0),
        "theta_initial": 8.0,
        "cor_angle": 9.0,
        "optics_mode": diffraction.OPTICS_MODE_EXACT,
    }

    with pytest.raises(RuntimeError, match="boom"):
        bundle.simulate_preview_style_peaks(miller_array, intensity_array, 32, param_set)

    diagnostics = bundle.last_simulation_diagnostics()
    assert diagnostics["stage"] == "simulate_preview_style_peaks"
    assert diagnostics["status"] == "exception"
    assert diagnostics["miller_shape"] == [1, 3]
    assert diagnostics["miller_count"] == 1
    assert diagnostics["intensity_count"] == 1
    assert diagnostics["exception_type"] == "RuntimeError"
    assert diagnostics["exception_message"] == "boom"
    assert diagnostics["exception"] == {
        "type": "RuntimeError",
        "message": "boom",
    }
    assert diagnostics["exceptions"] == [
        {
            "type": "RuntimeError",
            "message": "boom",
        }
    ]


def test_geometry_q_group_manager_runtime_value_callback_bundle_uses_live_values(
    monkeypatch,
) -> None:
    calls = []
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
            {
                "display_col": np.nan,
                "display_row": 9.0,
            },
        ],
        stored_sim_image=np.zeros((20, 30), dtype=float),
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("primary", 1, 0)},
        cached_entries=[_entry(("q_group", "primary", 1, 0), peak_count=2, total_intensity=10.0)],
    )
    live = {
        "primary_a": 7.0,
        "primary_c": 9.0,
        "var_names": ["gamma"],
        "image_size": 64,
    }

    monkeypatch.setattr(
        geometry_q_group_manager,
        "filter_geometry_fit_simulated_peaks",
        lambda *args, **kwargs: (
            calls.append(("filter_peaks", args, kwargs)) or ([{"filtered": True}], 1, 2)
        ),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "collapse_geometry_fit_simulated_peaks",
        lambda *args, **kwargs: (
            calls.append(("collapse_peaks", args, kwargs)) or ([{"collapsed": True}], 3)
        ),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_entries",
        lambda *args, **kwargs: calls.append(("build_entries", args, kwargs)) or [{"entry": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_export_rows",
        lambda *args, **kwargs: calls.append(("export_rows", args, kwargs)) or [{"row": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "clone_geometry_q_group_entries",
        lambda entries: calls.append(("clone_entries", (entries,), {})) or [{"cloned": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_window_status_text",
        lambda *args, **kwargs: calls.append(("window_status", args, kwargs)) or "status-text",
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_preview_exclude_button_label",
        lambda *args, **kwargs: calls.append(("button_label", args, kwargs)) or "button-label",
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 4}}},
        current_geometry_fit_var_names_factory=lambda: live["var_names"],
        primary_a_factory=lambda: live["primary_a"],
        primary_c_factory=lambda: live["primary_c"],
        image_size_factory=lambda: live["image_size"],
        native_sim_to_display_coords="native-to-display",
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 4.0
    assert cached_preview_peaks[0]["sim_row"] == 5.0
    assert cached_preview_peaks[0]["weight"] == 7.0
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 0)
    assert cached_preview_peaks[0]["label"] == "1,0,0"
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert bundle.filter_simulated_peaks([{"seed": True}]) == ([{"filtered": True}], 1, 2)
    assert bundle.collapse_simulated_peaks(
        [{"seed": True}],
        merge_radius_px=4.5,
    ) == ([{"collapsed": True}], 3)
    assert bundle.build_entries_snapshot() == [{"entry": True}]
    assert bundle.clone_entries([{"a": 1}]) == [{"cloned": True}]
    assert bundle.listed_entries() == [{"cloned": True}]
    assert bundle.listed_keys() == {("q_group", "primary", 1, 0)}
    assert bundle.key_from_jsonable(["q_group", "primary", 1, 0]) == (
        "q_group",
        "primary",
        1,
        0,
    )
    assert bundle.export_rows() == [{"row": True}]
    assert "primary" in bundle.format_line(q_group_state.cached_entries[0])
    assert bundle.current_min_matches() == 4
    assert bundle.excluded_count(q_group_state.cached_entries) == 1
    assert bundle.build_window_status() == "status-text"
    assert bundle.build_preview_exclude_button_label() == "button-label"
    preview_entry = {
        "hkl": (1, 0, 0),
        "source_label": "primary",
        "source_table_index": 0,
        "source_row_index": 1,
    }
    preview_state.excluded_keys = {("peak", "primary", 0, 1, 1, 0, 0)}
    assert bundle.live_preview_match_key(preview_entry) == (
        "peak",
        "primary",
        0,
        1,
        1,
        0,
        0,
    )
    assert bundle.live_preview_match_hkl(preview_entry) == (1, 0, 0)
    assert bundle.live_preview_match_is_excluded(preview_entry) is True
    assert bundle.filter_live_preview_matches([preview_entry, {"hkl": (2, 0, 0)}]) == (
        [{"hkl": (2, 0, 0)}],
        1,
    )
    filtered_pairs, preview_stats, excluded_total = bundle.apply_live_preview_match_exclusions(
        [preview_entry, {"hkl": (2, 0, 0), "distance_px": 3.0}],
        {"search_radius_px": 18.0},
    )
    assert filtered_pairs == [{"hkl": (2, 0, 0), "distance_px": 3.0}]
    assert preview_stats["excluded_count"] == 1
    assert excluded_total == 1

    assert (
        "filter_peaks",
        ([{"seed": True}],),
        {
            "listed_keys": {("q_group", "primary", 1, 0)},
            "q_group_state": q_group_state,
        },
    ) in calls
    assert any(
        name == "collapse_peaks"
        and args == ([{"seed": True}],)
        and kwargs.get("merge_radius_px") == 4.5
        and "profile_cache" in kwargs
        for name, args, kwargs in calls
    )
    assert (
        "build_entries",
        (None,),
        {
            "peak_table_lattice": [(3.0, 5.0, "primary")],
            "peak_records": [dict(runtime_state.peak_records[0])],
            "primary_a": 7.0,
            "primary_c": 9.0,
            "allow_nominal_hkl_indices": False,
        },
    ) in calls
    assert (
        "clone_entries",
        ([{"a": 1}],),
        {},
    ) in calls
    assert (
        "export_rows",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "entries": None,
        },
    ) in calls
    assert (
        "window_status",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "fit_config": {"geometry": {"auto_match": {"min_matches": 4}}},
            "current_geometry_fit_var_names": ["gamma"],
            "entries": None,
        },
    ) in calls
    assert (
        "button_label",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "entries": None,
        },
    ) in calls

    runtime_state.peak_records = []
    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert cached_preview_peaks == []

    runtime_state.stored_max_positions_local = None
    runtime_state.stored_sim_image = None
    runtime_state.stored_peak_table_lattice = None
    runtime_state.peak_records = [
        {
            "display_col": 30.25,
            "display_row": -57.5,
            "native_col": 1.5,
            "native_row": 2.5,
            "sim_col_raw": 1.5,
            "sim_row_raw": 2.5,
            "hkl_raw": [1, 0, 0],
            "intensity": 7.0,
            "phi": 15.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_peak_index": 13,
            "q_group_key": ("q_group", "primary", 1, 0),
            "caked_x": 30.25,
            "caked_y": -57.5,
        },
        {
            "display_col": 31.25,
            "display_row": -56.5,
            "hkl_raw": [2, 0, 0],
            "intensity": 4.0,
            "phi": -15.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 2,
            "source_peak_index": 0,
            "q_group_key": ("q_group", "primary", 2, 0),
            "caked_x": 31.25,
            "caked_y": -56.5,
        },
    ]
    live["primary_a"] = 11.0
    live["primary_c"] = 13.0
    live["image_size"] = 48
    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 1.5
    assert cached_preview_peaks[0]["sim_row"] == 2.5
    assert cached_preview_peaks[0]["display_col"] == 1.5
    assert cached_preview_peaks[0]["display_row"] == 2.5
    assert cached_preview_peaks[0]["weight"] == 7.0
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 0)
    assert cached_preview_peaks[0]["label"] == "1,0,0"
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert cached_preview_peaks[0]["source_branch_index"] == 1
    assert cached_preview_peaks[0]["source_peak_index"] == 1
    assert "source_reflection_index" not in cached_preview_peaks[0]
    runtime_state.peak_records[0].pop("q_group_key")
    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)


def test_geometry_q_group_manager_seeds_peak_records_from_overlay_builder(
    monkeypatch,
) -> None:
    from ra_sim.gui import peak_selection

    runtime_state = state.SimulationRuntimeState(
        peak_records=[],
        stored_sim_image=np.zeros((20, 20), dtype=float),
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=float)],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState()
    seeded_records = [
        {
            "display_col": 1.5,
            "display_row": 2.5,
            "native_col": 4.0,
            "native_row": 5.0,
            "sim_col_raw": 4.0,
            "sim_row_raw": 5.0,
            "hkl_raw": [1, 0, 0],
            "intensity": 7.0,
            "phi": 15.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 1,
            "q_group_key": ("q_group", "primary", 1, 0),
        }
    ]
    overlay_calls: list[dict[str, object]] = []
    build_entries_calls: list[dict[str, object]] = []

    def _seed_overlay_records(simulation_runtime_state, **kwargs):
        overlay_calls.append(dict(kwargs))
        simulation_runtime_state.peak_records = [dict(record) for record in seeded_records]
        return True

    monkeypatch.setattr(
        peak_selection,
        "ensure_runtime_peak_overlay_data",
        _seed_overlay_records,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_entries",
        lambda *args, **kwargs: (
            build_entries_calls.append(dict(kwargs))
            or [{"records": list(kwargs.get("peak_records", ()) or ())}]
        ),
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={},
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 7.0,
        primary_c_factory=lambda: 9.0,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _image_shape: (float(col), float(row)),
        caked_view_enabled_factory=lambda: True,
        native_detector_coords_to_caked_display_coords=(
            lambda col, row: (float(col) + 10.0, float(row) - 10.0)
        ),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert overlay_calls
    assert runtime_state.peak_records == seeded_records
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 4.0
    assert cached_preview_peaks[0]["sim_row"] == 5.0
    assert cached_preview_peaks[0]["caked_x"] == 14.0
    assert cached_preview_peaks[0]["caked_y"] == -5.0
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert bundle.build_entries_snapshot() == [{"records": seeded_records}]
    assert build_entries_calls[0]["peak_records"] == seeded_records
    assert callable(overlay_calls[0]["caked_view_enabled_factory"])
    assert overlay_calls[0]["caked_view_enabled_factory"]() is True
    assert overlay_calls[0]["native_detector_coords_to_caked_display_coords"](
        4.0,
        5.0,
    ) == (14.0, -5.0)


def test_geometry_q_group_manager_rebuilds_stale_peak_records_from_overlay_builder(
    monkeypatch,
) -> None:
    from ra_sim.gui import peak_selection

    runtime_state = state.SimulationRuntimeState(
        peak_records=[{"stale": True}],
        stored_sim_image=np.zeros((20, 20), dtype=float),
        stored_primary_intersection_cache=[np.ones((1, 17), dtype=float)],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState()
    seeded_records = [
        {
            "display_col": 1.5,
            "display_row": 2.5,
            "native_col": 4.0,
            "native_row": 5.0,
            "sim_col_raw": 4.0,
            "sim_row_raw": 5.0,
            "hkl_raw": [1, 0, 0],
            "intensity": 7.0,
            "phi": 15.0,
            "source_label": "primary",
            "source_table_index": 0,
            "source_row_index": 1,
            "q_group_key": ("q_group", "primary", 1, 0),
        }
    ]
    overlay_calls: list[dict[str, object]] = []

    def _seed_overlay_records(simulation_runtime_state, **kwargs):
        overlay_calls.append(dict(kwargs))
        simulation_runtime_state.peak_records = [dict(record) for record in seeded_records]
        return True

    monkeypatch.setattr(
        peak_selection,
        "ensure_runtime_peak_overlay_data",
        _seed_overlay_records,
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={},
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 7.0,
        primary_c_factory=lambda: 9.0,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _image_shape: (float(col), float(row)),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert overlay_calls
    assert runtime_state.peak_records == seeded_records
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 0)
    entries = bundle.build_entries_snapshot()
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 0)


def test_geometry_q_group_manager_rebuilds_detector_display_coords_from_detector_projection() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 1.5,
                "native_row": 2.5,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "source_peak_index": 13,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_sim_image=np.zeros((48, 48), dtype=float),
    )
    sim_display_calls: list[tuple[float, float, tuple[int, int]]] = []
    detector_display_calls: list[tuple[float, float]] = []

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 11.0,
        primary_c_factory=lambda: 13.0,
        image_size_factory=lambda: 48,
        native_sim_to_display_coords=lambda col, row, image_shape: (
            sim_display_calls.append((float(col), float(row), image_shape))
            or (float(col) + 100.0, float(row) + 200.0)
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            detector_display_calls.append((float(col), float(row)))
            or (float(col) + 10.0, float(row) + 20.0)
        ),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 11.5
    assert cached_preview_peaks[0]["sim_row"] == 22.5
    assert cached_preview_peaks[0]["display_col"] == 11.5
    assert cached_preview_peaks[0]["display_row"] == 22.5
    assert cached_preview_peaks[0]["sim_col_raw"] == 11.5
    assert cached_preview_peaks[0]["sim_row_raw"] == 22.5
    assert detector_display_calls
    assert all(call == (1.5, 2.5) for call in detector_display_calls)
    assert sim_display_calls == []


def test_geometry_q_group_manager_reprojects_peak_records_into_current_view() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 300.25,
                "display_row": -500.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_sim_image=np.zeros((20, 30), dtype=float),
    )
    projection_inputs: list[list[dict[str, object]]] = []

    def _project(entries):
        copied_entries = [dict(entry) for entry in (entries or ()) if isinstance(entry, dict)]
        projection_inputs.append(copied_entries)
        return [
            dict(
                entry,
                sim_col=104.0,
                sim_row=205.0,
                display_col=104.0,
                display_row=205.0,
                caked_x=104.0,
                caked_y=205.0,
            )
            for entry in copied_entries
        ]

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 7.0,
        primary_c_factory=lambda: 9.0,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        project_peaks_to_current_view=_project,
        caked_view_enabled_factory=lambda: True,
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert projection_inputs
    assert projection_inputs[0][0]["display_col"] == 4.0
    assert projection_inputs[0][0]["native_col"] == 4.0
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 104.0
    assert cached_preview_peaks[0]["sim_row"] == 205.0
    assert cached_preview_peaks[0]["display_col"] == 104.0
    assert cached_preview_peaks[0]["display_row"] == 205.0


def test_geometry_q_group_manager_keeps_detector_display_when_caked_view_disabled() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 300.25,
                "display_row": -500.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_sim_image=np.zeros((20, 30), dtype=float),
    )
    projection_inputs: list[list[dict[str, object]]] = []

    def _project(entries):
        copied_entries = [dict(entry) for entry in (entries or ()) if isinstance(entry, dict)]
        projection_inputs.append(copied_entries)
        return [
            dict(
                entry,
                sim_col=104.0,
                sim_row=205.0,
                display_col=104.0,
                display_row=205.0,
                caked_x=104.0,
                caked_y=205.0,
            )
            for entry in copied_entries
        ]

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 7.0,
        primary_c_factory=lambda: 9.0,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        project_peaks_to_current_view=_project,
        caked_view_enabled_factory=lambda: False,
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert projection_inputs == []
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 4.0
    assert cached_preview_peaks[0]["sim_row"] == 5.0
    assert cached_preview_peaks[0]["display_col"] == 4.0
    assert cached_preview_peaks[0]["display_row"] == 5.0


def test_geometry_q_group_manager_filters_stale_rows_from_mixed_peak_records() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
            {
                "hkl_raw": [1, 0, 0],
                "intensity": 99.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 9,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
        ],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config={},
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 7.0,
        primary_c_factory=lambda: 9.0,
        image_size_factory=lambda: 64,
        native_sim_to_display_coords=lambda col, row, _image_shape: (float(col), float(row)),
    )

    entries = bundle.build_entries_snapshot()

    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 0)
    assert entries[0]["peak_count"] == 1
    assert entries[0]["total_intensity"] == 7.0


def test_geometry_q_group_manager_peak_record_fallback_restores_trusted_provenance_for_matching_snapshot() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 1.5,
                "native_row": 2.5,
                "sim_col_raw": 1.5,
                "sim_row_raw": 2.5,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "source_peak_index": 13,
                "q_group_key": ("q_group", "primary", 1, 0),
                "caked_x": 30.25,
                "caked_y": -57.5,
            }
        ],
        stored_max_positions_local=None,
        stored_source_reflection_indices_local=[7],
        stored_hit_table_signature=("sig", 0),
        source_row_snapshots={
            0: {
                "background_index": 0,
                "simulation_signature": ("sig", 0),
                "rows": [
                    {
                        "hkl": (1, 0, 0),
                        "source_table_index": 0,
                        "source_row_index": 1,
                    }
                ],
                "row_count": 1,
                "created_from": "fresh_simulation",
            }
        },
    )
    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 11.0,
        primary_c_factory=lambda: 13.0,
        image_size_factory=lambda: 48,
        native_sim_to_display_coords="native-to-display",
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()

    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 1.5
    assert cached_preview_peaks[0]["sim_row"] == 2.5
    assert cached_preview_peaks[0]["source_reflection_index"] == 7
    assert cached_preview_peaks[0]["source_reflection_namespace"] == "full_reflection"
    assert cached_preview_peaks[0]["source_reflection_is_full"] is True
    assert cached_preview_peaks[0]["source_branch_index"] == 1
    assert cached_preview_peaks[0]["source_peak_index"] == 1
    assert callable(bundle.last_live_preview_cache_metadata)
    assert bundle.last_live_preview_cache_metadata()["cache_source"] == "peak_records"
    assert bundle.last_live_preview_cache_metadata()["active_signature_matches"] is True


def test_geometry_q_group_manager_runtime_value_callback_bundle_uses_nominal_cache_grouping() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 10.2,
                "display_row": 20.8,
                "native_col": 10.2,
                "native_row": 20.8,
                "sim_col_raw": 10.2,
                "sim_row_raw": 20.8,
                "hkl_raw": [1.0, 0.0, 1.29],
                "intensity": 12.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 0,
                "q_group_key": ("q_group", "primary", 1, 1),
            }
        ],
        stored_sim_image=np.zeros((32, 32), dtype=float),
        stored_primary_intersection_cache=[
            np.asarray(
                [
                    [
                        1.1,
                        1.2,
                        10.2,
                        20.8,
                        12.0,
                        0.0,
                        1.0,
                        0.0,
                        1.29,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ],
                dtype=float,
            )
        ],
    )
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState()

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    entries = bundle.build_entries_snapshot()

    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 1)
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert cached_preview_peaks[0]["q_group_nominal_hkl"] is True
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)


def test_geometry_q_group_manager_build_entries_snapshot_uses_intersection_cache_when_peak_records_empty() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[],
        stored_max_positions_local=[
            np.asarray(
                [[12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.29]],
                dtype=float,
            )
        ],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
        stored_sim_image=np.zeros((32, 32), dtype=float),
        stored_primary_intersection_cache=[
            np.asarray(
                [
                    [
                        1.1,
                        1.2,
                        10.2,
                        20.8,
                        12.0,
                        0.0,
                        1.0,
                        0.0,
                        1.29,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ],
                dtype=float,
            )
        ],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    entries = bundle.build_entries_snapshot()

    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)
    assert entries[0]["peak_count"] == 1


def test_geometry_q_group_manager_build_entries_snapshot_falls_back_when_stored_hit_tables_empty_list() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_max_positions_local=[],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    entries = bundle.build_entries_snapshot()

    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 0)
    assert entries[0]["peak_count"] == 1


def test_geometry_q_group_manager_build_entries_snapshot_prefers_valid_stored_hit_tables() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_max_positions_local=[
            np.asarray(
                [[12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.0]],
                dtype=float,
            )
        ],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    entries = bundle.build_entries_snapshot()

    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)
    assert entries[0]["peak_count"] == 1


def test_geometry_q_group_manager_build_entries_snapshot_falls_back_when_stored_hit_tables_are_empty_arrays() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_max_positions_local=[np.asarray([], dtype=float)],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    entries = bundle.build_entries_snapshot()

    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 0)
    assert entries[0]["peak_count"] == 1


def test_geometry_q_group_manager_build_entries_snapshot_falls_back_from_empty_cached_stored_entries(
    monkeypatch,
) -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "native_col": 4.0,
                "native_row": 5.0,
                "sim_col_raw": 4.0,
                "sim_row_raw": 5.0,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "phi": 15.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ],
        stored_max_positions_local=[
            np.asarray(
                [[12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.0]],
                dtype=float,
            )
        ],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    build_entries_calls: list[dict[str, object]] = []

    def _build_entries(max_positions_local, *args, **kwargs):
        build_entries_calls.append(
            {
                "has_max_positions_local": max_positions_local is not None,
                "peak_records": list(kwargs.get("peak_records", ()) or ()),
            }
        )
        if max_positions_local is not None:
            return []
        return [{"records": list(kwargs.get("peak_records", ()) or ())}]

    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_entries",
        _build_entries,
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    first_entries = bundle.build_entries_snapshot()
    second_entries = bundle.build_entries_snapshot()

    assert first_entries == [{"records": [dict(runtime_state.peak_records[0])]}]
    assert second_entries == first_entries
    assert build_entries_calls == [
        {
            "has_max_positions_local": True,
            "peak_records": [],
        },
        {
            "has_max_positions_local": False,
            "peak_records": [dict(runtime_state.peak_records[0])],
        },
        {
            "has_max_positions_local": False,
            "peak_records": [dict(runtime_state.peak_records[0])],
        },
    ]


def test_geometry_q_group_manager_caked_preview_uses_stored_hit_tables_without_peak_records() -> (
    None
):
    runtime_state = state.SimulationRuntimeState(
        peak_records=[],
        stored_max_positions_local=[
            np.asarray(
                [
                    [12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 10.0],
                    [8.0, 12.0, 24.0, 0.0, -1.0, 0.0, 10.0],
                ],
                dtype=float,
            )
        ],
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
        stored_hit_table_signature=("cif", "Bi2Se3", 1),
        stored_sim_image=np.zeros((32, 32), dtype=float),
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        caked_view_enabled_factory=lambda: True,
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            0.1 * col,
            0.2 * row,
        ),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    entries = bundle.build_entries_snapshot()

    assert len(cached_preview_peaks) == 2
    assert {entry["q_group_key"] for entry in cached_preview_peaks} == {
        ("q_group", "primary", 1, 10)
    }
    assert cached_preview_peaks[0]["caked_x"] == 1.0
    assert cached_preview_peaks[0]["caked_y"] == 4.0
    assert bundle.last_live_preview_cache_metadata()["cache_source"] == "stored_hit_tables"
    assert bundle.last_live_preview_cache_metadata()["max_positions_row_count"] == 2
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 10)
    assert entries[0]["peak_count"] == 2
    assert runtime_state.geometry_q_group_entries_cache_signature is not None


def test_geometry_q_group_manager_source_row_content_signature_ignores_view_fields() -> None:
    base_rows = [
        {
            "source_table_index": 0,
            "source_row_index": 1,
            "source_reflection_index": 7,
            "source_label": "primary",
            "hkl_raw": [1.0, 0.0, 0.0],
            "intensity": 12.0,
            "native_col": 10.0,
            "native_row": 20.0,
            "display_col": 99.0,
            "display_row": -12.0,
            "sim_col": 88.0,
            "sim_row": 77.0,
            "caked_x": 1.5,
            "caked_y": -3.0,
        }
    ]

    view_variant_rows = [
        dict(
            base_rows[0],
            display_col=-500.0,
            display_row=800.0,
            sim_col=-9.0,
            sim_row=-7.0,
            caked_x=25.0,
            caked_y=35.0,
            current_view_mode="caked",
        )
    ]

    base_signature = geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
        base_rows
    )
    view_variant_signature = (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            view_variant_rows
        )
    )

    assert base_signature == view_variant_signature
    assert (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            [dict(base_rows[0], native_col=11.0)]
        )
        != base_signature
    )
    assert (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            [dict(base_rows[0], hkl_raw=[1.0, 0.0, 1.0])]
        )
        != base_signature
    )
    assert (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            [dict(base_rows[0], intensity=15.0)]
        )
        != base_signature
    )
    assert (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            [dict(base_rows[0], source_row_index=2)]
        )
        != base_signature
    )


def test_geometry_q_group_manager_source_row_content_signature_tracks_theta_initial() -> None:
    base_rows = [
        {
            "source_table_index": 0,
            "source_row_index": 1,
            "source_reflection_index": 7,
            "source_label": "primary",
            "hkl_raw": [1.0, 0.0, 0.0],
            "intensity": 12.0,
            "theta_initial": 5.0,
        }
    ]

    base_signature = geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
        base_rows
    )
    theta_variant_signature = (
        geometry_q_group_manager._geometry_q_group_content_signature_from_source_rows(
            [dict(base_rows[0], theta_initial=7.5)]
        )
    )

    assert theta_variant_signature != base_signature


def test_geometry_q_group_manager_build_entries_snapshot_invalidates_on_q_group_content_signature_change() -> (
    None
):
    first_hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 5.0]],
            dtype=float,
        )
    ]
    second_hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5.0]],
            dtype=float,
        )
    ]
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=first_hit_tables,
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
        stored_hit_table_signature=("sig", 0),
        stored_q_group_content_signature=(
            geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
                first_hit_tables
            )
        ),
    )
    bundle = _make_runtime_q_group_bundle(runtime_state)

    first_entries = bundle.build_entries_snapshot()
    first_signature = runtime_state.geometry_q_group_entries_cache_signature

    runtime_state.stored_max_positions_local = second_hit_tables
    runtime_state.stored_q_group_content_signature = (
        geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(
            second_hit_tables
        )
    )

    second_entries = bundle.build_entries_snapshot()
    second_signature = runtime_state.geometry_q_group_entries_cache_signature

    assert [entry["key"] for entry in first_entries] == [("q_group", "primary", 1, 0)]
    assert [entry["key"] for entry in second_entries] == [("q_group", "primary", 1, 1)]
    assert first_signature != second_signature


def test_geometry_q_group_manager_build_entries_snapshot_is_view_independent() -> None:
    hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5.0]],
            dtype=float,
        )
    ]
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=hit_tables,
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
        stored_hit_table_signature=("sig", 1),
        stored_q_group_content_signature=(
            geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(hit_tables)
        ),
    )
    detector_bundle = _make_runtime_q_group_bundle(
        runtime_state,
        caked_view_enabled_factory=lambda: False,
        project_peaks_to_current_view=lambda entries: [
            dict(entry, display_col=100.0, display_row=200.0)
            for entry in (entries or ())
            if isinstance(entry, dict)
        ],
    )
    caked_bundle = _make_runtime_q_group_bundle(
        runtime_state,
        caked_view_enabled_factory=lambda: True,
        project_peaks_to_current_view=lambda entries: [
            dict(entry, caked_x=-10.0, caked_y=50.0)
            for entry in (entries or ())
            if isinstance(entry, dict)
        ],
    )

    detector_entries = detector_bundle.build_entries_snapshot()
    detector_signature = runtime_state.geometry_q_group_entries_cache_signature
    caked_entries = caked_bundle.build_entries_snapshot()
    caked_signature = runtime_state.geometry_q_group_entries_cache_signature

    assert [entry["key"] for entry in detector_entries] == [("q_group", "primary", 1, 1)]
    assert [entry["key"] for entry in caked_entries] == [("q_group", "primary", 1, 1)]
    assert detector_signature == caked_signature


def test_geometry_q_group_manager_build_entries_snapshot_invalidates_on_lattice_change() -> None:
    hit_tables = [
        np.asarray(
            [[12.0, 10.0, 20.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 5.0]],
            dtype=float,
        )
    ]
    primary_a = {"value": 3.0}
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=hit_tables,
        stored_hit_table_signature=("sig", 2),
        stored_q_group_content_signature=(
            geometry_q_group_manager._geometry_q_group_content_signature_from_hit_tables(hit_tables)
        ),
    )
    bundle = _make_runtime_q_group_bundle(
        runtime_state,
        primary_a_factory=lambda: primary_a["value"],
        primary_c_factory=5.0,
    )

    first_entries = bundle.build_entries_snapshot()
    first_signature = runtime_state.geometry_q_group_entries_cache_signature

    primary_a["value"] = 6.0
    second_entries = bundle.build_entries_snapshot()
    second_signature = runtime_state.geometry_q_group_entries_cache_signature

    assert first_signature != second_signature
    assert np.isclose(first_entries[0]["qr"], (2.0 * np.pi / 3.0) * np.sqrt(4.0 / 3.0))
    assert np.isclose(
        second_entries[0]["qr"],
        (2.0 * np.pi / 6.0) * np.sqrt(4.0 / 3.0),
    )


def test_geometry_q_group_manager_live_preview_exclusion_helpers() -> None:
    excluded_entry = {
        "hkl": (1, 0, 0),
        "source_label": "primary",
        "source_table_index": 2,
        "source_row_index": 3,
        "distance_px": 9.0,
        "confidence": 0.1,
    }
    indexed_entry = {
        "hkl": (2, 0, 1),
        "source_peak_index": 5,
        "distance_px": 2.0,
        "confidence": 0.8,
    }
    row_and_peak_entry = {
        "hkl": (2, 0, 1),
        "source_label": "secondary",
        "source_table_index": 4,
        "source_row_index": 6,
        "source_peak_index": 99,
    }
    branch_and_peak_entry = {
        "hkl": (3, 0, 2),
        "phi": 15.0,
        "source_peak_index": 99,
    }
    coord_entry = {
        "hkl": (3, 0, 2),
        "sim_x": 11.24,
        "sim_y": 8.66,
        "distance_px": 4.0,
        "confidence": 0.4,
    }
    excluded_key = geometry_q_group_manager.live_geometry_preview_match_key(excluded_entry)
    preview_state = state.GeometryPreviewState(excluded_keys={excluded_key})
    preview_state.overlay.pairs = [dict(excluded_entry)]

    assert excluded_key == ("peak", "primary", 2, 3, 1, 0, 0)
    assert geometry_q_group_manager.live_geometry_preview_match_key(row_and_peak_entry) == (
        "peak",
        "secondary",
        4,
        6,
        2,
        0,
        1,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_key(indexed_entry) == (
        "peak_index",
        5,
        2,
        0,
        1,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_key(branch_and_peak_entry) == (
        "peak_index",
        1,
        3,
        0,
        2,
    )
    assert geometry_q_group_manager._live_geometry_preview_compatible_match_keys(
        row_and_peak_entry
    ) == (
        ("peak", "secondary", 4, 6, 2, 0, 1),
        ("peak_index", 99, 2, 0, 1),
    )
    assert geometry_q_group_manager._live_geometry_preview_compatible_match_keys(
        branch_and_peak_entry
    ) == (
        ("peak_index", 1, 3, 0, 2),
        ("peak_index", 99, 3, 0, 2),
    )
    assert geometry_q_group_manager.live_geometry_preview_match_key(coord_entry) == (
        "hkl_coord",
        3,
        0,
        2,
        11.2,
        8.7,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_hkl(coord_entry) == (
        3,
        0,
        2,
    )
    assert (
        geometry_q_group_manager.live_geometry_preview_match_is_excluded(
            preview_state,
            excluded_entry,
        )
        is True
    )

    filtered, excluded_count = geometry_q_group_manager.filter_live_geometry_preview_matches(
        preview_state,
        [excluded_entry, indexed_entry, coord_entry, "bad"],
    )
    assert filtered == [indexed_entry, coord_entry]
    assert excluded_count == 1

    filtered_pairs, stats, excluded_total = (
        geometry_q_group_manager.apply_live_geometry_preview_match_exclusions(
            preview_state,
            [excluded_entry, indexed_entry, coord_entry],
            {"search_radius_px": 18.0},
        )
    )
    assert filtered_pairs == [indexed_entry, coord_entry]
    assert excluded_total == 1
    assert stats["excluded_count"] == 1
    assert stats["matched_count"] == 2
    assert stats["matched_after_exclusions"] == 2
    assert np.isclose(stats["mean_match_distance_px"], 3.0)
    assert np.isclose(stats["p90_match_distance_px"], 3.8)
    assert np.isclose(stats["median_match_confidence"], 0.6)


def test_geometry_q_group_manager_live_preview_source_peak_precedes_coord_key() -> None:
    entry = {
        "hkl": (2, 0, 1),
        "source_peak_index": 5,
        "sim_x": 1.2,
        "sim_y": 3.4,
    }

    assert geometry_q_group_manager.live_geometry_preview_match_key(entry) == (
        "peak_index",
        5,
        2,
        0,
        1,
    )
    assert geometry_q_group_manager._live_geometry_preview_compatible_match_keys(entry) == (
        ("peak_index", 5, 2, 0, 1),
        ("hkl_coord", 2, 0, 1, 1.2, 3.4),
    )


def test_geometry_q_group_manager_live_preview_exclusions_keep_legacy_aliases() -> None:
    legacy_entry = {
        "hkl": (4, 0, 1),
        "sim_x": 7.24,
        "sim_y": 9.66,
    }
    richer_entry = {
        **legacy_entry,
        "source_label": "secondary",
        "source_table_index": 4,
        "source_row_index": 6,
        "source_peak_index": 99,
    }
    preview_state = state.GeometryPreviewState(
        excluded_keys={geometry_q_group_manager.live_geometry_preview_match_key(legacy_entry)}
    )

    assert geometry_q_group_manager.live_geometry_preview_match_key(legacy_entry) == (
        "hkl_coord",
        4,
        0,
        1,
        7.2,
        9.7,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_key(richer_entry) == (
        "peak",
        "secondary",
        4,
        6,
        4,
        0,
        1,
    )
    assert (
        geometry_q_group_manager.live_geometry_preview_match_is_excluded(
            preview_state,
            richer_entry,
        )
        is True
    )
    filtered, excluded_count = geometry_q_group_manager.filter_live_geometry_preview_matches(
        preview_state,
        [richer_entry],
    )
    assert filtered == []
    assert excluded_count == 1


def test_geometry_q_group_manager_filters_simulated_peaks_by_listed_keys_and_exclusions() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "primary", 1, 1)
    q_group_state = state.GeometryQGroupState(disabled_qz_sections={("primary", 1, 1)})
    filtered, excluded_count, total_groups = (
        geometry_q_group_manager.filter_geometry_fit_simulated_peaks(
            [
                {"hkl": (1, 0, 0), "source_label": "primary", "av": 3.0, "cv": 5.0},
                {"hkl": (1, 0, 1), "source_label": "primary", "av": 3.0, "cv": 5.0},
                {"hkl": (2, 0, 0), "source_label": "secondary", "av": 4.0, "cv": 6.0},
                {"hkl": "bad"},
            ],
            listed_keys=[key1, key2],
            q_group_state=q_group_state,
        )
    )

    assert [entry["q_group_key"] for entry in filtered] == [key1]
    assert excluded_count == 3
    assert total_groups == 2


def test_geometry_q_group_manager_collapses_degenerate_simulated_peaks() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 3, 0)
    collapsed, collapsed_count = geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
        [
            {
                "q_group_key": key1,
                "hkl": (1, 0, 0),
                "label": "1,0,0",
                "sim_col": 10.0,
                "sim_row": 10.0,
                "weight": 2.0,
                "source_peak_index": 5,
            },
            {
                "q_group_key": key1,
                "hkl": (0, 1, 0),
                "label": "0,1,0",
                "sim_col": 11.0,
                "sim_row": 12.0,
                "weight": 3.0,
                "source_peak_index": 1,
            },
            {
                "q_group_key": key1,
                "hkl": (1, 0, 1),
                "label": "1,0,1",
                "sim_col": 30.0,
                "sim_row": 30.0,
                "weight": 0.0,
                "source_peak_index": 7,
            },
            {
                "q_group_key": key2,
                "hkl": (1, 1, 0),
                "label": "1,1,0",
                "sim_col": 50.0,
                "sim_row": 50.0,
                "weight": 4.0,
                "source_peak_index": 0,
            },
        ],
        merge_radius_px=3.0,
        one_per_q_group=True,
    )

    assert collapsed_count == 2
    assert [entry["q_group_key"] for entry in collapsed] == [key1, key2]
    assert collapsed[0]["source_peak_index"] == 5
    assert collapsed[0]["weight"] == 6.0
    assert collapsed[0]["degenerate_count"] == 3
    assert collapsed[0]["degenerate_hkls"] == [(1, 0, 0), (0, 1, 0), (1, 0, 1)]
    assert collapsed[0]["selection_reason"] == "mosaic_top_per_q_group"
    assert collapsed[0]["selection_scope"] == "q_group"
    assert collapsed[1]["weight"] == 4.0


def test_collapse_geometry_fit_simulated_peaks_prefers_mosaic_top_per_branch() -> None:
    key = ("q_group", "primary", 1, 0)
    raw_entries = [
        {
            "q_group_key": key,
            "branch_id": "+x",
            "branch_source": "generated",
            "hkl": (1, 0, 0),
            "sim_col": 10.0,
            "sim_row": 10.0,
            "weight": 99.0,
            "mosaic_weight": 0.1,
            "best_sample_index": 3,
            "source_row_index": 30,
        },
        {
            "q_group_key": key,
            "branch_id": "+x",
            "branch_source": "generated",
            "hkl": (1, 0, 0),
            "sim_col": 11.0,
            "sim_row": 10.0,
            "weight": 1.0,
            "mosaic_weight": 0.9,
            "best_sample_index": 0,
            "source_row_index": 31,
        },
        {
            "q_group_key": key,
            "branch_id": "-x",
            "branch_source": "generated",
            "hkl": (0, 1, 0),
            "sim_col": 20.0,
            "sim_row": 20.0,
            "weight": 88.0,
            "mosaic_weight": 0.2,
            "best_sample_index": 4,
            "source_row_index": 40,
        },
        {
            "q_group_key": key,
            "branch_id": "-x",
            "branch_source": "generated",
            "hkl": (0, 1, 0),
            "sim_col": 21.0,
            "sim_row": 20.0,
            "weight": 1.0,
            "mosaic_weight": 0.8,
            "best_sample_index": 1,
            "source_row_index": 41,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
        raw_entries,
        merge_radius_px=100.0,
    )

    assert len(raw_entries) == 4
    assert all("mosaic_top_rank_key" not in entry for entry in raw_entries)
    assert collapsed_count == 2
    by_branch = {entry["branch_id"]: entry for entry in collapsed}
    assert set(by_branch) == {"+x", "-x"}
    assert by_branch["+x"]["best_sample_index"] == 0
    assert by_branch["+x"]["source_row_index"] == 31
    assert by_branch["-x"]["best_sample_index"] == 1
    assert by_branch["-x"]["source_row_index"] == 41
    assert all(entry["selection_reason"] == "mosaic_top_per_branch" for entry in collapsed)
    assert all(isinstance(entry["mosaic_top_rank_key"], tuple) for entry in collapsed)


def test_collapse_geometry_fit_simulated_peaks_group_wide_keeps_top_provenance() -> None:
    key = ("q_group", "primary", 2, 4)
    raw_entries = [
        {
            "q_group_key": key,
            "branch_id": "+x",
            "hkl": (2, 0, 4),
            "source_hkl": (2, 0, 4),
            "source_branch_index": 0,
            "source_reflection_index": 20,
            "source_reflection_key": ("full", 20),
            "source_ray_id": "ray-low",
            "mosaic_weight": 0.2,
            "source_row_index": 1,
            "weight": 1.0,
        },
        {
            "q_group_key": key,
            "branch_id": "-x",
            "hkl": (-2, 0, 4),
            "source_hkl": (-2, 0, 4),
            "source_branch_index": 1,
            "source_reflection_index": 21,
            "source_reflection_key": ("full", 21),
            "source_ray_id": "ray-top",
            "mosaic_weight": 0.95,
            "source_row_index": 2,
            "weight": 1.0,
        },
        {
            "q_group_key": key,
            "branch_id": "+x",
            "hkl": (2, 0, 4),
            "source_hkl": (2, 0, 4),
            "source_branch_index": 0,
            "source_reflection_index": 22,
            "source_reflection_key": ("full", 22),
            "source_ray_id": "ray-mid",
            "mosaic_weight": 0.55,
            "source_row_index": 3,
            "weight": 1.0,
        },
    ]
    original_entries = [dict(entry) for entry in raw_entries]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
        raw_entries,
        one_per_q_group=True,
    )

    assert raw_entries == original_entries
    assert len(collapsed) == 1
    assert collapsed_count == len(raw_entries) - 1
    kept = collapsed[0]
    assert kept["mosaic_weight"] == 0.95
    assert kept["selection_reason"] == "mosaic_top_per_q_group"
    assert kept["selection_scope"] == "q_group"
    assert kept["selected_q_group_key"] == key
    assert kept["branch_id"] == "-x"
    assert kept["source_branch_index"] == 1
    assert kept["source_reflection_index"] == 21
    assert kept["source_reflection_key"] == ("full", 21)
    assert kept["source_ray_id"] == "ray-top"
    assert kept["hkl"] == (-2, 0, 4)
    assert kept["source_hkl"] == (-2, 0, 4)


def test_collapse_geometry_fit_simulated_peaks_default_remains_per_branch() -> None:
    key = ("q_group", "primary", 2, 4)
    raw_entries = [
        {
            "q_group_key": key,
            "branch_id": "+x",
            "hkl": (2, 0, 4),
            "source_branch_index": 0,
            "source_reflection_index": 20,
            "mosaic_weight": 0.2,
            "source_row_index": 1,
        },
        {
            "q_group_key": key,
            "branch_id": "-x",
            "hkl": (-2, 0, 4),
            "source_branch_index": 1,
            "source_reflection_index": 21,
            "mosaic_weight": 0.95,
            "source_row_index": 2,
        },
        {
            "q_group_key": key,
            "branch_id": "+x",
            "hkl": (2, 0, 4),
            "source_branch_index": 0,
            "source_reflection_index": 22,
            "mosaic_weight": 0.55,
            "source_row_index": 3,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
        raw_entries,
    )

    assert len(collapsed) == 2
    assert collapsed_count == 1
    by_branch = {entry["branch_id"]: entry for entry in collapsed}
    assert by_branch["+x"]["source_reflection_index"] == 22
    assert by_branch["-x"]["source_reflection_index"] == 21
    assert all(entry["selection_reason"] == "mosaic_top_per_branch" for entry in collapsed)
    assert all("selection_scope" not in entry for entry in collapsed)


def test_collapse_qr_qz_selection_peaks_keeps_one_intersection_per_source_branch() -> None:
    key = ("q_group", "primary", 5, 2)
    raw_entries = [
        {
            "q_group_key": key,
            "hkl": (5, 0, 2),
            "source_branch_index": 0,
            "source_reflection_index": 50,
            "source_reflection_key": ("full", 50),
            "source_ray_id": "branch-0-low",
            "mosaic_weight": 0.1,
            "source_row_index": 1,
        },
        {
            "q_group_key": key,
            "hkl": (5, 0, 2),
            "source_branch_index": 0,
            "source_reflection_index": 52,
            "source_reflection_key": ("full", 52),
            "source_ray_id": "branch-0-top",
            "mosaic_weight": 0.8,
            "source_row_index": 2,
        },
        {
            "q_group_key": key,
            "hkl": (-5, 0, 2),
            "source_branch_index": 1,
            "source_reflection_index": 60,
            "source_reflection_key": ("full", 60),
            "source_ray_id": "branch-1-low",
            "mosaic_weight": 0.2,
            "source_row_index": 3,
        },
        {
            "q_group_key": key,
            "hkl": (-5, 0, 2),
            "source_branch_index": 1,
            "source_reflection_index": 61,
            "source_reflection_key": ("full", 61),
            "source_ray_id": "branch-1-top",
            "mosaic_weight": 0.7,
            "source_row_index": 4,
        },
    ]
    original_entries = [dict(entry) for entry in raw_entries]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        raw_entries,
    )

    assert raw_entries == original_entries
    assert len(collapsed) == 2
    assert collapsed_count == 2
    by_source_branch = {entry["source_branch_index"]: entry for entry in collapsed}
    assert set(by_source_branch) == {0, 1}
    assert by_source_branch[0]["source_reflection_index"] == 52
    assert by_source_branch[0]["source_reflection_key"] == ("full", 52)
    assert by_source_branch[0]["source_ray_id"] == "branch-0-top"
    assert by_source_branch[0]["hkl"] == (5, 0, 2)
    assert by_source_branch[0]["selection_reason"] == "mosaic_top_per_branch"
    assert by_source_branch[1]["source_reflection_index"] == 61
    assert by_source_branch[1]["source_reflection_key"] == ("full", 61)
    assert by_source_branch[1]["source_ray_id"] == "branch-1-top"
    assert by_source_branch[1]["hkl"] == (-5, 0, 2)
    assert by_source_branch[1]["selection_reason"] == "mosaic_top_per_branch"
    assert by_source_branch[0]["branch_id"] != by_source_branch[1]["branch_id"]


def test_collapse_qr_qz_selection_peaks_keeps_detector_distinct_unknown_branches() -> None:
    key = ("q_group", "primary", 5, 2)
    raw_entries = [
        {
            "q_group_key": key,
            "hkl": (5, 0, 2),
            "source_label": "primary",
            "native_col": 10.0,
            "native_row": 20.0,
            "weight": 1.0,
            "mosaic_weight": 0.4,
        },
        {
            "q_group_key": key,
            "hkl": (5, 0, 2),
            "source_label": "primary",
            "native_col": 90.0,
            "native_row": 20.0,
            "weight": 1.0,
            "mosaic_weight": 0.6,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        raw_entries,
        merge_radius_px=6.0,
    )

    assert collapsed_count == 0
    assert len(collapsed) == 2
    branch_ids = {entry["branch_id"] for entry in collapsed}
    assert len(branch_ids) == 2
    assert all(str(branch_id).startswith("unknown:") for branch_id in branch_ids)
    assert {entry["native_col"] for entry in collapsed} == {10.0, 90.0}


def test_collapse_qr_qz_selection_peaks_one_per_q_group_keeps_single_non_00l() -> None:
    key = ("q_group", "primary", 5, 2)
    raw_entries = [
        {
            "q_group_key": key,
            "hkl": (5, 0, 2),
            "branch_id": "+x",
            "branch_source": "generated",
            "mosaic_weight": 0.2,
            "source_row_index": 10,
            "weight": 1.0,
        },
        {
            "q_group_key": key,
            "hkl": (-5, 0, 2),
            "branch_id": "-x",
            "branch_source": "generated",
            "mosaic_weight": 0.9,
            "source_row_index": 11,
            "weight": 1.0,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        raw_entries,
        one_per_q_group=True,
    )

    assert len(collapsed) == 1
    assert collapsed_count == 1
    assert collapsed[0]["branch_id"] == "-x"
    assert collapsed[0]["source_row_index"] == 11
    assert collapsed[0]["selection_reason"] == "mosaic_top_per_q_group"


def test_collapse_qr_qz_selection_peaks_00l_has_single_branch() -> None:
    key = ("q_group", "primary", 0, 3)
    raw_entries = [
        {
            "q_group_key": key,
            "hkl": (0, 0, 3),
            "source_branch_index": 0,
            "source_reflection_index": 70,
            "source_reflection_key": ("full", 70),
            "source_ray_id": "00l-low",
            "mosaic_weight": 0.2,
            "source_row_index": 1,
        },
        {
            "q_group_key": key,
            "hkl": (0, 0, 3),
            "source_branch_index": 1,
            "source_reflection_index": 71,
            "source_reflection_key": ("full", 71),
            "source_ray_id": "00l-top",
            "mosaic_weight": 0.9,
            "source_row_index": 2,
        },
        {
            "q_group_key": key,
            "hkl": (0, 0, 3),
            "source_branch_index": 0,
            "source_reflection_index": 72,
            "source_reflection_key": ("full", 72),
            "source_ray_id": "00l-mid",
            "mosaic_weight": 0.5,
            "source_row_index": 3,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        raw_entries,
    )

    assert len(collapsed) == 1
    assert collapsed_count == len(raw_entries) - 1
    kept = collapsed[0]
    assert kept["branch_id"] == "00l"
    assert kept["branch_source"] == "generated"
    assert kept["source_branch_index"] == 1
    assert kept["source_reflection_index"] == 71
    assert kept["source_reflection_key"] == ("full", 71)
    assert kept["source_ray_id"] == "00l-top"
    assert kept["mosaic_weight"] == 0.9
    assert kept["selection_reason"] == "mosaic_top_per_branch"
    assert kept["is_00l_collapsed"] is True
    assert len(kept["source_coverage_aliases"]) == 3
    assert {
        (tuple(alias["hkl"]), alias["branch_slot"], alias["source_reflection_index"])
        for alias in kept["source_coverage_aliases"]
    } == {
        ((0, 0, 3), "00l_collapsed", 70),
        ((0, 0, 3), "00l_collapsed", 71),
        ((0, 0, 3), "00l_collapsed", 72),
    }


def test_00l_visual_collapse_still_one_row() -> None:
    key = ("q_group", "primary", 0, 3)
    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        [
            {
                "q_group_key": key,
                "hkl": (0, 0, 3),
                "source_branch_index": 0,
                "source_reflection_index": 70,
                "source_row_index": 1,
                "mosaic_weight": 0.2,
            },
            {
                "q_group_key": key,
                "hkl": (0, 0, 3),
                "source_branch_index": 1,
                "source_reflection_index": 71,
                "source_row_index": 2,
                "mosaic_weight": 0.9,
            },
        ],
    )

    assert len(collapsed) == 1
    assert collapsed_count == 1
    assert len(collapsed[0]["source_coverage_aliases"]) == 2


def test_collapse_qr_qz_selection_peaks_leaves_ungrouped_rows_independent() -> None:
    raw_entries = [
        {
            "q_group_key": None,
            "branch_id": "+x",
            "hkl": (1, 0, 0),
            "mosaic_weight": 0.1,
            "source_row_index": 1,
        },
        {
            "q_group_key": None,
            "branch_id": "-x",
            "hkl": (0, 1, 0),
            "mosaic_weight": 0.9,
            "source_row_index": 2,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        raw_entries,
    )

    assert len(collapsed) == 2
    assert collapsed_count == 0
    assert [entry["source_row_index"] for entry in collapsed] == [1, 2]
    assert all(entry.get("q_group_key") is None for entry in collapsed)


def _single_collapsed_source_row(
    rows: list[dict[str, object]],
    *,
    profile_cache: dict[str, object] | None = None,
) -> int:
    collapsed, _collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        rows,
        profile_cache=profile_cache,
    )

    assert len(collapsed) == 1
    return int(collapsed[0]["source_row_index"])


def test_qr_selection_mosaic_top_rank_uses_fallbacks_before_intensity() -> None:
    key = ("q_group", "primary", 1, 2)

    def _row(source_row: int, **overrides) -> dict[str, object]:
        row = {
            "q_group_key": key,
            "branch_id": "+x",
            "branch_source": "generated",
            "hkl": (1, 0, 2),
            "mosaic_weight": 0.9,
            "theta_offset": 0.0,
            "phi_offset": 0.0,
            "beam_x_offset": 0.0,
            "beam_y_offset": 0.0,
            "wavelength_offset": 0.0,
            "weight": 1.0,
            "intensity": 1.0,
            "source_row_index": source_row,
        }
        row.update(overrides)
        return row

    assert (
        _single_collapsed_source_row(
            [
                _row(10, theta_offset=0.3, intensity=100.0, weight=100.0),
                _row(11, theta_offset=0.01),
            ]
        )
        == 11
    )
    assert (
        _single_collapsed_source_row(
            [
                _row(20, beam_x_offset=0.3, intensity=100.0, weight=100.0),
                _row(21, beam_x_offset=0.01),
            ]
        )
        == 21
    )
    assert (
        _single_collapsed_source_row(
            [
                _row(30, best_sample_index=0, intensity=100.0, weight=100.0),
                _row(31, best_sample_index=1),
            ],
            profile_cache={
                "wavelength_array": np.asarray([1.50, 1.54, 1.60], dtype=float),
            },
        )
        == 31
    )


def _weighted_event_representative_qr_selection_context(monkeypatch):
    profile_cache = {
        "beam_x_array": np.asarray([0.5, 0.5], dtype=float),
        "beam_y_array": np.asarray([0.0, 0.0], dtype=float),
        "theta_array": np.asarray([0.0, 0.2], dtype=float),
        "phi_array": np.asarray([0.0, 0.2], dtype=float),
        "wavelength_array": np.asarray([1.54, 1.56], dtype=float),
        "sample_weights": np.asarray([0.9, 0.1], dtype=float),
    }
    sampled_hit_tables = [
        np.asarray(
            [[100.0, 10.0, 10.0, 0.5, 1.0, 0.0, 2.0, 0.0, 101.0, 1.0]],
            dtype=float,
        )
    ]
    representative_hit_tables = [
        np.asarray(
            [[1.0, 20.0, 20.0, 0.5, 1.0, 0.0, 2.0, 0.0, 100.0, 0.0]],
            dtype=float,
        )
    ]

    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)
    diffraction._set_last_intersection_cache([])
    diffraction._set_last_intersection_cache_from_hit_tables(
        sampled_hit_tables,
        5.0,
        7.0,
        beam_x_array=profile_cache["beam_x_array"],
        beam_y_array=profile_cache["beam_y_array"],
        theta_array=profile_cache["theta_array"],
        phi_array=profile_cache["phi_array"],
        wavelength_array=profile_cache["wavelength_array"],
        representative_hit_tables=representative_hit_tables,
    )

    representative_hit_tables_for_gui = diffraction.intersection_cache_to_hit_tables(
        diffraction.get_last_intersection_cache()
    )
    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        representative_hit_tables_for_gui,
        image_shape=(64, 64),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        primary_a=5.0,
        primary_c=7.0,
        profile_cache=profile_cache,
    )
    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        peaks,
        profile_cache=profile_cache,
    )
    return {
        "profile_cache": profile_cache,
        "sampled_cache": diffraction.get_last_intersection_cache_views()["sampled_event_rows"],
        "representative_cache": diffraction.get_last_intersection_cache(),
        "peaks": peaks,
        "collapsed": collapsed,
        "collapsed_count": collapsed_count,
    }


def test_qr_selection_uses_weighted_event_mosaic_top_representative(monkeypatch) -> None:
    context = _weighted_event_representative_qr_selection_context(monkeypatch)

    assert len(context["peaks"]) == 1
    assert len(context["collapsed"]) == 1
    selected = context["collapsed"][0]
    assert selected["best_sample_index"] == 0
    assert selected["source_row_index"] == 100
    assert selected["mosaic_weight"] == pytest.approx(0.9)
    assert selected["weight"] == pytest.approx(1.0)
    assert context["collapsed_count"] == 0


def test_qr_selection_does_not_use_weighted_sampled_event_when_representative_exists(
    monkeypatch,
) -> None:
    context = _weighted_event_representative_qr_selection_context(monkeypatch)

    sampled_rows = np.vstack(context["sampled_cache"])
    assert sampled_rows[0, cache_schema.CACHE_COL_INTENSITY] == pytest.approx(100.0)
    assert sampled_rows[0, cache_schema.CACHE_COL_BEST_SAMPLE_INDEX] == pytest.approx(1.0)

    representative_rows = np.vstack(context["representative_cache"])
    assert representative_rows[0, cache_schema.CACHE_COL_INTENSITY] == pytest.approx(1.0)
    assert representative_rows[0, cache_schema.CACHE_COL_BEST_SAMPLE_INDEX] == pytest.approx(0.0)

    selected = context["collapsed"][0]
    assert selected["best_sample_index"] == 0
    assert selected["source_row_index"] == 100


def test_qr_selection_preserves_clicked_branch_then_mosaic_top_candidate() -> None:
    key = ("q_group", "primary", 1, 2)
    rows = [
        {
            "q_group_key": key,
            "branch_id": "+x",
            "branch_source": "generated",
            "hkl": (1, 0, 2),
            "best_sample_index": 1,
            "source_row_index": 30,
            "mosaic_weight": 0.1,
            "weight": 100.0,
        },
        {
            "q_group_key": key,
            "branch_id": "+x",
            "branch_source": "generated",
            "hkl": (1, 0, 2),
            "best_sample_index": 0,
            "source_row_index": 31,
            "mosaic_weight": 0.9,
            "weight": 1.0,
        },
        {
            "q_group_key": key,
            "branch_id": "-x",
            "branch_source": "generated",
            "hkl": (1, 0, 2),
            "best_sample_index": 2,
            "source_row_index": 41,
            "mosaic_weight": 0.99,
            "weight": 200.0,
        },
    ]

    collapsed, collapsed_count = geometry_q_group_manager.collapse_qr_qz_selection_peaks(
        rows,
        profile_cache={"sample_weights": np.asarray([0.9, 0.1, 0.99], dtype=float)},
    )

    assert collapsed_count == 1
    by_branch = {entry["branch_id"]: entry for entry in collapsed}
    assert set(by_branch) == {"+x", "-x"}
    assert by_branch["+x"]["best_sample_index"] == 0
    assert by_branch["+x"]["source_row_index"] == 31
    assert by_branch["+x"]["mosaic_weight"] == pytest.approx(0.9)
    assert by_branch["-x"]["source_row_index"] == 41


def test_collapse_geometry_fit_simulated_peaks_uses_profile_cache_branch_before_unknown() -> None:
    key = ("q_group", "primary", 1, 0)
    raw_entries = [
        {
            "q_group_key": key,
            "hkl": (1, 0, 0),
            "sim_col": 10.0,
            "sim_row": 10.0,
            "weight": 99.0,
            "best_sample_index": 0,
            "source_row_index": 30,
        },
        {
            "q_group_key": key,
            "hkl": (1, 0, 0),
            "sim_col": 11.0,
            "sim_row": 10.0,
            "weight": 1.0,
            "best_sample_index": 1,
            "source_row_index": 31,
        },
    ]
    profile_cache = {
        "beam_x_array": np.asarray([-0.5, 0.5], dtype=float),
        "sample_weights": np.asarray([0.2, 0.9], dtype=float),
    }

    collapsed, collapsed_count = geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
        raw_entries,
        merge_radius_px=100.0,
        profile_cache=profile_cache,
    )

    assert all("branch_id" not in entry for entry in raw_entries)
    assert all("mosaic_top_rank_key" not in entry for entry in raw_entries)
    assert collapsed_count == 0
    by_branch = {entry["branch_id"]: entry for entry in collapsed}
    assert set(by_branch) == {"-x", "+x"}
    assert by_branch["-x"]["source_row_index"] == 30
    assert by_branch["-x"]["branch_source"] == "generated"
    assert by_branch["-x"]["mosaic_weight"] == 0.2
    assert by_branch["+x"]["source_row_index"] == 31
    assert by_branch["+x"]["branch_source"] == "generated"
    assert by_branch["+x"]["mosaic_weight"] == 0.9


def test_build_geometry_fit_simulated_peaks_uses_profile_cache_branch_and_mosaic() -> None:
    rows = [
        np.asarray([10.0, 1.0, 2.0, 45.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.asarray([20.0, 3.0, 4.0, 45.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=float),
    ]
    profile_cache = {
        "beam_x_array": np.asarray([-0.5, 0.5], dtype=float),
        "beam_y_array": np.asarray([0.05, 0.15], dtype=float),
        "theta_array": np.asarray([0.01, 0.02], dtype=float),
        "phi_array": np.asarray([0.03, 0.04], dtype=float),
        "wavelength_array": np.asarray([1.54, 1.55], dtype=float),
        "sample_weights": np.asarray([0.25, 0.75], dtype=float),
    }

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [np.asarray(rows, dtype=float)],
        image_shape=(64, 64),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        profile_cache=profile_cache,
    )

    by_sample = {entry["best_sample_index"]: entry for entry in peaks}
    assert by_sample[0]["branch_id"] == "-x"
    assert by_sample[1]["branch_id"] == "+x"
    assert by_sample[0]["source_branch_index"] == 1
    assert by_sample[1]["source_branch_index"] == 1
    assert by_sample[0]["mosaic_weight"] == 0.25
    assert by_sample[1]["mosaic_weight"] == 0.75
    assert by_sample[0]["beam_x_offset"] == -0.5
    assert by_sample[1]["beam_y_offset"] == 0.15
    assert by_sample[0]["theta_offset"] == 0.01
    assert by_sample[1]["phi_offset"] == 0.04
    assert by_sample[1]["wavelength_offset"] == 1.55


def test_build_geometry_fit_simulated_peaks_reannotates_after_mirrored_branch_repair() -> None:
    rows = [
        np.asarray([10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.asarray([10.0, 50.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=float),
    ]

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [np.asarray(rows, dtype=float)],
        image_shape=(64, 64),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
    )

    assert {entry["source_branch_index"] for entry in peaks} == {0, 1}
    branch_ids = {entry["branch_id"] for entry in peaks}
    assert len(branch_ids) == 2
    assert all(str(branch_id).startswith("unknown:") for branch_id in branch_ids)


def test_geometry_q_group_manager_formats_lines_and_builds_status_text() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("secondary", 2, 1)},
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ],
    )

    line = geometry_q_group_manager.format_geometry_q_group_line(q_group_state.cached_entries[0])
    status = geometry_q_group_manager.build_geometry_q_group_window_status_text(
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 4}}},
        current_geometry_fit_var_names=["gamma"],
    )

    assert "primary" in line
    assert "Qr=    1.25" in line
    assert "m=   1" in line
    assert "L=   0" in line
    assert "Gz=" not in line
    assert "Qz=    0.50" in line
    assert "1.2500" not in line
    assert "0.5000" not in line
    assert "HKL=" in line
    assert ", ..." in line
    assert (
        geometry_q_group_manager.current_geometry_auto_match_min_matches(
            {},
            ["gamma", "chi"],
        )
        == 6
    )
    assert (
        geometry_q_group_manager.geometry_q_group_excluded_count(
            preview_state,
            q_group_state,
        )
        == 1
    )
    assert (
        geometry_q_group_manager.build_geometry_preview_exclude_button_label(
            preview_state=preview_state,
            q_group_state=q_group_state,
        )
        == "Choose Active Qr/Qz Groups (1 off)"
    )
    assert "Included Qr/Qz groups: 1/2" in status
    assert "Selected peaks: 2/5" in status
    assert "Need >= 4  short 2" in status
    assert "Intensity=10.000/30.000" in status


def test_geometry_q_group_manager_live_preview_config_overlay_and_render_helpers() -> None:
    preview_cfg = geometry_q_group_manager.build_live_geometry_preview_auto_match_config(
        {
            "geometry": {
                "auto_match": {
                    "search_radius_px": 20.0,
                    "max_display_markers": 1,
                    "max_p90_distance_px": 30.0,
                    "max_mean_distance_px": 10.0,
                }
            }
        }
    )
    empty_overlay = geometry_q_group_manager.build_empty_live_geometry_preview_overlay_state(
        signature="empty-sig",
        min_matches=3,
        max_display_markers=5,
        q_group_total=4,
        q_group_excluded=1,
        excluded_q_peaks=2,
    )
    overlay_state = geometry_q_group_manager.build_live_geometry_preview_overlay_state(
        signature="sig",
        matched_pairs=[
            {"sim_x": 1.0, "sim_y": 2.0, "x": 11.0, "y": 12.0},
            {"sim_x": 3.0, "sim_y": 4.0, "x": 13.0, "y": 14.0},
        ],
        match_stats={
            "simulated_count": 4,
            "search_radius_px": 18.0,
            "p90_match_distance_px": 35.0,
            "mean_match_distance_px": 11.0,
        },
        preview_auto_match_cfg=preview_cfg,
        auto_match_attempts=[{"radius": 18.0}],
        min_matches=2,
        q_group_total=3,
        q_group_excluded=1,
        excluded_q_peaks=2,
        collapsed_degenerate_peaks=1,
    )

    assert preview_cfg["relax_on_low_matches"] is False
    assert preview_cfg["context_margin_px"] == 192.0
    assert empty_overlay["pairs"] == []
    assert empty_overlay["q_group_total"] == 4
    assert overlay_state["quality_fail"] is True
    assert overlay_state["max_display_markers"] == 1
    assert overlay_state["collapsed_degenerate_peaks"] == 1
    assert overlay_state["auto_match_attempts"] == [{"radius": 18.0}]

    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = list(overlay_state["pairs"])
    preview_state.overlay.simulated_count = int(overlay_state["simulated_count"])
    preview_state.overlay.min_matches = int(overlay_state["min_matches"])
    preview_state.overlay.best_radius = float(overlay_state["best_radius"])
    preview_state.overlay.mean_dist = float(overlay_state["mean_dist"])
    preview_state.overlay.p90_dist = float(overlay_state["p90_dist"])
    preview_state.overlay.quality_fail = bool(overlay_state["quality_fail"])
    preview_state.overlay.max_display_markers = int(overlay_state["max_display_markers"])
    preview_state.overlay.q_group_total = int(overlay_state["q_group_total"])
    preview_state.overlay.q_group_excluded = int(overlay_state["q_group_excluded"])
    preview_state.overlay.collapsed_degenerate_peaks = int(
        overlay_state["collapsed_degenerate_peaks"]
    )

    draw_calls = []
    status_messages = []
    rendered = geometry_q_group_manager.render_live_geometry_preview_overlay_state(
        preview_state=preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: draw_calls.append(
            (list(pairs), max_display_markers)
        ),
        filter_live_preview_matches=lambda pairs: (list(pairs), 1),
        set_status_text=lambda text: status_messages.append(text),
        update_status=True,
    )
    quiet_render = geometry_q_group_manager.render_live_geometry_preview_overlay_state(
        preview_state=preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: draw_calls.append(
            ("quiet", list(pairs), max_display_markers)
        ),
        filter_live_preview_matches=lambda pairs: ([], 0),
        set_status_text=lambda text: status_messages.append(f"quiet:{text}"),
        update_status=False,
    )

    assert rendered is True
    assert quiet_render is True
    assert draw_calls[0][1] == 1
    assert draw_calls[1][0] == "quiet"
    assert "Live auto-match preview: 2/4 active peaks" in status_messages[0]
    assert "Excluded=1." in status_messages[0]
    assert "Qr/Qz groups on=2/3." in status_messages[0]
    assert "Degenerate collapsed=1." in status_messages[0]
    assert "Geometry fit would stop on the quality gate." in status_messages[0]
    assert "Showing 1/2 overlays." in status_messages[0]
    assert len(status_messages) == 1


def test_geometry_q_group_manager_runtime_live_preview_render_helpers(
    monkeypatch,
) -> None:
    draw_calls = []
    status_messages = []
    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = [{"hkl": (1, 0, 0), "x": 1.0, "y": 2.0}]
    preview_state.overlay.simulated_count = 1
    preview_state.overlay.min_matches = 1
    preview_state.overlay.max_display_markers = 7

    monkeypatch.setattr(
        geometry_q_group_manager.gui_overlays,
        "draw_live_geometry_preview_overlay",
        lambda axis, pairs, **kwargs: draw_calls.append((axis, list(pairs), kwargs)),
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        axis="axis",
        geometry_preview_artists=[],
        draw_idle=lambda: None,
        normalize_hkl_key=lambda value: tuple(value) if isinstance(value, tuple) else None,
        live_preview_match_is_excluded=lambda _entry: False,
        filter_live_preview_matches=lambda pairs: (list(pairs), 0),
        set_status_text=lambda text: status_messages.append(text),
    )

    assert geometry_q_group_manager.runtime_live_geometry_preview_enabled(bindings) is True
    geometry_q_group_manager.draw_runtime_live_geometry_preview_overlay(
        bindings,
        [{"hkl": (2, 0, 0), "x": 3.0, "y": 4.0}],
        max_display_markers=9,
    )
    assert (
        geometry_q_group_manager.render_runtime_live_geometry_preview_state(
            bindings,
            update_status=True,
        )
        is True
    )

    assert draw_calls[0][0] == "axis"
    assert draw_calls[0][1] == [{"hkl": (2, 0, 0), "x": 3.0, "y": 4.0}]
    assert draw_calls[0][2]["max_display_markers"] == 9
    assert draw_calls[1][1] == preview_state.overlay.pairs
    assert draw_calls[1][2]["max_display_markers"] == 7
    assert "Live auto-match preview: 1/1 active peaks" in status_messages[0]


def test_geometry_q_group_manager_runtime_live_preview_simulated_peak_resolution() -> None:
    preferred_cache_calls: list[str] = []
    preferred_sim_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, dict[str, object]]] = []
    preferred_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: (
            preferred_cache_calls.append("cache") or [{"source": "cache"}]
        ),
        simulate_preview_style_peaks=lambda miller, intensities, image_size, params: (
            preferred_sim_calls.append(
                (tuple(miller.shape), tuple(intensities.shape), int(image_size), dict(params))
            )
            or [{"source": "central"}]
        ),
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=48,
        current_geometry_fit_params_factory=lambda: {"omega": 0.5},
    )

    preferred = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        preferred_bindings,
        update_status=True,
    )

    assert preferred == [{"source": "central"}]
    assert preferred_sim_calls == [((1, 3), (1,), 48, {"omega": 0.5})]
    assert preferred_cache_calls == []

    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        simulate_preview_style_peaks=lambda miller, intensities, image_size, params: [
            {
                "size": int(image_size),
                "miller_shape": tuple(miller.shape),
                "intensity_shape": tuple(intensities.shape),
                "params": dict(params),
            }
        ],
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=128,
        current_geometry_fit_params_factory=lambda: {"omega": 1.5},
    )

    simulated = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        success_bindings,
        update_status=True,
    )

    assert simulated == [
        {
            "size": 128,
            "miller_shape": (1, 3),
            "intensity_shape": (1,),
            "params": {"omega": 1.5},
        }
    ]

    failure_events = []
    failure_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        simulate_preview_style_peaks=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=64,
        current_geometry_fit_params_factory=lambda: {"omega": 2.0},
        clear_geometry_preview_artists=lambda: failure_events.append("clear"),
        set_status_text=lambda text: failure_events.append(text),
    )

    failed = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        failure_bindings,
        update_status=True,
    )

    assert failed is None
    assert failure_events == [
        "clear",
        "Live auto-match preview unavailable: failed to simulate peaks (boom).",
    ]

    empty_events = []
    empty_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=False,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        clear_geometry_preview_artists=lambda: empty_events.append("clear"),
        set_status_text=lambda text: empty_events.append(text),
    )

    empty = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        empty_bindings,
        update_status=True,
    )

    assert empty is None
    assert empty_events == [
        "clear",
        "Live auto-match preview unavailable: no simulated peaks are available.",
    ]


def test_geometry_q_group_manager_runtime_live_preview_background_resolution() -> None:
    disabled_events = []
    disabled_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        clear_geometry_preview_artists=lambda: disabled_events.append("clear"),
        set_status_text=lambda text: disabled_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            disabled_bindings,
            update_status=True,
        )
        is None
    )
    assert disabled_events == ["clear"]

    caked_events = []
    caked_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: True,
        clear_geometry_preview_artists=lambda: caked_events.append("clear"),
        set_status_text=lambda text: caked_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            caked_bindings,
            update_status=True,
        )
        is None
    )
    assert caked_events == [
        "clear",
        "Live auto-match preview unavailable in 2D caked view.",
    ]

    hidden_events = []
    hidden_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: False,
        background_visible=False,
        current_background_display_factory=lambda: None,
        clear_geometry_preview_artists=lambda: hidden_events.append("clear"),
        set_status_text=lambda text: hidden_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            hidden_bindings,
            update_status=True,
        )
        is None
    )
    assert hidden_events == [
        "clear",
        "Live auto-match preview unavailable: background image is hidden.",
    ]

    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: False,
        background_visible=True,
        current_background_display_factory=lambda: "background-image",
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            success_bindings,
            update_status=True,
        )
        == "background-image"
    )


def test_geometry_q_group_manager_runtime_live_preview_seed_state_resolution() -> None:
    preview_state = state.GeometryPreviewState()
    no_group_events = []
    no_group_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: ([], 2, 5),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            list(peaks or []),
            int(merge_radius_px),
        ),
        excluded_q_group_count=lambda: 3,
        clear_geometry_preview_artists=lambda: no_group_events.append("clear"),
        set_status_text=lambda text: no_group_events.append(text),
    )

    no_groups = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        no_group_bindings,
        [{"seed": True}],
        preview_auto_match_cfg={"max_display_markers": 11},
        min_matches=4,
        signature=("sig", 1),
        update_status=True,
    )

    assert no_groups is None
    assert no_group_events == [
        "clear",
        "Live auto-match preview unavailable: no Qr/Qz groups are selected.",
    ]
    assert preview_state.overlay.q_group_total == 5
    assert preview_state.overlay.q_group_excluded == 3
    assert preview_state.overlay.excluded_q_peaks == 2
    assert preview_state.overlay.max_display_markers == 11

    collapse_events = []
    collapse_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: (list(peaks or []), 1, 4),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            collapse_events.append(("collapse", list(peaks), merge_radius_px)),
            ([], 7),
        )[-1],
        clear_geometry_preview_artists=lambda: collapse_events.append("clear"),
        set_status_text=lambda text: collapse_events.append(text),
    )

    collapsed = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        collapse_bindings,
        [{"seed": 1}],
        preview_auto_match_cfg={"search_radius_px": 30.0},
        min_matches=4,
        signature=("sig", 2),
        update_status=True,
    )

    assert collapsed is None
    assert collapse_events == [
        ("collapse", [{"seed": 1}], 6.0),
        "clear",
        "Live auto-match preview unavailable: no geometry-fit seeds remain after Qr/Qz collapse.",
    ]

    success_events = []
    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: (list(peaks or []), 2, 6),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            success_events.append(("collapse", merge_radius_px)),
            ([{"collapsed": True}], 3),
        )[-1],
        set_status_text=lambda text: success_events.append(text),
    )

    success = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        success_bindings,
        [{"seed": 2}],
        preview_auto_match_cfg={"degenerate_merge_radius_px": 4.5},
        min_matches=5,
        signature=("sig", 3),
        update_status=True,
    )

    assert success == ([{"collapsed": True}], 2, 6, 3)
    assert success_events == [("collapse", 4.5)]


def test_qr_qz_initial_raw_cache_preview_draws_one_top_marker_per_branch() -> None:
    group_key = ("q_group", "primary", 3, 2)
    raw_cache_rows = [
        {
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 30,
            "source_reflection_key": ("full", 30),
            "source_ray_id": "ray-a",
            "hkl": (3, 0, 2),
            "mosaic_weight": 0.15,
            "sim_col": 10.0,
            "sim_row": 20.0,
        },
        {
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_branch_index": 1,
            "source_reflection_index": 31,
            "source_reflection_key": ("full", 31),
            "source_ray_id": "ray-winner",
            "hkl": (-3, 0, 2),
            "mosaic_weight": 0.85,
            "sim_col": 11.0,
            "sim_row": 21.0,
        },
        {
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 32,
            "source_reflection_key": ("full", 32),
            "source_ray_id": "ray-b",
            "hkl": (3, 0, 2),
            "mosaic_weight": 0.45,
            "sim_col": 12.0,
            "sim_row": 22.0,
        },
    ]
    preview_state = state.GeometryPreviewState()
    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: (list(peaks or []), 0, 1),
        collapse_simulated_peaks=geometry_q_group_manager.collapse_qr_qz_selection_peaks,
    )

    seed_state = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        bindings,
        raw_cache_rows,
        preview_auto_match_cfg={"degenerate_merge_radius_px": 4.0},
        min_matches=1,
        signature=("raw-cache",),
    )

    assert seed_state is not None
    visible_rows, _excluded, _total, collapsed_count = seed_state
    visible_for_group = [row for row in visible_rows if row.get("q_group_key") == group_key]
    assert len(visible_for_group) == 2
    by_branch = {row["branch_id"]: row for row in visible_for_group}
    assert by_branch["+x"]["mosaic_weight"] == 0.45
    assert by_branch["+x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["+x"]["source_branch_index"] == 0
    assert by_branch["+x"]["source_reflection_index"] == 32
    assert by_branch["+x"]["source_reflection_key"] == ("full", 32)
    assert by_branch["+x"]["source_ray_id"] == "ray-b"
    assert by_branch["-x"]["mosaic_weight"] == 0.85
    assert by_branch["-x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["-x"]["source_branch_index"] == 1
    assert by_branch["-x"]["source_reflection_index"] == 31
    assert by_branch["-x"]["source_reflection_key"] == ("full", 31)
    assert by_branch["-x"]["source_ray_id"] == "ray-winner"
    assert collapsed_count == len(raw_cache_rows) - 2

    overlay_state = geometry_q_group_manager.build_live_geometry_preview_overlay_state(
        signature=("raw-cache", "draw"),
        matched_pairs=[
            {
                **dict(row),
                "x": 100.0,
                "y": 200.0,
                "bg_x": 100.0,
                "bg_y": 200.0,
            }
            for row in visible_for_group
        ],
        match_stats={
            "simulated_count": len(visible_rows),
            "search_radius_px": 4.0,
            "mean_match_distance_px": 0.0,
            "p90_match_distance_px": 0.0,
        },
        preview_auto_match_cfg={"max_display_markers": 5},
        auto_match_attempts=[],
        min_matches=1,
        q_group_total=1,
        q_group_excluded=0,
        excluded_q_peaks=0,
        collapsed_degenerate_peaks=collapsed_count,
    )
    preview_state.overlay.pairs = list(overlay_state["pairs"])
    preview_state.overlay.simulated_count = int(overlay_state["simulated_count"])
    preview_state.overlay.min_matches = int(overlay_state["min_matches"])
    preview_state.overlay.best_radius = float(overlay_state["best_radius"])
    preview_state.overlay.mean_dist = float(overlay_state["mean_dist"])
    preview_state.overlay.p90_dist = float(overlay_state["p90_dist"])
    preview_state.overlay.quality_fail = bool(overlay_state["quality_fail"])
    preview_state.overlay.max_display_markers = int(overlay_state["max_display_markers"])
    preview_state.overlay.q_group_total = int(overlay_state["q_group_total"])
    preview_state.overlay.q_group_excluded = int(overlay_state["q_group_excluded"])
    preview_state.overlay.collapsed_degenerate_peaks = int(
        overlay_state["collapsed_degenerate_peaks"]
    )
    draw_calls: list[tuple[list[dict[str, object]], int]] = []

    rendered = geometry_q_group_manager.render_live_geometry_preview_overlay_state(
        preview_state=preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: draw_calls.append(
            (list(pairs), int(max_display_markers))
        ),
        filter_live_preview_matches=lambda pairs: (list(pairs), 0),
        set_status_text=lambda _text: None,
        update_status=True,
    )

    assert rendered is True
    drawn_pairs = draw_calls[0][0]
    drawn_for_group = [row for row in drawn_pairs if row.get("q_group_key") == group_key]
    assert len(drawn_for_group) == 2
    drawn_by_branch = {row["branch_id"]: row for row in drawn_for_group}
    assert drawn_by_branch["+x"]["source_reflection_index"] == 32
    assert drawn_by_branch["+x"]["source_ray_id"] == "ray-b"
    assert drawn_by_branch["-x"]["source_reflection_index"] == 31
    assert drawn_by_branch["-x"]["source_ray_id"] == "ray-winner"


def test_qr_qz_ui_paths_use_selection_collapse_wrapper() -> None:
    gui_dir = Path(geometry_q_group_manager.__file__).parent
    checked_paths = [
        gui_dir / "geometry_q_group_manager.py",
        gui_dir / "manual_geometry.py",
    ]
    unexpected: list[str] = []
    for path in checked_paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "collapse_geometry_fit_simulated_peaks(" not in line:
                continue
            if "def collapse_geometry_fit_simulated_peaks" in line:
                continue
            if "return collapse_geometry_fit_simulated_peaks" in line:
                continue
            unexpected.append(f"{path.name}:{line_number}:{line.strip()}")

    assert unexpected == []


def test_runtime_session_keeps_qr_qz_collapse_out_of_fit_internals() -> None:
    runtime_path = (
        Path(geometry_q_group_manager.__file__).parent / "_runtime" / "runtime_session.py"
    )
    runtime_source = runtime_path.read_text(encoding="utf-8")

    assert "collapse_simulated_peaks=_collapse_qr_qz_selection_peaks" in runtime_source
    assert 'globals()["_collapse_qr_qz_selection_peaks"]' in runtime_source
    assert "def _collapse_geometry_fit_simulated_peaks" in runtime_source
    assert "gui_geometry_q_group_manager.collapse_geometry_fit_simulated_peaks" in runtime_source
    assert (
        "_collapse_geometry_fit_simulated_peaks = (\n"
        "        geometry_q_group_runtime_value_callbacks.collapse_simulated_peaks"
        not in runtime_source
    )


def test_runtime_session_auto_refreshes_listed_qr_qz_after_simulation_update() -> None:
    runtime_path = (
        Path(geometry_q_group_manager.__file__).parent / "_runtime" / "runtime_session.py"
    )
    runtime_source = runtime_path.read_text(encoding="utf-8")

    assert "q_group_auto_refresh_needed = bool(" in runtime_source
    assert "or q_group_auto_refresh_needed" in runtime_source
    assert "auto_q_group_list_refresh = bool(" in runtime_source
    assert 'getattr(geometry_q_group_state, "refresh_requested", False)' in runtime_source
    assert "refresh_q_group_listing_requested or auto_q_group_list_refresh" in runtime_source
    assert "capture_runtime_geometry_q_group_entries_snapshot" in runtime_source
    capture_gate = runtime_source.index(
        "if not need_hit_table_refresh and (\n"
        "        refresh_q_group_listing_requested or auto_q_group_list_refresh"
    )
    consume_call = runtime_source.index(
        "gui_controllers.consume_geometry_q_group_refresh_request",
        capture_gate,
    )
    assert consume_call > capture_gate


def test_runtime_manual_projection_caked_mode_uses_active_primary_view() -> None:
    runtime_path = (
        Path(geometry_q_group_manager.__file__).parent / "_runtime" / "runtime_session.py"
    )
    runtime_source = runtime_path.read_text(encoding="utf-8")

    assert "caked_view_enabled=lambda: _active_caked_primary_view()" in runtime_source


def test_geometry_q_group_manager_runtime_live_preview_match_result_application(
    monkeypatch,
) -> None:
    events = []
    preview_state = state.GeometryPreviewState()

    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "replace_geometry_preview_overlay_state",
        lambda preview_state_value, overlay_state: events.append(
            ("replace", preview_state_value, overlay_state)
        ),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "render_runtime_live_geometry_preview_state",
        lambda bindings, *, update_status=True: (
            events.append(("render", bindings, update_status)),
            True,
        )[-1],
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        excluded_q_group_count=lambda: 2,
    )

    applied = geometry_q_group_manager.apply_runtime_live_geometry_preview_match_results(
        bindings,
        signature=("sig", 4),
        matched_pairs=[{"x": 1.0, "y": 2.0, "sim_x": 0.5, "sim_y": 1.5}],
        match_stats={
            "simulated_count": 4,
            "search_radius_px": 18.0,
            "mean_match_distance_px": 6.0,
            "p90_match_distance_px": 9.0,
        },
        preview_auto_match_cfg={"max_display_markers": 6},
        auto_match_attempts=[{"radius": 18.0}],
        min_matches=3,
        q_group_total=5,
        excluded_q_peaks=1,
        collapsed_deg_preview=2,
        update_status=False,
    )

    assert applied is True
    assert events[0][0] == "replace"
    assert events[0][1] is preview_state
    assert events[0][2]["signature"] == ("sig", 4)
    assert events[0][2]["q_group_total"] == 5
    assert events[0][2]["q_group_excluded"] == 2
    assert events[0][2]["excluded_q_peaks"] == 1
    assert events[0][2]["collapsed_degenerate_peaks"] == 2
    assert events[0][2]["auto_match_attempts"] == [{"radius": 18.0}]
    assert events[1] == ("render", bindings, False)


def test_refresh_geometry_q_group_window_uses_cached_entries_and_view_helpers(
    monkeypatch,
) -> None:
    calls = {}
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(("q_group", "primary", 1, 0), peak_count=2, total_intensity=10.0)]
    )

    def _refresh(**kwargs):
        calls["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "refresh_geometry_q_group_window",
        _refresh,
    )

    ok = geometry_q_group_manager.refresh_geometry_q_group_window(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 1}}},
        current_geometry_fit_var_names=["gamma"],
        on_toggle=lambda key, row_var: (key, row_var),
    )

    assert ok is True
    assert calls["kwargs"]["entries"] == q_group_state.cached_entries
    assert calls["kwargs"]["format_line"] is geometry_q_group_manager.format_geometry_q_group_line
    assert "ready" in calls["kwargs"]["status_text"]


def test_geometry_q_group_manager_toggle_and_bulk_enable_update_preview_state() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            q_group_state,
            key1,
            _FakeVar(False),
        )
        == "Disabled"
    )
    assert q_group_state.disabled_qz_sections == {("primary", 1, 0)}

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            q_group_state,
            key1,
            _FakeVar(True),
        )
        == "Enabled"
    )
    assert q_group_state.disabled_qz_sections == set()

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=False,
    )
    assert (action, count) == ("Disabled", 2)
    assert q_group_state.disabled_qr_sets == {("primary", 1), ("secondary", 2)}

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=True,
    )
    assert (action, count) == ("Enabled", 2)
    assert q_group_state.disabled_qr_sets == set()


def test_geometry_q_group_manager_save_load_helpers_round_trip() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("secondary", 2, 1)},
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ],
    )

    export_rows = geometry_q_group_manager.build_geometry_q_group_export_rows(
        preview_state=preview_state,
        q_group_state=q_group_state,
    )
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        export_rows,
        q_group_state=q_group_state,
        saved_at="2026-03-26T12:00:00",
    )
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state(payload)

    assert error is None
    assert payload["included_count"] == 1
    assert export_rows[0]["included"] is True
    assert export_rows[1]["included"] is False
    assert export_rows[0]["m_index"] == 1
    assert export_rows[0]["l_index"] == 0
    assert export_rows[0]["gz_index"] == 0
    assert "Qr=" in export_rows[0]["display_label"]
    assert "m=" in export_rows[0]["display_label"]
    assert "L=" in export_rows[0]["display_label"]
    assert "Gz=" not in export_rows[0]["display_label"]
    assert saved_state == {
        "saved_rows": {key1: True, key2: False},
        "disabled_qr_sets": [],
        "disabled_qz_sections": [("secondary", 2, 1)],
        "explicit_masks_present": True,
    }

    target_q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=state.GeometryPreviewState(),
        q_group_state=target_q_group_state,
        saved_state=saved_state,
    )

    assert error is None
    assert summary == {
        "matched_total": 2,
        "included_total": 1,
        "current_only": 1,
        "saved_only": 0,
    }
    assert target_q_group_state.disabled_qz_sections == {("secondary", 2, 1)}
    assert target_q_group_state.disabled_qr_sets == set()


def test_geometry_q_group_manager_load_helpers_report_validation_errors() -> None:
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state("bad")
    assert saved_state is None
    assert error == "Invalid Qr/Qz peak list file: expected a JSON object."

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        saved_state={},
    )
    assert summary is None
    assert error == "Loaded Qr/Qz peak list does not contain any valid rows."


def test_geometry_q_group_manager_checkbox_side_effects_update_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(key1, peak_count=1, total_intensity=1.0)]
    )
    events = []

    changed = geometry_q_group_manager.apply_geometry_q_group_checkbox_change_with_side_effects(
        preview_state=preview_state,
        q_group_state=q_group_state,
        group_key=key1,
        row_var=_FakeVar(False),
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        update_geometry_q_group_window_status=lambda: events.append("status"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("refresh_live"),
        set_status_text=lambda text: events.append(text),
    )

    assert changed is True
    assert q_group_state.disabled_qz_sections == {("primary", 1, 0)}
    assert events == [
        "invalidate",
        "label",
        "status",
        "Disabled one Qr/Qz group.",
    ]


def test_geometry_q_group_checkbox_side_effects_warm_caked_cache_without_view_toggle() -> None:
    key1 = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(key1, peak_count=1, total_intensity=1.0)]
    )
    view_mode = {"value": "detector"}
    events = []

    changed = geometry_q_group_manager.apply_geometry_q_group_checkbox_change_with_side_effects(
        preview_state=preview_state,
        q_group_state=q_group_state,
        group_key=key1,
        row_var=_FakeVar(False),
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        update_geometry_q_group_window_status=lambda: events.append("status"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("refresh_live"),
        set_status_text=lambda text: events.append(text),
        warm_detector_mode_qr_caked_cache=lambda: events.append(
            f"warm:{view_mode['value']}"
        )
        or True,
    )

    assert changed is True
    assert view_mode["value"] == "detector"
    assert events == [
        "invalidate",
        "label",
        "status",
        "Disabled one Qr/Qz group.",
        "warm:detector",
    ]


def test_geometry_q_group_manager_bulk_enable_side_effects_cover_empty_and_live_refresh() -> None:
    preview_state = state.GeometryPreviewState()
    empty_state = state.GeometryQGroupState()
    empty_messages = []

    changed = geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
        preview_state=preview_state,
        q_group_state=empty_state,
        enabled=False,
        invalidate_geometry_manual_pick_cache=lambda: empty_messages.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: empty_messages.append("label"),
        refresh_geometry_q_group_window=lambda: empty_messages.append("refresh"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: empty_messages.append("live"),
        set_status_text=lambda text: empty_messages.append(text),
    )

    assert changed is False
    assert empty_messages == [
        'No listed Qr/Qz groups are available. Press "Update Listed Peaks" first.'
    ]

    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )
    events = []

    changed = geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
        preview_state=preview_state,
        q_group_state=q_group_state,
        enabled=False,
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        refresh_geometry_q_group_window=lambda: events.append("refresh"),
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: events.append("live"),
        set_status_text=lambda text: events.append(text),
    )

    assert changed is True
    assert q_group_state.disabled_qr_sets == {("primary", 1), ("secondary", 2)}
    assert preview_state.excluded_q_groups == set()
    assert events == ["invalidate", "label", "refresh", "live"]


def test_geometry_q_group_bulk_side_effects_warm_caked_cache() -> None:
    key1 = ("q_group", "primary", 1, 0)
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(key1, peak_count=2, total_intensity=10.0)]
    )
    events = []

    changed = geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
        preview_state=state.GeometryPreviewState(),
        q_group_state=q_group_state,
        enabled=False,
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        refresh_geometry_q_group_window=lambda: events.append("refresh"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("live"),
        set_status_text=lambda text: events.append(text),
        warm_detector_mode_qr_caked_cache=lambda: events.append("warm") or True,
    )

    assert changed is True
    assert events == [
        "invalidate",
        "label",
        "refresh",
        "Disabled 1 Qr/Qz groups.",
        "warm",
    ]


def test_geometry_q_group_manager_request_update_side_effects_marks_refresh() -> None:
    q_group_state = state.GeometryQGroupState()
    events = []

    geometry_q_group_manager.request_geometry_q_group_window_update_with_side_effects(
        q_group_state=q_group_state,
        clear_last_simulation_signature=lambda: events.append("clear_signature"),
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        set_status_text=lambda text: events.append(text),
        schedule_update=lambda: events.append("schedule"),
    )

    assert q_group_state.refresh_requested is True
    assert events == [
        "clear_signature",
        "invalidate",
        "Updating listed Qr/Qz peaks from the current simulation...",
        "schedule",
    ]


def test_geometry_q_group_manager_snapshot_replace_side_effects_trim_exclusions() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key1, ("q_group", "stale", 9, 9)})
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("primary", 1, 0), ("stale", 9, 9)}
    )
    events = []

    entries = geometry_q_group_manager.replace_geometry_q_group_entries_snapshot_with_side_effects(
        preview_state=preview_state,
        q_group_state=q_group_state,
        entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
    )

    assert [entry["key"] for entry in entries] == [key1, key2]
    assert q_group_state.cached_entries == entries
    assert q_group_state.disabled_qz_sections == {("primary", 1, 0)}
    assert preview_state.excluded_q_groups == {
        key1,
        ("q_group", "stale", 9, 9),
    }
    assert events == ["invalidate", "label"]


def test_geometry_q_group_manager_preview_exclusion_toggle_and_clear_helpers() -> None:
    key1 = ("pair", 1)
    q_group_key = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState(
        excluded_q_groups={q_group_key},
    )
    preview_state.overlay.pairs = [
        {
            "id": 1,
            "sim_x": 10.0,
            "sim_y": 10.0,
            "x": 20.0,
            "y": 10.0,
        }
    ]
    toggle_events = []

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )
    missed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=100.0,
        row=100.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("unexpected"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )

    assert changed is True
    assert changed_again is True
    assert missed is False
    assert key1 not in preview_state.excluded_keys
    assert toggle_events == [
        "render",
        "Excluded live preview peak HKL=(1, 0, 0) from geometry fit.",
        "render",
        "Included live preview peak HKL=(1, 0, 0) from geometry fit.",
        "No preview pair within 5px to toggle.",
    ]

    clear_events = []
    preview_state.excluded_keys = {key1}
    geometry_q_group_manager.clear_live_geometry_preview_exclusions_with_side_effects(
        preview_state=preview_state,
        invalidate_geometry_manual_pick_cache=lambda: clear_events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: clear_events.append("label"),
        refresh_geometry_q_group_window=lambda: clear_events.append("refresh"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: clear_events.append("live"),
        set_status_text=lambda text: clear_events.append(text),
    )

    assert preview_state.excluded_keys == set()
    assert preview_state.excluded_q_groups == {q_group_key}
    assert clear_events == [
        "invalidate",
        "label",
        "refresh",
        "Reset live preview pair exclusions.",
    ]

    callback_only_entry = {
        "id": 1,
        "sim_x": 10.0,
        "sim_y": 10.0,
        "x": 20.0,
        "y": 10.0,
    }
    callback_only_key = lambda entry: ("pair", int(entry["id"]))
    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = [dict(callback_only_entry)]

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=callback_only_key,
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: None,
        max_distance_px=5.0,
        set_status_text=lambda _text: None,
    )
    callback_only_filtered, callback_only_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(callback_only_entry)],
            live_preview_match_key=callback_only_key,
        )
    )
    callback_only_excluded_keys = set(preview_state.excluded_keys)
    callback_only_is_excluded = geometry_q_group_manager.live_geometry_preview_match_is_excluded(
        preview_state,
        callback_only_entry,
        live_preview_match_key=callback_only_key,
    )
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=callback_only_key,
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: None,
        max_distance_px=5.0,
        set_status_text=lambda _text: None,
    )

    assert changed is True
    assert callback_only_excluded_keys == {("pair", 1)}
    assert callback_only_is_excluded is True
    assert callback_only_filtered == []
    assert callback_only_excluded == 1
    assert changed_again is True
    assert preview_state.excluded_keys == set()

    callback_hkl_entry_1 = {
        "id": 1,
        "hkl": (4, 0, 1),
    }
    callback_hkl_entry_2 = {
        "id": 2,
        "hkl": (4, 0, 1),
    }
    preview_state = state.GeometryPreviewState(excluded_keys={("pair", 1)})
    preview_state._live_geometry_preview_exclusion_groups = [
        geometry_q_group_manager._live_geometry_preview_exclusion_descriptor(
            callback_hkl_entry_1,
            callback_key=callback_only_key(callback_hkl_entry_1),
        )
    ]

    assert (
        geometry_q_group_manager.live_geometry_preview_match_is_excluded(
            preview_state,
            callback_hkl_entry_1,
            live_preview_match_key=callback_only_key,
        )
        is True
    )
    assert (
        geometry_q_group_manager.live_geometry_preview_match_is_excluded(
            preview_state,
            callback_hkl_entry_2,
            live_preview_match_key=callback_only_key,
        )
        is False
    )
    callback_hkl_filtered, callback_hkl_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(callback_hkl_entry_1), dict(callback_hkl_entry_2)],
            existing_pairs=[dict(callback_hkl_entry_1)],
            live_preview_match_key=callback_only_key,
        )
    )

    assert callback_hkl_filtered == [callback_hkl_entry_2]
    assert callback_hkl_excluded == 1

    alias_entry = {
        "hkl": (4, 0, 1),
        "sim_x": 10.0,
        "sim_y": 10.0,
        "x": 20.0,
        "y": 10.0,
        "source_label": "secondary",
        "source_table_index": 4,
        "source_row_index": 6,
        "source_peak_index": 99,
    }
    legacy_alias_key = geometry_q_group_manager.live_geometry_preview_match_key(
        {
            "hkl": alias_entry["hkl"],
            "sim_x": alias_entry["sim_x"],
            "sim_y": alias_entry["sim_y"],
        }
    )
    preview_state = state.GeometryPreviewState(excluded_keys={legacy_alias_key})
    preview_state.overlay.pairs = [dict(alias_entry)]
    alias_events = []
    alias_canonical_key = ("peak", "secondary", 4, 6, 4, 0, 1)

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=geometry_q_group_manager.live_geometry_preview_match_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: alias_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: alias_events.append(text),
    )
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=geometry_q_group_manager.live_geometry_preview_match_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: alias_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: alias_events.append(text),
    )

    assert changed is True
    assert changed_again is True
    assert preview_state.excluded_keys == {alias_canonical_key}
    assert alias_events == [
        "render",
        "Included live preview peak HKL=(4, 0, 1) from geometry fit.",
        "render",
        "Excluded live preview peak HKL=(4, 0, 1) from geometry fit.",
    ]

    custom_key = ("pair", 42)
    internal_key = ("peak", "secondary", 4, 6, 4, 0, 1)
    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = [dict(alias_entry)]
    custom_events = []
    poorer_entry = {
        "hkl": alias_entry["hkl"],
        "sim_x": alias_entry["sim_x"],
        "sim_y": alias_entry["sim_y"],
        "x": alias_entry["x"],
        "y": alias_entry["y"],
    }

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda _entry: custom_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: custom_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: custom_events.append(text),
    )
    filtered_custom, excluded_custom = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(alias_entry)],
        )
    )
    poorer_filtered, poorer_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(poorer_entry)],
        )
    )

    assert changed is True
    assert preview_state.excluded_keys == {custom_key, internal_key}
    assert geometry_q_group_manager.live_geometry_preview_match_is_excluded(
        preview_state,
        alias_entry,
    )
    assert filtered_custom == []
    assert excluded_custom == 1
    assert poorer_filtered == []
    assert poorer_excluded == 1
    preview_state.overlay.pairs = [dict(poorer_entry)]
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda _entry: custom_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: custom_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: custom_events.append(text),
    )
    assert changed_again is True
    assert preview_state.excluded_keys == set()
    rich_filtered, rich_excluded = geometry_q_group_manager.filter_live_geometry_preview_matches(
        preview_state,
        [dict(alias_entry)],
    )
    assert rich_filtered == [alias_entry]
    assert rich_excluded == 0
    assert custom_events == [
        "render",
        "Excluded live preview peak HKL=(4, 0, 1) from geometry fit.",
        "render",
        "Included live preview peak HKL=(4, 0, 1) from geometry fit.",
    ]

    primary_entry = {
        "hkl": (4, 0, 1),
        "sim_x": 10.0,
        "sim_y": 10.0,
        "x": 20.0,
        "y": 10.0,
        "source_label": "primary",
        "source_table_index": 1,
        "source_row_index": 6,
    }
    secondary_entry = {
        "hkl": (4, 0, 1),
        "sim_x": 10.0,
        "sim_y": 10.0,
        "x": 20.0,
        "y": 10.0,
        "source_label": "secondary",
        "source_table_index": 2,
        "source_row_index": 6,
    }
    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = [dict(primary_entry)]

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=geometry_q_group_manager.live_geometry_preview_match_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: None,
        max_distance_px=5.0,
        set_status_text=lambda _text: None,
    )
    assert changed is True
    assert preview_state.excluded_keys == {
        geometry_q_group_manager.live_geometry_preview_match_key(primary_entry)
    }

    preview_state.overlay.pairs = [dict(secondary_entry)]
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=geometry_q_group_manager.live_geometry_preview_match_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: None,
        max_distance_px=5.0,
        set_status_text=lambda _text: None,
    )
    primary_filtered, primary_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(primary_entry)],
        )
    )
    secondary_filtered, secondary_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(secondary_entry)],
        )
    )

    assert changed_again is True
    assert preview_state.excluded_keys == {
        geometry_q_group_manager.live_geometry_preview_match_key(primary_entry),
        geometry_q_group_manager.live_geometry_preview_match_key(secondary_entry),
    }
    assert primary_filtered == []
    assert primary_excluded == 1
    assert secondary_filtered == []
    assert secondary_excluded == 1

    preview_state.overlay.pairs = [dict(primary_entry)]
    changed_third = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=geometry_q_group_manager.live_geometry_preview_match_key,
        live_preview_match_hkl=geometry_q_group_manager.live_geometry_preview_match_hkl,
        render_live_geometry_preview_state=lambda: None,
        max_distance_px=5.0,
        set_status_text=lambda _text: None,
    )
    primary_filtered, primary_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(primary_entry)],
        )
    )
    secondary_filtered, secondary_excluded = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [dict(secondary_entry)],
        )
    )

    assert changed_third is True
    assert preview_state.excluded_keys == {
        geometry_q_group_manager.live_geometry_preview_match_key(secondary_entry)
    }
    assert primary_filtered == [primary_entry]
    assert primary_excluded == 0
    assert secondary_filtered == []
    assert secondary_excluded == 1


def test_geometry_q_group_manager_runtime_preview_exclude_mode_helper_updates_hkl_and_status(
    monkeypatch,
) -> None:
    preview_state = state.GeometryPreviewState()
    events = []

    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "set_geometry_preview_exclude_mode",
        lambda state_value, enabled: events.append(("mode", state_value, enabled)) or True,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("refresh"),
        set_hkl_pick_mode=lambda enabled: events.append(("hkl", enabled)),
        set_status_text=lambda text: events.append(("status", text)),
    )

    changed = geometry_q_group_manager.set_runtime_geometry_preview_exclude_mode(
        bindings,
        True,
        message="armed",
    )

    assert changed is True
    assert events == [
        ("mode", preview_state, True),
        ("hkl", False),
        "label",
        ("status", "armed"),
    ]


def test_geometry_q_group_manager_runtime_snapshot_capture_refreshes_open_window(
    monkeypatch,
) -> None:
    key1 = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("primary", 1, 0), ("stale", 9, 9)}
    )
    events = []

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "geometry_q_group_window_open",
        lambda view_state: True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "refresh_runtime_geometry_q_group_window",
        lambda bindings: events.append("refresh") or True,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("live"),
        build_entries_snapshot=lambda: [
            _entry(key1, peak_count=2, total_intensity=10.0),
        ],
    )

    entries = geometry_q_group_manager.capture_runtime_geometry_q_group_entries_snapshot(bindings)

    assert [entry["key"] for entry in entries] == [key1]
    assert q_group_state.cached_entries == entries
    assert q_group_state.disabled_qz_sections == {("primary", 1, 0)}
    assert events == ["invalidate", "label", "refresh"]


def test_geometry_q_group_manager_snapshot_capture_preserves_imported_rows_during_empty_refresh(
    monkeypatch,
) -> None:
    key1 = ("q_group", "primary", 1, 0)
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(key1, peak_count=2, total_intensity=10.0)]
    )
    q_group_state.restored_q_group_rows_pending_live_refresh = True
    events = []

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "geometry_q_group_window_open",
        lambda view_state: True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "refresh_runtime_geometry_q_group_window",
        lambda bindings: events.append("refresh") or True,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=state.GeometryPreviewState(),
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("live"),
        build_entries_snapshot=lambda: [],
        has_cached_hit_tables=lambda: False,
    )

    entries = geometry_q_group_manager.capture_runtime_geometry_q_group_entries_snapshot(bindings)

    assert [entry["key"] for entry in entries] == [key1]
    assert [entry["key"] for entry in q_group_state.cached_entries] == [key1]
    assert q_group_state.restored_q_group_rows_pending_live_refresh is True
    assert events == ["refresh"]


def test_geometry_q_group_manager_preview_exclusion_open_reports_status(
    monkeypatch,
) -> None:
    events = []

    def _open_window(*, root, bindings_factory):
        events.append(("open", root, bindings_factory()))
        return False

    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_window",
        _open_window,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        set_status_text=lambda text: events.append(("status", text)),
    )

    opened = geometry_q_group_manager.open_runtime_geometry_q_group_preview_exclusion_window(
        root="root-window",
        bindings_factory=lambda: bindings,
    )

    assert opened is False
    assert events[0] == ("open", "root-window", bindings)
    assert events[1] == (
        "status",
        "Opened the Qr/Qz group selector. Unchecked rows are skipped during manual picking and geometry fitting.",
    )


def test_geometry_q_group_manager_live_preview_toggle_helper_covers_disabled_scheduled_and_refreshed_paths() -> (
    None
):
    disabled_events = []

    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=False,
        disable_preview_exclude_mode=lambda: disabled_events.append("disable"),
        clear_geometry_preview_artists=lambda: disabled_events.append("clear"),
        open_geometry_q_group_window=lambda: disabled_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: disabled_events.append("schedule"),
        refresh_live_geometry_preview=lambda: disabled_events.append("refresh") or True,
        set_status_text=lambda text: disabled_events.append(text),
    )

    assert refreshed is False
    assert disabled_events == [
        "disable",
        "clear",
        "Live auto-match preview disabled.",
    ]

    scheduled_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: scheduled_events.append("disable"),
        clear_geometry_preview_artists=lambda: scheduled_events.append("clear"),
        open_geometry_q_group_window=lambda: scheduled_events.append("open"),
        update_running=True,
        has_cached_hit_tables=False,
        schedule_update=lambda: scheduled_events.append("schedule"),
        refresh_live_geometry_preview=lambda: scheduled_events.append("refresh") or True,
        set_status_text=lambda text: scheduled_events.append(text),
    )

    assert refreshed is True
    assert scheduled_events == ["open", "schedule"]

    failed_refresh_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: failed_refresh_events.append("disable"),
        clear_geometry_preview_artists=lambda: failed_refresh_events.append("clear"),
        open_geometry_q_group_window=lambda: failed_refresh_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: failed_refresh_events.append("schedule"),
        refresh_live_geometry_preview=lambda: failed_refresh_events.append("refresh") or False,
        set_status_text=lambda text: failed_refresh_events.append(text),
    )

    assert refreshed is False
    assert failed_refresh_events == ["open", "refresh", "schedule"]

    success_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: success_events.append("disable"),
        clear_geometry_preview_artists=lambda: success_events.append("clear"),
        open_geometry_q_group_window=lambda: success_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: success_events.append("schedule"),
        refresh_live_geometry_preview=lambda: success_events.append("refresh") or True,
        set_status_text=lambda text: success_events.append(text),
    )

    assert refreshed is True
    assert success_events == ["open", "refresh"]


def test_geometry_q_group_manager_save_dialog_workflow_writes_payload_and_reports_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        disabled_qz_sections={("secondary", 2, 1)},
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ],
    )
    captured = {}
    messages = []

    def _asksaveasfilename(**kwargs):
        captured["dialog"] = kwargs
        return "C:/tmp/groups.json"

    saved = geometry_q_group_manager.save_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        asksaveasfilename=_asksaveasfilename,
        set_status_text=lambda text: messages.append(text),
        save_payload=lambda path, payload: captured.update({"path": path, "payload": payload}),
        now=lambda: datetime(2026, 3, 26, 12, 34, 56),
    )

    assert saved is True
    assert captured["dialog"]["initialdir"] == "C:/dialogs"
    assert captured["dialog"]["initialfile"] == "geometry_q_groups_20260326_123456.json"
    assert captured["path"] == "C:/tmp/groups.json"
    assert captured["payload"]["saved_at"] == "2026-03-26T12:34:56"
    assert captured["payload"]["included_count"] == 1
    assert messages == ["Saved 2 Qr/Qz groups to C:/tmp/groups.json"]


def test_geometry_q_group_manager_load_dialog_workflow_applies_state_and_refreshes() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )
    events = []
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        [
            {"key": ["q_group", "primary", 1, 0], "included": True},
            {"key": ["q_group", "secondary", 2, 1], "included": False},
        ],
        q_group_state=state.GeometryQGroupState(disabled_qz_sections={("secondary", 2, 1)}),
        saved_at="2026-03-26T12:00:00",
    )

    def _askopenfilename(**kwargs):
        events.append(("dialog", kwargs))
        return "C:/tmp/selector.json"

    loaded = geometry_q_group_manager.load_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        askopenfilename=_askopenfilename,
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        refresh_geometry_q_group_window=lambda: events.append("refresh"),
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: events.append("live"),
        set_status_text=lambda text: events.append(text),
        load_payload=lambda path: events.append(("load", path)) or payload,
    )

    assert loaded is True
    assert q_group_state.disabled_qz_sections == {("secondary", 2, 1)}
    assert events[0][0] == "dialog"
    assert events[0][1]["initialdir"] == "C:/dialogs"
    assert events[1] == ("load", "C:/tmp/selector.json")
    assert events[2:6] == ["invalidate", "label", "refresh", "live"]
    assert (
        events[6]
        == "Loaded Qr/Qz peak list from selector.json: matched 2, enabled 1, current-only unmatched 1, saved-only missing 0."
    )


def test_geometry_q_group_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"status": 0, "schedule": 0, "dir": 0}

    monkeypatch.setattr(
        geometry_q_group_manager,
        "GeometryQGroupRuntimeBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    def build_schedule():
        counters["schedule"] += 1
        idx = counters["schedule"]
        return lambda: f"schedule-{idx}"

    def build_dialog_dir():
        counters["dir"] += 1
        return f"C:/dialogs/{counters['dir']}"

    factory = geometry_q_group_manager.make_runtime_geometry_q_group_bindings_factory(
        view_state="view-state",
        preview_state="preview-state",
        q_group_state="q-group-state",
        fit_config={"geometry": {}},
        current_geometry_fit_var_names_factory=lambda: ["gamma"],
        build_entries_snapshot=lambda: [{"key": ("q_group", "primary", 1, 0)}],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        refresh_live_geometry_preview_quiet=lambda: None,
        clear_last_simulation_signature=lambda: None,
        schedule_update_factory=build_schedule,
        set_status_text_factory=build_status,
        file_dialog_dir_factory=build_dialog_dir,
        asksaveasfilename=lambda **kwargs: "save.json",
        askopenfilename=lambda **kwargs: "load.json",
    )

    assert factory()["view_state"] == "view-state"
    assert factory()["view_state"] == "view-state"
    assert calls[0]["preview_state"] == "preview-state"
    assert calls[0]["q_group_state"] == "q-group-state"
    assert calls[0]["fit_config"] == {"geometry": {}}
    assert callable(calls[0]["set_status_text"])
    assert callable(calls[0]["schedule_update"])
    assert callable(calls[0]["build_entries_snapshot"])
    assert calls[0]["build_entries_snapshot"]() == [{"key": ("q_group", "primary", 1, 0)}]
    assert calls[0]["file_dialog_dir"] == "C:/dialogs/1"
    assert calls[1]["file_dialog_dir"] == "C:/dialogs/2"
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]
    assert calls[0]["schedule_update"] is not calls[1]["schedule_update"]


def test_geometry_q_group_runtime_callback_bundle_delegates_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        geometry_q_group_manager,
        "update_runtime_geometry_q_group_window_status",
        lambda bindings, entries=None: calls.append(("status", bindings, entries)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "refresh_runtime_geometry_q_group_window",
        lambda bindings: calls.append(("refresh", bindings)) or True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "on_runtime_geometry_q_group_checkbox_changed",
        lambda bindings, group_key, row_var: (
            calls.append(("toggle", bindings, group_key, row_var)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "set_all_geometry_q_groups_enabled_runtime",
        lambda bindings, *, enabled: (
            calls.append(("bulk", bindings, enabled)),
            enabled,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "request_runtime_geometry_q_group_window_update",
        lambda bindings: calls.append(("update", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "save_geometry_q_group_selection_runtime",
        lambda bindings: calls.append(("save", bindings)) or True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "load_geometry_q_group_selection_runtime",
        lambda bindings: calls.append(("load", bindings)) or False,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "close_runtime_geometry_q_group_window",
        lambda bindings: calls.append(("close", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_window",
        lambda *, root, bindings_factory: (
            calls.append(("open", root, bindings_factory())),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_preview_exclusion_window",
        lambda *, root, bindings_factory: (
            calls.append(("open-preview", root, bindings_factory())),
            False,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "set_runtime_geometry_preview_exclude_mode",
        lambda bindings, enabled, *, message=None: (
            calls.append(("mode", bindings, enabled, message)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "clear_runtime_live_geometry_preview_exclusions",
        lambda bindings: calls.append(("clear-preview", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "toggle_runtime_live_geometry_preview_exclusion_at",
        lambda bindings, col, row: (
            calls.append(("toggle-preview", bindings, col, row)),
            False,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "toggle_runtime_live_geometry_preview",
        lambda bindings, *, root, bindings_factory: (
            calls.append(("live-toggle", bindings, root)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "runtime_live_geometry_preview_enabled",
        lambda bindings: calls.append(("preview-enabled", bindings)) or False,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "render_runtime_live_geometry_preview_state",
        lambda bindings, *, update_status=True: (
            calls.append(("render-preview", bindings, update_status)),
            True,
        )[-1],
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = geometry_q_group_manager.make_runtime_geometry_q_group_callbacks(
        root="root-window",
        bindings_factory=build_bindings,
    )

    entries = [{"key": ("q_group", "primary", 1, 0)}]
    row_var = _FakeVar(True)
    group_key = ("q_group", "primary", 1, 0)

    callbacks.update_window_status(entries)
    assert callbacks.refresh_window() is True
    assert callbacks.on_toggle(group_key, row_var) is True
    assert callbacks.include_all() is True
    assert callbacks.exclude_all() is False
    callbacks.update_listed_peaks()
    assert callbacks.save_selection() is True
    assert callbacks.load_selection() is False
    callbacks.close_window()
    assert callbacks.open_window() is True
    assert callbacks.open_preview_exclusion_window() is False
    assert callbacks.set_preview_exclude_mode(True, message="armed") is True
    callbacks.clear_preview_exclusions()
    assert callbacks.toggle_preview_exclusion_at(1.5, 2.5) is False
    assert callbacks.toggle_live_preview() is True
    assert callbacks.live_preview_enabled() is False
    assert callbacks.render_live_preview_state(update_status=False) is True

    assert calls == [
        ("status", "bindings-1", entries),
        ("refresh", "bindings-2"),
        ("toggle", "bindings-3", group_key, row_var),
        ("bulk", "bindings-4", True),
        ("bulk", "bindings-5", False),
        ("update", "bindings-6"),
        ("save", "bindings-7"),
        ("load", "bindings-8"),
        ("close", "bindings-9"),
        ("open", "root-window", "bindings-10"),
        ("open-preview", "root-window", "bindings-11"),
        ("mode", "bindings-12", True, "armed"),
        ("clear-preview", "bindings-13"),
        ("toggle-preview", "bindings-14", 1.5, 2.5),
        ("live-toggle", "bindings-15", "root-window"),
        ("preview-enabled", "bindings-16"),
        ("render-preview", "bindings-17", False),
    ]
