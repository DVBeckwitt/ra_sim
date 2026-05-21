from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import runtime_primary_cache
from ra_sim.gui._runtime import primary_cache_helpers
from ra_sim.simulation.diffraction import intersection_cache_to_hit_tables
from ra_sim.simulation import intersection_cache_schema as cache_schema
from ra_sim.simulation.engine import simulate
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


def _build_request(image_size: int = 16) -> SimulationRequest:
    return SimulationRequest(
        miller=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        geometry=DetectorGeometry(
            image_size=image_size,
            av=4.0,
            cv=7.0,
            lambda_angstrom=1.54,
            distance_m=0.1,
            gamma_deg=0.0,
            Gamma_deg=0.0,
            chi_deg=0.0,
            psi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            center=np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64),
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ),
        beam=BeamSamples(
            beam_x_array=np.zeros(1, dtype=np.float64),
            beam_y_array=np.zeros(1, dtype=np.float64),
            theta_array=np.zeros(1, dtype=np.float64),
            phi_array=np.zeros(1, dtype=np.float64),
            wavelength_array=np.ones(1, dtype=np.float64),
        ),
        mosaic=MosaicParams(
            sigma_mosaic_deg=0.5,
            gamma_mosaic_deg=0.4,
            eta=0.2,
        ),
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=np.zeros((image_size, image_size), dtype=np.float64),
        collect_hit_tables=True,
    )


def test_build_primary_subset_payload_for_miller_uses_raw_row_indices() -> None:
    payload = runtime_primary_cache.build_primary_subset_payload(
        source_mode="miller",
        all_primary_qr={},
        all_miller=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        all_intensities=np.array([10.0, 5.0, 8.0], dtype=np.float64),
        requested_keys=[2, 0],
    )

    assert payload["primary_contribution_keys"] == [2, 0]
    assert np.array_equal(
        payload["primary_data"],
        np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(
        payload["primary_intensities"],
        np.array([8.0, 10.0], dtype=np.float64),
    )


def test_build_primary_subset_payload_for_qr_preserves_sorted_m_local_order() -> None:
    payload = runtime_primary_cache.build_primary_subset_payload(
        source_mode="qr",
        all_primary_qr={
            3: {
                "hk": (1, 1),
                "L": np.array([0.0, 1.0], dtype=np.float64),
                "I": np.array([8.0, 4.0], dtype=np.float64),
                "deg": 1,
            },
            1: {
                "hk": (1, 0),
                "L": np.array([0.0, 2.0, 3.0], dtype=np.float64),
                "I": np.array([10.0, 3.0, 1.0], dtype=np.float64),
                "deg": 1,
            },
        },
        all_miller=np.empty((0, 3), dtype=np.float64),
        all_intensities=np.empty((0,), dtype=np.float64),
        requested_keys=[(3, 1), (1, 2), (1, 0)],
    )

    assert payload["primary_contribution_keys"] == [(1, 2), (1, 0), (3, 1)]
    assert list(payload["primary_data"]) == [1, 3]
    assert np.array_equal(
        payload["primary_data"][1]["L"],
        np.array([3.0, 0.0], dtype=np.float64),
    )
    assert np.array_equal(
        payload["primary_data"][3]["I"],
        np.array([4.0], dtype=np.float64),
    )


def test_copy_intersection_cache_tables_preserves_supported_schema_widths() -> None:
    copied = primary_cache_helpers.copy_intersection_cache_tables(
        [
            np.empty((0, 14), dtype=np.float64),
            np.empty((0, 16), dtype=np.float64),
            np.empty((0, 17), dtype=np.float64),
            np.empty((0, 19), dtype=np.float64),
            np.ones((1, 7), dtype=np.float64),
            object(),
        ]
    )

    assert [table.shape for table in copied[:4]] == [
        (0, 14),
        (0, 16),
        (0, 17),
        (0, 19),
    ]
    assert copied[4].shape == (0, 17)
    assert copied[5].shape == (0, 17)


def test_resolve_incremental_sf_prune_action_only_requests_missing_added_keys() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("base", 1),
        cached_signature=("base", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 1, 2, 3],
        previous_active_keys=[0, 3],
        primary_hit_table_cache={
            0: np.empty((0, 7), dtype=np.float64),
            2: np.empty((0, 7), dtype=np.float64),
            3: np.empty((0, 7), dtype=np.float64),
        },
    )

    assert action.mode == "fill"
    assert action.added_keys == (1, 2)
    assert action.removed_keys == ()
    assert action.missing_keys == (1,)


def test_prune_reuse_does_not_request_full_simulation_when_active_keys_cached() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("physics", 1),
        cached_signature=("physics", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 2],
        previous_active_keys=[0, 1, 2],
        primary_hit_table_cache={
            0: np.empty((1, 7), dtype=np.float64),
            1: np.empty((1, 7), dtype=np.float64),
            2: np.empty((1, 7), dtype=np.float64),
        },
    )

    assert action.mode == "reuse"
    assert action.missing_keys == ()
    assert action.reason == "all_keys_cached"


def test_prune_fill_requests_only_missing_contribution_keys() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("physics", 1),
        cached_signature=("physics", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 1, 2, 3],
        previous_active_keys=[0],
        primary_hit_table_cache={
            0: np.empty((1, 7), dtype=np.float64),
            2: np.empty((1, 7), dtype=np.float64),
        },
    )

    assert action.mode == "fill"
    assert action.added_keys == (1, 2, 3)
    assert action.missing_keys == (1, 3)
    assert action.reason == "fill_missing_keys"


def test_prune_cache_incompatible_physics_returns_full() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("physics", 2),
        cached_signature=("physics", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 1],
        previous_active_keys=[0],
        primary_hit_table_cache={0: np.empty((1, 7), dtype=np.float64)},
    )

    assert action.mode == "full"
    assert action.reason == "cache_signature_changed"
    assert action.missing_keys == ()


def test_resolve_incremental_sf_prune_action_reuses_cache_when_bias_increases() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("base", 1),
        cached_signature=("base", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 3],
        previous_active_keys=[0, 1, 2, 3],
        primary_hit_table_cache={
            0: np.empty((0, 7), dtype=np.float64),
            1: np.empty((0, 7), dtype=np.float64),
            2: np.empty((0, 7), dtype=np.float64),
            3: np.empty((0, 7), dtype=np.float64),
        },
    )

    assert action.mode == "reuse"
    assert action.added_keys == ()
    assert action.removed_keys == (1, 2)
    assert action.missing_keys == ()


def test_resolve_incremental_sf_prune_action_toggle_reuses_cached_keys() -> None:
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        cache_signature=("base", 1),
        cached_signature=("base", 1),
        source_mode="miller",
        cached_source_mode="miller",
        active_keys=[0, 3],
        previous_active_keys=[0, 1, 2, 3],
        primary_hit_table_cache={
            0: np.empty((0, 7), dtype=np.float64),
            1: np.empty((0, 7), dtype=np.float64),
            2: np.empty((0, 7), dtype=np.float64),
            3: np.empty((0, 7), dtype=np.float64),
        },
    )

    assert action.mode == "reuse"
    assert action.missing_keys == ()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {
                "cached_signature": ("other", 1),
            },
            "cache_signature_changed",
        ),
        (
            {
                "cached_source_mode": "qr",
            },
            "source_mode_changed",
        ),
        (
            {
                "primary_hit_table_cache": {},
            },
            "missing_primary_cache",
        ),
        (
            {
                "active_job": {
                    "run_primary": True,
                    "primary_contribution_cache_signature": ("wrong", 1),
                    "primary_source_mode": "miller",
                },
            },
            "worker_incompatible",
        ),
    ],
)
def test_resolve_incremental_sf_prune_action_falls_back_for_incompatible_state(
    kwargs: dict[str, object],
    reason: str,
) -> None:
    base_kwargs = {
        "cache_signature": ("base", 1),
        "cached_signature": ("base", 1),
        "source_mode": "miller",
        "cached_source_mode": "miller",
        "active_keys": [0, 1],
        "previous_active_keys": [0],
        "primary_hit_table_cache": {0: np.empty((0, 7), dtype=np.float64)},
    }
    base_kwargs.update(kwargs)
    action = runtime_primary_cache.resolve_incremental_sf_prune_action(
        **base_kwargs,
    )

    assert action.mode == "full"
    assert action.reason == reason


def test_rasterize_hit_tables_to_image_matches_simulation_image() -> None:
    request = _build_request()
    result = simulate(request)

    image = runtime_primary_cache.rasterize_hit_tables_to_image(
        result.hit_tables,
        image_size=request.geometry.image_size,
    )

    assert np.allclose(image, result.image)


def test_rasterize_hit_tables_to_image_handles_edges_and_duplicate_hits() -> None:
    hit_tables = [
        np.array(
            [
                [10.0, 1.25, 2.5, 0.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [8.0, -0.25, 0.0, 0.0, 0.0, 0.0, 1.0],
                [99.0, np.nan, 1.0, 0.0, 0.0, 0.0, 1.0],
                [99.0, 1.0, np.inf, 0.0, 0.0, 0.0, 1.0],
                [99.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0],
                [99.0, 1.0e300, 1.0, 0.0, 0.0, 0.0, 1.0],
                [99.0, 1.0, 1.0e300, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.array([4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = runtime_primary_cache.rasterize_hit_tables_to_image(
            hit_tables,
            image_size=5,
        )

    expected = np.zeros((5, 5), dtype=np.float64)
    expected[2, 1] += 3.75
    expected[2, 2] += 1.25
    expected[3, 1] += 3.75
    expected[3, 2] += 1.25
    expected[1, 1] += 5.0
    expected[0, 0] += 6.0
    expected[0, 2] += 4.0
    assert np.allclose(image, expected)


def test_rematerialize_primary_artifacts_matches_full_simulation_artifacts() -> None:
    request = _build_request()
    result = simulate(request)

    payload = runtime_primary_cache.rematerialize_primary_artifacts(
        primary_hit_table_cache={0: np.asarray(result.hit_tables[0], dtype=np.float64)},
        active_keys=[0],
        image_size=request.geometry.image_size,
        a_primary=request.geometry.av,
        c_primary=request.geometry.cv,
        beam_x_array=request.beam.beam_x_array,
        beam_y_array=request.beam.beam_y_array,
        theta_array=request.beam.theta_array,
        phi_array=request.beam.phi_array,
        wavelength_array=request.beam.wavelength_array,
    )

    expected_peak_tables = intersection_cache_to_hit_tables(result.intersection_cache)
    if not expected_peak_tables:
        expected_peak_tables = [np.asarray(result.hit_tables[0], dtype=np.float64)]
    assert np.allclose(payload["image"], result.image)
    assert len(payload["intersection_cache"]) == len(result.intersection_cache)
    for rebuilt, full in zip(payload["intersection_cache"], result.intersection_cache):
        assert np.allclose(rebuilt, full)
    assert len(payload["peak_tables"]) == len(expected_peak_tables)
    for rebuilt, expected in zip(payload["peak_tables"], expected_peak_tables):
        assert np.allclose(rebuilt, expected)


def test_prune_reuse_image_equals_sum_of_selected_cached_contributions() -> None:
    hit_table_cache = {
        0: np.asarray([[4.0, 1.0, 1.0]], dtype=np.float64),
        1: np.asarray([[99.0, 2.0, 2.0]], dtype=np.float64),
        2: np.asarray([[6.0, 3.0, 1.0]], dtype=np.float64),
    }

    payload = runtime_primary_cache.rematerialize_primary_artifacts(
        primary_hit_table_cache=hit_table_cache,
        active_keys=[0, 2],
        image_size=5,
        a_primary=4.0,
        c_primary=7.0,
    )

    expected = runtime_primary_cache.rasterize_hit_tables_to_image(
        [hit_table_cache[0], hit_table_cache[2]],
        image_size=5,
    )
    excluded = runtime_primary_cache.rasterize_hit_tables_to_image(
        [hit_table_cache[1]],
        image_size=5,
    )
    assert np.allclose(payload["image"], expected)
    assert not np.any((payload["image"] > 0.0) & (excluded > 0.0))


def test_rematerialize_primary_artifacts_prefers_cached_representative_intersection_cache() -> None:
    sampled_hit_table = np.asarray(
        [[5.0, 20.0, 12.0, -0.2, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    representative_cache = np.full((1, cache_schema.CURRENT_DETECTOR_CACHE_WIDTH), np.nan)
    representative_cache[0, :9] = [
        1.0,
        2.0,
        10.0,
        12.0,
        0.0,
        -0.2,
        1.0,
        0.0,
        2.0,
    ]
    representative_cache[0, 9:14] = 0.0
    representative_cache[0, cache_schema.CACHE_COL_SOURCE_TABLE_INDEX] = 0.0
    representative_cache[0, cache_schema.CACHE_COL_SOURCE_ROW_INDEX] = 0.0

    payload = runtime_primary_cache.rematerialize_primary_artifacts(
        primary_hit_table_cache={0: sampled_hit_table},
        primary_intersection_cache_cache={0: [representative_cache]},
        active_keys=[0],
        image_size=32,
        a_primary=4.0,
        c_primary=7.0,
    )

    assert float(np.sum(payload["image"])) == pytest.approx(5.0)
    assert len(payload["intersection_cache"]) == 1
    cache_row = np.asarray(payload["intersection_cache"][0], dtype=np.float64)[0]
    assert float(cache_row[cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(10.0)
    assert float(cache_row[cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.0)
    np.testing.assert_allclose(cache_row[9:14], np.zeros(5, dtype=np.float64))
    peak_rows = payload["peak_tables"]
    assert len(peak_rows) == 1
    assert float(np.asarray(peak_rows[0], dtype=np.float64)[0, 1]) == pytest.approx(10.0)


def test_rematerialize_primary_artifacts_treats_bad_best_sample_indices_as_missing(
    monkeypatch,
) -> None:
    seen_best_sample_indices: list[np.ndarray] = []

    def fake_build_intersection_cache(*_args, **kwargs):
        seen_best_sample_indices.append(
            np.asarray(kwargs["best_sample_indices_out"], dtype=np.int64).copy()
        )
        return []

    monkeypatch.setattr(
        runtime_primary_cache,
        "build_intersection_cache",
        fake_build_intersection_cache,
    )

    payload = runtime_primary_cache.rematerialize_primary_artifacts(
        primary_hit_table_cache={
            0: np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        },
        primary_best_sample_index_cache={0: None},
        active_keys=[0],
        image_size=4,
        a_primary=4.0,
        c_primary=7.0,
    )

    assert len(payload["raw_hit_tables"]) == 1
    assert len(payload["peak_tables"]) == 1
    assert len(seen_best_sample_indices) == 1
    assert np.array_equal(seen_best_sample_indices[0], np.asarray([-1], dtype=np.int64))


def test_copy_hit_tables_returns_independent_arrays() -> None:
    original = [np.array([[1.0, 2.0, 3.0]], dtype=np.float64)]

    copied = primary_cache_helpers.copy_hit_tables(original)
    copied[0][0, 0] = 99.0

    assert original[0][0, 0] == 1.0


def test_store_primary_cache_payload_handles_discard_path() -> None:
    state = SimpleNamespace(
        primary_contribution_cache_signature=("sig", 1),
        primary_source_mode="miller",
        primary_active_contribution_keys=[0],
        primary_hit_table_cache={0: np.ones((1, 7), dtype=np.float64)},
        primary_filter_signature=None,
    )
    events: list[dict[str, object]] = []

    def trace_live_cache_event(family: str, action: str, **fields: object) -> None:
        events.append({"family": family, "action": action, **fields})

    primary_cache_helpers.store_primary_cache_payload(
        state,
        cache_signature=("sig", 1),
        source_mode="miller",
        active_keys=[1],
        contribution_keys=[1],
        raw_hit_tables=[np.ones((1, 7), dtype=np.float64)],
        best_sample_indices=None,
        retain_runtime_optional_cache=lambda *_args, **_kwargs: False,
        trace_live_cache_event=trace_live_cache_event,
        live_cache_count=lambda value: len(value) if value is not None else 0,
        live_cache_signature_summary=lambda value: str(value),
    )

    assert state.primary_hit_table_cache == {}
    assert state.primary_active_contribution_keys == [1]
    assert events[-1]["outcome"] == "discarded"
    assert events[-1]["stored_table_count"] == 0


def test_store_primary_cache_payload_can_store_detector_relative_hit_tables() -> None:
    state = SimpleNamespace(
        primary_contribution_cache_signature=None,
        primary_source_mode=None,
        primary_active_contribution_keys=[],
        primary_hit_table_cache={},
        primary_best_sample_index_cache={},
        primary_filter_signature=None,
    )
    raw_hit_table = np.asarray(
        [[3.0, 12.0, 23.0, 99.0], [5.0, 8.0, 19.0, 42.0]],
        dtype=np.float64,
    )

    primary_cache_helpers.store_primary_cache_payload(
        state,
        cache_signature=("sig", 2),
        source_mode="miller",
        active_keys=[10],
        contribution_keys=[10],
        raw_hit_tables=[raw_hit_table],
        best_sample_indices=[4],
        retain_runtime_optional_cache=lambda *_args, **_kwargs: True,
        trace_live_cache_event=lambda *_args, **_kwargs: None,
        live_cache_count=lambda value: len(value) if value is not None else 0,
        live_cache_signature_summary=lambda value: str(value),
        detector_center=(10.0, 20.0),
        store_detector_relative_hit_tables=True,
    )

    assert state.primary_relative_hit_table_cache_center == (10.0, 20.0)
    assert state.primary_relative_hit_table_cache_signature == ("sig", 2)
    assert np.allclose(
        state.primary_relative_hit_table_cache[10],
        np.asarray([[3.0, 2.0, 3.0, 99.0], [5.0, -2.0, -1.0, 42.0]]),
    )
    assert np.allclose(raw_hit_table[:, 1:3], [[12.0, 23.0], [8.0, 19.0]])


def test_store_primary_cache_payload_stores_representative_intersection_cache() -> None:
    state = SimpleNamespace(
        primary_contribution_cache_signature=None,
        primary_source_mode=None,
        primary_active_contribution_keys=[],
        primary_hit_table_cache={},
        primary_best_sample_index_cache={},
        primary_filter_signature=None,
    )
    sampled_hit_table = np.asarray(
        [[5.0, 20.0, 12.0, -0.2, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    representative_cache = np.full((1, cache_schema.CURRENT_DETECTOR_CACHE_WIDTH), np.nan)
    representative_cache[0, :9] = [
        1.0,
        2.0,
        10.0,
        12.0,
        0.0,
        -0.2,
        1.0,
        0.0,
        2.0,
    ]
    representative_cache[0, 9:14] = 0.0
    representative_cache[0, cache_schema.CACHE_COL_SOURCE_TABLE_INDEX] = 0.0
    representative_cache[0, cache_schema.CACHE_COL_SOURCE_ROW_INDEX] = 0.0

    primary_cache_helpers.store_primary_cache_payload(
        state,
        cache_signature=("sig", 3),
        source_mode="miller",
        active_keys=["peak-a"],
        contribution_keys=["peak-a"],
        raw_hit_tables=[sampled_hit_table],
        best_sample_indices=[1],
        representative_intersection_cache=[representative_cache],
        retain_runtime_optional_cache=lambda *_args, **_kwargs: True,
        trace_live_cache_event=lambda *_args, **_kwargs: None,
        live_cache_count=lambda value: len(value) if value is not None else 0,
        live_cache_signature_summary=lambda value: str(value),
    )

    payload = primary_cache_helpers.rematerialize_primary_cache_artifacts(
        state,
        image_size=32,
        mosaic_params={
            "beam_x_array": np.zeros(2, dtype=np.float64),
            "beam_y_array": np.zeros(2, dtype=np.float64),
            "theta_array": np.zeros(2, dtype=np.float64),
            "phi_array": np.zeros(2, dtype=np.float64),
            "wavelength_array": np.ones(2, dtype=np.float64),
        },
        a_primary=4.0,
        c_primary=7.0,
        trace_live_cache_event=lambda *_args, **_kwargs: None,
        live_cache_signature_summary=lambda value: str(value),
        live_cache_shape=lambda value: list(np.asarray(value).shape),
        live_cache_count=lambda value: len(value) if value is not None else 0,
    )

    cache_row = np.asarray(payload["intersection_cache"][0], dtype=np.float64)[0]
    assert float(np.sum(payload["image"])) == pytest.approx(5.0)
    assert float(cache_row[cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(10.0)
    assert float(cache_row[cache_schema.CACHE_COL_INTENSITY]) == pytest.approx(0.0)


def test_store_primary_cache_payload_drops_stale_representative_intersection_cache() -> None:
    state = SimpleNamespace(
        primary_contribution_cache_signature=("sig", 3),
        primary_source_mode="miller",
        primary_active_contribution_keys=["peak-a"],
        primary_hit_table_cache={"peak-a": np.ones((1, 10), dtype=np.float64)},
        primary_best_sample_index_cache={"peak-a": 0},
        primary_intersection_cache_entry_cache={
            "peak-a": [np.ones((1, cache_schema.CURRENT_DETECTOR_CACHE_WIDTH), dtype=np.float64)]
        },
        primary_filter_signature=None,
    )

    primary_cache_helpers.store_primary_cache_payload(
        state,
        cache_signature=("sig", 3),
        source_mode="miller",
        active_keys=["peak-a"],
        contribution_keys=["peak-a"],
        raw_hit_tables=[np.asarray([[1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 2.0]], dtype=np.float64)],
        best_sample_indices=[0],
        representative_intersection_cache=None,
        retain_runtime_optional_cache=lambda *_args, **_kwargs: True,
        trace_live_cache_event=lambda *_args, **_kwargs: None,
        live_cache_count=lambda value: len(value) if value is not None else 0,
        live_cache_signature_summary=lambda value: str(value),
    )

    assert state.primary_intersection_cache_entry_cache == {}


def test_translate_intersection_cache_entry_cache_for_center_delta() -> None:
    representative_cache = np.full((1, cache_schema.CURRENT_DETECTOR_CACHE_WIDTH), np.nan)
    representative_cache[0, cache_schema.CACHE_COL_DETECTOR_COL] = 10.0
    representative_cache[0, cache_schema.CACHE_COL_DETECTOR_ROW] = 12.0
    representative_cache[0, cache_schema.CACHE_COL_INTENSITY] = 0.0
    representative_cache[0, cache_schema.CACHE_COL_H : cache_schema.CACHE_COL_L + 1] = [
        1.0,
        0.0,
        2.0,
    ]

    translated = runtime_primary_cache.translate_intersection_cache_entry_cache_for_center_delta(
        {"peak-a": [representative_cache]},
        delta_row=3.0,
        delta_col=-2.0,
    )

    translated_row = np.asarray(translated["peak-a"][0], dtype=np.float64)[0]
    assert float(translated_row[cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(8.0)
    assert float(translated_row[cache_schema.CACHE_COL_DETECTOR_ROW]) == pytest.approx(15.0)
    assert float(representative_cache[0, cache_schema.CACHE_COL_DETECTOR_COL]) == pytest.approx(
        10.0
    )


def test_clear_primary_contribution_cache_clears_detector_relative_hit_tables() -> None:
    state = SimpleNamespace(
        primary_contribution_cache_signature=("sig", 1),
        primary_active_contribution_keys=[1],
        primary_hit_table_cache={1: np.ones((1, 4), dtype=np.float64)},
        primary_best_sample_index_cache={1: 0},
        primary_intersection_cache_entry_cache={1: [np.ones((1, 17), dtype=np.float64)]},
        primary_relative_hit_table_cache={1: np.ones((1, 4), dtype=np.float64)},
        primary_relative_hit_table_cache_center=(10.0, 20.0),
        primary_relative_hit_table_cache_signature=("sig", 1),
        primary_filter_signature=("filter", 1),
    )

    primary_cache_helpers.clear_primary_contribution_cache(state)

    assert state.primary_relative_hit_table_cache == {}
    assert state.primary_intersection_cache_entry_cache == {}
    assert state.primary_relative_hit_table_cache_center is None
    assert state.primary_relative_hit_table_cache_signature is None
