from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import runtime_primary_cache
from ra_sim.gui._runtime import primary_cache_helpers
from ra_sim.simulation.diffraction import intersection_cache_to_hit_tables
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
        retain_runtime_optional_cache=lambda *_args, **_kwargs: False,
        trace_live_cache_event=trace_live_cache_event,
        live_cache_count=lambda value: len(value) if value is not None else 0,
        live_cache_signature_summary=lambda value: str(value),
    )

    assert state.primary_hit_table_cache == {}
    assert state.primary_active_contribution_keys == [1]
    assert events[-1]["outcome"] == "discarded"
    assert events[-1]["stored_table_count"] == 0
