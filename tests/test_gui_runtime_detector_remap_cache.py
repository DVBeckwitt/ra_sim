from __future__ import annotations

import numpy as np

from ra_sim.gui import runtime_detector_remap_cache as remap_cache
from ra_sim.gui import runtime_primary_cache


def test_translate_hit_tables_for_center_delta_shifts_only_col_row() -> None:
    hit_table = np.asarray(
        [
            [10.0, 1.25, 2.5, 7.0, 8.0],
            [5.0, -0.5, 4.0, 9.0, 10.0],
        ],
        dtype=np.float64,
    )

    translated = remap_cache.translate_hit_tables_for_center_delta(
        [hit_table],
        delta_row=-1.5,
        delta_col=3.0,
        image_size=16,
    )

    assert len(translated) == 1
    assert np.allclose(translated[0][:, 0], hit_table[:, 0])
    assert np.allclose(translated[0][:, 1], hit_table[:, 1] + 3.0)
    assert np.allclose(translated[0][:, 2], hit_table[:, 2] - 1.5)
    assert np.allclose(translated[0][:, 3:], hit_table[:, 3:])
    assert np.allclose(hit_table[:, 1], [1.25, -0.5])


def test_materialize_absolute_hit_tables_from_relative_uses_native_row_col_center() -> None:
    relative_table = np.asarray(
        [
            [12.0, -2.0, 3.5, 42.0],
            [6.0, 1.25, -4.0, 24.0],
        ],
        dtype=np.float64,
    )

    absolute = remap_cache.materialize_absolute_hit_tables_from_relative(
        [relative_table],
        new_center=(10.0, 20.0),
    )

    assert len(absolute) == 1
    assert np.allclose(absolute[0][:, 0], relative_table[:, 0])
    assert np.allclose(absolute[0][:, 1], [18.0, 21.25])
    assert np.allclose(absolute[0][:, 2], [13.5, 6.0])
    assert np.allclose(absolute[0][:, 3:], relative_table[:, 3:])


def test_detector_relative_center_cache_uses_row_col_not_col_row() -> None:
    hit_table = np.asarray([[9.0, 1607.0, 1544.0, 1.0]], dtype=np.float64)

    relative = remap_cache.make_relative_hit_tables_for_center(
        [hit_table],
        center=(1544.0, 1607.0),
    )
    absolute = remap_cache.materialize_absolute_hit_tables_from_relative(
        relative,
        new_center=(1500.0, 1700.0),
    )

    np.testing.assert_allclose(relative[0][:, 1:3], [[0.0, 0.0]])
    np.testing.assert_allclose(absolute[0][:, 1:3], [[1700.0, 1500.0]])


def test_center_remap_falls_back_when_cache_missing() -> None:
    assert (
        remap_cache.can_remap_detector_center_exactly(
            None,
            old_center=(10.0, 20.0),
            new_center=(11.0, 19.5),
            image_size=128,
        )
        is False
    )


def test_center_remap_falls_back_when_cache_is_clipped_only() -> None:
    clipped_table = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    state = remap_cache.DetectorCenterRemapCacheState(
        clipped_hit_tables=[clipped_table],
    )

    assert (
        remap_cache.can_remap_detector_center_exactly(
            state,
            old_center=(10.0, 20.0),
            new_center=(11.0, 19.5),
            image_size=128,
        )
        is False
    )


def test_center_remap_never_uses_clipped_only_cache() -> None:
    clipped_table = np.asarray([[7.0, 4.0, 5.0, 1.0, 2.0]], dtype=np.float64)
    state = {
        "clipped_hit_tables": [clipped_table],
        "detector_relative_hit_tables": None,
        "unclipped_hit_tables": None,
        "has_detector_relative_hit_tables": False,
        "has_unclipped_hit_tables": False,
    }

    assert (
        remap_cache.can_remap_detector_center_exactly(
            state,
            old_center=(10.0, 10.0),
            new_center=(11.0, 9.0),
            image_size=64,
        )
        is False
    )


def test_center_remap_accepts_detector_relative_cache() -> None:
    relative_table = np.asarray([[1.0, -2.0, 3.0]], dtype=np.float64)
    state = remap_cache.DetectorCenterRemapCacheState(
        detector_relative_hit_tables=[relative_table],
    )

    assert (
        remap_cache.can_remap_detector_center_exactly(
            state,
            old_center=(10.0, 20.0),
            new_center=(11.0, 19.5),
            image_size=128,
        )
        is True
    )


def test_center_remap_matches_full_simulation_for_safe_in_bounds_shift() -> None:
    old_hit_table = np.asarray(
        [[4.0, 1.25, 1.5, 0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    relative_tables = remap_cache.make_relative_hit_tables_for_center(
        [old_hit_table],
        center=(1.0, 1.0),
    )
    remapped_tables = remap_cache.materialize_absolute_hit_tables_from_relative(
        relative_tables,
        new_center=(1.5, 1.25),
    )
    full_simulation_at_new_center_tables = [
        np.asarray([[4.0, 1.5, 2.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    ]

    remapped_image = runtime_primary_cache.rasterize_hit_tables_to_image(
        remapped_tables,
        image_size=4,
    )
    full_image_at_new_center = runtime_primary_cache.rasterize_hit_tables_to_image(
        full_simulation_at_new_center_tables,
        image_size=4,
    )

    np.testing.assert_allclose(
        remapped_image,
        full_image_at_new_center,
        rtol=1e-10,
        atol=1e-10,
    )


def test_translate_hit_tables_keeps_non_position_metadata_unchanged() -> None:
    hit_table = np.asarray(
        [[4.0, 1.25, 1.5, 9.0, 8.0, 7.0, 6.0]],
        dtype=np.float64,
    )

    translated = remap_cache.translate_hit_tables_for_center_delta(
        [hit_table],
        delta_row=0.25,
        delta_col=0.5,
        image_size=4,
    )[0]

    np.testing.assert_allclose(translated[:, 0], hit_table[:, 0])
    np.testing.assert_allclose(translated[:, 1], hit_table[:, 1] + 0.5)
    np.testing.assert_allclose(translated[:, 2], hit_table[:, 2] + 0.25)
    np.testing.assert_allclose(translated[:, 3:], hit_table[:, 3:])
