import numpy as np
import pytest

from ra_sim.simulation import intersection_cache_schema as schema


@pytest.mark.parametrize(
    ("width", "kind", "has_provenance", "has_caked_angles"),
    [
        (schema.LEGACY_DETECTOR_CACHE_WIDTH, "detector", False, False),
        (schema.CURRENT_DETECTOR_CACHE_WIDTH, "detector", True, False),
        (schema.LEGACY_CAKED_CACHE_WIDTH, "caked", False, True),
        (schema.CURRENT_CAKED_CACHE_WIDTH, "caked", True, True),
    ],
)
def test_classify_intersection_cache_table_supports_known_layouts(
    width: int,
    kind: str,
    has_provenance: bool,
    has_caked_angles: bool,
) -> None:
    layout = schema.classify_intersection_cache_table(
        np.empty((0, width), dtype=np.float64)
    )

    assert layout.is_valid is True
    assert layout.cache_kind == kind
    assert layout.has_provenance is has_provenance
    assert layout.has_cached_caked_angles is has_caked_angles
    assert layout.hit_row_width == (
        schema.HIT_ROW_WITH_PROVENANCE_WIDTH
        if has_provenance
        else schema.BASE_HIT_ROW_WIDTH
    )


def test_coerce_float64_table_returns_detached_2d_or_canonical_empty() -> None:
    row = np.arange(schema.LEGACY_DETECTOR_CACHE_WIDTH, dtype=np.float32)
    coerced = schema.coerce_float64_table(row)
    row[0] = -99.0

    assert coerced.shape == (1, schema.LEGACY_DETECTOR_CACHE_WIDTH)
    assert coerced.dtype == np.float64
    assert coerced[0, 0] == 0.0

    invalid = schema.coerce_float64_table(object(), empty_width=schema.CURRENT_CAKED_CACHE_WIDTH)
    assert invalid.shape == (0, schema.CURRENT_CAKED_CACHE_WIDTH)


def test_classify_abbreviated_detector_cache_as_invalid_layout() -> None:
    layout = schema.classify_intersection_cache_table(np.empty((0, 9), dtype=np.float64))

    assert layout.is_valid is False
    assert schema.is_intersection_cache_table(np.empty((0, 9), dtype=np.float64)) is False
    assert schema.cache_table_to_hit_table(np.empty((1, 9), dtype=np.float64)).shape == (
        0,
        schema.BASE_HIT_ROW_WIDTH,
    )


def test_cache_table_to_hit_table_supports_all_known_layouts() -> None:
    detector_legacy = np.array(
        [[0.1, 0.2, 10.5, 20.5, 7.0, 30.0, 1.0, 0.0, 2.0, 0, 0, 0, 0, 0]],
        dtype=np.float64,
    )
    detector_current = np.array(
        [[0.3, 0.4, 11.5, 21.5, 3.0, 31.0, 1.0, 0.0, 2.0, 0, 0, 0, 0, 0, 5, 6, 7]],
        dtype=np.float64,
    )
    caked_legacy = np.array(
        [[0.5, 0.6, 12.5, 22.5, 4.0, 32.0, 2.0, 1.0, 3.0, 0, 0, 0, 0, 0, 17.5, -32.0]],
        dtype=np.float64,
    )
    caked_current = np.array(
        [
            [
                0.7,
                0.8,
                13.5,
                23.5,
                5.0,
                33.0,
                2.0,
                1.0,
                4.0,
                0,
                0,
                0,
                0,
                0,
                8,
                9,
                10,
                18.5,
                -31.0,
            ]
        ],
        dtype=np.float64,
    )

    legacy_hit = schema.cache_table_to_hit_table(detector_legacy)
    current_hit = schema.cache_table_to_hit_table(detector_current)
    legacy_caked_hit = schema.cache_table_to_hit_table(caked_legacy)
    current_caked_hit = schema.cache_table_to_hit_table(caked_current)

    assert legacy_hit.shape == (1, schema.BASE_HIT_ROW_WIDTH)
    assert legacy_caked_hit.shape == (1, schema.BASE_HIT_ROW_WIDTH)
    assert current_hit.shape == (1, schema.HIT_ROW_WITH_PROVENANCE_WIDTH)
    assert current_caked_hit.shape == (1, schema.HIT_ROW_WITH_PROVENANCE_WIDTH)
    np.testing.assert_allclose(legacy_hit[0], [7.0, 10.5, 20.5, 30.0, 1.0, 0.0, 2.0])
    np.testing.assert_allclose(
        current_hit[0],
        [3.0, 11.5, 21.5, 31.0, 1.0, 0.0, 2.0, 5.0, 6.0, 7.0],
    )
    np.testing.assert_allclose(legacy_caked_hit[0], [4.0, 12.5, 22.5, 32.0, 2.0, 1.0, 3.0])
    np.testing.assert_allclose(
        current_caked_hit[0],
        [5.0, 13.5, 23.5, 33.0, 2.0, 1.0, 4.0, 8.0, 9.0, 10.0],
    )


def test_extract_provenance_helpers_return_optional_indices() -> None:
    cache_row = np.array(
        [0, 0, 10, 20, 8, 0.25, 1, 0, 2, 0, 0, 0, 0, 0, 4, 5, 6, 17.5, -32.0],
        dtype=np.float64,
    )
    hit_row = np.array([8.0, 10.0, 20.0, 0.25, 1.0, 0.0, 2.0, 4.0, 5.0, 6.0], dtype=np.float64)
    legacy_caked_row = np.array(
        [0, 0, 10, 20, 8, 0.25, 1, 0, 2, 0, 0, 0, 0, 0, 17.5, -32.0],
        dtype=np.float64,
    )

    assert schema.extract_cache_row_provenance(cache_row) == (4, 5, 6)
    assert schema.extract_hit_row_provenance(hit_row) == (4, 5, 6)
    assert schema.extract_cache_row_provenance(legacy_caked_row) == (None, None, None)
    assert schema.extract_hit_row_provenance(hit_row[:7]) == (None, None, None)


def test_extract_cached_caked_angles_reads_legacy_and_current_layouts() -> None:
    legacy_row = np.array(
        [0, 0, 10, 20, 8, 0.25, 1, 0, 2, 0, 0, 0, 0, 0, 17.5, -32.0],
        dtype=np.float64,
    )
    current_row = np.array(
        [0, 0, 10, 20, 8, 0.25, 1, 0, 2, 0, 0, 0, 0, 0, 4, 5, 6, 18.5, -31.0],
        dtype=np.float64,
    )
    detector_row = np.array(
        [0, 0, 10, 20, 8, 0.25, 1, 0, 2, 0, 0, 0, 0, 0, 4, 5, 6],
        dtype=np.float64,
    )

    assert schema.extract_cached_caked_angles(legacy_row) == (17.5, -32.0)
    assert schema.extract_cached_caked_angles(current_row) == (18.5, -31.0)
    missing_two_theta, missing_phi = schema.extract_cached_caked_angles(detector_row)
    assert np.isnan(missing_two_theta)
    assert np.isnan(missing_phi)


def test_invalid_cache_layout_fails_closed() -> None:
    invalid_table = np.ones((2, 15), dtype=np.float64)

    assert schema.is_intersection_cache_table(invalid_table) is False
    converted = schema.cache_table_to_hit_table(invalid_table)
    assert converted.shape == (0, schema.BASE_HIT_ROW_WIDTH)
