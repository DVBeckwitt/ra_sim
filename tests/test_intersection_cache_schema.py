import json

import numpy as np
import pytest

from ra_sim.gui import mosaic_top_selection
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
    layout = schema.classify_intersection_cache_table(np.empty((0, width), dtype=np.float64))

    assert layout.is_valid is True
    assert layout.cache_kind == kind
    assert layout.has_provenance is has_provenance
    assert layout.has_cached_caked_angles is has_caked_angles
    assert layout.hit_row_width == (
        schema.HIT_ROW_WITH_PROVENANCE_WIDTH if has_provenance else schema.BASE_HIT_ROW_WIDTH
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


def test_coerce_intersection_cache_table_rejects_wrong_width_numeric_tables() -> None:
    wrong_width = np.ones((1, schema.BASE_HIT_ROW_WIDTH), dtype=np.float64)
    coerced = schema.coerce_intersection_cache_table(wrong_width)

    assert coerced.shape == (0, schema.CURRENT_DETECTOR_CACHE_WIDTH)


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


def test_cache_table_to_hit_table_preserves_representative_best_sample_index() -> None:
    detector_current = np.array(
        [[0.3, 0.4, 11.5, 21.5, 3.0, 31.0, 1.0, 0.0, 2.0, 0, 0, 0, 0, 0, 5, 6, 12]],
        dtype=np.float64,
    )

    hit_table = schema.cache_table_to_hit_table(detector_current)

    assert hit_table.shape == (1, schema.HIT_ROW_WITH_PROVENANCE_WIDTH)
    assert float(hit_table[0, schema.HIT_ROW_COL_SOURCE_TABLE_INDEX]) == pytest.approx(5.0)
    assert float(hit_table[0, schema.HIT_ROW_COL_SOURCE_ROW_INDEX]) == pytest.approx(6.0)
    assert float(hit_table[0, schema.HIT_ROW_COL_BEST_SAMPLE_INDEX]) == pytest.approx(12.0)


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


def test_mosaic_top_selection_keeps_raw_cache_intact_and_uses_dedicated_weight() -> None:
    raw_cache = [
        {
            "q_group_key": ("q_group", "primary", 1, 0),
            "branch_id": "+x",
            "branch_source": "generated",
            "source_row_index": 1,
            "best_sample_index": 1,
            "weight": 999.0,
            "intensity": 999.0,
            "mosaic_weight": 0.1,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 0),
            "branch_id": "+x",
            "branch_source": "generated",
            "source_row_index": 2,
            "best_sample_index": 0,
            "weight": 1.0,
            "intensity": 1.0,
            "mosaic_weight": 0.9,
        },
    ]
    raw_digest = [
        (id(entry), tuple(sorted(entry.keys())), entry["source_row_index"]) for entry in raw_cache
    ]

    selection_cache = mosaic_top_selection.build_selection_cache(raw_cache)

    assert [
        (id(entry), tuple(sorted(entry.keys())), entry["source_row_index"]) for entry in raw_cache
    ] == raw_digest
    assert all("mosaic_top_rank_key" not in entry for entry in raw_cache)
    assert len(selection_cache) == 1
    selected = selection_cache[0]
    assert selected["source_row_index"] == 2
    assert selected["mosaic_weight"] == 0.9
    assert selected["selection_reason"] == "mosaic_top_per_branch"
    assert isinstance(selected["mosaic_top_rank_key"], tuple)
    json.dumps(selected["mosaic_top_rank_key"], allow_nan=False)


def test_mosaic_top_selection_uses_profile_cache_sample_weights_and_beam_x() -> None:
    target = ("q_group", "primary", 1, 0)
    raw_cache = [
        {
            "q_group_key": target,
            "source_branch_index": 0,
            "best_sample_index": 0,
            "source_row_index": 1,
            "intensity": 999.0,
        },
        {
            "q_group_key": target,
            "source_branch_index": 1,
            "best_sample_index": 1,
            "source_row_index": 2,
            "intensity": 1.0,
        },
    ]
    profile_cache = {
        "beam_x_array": np.asarray([1.0, 1.0], dtype=float),
        "sample_weights": np.asarray([0.1, 0.9], dtype=float),
    }

    selected = mosaic_top_selection.select_mosaic_top_representative(
        raw_cache,
        branch_id="+x",
        target_key=target,
        profile_cache=profile_cache,
    )

    assert all("mosaic_weight" not in entry for entry in raw_cache)
    assert selected["source_row_index"] == 2
    assert selected["branch_id"] == "+x"
    assert selected["branch_source"] == "generated"
    assert selected["mosaic_weight"] == 0.9


def test_mosaic_top_rank_edges_are_weight_angular_tie_then_source_order() -> None:
    target = ("q_group", "primary", 1, 0)
    finite_weight_wins = mosaic_top_selection.select_mosaic_top_representative(
        [
            {
                "q_group_key": target,
                "branch_id": "+x",
                "mosaic_weight": 0.1,
                "theta_offset": 0.0,
                "phi_offset": 0.0,
                "source_row_index": 1,
            },
            {
                "q_group_key": target,
                "branch_id": "+x",
                "mosaic_weight": 0.9,
                "theta_offset": 9.0,
                "phi_offset": 9.0,
                "source_row_index": 2,
            },
        ],
        branch_id="+x",
        target_key=target,
    )
    assert finite_weight_wins["source_row_index"] == 2

    angular_fallback = mosaic_top_selection.select_mosaic_top_representative(
        [
            {
                "q_group_key": target,
                "branch_id": "+x",
                "mosaic_weight": np.nan,
                "theta_offset": 0.4,
                "phi_offset": 0.4,
                "source_row_index": 3,
                "intensity": 999.0,
            },
            {
                "q_group_key": target,
                "branch_id": "+x",
                "theta_offset": 0.0,
                "phi_offset": 0.1,
                "source_row_index": 4,
                "intensity": 1.0,
            },
        ],
        branch_id="+x",
        target_key=target,
    )
    assert angular_fallback["source_row_index"] == 4

    source_order_fallback = mosaic_top_selection.select_mosaic_top_representative(
        [
            {"q_group_key": target, "branch_id": "+x", "source_row_index": 5},
            {"q_group_key": target, "branch_id": "+x", "source_row_index": 6},
        ],
        branch_id="+x",
        target_key=target,
    )
    assert source_order_fallback["source_row_index"] == 5

    intensity_tie = mosaic_top_selection.select_mosaic_top_representative(
        [
            {
                "q_group_key": target,
                "branch_id": "+x",
                "theta_offset": 0.2,
                "phi_offset": 0.0,
                "source_row_index": 7,
                "intensity": 1.0,
            },
            {
                "q_group_key": target,
                "branch_id": "+x",
                "theta_offset": 0.2,
                "phi_offset": 0.0,
                "source_row_index": 8,
                "intensity": 2.0,
            },
        ],
        branch_id="+x",
        target_key=target,
    )
    assert intensity_tie["source_row_index"] == 8


def test_branch_ids_are_generated_or_stable_unknown_without_source_branch_mapping() -> None:
    target = ("q_group", "primary", 1, 0)

    assert mosaic_top_selection.normalize_branch_id(
        {"signed_x_branch": 1.0},
        target_key=target,
    ) == ("+x", "generated")
    assert mosaic_top_selection.normalize_branch_id(
        {"signed_x_branch": -1.0},
        target_key=target,
    ) == ("-x", "generated")

    inferred_from_legacy = mosaic_top_selection.normalize_branch_id(
        {"source_branch_index": 0, "q_group_key": target},
        target_key=target,
    )
    assert inferred_from_legacy[0].startswith("unknown:")
    assert inferred_from_legacy[1] == "unknown"
    other_legacy = mosaic_top_selection.normalize_branch_id(
        {"source_branch_index": 1, "q_group_key": target},
        target_key=target,
    )
    assert other_legacy[0].startswith("unknown:")
    assert other_legacy[1] == "unknown"
    assert other_legacy != inferred_from_legacy

    cache = mosaic_top_selection.build_selection_cache(
        [
            {"q_group_key": target, "signed_x_branch": 1.0, "source_row_index": 1},
            {"q_group_key": target, "signed_x_branch": -1.0, "source_row_index": 2},
            {"q_group_key": target, "source_branch_index": 0, "source_row_index": 3},
        ]
    )
    by_branch = {entry["branch_id"]: entry for entry in cache}
    assert "+x" in by_branch
    assert "-x" in by_branch
    assert any(branch.startswith("unknown:") for branch in by_branch)
