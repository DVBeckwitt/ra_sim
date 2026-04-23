from types import SimpleNamespace

import numpy as np

from ra_sim.fitting.background_peak_matching import build_background_peak_context
from ra_sim.gui import manual_geometry as mg
from ra_sim.gui import peak_selection as ps
from ra_sim.simulation import exact_cake_portable


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyAxis:
    def __init__(self, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
        self._xlim = tuple(float(v) for v in xlim)
        self._ylim = tuple(float(v) for v in ylim)

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, left, right):
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom, top):
        self._ylim = (float(bottom), float(top))


class _DummyCanvas:
    def __init__(self) -> None:
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


def _pairs_for_index(pairs_by_background: dict[int, list[dict[str, object]]], index: int):
    return mg.geometry_manual_pairs_for_index(
        index,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _set_pairs(
    pairs_by_background: dict[int, list[dict[str, object]]],
    index: int,
    entries,
):
    return mg.set_geometry_manual_pairs_for_index(
        index,
        entries,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _group_count(
    pairs_by_background: dict[int, list[dict[str, object]]],
    index: int,
) -> int:
    return mg.geometry_manual_pair_group_count(
        index,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _wrap_phi_range(phi_values):
    return ((np.asarray(phi_values) + 180.0) % 360.0) - 180.0


def _ai_with_live_bundle(bundle):
    ai = type("_LiveBundleAI", (), {})()
    ai._live_caked_transform_bundle = bundle
    return ai


def _dummy_transform_bundle(detector_shape=(6, 6)):
    return mg.CakeTransformBundle(
        detector_shape=tuple(int(v) for v in detector_shape),
        radial_deg=np.array([10.0, 12.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-5.0, 5.0], dtype=np.float64),
        gui_azimuth_deg=np.array([5.0, -5.0], dtype=np.float64),
        lut=object(),
    )


def _real_transform_bundle(detector_shape=(5, 5)):
    image_shape = tuple(int(v) for v in detector_shape)
    image = np.arange(1, int(np.prod(image_shape)) + 1, dtype=np.float32).reshape(image_shape)
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.075,
        poni1=2.5e-4,
        poni2=2.5e-4,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    result = integrator.integrate2d(
        image,
        npt_rad=64,
        npt_azim=180,
        method="lut",
        unit="2th_deg",
        workers=1,
    )
    bundle = exact_cake_portable.build_cake_transform_bundle_from_result(
        integrator,
        image.shape,
        result,
        workers=1,
    )
    assert isinstance(bundle, mg.CakeTransformBundle)
    return image, bundle


def _fail_projection_legacy_path(message: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(message)

    return _fail


def _group_candidates(entries):
    grouped = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        group_key = raw_entry.get("q_group_key")
        if not isinstance(group_key, tuple):
            continue
        grouped.setdefault(group_key, []).append(dict(raw_entry))
    return grouped


def _source_key(entry):
    return mg.geometry_manual_candidate_source_key(dict(entry) if isinstance(entry, dict) else None)


def _build_lookup(entries):
    lookup = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        key = _source_key(raw_entry)
        if key is None:
            continue
        entry = dict(raw_entry)
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = entry
        elif isinstance(existing, list):
            existing.append(entry)
        else:
            lookup[key] = [dict(existing), entry]
    return lookup


def _candidate_multiset(entries):
    normalized = []
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        hkl_value = raw_entry.get("hkl")
        if isinstance(hkl_value, (list, tuple)) and len(hkl_value) >= 3:
            hkl_value = tuple(int(v) for v in hkl_value[:3])
        else:
            hkl_value = None
        normalized.append(
            {
                "candidate_source_key": _source_key(raw_entry),
                "hkl": hkl_value,
                "source_reflection_index": raw_entry.get("source_reflection_index"),
                "source_reflection_namespace": raw_entry.get("source_reflection_namespace"),
                "source_reflection_is_full": raw_entry.get("source_reflection_is_full"),
                "source_branch_index": raw_entry.get("source_branch_index"),
                "source_peak_index": raw_entry.get("source_peak_index"),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            repr(item["candidate_source_key"]),
            repr(item["hkl"]),
        ),
    )


def test_manual_pair_store_keeps_backgrounds_separate() -> None:
    pairs_by_background: dict[int, list[dict[str, object]]] = {}

    bg0_pairs = _set_pairs(
        pairs_by_background,
        0,
        [
            {
                "label": "1,0,2",
                "x": "10.5",
                "y": 12,
                "q_group_key": ["q_group", "primary", 1, 2],
                "source_table_index": "4",
                "source_row_index": "7",
                "raw_x": "9.5",
                "raw_y": 11.5,
            }
        ],
    )
    bg1_pairs = _set_pairs(
        pairs_by_background,
        1,
        [
            {
                "hkl": (2, 0, 0),
                "x": 5,
                "y": 6,
                "q_group_key": ("q_group", "primary", 2, 0),
            }
        ],
    )

    assert bg0_pairs[0]["hkl"] == (1, 0, 2)
    assert bg0_pairs[0]["source_table_index"] == 4
    assert bg0_pairs[0]["source_row_index"] == 7
    assert bg0_pairs[0]["q_group_key"] == ("q_group", "primary", 1, 2)
    assert bg0_pairs[0]["raw_x"] == 9.5
    assert bg0_pairs[0]["raw_y"] == 11.5
    assert bg0_pairs[0]["placement_error_px"] > 0.0
    assert bg0_pairs[0]["sigma_px"] > bg0_pairs[0]["placement_error_px"]

    assert len(_pairs_for_index(pairs_by_background, 0)) == 1
    assert len(_pairs_for_index(pairs_by_background, 1)) == 1
    assert _group_count(pairs_by_background, 0) == 1
    assert _group_count(pairs_by_background, 1) == 1
    assert (
        _pairs_for_index(pairs_by_background, 0)[0]["hkl"]
        != _pairs_for_index(
            pairs_by_background,
            1,
        )[0]["hkl"]
    )


def test_peak_maximum_near_in_image_returns_local_brightest_pixel() -> None:
    image = np.zeros((9, 9), dtype=float)
    image[4, 4] = 2.0
    image[6, 5] = 9.5
    image[2, 2] = 7.0

    assert mg.peak_maximum_near_in_image(image, 4.2, 4.1, search_radius=1) == (4.0, 4.0)
    assert mg.peak_maximum_near_in_image(image, 4.9, 5.8, search_radius=2) == (5.0, 6.0)


def test_peak_maximum_near_in_image_respects_inverted_display_extent() -> None:
    image = np.zeros((3, 3), dtype=float)
    image[0, 0] = 9.0
    image[2, 0] = 1.0

    peak = mg.peak_maximum_near_in_image(
        image,
        0.2,
        2.8,
        search_radius=1,
        display_extent=(0.0, 3.0, 3.0, 0.0),
    )

    assert peak == (0.5, 2.5)


def test_caked_axis_index_helpers_round_trip() -> None:
    axis = np.linspace(-30.0, 30.0, 121)
    idx = mg.caked_axis_to_image_index(7.5, axis)
    restored = mg.caked_image_index_to_axis(idx, axis)

    assert np.isfinite(idx)
    assert abs(restored - 7.5) < 1e-9


def test_refine_caked_peak_center_finds_ridge_crest() -> None:
    radial = np.linspace(10.0, 20.0, 201)
    azimuth = np.linspace(-30.0, 30.0, 301)
    radial_grid, azimuth_grid = np.meshgrid(radial, azimuth)
    image = 2.0 + 6.0 * np.exp(-0.5 * ((radial_grid - 15.2) / 0.22) ** 2) * np.exp(
        -0.5 * ((azimuth_grid - 7.5) / 4.2) ** 2
    )

    refined_tth, refined_phi = mg.refine_caked_peak_center(image, radial, azimuth, 14.7, 11.0)

    assert abs(refined_tth - 15.2) < 0.08
    assert abs(refined_phi - 7.5) < 0.35


def test_geometry_manual_candidate_source_key_prefers_source_indices() -> None:
    assert mg.geometry_manual_candidate_source_key(
        {
            "source_table_index": "3",
            "source_branch_index": 1,
            "source_peak_index": 5,
        }
    ) == ("source_branch", 3, 1)
    assert mg.geometry_manual_candidate_source_key(
        {"source_table_index": "3", "source_row_index": 9, "source_peak_index": 1}
    ) == ("source_branch", 3, 1)
    assert mg.geometry_manual_candidate_source_key(
        {"source_table_index": "3", "source_row_index": 9, "source_peak_index": 5}
    ) == (
        "source",
        3,
        9,
    )
    assert mg.geometry_manual_candidate_source_key({"hkl": (1, 2, 3)}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "1,2,3"}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "left peak"}) == ("label", "left peak")


def test_geometry_manual_lookup_source_entry_does_not_alias_table_and_reflection_branch_keys() -> (
    None
):
    legacy_saved_entry = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    wrong_candidate = {
        "label": "wrong",
        "hkl": (2, 0, 1),
        "source_table_index": 12,
        "source_row_index": 7,
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    right_candidate = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    simulated_lookup = {
        _source_key(wrong_candidate): dict(wrong_candidate),
        _source_key(right_candidate): dict(right_candidate),
    }

    resolved = mg.geometry_manual_lookup_source_entry(
        simulated_lookup,
        legacy_saved_entry,
    )

    assert resolved is not None
    assert resolved["label"] == "right"
    assert resolved["source_reflection_index"] == 203


def test_geometry_manual_tagged_candidate_from_session_returns_matching_entry() -> None:
    candidate_entries = [
        {"label": "left", "source_table_index": 1, "source_row_index": 2},
        {"label": "right", "source_table_index": 1, "source_row_index": 3},
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {"tagged_candidate_key": ("source", 1, 3)},
        candidate_entries,
    )

    assert tagged is not None
    assert tagged["label"] == "right"
    assert (
        mg.geometry_manual_tagged_candidate_from_session(
            {"tagged_candidate_key": ("source", 9, 9)},
            candidate_entries,
        )
        is None
    )


def test_geometry_manual_tagged_candidate_from_session_prefers_tagged_entry_identity() -> None:
    tagged_candidate = {
        "label": "target-right",
        "hkl": (-1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 3,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    candidate_entries = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "source_table_index": 1,
            "source_row_index": 2,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
        dict(tagged_candidate),
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {
            "tagged_candidate_key": ("source_branch", 1, 1),
            "tagged_candidate": dict(tagged_candidate),
        },
        candidate_entries,
    )

    assert tagged is not None
    assert tagged["label"] == "target-right"
    assert tagged["source_row_index"] == 3


def test_geometry_manual_tagged_candidate_from_session_returns_none_when_stored_tagged_candidate_misses_identity() -> (
    None
):
    candidate_entries = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "source_table_index": 1,
            "source_row_index": 2,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
        {
            "label": "left",
            "hkl": (1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 1,
            "source_branch_index": 0,
            "source_peak_index": 0,
        },
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {
            "tagged_candidate_key": ("source_branch", 1, 1),
            "tagged_candidate": {
                "label": "missing-right",
                "hkl": (-1, 0, 5),
                "source_table_index": 1,
                "source_row_index": 99,
                "source_branch_index": 1,
                "source_peak_index": 1,
            },
        },
        candidate_entries,
    )

    assert tagged is None


def test_geometry_manual_tagged_candidate_from_session_returns_none_for_nonbranch_same_key_identity_miss() -> (
    None
):
    candidate_entries = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "source_table_index": 1,
            "source_row_index": 3,
        },
        {
            "label": "left",
            "hkl": (1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 1,
        },
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {
            "tagged_candidate_key": ("source", 1, 3),
            "tagged_candidate": {
                "label": "missing-right",
                "hkl": (-1, 0, 5),
                "source_table_index": 1,
                "source_row_index": 3,
            },
        },
        candidate_entries,
    )

    assert tagged is None


def test_geometry_manual_tagged_candidate_from_session_returns_none_after_cleared_identity_locked_snapshot() -> (
    None
):
    candidate_entries = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "source_table_index": 1,
            "source_row_index": 3,
        },
        {
            "label": "left",
            "hkl": (1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 1,
        },
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {
            "tagged_candidate_key": ("source", 1, 3),
            "_tagged_candidate_requires_identity": True,
        },
        candidate_entries,
    )

    assert tagged is None


def test_geometry_manual_resolve_source_entry_index_disambiguates_nonbranch_same_key_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 110.0,
            "sim_row": 120.0,
        },
        {
            "label": "target-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 210.0,
            "sim_row": 220.0,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_lookup_source_entry_disambiguates_nonbranch_same_key_bucket() -> None:
    target_entry = {
        "label": "target-right",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
    }
    simulated_lookup = {
        ("source", 9, 0): [
            {
                "label": "other-right",
                "hkl": (-2, 0, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "sim_col": 110.0,
                "sim_row": 120.0,
            },
            {
                "label": "target-right",
                "hkl": (-1, 0, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "sim_col": 210.0,
                "sim_row": 220.0,
            },
        ]
    }

    resolved = mg.geometry_manual_lookup_source_entry(simulated_lookup, target_entry)

    assert resolved is not None
    assert resolved["label"] == "target-right"
    assert resolved["hkl"] == (-1, 0, 5)


def test_geometry_manual_resolve_source_entry_index_prefers_caked_match_over_detector_refined_coords() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 13.1,
            "display_row": 2.1,
            "caked_x": 13.1,
            "caked_y": 2.1,
            "sim_col": 500.0,
            "sim_row": 600.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 30.0,
            "display_row": 40.0,
            "caked_x": 30.0,
            "caked_y": 40.0,
            "sim_col": 300.2,
            "sim_row": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "refined_sim_x": 300.0,
            "refined_sim_y": 400.0,
            "refined_sim_caked_x": 13.0,
            "refined_sim_caked_y": 2.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_prefers_current_view_display_over_saved_xy() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 13.1,
            "display_row": 2.1,
            "sim_col": 500.0,
            "sim_row": 600.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 300.2,
            "display_row": 400.2,
            "sim_col": 30.0,
            "sim_row": 40.0,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 300.0,
            "y": 400.0,
            "display_col": 13.0,
            "display_row": 2.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_prefers_legacy_caked_candidate_for_current_view_display() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "caked_x": 13.1,
            "caked_y": 2.1,
            "sim_col": 500.0,
            "sim_row": 600.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "caked_x": 30.0,
            "caked_y": 40.0,
            "sim_col": 13.2,
            "sim_row": 2.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 13.0,
            "display_row": 2.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_prefers_saved_detector_hints_over_current_view_display_when_both_exist() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "caked_x": 101.0,
            "caked_y": 101.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_saved_native_hints_over_current_view_display_when_both_exist() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "caked_x": 101.0,
            "caked_y": 101.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "sim_native_x": 300.0,
            "sim_native_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_stronger_native_hints_over_saved_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 500.2,
            "sim_native_y": 600.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "sim_native_x": 500.0,
            "sim_native_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_simulated_detector_hints_over_saved_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 500.2,
            "sim_native_y": 600.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "simulated_detector_x": 500.0,
            "simulated_detector_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_simulated_detector_hints_over_saved_detector_hints_for_mixed_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.2,
            "simulated_detector_x": 999.0,
            "simulated_detector_y": 999.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 999.0,
            "detector_y": 999.0,
            "simulated_detector_x": 500.2,
            "simulated_detector_y": 600.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "simulated_detector_x": 500.0,
            "simulated_detector_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_keeps_detector_hint_priority_for_detector_only_entries_with_mixed_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.2,
            "simulated_detector_x": 999.0,
            "simulated_detector_y": 999.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 999.0,
            "detector_y": 999.0,
            "simulated_detector_x": 300.2,
            "simulated_detector_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_uses_current_view_display_to_break_stronger_native_hint_tie_before_saved_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 500.2,
            "sim_native_y": 600.0,
            "caked_x": 100.1,
            "caked_y": 100.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 499.8,
            "sim_native_y": 600.0,
            "caked_x": 200.1,
            "caked_y": 200.1,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "sim_native_x": 500.0,
            "sim_native_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_uses_current_view_display_to_break_simulated_detector_hint_tie_before_saved_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 500.2,
            "sim_native_y": 600.0,
            "caked_x": 100.1,
            "caked_y": 100.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 499.8,
            "sim_native_y": 600.0,
            "caked_x": 200.1,
            "caked_y": 200.1,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "simulated_detector_x": 500.0,
            "simulated_detector_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_uses_current_view_display_to_break_mixed_candidate_simulated_detector_hint_tie_before_weaker_detector_anchor() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.0,
            "simulated_detector_x": 500.2,
            "simulated_detector_y": 600.0,
            "caked_x": 200.1,
            "caked_y": 200.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 299.8,
            "detector_y": 400.0,
            "simulated_detector_x": 499.8,
            "simulated_detector_y": 600.0,
            "caked_x": 100.1,
            "caked_y": 100.1,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "simulated_detector_x": 500.0,
            "simulated_detector_y": 600.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_uses_current_view_display_to_break_detector_hint_native_tie() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
            "caked_x": 100.1,
            "caked_y": 100.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
            "caked_x": 200.1,
            "caked_y": 200.1,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_uses_current_view_display_to_break_native_hint_tie() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
            "caked_x": 100.1,
            "caked_y": 100.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
            "caked_x": 200.1,
            "caked_y": 200.1,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "display_col": 100.0,
            "display_row": 100.0,
            "sim_native_x": 300.0,
            "sim_native_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 0


def test_geometry_manual_resolve_source_entry_index_uses_detector_xy_aliases_when_display_is_stale() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 13.1,
            "sim_row": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 300.2,
            "sim_row": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 300.0,
            "y": 400.0,
            "display_col": 13.0,
            "display_row": 2.0,
            "stale_caked_fields": True,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_uses_saved_xy_when_current_view_display_is_missing() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 13.1,
            "sim_row": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 300.2,
            "sim_row": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 300.0,
            "y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_uses_detector_hints_for_legacy_background_detector_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 13.1,
            "detector_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_uses_detector_hints_for_native_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "background_detector_x": 13.1,
            "background_detector_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "background_detector_x": 300.2,
            "background_detector_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_uses_detector_hints_for_sim_native_candidates() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 13.1,
            "sim_native_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_detector_hints_over_legacy_background_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 13.1,
            "detector_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.0,
            "detector_y": 400.0,
            "background_detector_x": 13.0,
            "background_detector_y": 2.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_prefers_saved_detector_hints_over_saved_xy_detector_display() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "sim_col": 101.0,
            "sim_row": 101.0,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 500.0,
            "sim_row": 500.0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 100.0,
            "y": 100.0,
            "detector_x": 300.0,
            "detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_falls_back_to_legacy_background_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 13.1,
            "detector_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "detector_x": 300.2,
            "detector_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "background_detector_x": 300.0,
            "background_detector_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index == 1


def test_geometry_manual_resolve_source_entry_index_does_not_use_raw_display_clicks_as_native_detector_hints() -> (
    None
):
    candidate_entries = [
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 13.1,
            "sim_native_y": 2.1,
        },
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_native_x": 300.2,
            "sim_native_y": 400.2,
        },
    ]

    resolved_index = mg.geometry_manual_resolve_source_entry_index(
        {
            "label": "target",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "raw_x": 300.0,
            "raw_y": 400.0,
        },
        candidate_entries,
    )

    assert resolved_index is None


def test_current_geometry_manual_match_config_reuses_auto_match_defaults() -> None:
    cfg = mg.current_geometry_manual_match_config(
        {
            "geometry": {
                "auto_match": {
                    "search_radius_px": 17.5,
                    "min_match_prominence_sigma": 3.25,
                }
            }
        }
    )

    assert cfg["search_radius_px"] == 17.5
    assert cfg["min_match_prominence_sigma"] == 3.25
    assert cfg["console_progress"] is False
    assert cfg["relax_on_low_matches"] is False
    assert cfg["require_candidate_ownership"] is True


def test_geometry_manual_choose_group_at_picks_nearest_seed() -> None:
    grouped_candidates = {
        ("q_group", "primary", 1, 0): [
            {"label": "1,0,0", "sim_col": 20.0, "sim_row": 24.0},
            {"label": "-1,0,0", "sim_col": 42.0, "sim_row": 24.0},
        ],
        ("q_group", "primary", 3, 0): [{"label": "2,1,0", "sim_col": 75.0, "sim_row": 24.0}],
    }

    group_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        grouped_candidates,
        19.5,
        23.5,
        window_size_px=50.0,
    )

    assert group_key == ("q_group", "primary", 1, 0)
    assert len(entries) == 2
    assert best_dist < 1.0


def test_geometry_manual_choose_group_at_uses_visible_detector_display_over_raw_provenance() -> (
    None
):
    group_key = ("q_group", "primary", 3, 4)
    visible_seed = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "visible-ray",
        "hkl": (-3, 0, 4),
        "display_col": 150.0,
        "display_row": 75.0,
        "display_frame": "detector_display",
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }

    found_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        {group_key: [visible_seed]},
        150.0,
        75.0,
        window_size_px=10.0,
        use_caked_display=False,
    )

    assert found_key == group_key
    assert best_dist < 1.0
    assert len(entries) == 1
    assert entries[0]["branch_id"] == "-x"
    assert entries[0]["source_branch_index"] == 1
    assert entries[0]["source_reflection_index"] == 17
    assert entries[0]["source_ray_id"] == "visible-ray"


def test_geometry_manual_choose_group_at_honors_detector_display_frame_when_values_match_caked() -> (
    None
):
    group_key = ("q_group", "primary", 3, 4)
    visible_seed = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "same-values-ray",
        "display_col": 13.0,
        "display_row": 24.0,
        "display_frame": "detector_display",
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }

    found_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        {group_key: [visible_seed]},
        13.0,
        24.0,
        window_size_px=10.0,
        use_caked_display=False,
    )

    assert found_key == group_key
    assert best_dist < 1.0
    assert entries[0]["source_ray_id"] == "same-values-ray"


def test_geometry_manual_choose_group_at_ignores_peaks_outside_50px_window() -> None:
    group_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        {("q_group", "primary", 1, 0): [{"label": "1,0,0", "sim_col": 80.0, "sim_row": 80.0}]},
        20.0,
        20.0,
        window_size_px=50.0,
    )

    assert group_key is None
    assert entries == []
    assert np.isnan(best_dist)


def test_geometry_manual_zoom_bounds_returns_clamped_100px_window() -> None:
    assert mg.geometry_manual_zoom_bounds(150.0, 80.0, (200, 300), window_size_px=100.0) == (
        100.0,
        200.0,
        30.0,
        130.0,
    )
    assert mg.geometry_manual_zoom_bounds(12.0, 15.0, (200, 300), window_size_px=100.0) == (
        0.0,
        100.0,
        0.0,
        100.0,
    )


def test_geometry_manual_anchor_axis_limits_preserves_click_fraction() -> None:
    assert mg.geometry_manual_anchor_axis_limits(150.0, 100.0, 0.25, 0.0, 300.0) == (
        125.0,
        225.0,
    )
    assert mg.geometry_manual_anchor_axis_limits(80.0, -100.0, 0.75, 0.0, 200.0) == (
        155.0,
        55.0,
    )
    assert mg.geometry_manual_anchor_axis_limits(12.0, 100.0, 0.2, 0.0, 300.0) == (
        0.0,
        100.0,
    )


def test_geometry_manual_group_target_count_uses_single_bg_peak_for_00l() -> None:
    assert (
        mg.geometry_manual_group_target_count(
            ("q_group", "primary", 0, 3),
            [{"hkl": (0, 0, 3), "label": "0,0,3"}, {"hkl": (0, 0, 3), "label": "0,0,3"}],
        )
        == 1
    )
    assert (
        mg.geometry_manual_group_target_count(
            ("q_group", "primary", 1, 2),
            [{"hkl": (1, 0, 2), "label": "1,0,2"}, {"hkl": (-1, 0, 2), "label": "-1,0,2"}],
        )
        == 2
    )


def test_geometry_manual_prioritize_candidate_entries_moves_preferred_entry_first() -> None:
    prioritized = mg.geometry_manual_prioritize_candidate_entries(
        [
            {
                "label": "left",
                "source_table_index": 1,
                "source_row_index": 10,
                "sim_col": 10.0,
                "sim_row": 10.0,
            },
            {
                "label": "center",
                "source_table_index": 1,
                "source_row_index": 20,
                "sim_col": 20.0,
                "sim_row": 20.0,
            },
            {
                "label": "right",
                "source_table_index": 1,
                "source_row_index": 35,
                "sim_col": 35.0,
                "sim_row": 35.0,
            },
        ],
        {
            "label": "center-tag",
            "source_table_index": 1,
            "source_row_index": 20,
        },
    )

    assert [entry["label"] for entry in prioritized] == ["center", "left", "right"]


def test_geometry_manual_nearest_candidate_to_point_selects_closest_simulated_peak() -> None:
    candidate, dist = mg.geometry_manual_nearest_candidate_to_point(
        28.0,
        15.5,
        [
            {"label": "left", "sim_col": 12.0, "sim_row": 15.0},
            {"label": "right", "sim_col": 30.0, "sim_row": 16.0},
        ],
    )

    assert isinstance(candidate, dict)
    assert candidate["label"] == "right"
    assert dist < 3.0


def test_geometry_manual_candidate_helpers_prefer_caked_coords_in_caked_view() -> None:
    caked_near = {
        "label": "caked-near",
        "caked_x": 13.1,
        "caked_y": 2.1,
        "sim_col": 500.0,
        "sim_row": 600.0,
    }
    detector_near = {
        "label": "detector-near",
        "caked_x": 30.0,
        "caked_y": 40.0,
        "sim_col": 13.2,
        "sim_row": 2.2,
    }

    caked_near_dist = mg.geometry_manual_candidate_distance_to_point(
        13.0,
        2.0,
        caked_near,
        use_caked_display=True,
    )
    detector_near_dist = mg.geometry_manual_candidate_distance_to_point(
        13.0,
        2.0,
        detector_near,
        use_caked_display=True,
    )
    candidate, dist = mg.geometry_manual_nearest_candidate_to_point(
        13.0,
        2.0,
        [caked_near, detector_near],
        use_caked_display=True,
    )

    assert caked_near_dist < 1.0
    assert detector_near_dist > 10.0
    assert candidate is not None
    assert candidate["label"] == "caked-near"
    assert abs(dist - caked_near_dist) < 1.0e-9


def test_geometry_manual_choose_group_at_does_not_match_detector_pixels_in_caked_view() -> None:
    group_key, entries, dist = mg.geometry_manual_choose_group_at(
        {
            ("q_group", "primary", 1, 0): [
                {
                    "label": "detector-only",
                    "sim_col": 13.0,
                    "sim_row": 2.0,
                    "display_col": 13.0,
                    "display_row": 2.0,
                }
            ]
        },
        13.0,
        2.0,
        window_size_px=4.0,
        use_caked_display=True,
    )

    assert group_key is None
    assert entries == []
    assert np.isnan(dist)


def test_geometry_manual_pair_entry_from_candidate_preserves_caked_coords() -> None:
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "source_table_index": 3,
            "source_row_index": 8,
            "source_peak_index": 2,
            "qr": 1.234567890123,
            "qz": -2.345678901234,
        },
        120.0,
        240.0,
        group_key=("q_group", "primary", 1, 2),
        detector_col=119.5,
        detector_row=239.5,
        raw_col=118.5,
        raw_row=239.0,
        caked_col=13.2,
        caked_row=-7.4,
        raw_caked_col=13.0,
        raw_caked_row=-7.0,
        placement_error_px=1.7,
        sigma_px=1.9,
    )

    assert entry is not None
    assert entry["x"] == 120.0
    assert entry["y"] == 240.0
    assert entry["detector_x"] == 119.5
    assert entry["detector_y"] == 239.5
    assert entry["caked_x"] == 13.2
    assert entry["caked_y"] == -7.4
    assert entry["raw_caked_x"] == 13.0
    assert entry["raw_caked_y"] == -7.0
    assert entry["source_branch_index"] == 0
    assert entry["source_peak_index"] == 0
    assert entry["qr"] == 1.234567890123
    assert entry["qz"] == -2.345678901234


def test_geometry_manual_pair_entry_from_candidate_preserves_simulated_anchor_separately_from_caked_background() -> (
    None
):
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "-1,0,5",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "display_col": 30.25,
            "display_row": -57.5,
            "sim_col_raw": 190.0,
            "sim_row_raw": 96.0,
            "native_col": 189.5,
            "native_row": 95.5,
            "caked_x": 30.25,
            "caked_y": -57.5,
        },
        102.5,
        213.2,
        group_key=("q_group", "primary", 1, 5),
        detector_col=101.5,
        detector_row=211.2,
        raw_col=102.0,
        raw_row=213.0,
        caked_col=13.2,
        caked_row=2.5,
        raw_caked_col=13.0,
        raw_caked_row=2.0,
    )

    assert entry is not None
    assert entry["x"] == 102.5
    assert entry["y"] == 213.2
    assert entry["caked_x"] == 13.2
    assert entry["caked_y"] == 2.5
    assert entry["refined_sim_x"] == 190.0
    assert entry["refined_sim_y"] == 96.0
    assert entry["refined_sim_native_x"] == 189.5
    assert entry["refined_sim_native_y"] == 95.5
    assert "refined_sim_caked_x" not in entry
    assert "refined_sim_caked_y" not in entry


def test_geometry_manual_pair_json_roundtrip_preserves_selection_provenance() -> None:
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "source_table_index": 3,
            "source_row_index": 8,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "branch_id": "+x",
            "branch_source": "generated",
            "best_sample_index": 4,
            "mosaic_weight": 0.75,
            "mosaic_top_rank_key": (0, 0, -0.75, 1.0e300, 0.0, 0.0, -5.0, 2),
            "selection_reason": "mosaic_top_per_branch",
        },
        120.0,
        240.0,
        group_key=("q_group", "primary", 1, 2),
    )
    assert entry is not None

    serialized = mg.geometry_manual_pair_entry_to_jsonable(entry)
    restored = mg.geometry_manual_pair_entry_from_jsonable(serialized)

    assert serialized["branch_id"] == "+x"
    assert serialized["branch_source"] == "generated"
    assert serialized["best_sample_index"] == 4
    assert serialized["mosaic_weight"] == 0.75
    assert serialized["selection_reason"] == "mosaic_top_per_branch"
    assert serialized["mosaic_top_rank_key"] == [
        0,
        0,
        -0.75,
        1.0e300,
        0.0,
        0.0,
        -5.0,
        2,
    ]
    assert restored["branch_id"] == "+x"
    assert restored["branch_source"] == "generated"
    assert restored["best_sample_index"] == 4
    assert restored["mosaic_weight"] == 0.75
    assert restored["mosaic_top_rank_key"] == tuple(serialized["mosaic_top_rank_key"])


def test_geometry_manual_pair_entry_sanitizes_wide_provenance_values() -> None:
    raw_payload = object()
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "source_table_index": 3,
            "source_row_index": 8,
            "source_branch_index": 1,
            "source_reflection_index": 9,
            "source_reflection_key": ("full", 9),
            "source_ray_id": "ray-9",
            "source_payload": raw_payload,
            "ray_vector": np.asarray([1.0, 2.0, 3.0], dtype=float),
            "reflection_payload": {"array": np.asarray([4, 5], dtype=np.int64)},
            "branch_id": "-x",
            "selection_reason": "mosaic_top_per_q_group",
        },
        120.0,
        240.0,
        group_key=("q_group", "primary", 1, 2),
    )

    assert entry is not None
    assert entry["source_reflection_key"] == ("full", 9)
    assert entry["source_ray_id"] == "ray-9"
    assert entry["ray_vector"] == (1.0, 2.0, 3.0)
    assert entry["reflection_payload"] == {"array": (4, 5)}
    assert isinstance(entry["source_payload"], str)
    assert entry["source_payload"] != raw_payload


def test_caked_manual_pair_redraws_in_detector_view_from_saved_simulated_anchor_without_live_lookup() -> (
    None
):
    saved_pair = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "-1,0,5",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "display_col": 30.25,
            "display_row": -57.5,
            "sim_col_raw": 190.0,
            "sim_row_raw": 96.0,
            "native_col": 189.5,
            "native_row": 95.5,
            "caked_x": 30.25,
            "caked_y": -57.5,
        },
        102.5,
        213.2,
        group_key=("q_group", "primary", 1, 5),
        detector_col=101.5,
        detector_row=211.2,
        raw_col=102.0,
        raw_row=213.0,
        caked_col=13.2,
        caked_row=2.5,
        raw_caked_col=13.0,
        raw_caked_row=2.0,
    )
    assert saved_pair is not None

    detector_measured, detector_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )
    caked_measured, caked_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert detector_measured[0]["caked_x"] == 13.2
    assert caked_measured[0]["x"] == 102.5
    assert detector_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (102.5, 213.2),
            "sim_display": (190.0, 96.0),
        }
    ]
    assert caked_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (13.2, 2.5),
            "sim_display_unresolved": True,
        }
    ]


def test_geometry_manual_pair_entry_from_candidate_rejects_caked_alias_as_detector_anchor() -> None:
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {
            "label": "-1,0,5",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "display_col": 30.25,
            "display_row": -57.5,
            "sim_col": 30.25,
            "sim_row": -57.5,
            "caked_x": 30.25,
            "caked_y": -57.5,
        },
        102.5,
        213.2,
        group_key=("q_group", "primary", 1, 5),
        detector_col=101.5,
        detector_row=211.2,
        caked_col=13.2,
        caked_row=2.5,
    )

    assert entry is not None
    assert "refined_sim_x" not in entry
    assert "refined_sim_y" not in entry
    assert "refined_sim_caked_x" not in entry
    assert "refined_sim_caked_y" not in entry


def test_make_runtime_geometry_manual_callbacks_render_current_pairs_uses_live_state() -> None:
    events: list[tuple[object, ...]] = []
    status_texts: list[str] = []

    callbacks = mg.make_runtime_geometry_manual_callbacks(
        background_visible=lambda: True,
        current_background_index=lambda: 2,
        current_background_image=lambda: np.ones((4, 4), dtype=float),
        pick_session=lambda: None,
        build_initial_pairs_display=lambda index, *, prefer_cache: (
            [{"measured": int(index)}],
            [{"saved": int(index)}],
        ),
        session_initial_pairs_display=lambda: [{"pending": True}],
        clear_geometry_pick_artists=lambda *args, **kwargs: events.append(("clear", args, kwargs)),
        draw_initial_geometry_pairs_overlay=lambda pairs, *, max_display_markers: events.append(
            ("draw", list(pairs), int(max_display_markers))
        ),
        update_button_label=lambda: events.append(("button",)),
        set_background_file_status_text=lambda: events.append(("background-status",)),
        pair_group_count=lambda index: 1,
        set_status_text=lambda text: status_texts.append(str(text)),
        get_cache_data=lambda **kwargs: {},
        set_pairs_for_index=lambda index, entries: list(entries or []),
        pairs_for_index=lambda index: [{"pair": int(index)}],
        set_pick_session=lambda session: events.append(("set-session", dict(session))),
        restore_view=lambda **kwargs: events.append(("restore", kwargs)),
        clear_preview_artists=lambda **kwargs: events.append(("clear-preview", kwargs)),
        use_caked_space=False,
        pick_search_window_px=50.0,
        refine_preview_point=lambda *args, **kwargs: (0.0, 0.0),
        remaining_candidates=lambda: [],
        preview_due=lambda col, row: True,
        show_preview=lambda *args, **kwargs: events.append(("show-preview", args, kwargs)),
    )

    assert callbacks.render_current_pairs(update_status=True) is True
    assert events == [
        ("draw", [{"saved": 2}, {"pending": True}], 2),
        ("button",),
        ("background-status",),
    ]
    assert status_texts == ["Current background has 1 saved manual points across 1 Qr/Qz groups."]


def test_make_runtime_geometry_manual_callbacks_refresh_pick_session_before_delegate() -> None:
    events: list[tuple[object, ...]] = []

    def _refresh_pick_session(session: dict[str, object] | None) -> dict[str, object]:
        events.append(("refresh", dict(session or {})))
        refreshed = dict(session or {})
        refreshed["mode"] = "refreshed"
        return refreshed

    def _fake_place(col: float, row: float, **kwargs):
        events.append(
            (
                "place",
                float(col),
                float(row),
                dict(kwargs["pick_session"]),
                callable(kwargs.get("refine_saved_pair_entry_fn")),
            )
        )
        return False, {}

    original_place = mg.geometry_manual_place_selection_at
    try:
        mg.geometry_manual_place_selection_at = _fake_place
        callbacks = mg.make_runtime_geometry_manual_callbacks(
            background_visible=lambda: True,
            current_background_index=lambda: 0,
            current_background_image=lambda: "bg-image",
            pick_session=lambda: {"mode": "start"},
            build_initial_pairs_display=lambda index, *, prefer_cache: ([], []),
            session_initial_pairs_display=lambda: [],
            clear_geometry_pick_artists=lambda *args, **kwargs: None,
            draw_initial_geometry_pairs_overlay=lambda pairs, *, max_display_markers: None,
            update_button_label=lambda: None,
            set_background_file_status_text=lambda: None,
            pair_group_count=lambda index: 0,
            set_status_text=None,
            get_cache_data=lambda **kwargs: {},
            set_pairs_for_index=lambda index, entries: list(entries or []),
            pairs_for_index=lambda index: [],
            set_pick_session=lambda session: None,
            restore_view=lambda **kwargs: None,
            clear_preview_artists=lambda **kwargs: None,
            use_caked_space=False,
            pick_search_window_px=25.0,
            refine_preview_point=lambda *args, **kwargs: (0.0, 0.0),
            refine_saved_pair_entry=lambda entry, candidate=None: dict(entry),
            remaining_candidates=lambda: [],
            preview_due=lambda col, row: False,
            refresh_pick_session=_refresh_pick_session,
        )

        assert callbacks.place_selection_at(3.0, 4.0) is False
    finally:
        mg.geometry_manual_place_selection_at = original_place

    assert events == [
        ("refresh", {"mode": "start"}),
        ("place", 3.0, 4.0, {"mode": "refreshed"}, True),
    ]


def test_make_runtime_geometry_manual_callbacks_delegate_toggle_preview_and_cancel(
    monkeypatch,
) -> None:
    events: list[tuple[object, ...]] = []
    status_texts: list[str] = []
    pick_session_state: dict[str, object] = {"value": {"mode": "start"}}

    def _set_pick_session(session: dict[str, object]) -> None:
        pick_session_state["value"] = dict(session)
        events.append(("set-session", dict(session)))

    def _fake_toggle(col: float, row: float, **kwargs):
        events.append(
            (
                "toggle",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["use_caked_space"],
                dict(kwargs["pick_session"]),
            )
        )
        kwargs["set_pick_session_fn"]({"mode": "toggle"})
        return True, {"ignored": True}, True

    def _fake_place(col: float, row: float, **kwargs):
        events.append(
            (
                "place",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["use_caked_space"],
                dict(kwargs["pick_session"]),
            )
        )
        kwargs["set_pick_session_fn"]({"mode": "place"})
        return True, {"ignored": True}

    def _fake_preview_state(col: float, row: float, **kwargs):
        events.append(
            (
                "preview-state",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["force"],
                list(kwargs["remaining_candidates"]),
                kwargs["use_caked_space"],
            )
        )
        return {
            "raw_col": 5.0,
            "raw_row": 6.0,
            "refined_col": 7.5,
            "refined_row": 8.5,
            "delta": 1.25,
            "sigma_px": 1.46,
            "preview_color": "#2ecc71",
            "message": "preview ready",
        }

    def _fake_cancel(pick_session, **kwargs):
        events.append(
            (
                "cancel",
                dict(pick_session),
                kwargs["current_background_index"],
                kwargs["restore_view"],
                kwargs["redraw"],
                kwargs["message"],
            )
        )
        return {"mode": "cancel"}

    monkeypatch.setattr(mg, "geometry_manual_toggle_selection_at", _fake_toggle)
    monkeypatch.setattr(mg, "geometry_manual_place_selection_at", _fake_place)
    monkeypatch.setattr(mg, "geometry_manual_pick_preview_state", _fake_preview_state)
    monkeypatch.setattr(mg, "cancel_geometry_manual_pick_session", _fake_cancel)

    callbacks = mg.make_runtime_geometry_manual_callbacks(
        background_visible=lambda: True,
        current_background_index=lambda: 2,
        current_background_image=lambda: "bg-image",
        pick_session=lambda: pick_session_state["value"],
        build_initial_pairs_display=lambda index, *, prefer_cache: ([], []),
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda *args, **kwargs: None,
        draw_initial_geometry_pairs_overlay=lambda pairs, *, max_display_markers: None,
        update_button_label=lambda: events.append(("button",)),
        set_background_file_status_text=lambda: events.append(("background-status",)),
        pair_group_count=lambda index: 0,
        set_status_text=lambda text: status_texts.append(str(text)),
        get_cache_data=lambda **kwargs: {"cache": True},
        set_pairs_for_index=lambda index, entries: list(entries or []),
        pairs_for_index=lambda index: [],
        set_pick_session=_set_pick_session,
        restore_view=lambda **kwargs: events.append(("restore-view", kwargs)),
        clear_preview_artists=lambda **kwargs: events.append(("clear-preview", kwargs)),
        push_undo_state=lambda: events.append(("push-undo",)),
        listed_q_group_entries=lambda: [{"key": ("q", 1)}],
        format_q_group_line=lambda entry: "Q1",
        use_caked_space=lambda: True,
        pick_search_window_px=25.0,
        set_suppress_drag_press_once=lambda enabled: events.append(("suppress", enabled)),
        sync_peak_selection_state=lambda: events.append(("sync",)),
        refine_preview_point=lambda *args, **kwargs: (11.0, 12.0),
        remaining_candidates=lambda: [{"label": "cand"}],
        preview_due=lambda col, row: True,
        nearest_candidate_to_point=lambda col, row, candidates: (
            {"label": "cand"},
            1.5,
        ),
        caked_angles_to_background_display_coords=lambda col, row: (col + 100.0, row + 200.0),
        show_preview=lambda *args, **kwargs: events.append(("show-preview", args, kwargs)),
    )

    assert callbacks.toggle_selection_at(10.0, 20.0) is True
    assert callbacks.place_selection_at(30.0, 40.0) is True
    callbacks.update_pick_preview(5.0, 6.0, force=True)
    callbacks.cancel_pick_session(restore_view=False, redraw=False, message="bye")

    assert pick_session_state["value"] == {"mode": "cancel"}
    assert status_texts == ["preview ready"]
    assert events == [
        ("toggle", 10.0, 20.0, 2, True, {"mode": "start"}),
        ("set-session", {"mode": "toggle"}),
        ("suppress", True),
        ("sync",),
        ("place", 30.0, 40.0, 2, True, {"mode": "toggle"}),
        ("set-session", {"mode": "place"}),
        ("preview-state", 5.0, 6.0, 2, True, [{"label": "cand"}], True),
        (
            "show-preview",
            (5.0, 6.0, 7.5, 8.5),
            {
                "delta_px": 1.25,
                "sigma_px": 1.46,
                "preview_color": "#2ecc71",
            },
        ),
        ("cancel", {"mode": "place"}, 2, False, False, "bye"),
        ("set-session", {"mode": "cancel"}),
    ]


def test_ensure_geometry_fit_caked_view_switches_and_refreshes_immediately() -> None:
    calls: list[str] = []

    class _DummyRoot:
        def __init__(self) -> None:
            self.canceled: list[object] = []

        def after_cancel(self, token) -> None:
            self.canceled.append(token)

    update_pending, integration_update_pending = mg.ensure_geometry_fit_caked_view(
        show_caked_2d_var=_DummyVar(False),
        pick_uses_caked_space=lambda: False,
        toggle_caked_2d=lambda: calls.append("toggle"),
        do_update=lambda: calls.append("update"),
        schedule_update=lambda: calls.append("schedule"),
        root=_DummyRoot(),
        update_pending="update-token",
        integration_update_pending="range-token",
        update_running=False,
    )

    assert calls == ["toggle", "update"]
    assert update_pending is None
    assert integration_update_pending is None


def test_native_detector_coords_to_caked_display_coords_uses_bundle_frame_adapter(
    monkeypatch,
) -> None:
    bundle = object()
    projector_calls = []

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            projector_calls.append((live_bundle, float(col), float(row))) or (12.6, 169.2)
        ),
    )

    result = mg.native_detector_coords_to_caked_display_coords(
        0.9,
        0.1,
        ai=_ai_with_live_bundle(bundle),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
        caked_radial_values=np.array([10.0, 12.5, 15.0], dtype=float),
        caked_azimuth_values=np.array([-170.0, -10.0, 20.0, 170.0], dtype=float),
        native_detector_coords_to_bundle_detector_coords=lambda col, row: (
            float(col) + 1.0,
            float(row) + 2.0,
        ),
    )

    assert result == (12.6, 169.2)
    assert projector_calls == [(bundle, 1.9, 2.1)]


def test_native_detector_coords_to_caked_display_coords_uses_explicit_transform_bundle(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle()
    projector_calls = []

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            projector_calls.append((live_bundle, float(col), float(row))) or (14.2, -11.5)
        ),
    )

    result = mg.native_detector_coords_to_caked_display_coords(
        4.0,
        5.0,
        ai=object(),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        transform_bundle=bundle,
        wrap_phi_range=_wrap_phi_range,
    )

    assert result == (14.2, -11.5)
    assert projector_calls == [(bundle, 4.0, 5.0)]


def test_native_detector_coords_to_caked_display_coords_returns_none_without_live_bundle() -> None:
    result = mg.native_detector_coords_to_caked_display_coords(
        5.0,
        7.0,
        ai=object(),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
    )

    assert result is None


def test_caked_angles_to_background_display_coords_returns_none_without_native_background() -> None:
    result = mg.caked_angles_to_background_display_coords(
        12.0,
        30.0,
        ai=object(),
        native_background=None,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
            "analytic inverse fallback should not be used"
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
    )

    assert result == (None, None)


def test_caked_angles_to_background_display_coords_returns_none_when_inverse_lut_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mg, "_caked_point_to_detector_pixel", lambda *_args, **_kwargs: (None, None)
    )

    result = mg.caked_angles_to_background_display_coords(
        12.0,
        30.0,
        ai=object(),
        native_background=np.ones((8, 8), dtype=float),
        caked_radial_values=np.array([10.0, 12.0, 14.0], dtype=float),
        caked_azimuth_values=np.array([-30.0, 0.0, 30.0], dtype=float),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        scattering_angles_to_detector_pixel=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
    )

    assert result == (None, None)


def test_caked_angles_to_background_display_coords_applies_backend_inverse_to_lut_result(
    monkeypatch,
) -> None:
    inverse_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda *_args, **_kwargs: (0.0, 1.0),
    )

    result = mg.caked_angles_to_background_display_coords(
        12.0,
        30.0,
        ai=object(),
        native_background=np.ones((8, 8), dtype=float),
        caked_radial_values=np.array([10.0, 12.0, 14.0], dtype=float),
        caked_azimuth_values=np.array([-30.0, 0.0, 30.0], dtype=float),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        scattering_angles_to_detector_pixel=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        backend_detector_coords_to_native_detector_coords=lambda col, row: (
            inverse_calls.append((float(col), float(row))) or (101.0, 202.0)
        ),
    )

    assert result == (101.0, 202.0)
    assert inverse_calls == [(0.0, 1.0)]


def test_caked_angles_to_background_display_coords_prefers_bundle_frame_display_adapter(
    monkeypatch,
) -> None:
    backend_inverse_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda *_args, **_kwargs: (7.0, 8.0),
    )

    result = mg.caked_angles_to_background_display_coords(
        12.0,
        30.0,
        ai=object(),
        native_background=np.ones((8, 8), dtype=float),
        caked_radial_values=np.array([10.0, 12.0, 14.0], dtype=float),
        caked_azimuth_values=np.array([-30.0, 0.0, 30.0], dtype=float),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        scattering_angles_to_detector_pixel=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        backend_detector_coords_to_native_detector_coords=lambda col, row: (
            backend_inverse_calls.append((float(col), float(row))) or (101.0, 202.0)
        ),
        bundle_detector_coords_to_background_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
    )

    assert result == (107.0, 208.0)
    assert backend_inverse_calls == []


def test_geometry_manual_apply_refined_simulated_override_keeps_detector_coords_in_caked_view() -> (
    None
):
    refined = mg.geometry_manual_apply_refined_simulated_override(
        {
            "refined_sim_x": 30.0,
            "refined_sim_y": 40.0,
            "refined_sim_native_x": 5.0,
            "refined_sim_native_y": 6.0,
            "refined_sim_caked_x": 13.0,
            "refined_sim_caked_y": 2.0,
        },
        {
            "sim_col": 3.0,
            "sim_row": 4.0,
            "display_col": 3.0,
            "display_row": 4.0,
            "caked_x": 11.0,
            "caked_y": 12.0,
        },
        prefer_caked_display=True,
    )

    assert refined is not None
    assert refined["sim_col"] == 30.0
    assert refined["sim_row"] == 40.0
    assert refined["sim_col_raw"] == 30.0
    assert refined["sim_row_raw"] == 40.0
    assert refined["display_col"] == 13.0
    assert refined["display_row"] == 2.0
    assert refined["native_col"] == 5.0
    assert refined["native_row"] == 6.0
    assert refined["sim_native_x"] == 5.0
    assert refined["sim_native_y"] == 6.0
    assert refined["caked_x"] == 13.0
    assert refined["caked_y"] == 2.0
    assert refined["raw_caked_x"] == 13.0
    assert refined["raw_caked_y"] == 2.0
    assert refined["two_theta_deg"] == 13.0
    assert refined["phi_deg"] == 2.0


def test_geometry_manual_apply_refined_simulated_override_keeps_detector_view_display_when_not_caked() -> (
    None
):
    refined = mg.geometry_manual_apply_refined_simulated_override(
        {
            "refined_sim_x": 30.0,
            "refined_sim_y": 40.0,
            "refined_sim_caked_x": 13.0,
            "refined_sim_caked_y": 2.0,
        },
        {
            "sim_col": 3.0,
            "sim_row": 4.0,
            "display_col": 3.0,
            "display_row": 4.0,
            "native_col": 99.0,
            "native_row": 98.0,
            "sim_native_x": 97.0,
            "sim_native_y": 96.0,
            "caked_x": 11.0,
            "caked_y": 12.0,
        },
        prefer_caked_display=False,
    )

    assert refined is not None
    assert refined["sim_col"] == 30.0
    assert refined["sim_row"] == 40.0
    assert refined["display_col"] == 30.0
    assert refined["display_row"] == 40.0
    assert refined["x"] == 30.0
    assert refined["refined_sim_x"] == 30.0
    assert refined["refined_sim_y"] == 40.0
    assert refined["refined_sim_caked_x"] == 13.0
    assert refined["refined_sim_caked_y"] == 2.0
    assert refined["y"] == 40.0
    assert refined["caked_x"] == 13.0
    assert refined["caked_y"] == 2.0
    assert refined["raw_caked_x"] == 13.0
    assert refined["raw_caked_y"] == 2.0
    assert refined["two_theta_deg"] == 13.0
    assert refined["phi_deg"] == 2.0
    assert "native_col" not in refined
    assert "native_row" not in refined
    assert "sim_native_x" not in refined
    assert "sim_native_y" not in refined


def test_detector_to_caked_to_detector_round_trips_through_same_transform_bundle(
    monkeypatch,
) -> None:
    image, bundle = _real_transform_bundle()
    conflicting_ai = _ai_with_live_bundle(
        _dummy_transform_bundle(detector_shape=bundle.detector_shape)
    )
    forward_calls: list[object] = []
    inverse_calls: list[object] = []
    real_forward = mg._detector_pixel_to_caked_bin
    real_inverse = mg._caked_point_to_detector_pixel

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            forward_calls.append(live_bundle) or real_forward(live_bundle, col, row)
        ),
    )

    def _record_inverse(ai, detector_shape, radial_deg, azimuth_deg, two_theta, phi, **kwargs):
        inverse_calls.append(kwargs.get("transform_bundle"))
        return real_inverse(
            ai,
            detector_shape,
            radial_deg,
            azimuth_deg,
            two_theta,
            phi,
            **kwargs,
        )

    monkeypatch.setattr(mg, "_caked_point_to_detector_pixel", _record_inverse)

    caked_point = mg.native_detector_coords_to_caked_display_coords(
        1.0,
        1.0,
        ai=conflicting_ai,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic fallback should not be used"
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
        transform_bundle=bundle,
        caked_radial_values=bundle.radial_deg,
        caked_azimuth_values=bundle.gui_azimuth_deg,
    )
    assert caked_point is not None

    detector_point = mg.caked_angles_to_background_display_coords(
        float(caked_point[0]),
        float(caked_point[1]),
        ai=conflicting_ai,
        native_background=image,
        caked_radial_values=bundle.radial_deg,
        caked_azimuth_values=bundle.gui_azimuth_deg,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
            "analytic inverse fallback should not be used"
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        transform_bundle=bundle,
    )

    np.testing.assert_allclose(detector_point, (1.0, 1.0), atol=1.0e-6, rtol=0.0)
    assert forward_calls == [bundle]
    assert inverse_calls == [bundle]


def test_caked_to_detector_to_caked_round_trips_through_same_transform_bundle(
    monkeypatch,
) -> None:
    image, bundle = _real_transform_bundle()
    conflicting_ai = _ai_with_live_bundle(
        _dummy_transform_bundle(detector_shape=bundle.detector_shape)
    )
    forward_calls: list[object] = []
    inverse_calls: list[object] = []
    real_forward = mg._detector_pixel_to_caked_bin
    real_inverse = mg._caked_point_to_detector_pixel
    seed_caked_point = real_forward(bundle, 3.0, 1.0)

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            forward_calls.append(live_bundle) or real_forward(live_bundle, col, row)
        ),
    )

    def _record_inverse(ai, detector_shape, radial_deg, azimuth_deg, two_theta, phi, **kwargs):
        inverse_calls.append(kwargs.get("transform_bundle"))
        return real_inverse(
            ai,
            detector_shape,
            radial_deg,
            azimuth_deg,
            two_theta,
            phi,
            **kwargs,
        )

    monkeypatch.setattr(mg, "_caked_point_to_detector_pixel", _record_inverse)

    detector_point = mg.caked_angles_to_background_display_coords(
        float(seed_caked_point[0]),
        float(seed_caked_point[1]),
        ai=conflicting_ai,
        native_background=image,
        caked_radial_values=bundle.radial_deg,
        caked_azimuth_values=bundle.gui_azimuth_deg,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
            "analytic inverse fallback should not be used"
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        transform_bundle=bundle,
    )
    caked_point = mg.native_detector_coords_to_caked_display_coords(
        float(detector_point[0]),
        float(detector_point[1]),
        ai=conflicting_ai,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic fallback should not be used"
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
        transform_bundle=bundle,
        caked_radial_values=bundle.radial_deg,
        caked_azimuth_values=bundle.gui_azimuth_deg,
    )

    np.testing.assert_allclose(
        caked_point,
        seed_caked_point,
        atol=1.0e-6,
        rtol=0.0,
    )
    assert inverse_calls == [bundle]
    assert forward_calls == [bundle]


def test_geometry_manual_preview_due_throttles_small_motion() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [{"label": "1,0,0"}],
        "background_index": 0,
        "preview_last_t": 0.0,
        "preview_last_xy": None,
    }
    times = iter([1.0, 1.01, 1.05])

    assert mg.geometry_manual_preview_due(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )
    assert not mg.geometry_manual_preview_due(
        10.2,
        20.1,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )
    assert mg.geometry_manual_preview_due(
        10.2,
        20.1,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )


def test_should_collect_hit_tables_when_manual_geometry_overlay_is_visible() -> None:
    assert mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=2,
        skip_preview_once=False,
        manual_pick_armed=False,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda idx: [{"hkl": (1, 0, 2)}] if int(idx) == 2 else [],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_should_not_collect_hit_tables_for_hidden_manual_geometry_overlay() -> None:
    assert not mg.should_collect_hit_tables_for_update(
        background_visible=False,
        current_background_index=0,
        skip_preview_once=False,
        manual_pick_armed=False,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda _idx: [{"hkl": (1, 0, 2)}],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_should_not_collect_hit_tables_when_preview_skip_once_is_requested() -> None:
    assert not mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=2,
        skip_preview_once=True,
        manual_pick_armed=False,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: True,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda idx: [{"hkl": (1, 0, 2)}] if int(idx) == 2 else [],
        geometry_manual_pick_session_active=lambda: True,
    )


def test_should_collect_hit_tables_when_manual_pick_is_armed() -> None:
    assert mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=1,
        skip_preview_once=False,
        manual_pick_armed=True,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda _idx: [],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_geometry_manual_pair_json_round_trip_preserves_hkl_and_group_key() -> None:
    serialized = mg.geometry_manual_pair_entry_to_jsonable(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "x": 10.5,
            "y": 12.25,
            "detector_x": 10.0,
            "detector_y": 12.0,
            "background_detector_x": 9.5,
            "background_detector_y": 11.5,
            "q_group_key": ("q_group", "primary", 1.0, 2),
            "source_table_index": 4,
            "source_row_index": 7,
            "source_peak_index": 3,
            "raw_x": 9.0,
            "raw_y": 11.0,
            "background_two_theta_deg": 23.0,
            "background_phi_deg": -17.5,
            "caked_x": 23.0,
            "caked_y": -17.5,
            "raw_caked_x": 22.75,
            "raw_caked_y": -17.25,
            "placement_error_px": 1.25,
            "sigma_px": 1.46,
            "stale_caked_fields": True,
        },
        sigma_floor_px=0.75,
    )

    assert serialized["hkl"] == [1, 0, 2]
    assert serialized["q_group_key"] == ["q_group", "primary", 1.0, 2]
    assert serialized["background_two_theta_deg"] == 23.0
    assert serialized["background_phi_deg"] == -17.5
    assert serialized["caked_x"] == 23.0
    assert serialized["caked_y"] == -17.5
    assert serialized["raw_caked_x"] == 22.75
    assert serialized["raw_caked_y"] == -17.25
    assert serialized["placement_error_px"] == 1.25
    assert serialized["sigma_px"] == 1.46
    assert "stale_caked_fields" not in serialized

    restored = mg.geometry_manual_pair_entry_from_jsonable(serialized, sigma_floor_px=0.75)

    assert restored["hkl"] == (1, 0, 2)
    assert restored["q_group_key"] == ("q_group", "primary", 1.0, 2)
    assert restored["detector_x"] == 10.0
    assert restored["detector_y"] == 12.0
    assert restored["background_detector_x"] == 9.5
    assert restored["background_detector_y"] == 11.5
    assert restored["source_table_index"] == 4
    assert restored["source_row_index"] == 7
    assert restored["source_branch_index"] == 0
    assert restored["source_peak_index"] == 0
    assert restored["background_two_theta_deg"] == 23.0
    assert restored["background_phi_deg"] == -17.5
    assert restored["raw_x"] == 9.0
    assert restored["raw_y"] == 11.0
    assert restored["caked_x"] == 23.0
    assert restored["caked_y"] == -17.5
    assert restored["raw_caked_x"] == 22.75
    assert restored["raw_caked_y"] == -17.25
    assert restored["placement_error_px"] == 1.25
    assert restored["sigma_px"] == 1.46
    assert "stale_caked_fields" not in restored


def test_geometry_manual_pair_from_jsonable_accepts_legacy_stale_caked_field() -> None:
    restored = mg.geometry_manual_pair_entry_from_jsonable(
        {
            "label": "1,0,2",
            "hkl": [1, 0, 2],
            "x": 10.5,
            "y": 12.25,
            "caked_x": 23.0,
            "caked_y": -17.5,
            "stale_caked_fields": True,
        },
        sigma_floor_px=0.75,
    )

    assert restored is not None
    assert restored["caked_x"] == 23.0
    assert restored["caked_y"] == -17.5
    assert restored["stale_caked_fields"] is True


def test_geometry_manual_pair_from_jsonable_migrates_legacy_source_peak_alias_only_on_load() -> (
    None
):
    raw_entry = {
        "label": "1,0,2",
        "hkl": [1, 0, 2],
        "x": 10.5,
        "y": 12.25,
        "source_peak_index": 1,
    }

    normalized = mg.normalize_geometry_manual_pair_entry(dict(raw_entry))
    restored = mg.geometry_manual_pair_entry_from_jsonable(raw_entry)

    assert normalized is not None
    assert "source_branch_index" not in normalized
    assert normalized["source_peak_index"] == 1
    assert restored is not None
    assert restored["source_branch_index"] == 1
    assert restored["source_peak_index"] == 1


def test_geometry_manual_pair_from_jsonable_uses_legacy_branch_alias_when_phi_is_deadbanded() -> (
    None
):
    raw_entry = {
        "label": "1,0,2",
        "hkl": [1, 0, 2],
        "x": 10.5,
        "y": 12.25,
        "background_two_theta_deg": 23.0,
        "background_phi_deg": 0.0,
        "source_peak_index": 1,
    }

    normalized = mg.normalize_geometry_manual_pair_entry(dict(raw_entry))
    restored = mg.geometry_manual_pair_entry_from_jsonable(raw_entry)

    assert normalized is not None
    assert "source_branch_index" not in normalized
    assert normalized["source_peak_index"] == 1
    assert restored is not None
    assert restored["source_branch_index"] == 1
    assert restored["source_peak_index"] == 1


def test_geometry_manual_pair_from_jsonable_uses_legacy_branch_alias_when_caked_phi_is_deadbanded() -> (
    None
):
    raw_entry = {
        "label": "1,0,2",
        "hkl": [1, 0, 2],
        "x": 10.5,
        "y": 12.25,
        "caked_x": 23.0,
        "caked_y": 0.0,
        "source_peak_index": 1,
    }

    normalized = mg.normalize_geometry_manual_pair_entry(dict(raw_entry))
    restored = mg.geometry_manual_pair_entry_from_jsonable(raw_entry)

    assert normalized is not None
    assert "source_branch_index" not in normalized
    assert normalized["source_peak_index"] == 1
    assert restored is not None
    assert restored["background_two_theta_deg"] == 23.0
    assert restored["background_phi_deg"] == 0.0
    assert restored["source_branch_index"] == 1
    assert restored["source_peak_index"] == 1


def test_refresh_geometry_manual_pair_entry_prefers_native_coords_over_stale_display_aliases() -> (
    None
):
    background_shape = (100, 200)
    native_point = (20.0, 10.0)
    stale_display = (90.0, 20.0)

    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "1,0,0",
            "x": stale_display[0],
            "y": stale_display[1],
            "sim_col": stale_display[0],
            "sim_row": stale_display[1],
            "sim_col_raw": stale_display[0],
            "sim_row_raw": stale_display[1],
            "native_col": native_point[0],
            "native_row": native_point[1],
            "sim_native_x": native_point[0],
            "sim_native_y": native_point[1],
        },
        background_display_shape=background_shape,
        background_display_to_native_detector_coords=lambda col, row: mg._default_rotate_point(
            float(col),
            float(row),
            background_shape,
            1,
        ),
        rotate_point_for_display=mg._default_rotate_point,
        display_rotate_k=-1,
    )

    expected_display = mg._default_rotate_point(
        native_point[0],
        native_point[1],
        background_shape,
        -1,
    )

    assert refreshed is not None
    assert refreshed["detector_x"] == native_point[0]
    assert refreshed["detector_y"] == native_point[1]
    assert refreshed["native_col"] == native_point[0]
    assert refreshed["native_row"] == native_point[1]
    assert refreshed["sim_native_x"] == native_point[0]
    assert refreshed["sim_native_y"] == native_point[1]
    assert refreshed["x"] == expected_display[0]
    assert refreshed["y"] == expected_display[1]
    assert refreshed["display_col"] == expected_display[0]
    assert refreshed["display_row"] == expected_display[1]
    assert refreshed["sim_col"] == expected_display[0]
    assert refreshed["sim_row"] == expected_display[1]
    assert refreshed["sim_col_raw"] == expected_display[0]
    assert refreshed["sim_row_raw"] == expected_display[1]


def test_refresh_geometry_manual_pair_entry_accepts_projected_detector_rows_without_measured_xy() -> (
    None
):
    background_shape = (100, 200)
    native_point = (20.0, 10.0)
    detector_display = mg._default_rotate_point(
        native_point[0],
        native_point[1],
        background_shape,
        -1,
    )

    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "1,0,0",
            "display_col": detector_display[0],
            "display_row": detector_display[1],
            "sim_col": detector_display[0],
            "sim_row": detector_display[1],
            "sim_col_raw": detector_display[0],
            "sim_row_raw": detector_display[1],
            "native_col": native_point[0],
            "native_row": native_point[1],
            "sim_native_x": native_point[0],
            "sim_native_y": native_point[1],
            "caked_x": 35.0,
            "caked_y": -20.0,
            "raw_caked_x": 35.0,
            "raw_caked_y": -20.0,
        },
        background_display_shape=background_shape,
        background_display_to_native_detector_coords=lambda col, row: mg._default_rotate_point(
            float(col),
            float(row),
            background_shape,
            1,
        ),
        rotate_point_for_display=mg._default_rotate_point,
        display_rotate_k=-1,
    )

    assert refreshed is not None
    assert refreshed["detector_x"] == native_point[0]
    assert refreshed["detector_y"] == native_point[1]
    assert refreshed["x"] == detector_display[0]
    assert refreshed["y"] == detector_display[1]


def test_geometry_manual_pairs_export_rows_include_background_metadata() -> None:
    rows = mg.geometry_manual_pairs_export_rows(
        pairs_by_background={1: [{"label": "1,0,0", "x": 2.0, "y": 3.0}]},
        osc_files=["bg_0.osc", "bg_1.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "1,0,0", "x": 2.0, "y": 3.0}] if int(idx) == 1 else []
        ),
    )

    assert rows == [
        {
            "background_index": 1,
            "background_path": "bg_1.osc",
            "background_name": "bg_1.osc",
            "entries": [{"x": 2.0, "y": 3.0, "label": "1,0,0", "hkl": [1, 0, 0]}],
        }
    ]


def test_collect_geometry_manual_pairs_snapshot_records_loaded_backgrounds() -> None:
    snapshot = mg.collect_geometry_manual_pairs_snapshot(
        osc_files=["bg_0.osc", "bg_1.osc"],
        current_background_index=1,
        manual_pair_rows=[{"background_index": 1, "entries": []}],
    )

    assert snapshot == {
        "background_files": ["bg_0.osc", "bg_1.osc"],
        "current_background_index": 1,
        "manual_pairs": [{"background_index": 1, "entries": []}],
    }


def test_apply_geometry_manual_pairs_rows_replaces_state_and_refreshes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    replaced: dict[int, list[dict[str, object]]] = {}

    imported_backgrounds, imported_pairs, warnings = mg.apply_geometry_manual_pairs_rows(
        [
            {
                "background_path": "bg_1.osc",
                "background_name": "bg_1.osc",
                "entries": [{"label": "1,0,0", "x": 2.0, "y": 3.0}],
            }
        ],
        osc_files=["bg_0.osc", "bg_1.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "keep", "x": 9.0, "y": 10.0}] if int(idx) == 0 else []
        ),
        replace_pairs_by_background=lambda mapping: replaced.update(mapping),
        clear_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        cancel_pick_session=lambda **kwargs: calls.append(("cancel", kwargs)),
        invalidate_pick_cache=lambda: calls.append(("invalidate", None)),
        clear_manual_undo_stack=lambda: calls.append(("clear_manual", None)),
        clear_geometry_fit_undo_stack=lambda: calls.append(("clear_fit", None)),
        render_current_pairs=lambda **kwargs: calls.append(("render", kwargs)),
        update_button_label=lambda: calls.append(("button", None)),
        refresh_status=lambda: calls.append(("refresh", None)),
    )

    assert (imported_backgrounds, imported_pairs, warnings) == (1, 1, [])
    assert replaced == {1: [{"label": "1,0,0", "hkl": (1, 0, 0), "x": 2.0, "y": 3.0}]}
    assert ("clear", {"redraw": False}) in calls
    assert ("cancel", {"restore_view": True, "redraw": False}) in calls
    assert ("invalidate", None) in calls
    assert ("clear_manual", None) in calls
    assert ("clear_fit", None) in calls
    assert ("render", {"update_status": False}) in calls
    assert ("button", None) in calls
    assert ("refresh", None) in calls


def test_apply_geometry_manual_pairs_snapshot_reloads_backgrounds_before_apply(
    tmp_path,
) -> None:
    saved_backgrounds = [tmp_path / "bg_0.osc", tmp_path / "bg_1.osc"]
    for path in saved_backgrounds:
        path.write_text("", encoding="utf-8")

    calls: list[tuple[str, object]] = []
    message = mg.apply_geometry_manual_pairs_snapshot(
        {
            "background_files": [str(path) for path in saved_backgrounds],
            "current_background_index": 1,
            "manual_pairs": [{"background_index": 1, "entries": []}],
        },
        osc_files=["current_0.osc", "current_1.osc"],
        load_background_files=(lambda paths, index: calls.append(("load", (list(paths), index)))),
        apply_pairs_rows=(
            lambda rows, replace_existing=True: (
                calls.append(("apply", (list(rows), replace_existing))) or (1, 2, [])
            )
        ),
        schedule_update=lambda: calls.append(("schedule", None)),
    )

    assert calls[0] == (
        "load",
        ([str(path) for path in saved_backgrounds], 1),
    )
    assert calls[1] == (
        "apply",
        ([{"background_index": 1, "entries": []}], True),
    )
    assert calls[2] == ("schedule", None)
    assert message == "Imported 2 manual placement(s) across 1 background(s)."


def test_export_geometry_manual_pairs_runs_dialog_and_save_callback(tmp_path) -> None:
    statuses: list[str] = []
    calls: list[tuple[str, object]] = []
    save_path = tmp_path / "placements.json"

    result = mg.export_geometry_manual_pairs(
        osc_files=["bg_0.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "1,0,0", "x": 1.0, "y": 2.0}] if int(idx) == 0 else []
        ),
        collect_snapshot=lambda: {"manual_pairs": [{"background_index": 0}]},
        initial_dir=tmp_path,
        asksaveasfilename=(lambda **kwargs: calls.append(("dialog", kwargs)) or str(save_path)),
        save_file=(
            lambda path, payload, metadata=None: calls.append(("save", (path, payload, metadata)))
        ),
        set_status_text=statuses.append,
        stamp_factory=lambda: "20260328_140000",
    )

    assert result == str(save_path)
    assert calls[0][0] == "dialog"
    assert calls[0][1]["initialfile"] == "ra_sim_geometry_placements_20260328_140000.json"
    assert calls[1] == (
        "save",
        (
            str(save_path),
            {"manual_pairs": [{"background_index": 0}]},
            {"entrypoint": "python -m ra_sim gui"},
        ),
    )
    assert statuses[-1] == f"Saved manual geometry placements to {save_path}"


def test_manual_picker_truth_pairs_keep_detector_frame_for_caked_saved_pick() -> None:
    saved_entry = {
        "label": "3,0,4",
        "hkl": (3, 0, 4),
        "q_group_key": ("q_group", "primary", 3, 4),
        "source_reflection_index": 16,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 2,
        "source_row_index": 5,
        "x": 102.0,
        "y": 204.0,
        "background_detector_x": 2.0,
        "background_detector_y": 4.0,
        "caked_x": 12.0,
        "caked_y": 24.0,
        "background_two_theta_deg": 12.0,
        "background_phi_deg": 24.0,
        "refined_sim_x": 102.0,
        "refined_sim_y": 204.0,
        "refined_sim_native_x": 2.0,
        "refined_sim_native_y": 4.0,
        "refined_sim_caked_x": 12.0,
        "refined_sim_caked_y": 24.0,
        "simulated_two_theta_deg": 12.0,
        "simulated_phi_deg": 24.0,
    }

    truth_pairs = mg.build_geometry_manual_picker_truth_pairs(0, [saved_entry])

    assert len(truth_pairs) == 1
    truth = truth_pairs[0]
    assert truth["manual_background_frame"] == "display"
    assert truth["manual_background_point"] == [102.0, 204.0]
    assert truth["manual_selected_simulated_frame"] == "display"
    assert truth["manual_selected_simulated_point"] == [102.0, 204.0]
    assert truth["manual_selected_to_background_distance_px"] == 0.0
    assert truth["manual_picker_truth_available"] is True


def test_refresh_geometry_manual_pair_entry_preserves_caked_pick_visual_coords() -> None:
    candidate = {
        "label": "-3,0,4",
        "hkl": (-3, 0, 4),
        "q_group_key": ("q_group", "primary", 3, 4),
        "source_reflection_index": 17,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 2,
        "source_row_index": 17,
        "sim_col": 103.0,
        "sim_row": 204.0,
        "native_col": 3.0,
        "native_row": 4.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }
    entry = mg.geometry_manual_pair_entry_from_candidate(
        candidate,
        103.0,
        204.0,
        group_key=("q_group", "primary", 3, 4),
        detector_col=3.0,
        detector_row=4.0,
        caked_col=13.0,
        caked_row=24.0,
        raw_caked_col=13.0,
        raw_caked_row=24.0,
    )
    assert entry is not None

    refreshed = mg.refresh_geometry_manual_pair_entry(
        entry,
        background_display_shape=(64, 64),
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(two_theta) + 100.0,
            float(phi) + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 0.25,
            float(row) + 0.25,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
    )

    assert refreshed is not None
    assert refreshed["caked_x"] == 13.0
    assert refreshed["caked_y"] == 24.0
    assert refreshed["background_two_theta_deg"] == 13.0
    assert refreshed["background_phi_deg"] == 24.0
    assert refreshed["detector_x"] == 13.0
    assert refreshed["detector_y"] == 24.0
    assert refreshed["x"] == 113.0
    assert refreshed["y"] == 224.0


def test_import_geometry_manual_pairs_loads_snapshot_and_reports_caked_view_warning(
    tmp_path,
) -> None:
    statuses: list[str] = []
    apply_calls: list[tuple[dict[str, object], bool]] = []
    open_path = tmp_path / "placements.json"

    result = mg.import_geometry_manual_pairs(
        initial_dir=tmp_path,
        askopenfilename=lambda **_kwargs: str(open_path),
        load_file=lambda _path: {"state": {"manual_pairs": [{"background_index": 0}]}},
        apply_snapshot=(
            lambda snapshot, allow_background_reload=True: (
                apply_calls.append((dict(snapshot), bool(allow_background_reload)))
                or "Imported placements."
            )
        ),
        ensure_geometry_fit_caked_view=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        set_status_text=statuses.append,
    )

    assert apply_calls == [({"manual_pairs": [{"background_index": 0}]}, True)]
    assert result == (
        "Imported placements. Warning: imported placements but could not switch "
        "to 2D caked view (boom)."
    )
    assert statuses[-1] == result


def test_geometry_manual_unassigned_candidates_and_current_pending_candidate() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {"label": "1,0,0", "source_table_index": 1, "source_row_index": 2},
            {"label": "-1,0,0", "source_table_index": 1, "source_row_index": 3},
        ],
        "pending_entries": [
            {"label": "1,0,0", "source_table_index": 1, "source_row_index": 2, "x": 8.0, "y": 9.0}
        ],
        "background_index": 0,
    }

    remaining = mg.geometry_manual_unassigned_group_candidates(
        session,
        current_background_index=0,
    )
    pending = mg.geometry_manual_current_pending_candidate(
        session,
        current_background_index=0,
    )

    assert len(remaining) == 1
    assert remaining[0]["label"] == "-1,0,0"
    assert pending["label"] == "-1,0,0"


def test_geometry_manual_refine_preview_point_falls_back_to_local_peak_maximum() -> None:
    image = np.zeros((9, 9), dtype=float)
    image[6, 5] = 9.5

    refined = mg.geometry_manual_refine_preview_point(
        None,
        4.9,
        5.8,
        display_background=image,
        cache_data={"match_config": {}, "background_context": None},
        use_caked_space=False,
    )

    assert refined == (5.0, 6.0)


def test_geometry_manual_refine_preview_point_uses_candidate_sim_seed_for_peak_context() -> None:
    seen: dict[str, object] = {}

    def _fake_match(candidates, background_context, match_cfg):
        seen["candidate"] = dict(candidates[0])
        seen["background_context"] = dict(background_context)
        seen["match_cfg"] = dict(match_cfg)
        return ([{"x": 11.0, "y": 12.0}], {"status": "ok"})

    refined = mg.geometry_manual_refine_preview_point(
        {"sim_col": 30.0, "sim_row": 31.0},
        10.0,
        20.0,
        display_background=np.zeros((8, 8), dtype=float),
        cache_data={
            "match_config": {"search_radius_px": 6.0},
            "background_context": {"img_valid": True},
        },
        use_caked_space=False,
        match_simulated_peaks_to_peak_context=_fake_match,
    )

    assert refined == (11.0, 12.0)
    assert seen["candidate"]["sim_col"] == 30.0
    assert seen["candidate"]["sim_row"] == 31.0
    assert seen["background_context"] == {"img_valid": True}
    assert seen["match_cfg"] == {"search_radius_px": 6.0}


def test_refine_detector_pick_via_caked_background_projects_back_to_detector() -> None:
    refined = mg.refine_detector_pick_via_caked_background(
        None,
        3.0,
        4.0,
        detector_display_to_caked_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) - 5.0,
        ),
        caked_angles_to_detector_display_coords=lambda two_theta, phi: (
            float(two_theta) - 10.0,
            float(phi) + 5.0,
        ),
        caked_background=np.zeros((6, 6), dtype=float),
        radial_axis=np.linspace(10.0, 15.0, 6),
        azimuth_axis=np.linspace(-5.0, 0.0, 6),
        cache_data={"match_config": {}, "background_context": None},
        refine_caked_peak_center_fn=lambda *_args: (13.2, -1.5),
    )

    assert refined is not None
    assert refined["raw_caked_col"] == 13.0
    assert refined["raw_caked_row"] == -1.0
    assert refined["refined_caked_col"] == 13.2
    assert refined["refined_caked_row"] == -1.5
    assert np.isclose(refined["refined_display_col"], 3.2)
    assert refined["refined_display_row"] == 3.5


def test_refine_detector_pick_via_caked_background_uses_caked_candidate_seed() -> None:
    seen: dict[str, object] = {}
    radial_axis = np.array([10.0, 11.0, 12.0], dtype=float)
    azimuth_axis = np.array([-3.0, -2.0, -1.0], dtype=float)

    def _fake_match(candidates, background_context, match_cfg):
        seen["candidate"] = dict(candidates[0])
        seen["background_context"] = dict(background_context)
        seen["match_cfg"] = dict(match_cfg)
        return ([{"x": 1.0, "y": 2.0}], {"status": "ok"})

    refined = mg.refine_detector_pick_via_caked_background(
        {
            "sim_col": 300.0,
            "sim_row": 400.0,
            "caked_x": 12.0,
            "caked_y": -2.0,
        },
        3.0,
        4.0,
        detector_display_to_caked_coords=lambda col, row: (
            float(col) + 8.0,
            float(row) - 7.0,
        ),
        caked_angles_to_detector_display_coords=lambda two_theta, phi: (
            float(two_theta) - 8.0,
            float(phi) + 7.0,
        ),
        caked_background=np.zeros((6, 6), dtype=float),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        cache_data={
            "match_config": {"search_radius_px": 6.0},
            "background_context": {"img_valid": True},
        },
        match_simulated_peaks_to_peak_context=_fake_match,
    )

    assert refined is not None
    assert refined["raw_caked_col"] == 11.0
    assert refined["raw_caked_row"] == -3.0
    assert refined["refined_caked_col"] == 11.0
    assert refined["refined_caked_row"] == -1.0
    assert refined["refined_display_col"] == 3.0
    assert refined["refined_display_row"] == 6.0
    assert seen["candidate"]["sim_col"] == 12.0
    assert seen["candidate"]["sim_row"] == -2.0
    assert seen["candidate"]["sim_col_local"] == 2.0
    assert seen["candidate"]["sim_row_local"] == 1.0
    assert seen["background_context"] == {"img_valid": True}
    assert seen["match_cfg"] == {"search_radius_px": 6.0}


def test_resolve_background_pick_to_caked_angles_binds_refined_angles_not_raw_click() -> None:
    reverse_calls: list[tuple[float, float]] = []
    refine_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return float(two_theta) + 100.0, float(phi) + 200.0

    def _refine_detector(_candidate, raw_col, raw_row, **kwargs):
        refine_calls.append((float(raw_col), float(raw_row)))
        return float(raw_col) + 1.0, float(raw_row) + 2.0

    caked_image = np.zeros((3, 3), dtype=float)
    caked_image[2, 2] = 10.0
    resolved = mg.resolve_background_pick_to_caked_angles(
        {"label": "1,0,0"},
        1.0,
        1.0,
        active_view="caked",
        display_background=caked_image,
        refine_detector_pick_fn=_refine_detector,
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (float(col), float(row)),
        radial_axis=np.array([18.0, 20.0, 22.0], dtype=float),
        azimuth_axis=np.array([28.0, 30.0, 32.0], dtype=float),
    )

    assert resolved is not None
    assert reverse_calls == [(22.0, 32.0)]
    assert refine_calls == []
    assert resolved["raw_caked_display_col"] == 1.0
    assert resolved["raw_caked_display_row"] == 1.0
    assert resolved["raw_caked_two_theta_deg"] == 20.0
    assert resolved["raw_caked_phi_deg"] == 30.0
    assert "detector_seed_col" not in resolved
    assert "detector_seed_row" not in resolved
    assert resolved["refined_detector_display_col"] == 122.0
    assert resolved["refined_detector_display_row"] == 232.0
    assert resolved["refined_detector_native_col"] == 22.0
    assert resolved["refined_detector_native_row"] == 32.0
    assert resolved["background_detector_x"] == 22.0
    assert resolved["background_detector_y"] == 32.0
    assert resolved["refined_background_two_theta_deg"] == 22.0
    assert resolved["refined_background_phi_deg"] == 32.0


def test_caked_background_pick_refines_locally_binds_lut_and_separates_display() -> None:
    group_key = ("q_group", "primary", 1, 0)
    candidate = {
        "label": "1,0,0",
        "hkl": (1, 0, 0),
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 3,
        "source_branch_index": 0,
        "branch_id": "+x",
        "sim_col": 120.0,
        "sim_row": 200.0,
        "native_col": 490.0,
        "native_row": 580.0,
        "sim_native_x": 490.0,
        "sim_native_y": 580.0,
        "caked_x": 500.0,
        "caked_y": 600.0,
        "simulated_two_theta_deg": 500.0,
        "simulated_phi_deg": 600.0,
    }
    radial_axis = np.array([18.0, 20.0, 22.0], dtype=float)
    azimuth_axis = np.array([28.0, 30.0, 32.0], dtype=float)
    caked_image = np.zeros((3, 3), dtype=float)
    caked_image[2, 2] = 10.0
    reverse_calls: list[tuple[float, float]] = []
    refine_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return float(two_theta) + 100.0, float(phi) + 200.0

    def _display_to_native(col, row):
        return float(col) - 100.0, float(row) - 200.0

    def _native_to_caked(col, row):
        return float(col), float(row)

    def _refine_detector(_candidate, raw_col, raw_row, **kwargs):
        refine_calls.append((float(raw_col), float(raw_row)))
        return float(raw_col) + 1.0, float(raw_row) + 2.0

    def _pick_session() -> dict[str, object]:
        return {
            "group_key": group_key,
            "group_entries": [dict(candidate)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate": dict(candidate),
            "tagged_candidate_key": mg.geometry_manual_candidate_source_key(candidate),
        }

    def _project_caked_rows(rows):
        projected = []
        for row in rows or []:
            projected_row = dict(row)
            projected_row["caked_x"] = 500.0
            projected_row["caked_y"] = 600.0
            projected_row["display_col"] = 500.0
            projected_row["display_row"] = 600.0
            projected_row["sim_col_raw"] = 120.0
            projected_row["sim_row_raw"] = 230.0
            projected.append(projected_row)
        return projected

    saved_entry_sets: list[list[dict[str, object]]] = []
    handled, next_session = mg.geometry_manual_place_selection_at(
        1.0,
        1.0,
        pick_session=_pick_session(),
        current_background_index=0,
        display_background=caked_image,
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=_refine_detector,
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=_display_to_native,
        native_detector_coords_to_caked_display_coords=_native_to_caked,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )

    assert handled is True
    assert next_session == {}
    caked_entry = saved_entry_sets[-1][0]
    np.testing.assert_allclose(
        [
            caked_entry["background_two_theta_deg"],
            caked_entry["background_phi_deg"],
        ],
        [22.0, 32.0],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert refine_calls == []
    assert reverse_calls == [(22.0, 32.0)]
    assert caked_entry["raw_caked_display_col"] == 1.0
    assert caked_entry["raw_caked_display_row"] == 1.0
    assert caked_entry["raw_caked_two_theta_deg"] == 20.0
    assert caked_entry["raw_caked_phi_deg"] == 30.0
    assert caked_entry["background_detector_x"] == 22.0
    assert caked_entry["background_detector_y"] == 32.0

    poisoned = {
        **dict(caked_entry),
        "caked_x": -900.0,
        "caked_y": -901.0,
        "raw_caked_x": -902.0,
        "raw_caked_y": -903.0,
        "raw_caked_two_theta_deg": -904.0,
        "raw_caked_phi_deg": -905.0,
        "refined_sim_caked_x": -906.0,
        "refined_sim_caked_y": -907.0,
        "refined_sim_x": -908.0,
        "refined_sim_y": -909.0,
        "sim_display": (-910.0, -911.0),
    }

    active_display = mg.geometry_manual_session_initial_pairs_display(
        {
            "group_key": group_key,
            "group_entries": [dict(candidate)],
            "pending_entries": [dict(poisoned)],
            "target_count": 1,
            "background_index": 0,
        },
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=_project_caked_rows,
        entry_display_coords=lambda _entry: (-1.0, -1.0),
    )
    assert active_display[0]["sim_display"] == (500.0, 600.0)
    assert active_display[0]["bg_display"] == (22.0, 32.0)

    detector_reverse_calls: list[tuple[float, float]] = []
    detector_refreshed = mg.refresh_geometry_manual_pair_entry(
        poisoned,
        background_display_shape=(),
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            detector_reverse_calls.append((float(two_theta), float(phi)))
            or (float(two_theta) + 100.0, float(phi) + 200.0)
        ),
        background_display_to_native_detector_coords=_display_to_native,
        native_detector_coords_to_caked_display_coords=_native_to_caked,
    )
    assert detector_reverse_calls == []
    assert detector_refreshed["x"] == 122.0
    assert detector_refreshed["y"] == 232.0
    assert detector_refreshed["background_two_theta_deg"] == 22.0
    assert detector_refreshed["background_phi_deg"] == 32.0

    def _lookup(rows):
        lookup: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            key = mg.geometry_manual_candidate_source_key(row)
            if key is not None:
                lookup.setdefault(key, []).append(dict(row))
        return lookup

    _measured, saved_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=False,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(poisoned)],
        get_cache_data=lambda **_kwargs: {},
        source_rows_for_background=lambda *_args, **_kwargs: [dict(candidate)],
        build_simulated_lookup=_lookup,
        project_peaks_to_current_view=_project_caked_rows,
        entry_display_coords=lambda _entry: (-1.0, -1.0),
    )
    assert saved_display[0]["bg_display"] == active_display[0]["bg_display"]


def test_refresh_geometry_manual_pair_entry_recomputes_stale_detector_anchor_once() -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return float(two_theta) + 100.0, float(phi) + 200.0

    stale_entry = {
        "label": "1,0,0",
        "x": 999.0,
        "y": 999.0,
        "background_two_theta_deg": 22.0,
        "background_phi_deg": 32.0,
        "background_detector_x": 0.0,
        "background_detector_y": 0.0,
    }
    refreshed = mg.refresh_geometry_manual_pair_entry(
        stale_entry,
        background_display_shape=(),
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
    )

    assert refreshed is not None
    assert reverse_calls == [(22.0, 32.0)]
    assert refreshed["background_detector_x"] == 22.0
    assert refreshed["background_detector_y"] == 32.0
    assert refreshed["x"] == 122.0
    assert refreshed["y"] == 232.0

    reverse_calls.clear()
    refreshed_again = mg.refresh_geometry_manual_pair_entry(
        refreshed,
        background_display_shape=(),
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
    )

    assert refreshed_again is not None
    assert reverse_calls == []
    assert refreshed_again["background_detector_x"] == 22.0
    assert refreshed_again["background_detector_y"] == 32.0


def test_refresh_geometry_manual_pair_entry_sim_replay_uses_current_projected_caked_point_without_detector_alias_fallback() -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return float(two_theta) + 100.0, float(phi) + 200.0

    saved_entry = {
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "refined_sim_x": 901.0,
        "refined_sim_y": 902.0,
        "refined_sim_native_x": 903.0,
        "refined_sim_native_y": 904.0,
        "native_col": 905.0,
        "native_row": 906.0,
        "sim_native_x": 907.0,
        "sim_native_y": 908.0,
        "display_col": 909.0,
        "display_row": 910.0,
        "sim_col": 911.0,
        "sim_row": 912.0,
        "sim_col_raw": 913.0,
        "sim_row_raw": 914.0,
        "detector_x": 915.0,
        "detector_y": 916.0,
        "caked_x": -999.0,
        "caked_y": -998.0,
        "raw_caked_x": -997.0,
        "raw_caked_y": -996.0,
    }
    projected_sim_entry = {
        "_caked_qr_projection_cache": True,
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "caked_x": 23.0,
        "caked_y": -26.0,
        "two_theta_deg": 23.0,
        "phi_deg": -26.0,
    }

    refreshed = mg.refresh_geometry_manual_pair_entry(
        saved_entry,
        background_display_shape=(256, 256),
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        current_projected_sim_entry=projected_sim_entry,
    )

    assert refreshed is not None
    assert reverse_calls == [(23.0, -26.0)]
    assert refreshed["detector_x"] == 23.0
    assert refreshed["detector_y"] == -26.0
    assert refreshed["sim_detector_anchor_x"] == 23.0
    assert refreshed["sim_detector_anchor_y"] == -26.0
    assert refreshed["x"] == 33.0
    assert refreshed["y"] == -6.0
    assert refreshed["display_col"] == 33.0
    assert refreshed["display_row"] == -6.0
    assert refreshed["sim_detector_display_col"] == 33.0
    assert refreshed["sim_detector_display_row"] == -6.0
    assert refreshed["sim_detector_frame_provenance"] == "sim_reverse_lut_replay_cache"
    assert "background_detector_x" not in refreshed
    assert "background_detector_y" not in refreshed
    assert "background_detector_frame_provenance" not in refreshed


def test_refresh_geometry_manual_pair_entry_sim_replay_recomputes_stale_anchor_once() -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return float(two_theta) + 100.0, float(phi) + 200.0

    stale_entry = {
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "sim_detector_anchor_x": 999.0,
        "sim_detector_anchor_y": 998.0,
        "sim_detector_display_col": 997.0,
        "sim_detector_display_row": 996.0,
        "sim_detector_frame_provenance": "stale-anchor",
    }
    projected_sim_entry = {
        "_caked_qr_projection_cache": True,
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "caked_x": 23.0,
        "caked_y": -26.0,
        "two_theta_deg": 23.0,
        "phi_deg": -26.0,
    }

    refreshed = mg.refresh_geometry_manual_pair_entry(
        stale_entry,
        background_display_shape=(256, 256),
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        current_projected_sim_entry=projected_sim_entry,
    )

    assert refreshed is not None
    assert reverse_calls == [(23.0, -26.0)]
    assert refreshed["sim_detector_anchor_x"] == 23.0
    assert refreshed["sim_detector_anchor_y"] == -26.0
    assert refreshed["sim_detector_display_col"] == 33.0
    assert refreshed["sim_detector_display_row"] == -6.0

    reverse_calls.clear()
    refreshed_again = mg.refresh_geometry_manual_pair_entry(
        refreshed,
        background_display_shape=(256, 256),
        caked_angles_to_background_display_coords=_reverse_lut,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        current_projected_sim_entry=projected_sim_entry,
    )

    assert refreshed_again is not None
    assert reverse_calls == []
    assert refreshed_again["sim_detector_anchor_x"] == 23.0
    assert refreshed_again["sim_detector_anchor_y"] == -26.0
    assert refreshed_again["sim_detector_display_col"] == 33.0
    assert refreshed_again["sim_detector_display_row"] == -6.0


def test_caked_background_branch_association_uses_refined_peak_before_save() -> None:
    group_key = ("q_group", "primary", 1, 0)
    branch_zero = {
        "label": "branch-0",
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "branch_id": "+x",
        "caked_x": 20.0,
        "caked_y": 30.0,
    }
    branch_one = {
        "label": "branch-1",
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "branch_id": "-x",
        "caked_x": 22.0,
        "caked_y": 32.0,
    }
    radial_axis = np.array([18.0, 20.0, 22.0], dtype=float)
    azimuth_axis = np.array([28.0, 30.0, 32.0], dtype=float)
    caked_image = np.zeros((3, 3), dtype=float)
    caked_image[2, 2] = 10.0
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        1.0,
        1.0,
        pick_session={
            "group_key": group_key,
            "group_entries": [dict(branch_zero), dict(branch_one)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
        },
        current_background_index=0,
        display_background=caked_image,
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked background must not refine in detector space")
        ),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda tth, phi: (
            float(tth) + 100.0,
            float(phi) + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )

    assert handled is True
    assert next_session == {}
    saved = saved_entry_sets[-1][0]
    assert saved["source_branch_index"] == 1
    assert saved["source_row_index"] == 1
    assert saved["branch_id"] == "-x"
    assert saved["background_two_theta_deg"] == 22.0
    assert saved["background_phi_deg"] == 32.0

    measured, displayed = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved)],
        get_cache_data=lambda **_kwargs: {},
        source_rows_for_background=lambda *_args, **_kwargs: [
            dict(branch_zero),
            dict(branch_one),
        ],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=lambda rows: [dict(row) for row in (rows or [])],
        entry_display_coords=lambda _entry: (-1.0, -1.0),
    )

    assert measured[0]["source_branch_index"] == 1
    assert displayed[0]["bg_display"] == (22.0, 32.0)


def test_caked_background_same_branch_pick_replaces_existing_branch_only() -> None:
    group_key = ("q_group", "primary", 1, 0)
    old_branch_zero = {
        "label": "branch-0",
        "x": 110.0,
        "y": 210.0,
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "branch_id": "+x",
        "background_two_theta_deg": 10.0,
        "background_phi_deg": 20.0,
        "background_detector_x": 10.0,
        "background_detector_y": 20.0,
    }
    old_branch_one = {
        "label": "branch-1",
        "x": 150.0,
        "y": 250.0,
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "branch_id": "-x",
        "background_two_theta_deg": 50.0,
        "background_phi_deg": 60.0,
        "background_detector_x": 50.0,
        "background_detector_y": 60.0,
    }
    candidate = {
        "label": "branch-0",
        "q_group_key": group_key,
        "source_table_index": 7,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "branch_id": "+x",
        "caked_x": 22.0,
        "caked_y": 32.0,
    }
    radial_axis = np.array([18.0, 20.0, 22.0], dtype=float)
    azimuth_axis = np.array([28.0, 30.0, 32.0], dtype=float)
    caked_image = np.zeros((3, 3), dtype=float)
    caked_image[2, 2] = 10.0
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        1.0,
        1.0,
        pick_session={
            "group_key": group_key,
            "group_entries": [dict(candidate)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [dict(old_branch_zero), dict(old_branch_one)],
            "q_label": "selected group",
            "background_index": 0,
        },
        current_background_index=0,
        display_background=caked_image,
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked background must not refine in detector space")
        ),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda tth, phi: (
            float(tth) + 100.0,
            float(phi) + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )

    assert handled is True
    assert next_session == {}
    saved_entries = saved_entry_sets[-1]
    assert len(saved_entries) == 2
    by_branch = {int(entry["source_branch_index"]): entry for entry in saved_entries}
    assert by_branch[0]["background_two_theta_deg"] == 22.0
    assert by_branch[0]["background_phi_deg"] == 32.0
    assert by_branch[1]["background_two_theta_deg"] == 50.0
    assert by_branch[1]["background_phi_deg"] == 60.0


def test_update_geometry_manual_peak_record_cache_updates_cached_positions() -> None:
    peak_records = [
        {
            "source_table_index": 1,
            "source_row_index": 2,
            "display_col": 3.0,
            "display_row": 4.0,
            "sim_col": 3.0,
            "sim_row": 4.0,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "native_col": 5.0,
            "native_row": 6.0,
            "sim_native_x": 5.0,
            "sim_native_y": 6.0,
            "caked_x": 7.0,
            "caked_y": 8.0,
            "raw_caked_x": 7.0,
            "raw_caked_y": 8.0,
            "two_theta_deg": 7.0,
            "phi_deg": 8.0,
        },
        {
            "source_table_index": 9,
            "source_row_index": 10,
            "display_col": 30.0,
            "display_row": 40.0,
        },
    ]
    peak_positions = [(3.0, 4.0), (30.0, 40.0)]
    peak_overlay_cache = {
        "records": [dict(record) for record in peak_records],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(peak_records[0]),
        refined_caked=(11.0, 12.0),
        refined_native=(13.0, 14.0),
        refined_display=(15.0, 16.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["two_theta_deg"] == 11.0
    assert peak_records[0]["phi_deg"] == 12.0
    assert peak_records[0]["native_col"] == 13.0
    assert peak_records[0]["native_row"] == 14.0
    assert peak_records[0]["sim_native_x"] == 13.0
    assert peak_records[0]["sim_native_y"] == 14.0
    assert peak_records[0]["display_col"] == 15.0
    assert peak_records[0]["display_row"] == 16.0
    assert peak_records[0]["sim_col"] == 15.0
    assert peak_records[0]["sim_row"] == 16.0
    assert peak_records[0]["sim_col_raw"] == 15.0
    assert peak_records[0]["sim_row_raw"] == 16.0
    assert peak_records[0]["caked_x"] == 11.0
    assert peak_records[0]["caked_y"] == 12.0
    assert peak_records[0]["raw_caked_x"] == 11.0
    assert peak_records[0]["raw_caked_y"] == 12.0
    assert peak_records[1]["display_col"] == 30.0
    assert peak_positions[0] == (15.0, 16.0)
    assert peak_positions[1] == (30.0, 40.0)
    assert peak_overlay_cache["records"][0]["two_theta_deg"] == 11.0
    assert peak_overlay_cache["records"][0]["phi_deg"] == 12.0
    assert peak_overlay_cache["records"][0]["sim_col"] == 15.0
    assert peak_overlay_cache["records"][0]["sim_row"] == 16.0
    assert peak_overlay_cache["records"][0]["caked_x"] == 11.0
    assert peak_overlay_cache["records"][0]["caked_y"] == 12.0
    assert peak_overlay_cache["positions"][0] == (15.0, 16.0)
    assert peak_overlay_cache["click_spatial_index"] is None


def test_geometry_manual_live_peak_candidates_do_not_promote_display_to_detector_coords_when_native_available() -> (
    None
):
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 1.5,
                "native_row": 2.5,
                "hkl_raw": [1.0, 0.0, 0.0],
                "intensity": 7.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            }
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["display_col"] == 30.25
    assert candidates[0]["display_row"] == -57.5
    assert candidates[0]["native_col"] == 1.5
    assert candidates[0]["native_row"] == 2.5
    assert "sim_col_raw" not in candidates[0]
    assert "sim_row_raw" not in candidates[0]


def test_update_geometry_manual_peak_record_cache_updates_only_matching_mirrored_branch() -> None:
    left_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    peak_records = [dict(left_record), dict(right_record)]
    peak_positions = [
        (float(left_record["display_col"]), float(left_record["display_row"])),
        (float(right_record["display_col"]), float(right_record["display_row"])),
    ]
    peak_overlay_cache = {
        "records": [dict(left_record), dict(right_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(right_record),
        refined_display=(193.0, 99.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 193.0
    assert peak_records[1]["display_row"] == 99.0
    assert peak_positions == [(181.0, 95.0), (193.0, 99.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 193.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (193.0, 99.0)]
    assert peak_overlay_cache["click_spatial_index"] is None


def test_update_geometry_manual_peak_record_cache_matches_legacy_branch_key_to_current_record() -> (
    None
):
    legacy_saved_entry = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    left_record = {
        "label": "left",
        "hkl": (1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_record = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    peak_records = [dict(left_record), dict(right_record)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_record), dict(right_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(legacy_saved_entry),
        source_entry=legacy_saved_entry,
        refined_display=(191.0, 97.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 191.0
    assert peak_records[1]["display_row"] == 97.0
    assert peak_positions == [(181.0, 95.0), (191.0, 97.0)]
    assert peak_overlay_cache["records"][1]["display_col"] == 191.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (191.0, 97.0)]


def test_geometry_manual_source_key_matches_entry_does_not_alias_table_and_reflection_branch_keys() -> (
    None
):
    legacy_saved_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
    }
    current_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
    }

    assert (
        mg.geometry_manual_source_key_matches_entry(
            _source_key(legacy_saved_entry),
            current_record,
        )
        is False
    )


def test_geometry_manual_source_entries_share_identity_rejects_branchless_same_row_siblings_with_different_hkl() -> (
    None
):
    left_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "hkl": (1, 0, 5),
        "label": "1,0,5",
    }
    right_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "hkl": (-1, 0, 5),
        "label": "-1,0,5",
    }

    assert mg.geometry_manual_source_entries_share_identity(left_entry, right_entry) is False


def test_geometry_manual_source_entries_share_identity_rejects_untrusted_same_branch_siblings() -> (
    None
):
    left_entry = {
        "source_table_index": 9,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "hkl": (1, 0, 5),
        "label": "1,0,5-right",
    }
    right_entry = {
        "source_table_index": 9,
        "source_row_index": 3,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "hkl": (2, 0, 5),
        "label": "2,0,5-right",
    }

    assert mg.geometry_manual_source_entries_share_identity(left_entry, right_entry) is False


def test_update_geometry_manual_peak_record_cache_source_entry_does_not_fan_out_across_branchless_same_row_siblings() -> (
    None
):
    source_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "hkl": (1, 0, 5),
        "label": "1,0,5",
    }
    left_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "hkl": (1, 0, 5),
        "label": "1,0,5",
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "hkl": (-1, 0, 5),
        "label": "-1,0,5",
        "display_col": 190.0,
        "display_row": 96.0,
    }
    peak_records = [dict(left_record), dict(right_record)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_record), dict(right_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(source_entry),
        source_entry=source_entry,
        refined_display=(182.0, 94.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["display_col"] == 182.0
    assert peak_records[0]["display_row"] == 94.0
    assert peak_records[1]["display_col"] == 190.0
    assert peak_records[1]["display_row"] == 96.0
    assert peak_positions == [(182.0, 94.0), (190.0, 96.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 182.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(182.0, 94.0), (190.0, 96.0)]


def test_update_geometry_manual_peak_record_cache_source_entry_falls_back_to_row_key_when_identity_is_inconclusive() -> (
    None
):
    source_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    cache_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    peak_records = [dict(cache_record)]
    peak_positions = [(181.0, 95.0)]
    peak_overlay_cache = {
        "records": [dict(cache_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 1},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(source_entry),
        source_entry=source_entry,
        refined_display=(182.0, 94.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["display_col"] == 182.0
    assert peak_records[0]["display_row"] == 94.0
    assert peak_positions == [(182.0, 94.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 182.0
    assert peak_overlay_cache["positions"] == [(182.0, 94.0)]
    assert peak_overlay_cache["click_spatial_index"] is None


def test_update_geometry_manual_peak_record_cache_source_entry_only_updates_row_key_unique_matches() -> (
    None
):
    source_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    first_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    second_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    peak_records = [dict(first_record), dict(second_record)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(first_record), dict(second_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(source_entry),
        source_entry=source_entry,
        refined_display=(192.0, 98.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is False
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 190.0
    assert peak_records[1]["display_row"] == 96.0
    assert peak_positions == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["click_spatial_index"] == {"position_count": 2}


def test_update_geometry_manual_peak_record_cache_skips_branchless_row_key_for_mirrored_branches() -> (
    None
):
    legacy_saved_entry = {
        "source_table_index": 9,
        "source_row_index": 0,
        "label": "",
        "display_col": 190.0,
        "display_row": 96.0,
    }
    left_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_record = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    peak_records = [dict(left_record), dict(right_record)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_record), dict(right_record)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(legacy_saved_entry),
        refined_display=(191.0, 97.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is False
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 190.0
    assert peak_records[1]["display_row"] == 96.0
    assert peak_positions == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 190.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (190.0, 96.0)]
    assert peak_overlay_cache["click_spatial_index"] == {"position_count": 2}


def test_restore_geometry_manual_pick_view_resets_zoom_state() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "zoom_active": True,
        "zoom_center": (5.0, 6.0),
        "saved_xlim": (10.0, 20.0),
        "saved_ylim": (30.0, 40.0),
    }
    axis = _DummyAxis()
    canvas = _DummyCanvas()

    restored = mg.restore_geometry_manual_pick_view(session, axis=axis, canvas=canvas)

    assert restored is True
    assert axis.get_xlim() == (10.0, 20.0)
    assert axis.get_ylim() == (30.0, 40.0)
    assert session["zoom_active"] is False
    assert session["zoom_center"] is None
    assert session["saved_xlim"] is None
    assert session["saved_ylim"] is None
    assert canvas.draws == 1


def test_apply_geometry_manual_pick_zoom_updates_axis_and_session() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "background_index": 0,
        "zoom_active": False,
    }
    axis = _DummyAxis((0.0, 300.0), (0.0, 200.0))
    canvas = _DummyCanvas()

    updated = mg.apply_geometry_manual_pick_zoom(
        session,
        150.0,
        80.0,
        display_background=np.zeros((200, 300), dtype=float),
        axis=axis,
        canvas=canvas,
        use_caked_space=False,
        caked_zoom_tth_deg=4.0,
        caked_zoom_phi_deg=24.0,
        pick_zoom_window_px=100.0,
    )

    assert updated is True
    assert axis.get_xlim() == (100.0, 200.0)
    assert axis.get_ylim() == (30.0, 130.0)
    assert session["zoom_active"] is True
    assert session["zoom_center"] == (150.0, 80.0)
    assert session["saved_xlim"] == (0.0, 300.0)
    assert session["saved_ylim"] == (0.0, 200.0)
    assert canvas.draws == 1


def test_geometry_manual_pick_preview_state_builds_status_message() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "q_label": "test group",
        "group_entries": [{"label": "1,0,0"}],
        "background_index": 0,
    }

    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        force=True,
        remaining_candidates=[{"label": "right", "sim_col": 12.5, "sim_row": 14.5}],
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert preview["refined_col"] == 12.0
    assert preview["refined_row"] == 14.0
    assert preview["delta"] > 0.0
    assert preview["sigma_px"] > preview["delta"]
    assert preview["preview_color"] == mg.geometry_manual_preview_color(preview["sigma_px"])
    assert preview["quality_label"] == mg.geometry_manual_preview_quality_label(preview["sigma_px"])
    assert "test group" in preview["message"]
    assert "nearest sim [right]" in preview["message"]
    assert f"quality={preview['quality_label']}" in preview["message"]


def test_geometry_manual_pick_preview_state_prefers_tagged_candidate() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 2),
        "q_label": "test group",
        "group_entries": [{"label": "left"}, {"label": "right"}],
        "background_index": 0,
        "tagged_candidate_key": ("source", 1, 2),
    }

    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        force=True,
        remaining_candidates=[
            {
                "label": "left",
                "sim_col": 35.0,
                "sim_row": 30.0,
                "source_table_index": 1,
                "source_row_index": 2,
            },
            {
                "label": "right",
                "sim_col": 12.5,
                "sim_row": 14.5,
                "source_table_index": 1,
                "source_row_index": 3,
            },
        ],
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert preview["candidate"]["label"] == "left"
    assert "tagged sim [left]" in preview["message"]


def test_geometry_manual_pick_preview_state_refines_with_mosaic_top_context() -> None:
    key = ("q_group", "primary", 1, 2)
    clicked_seed = {
        "label": "near",
        "sim_col": 10.0,
        "sim_row": 20.0,
        "branch_id": "+x",
        "branch_source": "generated",
        "mosaic_weight": 0.1,
        "best_sample_index": 3,
        "source_table_index": 1,
        "source_row_index": 30,
    }
    selected_ray = {
        "label": "top",
        "sim_col": 35.0,
        "sim_row": 30.0,
        "branch_id": "+x",
        "branch_source": "generated",
        "mosaic_weight": 0.9,
        "best_sample_index": 0,
        "source_table_index": 1,
        "source_row_index": 31,
    }
    seen: dict[str, object] = {}

    def _refine_preview_point(source_entry, raw_col, raw_row, **_kwargs):
        seen["source_entry"] = dict(source_entry) if source_entry else None
        return float(raw_col) + 0.2, float(raw_row) + 0.4

    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session={
            "group_key": key,
            "q_label": "test group",
            "group_entries": [dict(clicked_seed), dict(selected_ray)],
            "background_index": 0,
            "tagged_candidate_key": _source_key(clicked_seed),
        },
        current_background_index=0,
        force=True,
        remaining_candidates=[dict(clicked_seed), dict(selected_ray)],
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=_refine_preview_point,
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert seen["source_entry"] is not None
    assert seen["source_entry"]["source_row_index"] == 31
    assert preview["candidate"]["source_row_index"] == 31
    assert preview["candidate"]["selection_reason"] == "mosaic_top_per_branch"


def test_geometry_manual_pick_preview_state_uses_profile_cache_mosaic_top() -> None:
    key = ("q_group", "primary", 1, 2)
    clicked_seed = {
        "label": "near",
        "sim_col": 10.0,
        "sim_row": 20.0,
        "best_sample_index": 0,
        "source_table_index": 1,
        "source_row_index": 30,
    }
    selected_ray = {
        "label": "top",
        "sim_col": 35.0,
        "sim_row": 30.0,
        "best_sample_index": 1,
        "source_table_index": 1,
        "source_row_index": 31,
    }
    other_branch = {
        "label": "other",
        "sim_col": 10.0,
        "sim_row": 20.2,
        "best_sample_index": 2,
        "source_table_index": 1,
        "source_row_index": 41,
    }
    profile_cache = {
        "beam_x_array": np.asarray([0.5, 0.5, -0.5], dtype=float),
        "sample_weights": np.asarray([0.1, 0.9, 9.0], dtype=float),
    }
    seen: dict[str, object] = {}

    def _refine_preview_point(source_entry, raw_col, raw_row, **_kwargs):
        seen["source_entry"] = dict(source_entry) if source_entry else None
        return float(raw_col), float(raw_row)

    entries = [dict(clicked_seed), dict(selected_ray), dict(other_branch)]
    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session={
            "group_key": key,
            "q_label": "test group",
            "group_entries": [dict(entry) for entry in entries],
            "background_index": 0,
            "tagged_candidate_key": _source_key(clicked_seed),
        },
        current_background_index=0,
        force=True,
        remaining_candidates=entries,
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=_refine_preview_point,
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
        profile_cache=profile_cache,
    )

    assert preview is not None
    assert all("mosaic_weight" not in entry for entry in entries)
    assert seen["source_entry"]["source_row_index"] == 31
    assert seen["source_entry"]["branch_id"] == "+x"
    assert seen["source_entry"]["selection_reason"] == "mosaic_top_per_branch"
    assert seen["source_entry"]["mosaic_weight"] == 0.9
    assert preview["candidate"]["source_row_index"] == 31


def test_geometry_manual_pick_preview_state_colors_from_match_confidence() -> None:
    background = np.zeros((33, 33), dtype=float)
    background[10, 10] = 1000.0
    background[10, 11] = 600.0
    background[11, 10] = 600.0
    state = {
        "match_config": {},
        "background_context": build_background_peak_context(background),
    }

    preview = mg.geometry_manual_pick_preview_state(
        24.0,
        24.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "q_label": "test group",
            "group_entries": [{"label": "1,0,0"}],
            "background_index": 0,
        },
        current_background_index=0,
        force=True,
        remaining_candidates=[
            {"label": "1,0,0", "sim_col": 10.0, "sim_row": 10.0},
        ],
        display_background=background,
        cache_data=state,
        refine_preview_point=lambda *_args, **_kwargs: (10.0, 10.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert np.isfinite(preview["match_confidence"])
    assert preview["preview_color"] == mg.geometry_manual_preview_confidence_color(
        preview["match_confidence"]
    )
    assert preview["preview_color"] != mg.geometry_manual_preview_color(preview["sigma_px"])
    assert "confidence=" in preview["message"]


def test_geometry_manual_preview_color_transitions_from_green_to_red() -> None:
    assert mg.geometry_manual_preview_color(0.75) == "#2ecc71"
    assert mg.geometry_manual_preview_color(12.0) == "#e74c3c"
    assert mg.geometry_manual_preview_color(6.0) not in {"#2ecc71", "#e74c3c"}


def test_geometry_manual_preview_confidence_color_transitions_from_red_to_green() -> None:
    assert mg.geometry_manual_preview_confidence_color(0.1) == "#e74c3c"
    assert mg.geometry_manual_preview_confidence_color(1.0) == "#2ecc71"
    assert mg.geometry_manual_preview_confidence_color(0.5) not in {
        "#2ecc71",
        "#e74c3c",
    }


def test_geometry_manual_preview_quality_label_tracks_sigma() -> None:
    assert mg.geometry_manual_preview_quality_label(0.75) == "good"
    assert mg.geometry_manual_preview_quality_label(4.0) == "warning"
    assert mg.geometry_manual_preview_quality_label(12.0) == "bad"


def test_geometry_manual_preview_confidence_quality_label_tracks_confidence() -> None:
    assert mg.geometry_manual_preview_confidence_quality_label(1.1) == "good"
    assert mg.geometry_manual_preview_confidence_quality_label(0.5) == "warning"
    assert mg.geometry_manual_preview_confidence_quality_label(0.1) == "bad"


def test_geometry_manual_session_initial_pairs_display_includes_pending_bg_points() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
                "qr": 1.2345678901,
                "qz": 2.3456789012,
            }
        ],
        "pending_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "x": 9.0,
                "y": 10.0,
            }
        ],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert len(entries) == 1
    assert entries[0]["sim_display"] == (5.0, 6.0)
    assert entries[0]["bg_display"] == (9.0, 10.0)
    assert entries[0]["qr"] == 1.2345678901
    assert entries[0]["qz"] == 2.3456789012


def test_geometry_manual_session_initial_pairs_display_projects_active_sim_display_geometry() -> (
    None
):
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
                "qr": 1.0,
                "qz": 2.0,
            }
        ],
        "pending_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "x": 9.0,
                "y": 10.0,
            }
        ],
        "background_index": 0,
    }
    projection_calls: list[dict[str, object]] = []

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        project_peaks_to_current_view=lambda entries: [
            (
                projection_calls.append(dict(entry))
                or {
                    **dict(entry),
                    "sim_col": 50.0,
                    "sim_row": 60.0,
                    "qr": 3.25,
                    "qz": 4.75,
                }
            )
            for entry in entries or []
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert len(projection_calls) == 1
    assert projection_calls[0]["sim_col"] == 5.0
    assert projection_calls[0]["sim_row"] == 6.0
    assert len(entries) == 1
    assert entries[0]["sim_display"] == (50.0, 60.0)
    assert entries[0]["bg_display"] == (9.0, 10.0)
    assert entries[0]["qr"] == 3.25
    assert entries[0]["qz"] == 4.75


def test_geometry_manual_session_initial_pairs_display_keeps_detector_sim_pixel_over_background_rotation() -> (
    None
):
    background_shape = (100, 200)
    native_point = (20.0, 10.0)
    background_rotated_point = mg._default_rotate_point(
        native_point[0],
        native_point[1],
        background_shape,
        -1,
    )
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": native_point[0],
                "sim_row": native_point[1],
                "native_col": native_point[0],
                "native_row": native_point[1],
            }
        ],
        "pending_entries": [],
        "background_index": 0,
    }
    refresh_calls: list[dict[str, object]] = []

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        refresh_entry_geometry=lambda entry: (
            refresh_calls.append(dict(entry or {}))
            or {
                **dict(entry or {}),
                "x": background_rotated_point[0],
                "y": background_rotated_point[1],
            }
        ),
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert refresh_calls == []
    assert len(entries) == 1
    assert entries[0]["sim_display"] == native_point
    assert entries[0]["sim_display"] != background_rotated_point


def test_geometry_manual_session_initial_pairs_display_refreshes_projected_detector_rows() -> None:
    background_shape = (100, 200)
    native_point = (20.0, 10.0)
    detector_display = mg._default_rotate_point(
        native_point[0],
        native_point[1],
        background_shape,
        -1,
    )
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "display_col": detector_display[0],
                "display_row": detector_display[1],
                "sim_col": detector_display[0],
                "sim_row": detector_display[1],
                "sim_col_raw": detector_display[0],
                "sim_row_raw": detector_display[1],
                "native_col": native_point[0],
                "native_row": native_point[1],
                "sim_native_x": native_point[0],
                "sim_native_y": native_point[1],
                "caked_x": 35.0,
                "caked_y": -20.0,
                "raw_caked_x": 35.0,
                "raw_caked_y": -20.0,
            }
        ],
        "pending_entries": [],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        refresh_entry_geometry=lambda entry: mg.refresh_geometry_manual_pair_entry(
            entry,
            background_display_shape=background_shape,
            background_display_to_native_detector_coords=lambda col, row: mg._default_rotate_point(
                float(col),
                float(row),
                background_shape,
                1,
            ),
            rotate_point_for_display=mg._default_rotate_point,
            display_rotate_k=-1,
        ),
        entry_display_coords=lambda entry: (
            float(entry["display_col"]),
            float(entry["display_row"]),
        ),
    )

    assert len(entries) == 1
    assert entries[0]["sim_display"] == detector_display


def test_geometry_manual_session_initial_pairs_display_refreshes_caked_view_when_projection_has_no_caked_coords() -> (
    None
):
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
            }
        ],
        "pending_entries": [],
        "background_index": 0,
    }
    projection_calls: list[dict[str, object]] = []
    refresh_calls: list[dict[str, object]] = []

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=lambda entries: [
            projection_calls.append(dict(entry)) or dict(entry) for entry in entries or []
        ],
        refresh_entry_geometry=lambda entry: (
            refresh_calls.append(dict(entry or {}))
            or {
                **dict(entry or {}),
                "caked_x": 30.0,
                "caked_y": -40.0,
            }
        ),
        entry_display_coords=lambda entry: (float(entry["sim_col"]), float(entry["sim_row"])),
    )

    assert projection_calls == []
    assert refresh_calls == []
    assert len(entries) == 1
    assert "sim_display" not in entries[0]
    assert entries[0]["sim_display_unresolved"] is True


def test_geometry_manual_session_initial_pairs_display_ignores_refresh_errors() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
                "qr": 1.0,
                "qz": 2.0,
            }
        ],
        "pending_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "x": 9.0,
                "y": 10.0,
            }
        ],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        refresh_entry_geometry=lambda _entry: (_ for _ in ()).throw(RuntimeError("refresh failed")),
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert len(entries) == 1
    assert entries[0]["sim_display"] == (5.0, 6.0)
    assert entries[0]["bg_display"] == (9.0, 10.0)
    assert entries[0]["qr"] == 1.0
    assert entries[0]["qz"] == 2.0


def test_geometry_manual_session_initial_pairs_display_uses_caked_coords_for_legacy_group_rows() -> (
    None
):
    session = {
        "group_key": ("q_group", "primary", 1, 5),
        "group_entries": [
            {
                "label": "-1,0,5",
                "hkl": (-1, 0, 5),
                "q_group_key": ("q_group", "primary", 1, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "caked_x": 30.25,
                "caked_y": -57.5,
                "sim_col": 190.0,
                "sim_row": 96.0,
            }
        ],
        "pending_entries": [],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        use_caked_display=True,
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert len(entries) == 1
    assert "sim_display" not in entries[0]
    assert entries[0]["sim_display_unresolved"] is True


def test_geometry_manual_session_initial_pairs_display_resolves_colliding_pending_branch_entries() -> (
    None
):
    session = {
        "group_key": ("q_group", "primary", 1, 5),
        "group_entries": [
            {
                "label": "target-right",
                "hkl": (-1, 0, 5),
                "source_table_index": 1,
                "source_row_index": 3,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "sim_col": 5.0,
                "sim_row": 6.0,
            }
        ],
        "pending_entries": [
            {
                "label": "other-right",
                "hkl": (-2, 0, 5),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "x": 9.0,
                "y": 10.0,
            },
            {
                "label": "target-right",
                "hkl": (-1, 0, 5),
                "source_table_index": 1,
                "source_row_index": 3,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "x": 19.0,
                "y": 21.0,
            },
        ],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert len(entries) == 1
    assert entries[0]["bg_display"] == (19.0, 21.0)


def test_refresh_geometry_manual_pick_session_candidates_keeps_one_branch_row_without_branch_hints() -> (
    None
):
    session = {
        "group_key": ("q_group", "primary", 1, 2),
        "group_entries": [
            {
                "label": "old-left",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
            },
            {
                "label": "old-right",
                "source_table_index": 1,
                "source_row_index": 3,
                "sim_col": 15.0,
                "sim_row": 16.0,
            },
        ],
        "pending_entries": [
            {
                "label": "old-right",
                "source_table_index": 1,
                "source_row_index": 3,
                "x": 20.0,
                "y": 30.0,
            }
        ],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source", 1, 3),
        "tagged_candidate": {
            "label": "old-right",
            "source_table_index": 1,
            "source_row_index": 3,
            "sim_col": 15.0,
            "sim_row": 16.0,
        },
    }

    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        session,
        grouped_candidates={
            ("q_group", "primary", 1, 2): [
                {
                    "label": "new-left",
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "sim_col": 50.0,
                    "sim_row": 60.0,
                },
                {
                    "label": "new-right",
                    "source_table_index": 1,
                    "source_row_index": 3,
                    "sim_col": 70.0,
                    "sim_row": 80.0,
                },
            ]
        },
        cache_signature=("sim", 2),
    )

    assert refreshed["cache_signature"] == ("sim", 2)
    assert refreshed["target_count"] == 1
    assert len(refreshed["group_entries"]) == 1
    assert refreshed["group_entries"][0]["label"] == "new-left"
    assert refreshed["group_entries"][0]["sim_col"] == 50.0
    assert refreshed["group_entries"][0]["selection_reason"] == "mosaic_top_per_branch"
    assert "tagged_candidate" not in refreshed
    assert refreshed["tagged_candidate_key"] == ("source", 1, 3)
    assert refreshed["pending_entries"][0]["source_row_index"] == 3


def test_refresh_geometry_manual_pick_session_candidates_keeps_one_top_row_per_branch() -> None:
    group_key = ("q_group", "primary", 4, 2)
    low_tagged = {
        "label": "low-tagged",
        "q_group_key": group_key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 40,
        "source_reflection_key": ("full", 40),
        "source_ray_id": "low-ray",
        "hkl": (4, 0, 2),
        "mosaic_weight": 0.2,
        "source_table_index": 1,
        "source_row_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    top = {
        "label": "top",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 41,
        "source_reflection_key": ("full", 41),
        "source_ray_id": "top-ray",
        "hkl": (-4, 0, 2),
        "mosaic_weight": 0.95,
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    session = {
        "group_key": group_key,
        "group_entries": [dict(low_tagged)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 1,
        "tagged_candidate_key": ("source_branch", 40, 0),
        "tagged_candidate": dict(low_tagged),
    }

    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        session,
        grouped_candidates={group_key: [dict(low_tagged), dict(top)]},
        cache_signature=("sig", "refresh"),
    )

    assert refreshed["target_count"] == 2
    assert len(refreshed["group_entries"]) == 2
    by_branch = {entry["branch_id"]: entry for entry in refreshed["group_entries"]}
    assert by_branch["+x"]["label"] == "low-tagged"
    assert by_branch["+x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["+x"]["source_branch_index"] == 0
    assert by_branch["+x"]["source_reflection_index"] == 40
    assert by_branch["+x"]["source_ray_id"] == "low-ray"
    assert by_branch["-x"]["label"] == "top"
    assert by_branch["-x"]["mosaic_weight"] == 0.95
    assert by_branch["-x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["-x"]["source_branch_index"] == 1
    assert by_branch["-x"]["source_reflection_index"] == 41
    assert by_branch["-x"]["source_reflection_key"] == ("full", 41)
    assert by_branch["-x"]["source_ray_id"] == "top-ray"
    assert refreshed["tagged_candidate"]["label"] == "low-tagged"
    displayed = mg.geometry_manual_session_initial_pairs_display(
        refreshed,
        current_background_index=0,
        entry_display_coords=lambda entry: (
            float(entry.get("sim_col")),
            float(entry.get("sim_row")),
        ),
    )
    assert len([entry for entry in displayed if entry.get("q_group_key") == group_key]) == 2


def test_refresh_geometry_manual_pick_session_candidates_keeps_tagged_identity_under_permutation() -> (
    None
):
    group_key = ("q_group", "primary", 1, 2)
    tagged_candidate = {
        "label": "right",
        "hkl": (-1, 0, 2),
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 1,
        "source_row_index": 3,
        "mosaic_weight": 0.9,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    other_candidate = {
        "label": "left",
        "hkl": (1, 0, 2),
        "source_reflection_index": 8,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 1,
        "source_row_index": 2,
        "mosaic_weight": 0.2,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(other_candidate), dict(tagged_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source_branch", 9, 1),
        "tagged_candidate": dict(tagged_candidate),
    }

    forward = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(other_candidate), dict(tagged_candidate)]},
        cache_signature=("sig", 1),
    )
    reversed_order = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(tagged_candidate), dict(other_candidate)]},
        cache_signature=("sig", 1),
    )

    for refreshed in (forward, reversed_order):
        assert refreshed["target_count"] == 2
        assert len(refreshed["group_entries"]) == 2
        assert refreshed["tagged_candidate_key"] == ("source_branch", 9, 1)
        assert refreshed["tagged_candidate"]["source_reflection_index"] == 9
        assert refreshed["tagged_candidate"]["source_reflection_namespace"] == ("full_reflection")
        assert refreshed["tagged_candidate"]["source_reflection_is_full"] is True
        assert refreshed["tagged_candidate"]["source_branch_index"] == 1
        assert refreshed["tagged_candidate"]["selection_reason"] == "mosaic_top_per_branch"
        by_branch = {entry["source_branch_index"]: entry for entry in refreshed["group_entries"]}
        assert by_branch[1]["source_reflection_index"] == 9
        assert by_branch[1]["source_branch_index"] == 1
        assert by_branch[0]["source_reflection_index"] == 8


def test_refresh_geometry_manual_pick_session_candidates_keeps_legacy_tagged_identity_under_permutation() -> (
    None
):
    group_key = ("q_group", "primary", 1, 5)
    tagged_candidate = {
        "label": "target-right",
        "hkl": (-1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 3,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "mosaic_weight": 0.9,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    sibling_candidate = {
        "label": "other-right",
        "hkl": (-2, 0, 5),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "mosaic_weight": 0.2,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(sibling_candidate), dict(tagged_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source_branch", 1, 1),
        "tagged_candidate": dict(tagged_candidate),
    }

    forward = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(sibling_candidate), dict(tagged_candidate)]},
        cache_signature=("sig", 1),
    )
    reversed_order = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(tagged_candidate), dict(sibling_candidate)]},
        cache_signature=("sig", 1),
    )

    for refreshed in (forward, reversed_order):
        assert refreshed["target_count"] == 1
        assert len(refreshed["group_entries"]) == 1
        assert refreshed["group_entries"][0]["label"] == "target-right"
        assert refreshed["group_entries"][0]["source_row_index"] == 3
        assert refreshed["tagged_candidate"]["label"] == "target-right"
        assert refreshed["tagged_candidate"]["source_row_index"] == 3
        assert refreshed["tagged_candidate"]["selection_reason"] == "mosaic_top_per_branch"


def test_refresh_geometry_manual_pick_session_candidates_does_not_rebind_missing_stored_tagged_candidate_by_key() -> (
    None
):
    group_key = ("q_group", "primary", 1, 5)
    left_candidate = {
        "label": "left",
        "hkl": (1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 1,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    sibling_candidate = {
        "label": "other-right",
        "hkl": (-2, 0, 5),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(left_candidate), dict(sibling_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source_branch", 1, 1),
        "tagged_candidate": {
            "label": "missing-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 99,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 99.0,
            "sim_row": 100.0,
        },
    }

    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(left_candidate), dict(sibling_candidate)]},
        cache_signature=("sig", 1),
    )

    assert refreshed["target_count"] == 2
    assert len(refreshed["group_entries"]) == 2
    assert refreshed["group_entries"][0]["label"] == "left"
    assert "tagged_candidate" not in refreshed
    assert refreshed["tagged_candidate_key"] == ("source_branch", 1, 1)


def test_refresh_geometry_manual_pick_session_candidates_does_not_rebind_missing_nonbranch_stored_tagged_candidate_by_key() -> (
    None
):
    group_key = ("q_group", "primary", 1, 5)
    left_candidate = {
        "label": "left",
        "hkl": (1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    sibling_candidate = {
        "label": "other-right",
        "hkl": (-2, 0, 5),
        "source_table_index": 1,
        "source_row_index": 3,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(left_candidate), dict(sibling_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source", 1, 3),
        "tagged_candidate": {
            "label": "missing-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 3,
            "sim_col": 99.0,
            "sim_row": 100.0,
        },
    }

    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(left_candidate), dict(sibling_candidate)]},
        cache_signature=("sig", 1),
    )

    assert refreshed["target_count"] == 2
    assert len(refreshed["group_entries"]) == 2
    assert refreshed["group_entries"][0]["label"] == "left"
    assert "tagged_candidate" not in refreshed
    assert refreshed["tagged_candidate_key"] == ("source", 1, 3)


def test_refresh_geometry_manual_pick_session_candidates_keeps_cleared_identity_lock_on_second_refresh() -> (
    None
):
    group_key = ("q_group", "primary", 1, 5)
    left_candidate = {
        "label": "left",
        "hkl": (1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    sibling_candidate = {
        "label": "other-right",
        "hkl": (-2, 0, 5),
        "source_table_index": 1,
        "source_row_index": 3,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(left_candidate), dict(sibling_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source", 1, 3),
        "tagged_candidate": {
            "label": "missing-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 1,
            "source_row_index": 3,
            "sim_col": 99.0,
            "sim_row": 100.0,
        },
    }

    first = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(left_candidate), dict(sibling_candidate)]},
        cache_signature=("sig", 1),
    )
    second = mg.refresh_geometry_manual_pick_session_candidates(
        first,
        grouped_candidates={group_key: [dict(left_candidate), dict(sibling_candidate)]},
        cache_signature=("sig", 2),
    )

    assert first["_tagged_candidate_requires_identity"] is True
    assert "tagged_candidate" not in first
    assert second["_tagged_candidate_requires_identity"] is True
    assert second["cache_signature"] == ("sig", 2)
    assert second["target_count"] == 2
    assert len(second["group_entries"]) == 2
    assert second["group_entries"][0]["label"] == "left"
    assert "tagged_candidate" not in second


def test_refresh_geometry_manual_pick_session_candidates_preserves_legacy_key_only_fallback() -> (
    None
):
    group_key = ("q_group", "primary", 1, 5)
    left_candidate = {
        "label": "left",
        "hkl": (1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    right_candidate = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_table_index": 1,
        "source_row_index": 3,
        "sim_col": 30.0,
        "sim_row": 40.0,
    }
    pick_session = {
        "group_key": group_key,
        "group_entries": [dict(left_candidate), dict(right_candidate)],
        "pending_entries": [],
        "background_index": 0,
        "target_count": 2,
        "tagged_candidate_key": ("source", 1, 3),
    }

    first = mg.refresh_geometry_manual_pick_session_candidates(
        pick_session,
        grouped_candidates={group_key: [dict(left_candidate), dict(right_candidate)]},
        cache_signature=("sig", 1),
    )
    second = mg.refresh_geometry_manual_pick_session_candidates(
        first,
        grouped_candidates={group_key: [dict(left_candidate), dict(right_candidate)]},
        cache_signature=("sig", 2),
    )

    for refreshed in (first, second):
        assert refreshed["_tagged_candidate_requires_identity"] is False
        assert refreshed["target_count"] == 2
        assert len(refreshed["group_entries"]) == 2
        assert any(entry["label"] == "right" for entry in refreshed["group_entries"])
        assert refreshed["tagged_candidate"]["label"] == "right"
        assert refreshed["tagged_candidate_key"] == ("source", 1, 3)


def test_cancel_geometry_manual_pick_session_clears_session_and_triggers_callbacks() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "background_index": 0,
    }
    calls: list[tuple[str, object]] = []

    cleared = mg.cancel_geometry_manual_pick_session(
        session,
        current_background_index=0,
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=lambda text: calls.append(("status", text)),
        message="done",
    )

    assert cleared == {}
    assert ("restore", False) in calls
    assert ("clear", False) in calls
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert ("status", "done") in calls


def test_cancel_geometry_manual_pick_session_skips_restore_in_detector_mode() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "background_index": 0,
    }
    calls: list[tuple[str, object]] = []

    cleared = mg.cancel_geometry_manual_pick_session(
        session,
        current_background_index=0,
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=lambda text: calls.append(("status", text)),
        message="done",
        use_caked_space=False,
    )

    assert cleared == {}
    assert ("restore", False) not in calls
    assert ("clear", False) in calls
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert ("status", "done") in calls


def test_match_geometry_manual_group_to_background_builds_source_lookup() -> None:
    matches = mg.match_geometry_manual_group_to_background(
        [{"label": "1,0,0", "source_table_index": 3, "source_row_index": 8}],
        background_image=np.zeros((8, 8), dtype=float),
        cache_data={
            "match_config": {"search_radius_px": 12.0},
            "background_context": {"img_valid": True},
        },
        match_simulated_peaks_to_peak_context=lambda entries, _context, _cfg: (
            [
                {
                    "source_table_index": entries[0]["source_table_index"],
                    "source_row_index": entries[0]["source_row_index"],
                    "x": 1.5,
                    "y": 2.5,
                }
            ],
            {},
        ),
    )

    assert matches == {("source", 3, 8): (1.5, 2.5)}


def test_geometry_manual_pick_cache_signature_tracks_background_state() -> None:
    placed_signature = mg.geometry_manual_pick_placed_cache_signature(
        source_snapshot_signature=("sim", 7),
        background_index=2,
        background_image=np.zeros((6, 5), dtype=np.float32),
        use_caked_space=True,
    )
    signature = mg.geometry_manual_pick_cache_signature(
        placed_cache_signature=placed_signature,
        disabled_qr_sets=[("primary", 1)],
        disabled_qz_sections=[("primary", 1, 0)],
    )

    assert placed_signature[0] == ("sim", 7)
    assert placed_signature[1] == 2
    assert placed_signature[2] is True
    assert placed_signature[3][1] == (6, 5)
    assert placed_signature[3][3] == "float32"
    assert signature[0] == placed_signature
    assert signature[1] == (("primary", 1),)
    assert signature[2] == (("primary", 1, 0),)


def test_build_geometry_manual_pick_cache_reuses_existing_current_background_state() -> None:
    existing_cache = {
        "signature": ("cached",),
        "value": 9,
        "grouped_candidates": {("q_group", "primary", 1, 0): [{"sim_col": 1.0}]},
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((3, 3), dtype=float),
        existing_cache_signature=("cached",),
        existing_cache_data=existing_cache,
        cache_signature_fn=lambda **_kwargs: ("cached",),
        simulated_peaks_for_params=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache miss")
        ),
        build_grouped_candidates=lambda _entries: {},
        build_simulated_lookup=lambda _entries: {},
        current_match_config=lambda: {},
    )

    assert cache_data is existing_cache
    assert next_sig == ("cached",)
    assert next_state is existing_cache


def test_build_geometry_manual_pick_cache_rebuilds_when_cached_groups_are_empty() -> None:
    simulation_calls: list[bool] = []
    existing_cache_data = {
        "grouped_candidates": {},
        "simulated_lookup": {},
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((3, 3), dtype=float),
        existing_cache_signature=("cached",),
        existing_cache_data=existing_cache_data,
        cache_signature_fn=lambda **_kwargs: ("cached",),
        simulated_peaks_for_params=lambda *_args, prefer_cache=False, **_kwargs: (
            simulation_calls.append(bool(prefer_cache))
            or [
                {
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "sim_col": 3.0,
                    "sim_row": 4.0,
                }
            ]
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert simulation_calls == []
    assert cache_data is existing_cache_data
    assert cache_data["grouped_candidates"] == {}
    assert next_sig == ("cached",)
    assert next_state is existing_cache_data


def test_build_geometry_manual_pick_cache_prefers_cached_preview_groups_when_cache_is_preferred() -> (
    None
):
    forwarded_prefer_cache: list[bool] = []

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((3, 3), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache))
            or (
                [
                    {
                        "q_group_key": ("q_group", "primary", 1, 0),
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "sim_col": 3.0,
                        "sim_row": 4.0,
                    }
                ]
                if prefer_cache
                else [
                    {
                        "q_group_key": ("q_group", "primary", 2, 0),
                        "source_table_index": 9,
                        "source_row_index": 8,
                        "sim_col": 30.0,
                        "sim_row": 40.0,
                    }
                ]
            )
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True]
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col"] == 3.0
    assert ("q_group", "primary", 1, 0) in cache_data["grouped_candidates"]
    assert next_sig == ("sig",)
    assert next_state["simulated_lookup"][(1, 2)]["sim_row"] == 4.0
    assert cache_data["cache_metadata"] == {
        "cache_action": "reused",
        "reused": True,
        "rebuilt": False,
        "stale_reason": None,
        "cache_source": "geometry_manual_simulated_peaks_for_params(prefer_cache=True)",
        "cache_provenance": [
            "geometry_manual_simulated_peaks_for_params(prefer_cache=True)",
            "build_grouped_candidates",
            "build_simulated_lookup",
        ],
        "prefer_cache": True,
        "background_index": 0,
        "current_background_index": 0,
        "simulated_peak_count": 1,
        "group_count": 1,
        "table_count": 1,
        "table_summaries": [
            {
                "source_table_index": 1,
                "nominal_hkl": None,
                "q_group_key": ["q_group", "primary", 1, 0],
                "qr": None,
                "qz": None,
                "row_count_before_grouping": 1,
                "row_count_after_grouping": 1,
                "dropped_nonfinite_row_count": 0,
                "nominal_hkl_recovery_count": 0,
                "merged_group_count": 0,
                "representative_row_indices_kept": [2],
            }
        ],
    }


def test_build_geometry_manual_pick_cache_falls_back_to_central_simulation_when_cached_groups_are_empty() -> (
    None
):
    forwarded_prefer_cache: list[bool] = []

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((3, 3), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache))
            or (
                []
                if prefer_cache
                else [
                    {
                        "q_group_key": ("q_group", "primary", 1, 0),
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "sim_col": 3.0,
                        "sim_row": 4.0,
                    }
                ]
            )
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col"] == 3.0
    assert ("q_group", "primary", 1, 0) in cache_data["grouped_candidates"]
    assert next_sig == ("sig",)
    assert next_state["simulated_lookup"][(1, 2)]["sim_row"] == 4.0
    assert cache_data["cache_metadata"]["cache_action"] == "rebuilt"
    assert cache_data["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
    )
    assert cache_data["cache_metadata"]["stale_reason"] == "cached preview rows were empty."
    assert cache_data["cache_metadata"]["table_summaries"][0][
        "representative_row_indices_kept"
    ] == [2]


def test_build_geometry_manual_pick_cache_reprojects_existing_rows_when_only_background_state_changes() -> (
    None
):
    forwarded_prefer_cache: list[bool] = []
    cached_entry = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 13.0,
        "sim_row": 2.0,
        "sim_col_raw": 3.0,
        "sim_row_raw": 4.0,
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            True,
            ("old-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                True,
                ("old-bg",),
                1,
                ("('q_group', 'primary', 1, 0)",),
            ),
            "simulated_peaks": [dict(cached_entry)],
            "simulated_lookup": {(1, 2): dict(cached_entry)},
            "grouped_candidates": {("q_group", "primary", 1, 0): [dict(cached_entry)]},
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            True,
            ("new-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache)) or []
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        project_peaks_to_current_view=lambda entries: [
            {
                **dict(entry),
                "sim_col": 30.0,
                "sim_row": 40.0,
                "caked_x": 30.0,
                "caked_y": 40.0,
            }
            for entry in entries or ()
            if isinstance(entry, dict)
        ],
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert ("q_group", "primary", 1, 0) in cache_data["grouped_candidates"]
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col"] == 30.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_row"] == 40.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col_raw"] == 3.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_row_raw"] == 4.0
    assert next_sig[3] == ("new-bg",)
    assert next_state["grouped_candidates"][("q_group", "primary", 1, 0)][0]["sim_row"] == 40.0
    assert cache_data["cache_metadata"]["cache_action"] == "reused"
    assert cache_data["cache_metadata"]["cache_source"] == (
        "existing_cache_data.simulated_peaks(reprojected)"
    )
    assert cache_data["cache_metadata"]["cache_provenance"] == [
        "existing_cache_data.simulated_peaks",
        "project_peaks_to_current_view",
        "build_grouped_candidates",
        "build_simulated_lookup",
    ]
    assert cache_data["cache_metadata"]["stale_reason"] == (
        "background-only cache signature change; reprojected cached simulated peaks."
    )
    assert cache_data["cache_metadata"]["table_summaries"][0][
        "representative_row_indices_kept"
    ] == [2]


def test_build_geometry_manual_pick_cache_background_churn_reuse_requires_raw_coords(
    monkeypatch,
) -> None:
    forwarded_prefer_cache: list[bool] = []
    project_calls: list[int] = []
    cached_entry = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 13.0,
        "sim_row": 2.0,
    }
    bundle = _dummy_transform_bundle(detector_shape=(8, 8))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        _fail_projection_legacy_path(
            "projection callback should not run when raw coords are missing"
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    def _recording_project_peaks_to_current_view(entries):
        cached_rows = [dict(entry) for entry in entries or () if isinstance(entry, dict)]
        project_calls.append(len(cached_rows))
        return callbacks.project_peaks_to_current_view(entries)

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            False,
            ("old-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                False,
                ("old-bg",),
                1,
                ("('q_group', 'primary', 1, 0)",),
            ),
            "simulated_peaks": [dict(cached_entry)],
            "simulated_lookup": {(1, 2): dict(cached_entry)},
            "grouped_candidates": {("q_group", "primary", 1, 0): [dict(cached_entry)]},
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            False,
            ("new-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache)) or []
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        project_peaks_to_current_view=_recording_project_peaks_to_current_view,
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert project_calls == []
    assert cache_data["grouped_candidates"] == {}
    assert cache_data["simulated_lookup"] == {}
    assert next_sig[3] == ("new-bg",)
    assert next_state["grouped_candidates"] == {}
    assert cache_data["cache_metadata"]["cache_action"] == "rebuilt"
    assert cache_data["cache_metadata"]["cache_source"] == ("source_snapshot_unavailable")
    assert cache_data["cache_metadata"]["stale_reason"] == (
        "background-only cache signature change; cached simulated peaks could not be reprojected."
    )


def test_build_geometry_manual_pick_cache_background_churn_reuse_requires_valid_caked_projection() -> (
    None
):
    forwarded_prefer_cache: list[bool] = []
    cached_entry = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 13.0,
        "sim_row": 2.0,
        "sim_col_raw": 3.0,
        "sim_row_raw": 4.0,
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            True,
            ("old-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                True,
                ("old-bg",),
                1,
                ("('q_group', 'primary', 1, 0)",),
            ),
            "simulated_peaks": [dict(cached_entry)],
            "simulated_lookup": {(1, 2): dict(cached_entry)},
            "grouped_candidates": {("q_group", "primary", 1, 0): [dict(cached_entry)]},
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            True,
            ("new-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache)) or []
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        project_peaks_to_current_view=lambda entries: [
            {
                **dict(entry),
                "sim_col": 30.0,
                "sim_row": 40.0,
            }
            for entry in entries or ()
            if isinstance(entry, dict)
        ],
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert cache_data["grouped_candidates"] == {}
    assert cache_data["simulated_lookup"] == {}
    assert next_sig[3] == ("new-bg",)
    assert next_state["grouped_candidates"] == {}
    assert cache_data["cache_metadata"]["cache_action"] == "rebuilt"
    assert cache_data["cache_metadata"]["cache_source"] == ("source_snapshot_unavailable")
    assert cache_data["cache_metadata"]["stale_reason"] == (
        "background-only cache signature change; cached simulated peaks could not be reprojected."
    )


def test_build_geometry_manual_pick_cache_rebuilds_before_reusing_existing_groups_on_background_churn() -> (
    None
):
    forwarded_prefer_cache: list[bool] = []
    stale_entry = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 13.0,
        "sim_row": 2.0,
        "candidate_source": "peak_records",
    }
    rebuilt_entry = {
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 30.0,
        "sim_row": 40.0,
        "candidate_source": "fresh_rebuild",
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            True,
            ("old-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                True,
                ("old-bg",),
                1,
                ("('q_group', 'primary', 1, 0)",),
            ),
            "simulated_peaks": [dict(stale_entry)],
            "simulated_lookup": {(1, 2): dict(stale_entry)},
            "grouped_candidates": {("q_group", "primary", 1, 0): [dict(stale_entry)]},
            "cache_metadata": {"cache_source": "peak_records"},
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            True,
            ("new-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache))
            or ([] if prefer_cache else [dict(rebuilt_entry)])
        ),
        build_grouped_candidates=lambda entries: {
            entry["q_group_key"]: [dict(entry)]
            for entry in entries or ()
            if isinstance(entry.get("q_group_key"), tuple)
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert ("q_group", "primary", 1, 0) in cache_data["grouped_candidates"]
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col"] == 30.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_row"] == 40.0
    assert cache_data["simulated_lookup"][(1, 2)]["candidate_source"] == ("fresh_rebuild")
    assert (
        cache_data["grouped_candidates"][("q_group", "primary", 1, 0)][0]["candidate_source"]
        == "fresh_rebuild"
    )
    assert next_sig[3] == ("new-bg",)
    assert next_state["simulated_lookup"][(1, 2)]["candidate_source"] == ("fresh_rebuild")
    assert cache_data["cache_metadata"]["cache_action"] == "rebuilt"
    assert cache_data["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
    )


def test_build_geometry_manual_initial_pairs_display_uses_cache_lookup() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {"sim_col": 13.5, "sim_row": 15.5},
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (13.5, 15.5),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_saved_refined_caked_coords() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "caked_x": 109.0,
                "caked_y": -11.0,
                "refined_sim_x": 13.5,
                "refined_sim_y": 15.5,
                "refined_sim_caked_x": 113.5,
                "refined_sim_caked_y": -12.5,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (109.0, -11.0),
            "sim_display_unresolved": True,
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_native_only_detector_rows() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "native_col": 105.0,
                    "native_row": 206.0,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_raw_detector_only_rows() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "sim_col_raw": 105.0,
                    "sim_row_raw": 206.0,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_detector_xy_only_lookup_rows() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "x": 105.0,
                    "y": 206.0,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_detector_xy_lookup_rows_when_they_only_match_caked_view() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "x": 30.25,
                    "y": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_prefers_raw_detector_over_stale_caked_display() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "sim_col_raw": 105.0,
                    "sim_row_raw": 206.0,
                    "display_col": 30.25,
                    "display_row": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_prefers_raw_detector_over_detector_xy_when_caked_present() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "x": 30.25,
                    "y": -57.5,
                    "sim_col_raw": 105.0,
                    "sim_row_raw": 206.0,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_accepts_display_only_lookup_rows_without_caked_markers() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "display_col": 30.25,
                    "display_row": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (30.25, -57.5),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_accepts_display_only_lookup_rows_without_caked_markers_for_non_native_saved_pairs() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "display_col": 30.25,
                    "display_row": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (30.25, -57.5),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_display_only_lookup_rows_with_caked_markers() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "display_col": 30.25,
                    "display_row": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_stale_caked_sim_coords_for_native_saved_pairs() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "sim_col": 30.25,
                    "sim_row": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_stale_caked_sim_coords_for_non_native_saved_pairs() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "sim_col": 30.25,
                    "sim_row": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_simulation_native_lookup_rows() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "simulated_x": 105.0,
                    "simulated_y": 206.0,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_rejects_simulation_native_lookup_rows_when_they_only_match_caked_view() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "refined_sim_native_x": 105.0,
                "refined_sim_native_y": 206.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "simulated_x": 30.25,
                    "simulated_y": -57.5,
                    "caked_x": 30.25,
                    "caked_y": -57.5,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_simulation_native_lookup_rows_for_non_native_saved_pairs() -> (
    None
):
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(
                    {
                        "source_table_index": 4,
                        "source_row_index": 7,
                    }
                ): {
                    "simulated_x": 105.0,
                    "simulated_y": 206.0,
                },
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (105.0, 206.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_branch_aware_cache_lookup_for_detector_view() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 182.0,
        "y": 138.0,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    left_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
    }
    right_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_branch_index"] == 1
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_branch_aware_detector_sim_with_caked_fields() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 182.0,
        "y": 138.0,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    left_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
        "caked_x": 28.0,
        "caked_y": -56.5,
    }
    right_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
        "caked_x": 30.25,
        "caked_y": -57.5,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_branch_index"] == 1
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_prefers_live_detector_candidate_over_saved_display_only_overlay() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 182.0,
        "y": 138.0,
        "display_col": 30.25,
        "display_row": -57.5,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    live_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(live_candidate): dict(live_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_branch_index"] == 1
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_matches_legacy_branch_entry_to_current_lookup() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 182.0,
        "y": 138.0,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    left_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
    }
    right_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_branch_index"] == 1
    assert "source_reflection_index" not in measured_display[0]
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_matches_branchless_legacy_entry_by_hkl() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 182.0,
        "y": 138.0,
        "source_table_index": 9,
        "source_row_index": 0,
    }
    left_candidate = {
        "label": "1,0,5",
        "hkl": (1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
    }
    right_candidate = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_row_index"] == 0
    assert "source_branch_index" not in measured_display[0]
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_ignores_stale_caked_coords_for_branchless_legacy_entry() -> (
    None
):
    saved_pair = {
        "label": "",
        "x": 190.0,
        "y": 96.0,
        "display_col": 29.0,
        "display_row": -58.5,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "stale_caked_fields": True,
        "source_table_index": 9,
        "source_row_index": 0,
    }
    left_candidate = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "mosaic_weight": 0.2,
    }
    right_candidate = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
        "caked_x": 30.25,
        "caked_y": -57.5,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["stale_caked_fields"] is True
    assert initial_pairs_display[0]["bg_display"] == (190.0, 96.0)
    assert initial_pairs_display[0] == {
        "overlay_match_index": 0,
        "hkl": "",
        "bg_display": (190.0, 96.0),
        "sim_display": (190.0, 96.0),
    }


def test_build_geometry_manual_initial_pairs_display_uses_fresh_caked_coords_for_branchless_legacy_entry() -> (
    None
):
    saved_pair = {
        "label": "",
        "x": 190.0,
        "y": 96.0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "stale_caked_fields": False,
        "source_table_index": 9,
        "source_row_index": 0,
    }
    left_candidate = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 95.0,
        "caked_x": 29.0,
        "caked_y": -58.5,
    }
    right_candidate = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
        "caked_x": 30.25,
        "caked_y": -57.5,
    }

    _measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert initial_pairs_display[0] == {
        "overlay_match_index": 0,
        "hkl": "",
        "bg_display": (190.0, 96.0),
        "sim_display": (190.0, 96.0),
    }


def test_build_geometry_manual_initial_pairs_display_skips_ambiguous_branchless_legacy_entry() -> (
    None
):
    saved_pair = {
        "label": "",
        "x": 185.5,
        "y": 138.0,
        "source_table_index": 9,
        "source_row_index": 0,
    }
    left_candidate = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 181.0,
        "sim_row": 96.0,
    }
    right_candidate = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 190.0,
        "sim_row": 96.0,
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                _source_key(left_candidate): dict(left_candidate),
                _source_key(right_candidate): dict(right_candidate),
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["source_row_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": "",
            "bg_display": (185.5, 138.0),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_uses_branch_aware_cache_lookup_for_caked_view() -> (
    None
):
    saved_pair = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "caked_x": 29.5,
        "caked_y": -58.0,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    left_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "native_col": 188.0,
        "native_row": 95.0,
        "sim_col_raw": 188.0,
        "sim_row_raw": 95.0,
        "display_col": 29.0,
        "display_row": -58.5,
        "caked_x": 29.0,
        "caked_y": -58.5,
    }
    right_candidate = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "native_col": 190.0,
        "native_row": 96.0,
        "sim_col_raw": 190.0,
        "sim_row_raw": 96.0,
        "display_col": 30.25,
        "display_row": -57.5,
        "caked_x": 30.25,
        "caked_y": -57.5,
    }
    projection_lookup = {
        tuple(left_candidate.get(field) for field in _CROSS_VIEW_ID_FIELDS): dict(left_candidate),
        tuple(right_candidate.get(field) for field in _CROSS_VIEW_ID_FIELDS): dict(right_candidate),
    }

    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {},
            "caked_qr_projection_lookup": projection_lookup,
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert measured_display[0]["source_branch_index"] == 1
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (29.5, -58.0),
            "sim_display": (30.25, -57.5),
        }
    ]


def test_build_geometry_manual_initial_pairs_display_preserves_canonical_pair_between_detector_and_caked_views() -> (
    None
):
    saved_pair = {
        "pair_id": "bg0:pair0",
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 1822.0,
        "y": 1375.0,
        "caked_x": 29.861040445064752,
        "caked_y": -59.079850372490654,
        "refined_sim_x": 1365.0,
        "refined_sim_y": 1168.0,
        "refined_sim_caked_x": 30.608825251597132,
        "refined_sim_caked_y": -58.571417149366184,
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    detector_measured, detector_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )
    caked_measured, caked_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    for measured_display in (detector_measured, caked_measured):
        assert measured_display[0]["pair_id"] == "bg0:pair0"
        assert measured_display[0]["source_reflection_index"] == 9
        assert measured_display[0]["source_reflection_namespace"] == "full_reflection"
        assert measured_display[0]["source_reflection_is_full"] is True
        assert measured_display[0]["source_branch_index"] == 1
        assert measured_display[0]["source_peak_index"] == 1

    assert detector_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (1822.0, 1375.0),
            "sim_display": (1365.0, 1168.0),
        }
    ]
    assert caked_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (29.861040445064752, -59.079850372490654),
            "sim_display_unresolved": True,
        }
    ]


def test_make_runtime_geometry_manual_cache_callbacks_store_cache_state_and_build_pairs() -> None:
    cache_state = {"signature": None, "data": {}}
    simulated_param_sets: list[dict[str, object]] = []

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        disabled_qr_sets=lambda: [],
        disabled_qz_sections=lambda: [],
        filter_active_rows=lambda rows: [dict(entry) for entry in rows or ()],
        stored_max_positions_local=lambda: [{"x": 1.0}],
        stored_peak_table_lattice=lambda: [{"hkl": (1, 0, 0)}],
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda idx: (
            [
                {
                    "label": "1,0,2",
                    "hkl": (1, 0, 2),
                    "x": 9.0,
                    "y": 11.0,
                    "source_table_index": 4,
                    "source_row_index": 7,
                }
            ]
            if int(idx) == 1
            else []
        ),
        simulated_peaks_for_params=lambda params, prefer_cache=True: (
            simulated_param_sets.append(dict(params or {}))
            or [
                {
                    "source_table_index": 4,
                    "source_row_index": 7,
                    "sim_col": 13.5,
                    "sim_row": 15.5,
                    "qr": 1.2345678901,
                    "qz": -0.4567890123,
                }
            ]
        ),
        build_grouped_candidates=lambda entries: {
            ("q_group", "primary", 1, 0): [dict(entry) for entry in entries or ()]
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        entry_display_coords=lambda entry: (
            (
                float(entry["x"]),
                float(entry["y"]),
            )
            if isinstance(entry, dict)
            else None
        ),
        auto_match_background_context=lambda image, cfg: (
            {**dict(cfg), "search_radius_px": 22.0},
            {"image_shape": np.asarray(image).shape},
        ),
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=False)
    measured_display, initial_pairs_display = callbacks.build_initial_pairs_display(
        1,
        prefer_cache=False,
    )

    assert callbacks.current_match_config()["search_radius_px"] == 18.0
    assert cache_data["match_config"]["search_radius_px"] == 22.0
    assert cache_state["signature"] == cache_data["signature"]
    assert cache_state["data"] == cache_data
    assert simulated_param_sets == [{"a": 2.0}, {"gamma": 1.25}]
    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (13.5, 15.5),
            "qr": 1.2345678901,
            "qz": -0.4567890123,
        }
    ]


def test_make_runtime_geometry_manual_cache_callbacks_fails_closed_when_mask_filter_raises() -> (
    None
):
    cache_state = {"signature": None, "data": {}}

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        disabled_qr_sets=lambda: [],
        disabled_qz_sections=lambda: [],
        filter_active_rows=lambda _rows: (_ for _ in ()).throw(RuntimeError("mask boom")),
        stored_max_positions_local=lambda: [],
        stored_peak_table_lattice=lambda: [],
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda _idx: [],
        simulated_peaks_for_params=lambda _params, prefer_cache=True: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 4,
                "source_row_index": 7,
                "sim_col": 13.5,
                "sim_row": 15.5,
            }
        ],
        build_grouped_candidates=lambda entries: (
            {("q_group", "primary", 1, 2): [dict(entry) for entry in entries or ()]}
            if list(entries or ())
            else {}
        ),
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        entry_display_coords=lambda entry: (
            (
                float(entry["x"]),
                float(entry["y"]),
            )
            if isinstance(entry, dict) and "x" in entry and "y" in entry
            else None
        ),
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=False)

    assert len(cache_data["simulated_peaks"]) == 1
    assert cache_data["active_simulated_peaks"] == []
    assert cache_data["grouped_candidates"] == {}
    assert cache_data["simulated_lookup"] == {}


def test_build_geometry_manual_pick_cache_mask_refresh_fails_closed_when_mask_filter_raises() -> (
    None
):
    placed_signature = ("placed", 1)
    existing_cache = {
        "placed_signature": placed_signature,
        "simulated_peaks": [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 4,
                "source_row_index": 7,
                "sim_col": 13.5,
                "sim_row": 15.5,
            }
        ],
    }

    cache_data, _next_sig, _next_state = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((4, 4), dtype=float),
        existing_cache_signature=("stale",),
        existing_cache_data=existing_cache,
        placed_cache_signature_fn=lambda **_kwargs: placed_signature,
        cache_signature_fn=lambda **_kwargs: ("fresh",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        peak_records=[],
        build_grouped_candidates=lambda entries: (
            {("q_group", "primary", 1, 2): [dict(entry) for entry in entries or ()]}
            if list(entries or ())
            else {}
        ),
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        filter_active_rows=lambda _rows: (_ for _ in ()).throw(RuntimeError("mask boom")),
        project_peaks_to_current_view=None,
        current_match_config=lambda: {"search_radius_px": 18.0},
        auto_match_background_context=lambda image, cfg: (
            {**dict(cfg), "search_radius_px": 18.0},
            {"image_shape": np.asarray(image).shape},
        ),
    )

    assert len(cache_data["simulated_peaks"]) == 1
    assert cache_data["active_simulated_peaks"] == []
    assert cache_data["grouped_candidates"] == {}
    assert cache_data["simulated_lookup"] == {}
    assert cache_data["cache_metadata"]["cache_source"] == (
        "existing_cache_data.simulated_peaks(mask_refresh)"
    )


def test_build_geometry_manual_initial_pairs_display_fails_closed_when_mask_filter_raises() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=False,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        source_rows_for_background=lambda *_args, **_kwargs: [
            {
                "source_table_index": 4,
                "source_row_index": 7,
                "sim_col": 13.5,
                "sim_row": 15.5,
            }
        ],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda entries: {
            _source_key(entry): dict(entry) for entry in entries or ()
        },
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in (entries or ()) if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        filter_active_rows=lambda _rows: (_ for _ in ()).throw(RuntimeError("mask boom")),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
        }
    ]


def test_geometry_manual_toggle_selection_at_uses_shared_live_preview_cache_when_other_sources_are_empty() -> (
    None
):
    cache_state = {"signature": None, "data": {}}
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    forwarded_prefer_cache: list[bool] = []
    group_key = ("q_group", "primary", 1, 0)

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    def _group_candidates(entries):
        grouped = {}
        for entry in entries or ():
            if not isinstance(entry, dict):
                continue
            key = entry.get("q_group_key")
            if not isinstance(key, tuple):
                continue
            grouped.setdefault(key, []).append(dict(entry))
        return grouped

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((8, 8), dtype=float),
        use_caked_space=lambda: False,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache))
            or (
                [
                    {
                        "label": "1,0,0",
                        "hkl": (1, 0, 0),
                        "q_group_key": group_key,
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "weight": 1.0,
                    }
                ]
                if prefer_cache
                else []
            )
        ),
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
            if isinstance(entry, dict)
        },
        entry_display_coords=lambda _entry: None,
        peak_records=lambda: [],
    )

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **kwargs: callbacks.get_pick_cache(**kwargs),
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert forwarded_prefer_cache == [True]
    assert next_session["group_key"] == group_key
    assert set_sessions[-1]["group_key"] == group_key
    assert cache_state["data"]["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=True)"
    )
    assert status_messages
    assert "No simulated Qr/Qz groups are available to pick" not in status_messages[-1]
    assert "Selected selected group" in status_messages[-1]


def test_make_runtime_geometry_manual_projection_callbacks_project_caked_view(
    monkeypatch,
) -> None:
    caked_image = np.zeros((6, 6), dtype=float)
    native_background = np.ones((6, 6), dtype=float)
    radial_axis = np.linspace(10.0, 15.0, 6)
    azimuth_axis = np.linspace(-2.0, 3.0, 6)
    bundle = object()
    ai = _ai_with_live_bundle(bundle)

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            {
                (2.0, 3.0): (12.0, 1.0),
                (4.0, 1.0): (14.0, -1.0),
                (3.0, 4.0): (13.0, 2.0),
            }.get((float(col), float(row)), (None, None))
            if live_bundle is bundle
            else (None, None)
        ),
    )
    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda *_args, **_kwargs: (3.0, 4.0),
    )

    def _cached_live_preview_peaks() -> list[dict[str, object]]:
        return [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "native_col": 3.0,
                "native_row": 4.0,
            }
        ]

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: caked_image,
        last_caked_radial_values=lambda: radial_axis,
        last_caked_azimuth_values=lambda: azimuth_axis,
        current_background_display=lambda: np.full((6, 6), 9.0, dtype=float),
        current_background_native=lambda: native_background,
        ai=lambda: ai,
        center=lambda: (0.0, 0.0),
        detector_distance=lambda: 1.0,
        pixel_size=lambda: 1.0,
        wrap_phi_range=lambda value: value,
        current_geometry_fit_params=lambda: {"gamma": 1.5},
        build_live_preview_simulated_peaks_from_cache=_cached_live_preview_peaks,
        miller=lambda: np.array([[1.0, 0.0, 0.0]], dtype=float),
        intensities=lambda: np.array([2.0], dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    assert callbacks.pick_uses_caked_space() is True
    assert callbacks.current_background_image() is caked_image
    assert callbacks.entry_display_coords({"x": 2.0, "y": 3.0}) == (12.0, 1.0)
    assert callbacks.entry_display_coords(
        {
            "x": 2.0,
            "y": 3.0,
            "detector_x": 4.0,
            "detector_y": 1.0,
        }
    ) == (14.0, -1.0)
    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (3.0, 4.0)

    projected = callbacks.simulated_peaks_for_params()

    assert projected[0]["caked_x"] == 13.0
    assert projected[0]["caked_y"] == 2.0
    assert projected[0]["sim_col"] == 3.0
    assert projected[0]["sim_row"] == 4.0
    assert projected[0]["display_col"] == 13.0
    assert projected[0]["display_row"] == 2.0
    assert projected[0]["sim_col_local"] == 3.0
    assert projected[0]["sim_row_local"] == 4.0

    grouped = callbacks.pick_candidates(projected)
    assert list(grouped) == [("q_group", "primary", 1, 0)]
    assert grouped[("q_group", "primary", 1, 0)][0]["display_col"] == 13.0

    lookup = callbacks.simulated_lookup(projected)
    assert lookup[_source_key(projected[0])]["display_row"] == 2.0


def test_make_runtime_geometry_manual_projection_callbacks_reprojects_cached_caked_rows_from_raw_coords(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(8, 8))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (13.0, 2.0)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (91.0, 82.0)
            if live_bundle is bundle and (float(col), float(row)) == (99.0, 88.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "native_col": 3.0,
                "native_row": 4.0,
                "sim_col": 99.0,
                "sim_row": 88.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "caked_x": 91.0,
                "caked_y": 82.0,
                "raw_caked_x": 90.0,
                "raw_caked_y": 81.0,
            }
        ]
    )

    assert projected == [
        {
            "label": "1,0,0",
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col": 3.0,
            "sim_row": 4.0,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "native_col": 3.0,
            "native_row": 4.0,
            "sim_native_x": 3.0,
            "sim_native_y": 4.0,
            "coordinate_frame": "simulation_native",
            "caked_x": 13.0,
            "caked_y": 2.0,
            "raw_caked_x": 13.0,
            "raw_caked_y": 2.0,
            "two_theta_deg": 13.0,
            "phi_deg": 2.0,
            "display_col": 13.0,
            "display_row": 2.0,
            "display_frame": "caked_display",
            "sim_col_global": 13.0,
            "sim_row_global": 2.0,
            "sim_col_local": 3.0,
            "sim_row_local": 6.0,
        }
    ]


def test_project_peaks_to_current_view_does_not_call_analytic_forward_projection(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(8, 8))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (17.0, -9.0)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-9.0, -2.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "native_col": 3.0,
                "native_row": 4.0,
                "sim_col": 90.0,
                "sim_row": 80.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "caked_x": 66.0,
                "caked_y": 55.0,
                "raw_caked_x": 65.0,
                "raw_caked_y": 54.0,
            }
        ]
    )

    assert projected == [
        {
            "label": "1,0,0",
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col": 3.0,
            "sim_row": 4.0,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "native_col": 3.0,
            "native_row": 4.0,
            "sim_native_x": 3.0,
            "sim_native_y": 4.0,
            "coordinate_frame": "simulation_native",
            "caked_x": 17.0,
            "caked_y": -9.0,
            "raw_caked_x": 17.0,
            "raw_caked_y": -9.0,
            "two_theta_deg": 17.0,
            "phi_deg": -9.0,
            "display_col": 17.0,
            "display_row": -9.0,
            "display_frame": "caked_display",
            "sim_col_global": 17.0,
            "sim_row_global": -9.0,
            "sim_col_local": 7.0,
            "sim_row_local": 0.0,
        }
    ]


def test_project_peaks_to_current_view_preserves_display_only_rows_as_frozen_caked_overlays() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: object(),
        image_size=lambda: 8,
        display_to_native_sim_coords=_fail_projection_legacy_path(
            "display coordinates should not be converted into detector coordinates"
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "display_col": 4.0,
                "display_row": 5.0,
            }
        ]
    )

    assert projected == []


def test_project_peaks_to_current_view_preserves_display_only_rows_as_frozen_detector_overlays() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: object(),
        image_size=lambda: 8,
        display_to_native_sim_coords=_fail_projection_legacy_path(
            "display coordinates should not be converted into detector coordinates"
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "display_col": 4.0,
                "display_row": 5.0,
            }
        ]
    )

    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["display_col"] == 4.0
    assert projected_entry["display_row"] == 5.0
    assert "caked_x" not in projected_entry
    assert "caked_y" not in projected_entry
    assert "sim_col" not in projected_entry
    assert "sim_row" not in projected_entry
    assert "sim_col_raw" not in projected_entry
    assert "sim_row_raw" not in projected_entry
    assert "native_col" not in projected_entry
    assert "native_row" not in projected_entry


def test_project_peaks_to_current_view_strips_stale_detector_fields_from_caked_only_rows() -> None:
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(10.0, 17.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: None,
        image_size=lambda: 8,
        display_to_native_sim_coords=_fail_projection_legacy_path(
            "stale caked legacy sim coords should not be inverted"
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
        scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
            "analytic inverse fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "sim_col": 5.5,
                "sim_row": 12.25,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
            }
        ]
    )

    assert projected == []


def test_geometry_manual_entry_detector_display_point_rejects_caked_trust_only_sim_coords() -> None:
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "sim_col": 5.5,
                "sim_row": 12.25,
                "display_col": 5.5,
                "display_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
            }
        )
        is None
    )


def test_geometry_manual_entry_detector_display_point_uses_explicit_detector_coords_for_caked_entries() -> (
    None
):
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "sim_col": 5.5,
                "sim_row": 12.25,
                "sim_col_raw": 100.0,
                "sim_row_raw": 200.0,
                "caked_x": 5.5,
                "caked_y": 12.25,
            }
        )
    ) == (100.0, 200.0)
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "x": 101.0,
                "y": 201.0,
                "sim_col": 5.5,
                "sim_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
            }
        )
    ) == (101.0, 201.0)
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "simulated_x": 102.0,
                "simulated_y": 202.0,
                "sim_col": 5.5,
                "sim_row": 12.25,
                "caked_x": 5.5,
                "caked_y": 12.25,
            }
        )
    ) == (102.0, 202.0)


def test_geometry_manual_entry_detector_display_point_rejects_caked_alias_detector_fields() -> None:
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "x": 30.25,
                "y": -57.5,
                "caked_x": 30.25,
                "caked_y": -57.5,
            }
        )
        is None
    )
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "simulated_x": 30.25,
                "simulated_y": -57.5,
                "caked_x": 30.25,
                "caked_y": -57.5,
            }
        )
        is None
    )


def test_geometry_manual_entry_detector_display_point_accepts_detector_aliases_when_caked_fields_are_stale() -> (
    None
):
    assert mg._geometry_manual_entry_detector_display_point(
        {
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "x": 101.0,
            "y": 201.0,
            "caked_x": 101.0,
            "caked_y": 201.0,
            "stale_caked_fields": True,
        }
    ) == (101.0, 201.0)
    assert mg._geometry_manual_entry_detector_display_point(
        {
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "simulated_x": 102.0,
            "simulated_y": 202.0,
            "caked_x": 102.0,
            "caked_y": 202.0,
            "stale_caked_fields": True,
        }
    ) == (102.0, 202.0)


def test_geometry_manual_entry_detector_display_point_uses_branch_aware_detector_sim_with_caked_fields() -> (
    None
):
    assert mg._geometry_manual_entry_detector_display_point(
        {
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 190.0,
            "sim_row": 96.0,
            "caked_x": 30.25,
            "caked_y": -57.5,
        }
    ) == (190.0, 96.0)


def test_geometry_manual_entry_detector_display_point_rejects_trusted_caked_live_source_sim_coords_without_explicit_detector_fields() -> (
    None
):
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "sim_col": 190.0,
                "sim_row": 96.0,
                "caked_x": 30.25,
                "caked_y": -57.5,
            }
        )
        is None
    )


def test_geometry_manual_entry_detector_display_point_rejects_source_row_identity_when_branch_only_comes_from_caked_fields() -> (
    None
):
    assert (
        mg._geometry_manual_entry_detector_display_point(
            {
                "source_table_index": 9,
                "source_row_index": 0,
                "sim_col": 190.0,
                "sim_row": 96.0,
                "caked_x": 30.25,
                "caked_y": -57.5,
            }
        )
        is None
    )


def test_project_peaks_to_current_view_rejects_trusted_caked_legacy_sim_without_explicit_detector_fields() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: None,
        image_size=lambda: 8,
        display_to_native_sim_coords=_fail_projection_legacy_path(
            "trusted caked legacy sim coords should not be inverted"
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
        scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
            "analytic inverse fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "-1,0,5",
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "sim_col": 190.0,
                "sim_row": 96.0,
                "caked_x": 30.25,
                "caked_y": -57.5,
                "hkl": (-1, 0, 5),
            }
        ]
    )

    assert projected == []


def test_project_peaks_to_current_view_recomputes_detector_display_from_native_background_frame() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((5, 5), dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((5, 5), dtype=float),
        current_background_native=lambda: np.ones((5, 5), dtype=float),
        image_size=lambda: 5,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda col, row, _shape: (_ for _ in ()).throw(
            AssertionError("native coordinates should be used directly")
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "native_col": 1.0,
                "native_row": 2.0,
                "sim_col": 99.0,
                "sim_row": 88.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "display_col": 400.0,
                "display_row": -500.0,
                "caked_x": 91.0,
                "caked_y": 82.0,
            }
        ]
    )

    expected_display = mg._default_rotate_point(1.0, 2.0, (5, 5), 3)

    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["native_col"] == 1.0
    assert projected_entry["native_row"] == 2.0
    assert projected_entry["sim_native_x"] == 1.0
    assert projected_entry["sim_native_y"] == 2.0
    assert projected_entry["sim_col_raw"] == float(expected_display[0])
    assert projected_entry["sim_row_raw"] == float(expected_display[1])
    assert projected_entry["sim_col"] == float(expected_display[0])
    assert projected_entry["sim_row"] == float(expected_display[1])
    assert projected_entry["display_col"] == float(expected_display[0])
    assert projected_entry["display_row"] == float(expected_display[1])
    assert "caked_x" not in projected_entry
    assert "caked_y" not in projected_entry


def test_project_peaks_to_current_view_uses_refined_detector_display_background_frame_when_native_is_stale() -> (
    None
):
    refined_native = (2.0, 1.0)
    refined_display = mg._default_rotate_point(
        float(refined_native[0]),
        float(refined_native[1]),
        (5, 5),
        3,
    )
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((5, 5), dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((5, 5), dtype=float),
        current_background_native=lambda: np.ones((5, 5), dtype=float),
        image_size=lambda: 5,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("refined detector display should use background inverse")
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )
    refined_entry = mg.geometry_manual_apply_refined_simulated_override(
        {
            "refined_sim_x": float(refined_display[0]),
            "refined_sim_y": float(refined_display[1]),
        },
        {
            "label": "1,0,0",
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "native_col": 1.0,
            "native_row": 2.0,
            "sim_native_x": 1.0,
            "sim_native_y": 2.0,
            "sim_col": 99.0,
            "sim_row": 88.0,
            "sim_col_raw": 77.0,
            "sim_row_raw": 66.0,
            "display_col": 55.0,
            "display_row": 44.0,
        },
        prefer_caked_display=False,
    )

    projected = callbacks.project_peaks_to_current_view([refined_entry])

    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["refined_sim_x"] == float(refined_display[0])
    assert projected_entry["refined_sim_y"] == float(refined_display[1])
    assert projected_entry["native_col"] == float(refined_native[0])
    assert projected_entry["native_row"] == float(refined_native[1])
    assert projected_entry["sim_native_x"] == float(refined_native[0])
    assert projected_entry["sim_native_y"] == float(refined_native[1])
    assert projected_entry["sim_col_raw"] == float(refined_display[0])
    assert projected_entry["sim_row_raw"] == float(refined_display[1])
    assert projected_entry["sim_col"] == float(refined_display[0])
    assert projected_entry["sim_row"] == float(refined_display[1])
    assert projected_entry["display_col"] == float(refined_display[0])
    assert projected_entry["display_row"] == float(refined_display[1])


def test_project_peaks_to_current_view_keeps_refined_detector_display_on_non_square_background() -> (
    None
):
    background_shape = (3, 5)
    refined_native = (2.0, 1.0)
    refined_display = mg._default_rotate_point(
        float(refined_native[0]),
        float(refined_native[1]),
        background_shape,
        3,
    )
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros(background_shape, dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros(background_shape, dtype=float),
        current_background_native=lambda: np.ones(background_shape, dtype=float),
        image_size=lambda: 9,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-square refined detector display should use background inverse")
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )
    refined_entry = mg.geometry_manual_apply_refined_simulated_override(
        {
            "refined_sim_x": float(refined_display[0]),
            "refined_sim_y": float(refined_display[1]),
        },
        {
            "label": "1,0,0",
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "native_col": 0.0,
            "native_row": 0.0,
            "sim_native_x": 0.0,
            "sim_native_y": 0.0,
            "sim_col": 99.0,
            "sim_row": 88.0,
            "sim_col_raw": 77.0,
            "sim_row_raw": 66.0,
            "display_col": 55.0,
            "display_row": 44.0,
        },
        prefer_caked_display=False,
    )

    projected = callbacks.project_peaks_to_current_view([refined_entry])

    assert len(projected) == 1
    projected_entry = projected[0]
    expected_native = mg._default_rotate_point(
        float(refined_display[0]),
        float(refined_display[1]),
        background_shape,
        -3,
    )
    assert projected_entry["native_col"] == float(expected_native[0])
    assert projected_entry["native_row"] == float(expected_native[1])
    assert projected_entry["sim_col_raw"] == float(refined_display[0])
    assert projected_entry["sim_row_raw"] == float(refined_display[1])
    assert projected_entry["sim_col"] == float(refined_display[0])
    assert projected_entry["sim_row"] == float(refined_display[1])
    assert projected_entry["display_col"] == float(refined_display[0])
    assert projected_entry["display_row"] == float(refined_display[1])


def test_project_peaks_to_current_view_prefers_simulation_native_caked_mapping() -> None:
    background_adapter_calls: list[tuple[float, float]] = []
    simulation_adapter_calls: list[tuple[float, float]] = []

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        native_detector_coords_to_bundle_detector_coords=lambda col, row: (
            background_adapter_calls.append((float(col), float(row)))
            or (float(col) + 100.0, float(row) + 100.0)
        ),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            simulation_adapter_calls.append((float(col), float(row)))
            or (20.0 + float(col), -30.0 + float(row))
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,10",
                "q_group_key": ("q_group", "primary", 1, 10),
                "source_table_index": 0,
                "source_row_index": 1,
                "native_col": 3.0,
                "native_row": 4.0,
                "sim_col": 3.0,
                "sim_row": 4.0,
            }
        ]
    )

    assert len(projected) == 1
    assert simulation_adapter_calls == [(3.0, 4.0)]
    assert background_adapter_calls == []
    assert projected[0]["caked_x"] == 23.0
    assert projected[0]["caked_y"] == -26.0
    assert projected[0]["two_theta_deg"] == 23.0
    assert projected[0]["phi_deg"] == -26.0
    assert projected[0]["display_col"] == 23.0
    assert projected[0]["display_row"] == -26.0


def test_project_peaks_to_current_view_recomputes_caked_angles_from_native_detector_coords(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(8, 8))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (11.5, 13.5)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(13.0, 20.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (_ for _ in ()).throw(
            AssertionError("native coordinates should be used directly")
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "native_col": 3.0,
                "native_row": 4.0,
                "sim_col": 90.0,
                "sim_row": 80.0,
                "sim_col_raw": 30.0,
                "sim_row_raw": 40.0,
                "caked_x": 66.0,
                "caked_y": 55.0,
                "raw_caked_x": 65.0,
                "raw_caked_y": 54.0,
                "two_theta_deg": 64.0,
                "phi_deg": 53.0,
            }
        ]
    )

    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["native_col"] == 3.0
    assert projected_entry["native_row"] == 4.0
    assert projected_entry["sim_native_x"] == 3.0
    assert projected_entry["sim_native_y"] == 4.0
    assert projected_entry["caked_x"] == 11.5
    assert projected_entry["caked_y"] == 13.5
    assert projected_entry["raw_caked_x"] == 11.5
    assert projected_entry["raw_caked_y"] == 13.5
    assert projected_entry["display_col"] == 11.5
    assert projected_entry["display_row"] == 13.5
    assert projected_entry["sim_col_global"] == 11.5
    assert projected_entry["sim_row_global"] == 13.5
    assert projected_entry["sim_col_local"] == 1.5
    assert projected_entry["sim_row_local"] == 0.5


def test_make_runtime_geometry_manual_projection_callbacks_clears_stale_caked_fields_when_reprojection_fails(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(8, 8))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda *_args, **_kwargs: (None, None),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 99.0,
                "sim_row": 88.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "caked_x": 91.0,
                "caked_y": 82.0,
                "raw_caked_x": 90.0,
                "raw_caked_y": 81.0,
                "sim_col_global": 91.0,
                "sim_row_global": 82.0,
                "sim_col_local": 7.0,
                "sim_row_local": 1.0,
            }
        ]
    )

    assert projected == []


def test_make_runtime_geometry_manual_projection_callbacks_pick_candidates_fall_back_when_filter_removes_every_group() -> (
    None
):
    filter_calls: list[int] = []
    collapse_calls: list[int] = []

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.zeros((6, 6), dtype=float),
        image_size=lambda: 6,
        filter_simulated_peaks=lambda entries: (
            filter_calls.append(len(entries or ())) or ([], 1, 0)
        ),
        collapse_simulated_peaks=lambda entries, *, merge_radius_px: (
            collapse_calls.append(len(entries or ()))
            or (list(entries or ()), int(round(float(merge_radius_px))))
        ),
    )

    grouped = callbacks.pick_candidates(
        [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
            }
        ]
    )

    assert filter_calls == [1]
    assert collapse_calls == [0]
    assert grouped == {}


def test_make_runtime_geometry_manual_projection_callbacks_back_projects_caked_through_inverse_lut(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle()
    inverse_calls: list[
        tuple[tuple[int, int], tuple[float, ...], tuple[float, ...], float, float, object]
    ] = []

    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda _ai, detector_shape, radial_deg, azimuth_deg, two_theta, phi, **kwargs: (
            inverse_calls.append(
                (
                    tuple(int(v) for v in tuple(detector_shape)[:2]),
                    tuple(float(v) for v in np.asarray(radial_deg, dtype=float)),
                    tuple(float(v) for v in np.asarray(azimuth_deg, dtype=float)),
                    float(two_theta),
                    float(phi),
                    kwargs.get("transform_bundle"),
                )
            )
            or (102.0, 213.0)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.ones((6, 6), dtype=float),
        center=lambda: (20.0, 30.0),
        detector_distance=lambda: 100.0,
        pixel_size=lambda: 0.25,
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("forward detector->angle conversion should not be used")
        ),
        scattering_angles_to_detector_pixel=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
    )

    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (102.0, 213.0)
    assert inverse_calls == [
        (
            (6, 6),
            (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
            (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
            13.0,
            2.0,
            bundle,
        )
    ]


def test_make_runtime_geometry_manual_projection_callbacks_back_projects_caked_through_backend_inverse(
    monkeypatch,
) -> None:
    inverse_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda *_args, **_kwargs: (0.0, 1.0),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((3, 4), dtype=float),
        current_background_native=lambda: np.ones((3, 4), dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("forward detector->angle conversion should not be used")
        ),
        backend_detector_coords_to_native_detector_coords=lambda col, row: (
            inverse_calls.append((float(col), float(row))) or (1.5, 2.5)
        ),
        scattering_angles_to_detector_pixel=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
    )

    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (1.5, 2.5)
    assert inverse_calls == [(0.0, 1.0)]


def test_make_runtime_geometry_manual_projection_callbacks_projects_detector_points_in_native_coords(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(4, 4))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (13.0, 2.0)
            if live_bundle is bundle and (float(col), float(row)) == (2.0, 2.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((4, 4), dtype=float),
        current_background_native=lambda: np.ones((4, 4), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
        backend_detector_coords_to_native_detector_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("backend inverse should not be used for detector->caked projection")
        ),
    )

    assert callbacks.native_detector_coords_to_caked_display_coords(2.0, 2.0) == (
        13.0,
        2.0,
    )
    assert callbacks.entry_display_coords(
        {
            "x": 99.0,
            "y": 99.0,
            "detector_x": 2.0,
            "detector_y": 2.0,
        }
    ) == (13.0, 2.0)


def test_make_runtime_geometry_manual_projection_callbacks_keep_detector_points_continuous(
    monkeypatch,
) -> None:
    bundle = _dummy_transform_bundle(detector_shape=(4, 4))

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (13.2, 1.8)
            if live_bundle is bundle and (float(col), float(row)) == (2.0, 2.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.array([10.0, 13.0, 15.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([-2.0, 2.0, 5.0], dtype=float),
        current_background_display=lambda: np.zeros((4, 4), dtype=float),
        current_background_native=lambda: np.ones((4, 4), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
        backend_detector_coords_to_native_detector_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("backend inverse should not be used for detector->caked projection")
        ),
    )

    assert callbacks.native_detector_coords_to_caked_display_coords(2.0, 2.0) == (
        13.2,
        1.8,
    )


def test_make_runtime_geometry_manual_projection_callbacks_prefer_cache_uses_live_preview_peaks() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.zeros((6, 6), dtype=float),
        current_geometry_fit_params=lambda: {"gamma": 1.5},
        build_live_preview_simulated_peaks_from_cache=lambda: [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "native_col": 3.0,
                "native_row": 4.0,
            }
        ],
        ensure_peak_overlay_data=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("live preview cache hit should not bootstrap overlay data")
        ),
        miller=lambda: np.array([[1.0, 0.0, 0.0]], dtype=float),
        intensities=lambda: np.array([2.0], dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)
    diagnostics = callbacks.last_simulation_diagnostics()

    assert len(projected) == 1
    assert {
        key: projected[0][key]
        for key in (
            "label",
            "q_group_key",
            "source_table_index",
            "source_row_index",
            "sim_col",
            "sim_row",
            "native_col",
            "native_row",
            "sim_native_x",
            "sim_native_y",
            "sim_col_raw",
            "sim_row_raw",
            "display_col",
            "display_row",
            "display_frame",
        )
    } == {
        "label": "1,0,0",
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 3.0,
        "sim_row": 4.0,
        "native_col": 3.0,
        "native_row": 4.0,
        "sim_native_x": 3.0,
        "sim_native_y": 4.0,
        "sim_col_raw": 3.0,
        "sim_row_raw": 4.0,
        "display_col": 3.0,
        "display_row": 4.0,
        "display_frame": "detector_display",
    }
    assert projected[0]["selection_reason"] == "mosaic_top_per_branch"
    assert diagnostics["source"] == "cache"
    assert diagnostics["status"] == "cache_hit"


def test_make_runtime_geometry_manual_projection_callbacks_prefer_cache_bootstraps_peak_overlay_cache() -> (
    None
):
    ensured: list[bool] = []
    cached_peaks: list[dict[str, object]] = []

    def _ensure_peak_overlay_data(*, force: bool = False) -> bool:
        ensured.append(bool(force))
        cached_peaks[:] = [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "native_col": 3.0,
                "native_row": 4.0,
            }
        ]
        return True

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.zeros((6, 6), dtype=float),
        current_geometry_fit_params=lambda: {"gamma": 1.5},
        build_live_preview_simulated_peaks_from_cache=lambda: list(cached_peaks),
        ensure_peak_overlay_data=_ensure_peak_overlay_data,
        miller=lambda: np.array([[1.0, 0.0, 0.0]], dtype=float),
        intensities=lambda: np.array([2.0], dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)
    diagnostics = callbacks.last_simulation_diagnostics()

    assert ensured == [False]
    assert len(projected) == 1
    assert projected[0]["label"] == "1,0,0"
    assert projected[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert projected[0]["source_table_index"] == 1
    assert projected[0]["source_row_index"] == 2
    assert projected[0]["sim_col"] == 3.0
    assert projected[0]["sim_row"] == 4.0
    assert projected[0]["native_col"] == 3.0
    assert projected[0]["native_row"] == 4.0
    assert projected[0]["sim_native_x"] == 3.0
    assert projected[0]["sim_native_y"] == 4.0
    assert projected[0]["sim_col_raw"] == 3.0
    assert projected[0]["sim_row_raw"] == 4.0
    assert projected[0]["display_col"] == 3.0
    assert projected[0]["display_row"] == 4.0
    assert projected[0]["selection_reason"] == "mosaic_top_per_branch"
    assert diagnostics["source"] == "cache"
    assert diagnostics["status"] == "cache_hit"


def test_render_current_geometry_manual_pairs_updates_active_session_status() -> None:
    calls: list[tuple[str, object]] = []
    status_messages: list[str] = []

    rendered = mg.render_current_geometry_manual_pairs(
        background_visible=True,
        current_background_index=2,
        current_background_image=np.zeros((5, 5), dtype=float),
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [{"label": "1,0,0"}, {"label": "-1,0,0"}, {"label": "0,0,2"}],
            "pending_entries": [{"label": "1,0,0", "x": 9.0, "y": 10.0}],
            "target_count": 3,
            "q_label": "test group",
            "background_index": 2,
        },
        build_initial_pairs_display=lambda *_args, **_kwargs: (
            [{"overlay_match_index": 0}],
            [{"overlay_match_index": 0, "bg_display": (1.0, 2.0)}],
        ),
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        draw_initial_geometry_pairs_overlay=lambda entries, **kwargs: calls.append(
            ("draw", (list(entries), kwargs.get("max_display_markers")))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_background_file_status_text_fn=lambda: calls.append(("background", None)),
        pair_group_count=lambda _idx: 4,
        set_status_text=status_messages.append,
        update_status=True,
    )

    assert rendered is True
    assert ("button", None) in calls
    assert ("background", None) in calls
    assert calls[0][0] == "draw"
    assert "Click background peak 2 of 3 for test group" in status_messages[-1]


def test_geometry_manual_pick_button_label_includes_progress() -> None:
    label = mg.geometry_manual_pick_button_label(
        armed=True,
        current_background_index=0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [{"label": "1,0,0"}, {"label": "-1,0,0"}],
            "pending_entries": [{"label": "1,0,0", "x": 9.0, "y": 10.0}],
            "target_count": 2,
            "background_index": 0,
        },
        pairs_for_index=lambda _idx: [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        pair_group_count=lambda _idx: 1,
    )

    assert label == "Pick Qr Sets on Image (Armed) [1 groups/2 pts] <placing 1/2>"


def test_geometry_manual_toggle_selection_at_starts_session() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []
    group_key = ("q_group", "primary", 1, 0)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,0",
                        "hkl": (1, 0, 0),
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    }
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["target_count"] == 1
    assert set_sessions[-1]["group_key"] == group_key
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert "Selected selected group" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_starts_two_branch_session_for_qr_qz_group() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []
    group_key = ("q_group", "primary", 1, 2)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,2",
                        "hkl": (1, 0, 2),
                        "branch_id": "+x",
                        "source_branch_index": 0,
                        "source_reflection_index": 20,
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "weight": 1.0,
                        "mosaic_weight": 0.2,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    },
                    {
                        "label": "right",
                        "hkl": (-1, 0, 2),
                        "branch_id": "-x",
                        "source_branch_index": 1,
                        "source_reflection_index": 21,
                        "sim_col": 30.0,
                        "sim_row": 40.0,
                        "weight": 1.0,
                        "mosaic_weight": 0.9,
                        "source_table_index": 1,
                        "source_row_index": 3,
                    },
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["target_count"] == 2
    assert set_sessions[-1]["target_count"] == 2
    assert len(next_session["group_entries"]) == 2
    by_branch = {entry["branch_id"]: entry for entry in next_session["group_entries"]}
    assert by_branch["+x"]["label"] == "1,0,2"
    assert by_branch["+x"]["source_branch_index"] == 0
    assert by_branch["+x"]["source_reflection_index"] == 20
    assert by_branch["+x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["-x"]["label"] == "right"
    assert by_branch["-x"]["source_branch_index"] == 1
    assert by_branch["-x"]["source_reflection_index"] == 21
    assert by_branch["-x"]["selection_reason"] == "mosaic_top_per_branch"
    displayed = mg.geometry_manual_session_initial_pairs_display(
        next_session,
        current_background_index=0,
        entry_display_coords=lambda entry: (
            float(entry.get("sim_col")),
            float(entry.get("sim_row")),
        ),
    )
    assert len([entry for entry in displayed if entry.get("q_group_key") == group_key]) == 2
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert "Tagged seed [1,0,2]" in status_messages[-1]
    assert "Click background peak 1 of 2" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_tags_clicked_seed_within_group() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    group_key = ("q_group", "primary", 1, 2)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        29.5,
        39.5,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,2",
                        "hkl": (1, 0, 2),
                        "branch_id": "+x",
                        "source_branch_index": 0,
                        "source_reflection_index": 20,
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "weight": 1.0,
                        "mosaic_weight": 0.2,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    },
                    {
                        "label": "right",
                        "hkl": (-1, 0, 2),
                        "branch_id": "-x",
                        "source_branch_index": 1,
                        "source_reflection_index": 21,
                        "sim_col": 30.0,
                        "sim_row": 40.0,
                        "weight": 1.0,
                        "mosaic_weight": 0.9,
                        "source_table_index": 1,
                        "source_row_index": 3,
                    },
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["target_count"] == 2
    assert len(next_session["group_entries"]) == 2
    assert next_session["tagged_candidate_key"] == ("source_branch", 1, 1)
    by_branch = {entry["branch_id"]: entry for entry in next_session["group_entries"]}
    assert by_branch["-x"]["source_row_index"] == 3
    assert by_branch["-x"]["source_branch_index"] == 1
    assert by_branch["-x"]["source_reflection_index"] == 21
    assert by_branch["-x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["+x"]["source_branch_index"] == 0
    assert next_session["tagged_candidate"]["label"] == "right"
    assert "Tagged seed [right]" in status_messages[-1]
    assert set_sessions[-1]["tagged_candidate_key"] == ("source_branch", 1, 1)


def test_geometry_manual_toggle_selection_at_keeps_selected_candidate_under_permutation() -> None:
    group_key = ("q_group", "primary", 1, 5)
    left = {
        "label": "left",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 8,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 8,
        "source_row_index": 0,
        "sim_col": 10.0,
        "sim_row": 20.0,
        "weight": 1.0,
        "mosaic_weight": 0.2,
    }
    right = {
        "label": "right",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 9,
        "source_row_index": 0,
        "sim_col": 30.0,
        "sim_row": 40.0,
        "weight": 1.0,
        "mosaic_weight": 0.9,
    }

    def _run(entries):
        handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
            29.5,
            39.5,
            pick_session={},
            current_background_index=0,
            display_background=np.zeros((8, 8), dtype=float),
            get_cache_data=lambda **_kwargs: {
                "signature": ("cache",),
                "grouped_candidates": {group_key: [dict(entry) for entry in entries]},
            },
            pairs_for_index=lambda _idx: [],
            set_pairs_for_index_fn=lambda _idx, rows: list(rows or []),
            set_pick_session_fn=lambda _session: None,
            restore_view_fn=lambda **_kwargs: None,
            clear_preview_artists_fn=lambda **_kwargs: None,
            render_current_pairs_fn=lambda **_kwargs: None,
            update_button_label_fn=lambda: None,
            set_status_text=lambda _text: None,
            listed_q_group_entries=lambda: [{"key": group_key}],
            format_q_group_line=lambda _entry: "selected group",
            use_caked_space=False,
            pick_search_window_px=50.0,
        )
        assert handled is True
        assert suppress_drag is True
        return next_session

    forward = _run([left, right])
    reversed_order = _run([right, left])

    for session in (forward, reversed_order):
        assert session["tagged_candidate_key"] == ("source_branch", 9, 1)
        assert session["tagged_candidate"]["source_reflection_index"] == 9
        assert session["tagged_candidate"]["source_reflection_namespace"] == ("full_reflection")
        assert session["tagged_candidate"]["source_reflection_is_full"] is True
        assert session["tagged_candidate"]["source_branch_index"] == 1
        assert session["tagged_candidate"]["selection_reason"] == "mosaic_top_per_branch"
        assert session["target_count"] == 2
        assert len(session["group_entries"]) == 2
        by_branch = {entry["source_branch_index"]: entry for entry in session["group_entries"]}
        assert by_branch[1]["source_reflection_index"] == 9
        assert by_branch[1]["source_branch_index"] == 1
        assert by_branch[0]["source_reflection_index"] == 8


def test_geometry_manual_toggle_selection_at_prefers_shared_peak_finder_group_when_window_choose_misses() -> (
    None
):
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    shared_search_limits: list[float] = []
    group_key = ("q_group", "primary", 1, 2)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "left",
                        "hkl": (1, 0, 2),
                        "q_group_key": group_key,
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "mosaic_weight": 0.2,
                        "sim_col": 200.0,
                        "sim_row": 210.0,
                    },
                    {
                        "label": "right",
                        "hkl": (-1, 0, 2),
                        "q_group_key": group_key,
                        "source_table_index": 1,
                        "source_row_index": 3,
                        "mosaic_weight": 0.9,
                        "sim_col": 220.0,
                        "sim_row": 230.0,
                    },
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, rows: list(rows or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        find_peak_record_for_click_fn=lambda col, row, max_axis_distance: (
            shared_search_limits.append(float(max_axis_distance))
            or (
                7,
                {
                    "label": "right",
                    "hkl": (-1, 0, 2),
                    "q_group_key": group_key,
                    "source_table_index": 1,
                    "source_row_index": 3,
                    "sim_col": 10.5,
                    "sim_row": 20.5,
                },
                0.75,
                True,
            )
        ),
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert shared_search_limits == [25.0]
    assert next_session["group_key"] == group_key
    assert next_session["tagged_candidate"]["label"] == "right"
    assert next_session["group_entries"][0]["source_row_index"] == 3
    assert set_sessions[-1]["tagged_candidate_key"] == ("source", 1, 3)
    assert "Tagged seed [right]." in status_messages[-1]
    assert "nearest Bragg seed 0.8px" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_shared_peak_finder_preserves_branch_from_phi() -> None:
    group_key = ("q_group", "primary", 1, 5)
    mirrored_entries = [
        {
            "label": "left",
            "hkl": (-1, 0, 5),
            "q_group_key": group_key,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_table_index": 9,
            "source_row_index": 0,
            "mosaic_weight": 0.2,
            "sim_col": 181.0,
            "sim_row": 95.0,
        },
        {
            "label": "right",
            "hkl": (-1, 0, 5),
            "q_group_key": group_key,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_table_index": 9,
            "source_row_index": 0,
            "mosaic_weight": 0.9,
            "sim_col": 190.0,
            "sim_row": 96.0,
        },
    ]

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        0.0,
        0.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {group_key: [dict(entry) for entry in mirrored_entries]},
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, rows: list(rows or []),
        set_pick_session_fn=lambda _session: None,
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        find_peak_record_for_click_fn=lambda _col, _row, _max_axis_distance: (
            1,
            {
                "label": "shared-right",
                "hkl": (-1, 0, 5),
                "q_group_key": group_key,
                "source_reflection_index": 203,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_table_index": 9,
                "source_row_index": 0,
                "phi": 15.0,
            },
            0.5,
            True,
        ),
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["tagged_candidate_key"] == ("source_branch", 203, 1)
    assert next_session["tagged_candidate"]["source_branch_index"] == 1
    assert next_session["tagged_candidate"]["source_peak_index"] == 1
    assert next_session["group_entries"][0]["source_branch_index"] == 1


def test_geometry_manual_toggle_selection_at_skips_shared_peak_finder_in_caked_view() -> None:
    status_messages: list[str] = []
    group_key = ("q_group", "primary", 1, 0)
    projected_grouped = {
        group_key: [
            {
                "label": "caked-near",
                "hkl": (1, 0, 0),
                "q_group_key": group_key,
                "source_table_index": 2,
                "source_row_index": 1,
                "native_col": 3.0,
                "native_row": 4.0,
                "sim_col_raw": 103.0,
                "sim_row_raw": 204.0,
                "display_col": 13.0,
                "display_row": 2.0,
                "caked_x": 13.0,
                "caked_y": 2.0,
                "_caked_qr_projection_cache": True,
            }
        ]
    }

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        13.0,
        2.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "caked-near",
                        "hkl": (1, 0, 0),
                        "q_group_key": group_key,
                        "source_table_index": 2,
                        "source_row_index": 1,
                        "sim_col": 500.0,
                        "sim_row": 600.0,
                        "caked_x": 40.0,
                        "caked_y": 50.0,
                    }
                ]
            },
            "caked_qr_projection_grouped_candidates": projected_grouped,
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, rows: list(rows or []),
        set_pick_session_fn=lambda _session: None,
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        find_peak_record_for_click_fn=lambda *_args: (_ for _ in ()).throw(
            AssertionError("caked Qr must not use shared peak finder")
        ),
        use_caked_space=True,
        pick_search_window_px=50.0,
        caked_search_tth_deg=2.0,
        caked_search_phi_deg=6.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["tagged_candidate"]["source_row_index"] == 1
    assert "nearest Bragg seed 0.0 deg" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_falls_back_when_shared_peak_has_no_group_key() -> None:
    group_key = ("q_group", "primary", 1, 0)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,0",
                        "hkl": (1, 0, 0),
                        "q_group_key": group_key,
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                    }
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, rows: list(rows or []),
        set_pick_session_fn=lambda _session: None,
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        find_peak_record_for_click_fn=lambda _col, _row, _max_axis_distance: (
            0,
            {"label": "missing-group-key"},
            0.1,
            True,
        ),
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["tagged_candidate_key"] == ("source", 1, 2)


def test_geometry_manual_place_selection_at_saves_completed_group() -> None:
    set_sessions: list[dict[str, object]] = []
    saved_entry_sets: list[list[dict[str, object]]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        4.8,
        5.9,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [
                {
                    "label": "1,0,0",
                    "hkl": (1, 0, 0),
                    "sim_col": 5.0,
                    "sim_row": 6.0,
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [{"label": "kept", "x": 1.0, "y": 2.0}],
            "q_label": "selected group",
            "background_index": 0,
            "zoom_active": True,
            "zoom_center": (5.0, 6.0),
            "saved_xlim": (0.0, 10.0),
            "saved_ylim": (0.0, 10.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (5.0, 6.0),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: calls.append(("undo", None)),
        use_caked_space=False,
    )

    assert handled is True
    assert next_session == {}
    assert set_sessions[-1] == {}
    assert ("undo", None) in calls
    assert ("clear", False) in calls
    assert ("restore", False) not in calls
    assert ("render", False) in calls
    assert saved_entry_sets
    assert saved_entry_sets[-1][0]["label"] == "kept"
    assert saved_entry_sets[-1][1]["label"] == "1,0,0"
    assert saved_entry_sets[-1][1]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert saved_entry_sets[-1][1]["placement_error_px"] > 0.0
    assert "Saved 1 manual background points for selected group" in status_messages[-1]


def test_geometry_manual_place_selection_at_applies_saved_pair_refinement_callback() -> None:
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        4.8,
        5.9,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [
                {
                    "label": "1,0,0",
                    "hkl": (1, 0, 0),
                    "sim_col": 5.0,
                    "sim_row": 6.0,
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (5.0, 6.0),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=False,
        refine_saved_pair_entry_fn=lambda entry, candidate=None: {
            **dict(entry),
            "refined_sim_caked_x": 11.0,
            "refined_sim_caked_y": 12.0,
            "refined_sim_x": 13.0,
            "refined_sim_y": 14.0,
            "source_row_index": candidate.get("source_row_index"),
        },
    )

    assert handled is True
    assert next_session == {}
    assert saved_entry_sets[-1][0]["refined_sim_caked_x"] == 11.0
    assert saved_entry_sets[-1][0]["refined_sim_caked_y"] == 12.0
    assert saved_entry_sets[-1][0]["refined_sim_x"] == 13.0
    assert saved_entry_sets[-1][0]["refined_sim_y"] == 14.0


def test_geometry_manual_place_selection_at_uses_tagged_candidate_first() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        11.8,
        14.2,
        pick_session={
            "group_key": ("q_group", "primary", 1, 2),
            "group_entries": [
                {
                    "label": "left",
                    "hkl": (1, 0, 2),
                    "sim_col": 35.0,
                    "sim_row": 30.0,
                    "source_table_index": 1,
                    "source_row_index": 2,
                },
                {
                    "label": "right",
                    "hkl": (-1, 0, 2),
                    "sim_col": 12.0,
                    "sim_row": 14.0,
                    "source_table_index": 1,
                    "source_row_index": 3,
                },
            ],
            "pending_entries": [],
            "target_count": 2,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate_key": ("source", 1, 2),
            "zoom_active": True,
            "zoom_center": (12.0, 14.0),
            "saved_xlim": (0.0, 40.0),
            "saved_ylim": (0.0, 40.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: None,
        use_caked_space=False,
    )

    assert handled is True
    assert next_session["pending_entries"][0]["label"] == "left"
    assert next_session["pending_entries"][0]["source_row_index"] == 2
    assert "Assigned to left" in status_messages[-1]
    assert set_sessions[-1]["pending_entries"][0]["source_row_index"] == 2


def test_geometry_manual_select_q_group_at_tags_branch_mosaic_top_candidate() -> None:
    key = ("q_group", "primary", 1, 0)
    entries = [
        {
            "label": "near-low",
            "hkl": (1, 0, 0),
            "sim_col": 10.0,
            "sim_row": 10.0,
            "branch_id": "+x",
            "branch_source": "generated",
            "mosaic_weight": 0.1,
            "best_sample_index": 3,
            "source_row_index": 30,
        },
        {
            "label": "top",
            "hkl": (1, 0, 0),
            "sim_col": 14.0,
            "sim_row": 10.0,
            "branch_id": "+x",
            "branch_source": "generated",
            "mosaic_weight": 0.9,
            "best_sample_index": 0,
            "source_row_index": 31,
        },
        {
            "label": "other-branch",
            "hkl": (1, 0, 0),
            "sim_col": 10.2,
            "sim_row": 10.2,
            "branch_id": "-x",
            "branch_source": "generated",
            "mosaic_weight": 9.0,
            "best_sample_index": 1,
            "source_row_index": 41,
        },
    ]
    sessions: list[dict[str, object]] = []

    handled, session, armed = mg.geometry_manual_toggle_selection_at(
        10.0,
        10.0,
        pick_session=None,
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {"grouped_candidates": {key: entries}},
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries_arg: list(entries_arg or []),
        set_pick_session_fn=lambda value: sessions.append(dict(value)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        listed_q_group_entries=lambda: [{"key": key}],
        format_q_group_line=lambda _entry: "selected group",
        choose_group_at_fn=lambda *_args, **_kwargs: (key, [dict(e) for e in entries], 0.0),
        use_caked_space=False,
        pick_search_window_px=20.0,
    )

    assert handled is True
    assert armed is True
    assert session["target_count"] == 2
    assert len(session["group_entries"]) == 2
    assert session["tagged_candidate"]["label"] == "top"
    assert session["tagged_candidate"]["branch_id"] == "+x"
    assert session["tagged_candidate"]["selection_reason"] == "mosaic_top_per_branch"
    assert session["tagged_candidate"]["best_sample_index"] == 0
    assert sessions[-1]["tagged_candidate"]["source_row_index"] == 31
    by_branch = {entry["branch_id"]: entry for entry in session["group_entries"]}
    assert by_branch["+x"]["source_row_index"] == 31
    assert by_branch["-x"]["source_row_index"] == 41


def test_geometry_manual_select_q_group_at_uses_profile_cache_sample_weights() -> None:
    key = ("q_group", "primary", 1, 0)
    entries = [
        {
            "label": "near-low",
            "hkl": (1, 0, 0),
            "sim_col": 10.0,
            "sim_row": 10.0,
            "best_sample_index": 0,
            "source_row_index": 30,
        },
        {
            "label": "top",
            "hkl": (1, 0, 0),
            "sim_col": 14.0,
            "sim_row": 10.0,
            "best_sample_index": 1,
            "source_row_index": 31,
        },
        {
            "label": "other-branch",
            "hkl": (1, 0, 0),
            "sim_col": 10.2,
            "sim_row": 10.2,
            "best_sample_index": 2,
            "source_row_index": 41,
        },
    ]
    profile_cache = {
        "beam_x_array": np.asarray([0.5, 0.5, -0.5], dtype=float),
        "sample_weights": np.asarray([0.1, 0.9, 9.0], dtype=float),
    }

    handled, session, armed = mg.geometry_manual_toggle_selection_at(
        10.0,
        10.0,
        pick_session=None,
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {"grouped_candidates": {key: entries}},
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries_arg: list(entries_arg or []),
        set_pick_session_fn=lambda _value: None,
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        listed_q_group_entries=lambda: [{"key": key}],
        format_q_group_line=lambda _entry: "selected group",
        choose_group_at_fn=lambda *_args, **_kwargs: (key, [dict(e) for e in entries], 0.0),
        use_caked_space=False,
        pick_search_window_px=20.0,
        profile_cache=profile_cache,
    )

    assert handled is True
    assert armed is True
    assert all("mosaic_weight" not in entry for entry in entries)
    assert session["target_count"] == 2
    assert len(session["group_entries"]) == 2
    assert session["tagged_candidate"]["label"] == "top"
    assert session["tagged_candidate"]["branch_id"] == "+x"
    assert session["tagged_candidate"]["selection_reason"] == "mosaic_top_per_branch"
    assert session["tagged_candidate"]["mosaic_weight"] == 0.9
    assert session["tagged_candidate"]["source_row_index"] == 31
    by_branch = {entry["branch_id"]: entry for entry in session["group_entries"]}
    assert by_branch["+x"]["source_row_index"] == 31
    assert by_branch["-x"]["source_row_index"] == 41


def test_geometry_manual_place_selection_at_refines_with_tagged_candidate_context() -> None:
    selected_ray = {
        "label": "tagged",
        "hkl": (1, 0, 0),
        "sim_col": 10.0,
        "sim_row": 10.0,
        "branch_id": "+x",
        "branch_source": "generated",
        "mosaic_weight": 0.9,
        "best_sample_index": 0,
        "source_table_index": 1,
        "source_row_index": 31,
    }
    seen: dict[str, object] = {}
    saved_entry_sets: list[list[dict[str, object]]] = []

    def _refine_preview_point(source_entry, raw_col, raw_row, **_kwargs):
        seen["source_entry"] = dict(source_entry) if source_entry else None
        return float(raw_col) + 0.2, float(raw_row) + 0.4

    handled, next_session = mg.geometry_manual_place_selection_at(
        5.0,
        6.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [dict(selected_ray)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate": dict(selected_ray),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=_refine_preview_point,
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        use_caked_space=False,
    )

    assert handled is True
    assert next_session == {}
    assert seen["source_entry"] is not None
    assert seen["source_entry"]["source_row_index"] == 31
    pair = saved_entry_sets[-1][0]
    assert pair["raw_x"] == 5.0
    assert pair["raw_y"] == 6.0
    assert pair["x"] == 5.2
    assert pair["y"] == 6.4
    assert pair["source_row_index"] == 31
    assert pair["best_sample_index"] == 0
    assert pair["mosaic_weight"] == 0.9


def test_geometry_manual_place_selection_at_uses_profile_cache_representative() -> None:
    key = ("q_group", "primary", 1, 0)
    candidates = [
        {
            "label": "near-low",
            "hkl": (1, 0, 0),
            "sim_col": 10.0,
            "sim_row": 10.0,
            "best_sample_index": 0,
            "source_table_index": 1,
            "source_row_index": 30,
        },
        {
            "label": "top",
            "hkl": (1, 0, 0),
            "sim_col": 14.0,
            "sim_row": 10.0,
            "best_sample_index": 1,
            "source_table_index": 1,
            "source_row_index": 31,
        },
    ]
    profile_cache = {
        "beam_x_array": np.asarray([0.5, 0.5], dtype=float),
        "sample_weights": np.asarray([0.1, 0.9], dtype=float),
    }
    seen: dict[str, object] = {}
    saved_entry_sets: list[list[dict[str, object]]] = []

    def _refine_preview_point(source_entry, raw_col, raw_row, **_kwargs):
        seen["source_entry"] = dict(source_entry)
        return float(raw_col), float(raw_row)

    handled, _next_session = mg.geometry_manual_place_selection_at(
        10.0,
        10.0,
        pick_session={
            "group_key": key,
            "group_entries": [dict(entry) for entry in candidates],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=_refine_preview_point,
        set_pairs_for_index_fn=lambda _idx, rows: (
            saved_entry_sets.append(list(rows or [])) or list(rows or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (float(col), float(row)),
        profile_cache=profile_cache,
    )

    assert handled is True
    assert seen["source_entry"]["label"] == "top"
    assert seen["source_entry"]["branch_id"] == "+x"
    assert seen["source_entry"]["mosaic_weight"] == 0.9
    assert saved_entry_sets[-1][0]["source_row_index"] == 31
    assert saved_entry_sets[-1][0]["mosaic_weight"] == 0.9


def test_geometry_manual_place_selection_at_uses_tagged_branch_representative() -> None:
    key = ("q_group", "primary", 6, 2)
    low = {
        "label": "low",
        "q_group_key": key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 60,
        "source_reflection_key": ("full", 60),
        "source_ray_id": "low-ray",
        "hkl": (6, 0, 2),
        "mosaic_weight": 0.2,
        "sim_col": 10.0,
        "sim_row": 10.0,
        "source_table_index": 1,
        "source_row_index": 60,
    }
    top = {
        "label": "top",
        "q_group_key": key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 61,
        "source_reflection_key": ("full", 61),
        "source_ray_id": "top-ray",
        "hkl": (-6, 0, 2),
        "mosaic_weight": 0.9,
        "sim_col": 20.0,
        "sim_row": 20.0,
        "source_table_index": 1,
        "source_row_index": 61,
    }
    seen: dict[str, object] = {}
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        10.0,
        10.0,
        pick_session={
            "group_key": key,
            "group_entries": [dict(low), dict(top)],
            "pending_entries": [],
            "target_count": 2,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate": dict(low),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda source_entry, raw_col, raw_row, **_kwargs: (
            seen.setdefault("source_entry", dict(source_entry)) and (float(raw_col), float(raw_row))
        ),
        set_pairs_for_index_fn=lambda _idx, rows: (
            saved_entry_sets.append(list(rows or [])) or list(rows or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (float(col), float(row)),
    )

    assert handled is True
    assert seen["source_entry"]["label"] == "low"
    assert seen["source_entry"]["selection_reason"] == "mosaic_top_per_branch"
    assert seen["source_entry"]["branch_id"] == "+x"
    assert seen["source_entry"]["source_branch_index"] == 0
    assert seen["source_entry"]["source_reflection_index"] == 60
    assert seen["source_entry"]["source_ray_id"] == "low-ray"
    assert saved_entry_sets == []
    assert next_session["pending_entries"][0]["source_reflection_index"] == 60
    assert next_session["pending_entries"][0]["source_ray_id"] == "low-ray"
    assert next_session["target_count"] == 2


def test_geometry_manual_place_selection_at_caked_background_refines_to_peak_top() -> None:
    radial = np.linspace(0.0, 30.0, 301)
    azimuth = np.linspace(-5.0, 5.0, 201)
    image = np.zeros((azimuth.size, radial.size), dtype=float)
    peak_col = int(np.argmin(np.abs(radial - 10.2)))
    peak_row = int(np.argmin(np.abs(azimuth - 0.3)))
    decoy_col = int(np.argmin(np.abs(radial - 25.0)))
    decoy_row = int(np.argmin(np.abs(azimuth - 0.0)))
    image[peak_row, peak_col] = 10.0
    image[decoy_row, decoy_col] = 1000.0
    saved_entry_sets: list[list[dict[str, object]]] = []
    candidate = {
        "label": "seed",
        "hkl": (1, 0, 0),
        "caked_x": 10.0,
        "caked_y": 0.0,
        "branch_id": "+x",
        "branch_source": "generated",
        "mosaic_weight": 1.0,
        "best_sample_index": 0,
        "source_table_index": 1,
        "source_row_index": 31,
    }

    handled, next_session = mg.geometry_manual_place_selection_at(
        10.0,
        0.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [dict(candidate)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate": dict(candidate),
        },
        current_background_index=0,
        display_background=image,
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda source_entry, raw_col, raw_row, **kwargs: (
            mg.geometry_manual_refine_preview_point(
                source_entry,
                raw_col,
                raw_row,
                use_caked_space=True,
                radial_axis=radial,
                azimuth_axis=azimuth,
                **kwargs,
            )
        ),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda tth, phi: (
            float(tth) * 10.0,
            float(phi) * 10.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
    )

    assert handled is True
    assert next_session == {}
    pair = saved_entry_sets[-1][0]
    assert pair["raw_caked_x"] == 10.0
    assert pair["raw_caked_y"] == 0.0
    assert np.isclose(pair["caked_x"], 10.2, atol=0.12)
    assert np.isclose(pair["caked_y"], 0.3, atol=0.12)
    assert pair["detector_x"] == pair["x"]
    assert pair["detector_y"] == pair["y"]
    assert pair["source_row_index"] == 31
    assert pair["best_sample_index"] == 0


def test_geometry_manual_place_selection_at_back_projects_caked_pick_to_detector_space() -> None:
    set_sessions: list[dict[str, object]] = []
    saved_entry_sets: list[list[dict[str, object]]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        13.0,
        2.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [
                {
                    "label": "1,0,0",
                    "hkl": (1, 0, 0),
                    "sim_col": 190.0,
                    "sim_row": 96.0,
                    "caked_x": 13.2,
                    "caked_y": 2.5,
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "zoom_active": True,
            "zoom_center": (13.0, 2.0),
            "saved_xlim": (10.0, 16.0),
            "saved_ylim": (-4.0, 4.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (13.2, 2.5),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(
            ("render", kwargs.get("update_status"))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: calls.append(("undo", None)),
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            phi + 100.0,
            two_theta + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 1.0,
            float(row) - 2.0,
        ),
    )

    assert handled is True
    assert next_session == {}
    assert set_sessions[-1] == {}
    assert ("undo", None) in calls
    assert saved_entry_sets
    assert saved_entry_sets[-1][0]["x"] == 102.5
    assert saved_entry_sets[-1][0]["y"] == 213.2
    assert saved_entry_sets[-1][0]["detector_x"] == 101.5
    assert saved_entry_sets[-1][0]["detector_y"] == 211.2
    assert saved_entry_sets[-1][0]["caked_x"] == 13.2
    assert saved_entry_sets[-1][0]["caked_y"] == 2.5


def test_geometry_manual_refine_preview_point_is_repeatable_with_peak_context_seed() -> None:
    calls: list[dict[str, object]] = []

    def _fake_match(candidates, background_context, match_cfg):
        calls.append(
            {
                "candidate": dict(candidates[0]),
                "background_context": dict(background_context),
                "match_cfg": dict(match_cfg),
            }
        )
        return ([{"x": 11.0, "y": 12.0}], {"status": "ok"})

    first = mg.geometry_manual_refine_preview_point(
        {"sim_col": 30.0, "sim_row": 31.0, "source_reflection_index": 9},
        10.0,
        20.0,
        display_background=np.zeros((8, 8), dtype=float),
        cache_data={
            "match_config": {"search_radius_px": 6.0},
            "background_context": {"img_valid": True},
        },
        use_caked_space=False,
        match_simulated_peaks_to_peak_context=_fake_match,
    )
    second = mg.geometry_manual_refine_preview_point(
        {"sim_col": 30.0, "sim_row": 31.0, "source_reflection_index": 9},
        10.0,
        20.0,
        display_background=np.zeros((8, 8), dtype=float),
        cache_data={
            "match_config": {"search_radius_px": 6.0},
            "background_context": {"img_valid": True},
        },
        use_caked_space=False,
        match_simulated_peaks_to_peak_context=_fake_match,
    )

    assert first == (11.0, 12.0)
    assert second == (11.0, 12.0)
    assert calls == [
        {
            "candidate": {"sim_col": 30.0, "sim_row": 31.0, "source_reflection_index": 9},
            "background_context": {"img_valid": True},
            "match_cfg": {"search_radius_px": 6.0},
        },
        {
            "candidate": {"sim_col": 30.0, "sim_row": 31.0, "source_reflection_index": 9},
            "background_context": {"img_valid": True},
            "match_cfg": {"search_radius_px": 6.0},
        },
    ]


def test_build_geometry_manual_pick_cache_matches_active_group_multiset_for_cache_and_rebuild() -> (
    None
):
    active_group_key = ("q_group", "primary", 1, 5)
    cached_rows = [
        {
            "label": "left",
            "hkl": (-1, 0, 5),
            "q_group_key": active_group_key,
            "source_reflection_index": 8,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_table_index": 8,
            "source_row_index": 0,
            "sim_col": 10.0,
            "sim_row": 20.0,
            "weight": 1.0,
        },
        {
            "label": "right",
            "hkl": (-1, 0, 5),
            "q_group_key": active_group_key,
            "source_reflection_index": 9,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_table_index": 9,
            "source_row_index": 0,
            "sim_col": 30.0,
            "sim_row": 40.0,
            "weight": 1.0,
        },
        {
            "label": "00l",
            "hkl": (0, 0, 3),
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_reflection_index": 1,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_table_index": 1,
            "source_row_index": 0,
            "sim_col": 50.0,
            "sim_row": 60.0,
            "weight": 1.0,
        },
    ]

    def _simulate(_params, *, prefer_cache):
        return list(reversed(cached_rows)) if prefer_cache else [dict(row) for row in cached_rows]

    cached_cache_data, _, _ = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((8, 8), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=_simulate,
        peak_records=[],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
    )
    rebuilt_cache_data, _, _ = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=False,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((8, 8), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=_simulate,
        peak_records=[],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
    )

    assert cached_cache_data["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=True)"
    )
    assert rebuilt_cache_data["cache_metadata"]["cache_source"] == (
        "geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
    )
    assert set(cached_cache_data["grouped_candidates"]) == set(
        rebuilt_cache_data["grouped_candidates"]
    )
    assert _candidate_multiset(cached_cache_data["grouped_candidates"][active_group_key]) == (
        _candidate_multiset(rebuilt_cache_data["grouped_candidates"][active_group_key])
    )
    assert mg.geometry_manual_group_target_count(
        active_group_key,
        cached_cache_data["grouped_candidates"][active_group_key],
    ) == mg.geometry_manual_group_target_count(
        active_group_key,
        rebuilt_cache_data["grouped_candidates"][active_group_key],
    )


def test_geometry_manual_choose_group_at_prefers_caked_coords_in_caked_view() -> None:
    caked_near = {
        "label": "caked-near",
        "caked_x": 13.1,
        "caked_y": 2.1,
        "sim_col": 500.0,
        "sim_row": 600.0,
    }
    detector_near = {
        "label": "detector-near",
        "caked_x": 30.0,
        "caked_y": 40.0,
        "sim_col": 13.2,
        "sim_row": 2.2,
    }

    group_key, group_entries, dist = mg.geometry_manual_choose_group_at(
        {
            ("good",): [caked_near],
            ("bad",): [detector_near],
        },
        13.0,
        2.0,
        window_size_px=4.0,
        use_caked_display=True,
    )

    assert group_key == ("good",)
    assert group_entries == [caked_near]
    assert dist < 1.0


def test_geometry_manual_place_selection_at_repeats_same_trusted_pair_for_caked_projection() -> (
    None
):
    candidate = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 9,
        "source_row_index": 0,
        "sim_col": 190.0,
        "sim_row": 96.0,
        "caked_x": 13.2,
        "caked_y": 2.5,
    }

    def _run_once():
        saved_entry_sets: list[list[dict[str, object]]] = []
        handled, next_session = mg.geometry_manual_place_selection_at(
            13.0,
            2.0,
            pick_session={
                "group_key": ("q_group", "primary", 1, 5),
                "group_entries": [dict(candidate)],
                "pending_entries": [],
                "target_count": 1,
                "base_entries": [],
                "q_label": "selected group",
                "background_index": 0,
                "zoom_active": True,
                "zoom_center": (13.0, 2.0),
                "saved_xlim": (10.0, 16.0),
                "saved_ylim": (-4.0, 4.0),
            },
            current_background_index=0,
            display_background=np.zeros((8, 8), dtype=float),
            get_cache_data=lambda **_kwargs: {},
            refine_preview_point=lambda *_args, **_kwargs: (13.2, 2.5),
            set_pairs_for_index_fn=lambda _idx, entries: (
                saved_entry_sets.append(list(entries or [])) or list(entries or [])
            ),
            set_pick_session_fn=lambda _session: None,
            clear_preview_artists_fn=lambda **_kwargs: None,
            restore_view_fn=lambda **_kwargs: None,
            render_current_pairs_fn=lambda **_kwargs: None,
            update_button_label_fn=lambda: None,
            set_status_text=lambda _text: None,
            push_undo_state_fn=lambda: None,
            use_caked_space=True,
            caked_angles_to_background_display_coords=lambda two_theta, phi: (
                float(phi) + 100.0,
                float(two_theta) + 200.0,
            ),
            background_display_to_native_detector_coords=lambda col, row: (
                float(col) - 1.0,
                float(row) - 2.0,
            ),
        )
        assert handled is True
        assert next_session == {}
        return dict(saved_entry_sets[-1][0])

    first = _run_once()
    second = _run_once()

    for entry in (first, second):
        assert entry["hkl"] == (-1, 0, 5)
        assert entry["source_reflection_index"] == 9
        assert entry["source_reflection_namespace"] == "full_reflection"
        assert entry["source_reflection_is_full"] is True
        assert entry["source_branch_index"] == 1
        assert entry["source_peak_index"] == 1
        assert entry["x"] == 102.5
        assert entry["y"] == 213.2
        assert entry["detector_x"] == 101.5
        assert entry["detector_y"] == 211.2
        assert entry["caked_x"] == 13.2
        assert entry["caked_y"] == 2.5
        assert entry["raw_caked_x"] == 13.0
        assert entry["raw_caked_y"] == 2.0
    assert first == second


def test_geometry_manual_place_selection_at_prefers_caked_nearest_candidate_in_caked_view() -> None:
    caked_near = {
        "label": "caked-near",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 9,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 9,
        "source_row_index": 0,
        "caked_x": 13.2,
        "caked_y": 2.4,
        "sim_col": 500.0,
        "sim_row": 600.0,
    }
    detector_near = {
        "label": "detector-near",
        "hkl": (1, 0, 5),
        "source_reflection_index": 10,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 9,
        "source_row_index": 1,
        "caked_x": 30.0,
        "caked_y": 40.0,
        "sim_col": 13.15,
        "sim_row": 2.35,
    }
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        13.0,
        2.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 5),
            "group_entries": [dict(caked_near), dict(detector_near)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "zoom_active": True,
            "zoom_center": (13.0, 2.0),
            "saved_xlim": (10.0, 16.0),
            "saved_ylim": (-4.0, 4.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (13.1, 2.2),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(phi) + 100.0,
            float(two_theta) + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 1.0,
            float(row) - 2.0,
        ),
    )

    assert handled is True
    assert next_session == {}
    assert saved_entry_sets[-1][0]["label"] == "caked-near"
    assert saved_entry_sets[-1][0]["source_row_index"] == 0


def test_runtime_projection_uses_sim_detector_adapter_after_caked_view() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    raw_row = {
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "winning-ray",
        "hkl": (-3, 0, 4),
        "mosaic_weight": 0.9,
        "native_col": 3.0,
        "native_row": 4.0,
    }

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: use_caked["value"],
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([10.0, 20.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([20.0, 30.0], dtype=float),
        current_background_display=lambda: np.zeros((64, 64), dtype=float),
        current_background_native=lambda: np.zeros((64, 64), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 900.0,
            float(row) + 900.0,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(raw_row)],
    )

    caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    assert len(caked_rows) == 1
    assert caked_rows[0]["display_col"] == 13.0
    assert caked_rows[0]["display_row"] == 24.0
    assert caked_rows[0]["sim_col_raw"] == 103.0
    assert caked_rows[0]["sim_row_raw"] == 204.0

    use_caked["value"] = False
    detector_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    assert len(detector_rows) == 1
    visible = detector_rows[0]
    assert visible["display_col"] == 103.0
    assert visible["display_row"] == 204.0
    assert visible["sim_col"] == 103.0
    assert visible["sim_row"] == 204.0
    assert visible["branch_id"] == "-x"
    assert visible["source_branch_index"] == 1
    assert visible["source_reflection_index"] == 17
    assert visible["source_ray_id"] == "winning-ray"


def _cross_view_selection_callbacks(use_caked, raw_rows):
    return mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: use_caked["value"],
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([10.0, 20.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([20.0, 30.0], dtype=float),
        current_background_display=lambda: np.zeros((256, 256), dtype=float),
        current_background_native=lambda: np.zeros((256, 256), dtype=float),
        image_size=256,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(row) for row in raw_rows],
    )


def _group_by_q_group(entries):
    grouped = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        group_key = raw_entry.get("q_group_key")
        if not isinstance(group_key, tuple):
            continue
        grouped.setdefault(group_key, []).append(dict(raw_entry))
    return grouped


def _build_cross_view_pick_cache(callbacks, raw_rows):
    cache_data, _next_sig, _next_state = mg.build_geometry_manual_pick_cache(
        param_set={"a": 1.0},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((256, 256), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            True,
            ("bg", 0),
            1,
            ("('q_group', 'primary', 3, 4)",),
        ),
        simulated_peaks_for_params=lambda *_args, **_kwargs: [dict(row) for row in raw_rows],
        build_grouped_candidates=callbacks.pick_candidates,
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        current_match_config=lambda: {},
    )
    return cache_data


def _cross_view_caked_projection_lookup_entry(cache_data, oracle):
    projection_lookup = cache_data["caked_qr_projection_lookup"]
    key = tuple(oracle["identity"])
    bucket = projection_lookup[key]
    if isinstance(bucket, list):
        assert len(bucket) == 1
        return dict(bucket[0])
    return dict(bucket)


def test_detector_qr_selection_uses_sim_display_frame_for_hit_table_rows() -> None:
    group_key = ("q_group", "primary", 3, 4)
    sim_row = {
        "label": "sim-hit",
        "q_group_key": group_key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 16,
        "source_ray_id": "sim-ray",
        "source_table_index": 0,
        "source_row_index": 2,
        "hkl": (3, 0, 4),
        "native_col": 10.0,
        "native_row": 20.0,
    }
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((80, 100), dtype=float),
        current_background_native=lambda: np.zeros((80, 100), dtype=float),
        image_size=100,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            79.0,
            10.0,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(sim_row)],
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)

    assert len(projected) == 1
    assert projected[0]["display_col"] == 10.0
    assert projected[0]["display_row"] == 20.0
    assert projected[0]["sim_col_raw"] == 10.0
    assert projected[0]["sim_row_raw"] == 20.0

    grouped = callbacks.pick_candidates(projected)
    group_key_found, group_entries, dist = mg.geometry_manual_choose_group_at(
        grouped,
        10.0,
        20.0,
        window_size_px=10.0,
        use_caked_display=False,
    )
    assert group_key_found == group_key
    assert len(group_entries) == 1
    assert dist < 1.0

    wrong_group_key, wrong_entries, wrong_dist = mg.geometry_manual_choose_group_at(
        grouped,
        79.0,
        10.0,
        window_size_px=10.0,
        use_caked_display=False,
    )
    assert wrong_group_key is None
    assert wrong_entries == []
    assert not np.isfinite(wrong_dist)

    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )
    _idx, record, pick_dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        10.0,
        20.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=5.0,
        simulation_point_candidates=[dict(projected[0])],
        use_caked_display=False,
    )
    assert within is True
    assert pick_dist < 1.0
    assert record is not None
    assert record["q_group_key"] == group_key
    assert record["branch_id"] == "+x"
    assert record["source_reflection_index"] == 16
    assert record["source_ray_id"] == "sim-ray"

    _idx, wrong_record, _wrong_pick_dist, wrong_within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        79.0,
        10.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=5.0,
        simulation_point_candidates=[dict(projected[0])],
        use_caked_display=False,
    )
    assert wrong_record is not None
    assert wrong_within is False


def test_detector_qr_selection_allows_explicit_background_detector_rows() -> None:
    group_key = ("q_group", "primary", 3, 4)
    background_row = {
        "label": "background-hit",
        "q_group_key": group_key,
        "coordinate_frame": "background_detector",
        "detector_display_source": "native_detector_coords_to_detector_display_coords",
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 16,
        "source_ray_id": "background-ray",
        "hkl": (3, 0, 4),
        "native_col": 10.0,
        "native_row": 20.0,
    }
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((80, 100), dtype=float),
        current_background_native=lambda: np.zeros((80, 100), dtype=float),
        image_size=100,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            79.0,
            10.0,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(background_row)],
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)

    assert len(projected) == 1
    assert projected[0]["display_col"] == 79.0
    assert projected[0]["display_row"] == 10.0
    assert projected[0]["sim_col_raw"] == 79.0
    assert projected[0]["sim_row_raw"] == 10.0


def _toggle_cross_view_selection(
    grouped,
    col,
    row,
    *,
    use_caked_space,
    group_key,
    cache_data=None,
):
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        float(col),
        float(row),
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((256, 256), dtype=float),
        get_cache_data=lambda **_kwargs: (
            dict(cache_data)
            if isinstance(cache_data, dict)
            else {
                "signature": ("cross-view", bool(use_caked_space)),
                "grouped_candidates": grouped,
                "caked_qr_projection_grouped_candidates": grouped if bool(use_caked_space) else {},
            }
        ),
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=bool(use_caked_space),
        pick_search_window_px=50.0,
    )
    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert set_sessions[-1]["group_key"] == group_key
    assert "Selected selected group" in status_messages[-1]
    return next_session


_CROSS_VIEW_ID_FIELDS = (
    "source_table_index",
    "source_row_index",
    "source_reflection_index",
    "source_branch_index",
    "source_ray_id",
    "branch_id",
)


def _selected_cross_view_group_entry(session):
    tagged_key = _source_key(session.get("tagged_candidate"))
    assert tagged_key is not None
    for index, raw_entry in enumerate(session.get("group_entries", ())):
        if _source_key(raw_entry) == tagged_key:
            return int(index), dict(raw_entry)
    raise AssertionError("selected group entry was not retained")


def _cross_view_caked_snapshot(callbacks, session):
    selected_index, selected_entry = _selected_cross_view_group_entry(session)
    projected_rows = callbacks.project_peaks_to_current_view([dict(selected_entry)])
    assert len(projected_rows) == 1
    projected = projected_rows[0]
    displayed = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        entry_display_coords=lambda _entry: None,
    )
    assert len(displayed) > selected_index
    assert "sim_display" in displayed[selected_index]
    return {
        "identity": tuple(projected.get(field) for field in _CROSS_VIEW_ID_FIELDS),
        "detector_display": (
            float(projected["sim_col_raw"]),
            float(projected["sim_row_raw"]),
        ),
        "caked_angles": (
            float(projected["two_theta_deg"]),
            float(projected["phi_deg"]),
        ),
        "caked_visual": (
            float(projected["display_col"]),
            float(projected["display_row"]),
        ),
        "caked_global": (
            float(projected["sim_col_global"]),
            float(projected["sim_row_global"]),
        ),
        "caked_sim_display": tuple(float(v) for v in displayed[selected_index]["sim_display"]),
    }


def _cross_view_detector_snapshot(callbacks, session):
    _selected_index, selected_entry = _selected_cross_view_group_entry(session)
    projected_rows = callbacks.project_peaks_to_current_view([dict(selected_entry)])
    assert len(projected_rows) == 1
    projected = projected_rows[0]
    return {
        "identity": tuple(projected.get(field) for field in _CROSS_VIEW_ID_FIELDS),
        "detector_visual": (
            float(projected["display_col"]),
            float(projected["display_row"]),
        ),
    }


def _assert_pair_close(left, right) -> None:
    np.testing.assert_allclose(
        np.asarray(left, dtype=float),
        np.asarray(right, dtype=float),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_build_geometry_manual_pick_cache_adds_caked_qr_projection_cache() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    clean_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_table_index": 0,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "qr": 1.5,
            "qz": 2.5,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "qr": 1.7,
            "qz": 2.7,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    poisoned_rows = []
    for index, row in enumerate(clean_rows):
        poisoned = dict(row)
        poisoned.update(
            {
                "refined_sim_x": 190.0 + float(index),
                "refined_sim_y": 96.0,
                "refined_sim_caked_x": 113.5 + float(index),
                "refined_sim_caked_y": -12.5,
                "display_col": 113.5 + float(index),
                "display_row": -12.5,
                "sim_col": 113.5 + float(index),
                "sim_row": -12.5,
            }
        )
        poisoned_rows.append(poisoned)

    raw_rows = [dict(row) for row in clean_rows]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)
    detector_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(callbacks.simulated_peaks_for_params(prefer_cache=True)),
        103.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )

    use_caked["value"] = True
    detector_to_caked = _cross_view_caked_snapshot(callbacks, detector_session)
    raw_rows[:] = [dict(row) for row in poisoned_rows]
    cache_data = _build_cross_view_pick_cache(callbacks, raw_rows)
    cache_entry = _cross_view_caked_projection_lookup_entry(cache_data, detector_to_caked)

    assert "caked_qr_projection_entries" in cache_data
    assert "caked_qr_projection_grouped_candidates" in cache_data
    assert "caked_qr_projection_lookup" in cache_data
    assert (
        tuple(cache_entry.get(field) for field in _CROSS_VIEW_ID_FIELDS)
        == detector_to_caked["identity"]
    )
    _assert_pair_close(
        (cache_entry["sim_col_raw"], cache_entry["sim_row_raw"]),
        detector_to_caked["detector_display"],
    )
    _assert_pair_close(
        (cache_entry["two_theta_deg"], cache_entry["phi_deg"]),
        detector_to_caked["caked_angles"],
    )
    _assert_pair_close(
        (cache_entry["display_col"], cache_entry["display_row"]),
        detector_to_caked["caked_visual"],
    )
    assert "refined_sim_caked_x" not in cache_entry
    assert "refined_sim_x" not in cache_entry


def test_manual_qr_caked_toggle_uses_projection_cache_for_hit_testing() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    stale_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
        "caked_x": 113.5,
        "caked_y": -12.5,
        "display_col": 113.5,
        "display_row": -12.5,
        "sim_col": 113.5,
        "sim_row": -12.5,
    }
    source_entry = {
        **dict(stale_entry),
        "caked_x": None,
        "caked_y": None,
        "display_col": None,
        "display_row": None,
        "sim_col": None,
        "sim_row": None,
    }
    raw_rows = [dict(source_entry)]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)
    projected_entry = callbacks.project_peaks_to_current_view([dict(source_entry)])[0]
    projected_entry["_caked_qr_projection_cache"] = True
    cache_data = {
        "signature": ("manual-cache",),
        "grouped_candidates": _group_by_q_group([dict(stale_entry)]),
        "caked_qr_projection_grouped_candidates": callbacks.pick_candidates(
            [dict(projected_entry)]
        ),
    }

    session = _toggle_cross_view_selection(
        cache_data["grouped_candidates"],
        float(projected_entry["display_col"]),
        float(projected_entry["display_row"]),
        use_caked_space=True,
        group_key=group_key,
        cache_data=cache_data,
    )
    selected_index, selected_entry = _selected_cross_view_group_entry(session)
    displayed = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=None,
        entry_display_coords=lambda _entry: None,
    )

    assert selected_index == 0
    assert tuple(selected_entry.get(field) for field in _CROSS_VIEW_ID_FIELDS) == tuple(
        projected_entry.get(field) for field in _CROSS_VIEW_ID_FIELDS
    )
    _assert_pair_close(
        displayed[0]["sim_display"],
        (projected_entry["display_col"], projected_entry["display_row"]),
    )


def test_manual_qr_caked_toggle_requires_projection_cache_without_grouped_fallback() -> None:
    group_key = ("q_group", "primary", 3, 4)
    stale_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
        "display_col": 13.0,
        "display_row": 24.0,
        "sim_col": 103.0,
        "sim_row": 204.0,
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
    }
    cache_data = {
        "signature": ("manual-cache",),
        "grouped_candidates": _group_by_q_group([dict(stale_entry)]),
        "caked_qr_projection_grouped_candidates": {},
    }
    status_messages: list[str] = []

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        13.0,
        24.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: dict(cache_data),
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda _session: None,
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=True,
        pick_search_window_px=50.0,
    )

    assert handled is False
    assert next_session == {}
    assert suppress_drag is False
    assert "No simulated Qr/Qz groups" in status_messages[-1]

    detector_sessions: list[dict[str, object]] = []
    detector_handled, detector_session, detector_suppress_drag = (
        mg.geometry_manual_toggle_selection_at(
            103.0,
            204.0,
            pick_session={},
            current_background_index=0,
            display_background=np.zeros((8, 8), dtype=float),
            get_cache_data=lambda **_kwargs: dict(cache_data),
            pairs_for_index=lambda _idx: [],
            set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
            set_pick_session_fn=lambda session: detector_sessions.append(dict(session)),
            restore_view_fn=lambda **_kwargs: None,
            clear_preview_artists_fn=lambda **_kwargs: None,
            render_current_pairs_fn=lambda **_kwargs: None,
            update_button_label_fn=lambda: None,
            set_status_text=lambda _text: None,
            listed_q_group_entries=lambda: [{"key": group_key}],
            format_q_group_line=lambda _entry: "selected group",
            use_caked_space=False,
            pick_search_window_px=50.0,
        )
    )

    assert detector_handled is True
    assert detector_suppress_drag is True
    assert detector_session["group_key"] == group_key
    assert detector_sessions[-1]["group_key"] == group_key


def test_caked_qr_projection_cache_rejects_alias_only_simulated_rows() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    alias_only_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "refined_sim_x": 900.0,
        "refined_sim_y": 901.0,
        "refined_sim_caked_x": 902.0,
        "refined_sim_caked_y": 903.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
        "display_col": 13.0,
        "display_row": 24.0,
        "sim_col": 13.0,
        "sim_row": 24.0,
    }
    callbacks = _cross_view_selection_callbacks(use_caked, [dict(alias_only_entry)])

    projected_rows = callbacks.project_peaks_to_current_view([dict(alias_only_entry)])
    cache_data = _build_cross_view_pick_cache(callbacks, [dict(alias_only_entry)])

    assert projected_rows == []
    assert cache_data["caked_qr_projection_entries"] == []
    assert cache_data["caked_qr_projection_grouped_candidates"] == {}
    assert cache_data["caked_qr_projection_lookup"] == {}


def test_project_peaks_to_current_view_caked_keeps_legacy_detector_raw_rows() -> None:
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([10.0, 20.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([20.0, 30.0], dtype=float),
        current_background_display=lambda: np.zeros((256, 256), dtype=float),
        current_background_native=lambda: np.zeros((256, 256), dtype=float),
        image_size=256,
        display_to_native_sim_coords=lambda col, row, _shape: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "legacy-detector",
                "q_group_key": ("q_group", "primary", 3, 4),
                "sim_col_raw": 103.0,
                "sim_row_raw": 204.0,
            }
        ]
    )

    assert len(projected) == 1
    row = projected[0]
    assert row["native_col"] == 3.0
    assert row["native_row"] == 4.0
    assert row["sim_col_raw"] == 103.0
    assert row["sim_row_raw"] == 204.0
    assert row["caked_x"] == 13.0
    assert row["caked_y"] == 24.0
    assert row["display_col"] == 13.0
    assert row["display_row"] == 24.0


def test_manual_qr_caked_saved_replay_uses_projection_cache_without_reprojection() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    source_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
    }
    callbacks = _cross_view_selection_callbacks(use_caked, [dict(source_entry)])
    projected_entry = callbacks.project_peaks_to_current_view([dict(source_entry)])[0]
    caked_lookup = {
        tuple(projected_entry.get(field) for field in _CROSS_VIEW_ID_FIELDS): dict(projected_entry)
    }
    saved = {
        **dict(source_entry),
        "x": 213.5,
        "y": 167.5,
        "detector_x": 113.5,
        "detector_y": -12.5,
        "caked_x": 113.5,
        "caked_y": -12.5,
        "background_two_theta_deg": 113.5,
        "background_phi_deg": -12.5,
        "refined_sim_native_x": 3.0,
        "refined_sim_native_y": 4.0,
        "refined_sim_caked_x": 113.5,
        "refined_sim_caked_y": -12.5,
    }

    _measured_display, saved_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {},
            "caked_qr_projection_lookup": caked_lookup,
        },
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=None,
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert saved_pairs[0]["bg_display"] == (113.5, -12.5)
    _assert_pair_close(
        saved_pairs[0]["sim_display"],
        (projected_entry["display_col"], projected_entry["display_row"]),
    )


def test_manual_qr_caked_saved_redraw_unresolved_without_projection_lookup() -> None:
    group_key = ("q_group", "primary", 3, 4)
    saved = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
        "x": 999.0,
        "y": 998.0,
        "detector_x": 997.0,
        "detector_y": 996.0,
        "caked_x": 113.5,
        "caked_y": -12.5,
        "background_two_theta_deg": 113.5,
        "background_phi_deg": -12.5,
        "refined_sim_caked_x": 902.0,
        "refined_sim_caked_y": 903.0,
        "display_col": 904.0,
        "display_row": 905.0,
        "sim_col": 906.0,
        "sim_row": 907.0,
    }

    _measured_display, saved_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {},
            "caked_qr_projection_lookup": {},
        },
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=lambda entries: [
            dict(entry) for entry in entries or () if isinstance(entry, dict)
        ],
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert saved_pairs[0]["bg_display"] == (113.5, -12.5)
    assert "sim_display" not in saved_pairs[0]
    assert saved_pairs[0]["sim_display_unresolved"] is True


def test_manual_qr_caked_active_background_refines_on_caked_image_separate_from_sim() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    radial = np.linspace(108.0, 116.0, 81)
    azimuth = np.linspace(-16.0, -8.0, 81)
    image = np.zeros((azimuth.size, radial.size), dtype=float)
    refined_bg = (113.8, -12.2)
    peak_col = int(np.argmin(np.abs(radial - refined_bg[0])))
    peak_row = int(np.argmin(np.abs(azimuth - refined_bg[1])))
    decoy_col = int(np.argmin(np.abs(radial - 110.0)))
    decoy_row = int(np.argmin(np.abs(azimuth - -15.0)))
    image[peak_row, peak_col] = 50.0
    image[decoy_row, decoy_col] = 5000.0
    source_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
        "refined_sim_x": 900.0,
        "refined_sim_y": 901.0,
        "refined_sim_caked_x": 902.0,
        "refined_sim_caked_y": 903.0,
        "display_col": 904.0,
        "display_row": 905.0,
        "sim_col": 906.0,
        "sim_row": 907.0,
    }
    callbacks = _cross_view_selection_callbacks(use_caked, [dict(source_entry)])
    projected_entry = callbacks.project_peaks_to_current_view([dict(source_entry)])[0]
    projected_entry["_caked_qr_projection_cache"] = True
    cache_data = {
        "signature": ("manual-cache",),
        "grouped_candidates": _group_by_q_group([dict(source_entry)]),
        "caked_qr_projection_grouped_candidates": callbacks.pick_candidates(
            [dict(projected_entry)]
        ),
    }

    session = _toggle_cross_view_selection(
        cache_data["grouped_candidates"],
        float(projected_entry["display_col"]),
        float(projected_entry["display_row"]),
        use_caked_space=True,
        group_key=group_key,
        cache_data=cache_data,
    )
    selected_index, selected_entry = _selected_cross_view_group_entry(session)
    refined = mg.geometry_manual_refine_preview_point(
        dict(selected_entry),
        113.5,
        -12.5,
        display_background=image,
        cache_data={},
        use_caked_space=True,
        radial_axis=radial,
        azimuth_axis=azimuth,
    )
    pending_entry = mg.geometry_manual_pair_entry_from_candidate(
        dict(selected_entry),
        313.8,
        147.8,
        group_key=group_key,
        detector_col=213.8,
        detector_row=-32.2,
        raw_col=313.5,
        raw_row=147.5,
        caked_col=float(refined[0]),
        caked_row=float(refined[1]),
        raw_caked_col=113.5,
        raw_caked_row=-12.5,
    )
    assert pending_entry is not None
    pending_entry.update(
        {
            "x": 999.0,
            "y": 998.0,
            "detector_x": 997.0,
            "detector_y": 996.0,
            "display_col": 995.0,
            "display_row": 994.0,
        }
    )
    active_session = dict(session)
    active_session["pending_entries"] = [dict(pending_entry)]

    displayed = mg.geometry_manual_session_initial_pairs_display(
        active_session,
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=None,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    _assert_pair_close(displayed[selected_index]["bg_display"], refined)
    _assert_pair_close(
        displayed[selected_index]["sim_display"],
        (projected_entry["display_col"], projected_entry["display_row"]),
    )


def test_manual_qr_caked_saved_background_redraw_uses_refined_caked_image_peak() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    radial = np.linspace(108.0, 116.0, 81)
    azimuth = np.linspace(-16.0, -8.0, 81)
    image = np.zeros((azimuth.size, radial.size), dtype=float)
    refined_bg = (113.8, -12.2)
    peak_col = int(np.argmin(np.abs(radial - refined_bg[0])))
    peak_row = int(np.argmin(np.abs(azimuth - refined_bg[1])))
    image[peak_row, peak_col] = 50.0
    source_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "native_col": 3.0,
        "native_row": 4.0,
        "refined_sim_x": 900.0,
        "refined_sim_y": 901.0,
        "refined_sim_caked_x": 902.0,
        "refined_sim_caked_y": 903.0,
        "display_col": 904.0,
        "display_row": 905.0,
        "sim_col": 906.0,
        "sim_row": 907.0,
    }
    callbacks = _cross_view_selection_callbacks(use_caked, [dict(source_entry)])
    projected_entry = callbacks.project_peaks_to_current_view([dict(source_entry)])[0]
    caked_lookup = {
        tuple(projected_entry.get(field) for field in _CROSS_VIEW_ID_FIELDS): dict(projected_entry)
    }
    session = {
        "group_key": group_key,
        "group_entries": [dict(projected_entry)],
        "pending_entries": [],
        "target_count": 1,
        "base_entries": [],
        "q_label": "selected group",
        "background_index": 0,
        "tagged_candidate": dict(projected_entry),
        "tagged_candidate_key": _source_key(projected_entry),
    }

    saved_entry_sets: list[list[dict[str, object]]] = []
    handled, next_session = mg.geometry_manual_place_selection_at(
        113.5,
        -12.5,
        pick_session=session,
        current_background_index=0,
        display_background=image,
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda source_entry_arg, raw_col, raw_row, **kwargs: (
            mg.geometry_manual_refine_preview_point(
                source_entry_arg,
                raw_col,
                raw_row,
                use_caked_space=True,
                radial_axis=radial,
                azimuth_axis=azimuth,
                **kwargs,
            )
        ),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(two_theta) + 200.0,
            float(phi) + 160.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
    )

    assert handled is True
    assert next_session == {}
    saved = dict(saved_entry_sets[-1][0])
    _assert_pair_close((saved["caked_x"], saved["caked_y"]), refined_bg)
    _assert_pair_close(
        (saved["background_two_theta_deg"], saved["background_phi_deg"]),
        refined_bg,
    )
    assert saved["manual_background_input_frame"] == "caked_2theta_phi"
    saved.update(
        {
            "x": 999.0,
            "y": 998.0,
            "detector_x": 997.0,
            "detector_y": 996.0,
            "display_col": 995.0,
            "display_row": 994.0,
        }
    )

    _measured_display, saved_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {},
            "caked_qr_projection_lookup": caked_lookup,
        },
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=None,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    _assert_pair_close(saved_pairs[0]["bg_display"], refined_bg)
    _assert_pair_close(
        saved_pairs[0]["sim_display"],
        (projected_entry["display_col"], projected_entry["display_row"]),
    )


def test_manual_qr_caked_direct_pick_matches_detector_origin_caked_baseline() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    clean_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_table_index": 0,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "mosaic_weight": 0.8,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "mosaic_weight": 0.9,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    poisoned_rows = []
    for index, row in enumerate(clean_rows):
        poisoned = dict(row)
        poisoned.update(
            {
                "refined_sim_x": 190.0 + float(index),
                "refined_sim_y": 96.0,
                "refined_sim_caked_x": 112.5 + float(index),
                "refined_sim_caked_y": -12.5,
                "display_col": 112.5 + float(index),
                "display_row": -12.5,
                "sim_col": 112.5 + float(index),
                "sim_row": -12.5,
            }
        )
        poisoned_rows.append(poisoned)
    raw_rows = [dict(row) for row in clean_rows]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)

    detector_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    detector_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(detector_rows),
        102.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )

    use_caked["value"] = True
    detector_to_caked = _cross_view_caked_snapshot(callbacks, detector_session)
    use_caked["value"] = False
    detector_baseline = _cross_view_detector_snapshot(callbacks, detector_session)

    raw_rows[:] = [dict(row) for row in poisoned_rows]
    use_caked["value"] = True
    direct_caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    direct_caked_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(direct_caked_rows),
        detector_to_caked["caked_visual"][0],
        detector_to_caked["caked_visual"][1],
        use_caked_space=True,
        group_key=group_key,
    )
    direct_caked = _cross_view_caked_snapshot(callbacks, direct_caked_session)

    use_caked["value"] = False
    direct_detector = _cross_view_detector_snapshot(callbacks, direct_caked_session)

    assert direct_caked["identity"] == detector_to_caked["identity"]
    assert direct_detector["identity"] == detector_baseline["identity"]
    for key in (
        "detector_display",
        "caked_angles",
        "caked_visual",
        "caked_global",
        "caked_sim_display",
    ):
        _assert_pair_close(direct_caked[key], detector_to_caked[key])
    _assert_pair_close(direct_detector["detector_visual"], detector_baseline["detector_visual"])
    _assert_pair_close(
        (
            direct_caked["caked_visual"][0] + 90.0,
            direct_caked["caked_visual"][1] + 180.0,
        ),
        direct_caked["detector_display"],
    )


def test_manual_qr_caked_projection_signature_tracks_axes_binning() -> None:
    image = np.zeros((8, 8), dtype=float)
    first = mg.geometry_manual_pick_placed_cache_signature(
        source_snapshot_signature=("sim", 1),
        background_index=0,
        background_image=image,
        use_caked_space=True,
        caked_projection_signature=(
            "caked",
            ("radial", 10.0, 20.0, 2),
            ("azimuth", 20.0, 30.0, 2),
        ),
    )
    changed_axes = mg.geometry_manual_pick_placed_cache_signature(
        source_snapshot_signature=("sim", 1),
        background_index=0,
        background_image=image,
        use_caked_space=True,
        caked_projection_signature=(
            "caked",
            ("radial", 10.0, 30.0, 3),
            ("azimuth", 20.0, 30.0, 2),
        ),
    )
    detector_mode = mg.geometry_manual_pick_placed_cache_signature(
        source_snapshot_signature=("sim", 1),
        background_index=0,
        background_image=image,
        use_caked_space=False,
        caked_projection_signature=(
            "caked",
            ("radial", 10.0, 30.0, 3),
            ("azimuth", 20.0, 30.0, 2),
        ),
    )

    assert first != changed_axes
    assert detector_mode[-1] is None


def test_manual_qr_caked_projection_signature_tracks_detector_display_transform() -> None:
    def _detector_display(col, row):
        return float(col), float(row)

    def _caked_display(col, row):
        return float(col), float(row)

    def _callbacks(display_rotate_k):
        return mg.make_runtime_geometry_manual_projection_callbacks(
            caked_view_enabled=lambda: True,
            last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
            last_caked_radial_values=lambda: np.array([10.0, 20.0], dtype=float),
            last_caked_azimuth_values=lambda: np.array([20.0, 30.0], dtype=float),
            current_background_display=lambda: np.zeros((64, 64), dtype=float),
            current_background_native=lambda: np.zeros((64, 64), dtype=float),
            image_size=64,
            native_detector_coords_to_detector_display_coords=_detector_display,
            simulation_native_detector_coords_to_caked_display_coords=_caked_display,
            display_rotate_k=int(display_rotate_k),
        )

    first = _callbacks(0).caked_projection_signature()
    rotated = _callbacks(1).caked_projection_signature()

    assert first != rotated


def test_manual_qr_caked_saved_replay_matches_detector_origin_caked_baseline() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    raw_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_table_index": 0,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "mosaic_weight": 0.8,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "mosaic_weight": 0.9,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)

    detector_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    detector_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(detector_rows),
        103.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )
    use_caked["value"] = True
    detector_to_caked = _cross_view_caked_snapshot(callbacks, detector_session)

    direct_caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    direct_caked_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(direct_caked_rows),
        detector_to_caked["caked_visual"][0],
        detector_to_caked["caked_visual"][1],
        use_caked_space=True,
        group_key=group_key,
    )
    direct_caked = _cross_view_caked_snapshot(callbacks, direct_caked_session)

    assert direct_caked["identity"] == detector_to_caked["identity"]
    _assert_pair_close(direct_caked["caked_sim_display"], detector_to_caked["caked_sim_display"])

    _selected_index, selected_entry = _selected_cross_view_group_entry(direct_caked_session)
    finish_session = dict(direct_caked_session)
    finish_session["group_entries"] = [dict(selected_entry)]
    finish_session["target_count"] = 1
    finish_session["pending_entries"] = []
    finish_session["base_entries"] = []
    finish_session["tagged_candidate"] = dict(selected_entry)
    finish_session["tagged_candidate_key"] = _source_key(selected_entry)
    clicked_caked = (113.5, -12.5)

    def _caked_to_detector(two_theta, phi):
        return float(two_theta) + 100.0, float(phi) + 180.0

    def _detector_to_native(col, row):
        return float(col) - 100.0, float(row) - 180.0

    def _poison_refined_sim_caked_with_measured_background(entry, candidate=None):
        poisoned = dict(entry)
        poisoned["refined_sim_caked_x"] = float(poisoned["caked_x"])
        poisoned["refined_sim_caked_y"] = float(poisoned["caked_y"])
        return poisoned

    saved_entry_sets: list[list[dict[str, object]]] = []
    handled, next_session = mg.geometry_manual_place_selection_at(
        clicked_caked[0],
        clicked_caked[1],
        pick_session=finish_session,
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: clicked_caked,
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=_caked_to_detector,
        background_display_to_native_detector_coords=_detector_to_native,
        refine_saved_pair_entry_fn=_poison_refined_sim_caked_with_measured_background,
    )

    assert handled is True
    assert next_session == {}
    saved = saved_entry_sets[-1][0]
    assert (
        tuple(saved.get(field) for field in _CROSS_VIEW_ID_FIELDS) == detector_to_caked["identity"]
    )
    assert saved["caked_x"] == clicked_caked[0]
    assert saved["caked_y"] == clicked_caked[1]
    assert saved["background_two_theta_deg"] == clicked_caked[0]
    assert saved["background_phi_deg"] == clicked_caked[1]
    assert saved["refined_sim_native_x"] == 3.0
    assert saved["refined_sim_native_y"] == 4.0

    _measured_display, saved_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(saved)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        source_rows_for_background=lambda *_args, **_kwargs: [dict(row) for row in raw_rows],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    assert saved_pairs[0]["bg_display"] == clicked_caked
    _assert_pair_close(saved_pairs[0]["sim_display"], detector_to_caked["caked_sim_display"])


def test_manual_qr_caked_saved_detector_replay_matches_detector_origin_baseline_after_finish() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    raw_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_table_index": 0,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "mosaic_weight": 0.8,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "mosaic_weight": 0.9,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)

    detector_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    detector_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(detector_rows),
        103.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )
    detector_baseline = _cross_view_detector_snapshot(callbacks, detector_session)

    use_caked["value"] = True
    detector_to_caked = _cross_view_caked_snapshot(callbacks, detector_session)
    direct_caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    direct_caked_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(direct_caked_rows),
        detector_to_caked["caked_visual"][0],
        detector_to_caked["caked_visual"][1],
        use_caked_space=True,
        group_key=group_key,
    )

    _selected_index, selected_entry = _selected_cross_view_group_entry(direct_caked_session)
    finish_session = dict(direct_caked_session)
    finish_session["group_entries"] = [dict(selected_entry)]
    finish_session["target_count"] = 1
    finish_session["pending_entries"] = []
    finish_session["base_entries"] = []
    finish_session["tagged_candidate"] = dict(selected_entry)
    finish_session["tagged_candidate_key"] = _source_key(selected_entry)
    clicked_caked = (113.5, -12.5)

    def _caked_to_detector(two_theta, phi):
        return float(two_theta) + 100.0, float(phi) + 180.0

    def _detector_to_native(col, row):
        return float(col) - 100.0, float(row) - 180.0

    def _poison_refined_sim_caked_with_measured_background(entry, candidate=None):
        poisoned = dict(entry)
        poisoned["refined_sim_caked_x"] = float(poisoned["caked_x"])
        poisoned["refined_sim_caked_y"] = float(poisoned["caked_y"])
        return poisoned

    saved_entry_sets: list[list[dict[str, object]]] = []
    handled, next_session = mg.geometry_manual_place_selection_at(
        clicked_caked[0],
        clicked_caked[1],
        pick_session=finish_session,
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: clicked_caked,
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=_caked_to_detector,
        background_display_to_native_detector_coords=_detector_to_native,
        refine_saved_pair_entry_fn=_poison_refined_sim_caked_with_measured_background,
    )

    assert handled is True
    assert next_session == {}
    saved = saved_entry_sets[-1][0]
    assert (
        tuple(saved.get(field) for field in _CROSS_VIEW_ID_FIELDS)
        == detector_baseline["identity"]
    )

    poisoned_saved = dict(saved)
    poisoned_saved.update(
        {
            "refined_sim_x": 913.5,
            "refined_sim_y": 812.5,
            "refined_sim_native_x": 703.0,
            "refined_sim_native_y": 704.0,
            "native_col": 705.0,
            "native_row": 706.0,
            "sim_native_x": 707.0,
            "sim_native_y": 708.0,
            "display_col": clicked_caked[0],
            "display_row": clicked_caked[1],
            "sim_col": clicked_caked[0],
            "sim_row": clicked_caked[1],
            "sim_col_raw": clicked_caked[0],
            "sim_row_raw": clicked_caked[1],
            "detector_x": 709.0,
            "detector_y": 710.0,
            "sim_detector_anchor_x": 711.0,
            "sim_detector_anchor_y": 712.0,
            "sim_detector_display_col": clicked_caked[0],
            "sim_detector_display_row": clicked_caked[1],
            "sim_detector_frame_provenance": "poisoned-cache",
        }
    )

    use_caked["value"] = False
    detector_cache = _build_cross_view_pick_cache(callbacks, raw_rows)
    _measured_display, detector_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(poisoned_saved)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: detector_cache,
        source_rows_for_background=lambda *_args, **_kwargs: [dict(row) for row in raw_rows],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        entry_display_coords=lambda entry: (
            float(entry["x"]),
            float(entry["y"]),
        ),
    )

    assert len(detector_pairs) == 1
    _assert_pair_close(detector_pairs[0]["sim_display"], detector_baseline["detector_visual"])
    assert tuple(float(v) for v in detector_pairs[0]["sim_display"]) != clicked_caked


def test_build_geometry_manual_initial_pairs_display_detector_replay_prefers_source_projection_over_detector_lookup() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    raw_rows = [
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_table_index": 0,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "native_col": 3.0,
            "native_row": 4.0,
        }
    ]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)
    detector_cache = _build_cross_view_pick_cache(callbacks, raw_rows)

    saved_entry = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "native_col": 3.0,
        "native_row": 4.0,
        "background_two_theta_deg": 23.0,
        "background_phi_deg": -26.0,
    }
    detector_lookup_row = {
        **saved_entry,
        "lookup_origin": "detector_lookup",
        "refined_sim_x": 901.0,
        "refined_sim_y": 902.0,
        "display_col": 903.0,
        "display_row": 904.0,
        "sim_col_raw": 905.0,
        "sim_row_raw": 906.0,
    }

    def _project_peaks_to_current_view(rows):
        projected_rows = []
        for row in rows or ():
            projected = dict(row)
            if projected.get("lookup_origin") == "detector_lookup":
                projected["refined_sim_x"] = 901.0
                projected["refined_sim_y"] = 902.0
                projected["display_col"] = 901.0
                projected["display_row"] = 902.0
                projected["sim_col_raw"] = 901.0
                projected["sim_row_raw"] = 902.0
            else:
                projected["display_col"] = 103.0
                projected["display_row"] = 204.0
                projected["sim_col_raw"] = 103.0
                projected["sim_row_raw"] = 204.0
            projected_rows.append(projected)
        return projected_rows

    cache_data = {
        "simulated_lookup": _build_lookup([detector_lookup_row]),
        "caked_qr_projection_lookup": detector_cache["caked_qr_projection_lookup"],
    }

    _measured_display, detector_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(saved_entry)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: cache_data,
        source_rows_for_background=lambda *_args, **_kwargs: [dict(row) for row in raw_rows],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=_project_peaks_to_current_view,
        entry_display_coords=lambda entry: (
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
        ),
    )

    assert len(detector_pairs) == 1
    assert detector_pairs[0]["sim_display"] == (103.0, 204.0)
    assert detector_pairs[0]["sim_display"] != (901.0, 902.0)


def test_project_peaks_to_current_view_native_provenance_beats_stale_live_aliases() -> None:
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        current_background_display=lambda: np.zeros((64, 64), dtype=float),
        current_background_native=lambda: np.zeros((64, 64), dtype=float),
        image_size=64,
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 20.0,
            float(row) - 30.0,
        ),
    )

    native_cases = [
        {"native_col": 3.0, "native_row": 4.0},
        {"sim_native_x": 3.0, "sim_native_y": 4.0},
    ]
    for native_fields in native_cases:
        projected = callbacks.project_peaks_to_current_view(
            [
                {
                    "label": "-3,0,4",
                    "q_group_key": ("q_group", "primary", 3, 4),
                    "branch_id": "-x",
                    "source_table_index": 0,
                    "source_row_index": 1,
                    "source_reflection_index": 17,
                    "source_branch_index": 1,
                    "source_ray_id": "minus-ray",
                    "refined_sim_x": 190.0,
                    "refined_sim_y": 96.0,
                    "refined_sim_caked_x": 113.5,
                    "refined_sim_caked_y": -12.5,
                    "display_col": 113.5,
                    "display_row": -12.5,
                    "sim_col": 113.5,
                    "sim_row": -12.5,
                    **native_fields,
                }
            ]
        )

        assert len(projected) == 1
        projected_entry = projected[0]
        assert projected_entry["native_col"] == 3.0
        assert projected_entry["native_row"] == 4.0
        assert projected_entry["sim_native_x"] == 3.0
        assert projected_entry["sim_native_y"] == 4.0
        assert projected_entry["caked_x"] == 23.0
        assert projected_entry["caked_y"] == -26.0
        assert projected_entry["two_theta_deg"] == 23.0
        assert projected_entry["phi_deg"] == -26.0
        assert projected_entry["display_col"] == 23.0
        assert projected_entry["display_row"] == -26.0
        assert projected_entry["sim_col_raw"] == 103.0
        assert projected_entry["sim_row_raw"] == 204.0


def test_project_peaks_to_current_view_detector_replay_keeps_reverse_lut_detector_display(
    monkeypatch,
) -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta_deg, phi_deg, **_kwargs):
        reverse_calls.append((float(two_theta_deg), float(phi_deg)))
        return float(two_theta_deg) - 20.0, float(phi_deg) + 30.0

    monkeypatch.setattr(mg, "caked_angles_to_background_display_coords", _reverse_lut)

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        current_background_display=lambda: np.zeros((256, 256), dtype=float),
        current_background_native=lambda: np.zeros((256, 256), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 1000.0,
            float(row) + 1000.0,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 20.0,
            float(row) - 30.0,
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "-3,0,4",
                "q_group_key": ("q_group", "primary", 3, 4),
                "branch_id": "-x",
                "source_table_index": 0,
                "source_row_index": 1,
                "source_reflection_index": 17,
                "source_branch_index": 1,
                "source_ray_id": "minus-ray",
                "coordinate_frame": "simulation_native",
                "native_col": 3.0,
                "native_row": 4.0,
                "refined_sim_x": 901.0,
                "refined_sim_y": 902.0,
                "display_col": 903.0,
                "display_row": 904.0,
                "sim_col": 905.0,
                "sim_row": 906.0,
            }
        ]
    )

    assert reverse_calls == [(23.0, -26.0)]
    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["sim_detector_anchor_x"] == 3.0
    assert projected_entry["sim_detector_anchor_y"] == 4.0
    assert projected_entry["sim_detector_display_col"] == 3.0
    assert projected_entry["sim_detector_display_row"] == 4.0
    assert projected_entry["display_col"] == 3.0
    assert projected_entry["display_row"] == 4.0
    assert projected_entry["sim_col_raw"] == 3.0
    assert projected_entry["sim_row_raw"] == 4.0
    assert projected_entry["display_col"] != 1003.0
    assert projected_entry["display_row"] != 1004.0


def test_project_peaks_to_current_view_detector_replay_recomputes_stale_display_cache_from_anchor(
    monkeypatch,
) -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta_deg, phi_deg, **_kwargs):
        reverse_calls.append((float(two_theta_deg), float(phi_deg)))
        return float(two_theta_deg) - 20.0, float(phi_deg) + 30.0

    monkeypatch.setattr(mg, "caked_angles_to_background_display_coords", _reverse_lut)

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([-30.0, -26.0, 0.0, 23.0], dtype=float),
        current_background_display=lambda: np.zeros((256, 256), dtype=float),
        current_background_native=lambda: np.zeros((256, 256), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 1000.0,
            float(row) + 1000.0,
        ),
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 20.0,
            float(row) - 30.0,
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "label": "-3,0,4",
                "q_group_key": ("q_group", "primary", 3, 4),
                "branch_id": "-x",
                "source_table_index": 0,
                "source_row_index": 1,
                "source_reflection_index": 17,
                "source_branch_index": 1,
                "source_ray_id": "minus-ray",
                "coordinate_frame": "simulation_native",
                "refined_sim_native_x": 3.0,
                "refined_sim_native_y": 4.0,
                "sim_detector_anchor_x": 3.0,
                "sim_detector_anchor_y": 4.0,
                "sim_detector_display_col": 903.0,
                "sim_detector_display_row": 904.0,
                "sim_detector_frame_provenance": "stale-display-cache",
            }
        ]
    )

    assert reverse_calls == []
    assert len(projected) == 1
    projected_entry = projected[0]
    assert projected_entry["sim_detector_anchor_x"] == 3.0
    assert projected_entry["sim_detector_anchor_y"] == 4.0
    assert projected_entry["sim_detector_display_col"] == 3.0
    assert projected_entry["sim_detector_display_row"] == 4.0
    assert projected_entry["display_col"] == 3.0
    assert projected_entry["display_row"] == 4.0
    assert projected_entry["sim_col_raw"] == 3.0
    assert projected_entry["sim_row_raw"] == 4.0


def test_geometry_manual_session_initial_pairs_display_uses_projected_caked_live_row() -> None:
    stale_live_row = {
        "label": "-3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_reflection_index": 17,
        "source_branch_index": 1,
        "source_ray_id": "minus-ray",
        "native_col": 3.0,
        "native_row": 4.0,
        "refined_sim_caked_x": 113.5,
        "refined_sim_caked_y": -12.5,
    }

    displayed = mg.geometry_manual_session_initial_pairs_display(
        {
            "group_key": ("q_group", "primary", 3, 4),
            "group_entries": [dict(stale_live_row)],
            "pending_entries": [],
            "background_index": 0,
        },
        current_background_index=0,
        use_caked_display=True,
        project_peaks_to_current_view=lambda entries: [
            {
                **dict(entry),
                "caked_x": 23.0,
                "caked_y": -26.0,
                "raw_caked_x": 23.0,
                "raw_caked_y": -26.0,
                "two_theta_deg": 23.0,
                "phi_deg": -26.0,
                "display_col": 23.0,
                "display_row": -26.0,
                "display_frame": "caked_display",
            }
            for entry in entries or ()
            if isinstance(entry, dict)
        ],
        entry_display_coords=lambda _entry: None,
    )

    assert len(displayed) == 1
    assert displayed[0]["sim_display"] == (23.0, -26.0)


def test_manual_qr_selection_works_detector_then_caked_after_view_change() -> None:
    use_caked = {"value": False}
    group_key = ("q_group", "primary", 3, 4)
    raw_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "mosaic_weight": 0.8,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "mosaic_weight": 0.9,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)

    detector_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    detector_by_branch = {row["branch_id"]: row for row in detector_rows}
    assert len(detector_rows) == 2
    assert detector_by_branch["+x"]["display_col"] == 102.0
    assert detector_by_branch["+x"]["display_row"] == 204.0

    detector_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(detector_rows),
        102.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )
    assert detector_session["target_count"] == 2
    assert len(detector_session["group_entries"]) == 2
    assert detector_session["tagged_candidate"]["branch_id"] == "+x"
    assert detector_session["tagged_candidate"]["source_reflection_index"] == 16
    assert detector_session["tagged_candidate"]["source_ray_id"] == "plus-ray"

    use_caked["value"] = True
    caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    caked_by_branch = {row["branch_id"]: row for row in caked_rows}
    assert len(caked_rows) == 2
    assert caked_by_branch["+x"]["display_col"] == 12.0
    assert caked_by_branch["+x"]["display_row"] == 24.0
    assert caked_by_branch["+x"]["sim_col_raw"] == 102.0
    assert caked_by_branch["+x"]["sim_row_raw"] == 204.0

    caked_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(caked_rows),
        12.0,
        24.0,
        use_caked_space=True,
        group_key=group_key,
    )
    assert caked_session["target_count"] == 2
    assert len(caked_session["group_entries"]) == 2
    assert caked_session["tagged_candidate"]["branch_id"] == "+x"
    assert caked_session["tagged_candidate"]["source_reflection_index"] == 16
    assert caked_session["tagged_candidate"]["source_ray_id"] == "plus-ray"


def test_manual_qr_selection_works_caked_then_detector_with_stale_caked_cache() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    raw_rows = [
        {
            "label": "3,0,4",
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 16,
            "source_ray_id": "plus-ray",
            "hkl": (3, 0, 4),
            "mosaic_weight": 0.8,
            "native_col": 2.0,
            "native_row": 4.0,
        },
        {
            "label": "-3,0,4",
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_branch_index": 1,
            "source_reflection_index": 17,
            "source_ray_id": "minus-ray",
            "hkl": (-3, 0, 4),
            "mosaic_weight": 0.9,
            "native_col": 3.0,
            "native_row": 4.0,
        },
    ]
    callbacks = _cross_view_selection_callbacks(use_caked, raw_rows)

    caked_rows = callbacks.simulated_peaks_for_params(prefer_cache=True)
    caked_by_branch = {row["branch_id"]: row for row in caked_rows}
    assert len(caked_rows) == 2
    assert caked_by_branch["-x"]["display_col"] == 13.0
    assert caked_by_branch["-x"]["display_row"] == 24.0
    assert caked_by_branch["-x"]["sim_col_raw"] == 103.0
    assert caked_by_branch["-x"]["sim_row_raw"] == 204.0

    caked_session = _toggle_cross_view_selection(
        callbacks.pick_candidates(caked_rows),
        13.0,
        24.0,
        use_caked_space=True,
        group_key=group_key,
    )
    assert caked_session["target_count"] == 2
    assert len(caked_session["group_entries"]) == 2
    assert caked_session["tagged_candidate"]["branch_id"] == "-x"
    assert caked_session["tagged_candidate"]["source_reflection_index"] == 17
    assert caked_session["tagged_candidate"]["source_ray_id"] == "minus-ray"

    stale_caked_rows = []
    for row in caked_rows:
        stale = dict(row)
        stale["sim_col"] = stale["display_col"]
        stale["sim_row"] = stale["display_row"]
        stale_caked_rows.append(stale)
    stale_grouped = callbacks.pick_candidates(stale_caked_rows)

    group_key_found, group_entries, dist = mg.geometry_manual_choose_group_at(
        stale_grouped,
        103.0,
        204.0,
        window_size_px=50.0,
        use_caked_display=False,
    )
    assert group_key_found == group_key
    assert len(group_entries) == 2
    assert dist < 1.0

    use_caked["value"] = False
    detector_session = _toggle_cross_view_selection(
        stale_grouped,
        103.0,
        204.0,
        use_caked_space=False,
        group_key=group_key,
    )
    assert detector_session["target_count"] == 2
    assert len(detector_session["group_entries"]) == 2
    assert detector_session["tagged_candidate"]["branch_id"] == "-x"
    assert detector_session["tagged_candidate"]["source_reflection_index"] == 17
    assert detector_session["tagged_candidate"]["source_ray_id"] == "minus-ray"


def test_peak_selection_detector_hit_test_prefers_detector_coords_over_stale_caked_display() -> (
    None
):
    group_key = ("q_group", "primary", 3, 4)
    candidate = {
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "hkl": (-3, 0, 4),
        "display_col": 13.0,
        "display_row": 24.0,
        "sim_col": 13.0,
        "sim_row": 24.0,
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
        "two_theta_deg": 13.0,
        "phi_deg": 24.0,
    }
    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )

    _idx, record, dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        103.0,
        204.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=25.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=False,
    )

    assert within is True
    assert dist < 1.0
    assert record is not None
    assert record["q_group_key"] == group_key
    assert record["branch_id"] == "-x"
    assert record["source_branch_index"] == 1
    assert record["source_reflection_index"] == 17
    assert record["source_ray_id"] == "minus-ray"

    _idx, caked_record, caked_dist, caked_within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        13.0,
        24.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=25.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=True,
    )

    assert caked_within is True
    assert caked_dist < 1.0
    assert caked_record is not None
    assert caked_record["branch_id"] == "-x"
    assert caked_record["source_reflection_index"] == 17


def test_peak_selection_detector_hit_test_uses_visible_detector_display_before_raw_provenance() -> (
    None
):
    group_key = ("q_group", "primary", 3, 4)
    candidate = {
        "q_group_key": group_key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 16,
        "source_ray_id": "visible-ray",
        "hkl": (3, 0, 4),
        "display_col": 150.0,
        "display_row": 75.0,
        "display_frame": "detector_display",
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }
    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )

    _idx, record, dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        150.0,
        75.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=10.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=False,
    )

    assert within is True
    assert dist < 1.0
    assert record is not None
    assert record["q_group_key"] == group_key
    assert record["branch_id"] == "+x"
    assert record["source_branch_index"] == 0
    assert record["source_reflection_index"] == 16
    assert record["source_ray_id"] == "visible-ray"


def test_peak_selection_detector_hit_test_honors_detector_display_frame_when_values_match_caked() -> (
    None
):
    group_key = ("q_group", "primary", 3, 4)
    candidate = {
        "q_group_key": group_key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 16,
        "source_ray_id": "same-values-ray",
        "hkl": (3, 0, 4),
        "display_col": 13.0,
        "display_row": 24.0,
        "display_frame": "detector_display",
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }
    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )

    _idx, record, dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        13.0,
        24.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=10.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=False,
    )

    assert within is True
    assert dist < 1.0
    assert record is not None
    assert record["source_ray_id"] == "same-values-ray"


def test_peak_selection_detector_hit_test_uses_raw_detector_even_when_values_match_caked() -> None:
    group_key = ("q_group", "primary", 3, 4)
    candidate = {
        "q_group_key": group_key,
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_reflection_index": 16,
        "source_ray_id": "raw-same-values-ray",
        "hkl": (3, 0, 4),
        "display_col": 999.0,
        "display_row": 999.0,
        "display_frame": "caked_display",
        "sim_col_raw": 13.0,
        "sim_row_raw": 24.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }
    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )

    _idx, record, dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        13.0,
        24.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=10.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=False,
    )

    assert within is True
    assert dist < 1.0
    assert record is not None
    assert record["source_ray_id"] == "raw-same-values-ray"


def test_peak_selection_detector_hit_test_rejects_legacy_caked_alias_coords() -> None:
    group_key = ("q_group", "primary", 3, 4)
    candidate = {
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "legacy-caked-ray",
        "hkl": (-3, 0, 4),
        "x": 13.0,
        "y": 24.0,
        "simulated_x": 13.0,
        "simulated_y": 24.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
        "two_theta_deg": 13.0,
        "phi_deg": 24.0,
    }
    runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_positions_filtered=False,
    )

    _idx, record, _dist, within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        13.0,
        24.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=10.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=False,
    )

    assert within is False
    assert record is None

    _idx, caked_record, caked_dist, caked_within = ps.find_peak_record_for_canvas_click(
        runtime_state,
        13.0,
        24.0,
        ensure_peak_overlay_data=lambda **_kwargs: None,
        max_axis_distance_px=10.0,
        simulation_point_candidates=[dict(candidate)],
        use_caked_display=True,
    )

    assert caked_within is True
    assert caked_dist < 1.0
    assert caked_record is not None
    assert caked_record["source_ray_id"] == "legacy-caked-ray"


def test_caked_manual_seed_returns_to_same_detector_visual_position() -> None:
    use_caked = {"value": True}
    group_key = ("q_group", "primary", 3, 4)
    raw_row = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "branch_id": "-x",
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "winning-ray",
        "hkl": (-3, 0, 4),
        "mosaic_weight": 0.9,
        "native_col": 3.0,
        "native_row": 4.0,
        "source_table_index": 2,
        "source_row_index": 17,
    }

    def _detector_display(col, row):
        return float(col) + 100.0, float(row) + 200.0

    def _caked_to_detector(two_theta, phi):
        return float(two_theta) + 90.0, float(phi) + 180.0

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: use_caked["value"],
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.array([10.0, 20.0], dtype=float),
        last_caked_azimuth_values=lambda: np.array([20.0, 30.0], dtype=float),
        current_background_display=lambda: np.zeros((64, 64), dtype=float),
        current_background_native=lambda: np.zeros((64, 64), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 900.0,
            float(row) + 900.0,
        ),
        native_detector_coords_to_detector_display_coords=_detector_display,
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(raw_row)],
    )
    caked_candidate = callbacks.simulated_peaks_for_params(prefer_cache=True)[0]

    saved_entry_sets: list[list[dict[str, object]]] = []
    handled, next_session = mg.geometry_manual_place_selection_at(
        13.0,
        24.0,
        pick_session={
            "group_key": group_key,
            "group_entries": [dict(caked_candidate)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate_key": _source_key(caked_candidate),
            "tagged_candidate": dict(caked_candidate),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (13.0, 24.0),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=True,
        caked_angles_to_background_display_coords=_caked_to_detector,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
    )

    assert handled is True
    assert next_session == {}
    saved = dict(saved_entry_sets[-1][0])
    assert saved["x"] == 103.0
    assert saved["y"] == 204.0
    assert saved["detector_x"] == 3.0
    assert saved["detector_y"] == 4.0
    assert saved["refined_sim_x"] == 103.0
    assert saved["refined_sim_y"] == 204.0
    assert saved["refined_sim_native_x"] == 3.0
    assert saved["refined_sim_native_y"] == 4.0
    assert "refined_sim_caked_x" not in saved
    assert "refined_sim_caked_y" not in saved
    assert saved["branch_id"] == "-x"
    assert saved["source_branch_index"] == 1
    assert saved["source_reflection_index"] == 17
    assert saved["source_ray_id"] == "winning-ray"

    stale_detector_saved = {
        **dict(saved),
        "x": 999.0,
        "y": 999.0,
        "display_col": 999.0,
        "display_row": 999.0,
        "detector_x": 30.0,
        "detector_y": 40.0,
        "native_col": 30.0,
        "native_row": 40.0,
        "sim_native_x": 30.0,
        "sim_native_y": 40.0,
    }
    refreshed = mg.refresh_geometry_manual_pair_entry(
        stale_detector_saved,
        background_display_shape=(64, 64),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 100.0,
            float(row) - 200.0,
        ),
        caked_angles_to_background_display_coords=_caked_to_detector,
        native_detector_coords_to_detector_display_coords=_detector_display,
        rotate_point_for_display=lambda *_args: (999.0, 999.0),
    )
    assert refreshed is not None
    assert refreshed["detector_x"] == 3.0
    assert refreshed["detector_y"] == 4.0
    assert refreshed["x"] == 103.0
    assert refreshed["y"] == 204.0
    assert refreshed["sim_col"] == 103.0
    assert refreshed["sim_row"] == 204.0

    displayed_caked = mg.geometry_manual_session_initial_pairs_display(
        {
            "group_key": group_key,
            "group_entries": [dict(caked_candidate)],
            "pending_entries": [dict(saved)],
            "target_count": 1,
            "background_index": 0,
        },
        current_background_index=0,
        use_caked_display=True,
        refresh_entry_geometry=callbacks.refresh_entry_geometry,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )
    assert len(displayed_caked) == 1
    assert displayed_caked[0]["sim_display"] == (13.0, 24.0)
    assert displayed_caked[0]["bg_display"] == (13.0, 24.0)

    use_caked["value"] = False
    displayed = mg.geometry_manual_session_initial_pairs_display(
        {
            "group_key": group_key,
            "group_entries": [dict(caked_candidate)],
            "pending_entries": [dict(saved)],
            "target_count": 1,
            "background_index": 0,
        },
        current_background_index=0,
        use_caked_display=False,
        refresh_entry_geometry=callbacks.refresh_entry_geometry,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )
    assert len(displayed) == 1
    assert displayed[0]["sim_display"] == (103.0, 204.0)
    assert displayed[0]["bg_display"] == (103.0, 204.0)


def test_fresh_emitted_pair_redraws_consistently_without_fit() -> None:
    sibling_candidate = {
        "label": "1,0,5",
        "hkl": (1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 9,
        "source_row_index": 0,
        "native_col": 181.0,
        "native_row": 95.0,
        "sim_col": 181.0,
        "sim_row": 95.0,
        "caked_x": 29.0,
        "caked_y": -58.5,
    }
    candidate = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": 9,
        "source_row_index": 0,
        "native_col": 188.0,
        "native_row": 94.0,
        "sim_col": 188.0,
        "sim_row": 94.0,
        "caked_x": 29.75,
        "caked_y": -57.8,
        "mosaic_weight": 0.9,
    }
    saved_entry_sets: list[list[dict[str, object]]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        181.0,
        137.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 5),
            "group_entries": [dict(sibling_candidate), dict(candidate)],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate_key": _source_key(candidate),
            "tagged_candidate": dict(candidate),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (182.0, 138.0),
        set_pairs_for_index_fn=lambda _idx, entries: (
            saved_entry_sets.append(list(entries or [])) or list(entries or [])
        ),
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=lambda _text: None,
        push_undo_state_fn=lambda: None,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        refine_saved_pair_entry_fn=lambda entry, candidate=None: {
            **dict(entry),
            "caked_x": 29.5,
            "caked_y": -58.0,
            "refined_sim_x": 190.0,
            "refined_sim_y": 96.0,
            "refined_sim_caked_x": 30.25,
            "refined_sim_caked_y": -57.5,
        },
    )

    assert handled is True
    assert next_session == {}
    emitted_pair = dict(saved_entry_sets[-1][0])
    assert emitted_pair["refined_sim_x"] == 190.0
    assert emitted_pair["refined_sim_y"] == 96.0
    assert emitted_pair["refined_sim_caked_x"] == 30.25
    assert emitted_pair["refined_sim_caked_y"] == -57.5

    def _project_detector_sources_to_caked(rows):
        projected = []
        for row in rows or ():
            entry = dict(row)
            branch_index = int(entry.get("source_branch_index", -1))
            if branch_index == 0:
                caked_point = (29.0, -58.5)
            elif branch_index == 1:
                caked_point = (29.75, -57.8)
            else:
                continue
            projected.append(
                {
                    **entry,
                    "sim_col_raw": float(entry["native_col"]),
                    "sim_row_raw": float(entry["native_row"]),
                    "display_col": caked_point[0],
                    "display_row": caked_point[1],
                    "caked_x": caked_point[0],
                    "caked_y": caked_point[1],
                    "two_theta_deg": caked_point[0],
                    "phi_deg": caked_point[1],
                }
            )
        return projected

    detector_measured, detector_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(emitted_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        source_rows_for_background=lambda *_args, **_kwargs: [
            dict(sibling_candidate),
            dict(candidate),
        ],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )
    caked_measured, caked_pairs = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(emitted_pair)],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {"simulated_lookup": {}},
        source_rows_for_background=lambda *_args, **_kwargs: [
            dict(sibling_candidate),
            dict(candidate),
        ],
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=_project_detector_sources_to_caked,
        entry_display_coords=lambda entry: (
            float(entry["caked_x"]),
            float(entry["caked_y"]),
        ),
    )

    for measured_display in (detector_measured, caked_measured):
        assert measured_display[0]["source_reflection_index"] == 203
        assert measured_display[0]["source_reflection_namespace"] == "full_reflection"
        assert measured_display[0]["source_reflection_is_full"] is True
        assert measured_display[0]["source_branch_index"] == 1
        assert measured_display[0]["source_peak_index"] == 1

    assert detector_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (182.0, 138.0),
            "sim_display": (190.0, 96.0),
        }
    ]
    assert caked_pairs == [
        {
            "overlay_match_index": 0,
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "bg_display": (29.5, -58.0),
            "sim_display": (29.75, -57.8),
        }
    ]

    peak_records = [
        {
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(record) for record in peak_records],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }

    updated = mg.update_geometry_manual_peak_record_cache(
        peak_records,
        source_key=_source_key(emitted_pair),
        source_entry=emitted_pair,
        refined_display=(191.0, 97.0),
        peak_positions=peak_positions,
        peak_overlay_cache=peak_overlay_cache,
    )

    assert updated is True
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 191.0
    assert peak_records[1]["display_row"] == 97.0
    assert peak_positions == [(181.0, 95.0), (191.0, 97.0)]


def test_runtime_projection_callbacks_report_rebuild_disabled_diagnostics() -> None:
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([]),
        last_caked_azimuth_values=lambda: np.array([]),
        current_background_display=lambda: np.zeros((4, 4), dtype=float),
        current_background_native=lambda: np.zeros((4, 4), dtype=float),
        current_geometry_fit_params=lambda: {
            "a": 4.1,
            "c": 28.6,
            "lambda": 1.5406,
            "corto_detector": 0.075,
            "gamma": 0.999,
            "Gamma": -3.579,
            "chi": 0.0,
            "psi": 0.0,
            "psi_z": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "n2": 1.0,
            "debye_x": 0.0,
            "debye_y": 0.0,
            "center": [1024.0, 1024.0],
            "theta_initial": 5.0,
            "mosaic_params": {
                "beam_x_array": np.array([0.0]),
                "beam_y_array": np.array([0.0]),
                "theta_array": np.array([0.0]),
                "phi_array": np.array([0.0]),
                "wavelength_array": np.array([1.5406]),
            },
        },
        miller=lambda: np.array([[0.0, 0.0, 3.0]], dtype=float),
        intensities=lambda: np.array([1.0], dtype=float),
        image_size=2048,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
    )

    assert callbacks.simulated_peaks_for_params(prefer_cache=False) == []
    diagnostics = callbacks.last_simulation_diagnostics()

    assert diagnostics["source"] == "fresh"
    assert diagnostics["requested_prefer_cache"] is False
    assert diagnostics["status"] == "simulated_peak_rebuild_disabled"


def test_runtime_projection_callbacks_collapse_raw_cache_qr_qz_rows_before_view_change() -> None:
    group_key = ("q_group", "primary", 5, 2)
    raw_rows = [
        {
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 50,
            "source_reflection_key": ("full", 50),
            "source_ray_id": "low-ray",
            "hkl": (5, 0, 2),
            "mosaic_weight": 0.1,
            "native_col": 10.0,
            "native_row": 20.0,
        },
        {
            "q_group_key": group_key,
            "branch_id": "-x",
            "source_branch_index": 1,
            "source_reflection_index": 51,
            "source_reflection_key": ("full", 51),
            "source_ray_id": "top-ray",
            "hkl": (-5, 0, 2),
            "mosaic_weight": 0.9,
            "native_col": 11.0,
            "native_row": 21.0,
        },
        {
            "q_group_key": group_key,
            "branch_id": "+x",
            "source_branch_index": 0,
            "source_reflection_index": 52,
            "source_reflection_key": ("full", 52),
            "source_ray_id": "mid-ray",
            "hkl": (5, 0, 2),
            "mosaic_weight": 0.4,
            "native_col": 12.0,
            "native_row": 22.0,
        },
    ]

    def _old_collapse_stub(entries, *, merge_radius_px):
        return (list(entries or []), int(merge_radius_px))

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([]),
        last_caked_azimuth_values=lambda: np.array([]),
        current_background_display=lambda: np.zeros((64, 64), dtype=float),
        current_background_native=lambda: np.zeros((64, 64), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(row) for row in raw_rows],
        collapse_simulated_peaks=_old_collapse_stub,
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)
    grouped = callbacks.pick_candidates(projected)

    assert len([row for row in projected if row.get("q_group_key") == group_key]) == 2
    assert len(grouped[group_key]) == 2
    by_branch = {row["branch_id"]: row for row in grouped[group_key]}
    assert by_branch["+x"]["mosaic_weight"] == 0.4
    assert by_branch["+x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["+x"]["source_branch_index"] == 0
    assert by_branch["+x"]["source_reflection_index"] == 52
    assert by_branch["+x"]["source_reflection_key"] == ("full", 52)
    assert by_branch["+x"]["source_ray_id"] == "mid-ray"
    assert by_branch["-x"]["mosaic_weight"] == 0.9
    assert by_branch["-x"]["selection_reason"] == "mosaic_top_per_branch"
    assert by_branch["-x"]["source_branch_index"] == 1
    assert by_branch["-x"]["source_reflection_index"] == 51
    assert by_branch["-x"]["source_reflection_key"] == ("full", 51)
    assert by_branch["-x"]["source_ray_id"] == "top-ray"


def test_runtime_projection_callbacks_collapse_raw_cache_00l_rows_before_first_click() -> None:
    group_key = ("q_group", "primary", 0, 3)
    raw_rows = [
        {
            "q_group_key": group_key,
            "source_branch_index": 0,
            "source_reflection_index": 70,
            "source_reflection_key": ("full", 70),
            "source_ray_id": "00l-low",
            "hkl": (0, 0, 3),
            "mosaic_weight": 0.1,
            "native_col": 10.0,
            "native_row": 20.0,
            "source_table_index": 1,
            "source_row_index": 70,
        },
        {
            "q_group_key": group_key,
            "source_branch_index": 1,
            "source_reflection_index": 71,
            "source_reflection_key": ("full", 71),
            "source_ray_id": "00l-top",
            "hkl": (0, 0, 3),
            "mosaic_weight": 0.9,
            "native_col": 11.0,
            "native_row": 21.0,
            "source_table_index": 1,
            "source_row_index": 71,
        },
        {
            "q_group_key": group_key,
            "source_branch_index": 0,
            "source_reflection_index": 72,
            "source_reflection_key": ("full", 72),
            "source_ray_id": "00l-mid",
            "hkl": (0, 0, 3),
            "mosaic_weight": 0.4,
            "native_col": 12.0,
            "native_row": 22.0,
            "source_table_index": 1,
            "source_row_index": 72,
        },
    ]

    def _old_collapse_stub(entries, *, merge_radius_px):
        return (list(entries or []), int(merge_radius_px))

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([]),
        last_caked_azimuth_values=lambda: np.array([]),
        current_background_display=lambda: np.zeros((64, 64), dtype=float),
        current_background_native=lambda: np.zeros((64, 64), dtype=float),
        image_size=64,
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        build_live_preview_simulated_peaks_from_cache=lambda: [dict(row) for row in raw_rows],
        collapse_simulated_peaks=_old_collapse_stub,
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)
    grouped = callbacks.pick_candidates(projected)

    visible_for_group = [row for row in projected if row.get("q_group_key") == group_key]
    assert len(visible_for_group) == 1
    assert len(grouped[group_key]) == 1
    kept = grouped[group_key][0]
    assert kept["branch_id"] == "00l"
    assert kept["selection_reason"] == "mosaic_top_per_branch"
    assert kept["source_branch_index"] == 1
    assert kept["source_reflection_index"] == 71
    assert kept["source_reflection_key"] == ("full", 71)
    assert kept["source_ray_id"] == "00l-top"
    assert kept["mosaic_weight"] == 0.9

    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        11.0,
        21.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((64, 64), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("raw-cache",),
            "grouped_candidates": grouped,
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["target_count"] == 1
    assert len(next_session["group_entries"]) == 1
    assert next_session["group_entries"][0]["branch_id"] == "00l"
    assert next_session["group_entries"][0]["source_reflection_index"] == 71
    assert set_sessions[-1]["target_count"] == 1
    assert "Click background peak 1 of 1" in status_messages[-1]
    displayed = mg.geometry_manual_session_initial_pairs_display(
        next_session,
        current_background_index=0,
        entry_display_coords=lambda entry: (
            float(entry.get("sim_col")),
            float(entry.get("sim_row")),
        ),
    )
    assert len([entry for entry in displayed if entry.get("q_group_key") == group_key]) == 1
