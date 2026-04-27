from collections.abc import Mapping, Sequence
import contextlib
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ra_sim.fitting.background_peak_matching import build_background_peak_context
from ra_sim import headless_geometry_fit as hgf
from ra_sim.gui import geometry_fit as gf
from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui import geometry_overlay
from ra_sim.gui import manual_geometry as mg
from ra_sim.gui import mosaic_top_selection
from ra_sim.gui import peak_selection as ps
from ra_sim.io.data_loading import load_gui_state_file
from ra_sim.simulation import diffraction, exact_cake_portable


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


def test_geometry_manual_choose_group_at_finds_qr_group_from_representative_cache_in_caked_view() -> (
    None
):
    representative_cache = diffraction.build_branch_representative_intersection_cache(
        [
            np.array(
                [[1.0, 10.0, 20.0, -0.2, 1.0, 0.0, 2.0, 7.0, 3.0, 11.0]],
                dtype=np.float64,
            )
        ],
        4.0,
        7.0,
    )
    representative_hit_tables = diffraction.intersection_cache_to_hit_tables(representative_cache)
    simulated_rows = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        representative_hit_tables,
        image_shape=(128, 128),
        native_sim_to_display_coords=lambda col, row, _shape: (float(col), float(row)),
        primary_a=4.0,
        primary_c=7.0,
    )

    projected_rows, grouped_candidates, projection_lookup = (
        mg._geometry_manual_build_caked_qr_projection_cache(
            simulated_rows,
            lambda rows: [
                dict(
                    entry,
                    display_col=13.0,
                    display_row=2.0,
                    sim_col_raw=13.0,
                    sim_row_raw=2.0,
                    caked_x=13.0,
                    caked_y=2.0,
                )
                for entry in (rows or ())
            ],
            _group_candidates,
            _build_lookup,
            None,
        )
    )

    assert len(projected_rows) == 1
    assert projection_lookup
    found_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        grouped_candidates,
        13.0,
        2.0,
        window_size_px=5.0,
        use_caked_display=True,
    )

    assert found_key == projected_rows[0]["q_group_key"]
    assert best_dist < 1.0
    assert len(entries) == 1
    assert entries[0]["source_table_index"] == 7
    assert entries[0]["source_row_index"] == 3
    assert entries[0]["best_sample_index"] == 11
    assert entries[0]["source_branch_index"] == 0


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


def test_refresh_geometry_manual_pair_entry_sim_replay_ignores_saved_background_angles_and_poisoned_aliases() -> None:
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
        "background_two_theta_deg": -7.0,
        "background_phi_deg": 8.0,
        "background_detector_x": 1001.0,
        "background_detector_y": 1002.0,
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


def test_refresh_geometry_manual_pair_entry_sim_replay_drops_stale_saved_display_aliases_without_current_detector_display_projection() -> None:
    saved_entry = {
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "native_col": 3.0,
        "native_row": 4.0,
        "sim_native_x": 3.0,
        "sim_native_y": 4.0,
        "sim_detector_anchor_x": 3.0,
        "sim_detector_anchor_y": 4.0,
        "x": 901.0,
        "y": 902.0,
        "display_col": 903.0,
        "display_row": 904.0,
        "sim_col_raw": 905.0,
        "sim_row_raw": 906.0,
        "sim_col": 907.0,
        "sim_row": 908.0,
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
        background_display_shape=(),
        background_display_to_native_detector_coords=None,
        caked_angles_to_background_display_coords=None,
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 20.0,
            float(row) - 30.0,
        ),
        native_detector_coords_to_detector_display_coords=None,
        current_projected_sim_entry=projected_sim_entry,
    )

    assert refreshed is not None
    assert refreshed["detector_x"] == 3.0
    assert refreshed["detector_y"] == 4.0
    assert refreshed["sim_detector_anchor_x"] == 3.0
    assert refreshed["sim_detector_anchor_y"] == 4.0
    assert refreshed["sim_detector_frame_provenance"] == "sim_reverse_lut_replay_cache"
    assert "x" not in refreshed
    assert "y" not in refreshed
    assert "display_col" not in refreshed
    assert "display_row" not in refreshed
    assert "sim_col_raw" not in refreshed
    assert "sim_row_raw" not in refreshed


def test_refresh_geometry_manual_pair_entry_sim_replay_reverse_lut_failure_leaves_unresolved_without_fallback() -> None:
    reverse_calls: list[tuple[float, float]] = []

    def _reverse_lut(two_theta, phi):
        reverse_calls.append((float(two_theta), float(phi)))
        return None

    saved_entry = {
        "label": "3,0,4",
        "q_group_key": ("q_group", "primary", 3, 4),
        "branch_id": "-x",
        "source_table_index": 0,
        "source_row_index": 1,
        "source_branch_index": 1,
        "source_reflection_index": 17,
        "source_ray_id": "minus-ray",
        "background_two_theta_deg": -7.0,
        "background_phi_deg": 8.0,
        "background_detector_x": 1001.0,
        "background_detector_y": 1002.0,
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
        "sim_detector_anchor_x": 917.0,
        "sim_detector_anchor_y": 918.0,
        "sim_detector_display_col": 919.0,
        "sim_detector_display_row": 920.0,
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
    for missing_key in (
        "x",
        "y",
        "display_col",
        "display_row",
        "detector_x",
        "detector_y",
        "refined_sim_x",
        "refined_sim_y",
        "native_col",
        "native_row",
        "refined_sim_native_x",
        "refined_sim_native_y",
        "sim_native_x",
        "sim_native_y",
        "sim_col",
        "sim_row",
        "sim_col_raw",
        "sim_row_raw",
        "sim_detector_anchor_x",
        "sim_detector_anchor_y",
        "sim_detector_display_col",
        "sim_detector_display_row",
        "sim_detector_frame_provenance",
    ):
        assert missing_key not in refreshed
    assert "sim_col" not in refreshed
    assert "sim_row" not in refreshed
    assert "sim_detector_display_col" not in refreshed
    assert "sim_detector_display_row" not in refreshed


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


def test_project_peaks_to_current_view_detector_replay_drops_stale_display_cache_without_detector_display_callback(
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
        simulation_native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 20.0,
            float(row) - 30.0,
        ),
        display_rotate_k=1,
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
    assert "sim_detector_display_col" not in projected_entry
    assert "sim_detector_display_row" not in projected_entry
    assert projected_entry["display_col"] == 251.0
    assert projected_entry["display_row"] == 3.0
    assert projected_entry["sim_col_raw"] == 251.0
    assert projected_entry["sim_row_raw"] == 3.0


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


_QR_PICKER_DIAG_STATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "geometry_fit_gui_states"
    / "new4.json"
)
_QR_PICKER_TARGET_Q_GROUP_KEY = ("q_group", "primary", 1, 10)
_QR_PICKER_TARGET_HKL = (-1, 0, 10)


def _diag_plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_diag_plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return tuple(_diag_plain(item) for item in value)
    return value


def _diag_tuple(value):
    value = _diag_plain(value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, list):
        return tuple(value)
    return None


def _diag_hkl(entry):
    value = _diag_tuple(entry.get("hkl") if isinstance(entry, Mapping) else None)
    if value is None or len(value) < 3:
        return None
    try:
        return tuple(int(np.rint(float(item))) for item in value[:3])
    except Exception:
        return None


def _diag_q_group_key(entry):
    value = _diag_tuple(entry.get("q_group_key") if isinstance(entry, Mapping) else None)
    if value is None:
        return None
    normalized = []
    for item in value:
        try:
            numeric = float(item)
        except Exception:
            normalized.append(item)
            continue
        if np.isfinite(numeric) and abs(numeric - round(numeric)) < 1.0e-9:
            normalized.append(int(round(numeric)))
        else:
            normalized.append(float(numeric))
    return tuple(normalized)


def _diag_finite_pair(entry, key_pairs):
    if not isinstance(entry, Mapping):
        return None
    for x_key, y_key in key_pairs:
        try:
            col = float(entry.get(x_key, np.nan))
            row = float(entry.get(y_key, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    return None


def _diag_shape_hw(image):
    arr = np.asarray(image)
    if arr.ndim < 2:
        return 0, 0
    return int(arr.shape[0]), int(arr.shape[1])


def _diag_inside(point, width, height):
    if point is None:
        return False
    return bool(0.0 <= float(point[0]) < float(width) and 0.0 <= float(point[1]) < float(height))


def _diag_point_text(point, reason="missing"):
    if point is None:
        if isinstance(reason, str) and reason.startswith("<unavailable"):
            return reason
        return f"<unavailable reason={reason}>"
    return f"({float(point[0]):.3f}, {float(point[1]):.3f})"


def _diag_float_text(value):
    try:
        number = float(value)
    except Exception:
        return "<unavailable>"
    if not np.isfinite(number):
        return "<unavailable>"
    return f"{number:.6f}"


def _diag_function_name(fn):
    if fn is None:
        return "<unavailable>"
    return getattr(fn, "__name__", type(fn).__name__)


def _diag_source_identity(entry):
    if not isinstance(entry, Mapping):
        return None
    key = _source_key(dict(entry))
    if key is not None:
        return key
    return (
        _diag_q_group_key(entry),
        _diag_hkl(entry),
        entry.get("source_branch_index"),
        entry.get("source_table_index"),
        entry.get("source_row_index"),
        entry.get("source_peak_index"),
        entry.get("branch_id"),
    )


def _diag_sorted(entries):
    def _sort_key(entry):
        return (
            repr(_diag_q_group_key(entry)),
            repr(_diag_hkl(entry)),
            int(entry.get("source_branch_index", -1) or -1),
            int(entry.get("source_table_index", -1) or -1),
            int(entry.get("source_row_index", -1) or -1),
            int(entry.get("source_peak_index", -1) or -1),
            str(entry.get("branch_id", "")),
        )

    return sorted(entries, key=_sort_key)


def _diag_runtime_value(value):
    return value() if callable(value) else value


def _diag_load_saved_state():
    loaded = load_gui_state_file(_QR_PICKER_DIAG_STATE_PATH)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("state"), Mapping):
        return dict(loaded["state"])
    return dict(loaded)


def test_headless_native_to_display_transform_is_bound_to_requested_background() -> None:
    calls = []
    backgrounds = {
        0: np.zeros((10, 20), dtype=float),
        2: np.zeros((30, 40), dtype=float),
    }

    def load_background(index):
        calls.append(int(index))
        image = backgrounds[int(index)]
        return image, image

    callback = hgf._headless_native_detector_coords_to_detector_display_coords_for_background(
        load_background,
        2,
        display_rotate_k=0,
    )

    assert callable(callback)
    assert calls == [2]
    calls.clear()
    assert callback(3.0, 4.0) == (3.0, 4.0)
    assert calls == []
    assert "bg_2" in getattr(callback, "__name__", "")


def _diag_capture_startup_runtime(tmp_path):
    saved_state = _diag_load_saved_state()
    captured = []
    original_factory = mg.make_runtime_geometry_manual_projection_callbacks

    def _capture_factory(**kwargs):
        callbacks = original_factory(**kwargs)
        captured.append((dict(kwargs), callbacks))
        return callbacks

    runtime_error = None
    mg.make_runtime_geometry_manual_projection_callbacks = _capture_factory
    try:
        try:
            hgf.run_headless_geometry_fit(
                saved_state,
                state_path=_QR_PICKER_DIAG_STATE_PATH,
                downloads_dir=tmp_path,
                stamp="qr_picker_detector_pixel_positions",
            )
        except RuntimeError as exc:
            runtime_error = str(exc)
    finally:
        mg.make_runtime_geometry_manual_projection_callbacks = original_factory

    assert captured, "GUI runtime did not build manual geometry projection callbacks"
    projection_kwargs, projection_callbacks = captured[-1]
    detector_kwargs = dict(projection_kwargs)
    detector_kwargs["caked_view_enabled"] = lambda: False
    detector_callbacks = original_factory(**detector_kwargs)
    return {
        "saved_state": saved_state,
        "projection_kwargs": detector_kwargs,
        "projection_callbacks": detector_callbacks,
        "captured_projection_callbacks": projection_callbacks,
        "runtime_error": runtime_error,
    }


def _diag_native_shape(context):
    native = _diag_runtime_value(context["projection_kwargs"]["current_background_native"])
    return _diag_shape_hw(native)


def _diag_display_shape(context):
    display = _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    return _diag_shape_hw(display)


def _diag_native_to_display(context, point):
    if point is None:
        return None
    fn = context["projection_kwargs"].get("native_detector_coords_to_detector_display_coords")
    if callable(fn):
        result = fn(float(point[0]), float(point[1]))
        if result is not None:
            try:
                return float(result[0]), float(result[1])
            except Exception:
                return None
    native_h, native_w = _diag_native_shape(context)
    rotate_fn = context["projection_kwargs"].get("rotate_point_for_display")
    rotate_k = int(context["projection_kwargs"].get("display_rotate_k", 0))
    if callable(rotate_fn):
        return rotate_fn(float(point[0]), float(point[1]), (native_h, native_w), rotate_k)
    return geometry_overlay.rotate_point_for_display(
        float(point[0]),
        float(point[1]),
        (native_h, native_w),
        rotate_k,
    )


def _diag_display_to_native(context, point):
    if point is None:
        return None
    callbacks = context["projection_callbacks"]
    fn = getattr(callbacks, "background_display_to_native_detector_coords", None)
    if not callable(fn):
        return None
    result = fn(float(point[0]), float(point[1]))
    if result is None:
        return None
    try:
        return float(result[0]), float(result[1])
    except Exception:
        return None


def _diag_manual_entries_for_active_background(saved_state):
    geometry_state = saved_state.get("geometry", {}) if isinstance(saved_state, Mapping) else {}
    manual_pairs = (
        geometry_state.get("manual_pairs", []) if isinstance(geometry_state, Mapping) else []
    )
    for item in manual_pairs or ():
        if not isinstance(item, Mapping):
            continue
        try:
            background_index = int(item.get("background_index"))
        except Exception:
            continue
        if background_index != 0:
            continue
        entries = item.get("entries", [])
        return [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    return []


def _diag_prepare_saved_manual_source_rows(context, profile_cache):
    rows = []
    for raw in _diag_manual_entries_for_active_background(context["saved_state"]):
        entry = dict(raw)
        q_group_key = _diag_q_group_key(entry)
        hkl = _diag_hkl(entry)
        if q_group_key is not None:
            entry["q_group_key"] = q_group_key
        if hkl is not None:
            entry["hkl"] = hkl
            entry["label"] = ",".join(str(item) for item in hkl)
        native = _diag_finite_pair(
            entry,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("sim_native_x", "sim_native_y"),
                ("native_col", "native_row"),
                ("detector_x", "detector_y"),
            ),
        )
        if native is not None:
            display = _diag_native_to_display(context, native)
            entry["native_col"] = float(native[0])
            entry["native_row"] = float(native[1])
            entry["sim_native_x"] = float(native[0])
            entry["sim_native_y"] = float(native[1])
            entry["detector_native_col"] = float(native[0])
            entry["detector_native_row"] = float(native[1])
            if display is not None:
                entry["display_col"] = float(display[0])
                entry["display_row"] = float(display[1])
                entry["sim_col"] = float(display[0])
                entry["sim_row"] = float(display[1])
                entry["sim_col_raw"] = float(display[0])
                entry["sim_row_raw"] = float(display[1])
        entry.pop("refined_sim_x", None)
        entry.pop("refined_sim_y", None)
        entry["coordinate_frame"] = "simulation_native"
        entry["diagnostic_source"] = "manual_saved_pair"
        entry["included_in_manual_source_rows"] = True
        entry.setdefault("best_sample_index", entry.get("source_row_index"))
        entry["mosaic_weight"] = float(entry.get("mosaic_weight", 1.0))
        branch_id, branch_source = mosaic_top_selection.normalize_branch_id(
            entry,
            target_key=q_group_key,
            profile_cache=profile_cache,
        )
        entry["branch_id"] = str(branch_id)
        entry["branch_source"] = str(branch_source)
        rows.append(
            mosaic_top_selection.annotate_selection_metadata(
                entry,
                target_key=q_group_key,
                profile_cache=profile_cache,
            )
        )
    return rows


def _diag_fresh_source_rows(context, profile_cache):
    kwargs = context["projection_kwargs"]
    params = dict(_diag_runtime_value(kwargs["current_geometry_fit_params"]))
    miller = np.asarray(_diag_runtime_value(kwargs["miller"]), dtype=np.float64)
    intensities = np.asarray(_diag_runtime_value(kwargs["intensities"]), dtype=np.float64)
    image_size = int(_diag_runtime_value(kwargs["image_size"]))
    mosaic = dict(params.get("mosaic_params", {}) or {})
    hit_tables = geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        miller,
        intensities,
        image_size,
        params,
        build_geometry_fit_central_mosaic_params=None,
        process_peaks_parallel=diffraction.process_peaks_parallel,
        default_solve_q_steps=int(mosaic.get("solve_q_steps", params.get("solve_q_steps", 1000))),
        default_solve_q_rel_tol=float(
            mosaic.get(
                "solve_q_rel_tol",
                params.get("solve_q_rel_tol", diffraction.DEFAULT_SOLVE_Q_REL_TOL),
            )
        ),
        default_solve_q_mode=int(
            mosaic.get("solve_q_mode", params.get("solve_q_mode", diffraction.DEFAULT_SOLVE_Q_MODE))
        ),
    )
    source_reflection_indices = (
        geometry_q_group_manager.audited_full_order_source_reflection_indices(
            hit_tables,
            owner="test_qr_picker_detector_pixel_positions",
        )
    )
    rows = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        hit_tables,
        image_shape=(image_size, image_size),
        native_sim_to_display_coords=lambda col, row, shape: (
            geometry_overlay.native_sim_to_display_coords(
                col,
                row,
                shape,
                sim_display_rotate_k=hgf.SIM_DISPLAY_ROTATE_K,
            )
        ),
        peak_table_lattice=[
            (params.get("a", np.nan), params.get("c", np.nan), "primary")
            for _ in hit_tables
        ],
        source_reflection_indices=source_reflection_indices,
        primary_a=params.get("a", np.nan),
        primary_c=params.get("c", np.nan),
        default_source_label="primary",
        round_pixel_centers=True,
        allow_nominal_hkl_indices=True,
        profile_cache=profile_cache,
    )
    output = []
    for row in rows:
        entry = dict(row)
        entry["diagnostic_source"] = "fresh_full_order_source_rows"
        entry["included_in_manual_source_rows"] = True
        q_group_key = _diag_q_group_key(entry)
        if q_group_key is not None:
            entry["q_group_key"] = q_group_key
        hkl = _diag_hkl(entry)
        if hkl is not None:
            entry["hkl"] = hkl
        output.append(entry)
    return output


def _diag_project_detector_entry(context, raw_entry):
    entry = dict(raw_entry)
    native = _diag_finite_pair(
        entry,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("sim_native_x", "sim_native_y"),
            ("native_col", "native_row"),
            ("detector_x", "detector_y"),
            ("background_detector_x", "background_detector_y"),
        ),
    )
    if native is None:
        display = _diag_finite_pair(
            entry,
            (
                ("display_col", "display_row"),
                ("sim_col_raw", "sim_row_raw"),
                ("sim_col", "sim_row"),
                ("x", "y"),
            ),
        )
        native = _diag_display_to_native(context, display)
    display = _diag_native_to_display(context, native) if native is not None else None
    if native is not None:
        entry["native_col"] = float(native[0])
        entry["native_row"] = float(native[1])
        entry["sim_native_x"] = float(native[0])
        entry["sim_native_y"] = float(native[1])
        entry["detector_native_col"] = float(native[0])
        entry["detector_native_row"] = float(native[1])
    if display is not None:
        entry["display_col"] = float(display[0])
        entry["display_row"] = float(display[1])
        entry["sim_col"] = float(display[0])
        entry["sim_row"] = float(display[1])
        entry["sim_col_raw"] = float(display[0])
        entry["sim_row_raw"] = float(display[1])
        entry["display_frame"] = "detector_display"
        entry["detector_display_source"] = "diagnostic_native_to_display"
    entry.pop("refined_sim_x", None)
    entry.pop("refined_sim_y", None)
    return entry


def _diag_group_and_project_rows(context):
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    profile_cache = dict(params.get("mosaic_params", {}) or {})
    manual_rows = _diag_prepare_saved_manual_source_rows(context, profile_cache)
    fresh_rows = _diag_fresh_source_rows(context, profile_cache)
    combined = [*manual_rows, *fresh_rows]
    collapsed = mg._geometry_manual_collapse_q_group_representatives(
        combined,
        profile_cache=profile_cache,
    )
    projected = [
        _diag_project_detector_entry(context, entry)
        for entry in collapsed
        if isinstance(entry, Mapping)
    ]
    grouped = context["projection_callbacks"].pick_candidates(projected)
    picker_rows = []
    for key, entries in grouped.items():
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if isinstance(entry, Mapping):
                item = dict(entry)
                item["q_group_key"] = _diag_q_group_key({"q_group_key": key}) or key
                picker_rows.append(item)
    return {
        "profile_cache": profile_cache,
        "manual_rows": manual_rows,
        "fresh_rows": fresh_rows,
        "combined_source_rows": combined,
        "collapsed_source_rows": collapsed,
        "overlay_rows": [dict(entry) for entry in projected if isinstance(entry, Mapping)],
        "picker_rows": picker_rows,
    }


_QR_PICKER_STARTUP_CACHE = {}


def _diag_startup_context_and_rows(tmp_path):
    if "payload" not in _QR_PICKER_STARTUP_CACHE:
        context = _diag_capture_startup_runtime(tmp_path)
        rows = _diag_group_and_project_rows(context)
        _QR_PICKER_STARTUP_CACHE["payload"] = (context, rows)
    return _QR_PICKER_STARTUP_CACHE["payload"]


def _diag_detector_picker_cache(source_rows, *, overlay_grouped=None):
    source_rows = [dict(entry) for entry in source_rows if isinstance(entry, Mapping)]
    if overlay_grouped is None:
        overlay_grouped = {}
    return {
        "signature": ("detector-picker-diagnostic",),
        "simulated_peaks": [dict(entry) for entry in source_rows],
        "active_simulated_peaks": [dict(entry) for entry in source_rows],
        "detector_picker_source_rows": [dict(entry) for entry in source_rows],
        "grouped_candidates": dict(overlay_grouped),
        "caked_qr_projection_grouped_candidates": {},
        "cache_metadata": {
            "cache_source": "startup_sim_ready_detector_rows",
            "simulated_peak_count": len(source_rows),
        },
    }


def _diag_flatten_grouped(grouped):
    rows = []
    for key, entries in (grouped or {}).items():
        for entry in entries or ():
            if isinstance(entry, Mapping):
                item = dict(entry)
                item.setdefault("q_group_key", key)
                rows.append(item)
    return rows


def _diag_target_rows_by_branch(grouped):
    target_rows = [
        row
        for row in _diag_flatten_grouped(grouped)
        if _diag_q_group_key(row) == _QR_PICKER_TARGET_Q_GROUP_KEY
        and _diag_hkl(row) == _QR_PICKER_TARGET_HKL
    ]
    return {
        int(row.get("source_branch_index")): row
        for row in target_rows
        if row.get("source_branch_index") is not None
    }


def _diag_toggle_detector_click(cache_data, click_px, profile_cache, trace_output=None):
    statuses = []
    sessions = []
    handled, session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        float(click_px[0]),
        float(click_px[1]),
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((3000, 3000), dtype=float),
        get_cache_data=lambda **_kwargs: dict(cache_data),
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda value: sessions.append(dict(value)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=statuses.append,
        listed_q_group_entries=lambda: [{"key": _QR_PICKER_TARGET_Q_GROUP_KEY}],
        format_q_group_line=lambda _entry: "target q_group",
        use_caked_space=False,
        pick_search_window_px=50.0,
        profile_cache=profile_cache,
        trace_picker_input_fn=(
            (lambda trace: trace_output.append(dict(trace)))
            if isinstance(trace_output, list)
            else None
        ),
    )
    assert handled is True
    assert suppress_drag is True
    assert sessions
    assert session == sessions[-1]
    assert not any("No simulated Qr/Qz groups are available" in text for text in statuses)
    return session, statuses


def _diag_detector_display_point(entry):
    return mg._geometry_manual_entry_detector_display_point(entry)


def _diag_detector_native_point(entry):
    return mg._geometry_manual_entry_native_point(entry)


def _diag_detector_point_is_caked(entry, point):
    if point is None:
        return False
    for key_pair in (
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
        ("two_theta_deg", "phi_deg"),
        ("refined_sim_caked_x", "refined_sim_caked_y"),
        ("background_two_theta_deg", "background_phi_deg"),
    ):
        caked = _diag_finite_pair(entry, (key_pair,))
        if caked is not None and np.allclose(point, caked, atol=1.0e-6, rtol=0.0):
            return True
    return False


def _diag_source_branch(entry):
    if not isinstance(entry, Mapping):
        return None
    for key in ("source_branch_index", "branch_index"):
        if key in entry:
            try:
                return int(entry.get(key))
            except Exception:
                return None
    return None


def _diag_branch_map(entries):
    mapped = {}
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        branch = _diag_source_branch(entry)
        if branch is not None:
            mapped[int(branch)] = dict(entry)
    return mapped


def _diag_pair_source(entry, key_pairs, source):
    pair = _diag_finite_pair(entry, key_pairs)
    return pair, source if pair is not None else "<unavailable>"


def _diag_observed_detector_display(entry):
    point = mg._geometry_manual_entry_detector_display_point(entry)
    if point is not None:
        return point, "entry_detector_display"
    return None, "<unavailable reason=no detector back-projection>"


def _diag_observed_detector_native(entry):
    return _diag_pair_source(
        entry,
        (
            ("refined_detector_native_col", "refined_detector_native_row"),
            ("background_detector_x", "background_detector_y"),
            ("detector_native_col", "detector_native_row"),
            ("native_col", "native_row"),
            ("detector_x", "detector_y"),
        ),
        "entry_detector_native",
    )


def _diag_observed_caked(entry):
    return _diag_pair_source(
        entry,
        (
            ("refined_background_two_theta_deg", "refined_background_phi_deg"),
            ("background_two_theta_deg", "background_phi_deg"),
            ("raw_caked_x", "raw_caked_y"),
            ("caked_x", "caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
        "entry_caked_deg",
    )


def _diag_sim_visual_detector_display(entry):
    tuple_point = mg._geometry_manual_tuple_point(
        entry,
        "sim_visual_detector_display_px",
    )
    if tuple_point is not None:
        return tuple_point, "entry_sim_visual_detector_display_px"
    return _diag_pair_source(
        entry,
        (
            ("refined_sim_x", "refined_sim_y"),
            ("sim_col_raw", "sim_row_raw"),
            ("sim_col", "sim_row"),
            ("display_col", "display_row"),
        ),
        "entry_visual_detector_display",
    )


def _diag_sim_visual_detector_native(entry):
    tuple_point = mg._geometry_manual_tuple_point(
        entry,
        "sim_visual_detector_native_px",
    )
    if tuple_point is not None:
        return tuple_point, "entry_sim_visual_detector_native_px"
    return _diag_pair_source(
        entry,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("sim_native_x", "sim_native_y"),
            ("native_col", "native_row"),
            ("detector_native_col", "detector_native_row"),
        ),
        "entry_visual_detector_native",
    )


def _diag_sim_visual_caked(entry):
    pair = _diag_tuple(entry.get("sim_visual_caked_deg") if isinstance(entry, Mapping) else None)
    if pair is not None and len(pair) >= 2:
        try:
            return (float(pair[0]), float(pair[1])), "entry_sim_visual_caked_deg"
        except Exception:
            pass
    pair = _diag_tuple(entry.get("sim_refined_caked_deg") if isinstance(entry, Mapping) else None)
    if pair is not None and len(pair) >= 2:
        try:
            return (float(pair[0]), float(pair[1])), "entry_sim_refined_caked_deg"
        except Exception:
            pass
    pair = _diag_tuple(entry.get("sim_visual_deg") if isinstance(entry, Mapping) else None)
    if pair is not None and len(pair) >= 2:
        try:
            return (float(pair[0]), float(pair[1])), "entry_sim_visual_deg"
        except Exception:
            pass
    pair = _diag_tuple(entry.get("sim_caked") if isinstance(entry, Mapping) else None)
    if pair is not None and len(pair) >= 2:
        try:
            return (float(pair[0]), float(pair[1])), "entry_sim_caked"
        except Exception:
            pass
    return _diag_pair_source(
        entry,
        (
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
        ),
        "entry_visual_caked",
    )


def _diag_legacy_sim_caked(entry):
    pair = _diag_tuple(entry.get("sim_caked") if isinstance(entry, Mapping) else None)
    if pair is not None and len(pair) >= 2:
        try:
            return (float(pair[0]), float(pair[1])), "legacy_sim_caked_tuple"
        except Exception:
            pass
    return _diag_pair_source(
        entry,
        (("simulated_two_theta_deg", "simulated_phi_deg"),),
        "legacy_simulated_caked_fields",
    )


def _diag_copy_projection_kwargs(context, *, use_caked):
    kwargs = dict(context["projection_kwargs"])
    kwargs["caked_view_enabled"] = lambda: bool(use_caked)
    return kwargs


def _diag_build_caked_callbacks(context):
    return mg.make_runtime_geometry_manual_projection_callbacks(
        **_diag_copy_projection_kwargs(context, use_caked=True)
    )


def _diag_build_caked_qr_cache(context, rows, detector_cache):
    callbacks = _diag_build_caked_callbacks(context)
    entries, grouped, lookup = mg._geometry_manual_build_caked_qr_projection_cache(
        rows["overlay_rows"],
        callbacks.project_peaks_to_current_view,
        callbacks.pick_candidates,
        callbacks.simulated_lookup,
        None,
    )
    cache_data = dict(detector_cache)
    cache_data["caked_qr_projection_entries"] = [dict(entry) for entry in entries]
    cache_data["caked_qr_projection_grouped_candidates"] = {
        key: [dict(entry) for entry in value]
        for key, value in (grouped or {}).items()
    }
    cache_data["caked_qr_projection_lookup"] = lookup
    return callbacks, cache_data, grouped


def _diag_poison_visual_grouped_candidates(grouped, poison_by_branch):
    poisoned = {}
    for key, entries in (grouped or {}).items():
        out_entries = []
        for raw_entry in entries or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            branch = _diag_source_branch(entry)
            poison = poison_by_branch.get(branch)
            if poison is not None:
                entry["sim_visual_deg"] = (float(poison[0]), float(poison[1]))
                entry["sim_caked"] = (float(poison[0]), float(poison[1]))
                entry["sim_refined_caked_deg"] = (float(poison[0]), float(poison[1]))
                entry["refined_sim_caked_x"] = float(poison[0])
                entry["refined_sim_caked_y"] = float(poison[1])
                entry["caked_x"] = float(poison[0])
                entry["caked_y"] = float(poison[1])
                entry["display_col"] = float(poison[0])
                entry["display_row"] = float(poison[1])
                entry["diagnostic_source"] = "cache_current_poison"
            out_entries.append(entry)
        poisoned[key] = out_entries
    return poisoned


def _diag_direct_native_to_caked(callbacks, point):
    if point is None:
        return None, "missing_native"
    fn = getattr(callbacks, "native_detector_coords_to_caked_display_coords", None)
    if not callable(fn):
        return None, "missing_converter"
    try:
        result = fn(float(point[0]), float(point[1]))
    except Exception as exc:
        return None, f"exception:{type(exc).__name__}"
    if result is None:
        return None, "outside_detector"
    try:
        caked = (float(result[0]), float(result[1]))
    except Exception:
        return None, "invalid_result"
    if not np.isfinite(caked[0]) or not np.isfinite(caked[1]):
        return None, "invalid_result"
    return caked, "ok"


def _diag_angle_delta(left, right):
    if left is None or right is None:
        return None
    return (
        float(left[0]) - float(right[0]),
        ((float(left[1]) - float(right[1]) + 180.0) % 360.0) - 180.0,
    )


def _diag_pair_distance_caked(left, right):
    delta = _diag_angle_delta(left, right)
    if delta is None:
        return None
    return float(np.hypot(delta[0], delta[1]))


def _diag_caked_visual_source_changed(before, after, *, tth_tol=0.25, phi_tol=0.5):
    if before is None or after is None:
        return True
    delta = _diag_angle_delta(before, after)
    if delta is None:
        return True
    return bool(abs(float(delta[0])) > float(tth_tol) or abs(float(delta[1])) > float(phi_tol))


def _diag_assert_caked_session_refresh_preserved_visual_source(
    before_map,
    after_map,
    *,
    incoming_candidate_source,
):
    changed = []
    for branch in (0, 1):
        before_visual, _before_source = _diag_sim_visual_caked(before_map.get(branch))
        after_visual, _after_source = _diag_sim_visual_caked(after_map.get(branch))
        if _diag_caked_visual_source_changed(before_visual, after_visual):
            changed.append((branch, before_visual, after_visual))
    if changed:
        branch, before_visual, after_visual = changed[0]
        print("caked_session_refresh_visual_source_changed=yes")
        print(f"changed_branch={branch}")
        print(f"before_sim_visual_caked_deg={_diag_point_text(before_visual)}")
        print(f"after_sim_visual_caked_deg={_diag_point_text(after_visual)}")
        print(f"incoming_candidate_source={incoming_candidate_source}")
    else:
        print("caked_session_refresh_visual_source_changed=no")
        print(f"incoming_candidate_source={incoming_candidate_source}")
    assert not changed


def _diag_matches_distance(value, expected, tol=1.0e-3):
    if value is None or expected is None:
        return False
    try:
        return abs(float(value) - float(expected)) <= float(tol)
    except Exception:
        return False


def _diag_preview_distance_match(observed_caked, visual_caked, cache_caked, preview_value):
    visual_distance = _diag_pair_distance_caked(observed_caked, visual_caked)
    cache_distance = _diag_pair_distance_caked(observed_caked, cache_caked)
    visual_match = _diag_matches_distance(preview_value, visual_distance)
    cache_match = _diag_matches_distance(preview_value, cache_distance)
    if visual_match and cache_match:
        match = "both"
    elif visual_match:
        match = "sim_visual"
    elif cache_match:
        match = "sim_cache_current"
    else:
        match = "neither"
    return match, visual_distance, cache_distance


def _diag_text_cell(value):
    if isinstance(value, tuple) and len(value) >= 2:
        return _diag_point_text(value)
    if value is None:
        return "<unavailable>"
    if isinstance(value, float):
        return _diag_float_text(value)
    return str(value)


def _diag_target_caked(entry):
    point, _source = _diag_observed_caked(entry)
    return point


def _diag_cache_current_map(context, detector_targets, caked_targets):
    cache_current = {}
    callbacks = _diag_build_caked_callbacks(context)
    for branch in (0, 1):
        detector_entry = detector_targets.get(branch)
        caked_entry = caked_targets.get(branch)
        display, _display_source = _diag_sim_visual_detector_display(detector_entry)
        native, _native_source = _diag_sim_visual_detector_native(detector_entry)
        caked = _diag_target_caked(caked_entry)
        source = "caked_qr_projection_cache" if caked is not None else "<unavailable>"
        if caked is None and native is not None:
            caked, status = _diag_direct_native_to_caked(callbacks, native)
            source = f"direct_current_native_projection:{status}"
        cache_current[branch] = {
            "source": source,
            "detector_display": display,
            "detector_native": native,
            "caked": caked,
        }
    return cache_current


def _diag_visual_map(detector_targets):
    visual = {}
    for branch, entry in detector_targets.items():
        display, display_source = _diag_sim_visual_detector_display(entry)
        native, native_source = _diag_sim_visual_detector_native(entry)
        caked, caked_source = _diag_sim_visual_caked(entry)
        visual[int(branch)] = {
            "source": caked_source,
            "detector_display": display,
            "detector_display_source": display_source,
            "detector_native": native,
            "detector_native_source": native_source,
            "caked": caked,
        }
    return visual


def _diag_build_refine_preview(use_caked_space, radial_axis, azimuth_axis):
    def _refine_preview(candidate, col, row, **kwargs):
        local_kwargs = dict(kwargs)
        local_kwargs.pop("force_detector_space", None)
        return mg.geometry_manual_refine_preview_point(
            candidate,
            col,
            row,
            use_caked_space=bool(use_caked_space),
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            **local_kwargs,
        )

    return _refine_preview


def _diag_live_toggle(cache_data, click, *, display_background, use_caked_space, profile_cache):
    statuses = []
    sessions = []
    handled, session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        float(click[0]),
        float(click[1]),
        pick_session={},
        current_background_index=0,
        display_background=display_background,
        get_cache_data=lambda **_kwargs: dict(cache_data),
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda value: sessions.append(dict(value)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=statuses.append,
        listed_q_group_entries=lambda: [{"key": _QR_PICKER_TARGET_Q_GROUP_KEY}],
        format_q_group_line=lambda _entry: "target q_group",
        use_caked_space=bool(use_caked_space),
        pick_search_window_px=50.0,
        profile_cache=profile_cache,
    )
    assert handled is True
    assert suppress_drag is True
    assert sessions
    assert session == sessions[-1]
    return session, statuses


def _diag_live_preview(
    session,
    click,
    *,
    cache_data,
    display_background,
    use_caked_space,
    callbacks,
    radial_axis,
    azimuth_axis,
    profile_cache,
):
    remaining = mg.geometry_manual_unassigned_group_candidates(
        session,
        current_background_index=0,
    )
    return mg.geometry_manual_pick_preview_state(
        float(click[0]),
        float(click[1]),
        pick_session=session,
        current_background_index=0,
        force=True,
        remaining_candidates=remaining,
        display_background=display_background,
        cache_data=dict(cache_data),
        refine_preview_point=_diag_build_refine_preview(
            use_caked_space,
            radial_axis,
            azimuth_axis,
        ),
        use_caked_space=bool(use_caked_space),
        caked_angles_to_background_display_coords=getattr(
            callbacks,
            "caked_angles_to_background_display_coords",
            None,
        ),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )


def _diag_live_place(
    session,
    click,
    *,
    cache_data,
    display_background,
    use_caked_space,
    callbacks,
    radial_axis,
    azimuth_axis,
    profile_cache,
):
    state = {"sessions": [], "saved": [], "statuses": []}
    handled, next_session = mg.geometry_manual_place_selection_at(
        float(click[0]),
        float(click[1]),
        pick_session=dict(session),
        current_background_index=0,
        display_background=display_background,
        get_cache_data=lambda **_kwargs: dict(cache_data),
        refine_preview_point=_diag_build_refine_preview(
            use_caked_space,
            radial_axis,
            azimuth_axis,
        ),
        set_pairs_for_index_fn=lambda _idx, entries: (
            state["saved"].append([dict(entry) for entry in (entries or [])])
            or list(entries or [])
        ),
        set_pick_session_fn=lambda value: state["sessions"].append(dict(value)),
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=state["statuses"].append,
        push_undo_state_fn=lambda: None,
        use_caked_space=bool(use_caked_space),
        caked_angles_to_background_display_coords=getattr(
            callbacks,
            "caked_angles_to_background_display_coords",
            None,
        ),
        background_display_to_native_detector_coords=getattr(
            callbacks,
            "background_display_to_native_detector_coords",
            None,
        ),
        native_detector_coords_to_caked_display_coords=getattr(
            callbacks,
            "native_detector_coords_to_caked_display_coords",
            None,
        ),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    assert handled is True
    if state["saved"]:
        entries = state["saved"][-1]
    else:
        entries = next_session.get("pending_entries", []) if isinstance(next_session, dict) else []
    return next_session, [dict(entry) for entry in entries if isinstance(entry, Mapping)], state


def _diag_runtime_refresh_pick_session(
    runtime_session,
    monkeypatch,
    *,
    session,
    cache_data,
    background_image,
    profile_cache,
    use_caked_space,
    detector_grouped=None,
):
    manual_state = SimpleNamespace(pick_session=dict(session))
    refresh_sources = []
    original_refresh = mg.refresh_geometry_manual_pick_session_candidates
    caked_grouped = (
        cache_data.get("caked_qr_projection_grouped_candidates")
        if isinstance(cache_data, Mapping)
        else None
    )
    grouped = cache_data.get("grouped_candidates") if isinstance(cache_data, Mapping) else None

    def _record_refresh(pick_session, *, grouped_candidates, **kwargs):
        if grouped_candidates is caked_grouped:
            refresh_sources.append("caked_qr_projection_grouped_candidates")
        elif grouped_candidates is detector_grouped:
            refresh_sources.append("detector_grouped_candidates_from_cache")
        elif grouped_candidates is grouped:
            refresh_sources.append("grouped_candidates")
        else:
            refresh_sources.append("other_grouped_candidates")
        return original_refresh(
            pick_session,
            grouped_candidates=grouped_candidates,
            **kwargs,
        )

    monkeypatch.setattr(runtime_session, "geometry_manual_state", manual_state, raising=False)
    monkeypatch.setattr(runtime_session.background_runtime_state, "current_background_index", 0)
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pick_session",
        lambda value: setattr(manual_state, "pick_session", dict(value)) or manual_state.pick_session,
    )
    monkeypatch.setattr(runtime_session.simulation_runtime_state, "profile_cache", profile_cache)
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: background_image,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_current_geometry_fit_params", lambda: {}, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: dict(cache_data),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: bool(use_caked_space),
        raising=False,
    )
    if detector_grouped is not None:
        monkeypatch.setattr(
            mg,
            "geometry_manual_detector_picker_grouped_candidates_from_cache",
            lambda *_args, **_kwargs: detector_grouped,
        )
    monkeypatch.setattr(mg, "refresh_geometry_manual_pick_session_candidates", _record_refresh)

    remaining = runtime_session._geometry_manual_unassigned_group_candidates()
    return manual_state.pick_session, remaining, list(refresh_sources)


def _diag_refresh_entries(callbacks, entries):
    refreshed = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        try:
            value = callbacks.refresh_entry_geometry(dict(entry))
        except Exception:
            value = None
        refreshed.append(dict(value) if isinstance(value, Mapping) else dict(entry))
    return refreshed


def _diag_pair_close(left, right, tol=1.0e-3):
    if left is None or right is None:
        return False
    return bool(np.allclose(left, right, atol=tol, rtol=0.0))


def _diag_field_label_errors(row, caked_points):
    errors = []
    for key in (
        "observed_detector_display_px",
        "observed_detector_native_px",
        "sim_visual_detector_display_px",
        "sim_visual_detector_native_px",
        "sim_cache_current_detector_display_px",
        "sim_cache_current_native_px",
    ):
        point = row.get(key)
        if point is None:
            continue
        for label, caked_point in caked_points:
            if _diag_pair_close(point, caked_point):
                errors.append(f"{key}_contains_{label}")
    return ",".join(errors) if errors else ""


def _diag_source_ledger_row(
    event,
    branch,
    entry,
    *,
    visual_map,
    cache_current,
    preview=None,
    direct_result=None,
):
    entry = dict(entry) if isinstance(entry, Mapping) else {}
    observed_display, _observed_display_source = _diag_observed_detector_display(entry)
    observed_native, _observed_native_source = _diag_observed_detector_native(entry)
    observed_caked, _observed_caked_source = _diag_observed_caked(entry)
    visual = visual_map.get(branch, {})
    visual_display = visual.get("detector_display")
    visual_native = visual.get("detector_native")
    visual_caked = visual.get("caked")
    visual_source = str(visual.get("source", "<unavailable>"))
    cache = cache_current.get(branch, {})
    cache_caked = cache.get("caked")
    legacy_caked, legacy_source = _diag_legacy_sim_caked(entry)
    preview_value = None
    if isinstance(preview, Mapping):
        preview_value = preview.get("sim_dist")
        if preview_value is None:
            preview_value = preview.get("printed_preview_distance")
    preview_source = "<unavailable>"
    if isinstance(preview, Mapping):
        preview_source = str(
            preview.get(
                "preview_distance_source",
                preview.get(
                    "sim_dist_source",
                    preview.get("preview_distance_matches", "<unavailable>"),
                ),
            )
        )
    delta = _diag_angle_delta(observed_caked, visual_caked)
    geometry_delta_source = "observed_minus_sim_visual" if delta is not None else "<unavailable>"
    direct_point = direct_result.get("point") if isinstance(direct_result, Mapping) else None
    direct_status = direct_result.get("status") if isinstance(direct_result, Mapping) else "<unavailable>"
    row = {
        "event": event,
        "branch": branch,
        "source_table_index": entry.get("source_table_index", "<none>"),
        "source_row_index": entry.get("source_row_index", "<none>"),
        "source_branch_index": entry.get("source_branch_index", "<none>"),
        "branch_id": entry.get("branch_id", "<none>"),
        "observed_detector_display_px": observed_display,
        "observed_detector_native_px": observed_native,
        "observed_caked_deg": observed_caked,
        "sim_visual_source": visual_source,
        "sim_visual_detector_display_px": visual_display,
        "sim_visual_detector_native_px": visual_native,
        "sim_visual_caked_deg": visual_caked,
        "sim_cache_current_source": cache.get("source", "<unavailable>"),
        "sim_cache_current_detector_display_px": cache.get("detector_display"),
        "sim_cache_current_native_px": cache.get("detector_native"),
        "sim_cache_current_caked_deg": cache_caked,
        "legacy_sim_caked_source": legacy_source,
        "legacy_sim_caked_deg": legacy_caked,
        "preview_distance_source": preview_source,
        "preview_distance_value": preview_value,
        "geometry_minus_sim_caked_source": geometry_delta_source,
        "geometry_minus_sim_caked_delta": delta,
        "direct_detector_native_to_caked_deg": direct_point,
        "direct_detector_native_to_caked_status": direct_status,
    }
    row["field_label_errors"] = _diag_field_label_errors(
        row,
        (
            ("observed_caked_deg", observed_caked),
            ("sim_visual_caked_deg", visual_caked),
            ("sim_cache_current_caked_deg", cache_caked),
            ("legacy_sim_caked_deg", legacy_caked),
        ),
    )
    return row


_SOURCE_LEDGER_FIELDS = (
    "event",
    "branch",
    "source_table_index",
    "source_row_index",
    "source_branch_index",
    "branch_id",
    "observed_detector_display_px",
    "observed_detector_native_px",
    "observed_caked_deg",
    "sim_visual_source",
    "sim_visual_detector_display_px",
    "sim_visual_detector_native_px",
    "sim_visual_caked_deg",
    "sim_cache_current_source",
    "sim_cache_current_detector_display_px",
    "sim_cache_current_native_px",
    "sim_cache_current_caked_deg",
    "legacy_sim_caked_source",
    "legacy_sim_caked_deg",
    "preview_distance_source",
    "preview_distance_value",
    "geometry_minus_sim_caked_source",
    "geometry_minus_sim_caked_delta",
    "direct_detector_native_to_caked_deg",
    "direct_detector_native_to_caked_status",
    "field_label_errors",
)


def _diag_print_source_ledger(rows):
    print("\nsource_ledger")
    print(" | ".join(_SOURCE_LEDGER_FIELDS))
    for row in rows:
        print(" | ".join(_diag_text_cell(row.get(field)) for field in _SOURCE_LEDGER_FIELDS))


def _diag_direct_detector_origin_checks(context, callbacks, entries, visual_map):
    checks = []
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        branch = _diag_source_branch(entry)
        if branch is None:
            continue
        raw_display = _diag_finite_pair(entry, (("raw_x", "raw_y"), ("detector_seed_col", "detector_seed_row"), ("x", "y")))
        raw_native = _diag_display_to_native(context, raw_display)
        geometry_native, _geometry_native_source = _diag_observed_detector_native(entry)
        sim_native = visual_map.get(branch, {}).get("detector_native")
        targets = (
            ("raw_detector_native_px", raw_native, _diag_finite_pair(entry, (("raw_caked_x", "raw_caked_y"),))),
            ("geometry_detector_native_px", geometry_native, _diag_observed_caked(entry)[0]),
            ("sim_visual_detector_native_px", sim_native, visual_map.get(branch, {}).get("caked")),
        )
        for field, native, trace_caked in targets:
            direct_caked, status = _diag_direct_native_to_caked(callbacks, native)
            comparison = "missing_trace"
            if direct_caked is not None and trace_caked is not None:
                comparison = (
                    "matches_trace"
                    if _diag_pair_close(direct_caked, trace_caked, tol=1.0e-2)
                    else "differs_from_trace"
                )
            checks.append(
                {
                    "branch": branch,
                    "field": field,
                    "native": native,
                    "direct": direct_caked,
                    "trace": trace_caked,
                    "status": status,
                    "comparison": comparison,
                }
            )
    print("\ndetector_origin_direct_conversion_check")
    print("branch | field | native_px | direct_caked_deg | trace_caked_deg | status | comparison")
    for check in checks:
        print(
            " | ".join(
                (
                    str(check["branch"]),
                    str(check["field"]),
                    _diag_point_text(check["native"]),
                    _diag_point_text(check["direct"]),
                    _diag_point_text(check["trace"]),
                    str(check["status"]),
                    str(check["comparison"]),
                )
            )
        )
    return checks


def _diag_print_frozen_visual_branch_map(session, visual_map, event_name):
    branch_map = _diag_branch_map(session.get("group_entries", []) if isinstance(session, Mapping) else [])
    print("\npending_visual_branch_map")
    print("branch | exists | sim_visual_caked_deg | sim_visual_source_event")
    for branch in (0, 1):
        exists = branch in branch_map
        visual = visual_map.get(branch, {})
        print(
            " | ".join(
                (
                    str(branch),
                    "yes" if exists else "no",
                    _diag_point_text(visual.get("caked")),
                    event_name if exists else "<missing>",
                )
            )
        )
    return branch_map


def _diag_make_preview_check(event, branch, observed_caked, visual_map, cache_current, preview):
    preview_value = None
    if isinstance(preview, Mapping):
        preview_value = preview.get("sim_dist")
    match, visual_distance, cache_distance = _diag_preview_distance_match(
        observed_caked,
        visual_map.get(branch, {}).get("caked"),
        cache_current.get(branch, {}).get("caked"),
        preview_value,
    )
    return {
        "event": event,
        "branch": branch,
        "printed_preview_distance": preview_value,
        "preview_distance_source": (
            str(preview.get("preview_distance_source"))
            if isinstance(preview, Mapping) and preview.get("preview_distance_source") is not None
            else "<unavailable>"
        ),
        "visual_distance": visual_distance,
        "cache_current_distance": cache_distance,
        "preview_distance_matches": match,
    }


def _diag_conclusion(failures):
    if not failures:
        return {
            "conclusion": "no_failure_found",
            "first_bad_event": "<none>",
            "first_bad_branch": "<none>",
            "first_bad_field": "<none>",
            "expected_source": "<none>",
            "actual_source": "<none>",
            "explanation": "all checked sources stayed visual",
        }
    order = {
        "visual_branch_map_missing_non_clicked_branch": 0,
        "detector_to_caked_trace_uses_wrong_or_double_transformed_native": 1,
        "preview_distance_uses_cache_current": 2,
        "legacy_sim_caked_uses_cache_current": 3,
        "caked_values_labeled_as_detector_px": 4,
    }
    return min(failures, key=lambda item: order.get(item["conclusion"], 99))


def _diag_entry_table_line(entry, picker_keys, overlay_keys, source_keys, display_w, display_h, native_w, native_h):
    display = _diag_detector_display_point(entry)
    native = _diag_detector_native_point(entry)
    identity = _diag_source_identity(entry)
    return " | ".join(
        [
            repr(_diag_q_group_key(entry)),
            repr(_diag_hkl(entry)),
            str(entry.get("source_branch_index", "<none>")),
            str(entry.get("source_table_index", "<none>")),
            str(entry.get("source_row_index", "<none>")),
            str(entry.get("source_peak_index", "<none>")),
            str(entry.get("branch_id", "<none>")),
            _diag_point_text(display, "no_detector_display_px"),
            _diag_point_text(native, "no_detector_native_px"),
            str(_diag_inside(display, display_w, display_h) and _diag_inside(native, native_w, native_h)),
            str(identity in picker_keys),
            str(identity in overlay_keys),
            str(entry.get("diagnostic_source", entry.get("source_label", "<none>"))),
        ]
    )


def _diag_rank_for_entry(entry, profile_cache, branch_id=None):
    return mosaic_top_selection.mosaic_top_rank_key(
        entry,
        branch_id=branch_id,
        source_order=0,
        profile_cache=profile_cache,
    )


def test_qr_sim_peak_refines_detector_candidate_to_sim_local_max() -> None:
    image = np.zeros((24, 24), dtype=float)
    image[6, 8] = 25.0
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_branch_index": 1,
        "display_col": 5.4,
        "display_row": 6.2,
        "native_col": 105.4,
        "native_row": 206.2,
        "caked_x": 11.0,
        "caked_y": 22.0,
    }

    refined = mg.geometry_manual_refine_qr_sim_peak_detector(
        candidate,
        detector_simulation_image=image,
        search_radius_px=5,
        detector_display_to_native_coords=lambda col, row: (col + 100.0, row + 200.0),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            col / 10.0,
            row / 10.0,
        ),
    )

    assert refined is not None
    assert refined["source_branch_index"] == 1
    assert refined["sim_nominal_detector_display_px"] == (5.4, 6.2)
    assert refined["sim_refined_detector_display_px"] == (8.0, 6.0)
    assert refined["sim_refined_detector_native_px"] == (108.0, 206.0)
    assert refined["sim_refined_caked_deg"] == (10.8, 20.6)
    assert refined["sim_refinement_status"] == "refined"
    assert refined["sim_refinement_source"] == "detector_simulation_image"
    assert refined["sim_visual_deg"] == refined["sim_refined_caked_deg"]


def test_qr_sim_peak_detector_blank_image_is_not_refined() -> None:
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_branch_index": 1,
        "display_col": 5.0,
        "display_row": 6.0,
        "native_col": 105.0,
        "native_row": 206.0,
        "caked_x": 11.0,
        "caked_y": 22.0,
    }

    refined = mg.geometry_manual_refine_qr_sim_peak_detector(
        candidate,
        detector_simulation_image=np.zeros((24, 24), dtype=float),
        search_radius_px=5,
        detector_display_to_native_coords=lambda col, row: (col + 100.0, row + 200.0),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            col / 10.0,
            row / 10.0,
        ),
    )

    assert refined is not None
    assert refined["sim_refinement_status"] == "no_peak_found"
    assert "sim_refined_detector_display_px" not in refined
    assert "sim_refined_detector_native_px" not in refined
    assert "sim_refined_caked_deg" not in refined
    assert refined["sim_visual_deg"] == refined["sim_nominal_caked_deg"]


def test_qr_sim_peak_refines_caked_candidate_to_sim_local_max() -> None:
    radial = np.linspace(0.0, 9.0, 10)
    azimuth = np.linspace(0.0, 9.0, 10)
    image = np.zeros((10, 10), dtype=float)
    image[4, 7] = 100.0
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_branch_index": 0,
        "caked_x": 5.2,
        "caked_y": 4.1,
    }

    refined = mg.geometry_manual_refine_qr_sim_peak_caked(
        candidate,
        caked_simulation_image=image,
        radial_axis=radial,
        azimuth_axis=azimuth,
    )

    assert refined is not None
    assert refined["source_branch_index"] == 0
    assert refined["sim_nominal_caked_deg"] == (5.2, 4.1)
    assert np.allclose(refined["sim_refined_caked_deg"], (7.0, 4.0), atol=1.0e-6)
    assert refined["sim_refinement_status"] == "refined"
    assert refined["sim_refinement_source"] == "caked_simulation_image"
    assert refined["sim_refined_caked_projection_status"] == "caked_simulation_image_axes"
    assert refined["sim_refined_caked_projection_real_callback"] is False
    assert refined["sim_visual_deg"] == refined["sim_refined_caked_deg"]
    assert "sim_refined_detector_display_px" not in refined
    assert "refined_sim_x" not in refined


def test_qr_sim_peak_caked_blank_image_is_not_refined() -> None:
    radial = np.linspace(0.0, 9.0, 10)
    azimuth = np.linspace(0.0, 9.0, 10)
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_branch_index": 0,
        "caked_x": 5.2,
        "caked_y": 4.1,
    }

    refined = mg.geometry_manual_refine_qr_sim_peak_caked(
        candidate,
        caked_simulation_image=np.zeros((10, 10), dtype=float),
        radial_axis=radial,
        azimuth_axis=azimuth,
    )

    assert refined is not None
    assert refined["sim_refinement_status"] == "no_peak_found"
    assert "sim_refined_caked_deg" not in refined
    assert "refined_sim_caked_x" not in refined
    assert refined["sim_visual_deg"] == refined["sim_nominal_caked_deg"]


def test_detector_picker_row_uses_refined_sim_detector_px_and_matching_native() -> None:
    entry = {
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_branch_index": 0,
        "display_col": 10.0,
        "display_row": 20.0,
        "native_col": 110.0,
        "native_row": 120.0,
        "sim_refined_detector_display_px": (12.0, 23.0),
        "sim_refined_detector_native_px": (112.0, 123.0),
    }

    row = mg.geometry_manual_detector_picker_row(
        entry,
        display_width=100,
        display_height=100,
        native_width=200,
        native_height=200,
    )

    assert row is not None
    assert row["detector_display_px"] == (12.0, 23.0)
    assert row["detector_native_px"] == (112.0, 123.0)
    assert row["display_col"] == 12.0
    assert row["display_row"] == 23.0
    assert row["detector_display_source"] == "sim_refined_detector_display_px"


def test_detector_picker_row_does_not_pair_refined_display_with_nominal_native() -> None:
    entry = {
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_branch_index": 0,
        "display_col": 10.0,
        "display_row": 20.0,
        "native_col": 110.0,
        "native_row": 120.0,
        "sim_refined_detector_display_px": (12.0, 23.0),
    }

    row = mg.geometry_manual_detector_picker_row(
        entry,
        display_width=100,
        display_height=100,
        native_width=200,
        native_height=200,
    )

    assert row is not None
    assert row["detector_display_px"] == (12.0, 23.0)
    assert "detector_native_px" not in row
    assert "native_col" not in row
    assert "native_row" not in row


def test_detector_picker_dedupe_uses_refined_sim_detector_px() -> None:
    nominal_entry = {
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_table_index": 160,
        "source_row_index": 24,
        "source_branch_index": 0,
        "source_peak_index": 24,
        "branch_id": "primary:1:10:0",
        "display_col": 10.0,
        "display_row": 20.0,
        "native_col": 110.0,
        "native_row": 120.0,
        "detector_picker_source": "manual_saved_pair",
    }
    refined_entry = {
        **nominal_entry,
        "sim_refined_detector_display_px": (12.0, 23.0),
        "sim_refined_detector_native_px": (112.0, 123.0),
        "detector_picker_source": "fresh_source_rows",
    }

    deduped = mg._geometry_manual_detector_picker_dedupe_rows(
        [nominal_entry, refined_entry]
    )

    assert len(deduped) == 2
    assert deduped[0]["detector_picker_source"] == "manual_saved_pair"
    assert deduped[1]["detector_picker_source"] == "fresh_source_rows"


def test_minus_1_0_10_sim_visual_uses_refined_peak_for_both_branches(tmp_path) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    cache_data = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        profile_cache=rows["profile_cache"],
    )
    target_by_branch = _diag_target_rows_by_branch(grouped)
    assert set(target_by_branch) == {0, 1}

    image = np.zeros((3000, 3000), dtype=float)
    branch_refined_display = {}
    for branch, entry in target_by_branch.items():
        nominal = _diag_detector_display_point(entry)
        assert nominal is not None
        refined = (
            float(round(nominal[0]) + (3 if branch == 0 else -4)),
            float(round(nominal[1]) + (2 if branch == 0 else -3)),
        )
        image[int(refined[1]), int(refined[0])] = 1000.0 + float(branch)
        branch_refined_display[int(branch)] = refined

    def display_to_native(col, row):
        return _diag_display_to_native(context, (float(col), float(row)))

    def native_to_caked(col, row):
        return (float(col) / 100.0, float(row) / 100.0)

    refined_cache = mg.geometry_manual_refine_qr_sim_candidates_in_cache(
        cache_data,
        detector_simulation_image=image,
        detector_display_to_native_coords=display_to_native,
        native_detector_coords_to_caked_display_coords=native_to_caked,
    )
    refined_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        refined_cache,
        profile_cache=rows["profile_cache"],
    )
    refined_target = _diag_target_rows_by_branch(refined_grouped)
    assert set(refined_target) == {0, 1}

    print("\nCMD sim refine (-1,0,10)")
    for branch in (0, 1):
        before = target_by_branch[branch]
        after = refined_target[branch]
        nominal_display = after.get("sim_nominal_detector_display_px")
        refined_display = after.get("sim_refined_detector_display_px")
        refined_caked = after.get("sim_refined_caked_deg")
        manual_caked = _diag_finite_pair(
            before,
            (
                ("background_two_theta_deg", "background_phi_deg"),
                ("caked_x", "caked_y"),
            ),
        )
        delta = (
            float(
                np.hypot(
                    float(manual_caked[0]) - float(refined_caked[0]),
                    float(manual_caked[1]) - float(refined_caked[1]),
                )
            )
            if manual_caked is not None and refined_caked is not None
            else float("nan")
        )
        print(
            "CMD before "
            f"branch={branch} nominal_sim_detector={_diag_point_text(nominal_display)} "
            f"nominal_native={_diag_point_text(after.get('sim_nominal_detector_native_px'))}"
        )
        print(
            "CMD after "
            f"branch={branch} refined_sim_detector={_diag_point_text(refined_display)} "
            f"refined_sim_caked={_diag_point_text(refined_caked)} "
            f"geometry_manual_peak={_diag_point_text(manual_caked)} "
            f"geometry_minus_refined_sim_delta={_diag_float_text(delta)}"
        )
        assert nominal_display is not None
        assert after.get("sim_nominal_detector_native_px") is not None
        assert np.allclose(refined_display, branch_refined_display[branch], atol=1.0e-6)
        assert after["sim_refinement_status"] == "refined"
        assert after["sim_refinement_source"] == "detector_simulation_image"
        assert after["sim_visual_deg"] == after["sim_refined_caked_deg"]
        assert after.get("sim_visual_source") == "sim_refined_caked_deg"
        assert after.get("sim_cache_current_deg") is None


def test_qr_sim_refinement_does_not_change_manual_background_refinement() -> None:
    background = np.zeros((20, 20), dtype=float)
    background[7, 8] = 50.0
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "display_col": 5.0,
        "display_row": 6.0,
    }

    before = mg.geometry_manual_refine_preview_point(
        candidate,
        5.2,
        6.1,
        display_background=background,
        cache_data={},
        use_caked_space=False,
    )
    sim_image = np.zeros((20, 20), dtype=float)
    sim_image[3, 4] = 100.0
    _refined_sim = mg.geometry_manual_refine_qr_sim_peak_detector(
        candidate,
        detector_simulation_image=sim_image,
        search_radius_px=5,
    )
    after = mg.geometry_manual_refine_preview_point(
        candidate,
        5.2,
        6.1,
        display_background=background,
        cache_data={},
        use_caked_space=False,
    )

    assert before == (8.0, 7.0)
    assert after == before


def test_qr_picker_detector_pixel_positions(tmp_path) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    display_h, display_w = _diag_display_shape(context)
    native_h, native_w = _diag_native_shape(context)
    flags = context["saved_state"].get("flags", {})
    display_to_native_fn = getattr(
        context["projection_callbacks"],
        "background_display_to_native_detector_coords",
        None,
    )
    native_to_display_fn = context["projection_kwargs"].get(
        "native_detector_coords_to_detector_display_coords"
    )

    overlay_rows = [
        row
        for row in rows["overlay_rows"]
        if _diag_q_group_key(row) is not None
        and _diag_inside(_diag_detector_display_point(row), display_w, display_h)
        and _diag_inside(_diag_detector_native_point(row), native_w, native_h)
    ]
    picker_rows = [
        row
        for row in rows["picker_rows"]
        if _diag_q_group_key(row) is not None
        and _diag_inside(_diag_detector_display_point(row), display_w, display_h)
        and _diag_inside(_diag_detector_native_point(row), native_w, native_h)
    ]
    source_rows = [
        row
        for row in rows["collapsed_source_rows"]
        if _diag_q_group_key(row) is not None
    ]
    picker_keys = {_diag_source_identity(row) for row in picker_rows}
    overlay_keys = {_diag_source_identity(row) for row in overlay_rows}
    source_keys = {_diag_source_identity(row) for row in source_rows}
    drawn_not_selectable = sorted(overlay_keys - picker_keys, key=repr)
    selectable_not_drawn = sorted(picker_keys - overlay_keys, key=repr)

    print("\nSection 1: detector geometry and orientation")
    print(f"detector image width/height: {native_w} / {native_h}")
    print(f"displayed background width/height: {display_w} / {display_h}")
    print(f"display rotation k: {context['projection_kwargs'].get('display_rotate_k')}")
    print("display flip settings: flip_x=False flip_y=False")
    print(
        "backend rotation/flip settings: "
        f"rotation_k={flags.get('background_backend_rotation_k')} "
        f"flip_x={flags.get('background_backend_flip_x')} "
        f"flip_y={flags.get('background_backend_flip_y')}"
    )
    print(
        "detector display -> native transform function name: "
        f"{_diag_function_name(display_to_native_fn)}"
    )
    print(
        "native -> display transform function name: "
        f"{_diag_function_name(native_to_display_fn) if native_to_display_fn is not None else 'rotate_point_for_display'}"
    )
    if context["runtime_error"]:
        print(f"startup runtime preflight error captured: {context['runtime_error']}")

    print("\nSection 2: all Qr picker candidates on detector")
    print(
        "q_group_key | hkl | branch | table | row | peak | branch_id | "
        "detector_display_px | detector_native_px | inside_detector | "
        "picker_candidate | overlay_drawn | source"
    )
    for entry in _diag_sorted(picker_rows):
        print(
            _diag_entry_table_line(
                entry,
                picker_keys,
                overlay_keys,
                source_keys,
                display_w,
                display_h,
                native_w,
                native_h,
            )
        )
        display = _diag_detector_display_point(entry)
        native = _diag_detector_native_point(entry)
        assert not _diag_detector_point_is_caked(entry, display)
        assert not _diag_detector_point_is_caked(entry, native)

    target_rows = [
        row
        for row in picker_rows
        if _diag_q_group_key(row) == _QR_PICKER_TARGET_Q_GROUP_KEY
        and _diag_hkl(row) == _QR_PICKER_TARGET_HKL
    ]
    target_by_branch = {
        int(row.get("source_branch_index")): row
        for row in target_rows
        if row.get("source_branch_index") is not None
    }

    print("\nSection 3: target group only")
    print(f"target q_group_key={_QR_PICKER_TARGET_Q_GROUP_KEY} hkl={_QR_PICKER_TARGET_HKL}")
    for branch in sorted(target_by_branch):
        entry = target_by_branch[branch]
        display = _diag_detector_display_point(entry)
        native = _diag_detector_native_point(entry)
        display_to_native = _diag_display_to_native(context, display)
        native_to_display = _diag_native_to_display(context, native)
        native_delta = (
            float(np.hypot(display_to_native[0] - native[0], display_to_native[1] - native[1]))
            if display_to_native is not None and native is not None
            else float("nan")
        )
        display_delta = (
            float(np.hypot(native_to_display[0] - display[0], native_to_display[1] - display[1]))
            if native_to_display is not None and display is not None
            else float("nan")
        )
        branch_id = str(entry.get("branch_id", ""))
        print(
            "branch "
            f"{branch}: detector_display_px={_diag_point_text(display)} "
            f"detector_native_px={_diag_point_text(native)} "
            f"display_to_native(detector_display_px)={_diag_point_text(display_to_native)} "
            f"native_to_display(detector_native_px)={_diag_point_text(native_to_display)} "
            f"roundtrip_delta_native={_diag_float_text(native_delta)} "
            f"roundtrip_delta_display={_diag_float_text(display_delta)} "
            f"inside_detector={_diag_inside(display, display_w, display_h) and _diag_inside(native, native_w, native_h)} "
            f"mosaic_top_rank_key={_diag_rank_for_entry(entry, rows['profile_cache'], branch_id=branch_id)} "
            f"mosaic_weight={entry.get('mosaic_weight', '<none>')} "
            f"best_sample_index={entry.get('best_sample_index', '<none>')}"
        )
        assert display is not None
        assert native is not None
        assert _diag_inside(display, display_w, display_h)
        assert _diag_inside(native, native_w, native_h)
        assert not _diag_detector_point_is_caked(entry, display)
        assert not _diag_detector_point_is_caked(entry, native)
        assert display_to_native is not None
        assert native_to_display is not None
        assert np.allclose(display_to_native, native, atol=1.0e-6, rtol=0.0)
        assert np.allclose(native_to_display, display, atol=1.0e-6, rtol=0.0)

    assert set(target_by_branch) == {0, 1}

    print("\nSection 4: click simulation for target group")
    payload = ps.build_hkl_pick_simulation_point_payload(picker_rows)
    runtime_state = SimpleNamespace(
        peak_records=[dict(row) for row in picker_rows],
        peak_positions=[_diag_detector_display_point(row) for row in picker_rows],
        peak_positions_filtered=False,
        profile_cache=rows["profile_cache"],
    )
    for branch in sorted(target_by_branch):
        entry = target_by_branch[branch]
        display = _diag_detector_display_point(entry)
        assert display is not None
        idx, selected, distance, within = ps.find_peak_record_for_canvas_click(
            runtime_state,
            float(display[0]),
            float(display[1]),
            ensure_peak_overlay_data=lambda force=False: None,
            max_axis_distance_px=25.0,
            simulation_point_candidates=payload,
            use_caked_display=False,
        )
        assert selected is not None
        selected_display = _diag_detector_display_point(selected)
        print(
            "branch "
            f"{branch}: click_px={_diag_point_text(display)} "
            f"nearest_picker_candidate_index={idx} "
            f"distance_px={_diag_float_text(distance)} within={within} "
            f"selected_q_group_key={_diag_q_group_key(selected)} "
            f"selected_source_branch_index={selected.get('source_branch_index')} "
            f"selected_source_table_index={selected.get('source_table_index')} "
            f"selected_source_row_index={selected.get('source_row_index')} "
            f"selected_branch_id={selected.get('branch_id')}"
        )
        assert int(selected.get("source_branch_index")) == branch
        assert _diag_q_group_key(selected) == _QR_PICKER_TARGET_Q_GROUP_KEY
        assert _diag_hkl(selected) == _QR_PICKER_TARGET_HKL
        assert selected_display is not None
        assert np.allclose(selected_display, display, atol=1.0e-6, rtol=0.0)
        assert not _diag_detector_point_is_caked(selected, selected_display)

    detector_visual_consistent = not drawn_not_selectable and not selectable_not_drawn
    print("\nSection 5: visual mismatch summary")
    print(f"total Qr candidates on detector: {len(picker_rows)}")
    print(f"total picker candidates: {len(picker_rows)}")
    print(f"total overlay-drawn points: {len(overlay_rows)}")
    print(f"candidates drawn but not selectable: {drawn_not_selectable}")
    print(f"candidates selectable but not drawn: {selectable_not_drawn}")
    for branch in (0, 1):
        point = _diag_detector_display_point(target_by_branch.get(branch))
        print(
            f"target (-1,0,10) branch {branch} detector px: "
            f"{_diag_point_text(point, 'missing_target_branch')}"
        )
    print(f"detector visual point source is consistent: {detector_visual_consistent}")

    assert detector_visual_consistent


def _diag_minus_1_0_10_caked_probe(tmp_path):
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    assert set(caked_targets) == {0, 1}
    visual_map = _diag_visual_map(detector_targets)
    cache_current = _diag_cache_current_map(context, detector_targets, caked_targets)

    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    caked_session, _statuses = _diag_live_toggle(
        caked_cache,
        caked_select_point,
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )
    caked_select_map = _diag_branch_map(caked_session.get("group_entries", []))

    click0 = _diag_target_caked(detector_targets[0])
    assert click0 is not None
    preview0 = _diag_live_preview(
        caked_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    check0 = _diag_make_preview_check(
        "caked_mode_placement_branch_0",
        0,
        (float(preview0["refined_col"]), float(preview0["refined_row"])),
        visual_map,
        cache_current,
        preview0,
    )
    caked_session, branch0_entries, place0_state = _diag_live_place(
        caked_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )

    click1 = _diag_target_caked(detector_targets[1])
    assert click1 is not None
    preview1 = _diag_live_preview(
        caked_session,
        click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    check1 = _diag_make_preview_check(
        "caked_mode_placement_branch_1",
        1,
        (float(preview1["refined_col"]), float(preview1["refined_row"])),
        visual_map,
        cache_current,
        preview1,
    )
    _next_session, saved_entries, place1_state = _diag_live_place(
        caked_session,
        click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    return {
        "context": context,
        "detector_targets": detector_targets,
        "caked_targets": caked_targets,
        "visual_map": visual_map,
        "cache_current": cache_current,
        "caked_select_map": caked_select_map,
        "preview0": preview0,
        "preview1": preview1,
        "preview_check0": check0,
        "preview_check1": check1,
        "branch0_entries": branch0_entries,
        "saved_entries": saved_entries,
        "place0_state": place0_state,
        "place1_state": place1_state,
    }


def test_minus_1_0_10_caked_preview_distance_uses_visual_sim(tmp_path) -> None:
    probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    print("\ncaked_preview_distance_before_after")
    print("before_source=sim_cache_current_caked_deg")
    print("after_source=sim_visual_caked_deg")
    for check in (probe["preview_check0"], probe["preview_check1"]):
        print(
            "branch={branch} printed={printed} visual={visual} cache_current={cache} "
            "source={source} match={match}".format(
                branch=check["branch"],
                printed=_diag_float_text(check["printed_preview_distance"]),
                visual=_diag_float_text(check["visual_distance"]),
                cache=_diag_float_text(check["cache_current_distance"]),
                source=check["preview_distance_source"],
                match=check["preview_distance_matches"],
            )
        )
        assert check["preview_distance_source"] == "sim_visual_caked_deg"
        assert check["preview_distance_matches"] == "sim_visual"
        assert _diag_matches_distance(
            check["printed_preview_distance"],
            check["visual_distance"],
        )
        assert not _diag_matches_distance(
            check["printed_preview_distance"],
            check["cache_current_distance"],
        )


def test_minus_1_0_10_caked_assignment_distance_uses_visual_sim(tmp_path) -> None:
    probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    branch0 = _diag_branch_map(probe["branch0_entries"])[0]
    saved = _diag_branch_map(probe["saved_entries"])
    branch1 = saved[1]
    status0 = probe["place0_state"]["statuses"][-1]
    visual0 = probe["preview_check0"]["visual_distance"]
    print("\ncaked_assignment_distance_before_after")
    print("before_source=sim_cache_current_caked_deg")
    print("after_source=sim_visual_caked_deg")
    print(f"branch=0 status={status0}")
    print(
        "branch=0 assignment={assignment} cache_current={cache}".format(
            assignment=_diag_float_text(branch0.get("assignment_distance_to_sim")),
            cache=_diag_float_text(branch0.get("assignment_distance_to_cache_current_sim")),
        )
    )
    print(
        "branch=1 assignment={assignment} cache_current={cache}".format(
            assignment=_diag_float_text(branch1.get("assignment_distance_to_sim")),
            cache=_diag_float_text(branch1.get("assignment_distance_to_cache_current_sim")),
        )
    )
    assert branch0["assignment_distance_source"] == "sim_visual_caked_deg"
    assert branch1["assignment_distance_source"] == "sim_visual_caked_deg"
    assert _diag_matches_distance(branch0["assignment_distance_to_sim"], visual0)
    assert f"({float(visual0):.2f} deg from sim)" in status0
    assert not _diag_matches_distance(
        branch0["assignment_distance_to_sim"],
        branch0.get("assignment_distance_to_cache_current_sim"),
    )
    assert not _diag_matches_distance(
        branch1["assignment_distance_to_sim"],
        branch1.get("assignment_distance_to_cache_current_sim"),
    )


def test_minus_1_0_10_caked_preview_status_line_uses_visual_sim(tmp_path) -> None:
    probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    print("\ncaked_preview_status_line_before_after")
    print("before=Manual pick preview ... tagged sim [-1,0,10] (78.85 deg)")
    for branch, preview, check in (
        (0, probe["preview0"], probe["preview_check0"]),
        (1, probe["preview1"], probe["preview_check1"]),
    ):
        message = str(preview["message"])
        print(f"branch={branch} after={message}")
        assert "Manual pick preview" in message
        assert "status_distance_source=sim_visual_caked_deg" in message
        assert "status_distance_units=deg" in message
        assert "78.85 deg" not in message
        assert "77.85 deg" not in message
        assert "80.32 deg" not in message
        assert check["preview_distance_source"] == "sim_visual_caked_deg"
        assert _diag_matches_distance(
            preview["status_distance_value"],
            check["visual_distance"],
        )
        assert not _diag_matches_distance(
            preview["status_distance_value"],
            check["cache_current_distance"],
        )
        assert float(preview["status_distance_value"]) < 2.0


def test_minus_1_0_10_caked_assignment_status_line_uses_visual_sim(tmp_path) -> None:
    probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    branch0 = _diag_branch_map(probe["branch0_entries"])[0]
    saved = _diag_branch_map(probe["saved_entries"])
    branch1 = saved[1]
    status0 = str(probe["place0_state"]["statuses"][-1])
    print("\ncaked_assignment_status_line_before_after")
    print("before=Placed peak ... Assigned ... (77.85 deg from sim)")
    print(f"branch=0 after={status0}")
    print(
        "branch=1 saved_assignment_source={source} saved_assignment_distance={distance}".format(
            source=branch1.get("assignment_distance_source"),
            distance=_diag_float_text(branch1.get("assignment_distance_to_sim")),
        )
    )
    assert "Assigned to" in status0
    assert "status_distance_source=sim_visual_caked_deg" in status0
    assert "status_distance_units=deg" in status0
    assert "77.85 deg from sim" not in status0
    assert "80.32 deg" not in status0
    assert branch0["assignment_distance_source"] == "sim_visual_caked_deg"
    assert branch1["assignment_distance_source"] == "sim_visual_caked_deg"
    assert _diag_matches_distance(
        branch0["status_distance_value"],
        probe["preview_check0"]["visual_distance"],
    )
    assert _diag_matches_distance(
        branch1["status_distance_value"],
        probe["preview_check1"]["visual_distance"],
    )
    assert float(branch0["status_distance_value"]) < 2.0
    assert float(branch1["status_distance_value"]) < 2.0


def test_manual_geometry_live_code_path_marker(capsys) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    stamp = mg.print_geometry_manual_live_code_path_stamp(force=True)
    output = capsys.readouterr().out
    print("\nmanual_geometry_live_code_path_marker")
    print(stamp)
    assert "source_path_marker=manual_preview_visual_distance_patch_v1" in output
    assert "manual_geometry.__file__=" in output
    assert "runtime_session.__file__=" in output
    assert str(Path(runtime_session.__file__).resolve()) in stamp
    assert "ra_sim" in stamp
    assert "manual_geometry.py" in stamp


def test_live_caked_visual_trace_without_run_id_stays_unattributed(capsys) -> None:
    active_run_id = mg.geometry_manual_start_run_id()
    legacy_entry = {
        "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "hkl": _QR_PICKER_TARGET_HKL,
        "source_branch_index": 0,
        "source_table_index": 160,
        "source_row_index": 24,
        "sim_visual_deg": (40.237466, 36.527645),
    }
    mg.geometry_manual_trace_live_caked_visual_source_event(
        "legacy_projection_without_run_id",
        selected_candidate=legacy_entry,
    )
    output = capsys.readouterr().out

    assert f"manual_geometry_run_id={active_run_id}" not in output
    assert f"manual_geometry_run_id={mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID}" in output
    assert "emitter=geometry_manual_trace_live_caked_visual_source_event" in output


def test_live_preview_and_place_without_session_run_id_stay_unattributed(tmp_path) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    session, _statuses = _diag_live_toggle(
        caked_cache,
        caked_select_point,
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )
    legacy_session = dict(session)
    legacy_session.pop("manual_geometry_run_id", None)
    legacy_session.pop("manual_trace_version", None)
    active_run_id = mg.geometry_manual_start_run_id()

    click0 = _diag_target_caked(caked_targets[0])
    preview = _diag_live_preview(
        legacy_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    assert preview["manual_geometry_run_id"] == mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID
    assert f"manual_geometry_run_id={active_run_id}" not in preview["message"]
    assert (
        f"manual_geometry_run_id={mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID}"
        in preview["message"]
    )

    _next_session, entries, place_state = _diag_live_place(
        legacy_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    status = str(place_state["statuses"][-1])
    assert f"manual_geometry_run_id={active_run_id}" not in status
    assert f"manual_geometry_run_id={mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID}" in status
    assert entries
    assert entries[0]["manual_geometry_run_id"] == mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID


def test_manual_trace_never_prints_caked_values_as_detector_px(tmp_path) -> None:
    probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    rows = []
    for branch in (0, 1):
        row = _diag_source_ledger_row(
            "caked_mode_qr_selection",
            branch,
            probe["caked_select_map"][branch],
            visual_map=probe["visual_map"],
            cache_current=probe["cache_current"],
            direct_result={"point": None, "status": "not_checked"},
        )
        rows.append(row)
    print("\nmanual_trace_no_caked_values_as_detector_px")
    for row in rows:
        print(
            "branch={branch} detector_display={display} caked={caked} errors={errors}".format(
                branch=row["branch"],
                display=_diag_point_text(row["observed_detector_display_px"]),
                caked=_diag_point_text(row["observed_caked_deg"]),
                errors=row["field_label_errors"] or "<none>",
            )
        )
        assert row["observed_caked_deg"] is not None
        assert row["field_label_errors"] == ""
        assert not _diag_pair_close(
            row["observed_detector_display_px"],
            row["observed_caked_deg"],
        )


def test_manual_trace_no_caked_values_as_detector_px(tmp_path) -> None:
    test_manual_trace_never_prints_caked_values_as_detector_px(tmp_path)


def test_minus_1_0_10_live_source_ledger(tmp_path) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_callbacks = context["projection_callbacks"]
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}

    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    assert set(caked_targets) == {0, 1}
    detector_cache.update(
        {
            "caked_qr_projection_entries": caked_cache["caked_qr_projection_entries"],
            "caked_qr_projection_grouped_candidates": caked_cache[
                "caked_qr_projection_grouped_candidates"
            ],
            "caked_qr_projection_lookup": caked_cache["caked_qr_projection_lookup"],
        }
    )

    visual_map = _diag_visual_map(detector_targets)
    cache_current = _diag_cache_current_map(context, detector_targets, caked_targets)
    ledger_rows = []
    preview_checks = []
    failures = []

    def _direct_for_row(branch, entry):
        native, _source = _diag_observed_detector_native(entry)
        if native is None:
            native = visual_map.get(branch, {}).get("detector_native")
        point, status = _diag_direct_native_to_caked(caked_callbacks, native)
        return {"point": point, "status": status}

    def _append_event(event, branch_entries, previews=None):
        branch_entries = dict(branch_entries or {})
        previews = dict(previews or {})
        for branch in (0, 1):
            entry = branch_entries.get(branch, {})
            direct = _direct_for_row(branch, entry)
            row = _diag_source_ledger_row(
                event,
                branch,
                entry,
                visual_map=visual_map,
                cache_current=cache_current,
                preview=previews.get(branch),
                direct_result=direct,
            )
            ledger_rows.append(row)
            if row["field_label_errors"]:
                failures.append(
                    {
                        "conclusion": "caked_values_labeled_as_detector_px",
                        "first_bad_event": event,
                        "first_bad_branch": branch,
                        "first_bad_field": row["field_label_errors"],
                        "expected_source": "detector_px_fields_have_detector_px",
                        "actual_source": "caked_deg",
                        "explanation": "angle-valued caked pair appeared in detector px field",
                    }
                )
            legacy = row.get("legacy_sim_caked_deg")
            visual = row.get("sim_visual_caked_deg")
            cache = row.get("sim_cache_current_caked_deg")
            if (
                legacy is not None
                and cache is not None
                and visual is not None
                and _diag_pair_close(legacy, cache, tol=1.0e-3)
                and not _diag_pair_close(legacy, visual, tol=1.0e-3)
            ):
                failures.append(
                    {
                        "conclusion": "legacy_sim_caked_uses_cache_current",
                        "first_bad_event": event,
                        "first_bad_branch": branch,
                        "first_bad_field": "legacy_sim_caked_deg",
                        "expected_source": "sim_visual_caked_deg",
                        "actual_source": "sim_cache_current_caked_deg",
                        "explanation": "legacy sim caked matched cache/current instead of visual source",
                    }
                )

    print("\nlive_source_ledger_target")
    print(f"q_group_key={_QR_PICKER_TARGET_Q_GROUP_KEY} hkl={_QR_PICKER_TARGET_HKL}")

    _append_event("startup_simulation_ready", detector_targets)

    detector_select_point = _diag_detector_display_point(detector_targets[0])
    assert detector_select_point is not None
    detector_session, detector_select_statuses = _diag_live_toggle(
        detector_cache,
        detector_select_point,
        display_background=display_background,
        use_caked_space=False,
        profile_cache=profile_cache,
    )
    frozen_detector_map = _diag_print_frozen_visual_branch_map(
        detector_session,
        visual_map,
        "detector_mode_qr_selection",
    )
    if 1 not in frozen_detector_map:
        print("first_failure=visual_branch_map_only_freezes_clicked_seed")
        failures.append(
            {
                "conclusion": "visual_branch_map_missing_non_clicked_branch",
                "first_bad_event": "detector_mode_qr_selection",
                "first_bad_branch": 1,
                "first_bad_field": "pending_visual_branch_map",
                "expected_source": "both_visual_branches",
                "actual_source": "clicked_seed_only",
                "explanation": "branch 1 missing after Qr simulation selection",
            }
        )
    _append_event("detector_mode_qr_selection", frozen_detector_map)

    detector_session, detector_branch0_entries, _detector_place0_state = _diag_live_place(
        detector_session,
        _diag_detector_display_point(detector_targets[0]),
        cache_data=detector_cache,
        display_background=display_background,
        use_caked_space=False,
        callbacks=detector_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    _append_event("detector_mode_placement_branch_0", _diag_branch_map(detector_branch0_entries))

    detector_session, detector_saved_entries, _detector_place1_state = _diag_live_place(
        detector_session,
        _diag_detector_display_point(detector_targets[1]),
        cache_data=detector_cache,
        display_background=display_background,
        use_caked_space=False,
        callbacks=detector_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    detector_saved_map = _diag_branch_map(detector_saved_entries)
    _append_event("detector_mode_placement_branch_1", detector_saved_map)

    direct_checks = _diag_direct_detector_origin_checks(
        context,
        caked_callbacks,
        detector_saved_entries,
        visual_map,
    )
    for check in direct_checks:
        if check["status"] == "ok" and check["comparison"] == "missing_trace":
            print("first_failure=trace_uses_wrong_field_or_double_transforms_native")
            failures.append(
                {
                    "conclusion": "detector_to_caked_trace_uses_wrong_or_double_transformed_native",
                    "first_bad_event": "detector_to_caked_switch",
                    "first_bad_branch": check["branch"],
                    "first_bad_field": check["field"],
                    "expected_source": "direct_native_detector_coords_to_caked",
                    "actual_source": "trace_unavailable",
                    "explanation": "direct native-to-caked succeeds but trace lacks caked result",
                }
            )

    detector_to_caked_entries = _diag_refresh_entries(caked_callbacks, detector_saved_entries)
    _append_event("detector_to_caked_switch", _diag_branch_map(detector_to_caked_entries))

    _append_event("clear", {})

    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    caked_session, _caked_select_statuses = _diag_live_toggle(
        caked_cache,
        caked_select_point,
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )
    caked_select_map = _diag_branch_map(caked_session.get("group_entries", []))
    if 1 not in caked_select_map:
        print("first_failure=visual_branch_map_only_freezes_clicked_seed")
        failures.append(
            {
                "conclusion": "visual_branch_map_missing_non_clicked_branch",
                "first_bad_event": "caked_mode_qr_selection",
                "first_bad_branch": 1,
                "first_bad_field": "pending_visual_branch_map",
                "expected_source": "both_visual_branches",
                "actual_source": "clicked_seed_only",
                "explanation": "branch 1 missing after caked Qr simulation selection",
            }
        )
    _append_event("caked_mode_qr_selection", caked_select_map)

    caked_click0 = _diag_target_caked(detector_targets[0])
    assert caked_click0 is not None
    preview0 = _diag_live_preview(
        caked_session,
        caked_click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    preview_check0 = _diag_make_preview_check(
        "caked_mode_placement_branch_0",
        0,
        (float(preview0["refined_col"]), float(preview0["refined_row"])),
        visual_map,
        cache_current,
        preview0,
    )
    preview_checks.append(preview_check0)
    if preview_check0["preview_distance_matches"] == "sim_cache_current":
        print("preview_distance_matches=sim_cache_current")
        failures.append(
            {
                "conclusion": "preview_distance_uses_cache_current",
                "first_bad_event": "caked_mode_placement_branch_0",
                "first_bad_branch": 0,
                "first_bad_field": "preview_distance_value",
                "expected_source": "sim_visual_caked_deg",
                "actual_source": "sim_cache_current_caked_deg",
                "explanation": "preview distance matches cache/current caked point, not visual caked point",
            }
        )
    caked_session, caked_branch0_entries, _caked_place0_state = _diag_live_place(
        caked_session,
        caked_click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    _append_event(
        "caked_mode_placement_branch_0",
        _diag_branch_map(caked_branch0_entries),
        previews={0: preview_check0},
    )

    caked_click1 = _diag_target_caked(detector_targets[1])
    assert caked_click1 is not None
    preview1 = _diag_live_preview(
        caked_session,
        caked_click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    preview_check1 = _diag_make_preview_check(
        "caked_mode_placement_branch_1",
        1,
        (float(preview1["refined_col"]), float(preview1["refined_row"])),
        visual_map,
        cache_current,
        preview1,
    )
    preview_checks.append(preview_check1)
    if preview_check1["preview_distance_matches"] == "sim_cache_current":
        print("preview_distance_matches=sim_cache_current")
        failures.append(
            {
                "conclusion": "preview_distance_uses_cache_current",
                "first_bad_event": "caked_mode_placement_branch_1",
                "first_bad_branch": 1,
                "first_bad_field": "preview_distance_value",
                "expected_source": "sim_visual_caked_deg",
                "actual_source": "sim_cache_current_caked_deg",
                "explanation": "preview distance matches cache/current caked point, not visual caked point",
            }
        )
    caked_session, caked_saved_entries, _caked_place1_state = _diag_live_place(
        caked_session,
        caked_click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    _append_event(
        "caked_mode_placement_branch_1",
        _diag_branch_map(caked_saved_entries),
        previews={1: preview_check1},
    )

    caked_to_detector_entries = _diag_refresh_entries(detector_callbacks, caked_saved_entries)
    _append_event("caked_to_detector_switch", _diag_branch_map(caked_to_detector_entries))

    _diag_print_source_ledger(ledger_rows)

    print("\npreview_distance_source_check")
    print(
        "event | branch | printed_preview_distance | visual_distance | "
        "cache_current_distance | preview_distance_matches"
    )
    for check in preview_checks:
        print(
            " | ".join(
                (
                    str(check["event"]),
                    str(check["branch"]),
                    _diag_float_text(check["printed_preview_distance"]),
                    _diag_float_text(check["visual_distance"]),
                    _diag_float_text(check["cache_current_distance"]),
                    str(check["preview_distance_matches"]),
                )
            )
        )

    direct_success = any(check["status"] == "ok" for check in direct_checks)
    branch1_frozen_exists = 1 in frozen_detector_map
    conclusion = _diag_conclusion(failures)
    print("\nsource_ledger_conclusion")
    for key in (
        "first_bad_event",
        "first_bad_branch",
        "first_bad_field",
        "expected_source",
        "actual_source",
        "explanation",
        "conclusion",
    ):
        print(f"{key}={conclusion[key]}")
    print(f"detector_native_to_caked_direct_success={direct_success}")
    print(f"frozen_visual_branch_1_exists={branch1_frozen_exists}")
    print(f"detector_select_status={detector_select_statuses[-1] if detector_select_statuses else '<none>'}")

    assert ledger_rows
    assert preview_checks
    assert direct_success
    assert branch1_frozen_exists


def test_minus_1_0_10_live_caked_visual_source_ledger(tmp_path, monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    assert set(caked_targets) == {0, 1}
    visual_map = _diag_visual_map(detector_targets)
    poison_by_branch = {
        0: (42.276621, -43.250000),
        1: (36.063607, -116.750000),
    }
    caked_cache["grouped_candidates"] = _diag_poison_visual_grouped_candidates(
        caked_grouped,
        poison_by_branch,
    )

    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    caked_session, caked_select_statuses = _diag_live_toggle(
        caked_cache,
        caked_select_point,
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )

    manual_state = SimpleNamespace(pick_session=dict(caked_session))
    monkeypatch.setattr(runtime_session, "geometry_manual_state", manual_state, raising=False)
    monkeypatch.setattr(runtime_session.background_runtime_state, "current_background_index", 0)
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pick_session",
        lambda value: setattr(manual_state, "pick_session", dict(value)) or manual_state.pick_session,
    )
    monkeypatch.setattr(runtime_session.simulation_runtime_state, "profile_cache", profile_cache)
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: caked_background,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_current_geometry_fit_params", lambda: {}, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: dict(caked_cache),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pick_uses_caked_space",
        lambda: True,
        raising=False,
    )

    refreshed_session = runtime_session._refresh_geometry_manual_pick_session()
    pending_map = _diag_branch_map(refreshed_session.get("group_entries", []))
    assert set(pending_map) == {0, 1}
    first_bad_event = "<none>"
    wrong_source = "<none>"
    expected_visual = None
    actual_visual = None
    for branch in (0, 1):
        expected_visual = visual_map[branch]["caked"]
        actual_visual, _source = _diag_sim_visual_caked(pending_map[branch])
        poison = poison_by_branch[branch]
        if _diag_pair_close(actual_visual, poison, tol=1.0e-3):
            first_bad_event = "pending_visual_map_built_from_cache_current"
            wrong_source = "cache_current"
            break
        assert _diag_pair_close(actual_visual, expected_visual, tol=1.0)
        assert not _diag_pair_close(actual_visual, poison, tol=1.0)

    click0 = _diag_target_caked(detector_targets[0])
    assert click0 is not None
    preview0 = _diag_live_preview(
        refreshed_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    pending0_visual, _source0 = _diag_sim_visual_caked(pending_map[0])
    preview0_visual = preview0["status_sim_visual_caked_deg"]
    if not _diag_pair_close(preview0_visual, pending0_visual, tol=1.0e-3):
        first_bad_event = "preview_reads_cache_current_instead_of_pending_visual"
        wrong_source = "cache_current"
        expected_visual = pending0_visual
        actual_visual = preview0_visual
    assert _diag_pair_close(preview0_visual, pending0_visual, tol=1.0e-3)
    assert float(preview0["status_distance_value"]) < 2.0

    next_session, branch0_entries, place0_state = _diag_live_place(
        refreshed_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    branch0 = _diag_branch_map(branch0_entries)[0]
    branch0_visual, _source = _diag_sim_visual_caked(branch0)
    if not _diag_pair_close(branch0_visual, pending0_visual, tol=1.0e-3):
        first_bad_event = "placement_overwrites_pending_visual_with_cache_current"
        wrong_source = "cache_current"
        expected_visual = pending0_visual
        actual_visual = branch0_visual
    assert _diag_pair_close(branch0_visual, pending0_visual, tol=1.0e-3)

    click1 = _diag_target_caked(detector_targets[1])
    assert click1 is not None
    preview1 = _diag_live_preview(
        next_session,
        click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    _final_session, saved_entries, _place1_state = _diag_live_place(
        next_session,
        click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    saved_map = _diag_branch_map(saved_entries)
    assert set(saved_map) == {0, 1}
    assert float(preview1["status_distance_value"]) < 2.0
    assert all(float(entry["status_distance_value"]) < 2.0 for entry in saved_map.values())

    before_status = (
        "Manual pick preview ... tagged sim [-1,0,10] (78.85 deg) "
        "status_sim_visual_caked_deg=(42.276621,-43.250000)"
    )
    after_status = str(preview0["message"])
    print("\nlive_caked_visual_source_ledger_result")
    print(f"selection_status={caked_select_statuses[-1]}")
    print(f"first_bad_event={first_bad_event}")
    print(f"expected visual caked value={_diag_point_text(expected_visual)}")
    print(f"actual visual caked value={_diag_point_text(actual_visual)}")
    print(f"source that supplied wrong value={wrong_source}")
    print(f"before_live_like_status={before_status}")
    print(f"after_live_like_status={after_status}")
    print(f"branch0_preview_status_sim_visual_caked_deg={_diag_point_text(preview0_visual)}")
    print(
        "branch1_preview_status_sim_visual_caked_deg="
        f"{_diag_point_text(preview1['status_sim_visual_caked_deg'])}"
    )

    assert first_bad_event == "<none>"
    assert wrong_source == "<none>"
    assert "78.85 deg" not in after_status
    assert "status_distance_source=sim_visual_caked_deg" in after_status


def _diag_minus_1_0_10_caked_refresh_probe(tmp_path, monkeypatch):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    assert set(caked_targets) == {0, 1}
    visual_map = _diag_visual_map(detector_targets)
    cache_current = _diag_cache_current_map(context, detector_targets, caked_targets)
    poison_by_branch = {
        0: (42.3514, -42.2500),
        1: (36.2090, -117.7514),
    }
    caked_cache["grouped_candidates"] = _diag_poison_visual_grouped_candidates(
        caked_grouped,
        poison_by_branch,
    )

    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    caked_session, select_statuses = _diag_live_toggle(
        caked_cache,
        caked_select_point,
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )
    before_map = _diag_branch_map(caked_session.get("group_entries", []))
    refreshed_session, remaining, refresh_sources = _diag_runtime_refresh_pick_session(
        runtime_session,
        monkeypatch,
        session=caked_session,
        cache_data=caked_cache,
        background_image=caked_background,
        profile_cache=profile_cache,
        use_caked_space=True,
    )
    after_map = _diag_branch_map(refreshed_session.get("group_entries", []))

    click0 = _diag_target_caked(detector_targets[0])
    click1 = _diag_target_caked(detector_targets[1])
    assert click0 is not None
    assert click1 is not None
    preview0 = _diag_live_preview(
        refreshed_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    next_session, branch0_entries, place0_state = _diag_live_place(
        refreshed_session,
        click0,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    preview1 = _diag_live_preview(
        next_session,
        click1,
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=caked_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    check0 = _diag_make_preview_check(
        "caked_session_refresh_branch_0",
        0,
        (float(preview0["refined_col"]), float(preview0["refined_row"])),
        visual_map,
        cache_current,
        preview0,
    )
    check1 = _diag_make_preview_check(
        "caked_session_refresh_branch_1",
        1,
        (float(preview1["refined_col"]), float(preview1["refined_row"])),
        visual_map,
        cache_current,
        preview1,
    )
    return {
        "select_statuses": select_statuses,
        "before_map": before_map,
        "after_map": after_map,
        "remaining": remaining,
        "refresh_sources": refresh_sources,
        "poison_by_branch": poison_by_branch,
        "preview0": preview0,
        "preview1": preview1,
        "check0": check0,
        "check1": check1,
        "next_session": next_session,
        "branch0_entries": branch0_entries,
        "place0_state": place0_state,
        "visual_map": visual_map,
        "cache_current": cache_current,
        "caked_cache": caked_cache,
        "detector_grouped": detector_grouped,
        "caked_grouped": caked_grouped,
        "caked_background": caked_background,
        "display_background": display_background,
        "profile_cache": profile_cache,
        "runtime_session": runtime_session,
    }


def test_minus_1_0_10_caked_session_refresh_preserves_visual_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    probe = _diag_minus_1_0_10_caked_refresh_probe(tmp_path, monkeypatch)
    before_map = probe["before_map"]
    after_map = probe["after_map"]
    incoming_source = probe["refresh_sources"][-1]
    print("\ncaked_session_refresh_preserves_visual_candidates")
    print("before_caked_refresh_source=caked_qr_projection_grouped_candidates")
    print(f"after_caked_refresh_source={incoming_source}")
    for branch in (0, 1):
        before_visual, _before_source = _diag_sim_visual_caked(before_map[branch])
        after_visual, _after_source = _diag_sim_visual_caked(after_map[branch])
        poison = probe["poison_by_branch"][branch]
        print(f"branch{branch}_visual_before_refresh={_diag_point_text(before_visual)}")
        print(f"branch{branch}_visual_after_refresh={_diag_point_text(after_visual)}")
        assert _diag_pair_close(after_visual, before_visual, tol=1.0e-6)
        assert not _diag_pair_close(after_visual, poison, tol=1.0e-3)
    _diag_assert_caked_session_refresh_preserved_visual_source(
        before_map,
        after_map,
        incoming_candidate_source=incoming_source,
    )
    assert set(after_map) == {0, 1}
    assert probe["refresh_sources"] == ["caked_qr_projection_grouped_candidates"]
    for check in (probe["check0"], probe["check1"]):
        print(
            "branch={branch} preview_distance_after_refresh={distance} "
            "source={source} match={match}".format(
                branch=check["branch"],
                distance=_diag_float_text(check["printed_preview_distance"]),
                source=check["preview_distance_source"],
                match=check["preview_distance_matches"],
            )
        )
        assert check["preview_distance_source"] == "sim_visual_caked_deg"
        assert check["preview_distance_matches"] == "sim_visual"
        assert float(check["printed_preview_distance"]) < 2.0
        assert float(check["printed_preview_distance"]) < 10.0


def test_minus_1_0_10_caked_refresh_uses_caked_projection_candidates_not_grouped_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    probe = _diag_minus_1_0_10_caked_refresh_probe(tmp_path, monkeypatch)
    assert probe["refresh_sources"] == ["caked_qr_projection_grouped_candidates"]

    runtime_session = probe["runtime_session"]
    base_session = {
        "background_index": 0,
        "group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "group_entries": list(probe["after_map"].values()),
        "pending_entries": [],
        "target_count": 2,
    }
    detector_session, _remaining, detector_sources = _diag_runtime_refresh_pick_session(
        runtime_session,
        monkeypatch,
        session=base_session,
        cache_data=probe["caked_cache"],
        background_image=probe["display_background"],
        profile_cache=probe["profile_cache"],
        use_caked_space=False,
        detector_grouped=probe["detector_grouped"],
    )
    assert detector_session.get("group_entries")
    assert detector_sources == ["detector_grouped_candidates_from_cache"]

    empty_caked_cache = dict(probe["caked_cache"])
    empty_caked_cache["caked_qr_projection_grouped_candidates"] = {}
    preserved_session, _remaining, empty_sources = _diag_runtime_refresh_pick_session(
        runtime_session,
        monkeypatch,
        session=base_session,
        cache_data=empty_caked_cache,
        background_image=probe["caked_background"],
        profile_cache=probe["profile_cache"],
        use_caked_space=True,
    )
    assert empty_sources == []
    assert preserved_session["group_entries"] == base_session["group_entries"]
    print("\ncaked_refresh_candidate_source_selection")
    print("caked_mode_used=caked_qr_projection_grouped_candidates")
    print("detector_mode_used=detector_grouped_candidates_from_cache")
    print("caked_empty_projection_preserved_existing_group_entries=yes")


def test_minus_1_0_10_caked_preview_after_refresh_uses_preserved_visual_source(
    tmp_path,
    monkeypatch,
) -> None:
    probe = _diag_minus_1_0_10_caked_refresh_probe(tmp_path, monkeypatch)
    print("\ncaked_preview_after_refresh_uses_preserved_visual_source")
    for branch, preview, check, bounds in (
        (0, probe["preview0"], probe["check0"], (0.6, 1.05)),
        (1, probe["preview1"], probe["check1"], (0.75, 1.05)),
    ):
        message = str(preview["message"])
        visual = preview["status_sim_visual_caked_deg"]
        cache = preview["status_sim_cache_current_caked_deg"]
        print(
            "branch={branch} preview_distance_after_refresh={distance} "
            "status_distance_source={source} visual={visual} cache_current={cache}".format(
                branch=branch,
                distance=_diag_float_text(preview["status_distance_value"]),
                source=preview["status_distance_source"],
                visual=_diag_point_text(visual),
                cache=_diag_point_text(cache),
            )
        )
        assert bounds[0] <= float(preview["status_distance_value"]) <= bounds[1]
        assert "77." not in message
        assert "78." not in message
        assert "79." not in message
        assert "80." not in message
        assert preview["status_distance_source"] == "sim_visual_caked_deg"
        assert "status_distance_source=sim_visual_caked_deg" in message
        assert check["preview_distance_matches"] == "sim_visual"
        assert not _diag_pair_close(visual, cache, tol=1.0e-3)


def test_manual_trace_never_routes_caked_angles_to_detector_px(tmp_path) -> None:
    bad_points = (
        (39.827, 35.250),
        (42.351, -42.250),
        (36.209, -117.751),
    )
    print("\nmanual_trace_never_routes_caked_angles_to_detector_px")
    for point in bad_points:
        entry = {
            "_caked_qr_projection_cache": True,
            "display_frame": "caked_display",
            "display_col": point[0],
            "display_row": point[1],
            "caked_x": point[0],
            "caked_y": point[1],
            "raw_caked_x": point[0],
            "raw_caked_y": point[1],
            "two_theta_deg": point[0],
            "phi_deg": point[1],
            "sim_visual_deg": point,
            "sim_visual_source": "sim_visual_caked_deg",
        }
        detector_display, detector_source = _diag_observed_detector_display(entry)
        observed_caked, caked_source = _diag_observed_caked(entry)
        print(
            "point={point} detector_px={detector} detector_source={detector_source} "
            "caked_deg={caked} caked_source={caked_source}".format(
                point=_diag_point_text(point),
                detector=_diag_point_text(detector_display),
                detector_source=detector_source,
                caked=_diag_point_text(observed_caked),
                caked_source=caked_source,
            )
        )
        assert detector_display is None
        assert detector_source == "<unavailable reason=no detector back-projection>"
        assert _diag_pair_close(observed_caked, point, tol=1.0e-6)


def test_manual_trace_never_routes_caked_row_generic_xy_to_detector_px() -> None:
    entry = {
        "_caked_qr_projection_cache": True,
        "display_frame": "caked_display",
        "x": 42.3514,
        "y": -42.25,
        "simulated_x": 36.209,
        "simulated_y": -117.7514,
        "sim_col_raw": 42.3514,
        "sim_row_raw": -42.25,
        "sim_col": 36.209,
        "sim_row": -117.7514,
        "two_theta_deg": 40.237466,
        "phi_deg": 36.527645,
    }
    detector_display, detector_source = _diag_observed_detector_display(entry)
    print("\nmanual_trace_never_routes_caked_row_generic_xy_to_detector_px")
    print(
        "generic_xy_detector_px={detector} detector_source={source}".format(
            detector=_diag_point_text(detector_display),
            source=detector_source,
        )
    )
    assert detector_display is None
    assert detector_source == "<unavailable reason=no detector back-projection>"


def test_caked_refresh_identity_keys_are_scoped_to_branch_identity() -> None:
    keys = mg._geometry_manual_refresh_branch_identity_keys(
        {
            "q_group_key": ("q_group", "primary", 1, 10),
            "hkl": (-1, 0, 10),
            "source_table_index": 160,
            "source_row_index": 24,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "branch_id": "branch-0",
        }
    )
    print("\ncaked_refresh_identity_keys_are_scoped_to_branch_identity")
    print(f"identity_keys={keys}")
    assert keys
    assert str(keys[0][0]).startswith("q_group_hkl_source_row")
    assert not any(key and key[0] == "candidate_source" for key in keys)
    assert all(key[0] != "q_group_hkl_source_branch" or key[1] for key in keys)


def test_caked_refresh_preserves_visual_by_specific_source_row_before_broad_hkl() -> None:
    group_key = ("q_group", "primary", 1, 10)
    first_visual = (40.0, 10.0)
    second_visual = (41.0, 20.0)
    incoming_visual = (42.0, 30.0)
    shared = {
        "q_group_key": group_key,
        "hkl": (-1, 0, 10),
        "source_branch_index": 0,
        "source_peak_index": 0,
        "_caked_qr_projection_cache": True,
        "display_frame": "caked_display",
    }
    session = {
        "background_index": 0,
        "group_key": group_key,
        "group_entries": [
            {
                **shared,
                "source_table_index": 160,
                "source_row_index": 24,
                "sim_visual_deg": first_visual,
                "sim_caked": first_visual,
            },
            {
                **shared,
                "source_table_index": 167,
                "source_row_index": 24,
                "sim_visual_deg": second_visual,
                "sim_caked": second_visual,
            },
        ],
        "pending_entries": [],
        "target_count": 2,
    }
    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        session,
        grouped_candidates={
            group_key: [
                {
                    **shared,
                    "source_table_index": 167,
                    "source_row_index": 24,
                    "sim_visual_deg": incoming_visual,
                    "sim_caked": incoming_visual,
                }
            ]
        },
    )
    refreshed_entry = refreshed["group_entries"][0]
    print("\ncaked_refresh_preserves_visual_by_specific_source_row_before_broad_hkl")
    print(f"preserved_visual={_diag_point_text(refreshed_entry.get('sim_visual_deg'))}")
    assert _diag_pair_close(refreshed_entry["sim_visual_deg"], second_visual)
    assert not _diag_pair_close(refreshed_entry["sim_visual_deg"], first_visual)


def test_caked_refresh_preserves_visual_but_not_cache_current_simulated_angles() -> None:
    group_key = ("q_group", "primary", 1, 10)
    session = {
        "background_index": 0,
        "group_key": group_key,
        "group_entries": [
            {
                "q_group_key": group_key,
                "hkl": (-1, 0, 10),
                "source_table_index": 160,
                "source_row_index": 24,
                "source_branch_index": 0,
                "source_peak_index": 0,
                "sim_visual_deg": (40.237466, 36.527645),
                "sim_caked": (40.237466, 36.527645),
                "simulated_two_theta_deg": 999.0,
                "simulated_phi_deg": 888.0,
                "_caked_qr_projection_cache": True,
                "display_frame": "caked_display",
            }
        ],
        "pending_entries": [],
        "target_count": 1,
    }
    incoming = {
        "q_group_key": group_key,
        "hkl": (-1, 0, 10),
        "source_table_index": 160,
        "source_row_index": 24,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_visual_deg": (42.3514, -42.25),
        "sim_caked": (42.3514, -42.25),
        "simulated_two_theta_deg": 40.142509,
        "simulated_phi_deg": 35.566836,
        "_caked_qr_projection_cache": True,
        "display_frame": "caked_display",
    }
    refreshed = mg.refresh_geometry_manual_pick_session_candidates(
        session,
        grouped_candidates={group_key: [incoming]},
    )
    refreshed_entry = refreshed["group_entries"][0]
    print("\ncaked_refresh_preserves_visual_but_not_cache_current_simulated_angles")
    print(f"sim_visual_deg={_diag_point_text(refreshed_entry.get('sim_visual_deg'))}")
    print(
        "simulated_angles=({tth:.6f},{phi:.6f})".format(
            tth=float(refreshed_entry["simulated_two_theta_deg"]),
            phi=float(refreshed_entry["simulated_phi_deg"]),
        )
    )
    assert _diag_pair_close(refreshed_entry["sim_visual_deg"], (40.237466, 36.527645))
    assert refreshed_entry["simulated_two_theta_deg"] == 40.142509
    assert refreshed_entry["simulated_phi_deg"] == 35.566836


def test_manual_trace_no_caked_angles_to_detector_px(tmp_path) -> None:
    test_manual_trace_never_routes_caked_angles_to_detector_px(tmp_path)


def test_detector_to_caked_projection_does_not_print_caked_angles_as_detector_px() -> None:
    entry = {
        "_caked_qr_projection_cache": True,
        "display_frame": "caked_display",
        "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "hkl": _QR_PICKER_TARGET_HKL,
        "source_branch_index": 0,
        "caked_x": 42.351,
        "caked_y": -42.250,
        "raw_caked_x": 36.209,
        "raw_caked_y": -117.751,
        "two_theta_deg": 39.827,
        "phi_deg": 35.250,
        "sim_visual_caked_deg": (39.827, 35.250),
    }
    with contextlib.redirect_stdout(io.StringIO()) as out:
        _diag_print_projection_block(
            "[ra-sim] detector -> caked Qr/Qz projection",
            [entry],
            {0: {"caked": (39.827, 35.250)}},
            "manual-angle-guard",
            emitter="test_detector_to_caked_projection_does_not_print_caked_angles_as_detector_px",
            event="detector_to_caked_projection",
        )
    text = out.getvalue()
    print("\ndetector_to_caked_projection_does_not_print_caked_angles_as_detector_px")
    print(text.strip())
    assert (
        "geometry_detector_display_px=<unavailable reason=no detector back-projection>"
        in text
    )
    bad_values = (
        "(42.351, -42.250)",
        "(36.209, -117.751)",
        "(39.827, 35.250)",
        "(42.351,-42.250)",
        "(36.209,-117.751)",
        "(39.827,35.250)",
    )
    detector_fields = (
        "geometry_detector_display_px",
        "geometry_detector_px",
        "sim_detector_display_px",
        "sim_detector_px",
    )
    for field in detector_fields:
        for value in bad_values:
            assert f"{field}={value}" not in text


def test_detector_to_caked_projection_no_caked_angles_as_detector_px() -> None:
    test_detector_to_caked_projection_does_not_print_caked_angles_as_detector_px()


def test_headless_replay_uses_same_caked_picker_builder_as_live(
    tmp_path,
    monkeypatch,
) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    calls = []
    original_builder = mg._geometry_manual_build_caked_qr_projection_cache
    marker_key = "live_caked_picker_builder_marker"

    def _mark_entry(entry):
        marked = dict(entry)
        marked[marker_key] = "yes"
        return marked

    def _mark_grouped(grouped):
        return {
            key: [_mark_entry(entry) for entry in value if isinstance(entry, Mapping)]
            for key, value in (grouped or {}).items()
        }

    def _mark_lookup(lookup):
        marked = {}
        for key, bucket in (lookup or {}).items():
            bucket_entries = mg._geometry_manual_lookup_bucket_entries(bucket)
            if len(bucket_entries) == 1:
                marked[key] = _mark_entry(bucket_entries[0])
            elif bucket_entries:
                marked[key] = [_mark_entry(entry) for entry in bucket_entries]
        return marked

    def _spy_builder(*args, **kwargs):
        calls.append("live_caked_picker_builder")
        entries, grouped, lookup = original_builder(*args, **kwargs)
        return (
            [_mark_entry(entry) for entry in entries],
            _mark_grouped(grouped),
            _mark_lookup(lookup),
        )

    monkeypatch.setattr(mg, "_geometry_manual_build_caked_qr_projection_cache", _spy_builder)
    callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    caked_select_point = _diag_target_caked(caked_targets[0])
    assert caked_select_point is not None
    statuses = []
    workflow, state = _diag_headless_workflow(
        cache_data=caked_cache,
        display_background=caked_background,
        use_caked_space=True,
        callbacks=callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
        status_sink=statuses,
    )
    workflow.toggle_selection_at(*caked_select_point)
    replay_entries = [
        entry
        for entry in state["pick_session"].get("group_entries", [])
        if isinstance(entry, Mapping)
        and _diag_q_group_key(entry) == _QR_PICKER_TARGET_Q_GROUP_KEY
        and _diag_hkl(entry) == _QR_PICKER_TARGET_HKL
    ]
    replay_used_builder_rows = bool(replay_entries) and all(
        entry.get(marker_key) == "yes" for entry in replay_entries
    )
    bypassed = not bool(calls)
    if bypassed or not replay_used_builder_rows:
        print("headless_replay_bypasses_live_caked_candidate_builder=yes")
    else:
        print("headless_replay_bypasses_live_caked_candidate_builder=no")
    print(f"headless_replay_builder_calls={len(calls)}")
    print(
        "headless_replay_caked_picker_candidate_count="
        f"{len(_diag_flatten_grouped(caked_grouped))}"
    )
    print(f"headless_replay_selected_builder_row_count={len(replay_entries)}")

    assert not bypassed
    assert calls
    assert caked_cache["caked_qr_projection_grouped_candidates"]
    assert caked_targets.keys() >= {0, 1}
    assert replay_entries
    assert replay_used_builder_rows


def _diag_status_capture(label, sink):
    def _status(message):
        text = f"[geometry] {message}"
        sink.append(text)
        print(text)

    return _status


def _diag_cmd_stamp(
    run_id,
    *,
    emitter,
    event,
    branch=None,
    actual_source=None,
    expected_source=None,
):
    return mg.geometry_manual_cmd_provenance_text(
        run_id=run_id,
        emitter=emitter,
        event=event,
        branch=branch,
        actual_source=actual_source,
        expected_source=expected_source,
    )


def _diag_headless_workflow(
    *,
    cache_data,
    display_background,
    use_caked_space,
    callbacks,
    radial_axis,
    azimuth_axis,
    profile_cache,
    status_sink,
):
    state = {"pick_session": {}, "pairs": [], "previews": []}

    def _set_pick_session(value):
        state["pick_session"] = dict(value) if isinstance(value, Mapping) else {}

    def _set_pairs_for_index(_idx, entries):
        state["pairs"] = [dict(entry) for entry in (entries or []) if isinstance(entry, Mapping)]
        return list(state["pairs"])

    def _show_preview(raw_col, raw_row, refined_col, refined_row, **kwargs):
        state["previews"].append(
            {
                "raw_col": float(raw_col),
                "raw_row": float(raw_row),
                "refined_col": float(refined_col),
                "refined_row": float(refined_row),
                **dict(kwargs),
            }
        )

    def _remaining_candidates():
        return mg.geometry_manual_unassigned_group_candidates(
            state["pick_session"],
            current_background_index=0,
        )

    workflow = mg.make_runtime_geometry_manual_callbacks(
        background_visible=True,
        current_background_index=0,
        current_background_image=display_background,
        pick_session=lambda: state["pick_session"],
        build_initial_pairs_display=lambda *_args, **_kwargs: ([], []),
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda **_kwargs: None,
        draw_initial_geometry_pairs_overlay=lambda *_args, **_kwargs: None,
        update_button_label=lambda: None,
        set_background_file_status_text=lambda: None,
        pair_group_count=lambda _idx: len(state["pairs"]),
        set_status_text=_diag_status_capture("manual_geometry", status_sink),
        get_cache_data=lambda **_kwargs: dict(cache_data),
        set_pairs_for_index=_set_pairs_for_index,
        pairs_for_index=lambda _idx: list(state["pairs"]),
        set_pick_session=_set_pick_session,
        restore_view=lambda **_kwargs: None,
        clear_preview_artists=lambda **_kwargs: None,
        push_undo_state=lambda: None,
        listed_q_group_entries=lambda: [{"key": _QR_PICKER_TARGET_Q_GROUP_KEY}],
        format_q_group_line=lambda _entry: f"group={_QR_PICKER_TARGET_Q_GROUP_KEY}",
        use_caked_space=bool(use_caked_space),
        pick_search_window_px=50.0,
        sync_peak_selection_state=lambda: None,
        refine_preview_point=_diag_build_refine_preview(
            use_caked_space,
            radial_axis,
            azimuth_axis,
        ),
        remaining_candidates=_remaining_candidates,
        preview_due=lambda _col, _row: True,
        caked_angles_to_background_display_coords=getattr(
            callbacks,
            "caked_angles_to_background_display_coords",
            None,
        ),
        last_caked_radial_values=radial_axis,
        last_caked_azimuth_values=azimuth_axis,
        background_display_to_native_detector_coords=getattr(
            callbacks,
            "background_display_to_native_detector_coords",
            None,
        ),
        native_detector_coords_to_caked_display_coords=getattr(
            callbacks,
            "native_detector_coords_to_caked_display_coords",
            None,
        ),
        refine_saved_pair_entry=getattr(callbacks, "refresh_entry_geometry", None),
        show_preview=_show_preview,
        profile_cache=profile_cache,
    )
    return workflow, state


def _diag_headless_detector_picker_resolution(context, picker_rows, target_by_branch, run_id):
    payload = ps.build_hkl_pick_simulation_point_payload(picker_rows)
    runtime_state = SimpleNamespace(
        peak_records=[dict(row) for row in picker_rows],
        peak_positions=[_diag_detector_display_point(row) for row in picker_rows],
        peak_positions_filtered=False,
        profile_cache=context["rows"]["profile_cache"],
    )
    resolved = {}
    print(
        "[ra-sim] Qr/Qz detector simulation selection "
        + _diag_cmd_stamp(
            run_id,
            emitter="_diag_headless_detector_picker_resolution",
            event="detector_simulation_selection",
        )
    )
    for branch in (0, 1):
        click = _diag_detector_display_point(target_by_branch[branch])
        idx, selected, distance, within = ps.find_peak_record_for_canvas_click(
            runtime_state,
            float(click[0]),
            float(click[1]),
            ensure_peak_overlay_data=lambda force=False: None,
            max_axis_distance_px=25.0,
            simulation_point_candidates=payload,
            use_caked_display=False,
        )
        resolved[branch] = dict(selected) if isinstance(selected, Mapping) else {}
        print(
            "branch={branch} click_px={click} picker_index={idx} distance_px={dist} "
            "within={within} q_group_key={q_group} hkl={hkl} table={table} row={row} "
            "source_branch={source_branch} {stamp}".format(
                branch=branch,
                click=_diag_point_text(click),
                idx=idx,
                dist=_diag_float_text(distance),
                within=within,
                q_group=_diag_q_group_key(selected),
                hkl=_diag_hkl(selected),
                table=selected.get("source_table_index") if isinstance(selected, Mapping) else "<none>",
                row=selected.get("source_row_index") if isinstance(selected, Mapping) else "<none>",
                source_branch=(
                    selected.get("source_branch_index")
                    if isinstance(selected, Mapping)
                    else "<none>"
                ),
                stamp=_diag_cmd_stamp(
                    run_id,
                    emitter="_diag_headless_detector_picker_resolution",
                    event="detector_simulation_selection_row",
                    branch=branch,
                    actual_source="detector_picker_payload",
                    expected_source="sim_visual_detector_display_px",
                ),
            )
        )
    return resolved


def _diag_print_projection_block(title, entries, visual_map, run_id, *, emitter, event):
    print(
        title
        + " "
        + _diag_cmd_stamp(
            run_id,
            emitter=emitter,
            event=event,
        )
    )
    for entry in _diag_sorted(entries):
        branch = _diag_source_branch(entry)
        observed_display, _display_source = _diag_observed_detector_display(entry)
        observed_native, _native_source = _diag_observed_detector_native(entry)
        observed_caked, _caked_source = _diag_observed_caked(entry)
        visual = visual_map.get(branch, {}) if branch is not None else {}
        print(
            "branch={branch} q_group_key={q_group} hkl={hkl} "
            "geometry_detector_display_px={display} geometry_detector_native_px={native} "
            "raw_caked_deg={raw_caked} geometry_caked_deg={geometry_caked} "
            "sim_caked_deg={sim_caked} {stamp}".format(
                branch=branch,
                q_group=_diag_q_group_key(entry),
                hkl=_diag_hkl(entry),
                display=_diag_point_text(displayed := observed_display, _display_source),
                native=_diag_point_text(observed_native, _native_source),
                raw_caked=_diag_point_text(_diag_finite_pair(entry, (("raw_caked_x", "raw_caked_y"),))),
                geometry_caked=_diag_point_text(observed_caked),
                sim_caked=_diag_point_text(visual.get("caked")),
                stamp=_diag_cmd_stamp(
                    run_id,
                    emitter=emitter,
                    event=f"{event}_row",
                    branch=branch,
                    actual_source="projection_refresh",
                    expected_source="sim_visual_caked_deg",
                ),
            )
        )
        del displayed


def _diag_detector_origin_to_caked_fixture(tmp_path):
    context, rows = _diag_startup_context_and_rows(tmp_path)
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_callbacks = context["projection_callbacks"]
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    assert set(_diag_target_rows_by_branch(caked_grouped)) == {0, 1}
    visual_map = _diag_visual_map(detector_targets)
    detector_clicks = {
        0: (1083.818, 1083.270),
        1: (1846.620, 1083.734),
    }
    session, _statuses = _diag_live_toggle(
        detector_cache,
        detector_clicks[0],
        display_background=display_background,
        use_caked_space=False,
        profile_cache=profile_cache,
    )
    run_id = session.get("manual_geometry_run_id")
    session, _branch0_entries, _state0 = _diag_live_place(
        session,
        detector_clicks[0],
        cache_data=detector_cache,
        display_background=display_background,
        use_caked_space=False,
        callbacks=detector_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    session, detector_saved, _state1 = _diag_live_place(
        session,
        detector_clicks[1],
        cache_data=detector_cache,
        display_background=display_background,
        use_caked_space=False,
        callbacks=detector_callbacks,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        profile_cache=profile_cache,
    )
    detector_to_caked = _diag_refresh_entries(caked_callbacks, detector_saved)
    return {
        "context": context,
        "rows": rows,
        "profile_cache": profile_cache,
        "detector_cache": detector_cache,
        "detector_targets": detector_targets,
        "detector_saved": detector_saved,
        "detector_to_caked": detector_to_caked,
        "visual_map": visual_map,
        "caked_callbacks": caked_callbacks,
        "run_id": run_id,
    }


def test_minus_1_0_10_detector_origin_direct_native_to_caked_succeeds(tmp_path) -> None:
    fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    callbacks = fixture["caked_callbacks"]
    expected = {
        0: ((1083.270, 1915.182), (40.10, 35.75)),
        1: ((1083.734, 1152.380), (40.75, -37.70)),
    }
    print("\ndetector_origin_direct_native_to_caked")
    for branch, (native, target) in expected.items():
        caked, status = _diag_direct_native_to_caked(callbacks, native)
        print(
            "branch={branch} native_px={native} direct_caked_deg={caked} "
            "expected_caked_deg={target} status={status}".format(
                branch=branch,
                native=_diag_point_text(native),
                caked=_diag_point_text(caked),
                target=_diag_point_text(target),
                status=status,
            )
        )
        assert status == "ok"
        assert caked is not None
        assert np.allclose(caked, target, atol=0.75, rtol=0.0)


def test_minus_1_0_10_detector_origin_sim_native_derived_from_display(tmp_path) -> None:
    fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    context = fixture["context"]
    entries = _diag_branch_map(fixture["detector_to_caked"])
    print("\ndetector_origin_sim_native_from_display")
    for branch in (0, 1):
        entry = entries[branch]
        display = mg._geometry_manual_tuple_point(entry, "sim_visual_detector_display_px")
        native = mg._geometry_manual_tuple_point(entry, "sim_visual_detector_native_px")
        existing = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_existing",
        )
        expected = _diag_display_to_native(context, display)
        confused = bool(
            existing is not None
            and display is not None
            and np.allclose(existing, display, atol=1.0e-6, rtol=0.0)
        )
        print(
            "branch={branch} sim_visual_detector_display_px={display} "
            "sim_visual_detector_native_existing={existing} "
            "sim_visual_display_to_native_px={expected} "
            "sim_visual_detector_native_px={native} "
            "display_native_confused_existing={confused} source={source}".format(
                branch=branch,
                display=_diag_point_text(display),
                existing=_diag_point_text(existing),
                expected=_diag_point_text(expected),
                native=_diag_point_text(native),
                confused="yes" if confused else "no",
                source=entry.get("sim_visual_detector_native_source", "<none>"),
            )
        )
        assert display is not None
        assert native is not None
        assert expected is not None
        assert np.allclose(native, expected, atol=1.0e-6, rtol=0.0)
        assert entry["sim_visual_detector_native_source"] == "display_to_native_callback"


def test_minus_1_0_10_detector_origin_to_caked_conversion_ledger(tmp_path) -> None:
    fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    context = fixture["context"]
    callbacks = fixture["caked_callbacks"]
    entries = _diag_branch_map(fixture["detector_to_caked"])
    print("\ndetector_origin_to_caked_conversion_ledger")
    for branch in (0, 1):
        entry = entries[branch]
        raw_display = mg._geometry_manual_tuple_point(entry, "raw_detector_display_px")
        raw_native = mg._geometry_manual_tuple_point(entry, "raw_detector_native_px")
        raw_display_to_native = _diag_display_to_native(context, raw_display)
        raw_native_to_caked, raw_status = _diag_direct_native_to_caked(
            callbacks,
            raw_native,
        )
        geometry_display = mg._geometry_manual_tuple_point(
            entry,
            "geometry_detector_display_px",
        )
        geometry_native = mg._geometry_manual_tuple_point(
            entry,
            "geometry_detector_native_px",
        )
        geometry_display_to_native = _diag_display_to_native(context, geometry_display)
        geometry_native_to_caked, geometry_status = _diag_direct_native_to_caked(
            callbacks,
            geometry_native,
        )
        sim_display = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_display_px",
        )
        sim_existing = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_existing",
        )
        sim_native = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_px",
        )
        sim_display_to_native = _diag_display_to_native(context, sim_display)
        sim_native_to_caked, sim_status = _diag_direct_native_to_caked(callbacks, sim_native)
        trace_raw = _diag_finite_pair(entry, (("raw_caked_x", "raw_caked_y"),))
        trace_geometry, _ = _diag_observed_caked(entry)
        trace_sim, _ = _diag_sim_visual_caked(entry)
        callback_available = callable(
            getattr(callbacks, "native_detector_coords_to_caked_display_coords", None)
        )
        projection_ready = bool(raw_status == geometry_status == sim_status == "ok")
        trace_deferred = "no"
        sim_existing_differs_from_canonical = bool(
            sim_existing is not None
            and sim_display_to_native is not None
            and not np.allclose(
                sim_existing,
                sim_display_to_native,
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        first_failure = "<none>"
        if not callback_available:
            first_failure = "trace_emitted_before_caked_projection_ready"
        elif geometry_native_to_caked is not None and trace_geometry is None:
            first_failure = "trace_not_using_saved_native_or_callback"
        elif sim_existing_differs_from_canonical:
            first_failure = "sim_native_display_confusion"
        print(
            "branch={branch} raw_detector_display_px={raw_display} "
            "raw_detector_native_px={raw_native} raw_display_to_native_px={raw_d2n} "
            "raw_native_to_caked_deg={raw_caked} "
            "geometry_detector_display_px={geometry_display} "
            "geometry_detector_native_px={geometry_native} "
            "geometry_display_to_native_px={geometry_d2n} "
            "geometry_native_to_caked_deg={geometry_caked} "
            "sim_visual_detector_display_px={sim_display} "
            "sim_visual_detector_native_existing={sim_existing} "
            "sim_visual_display_to_native_px={sim_d2n} "
            "sim_visual_native_to_caked_deg={sim_caked} "
            "trace_raw_caked_deg={trace_raw} "
            "trace_geometry_caked_deg={trace_geometry} "
            "trace_sim_caked_deg={trace_sim} "
            "caked_callback_available={callback_available} "
            "caked_projection_ready={projection_ready} "
            "trace_deferred_until_caked_ready={trace_deferred} "
            "trace_emitted_before_caked_ready=no first_failure={first_failure}".format(
                branch=branch,
                raw_display=_diag_point_text(raw_display),
                raw_native=_diag_point_text(raw_native),
                raw_d2n=_diag_point_text(raw_display_to_native),
                raw_caked=_diag_point_text(raw_native_to_caked),
                geometry_display=_diag_point_text(geometry_display),
                geometry_native=_diag_point_text(geometry_native),
                geometry_d2n=_diag_point_text(geometry_display_to_native),
                geometry_caked=_diag_point_text(geometry_native_to_caked),
                sim_display=_diag_point_text(sim_display),
                sim_existing=_diag_point_text(sim_existing),
                sim_d2n=_diag_point_text(sim_display_to_native),
                sim_caked=_diag_point_text(sim_native_to_caked),
                trace_raw=_diag_point_text(trace_raw),
                trace_geometry=_diag_point_text(trace_geometry),
                trace_sim=_diag_point_text(trace_sim),
                callback_available="yes" if callback_available else "no",
                projection_ready="yes" if projection_ready else "no",
                trace_deferred=trace_deferred,
                first_failure=first_failure,
            )
        )
        assert raw_native_to_caked is not None
        assert geometry_native_to_caked is not None
        assert sim_native_to_caked is not None
        assert trace_raw is not None
        assert trace_geometry is not None
        assert trace_sim is not None
        assert first_failure == "<none>"


def test_minus_1_0_10_detector_origin_to_caked_live_ledger(tmp_path) -> None:
    test_minus_1_0_10_detector_origin_to_caked_conversion_ledger(tmp_path)


def test_minus_1_0_10_detector_to_caked_trace_waits_for_caked_ready(
    monkeypatch,
    capsys,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    entry = {
        "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "hkl": _QR_PICKER_TARGET_HKL,
        "source_branch_index": 0,
        "manual_geometry_run_id": "manual-snapshot",
        "raw_detector_display_px": (10.0, 20.0),
        "raw_detector_native_px": (20.0, 40.0),
        "geometry_detector_display_px": (11.0, 21.0),
        "geometry_detector_native_px": (22.0, 42.0),
        "sim_visual_detector_display_px": (12.0, 22.0),
        "sim_visual_detector_native_px": (24.0, 44.0),
    }
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_saved_entries",
        lambda: [dict(entry)],
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_caked_display_coords",
        lambda col, row: (float(col) / 10.0, float(row) / 10.0),
    )
    states = [{"callback_available": False, "caked_projection_ready": False}]
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_projection_state",
        lambda: dict(states[-1]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_background_index",
        lambda: 0,
    )
    runtime_session.pending_detector_to_caked_manual_trace = None
    emitted = runtime_session._emit_or_defer_detector_to_caked_manual_trace("unit_before")
    before = capsys.readouterr().out
    assert emitted is False
    assert before == ""
    assert isinstance(runtime_session.pending_detector_to_caked_manual_trace, Mapping)
    assert runtime_session.pending_detector_to_caked_manual_trace["background_index"] == 0
    assert (
        runtime_session.pending_detector_to_caked_manual_trace["manual_geometry_run_id"]
        == "manual-snapshot"
    )
    assert (
        runtime_session.pending_detector_to_caked_manual_trace[
            "trace_deferred_until_caked_ready"
        ]
        is True
    )
    entry["source_branch_index"] = 99
    entry["raw_detector_native_px"] = (990.0, 990.0)
    entry["geometry_detector_native_px"] = (991.0, 991.0)
    entry["sim_visual_detector_native_px"] = (992.0, 992.0)
    states.append({"callback_available": True, "caked_projection_ready": True})
    flushed = runtime_session._emit_or_defer_detector_to_caked_manual_trace(
        "unit_after_ready"
    )
    after = capsys.readouterr().out
    print("\ndetector_to_caked_deferred_trace")
    print(after.strip())
    assert flushed is True
    assert "reason=unit_before" in after
    assert "reason=unit_after_ready" not in after
    assert "trace_deferred_until_caked_ready=yes" in after
    assert "pending_background_index=0" in after
    assert "pending_manual_geometry_run_id=manual-snapshot" in after
    assert "branch=0 " in after
    assert "branch=99 " not in after
    assert "raw_caked_deg=(2.000,4.000)" in after
    assert "geometry_caked_deg=(2.200,4.200)" in after
    assert "sim_caked_deg=(2.400,4.400)" in after
    assert "(99.000,99.000)" not in after


def test_minus_1_0_10_detector_to_caked_trace_drops_stale_background(
    monkeypatch,
    capsys,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    entry = {
        "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "hkl": _QR_PICKER_TARGET_HKL,
        "source_branch_index": 0,
        "raw_detector_native_px": (20.0, 40.0),
    }
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_saved_entries",
        lambda: [dict(entry)],
    )
    states = [{"callback_available": False, "caked_projection_ready": False}]
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_projection_state",
        lambda: dict(states[-1]),
    )
    background_indexes = [0]
    monkeypatch.setattr(
        runtime_session,
        "_detector_to_caked_manual_trace_background_index",
        lambda: background_indexes[-1],
    )
    runtime_session.pending_detector_to_caked_manual_trace = None
    assert runtime_session._emit_or_defer_detector_to_caked_manual_trace("unit_before") is False
    assert capsys.readouterr().out == ""
    background_indexes.append(1)
    states.append({"callback_available": True, "caked_projection_ready": True})
    assert runtime_session._emit_or_defer_detector_to_caked_manual_trace("unit_after_ready") is False
    assert capsys.readouterr().out == ""
    assert runtime_session.pending_detector_to_caked_manual_trace is None


def test_minus_1_0_10_detector_to_caked_trace_prints_raw_geometry_sim_caked(
    tmp_path,
) -> None:
    fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        _diag_print_projection_block(
            "[ra-sim] detector -> caked Qr/Qz projection",
            fixture["detector_to_caked"],
            fixture["visual_map"],
            fixture["run_id"],
            emitter="_diag_print_projection_block",
            event="detector_to_caked_projection",
        )
    text = out.getvalue()
    print("\ndetector_to_caked_raw_geometry_sim_trace")
    print(text.strip())
    assert "raw_caked_deg=<unavailable" not in text
    assert "geometry_caked_deg=<unavailable" not in text
    assert "sim_caked_deg=<unavailable" not in text
    assert "outside detector" not in text
    assert "no detector LUT" not in text
    assert "no matching branch row" not in text
    assert "raw_caked_deg=(" in text
    assert "geometry_caked_deg=(" in text
    assert "sim_caked_deg=(" in text


def test_minus_1_0_10_caked_preview_regression_still_fixed(tmp_path) -> None:
    fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    context = fixture["context"]
    rows = fixture["rows"]
    profile_cache = fixture["profile_cache"]
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        fixture["detector_cache"],
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    visual_map = fixture["visual_map"]
    session, _statuses = _diag_live_toggle(
        caked_cache,
        _diag_target_caked(caked_targets[0]),
        display_background=caked_background,
        use_caked_space=True,
        profile_cache=profile_cache,
    )
    for branch in (0, 1):
        preview = _diag_live_preview(
            session,
            _diag_target_caked(caked_targets[branch]),
            cache_data=caked_cache,
            display_background=caked_background,
            use_caked_space=True,
            callbacks=caked_callbacks,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            profile_cache=profile_cache,
        )
        observed = (float(preview["refined_col"]), float(preview["refined_row"]))
        visual_distance = _diag_pair_distance_caked(
            observed,
            visual_map[branch]["caked"],
        )
        print(
            "branch={branch} caked_preview_distance_source={source} "
            "preview_distance={preview_distance} visual_distance={visual_distance}".format(
                branch=branch,
                source=preview.get("preview_distance_source"),
                preview_distance=_diag_float_text(preview.get("sim_dist")),
                visual_distance=_diag_float_text(visual_distance),
            )
        )
        assert preview.get("preview_distance_source") == "sim_visual_caked_deg"
        assert _diag_matches_distance(preview.get("sim_dist"), visual_distance)
        assert float(preview.get("sim_dist")) < 2.0
        assert float(preview.get("sim_dist")) < 10.0
        if branch == 0:
            session, _entries, _state = _diag_live_place(
                session,
                _diag_target_caked(caked_targets[branch]),
                cache_data=caked_cache,
                display_background=caked_background,
                use_caked_space=True,
                callbacks=caked_callbacks,
                radial_axis=radial_axis,
                azimuth_axis=azimuth_axis,
                profile_cache=profile_cache,
            )


_QR_POINT_CONSISTENCY_RUNG_CACHE = {}


def _diag_rung_value_text(value):
    if isinstance(value, tuple):
        if len(value) >= 2:
            try:
                return _diag_point_text((float(value[0]), float(value[1])))
            except Exception:
                pass
        return "(" + ", ".join(_diag_rung_value_text(item) for item in value) + ")"
    if isinstance(value, list):
        if len(value) >= 2 and all(
            not isinstance(item, (Mapping, list, tuple)) for item in value[:2]
        ):
            try:
                return _diag_point_text((float(value[0]), float(value[1])))
            except Exception:
                pass
        return "[" + ", ".join(_diag_rung_value_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_diag_rung_value_text(item)}" for key, item in value.items())
    if isinstance(value, float):
        return _diag_float_text(value)
    if value is None:
        return "<unavailable>"
    return str(value)


def _diag_print_point_consistency_rungs(rows):
    print("\npoint_consistency_rungs")
    print("rung | name | branch | expected | actual | delta | status | first_failure_reason")
    for row in rows:
        print(
            " | ".join(
                (
                    str(row["rung"]),
                    str(row["name"]),
                    str(row["branch"]),
                    str(row["expected"]),
                    str(row["actual"]),
                    str(row["delta"]),
                    str(row["status"]),
                    str(row["first_failure_reason"]),
                )
            )
        )


def _diag_fail_point_consistency_rung(row):
    print(f"first_failing_rung={row['rung']}")
    print(f"first_failing_event={row['event']}")
    print(f"first_failing_branch={row['branch']}")
    print(f"expected_source={row['expected_source']}")
    print(f"actual_source={row['actual_source']}")
    print(f"expected_value={row['expected_value']}")
    print(f"actual_value={row['actual_value']}")
    print(f"suggested_fix_target={row['suggested_fix_target']}")


def _diag_add_point_consistency_rung(
    rows,
    rung,
    name,
    branch,
    *,
    ok,
    expected,
    actual,
    delta="<none>",
    failure_reason="<none>",
    event="<none>",
    expected_source="<none>",
    actual_source="<none>",
    expected_value="<none>",
    actual_value="<none>",
    suggested_fix_target="<none>",
):
    row = {
        "rung": int(rung),
        "name": str(name),
        "branch": branch,
        "expected": _diag_rung_value_text(expected),
        "actual": _diag_rung_value_text(actual),
        "delta": _diag_rung_value_text(delta),
        "status": "PASS" if ok else "FAIL",
        "first_failure_reason": "<none>" if ok else str(failure_reason),
        "event": str(event),
        "expected_source": str(expected_source),
        "actual_source": str(actual_source),
        "expected_value": _diag_rung_value_text(expected_value),
        "actual_value": _diag_rung_value_text(actual_value),
        "suggested_fix_target": str(suggested_fix_target),
    }
    rows.append(row)
    if not ok:
        _diag_print_point_consistency_rungs(rows)
        _diag_fail_point_consistency_rung(row)
        raise AssertionError(f"first_failing_rung={rung}: {failure_reason}")
    return row


def _diag_point_consistency_rung_status(rows, rung):
    return [
        row
        for row in rows
        if int(row.get("rung", -1)) == int(rung)
        and str(row.get("status")) != "PASS"
    ]


def _diag_detector_to_caked_deferral_trace_result():
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    entry = {
        "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
        "hkl": _QR_PICKER_TARGET_HKL,
        "source_branch_index": 0,
        "manual_geometry_run_id": "manual-snapshot",
        "raw_detector_display_px": (10.0, 20.0),
        "raw_detector_native_px": (20.0, 40.0),
        "geometry_detector_display_px": (11.0, 21.0),
        "geometry_detector_native_px": (22.0, 42.0),
        "sim_visual_detector_display_px": (12.0, 22.0),
        "sim_visual_detector_native_px": (24.0, 44.0),
    }
    originals = {
        "_detector_to_caked_manual_trace_saved_entries": getattr(
            runtime_session,
            "_detector_to_caked_manual_trace_saved_entries",
            None,
        ),
        "_native_detector_coords_to_caked_display_coords": getattr(
            runtime_session,
            "_native_detector_coords_to_caked_display_coords",
            None,
        ),
        "_detector_to_caked_manual_trace_projection_state": getattr(
            runtime_session,
            "_detector_to_caked_manual_trace_projection_state",
            None,
        ),
        "_detector_to_caked_manual_trace_background_index": getattr(
            runtime_session,
            "_detector_to_caked_manual_trace_background_index",
            None,
        ),
        "pending_detector_to_caked_manual_trace": getattr(
            runtime_session,
            "pending_detector_to_caked_manual_trace",
            None,
        ),
    }
    states = [{"callback_available": False, "caked_projection_ready": False}]
    try:
        runtime_session._detector_to_caked_manual_trace_saved_entries = lambda: [dict(entry)]
        runtime_session._native_detector_coords_to_caked_display_coords = (
            lambda col, row: (float(col) / 10.0, float(row) / 10.0)
        )
        runtime_session._detector_to_caked_manual_trace_projection_state = (
            lambda: dict(states[-1])
        )
        runtime_session._detector_to_caked_manual_trace_background_index = lambda: 0
        runtime_session.pending_detector_to_caked_manual_trace = None
        with contextlib.redirect_stdout(io.StringIO()) as before_out:
            emitted_before = runtime_session._emit_or_defer_detector_to_caked_manual_trace(
                "unit_before"
            )
        pending = dict(runtime_session.pending_detector_to_caked_manual_trace or {})
        entry["source_branch_index"] = 99
        entry["raw_detector_native_px"] = (990.0, 990.0)
        entry["geometry_detector_native_px"] = (991.0, 991.0)
        entry["sim_visual_detector_native_px"] = (992.0, 992.0)
        states.append({"callback_available": True, "caked_projection_ready": True})
        with contextlib.redirect_stdout(io.StringIO()) as after_out:
            flushed_after = runtime_session._emit_or_defer_detector_to_caked_manual_trace(
                "unit_after_ready"
            )
        after_text = after_out.getvalue()
        return {
            "emitted_before": bool(emitted_before),
            "before_text": before_out.getvalue(),
            "pending": pending,
            "flushed_after": bool(flushed_after),
            "after_text": after_text,
        }
    finally:
        for name, value in originals.items():
            setattr(runtime_session, name, value)


def _diag_records_same(left, right):
    try:
        _diag_records_close(left, right)
    except AssertionError:
        return False
    return True


def _diag_minus_1_0_10_point_consistency_rungs(tmp_path, monkeypatch, *, print_table):
    if "payload" in _QR_POINT_CONSISTENCY_RUNG_CACHE:
        rows = _QR_POINT_CONSISTENCY_RUNG_CACHE["payload"]
        if print_table:
            _diag_print_point_consistency_rungs(rows)
        return rows

    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        detector_fixture = _diag_detector_origin_to_caked_fixture(tmp_path)
    context = detector_fixture["context"]
    caked_callbacks = detector_fixture["caked_callbacks"]
    detector_saved = _diag_branch_map(detector_fixture["detector_saved"])
    detector_to_caked = _diag_branch_map(detector_fixture["detector_to_caked"])
    visual_map = detector_fixture["visual_map"]
    with contextlib.redirect_stdout(io.StringIO()):
        caked_probe = _diag_minus_1_0_10_caked_probe(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        caked_refresh = _diag_minus_1_0_10_caked_refresh_probe(tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        fit_context, fit_dataset, _fit_events = _diag_fit_handoff_dataset(tmp_path)
    fit_rows = _diag_fit_audit_rows(fit_dataset)
    deferral = _diag_detector_to_caked_deferral_trace_result()

    manual_lines = [
        line
        for line in str(deferral["after_text"]).splitlines()
        if "manual_geometry_run_id=" in line
        and ("Qr/Qz" in line or f"q_group_key={_QR_PICKER_TARGET_Q_GROUP_KEY!r}" in line)
    ]
    required_tokens = (
        "manual_geometry_run_id=",
        "manual_trace_version=",
        "source_path_marker=",
        "emitter=",
        "event=",
    )
    unmarked = [
        line for line in manual_lines if any(token not in line for token in required_tokens)
    ]
    legacy = [
        line
        for line in manual_lines
        if f"manual_geometry_run_id={mg.MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID}" in line
    ]
    _diag_add_point_consistency_rung(
        rows,
        0,
        "live path attribution",
        "all",
        ok=bool(manual_lines) and not unmarked and not legacy,
        expected="marked manual Qr/Qz trace lines, no legacy emitter",
        actual={
            "manual_lines": len(manual_lines),
            "unmarked": len(unmarked),
            "legacy": len(legacy),
        },
        event="detector_to_caked_projection_trace",
        expected_source="manual_geometry_cmd_provenance_text",
        actual_source="runtime detector_to_caked trace",
        expected_value="all required provenance fields",
        actual_value=manual_lines[:2],
        suggested_fix_target="ra_sim/gui/manual_geometry.py:geometry_manual_cmd_provenance_text",
        failure_reason="unmarked_or_legacy_manual_qr_qz_trace_line",
    )

    for branch in (0, 1):
        entry = detector_saved.get(branch)
        ok = bool(
            isinstance(entry, Mapping)
            and int(entry.get("source_branch_index", -1)) == branch
            and _diag_q_group_key(entry) == _QR_PICKER_TARGET_Q_GROUP_KEY
            and _diag_hkl(entry) == _QR_PICKER_TARGET_HKL
        )
        _diag_add_point_consistency_rung(
            rows,
            1,
            "detector picker identity",
            branch,
            ok=ok,
            expected={
                "q_group_key": _QR_PICKER_TARGET_Q_GROUP_KEY,
                "hkl": _QR_PICKER_TARGET_HKL,
                "source_branch_index": branch,
            },
            actual={
                "q_group_key": _diag_q_group_key(entry),
                "hkl": _diag_hkl(entry),
                "source_branch_index": entry.get("source_branch_index") if entry else None,
            },
            event="detector_mode_placement",
            expected_source="detector click target",
            actual_source="saved detector-origin pair",
            suggested_fix_target="ra_sim/gui/manual_geometry.py:geometry_manual_toggle_selection_at",
            failure_reason="detector_picker_branch_identity_unstable",
        )

    for branch in (0, 1):
        entry = detector_to_caked[branch]
        geometry_display = mg._geometry_manual_tuple_point(
            entry,
            "geometry_detector_display_px",
        )
        geometry_native = mg._geometry_manual_tuple_point(
            entry,
            "geometry_detector_native_px",
        )
        geometry_d2n = _diag_display_to_native(context, geometry_display)
        sim_display = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_display_px",
        )
        sim_native = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_px",
        )
        sim_d2n = _diag_display_to_native(context, sim_display)
        geometry_delta = (
            None
            if geometry_d2n is None or geometry_native is None
            else (
                float(geometry_d2n[0]) - float(geometry_native[0]),
                float(geometry_d2n[1]) - float(geometry_native[1]),
            )
        )
        sim_delta = (
            None
            if sim_d2n is None or sim_native is None
            else (
                float(sim_d2n[0]) - float(sim_native[0]),
                float(sim_d2n[1]) - float(sim_native[1]),
            )
        )
        copied_display = bool(
            geometry_display is not None
            and geometry_native is not None
            and np.allclose(geometry_display, geometry_native, atol=1.0e-6, rtol=0.0)
            and not np.allclose(geometry_d2n, geometry_native, atol=1.0e-6, rtol=0.0)
        )
        ok = bool(
            geometry_display is not None
            and geometry_native is not None
            and geometry_d2n is not None
            and sim_display is not None
            and sim_native is not None
            and sim_d2n is not None
            and np.allclose(geometry_d2n, geometry_native, atol=1.0e-6, rtol=0.0)
            and np.allclose(sim_d2n, sim_native, atol=1.0e-6, rtol=0.0)
            and entry.get("sim_visual_detector_native_source")
            == "display_to_native_callback"
            and not copied_display
        )
        _diag_add_point_consistency_rung(
            rows,
            2,
            "detector display/native consistency",
            branch,
            ok=ok,
            expected="display_to_native(display_px) == native_px; sim native canonical",
            actual={
                "geometry_display": geometry_display,
                "geometry_native": geometry_native,
                "geometry_d2n": geometry_d2n,
                "sim_display": sim_display,
                "sim_native": sim_native,
                "sim_d2n": sim_d2n,
                "sim_native_source": entry.get("sim_visual_detector_native_source"),
            },
            delta={"geometry": geometry_delta, "sim": sim_delta},
            event="detector_to_caked_switch",
            expected_source="display_to_native_callback",
            actual_source=str(entry.get("sim_visual_detector_native_source")),
            expected_value=geometry_native,
            actual_value=geometry_d2n,
            suggested_fix_target="ra_sim/gui/manual_geometry.py:_geometry_manual_apply_sim_visual_detector_fields",
            failure_reason="detector_display_native_inconsistent",
        )

    expected_geometry_caked = {
        0: (40.10, 35.75),
        1: (40.75, -37.70),
    }
    for branch in (0, 1):
        entry = detector_to_caked[branch]
        direct_results = {}
        for label, native in (
            (
                "raw",
                mg._geometry_manual_tuple_point(entry, "raw_detector_native_px"),
            ),
            (
                "geometry",
                mg._geometry_manual_tuple_point(entry, "geometry_detector_native_px"),
            ),
            (
                "sim",
                mg._geometry_manual_tuple_point(entry, "sim_visual_detector_native_px"),
            ),
        ):
            point, status = _diag_direct_native_to_caked(caked_callbacks, native)
            direct_results[label] = {"native": native, "caked": point, "status": status}
        geometry_caked = direct_results["geometry"]["caked"]
        expected = expected_geometry_caked[branch]
        ok = bool(
            all(item["status"] == "ok" and item["caked"] is not None for item in direct_results.values())
            and np.allclose(geometry_caked, expected, atol=0.75, rtol=0.0)
        )
        _diag_add_point_consistency_rung(
            rows,
            3,
            "direct detector native -> caked",
            branch,
            ok=ok,
            expected={"geometry_native_to_caked_approx": expected},
            actual=direct_results,
            delta=_diag_angle_delta(geometry_caked, expected),
            event="direct_native_to_caked",
            expected_source="saved detector native + caked callback",
            actual_source="native_detector_coords_to_caked_display_coords",
            expected_value=expected,
            actual_value=geometry_caked,
            suggested_fix_target="ra_sim/gui/manual_geometry.py:make_runtime_geometry_manual_projection_callbacks",
            failure_reason="valid_native_detector_coordinate_failed_caked_projection",
        )

    required_pair_fields = (
        "raw_detector_display_px",
        "raw_detector_native_px",
        "geometry_detector_display_px",
        "geometry_detector_native_px",
        "sim_visual_detector_display_px",
        "sim_visual_detector_native_px",
    )
    for branch in (0, 1):
        entry = detector_saved[branch]
        missing = [
            field
            for field in required_pair_fields
            if mg._geometry_manual_tuple_point(entry, field) is None
        ]
        sim_display = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_display_px",
        )
        sim_native = mg._geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_px",
        )
        sim_d2n = _diag_display_to_native(context, sim_display)
        ok = bool(
            not missing
            and sim_display is not None
            and sim_native is not None
            and sim_d2n is not None
            and np.allclose(sim_native, sim_d2n, atol=1.0e-6, rtol=0.0)
        )
        _diag_add_point_consistency_rung(
            rows,
            4,
            "detector-origin saved pair preservation",
            branch,
            ok=ok,
            expected="all detector raw/geometry/sim display+native fields preserved",
            actual={"missing": missing, "sim_native": sim_native, "sim_d2n": sim_d2n},
            event="detector_mode_placement_saved_pair",
            expected_source="saved detector-origin pair",
            actual_source="geometry manual pair entry",
            expected_value=required_pair_fields,
            actual_value={field: mg._geometry_manual_tuple_point(entry, field) for field in required_pair_fields},
            suggested_fix_target="ra_sim/gui/manual_geometry.py:geometry_manual_pair_entry_from_candidate",
            failure_reason="detector_origin_saved_pair_lost_coordinate_field",
        )

    after_text = str(deferral["after_text"])
    readiness_ok = bool(
        deferral["emitted_before"] is False
        and deferral["before_text"] == ""
        and deferral["pending"].get("trace_deferred_until_caked_ready") is True
        and deferral["flushed_after"] is True
        and "trace_deferred_until_caked_ready=yes" in after_text
        and "branch=0 " in after_text
        and "branch=99 " not in after_text
        and "raw_caked_deg=(2.000,4.000)" in after_text
        and "geometry_caked_deg=(2.200,4.200)" in after_text
        and "sim_caked_deg=(2.400,4.400)" in after_text
        and "raw_caked_deg=<unavailable" not in after_text
        and "geometry_caked_deg=<unavailable" not in after_text
        and "sim_caked_deg=<unavailable" not in after_text
    )
    _diag_add_point_consistency_rung(
        rows,
        5,
        "detector -> caked switch readiness",
        "all",
        ok=readiness_ok,
        expected="defer before caked ready; flush saved snapshot after ready",
        actual={
            "emitted_before": deferral["emitted_before"],
            "pending": bool(deferral["pending"]),
            "flushed_after": deferral["flushed_after"],
        },
        event="detector_to_caked_trace_defer_flush",
        expected_source="pending saved detector-native context",
        actual_source="runtime pending_detector_to_caked_manual_trace",
        expected_value="trace_deferred_until_caked_ready=yes then real caked operands",
        actual_value=after_text.strip().splitlines()[:2],
        suggested_fix_target="ra_sim/gui/_runtime/runtime_session.py:_emit_or_defer_detector_to_caked_manual_trace",
        failure_reason="detector_to_caked_trace_emitted_before_projection_ready",
    )

    for branch in (0, 1):
        entry = detector_to_caked[branch]
        raw_caked = _diag_finite_pair(entry, (("raw_caked_x", "raw_caked_y"),))
        geometry_caked, _geometry_source = _diag_observed_caked(entry)
        sim_caked, _sim_source = _diag_sim_visual_caked(entry)
        ok = raw_caked is not None and geometry_caked is not None and sim_caked is not None
        _diag_add_point_consistency_rung(
            rows,
            6,
            "detector-origin detector -> caked values",
            branch,
            ok=ok,
            expected="raw/geometry/sim caked operands real",
            actual={
                "raw_caked_deg": raw_caked,
                "geometry_caked_deg": geometry_caked,
                "sim_caked_deg": sim_caked,
            },
            delta=_diag_angle_delta(geometry_caked, sim_caked),
            event="detector_to_caked_switch",
            expected_source="saved native + caked callback",
            actual_source="refreshed detector-origin saved pair",
            expected_value="no unavailable caked operands",
            actual_value={
                "raw": raw_caked,
                "geometry": geometry_caked,
                "sim": sim_caked,
            },
            suggested_fix_target="ra_sim/gui/_runtime/runtime_session.py:_emit_detector_to_caked_manual_trace",
            failure_reason="detector_origin_caked_operand_unavailable",
        )

    caked_saved = _diag_branch_map(caked_probe["saved_entries"])
    refresh_checks = {
        0: caked_refresh["check0"],
        1: caked_refresh["check1"],
    }
    for branch, preview, check in (
        (0, caked_probe["preview0"], caked_probe["preview_check0"]),
        (1, caked_probe["preview1"], caked_probe["preview_check1"]),
    ):
        saved = caked_saved[branch]
        refresh_check = refresh_checks[branch]
        preview_distance = float(check["printed_preview_distance"])
        assignment_distance = float(saved.get("assignment_distance_to_sim"))
        refresh_distance = float(refresh_check["printed_preview_distance"])
        ok = bool(
            check["preview_distance_source"] == "sim_visual_caked_deg"
            and saved.get("assignment_distance_source") == "sim_visual_caked_deg"
            and check["preview_distance_matches"] == "sim_visual"
            and refresh_check["preview_distance_source"] == "sim_visual_caked_deg"
            and refresh_check["preview_distance_matches"] == "sim_visual"
            and preview_distance < 2.0
            and assignment_distance < 2.0
            and refresh_distance < 2.0
            and "77." not in str(preview.get("message"))
            and "78." not in str(preview.get("message"))
            and "79." not in str(preview.get("message"))
            and "80." not in str(preview.get("message"))
        )
        _diag_add_point_consistency_rung(
            rows,
            7,
            "caked-origin visual source preservation",
            branch,
            ok=ok,
            expected="preview/assignment/refresh source = sim_visual_caked_deg, distance < 2 deg",
            actual={
                "preview_source": check["preview_distance_source"],
                "assignment_source": saved.get("assignment_distance_source"),
                "refresh_source": refresh_check["preview_distance_source"],
                "preview_distance": preview_distance,
                "assignment_distance": assignment_distance,
                "refresh_distance": refresh_distance,
            },
            event="caked_origin_preview_assignment_refresh",
            expected_source="sim_visual_caked_deg",
            actual_source=str(check["preview_distance_source"]),
            expected_value="<2 deg and not 77-80 deg",
            actual_value=preview_distance,
            suggested_fix_target="ra_sim/gui/manual_geometry.py:geometry_manual_pick_preview_state",
            failure_reason="caked_origin_visual_source_regressed",
        )

    label_errors = []
    for branch in (0, 1):
        row = _diag_source_ledger_row(
            "caked_mode_qr_selection",
            branch,
            caked_probe["caked_select_map"][branch],
            visual_map=caked_probe["visual_map"],
            cache_current=caked_probe["cache_current"],
            direct_result={"point": None, "status": "not_checked"},
        )
        if row["field_label_errors"]:
            label_errors.append((branch, row["field_label_errors"]))
    synthetic_bad = []
    for point in ((39.827, 35.250), (42.351, -42.250), (36.209, -117.751)):
        detector_display, detector_source = _diag_observed_detector_display(
            {
                "_caked_qr_projection_cache": True,
                "display_frame": "caked_display",
                "display_col": point[0],
                "display_row": point[1],
                "caked_x": point[0],
                "caked_y": point[1],
                "raw_caked_x": point[0],
                "raw_caked_y": point[1],
                "two_theta_deg": point[0],
                "phi_deg": point[1],
            }
        )
        if detector_display is not None or detector_source != "<unavailable reason=no detector back-projection>":
            synthetic_bad.append((point, detector_display, detector_source))
    _diag_add_point_consistency_rung(
        rows,
        8,
        "coordinate field labeling",
        "all",
        ok=not label_errors and not synthetic_bad,
        expected="caked angle pairs never printed as detector px",
        actual={"field_label_errors": label_errors, "synthetic_bad": synthetic_bad},
        event="coordinate_label_guard",
        expected_source="detector px fields from detector projection only",
        actual_source="source ledger + caked projection guard",
        expected_value="<unavailable reason=no detector back-projection>",
        actual_value={"errors": label_errors, "synthetic_bad": synthetic_bad},
        suggested_fix_target="ra_sim/gui/geometry_fit.py:manual_trace_detector_point_label_guard",
        failure_reason="caked_angle_pair_routed_to_detector_px_field",
    )

    for branch in (0, 1):
        detector_geometry = _diag_observed_caked(detector_to_caked[branch])[0]
        caked_geometry = _diag_observed_caked(caked_saved[branch])[0]
        delta = _diag_angle_delta(detector_geometry, caked_geometry)
        ok = bool(
            delta is not None
            and abs(float(delta[0])) <= 0.25
            and abs(float(delta[1])) <= 0.5
        )
        _diag_add_point_consistency_rung(
            rows,
            9,
            "cross-origin equivalence",
            branch,
            ok=ok,
            expected="detector-origin and caked-origin refined caked points agree",
            actual={
                "detector_origin_geometry_caked_deg": detector_geometry,
                "caked_origin_geometry_caked_deg": caked_geometry,
            },
            delta=delta,
            event="cross_origin_compare",
            expected_source="same physical Qr branch point",
            actual_source="detector-origin vs caked-origin saved geometry",
            expected_value="2theta<=0.25 deg, wrapped phi<=0.5 deg",
            actual_value=delta,
            suggested_fix_target="ra_sim/gui/manual_geometry.py:refresh_geometry_manual_pair_entry",
            failure_reason="detector_origin_caked_origin_not_equivalent",
        )

    recompute_first = _diag_recompute_fit_audit_rows(fit_context, fit_dataset)
    recompute_second = _diag_recompute_fit_audit_rows(fit_context, fit_dataset)
    records_first = _diag_records_from_audit_rows(recompute_first)
    records_second = _diag_records_from_audit_rows(recompute_second)
    recompute_same = _diag_records_same(records_first, records_second)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(fit_context, fit_dataset)
    no_fake_after_fit = True
    if not fit_run["step_executed"]:
        baseline_records = _diag_records_from_audit_rows(fit_rows)
        repeated_records = _diag_records_from_audit_rows(
            _diag_recompute_fit_audit_rows(fit_context, fit_dataset)
        )
        no_fake_after_fit = _diag_records_same(baseline_records, repeated_records)
    allowed_prediction_sources = {"dynamic_current_simulation", "saved_visual_sim_refined"}
    for branch in (0, 1):
        row = fit_rows[branch]
        prediction_source = str(row.get("fit_prediction_source"))
        observed_match = bool(
            row.get("observed_visual_to_fit_observed_match") == "yes"
            and float(row.get("fit_observed_minus_observed_refined_detector_delta_px", np.inf)) <= 1.0
        )
        caked_delta = row.get("fit_observed_minus_observed_refined_caked_delta_deg")
        try:
            _diag_delta_pair_close(caked_delta)
            caked_match = True
        except AssertionError:
            caked_match = False
        prediction_source_ok = bool(
            prediction_source in allowed_prediction_sources
            or prediction_source.startswith("unavailable_reason:")
        )
        ok = bool(
            observed_match
            and caked_match
            and prediction_source_ok
            and recompute_same
            and no_fake_after_fit
        )
        _diag_add_point_consistency_rung(
            rows,
            10,
            "fit handoff consistency",
            branch,
            ok=ok,
            expected="fit observed=manual refined; prediction explicit; same params stable; no fake failed-fit state",
            actual={
                "fit_observed_caked_deg": row.get("fit_observed_caked_deg"),
                "fit_prediction_caked_deg": row.get("fit_prediction_caked_deg"),
                "fit_prediction_source": prediction_source,
                "recompute_same": recompute_same,
                "no_fake_after_fit": no_fake_after_fit,
            },
            delta=row.get("fit_observed_minus_observed_refined_caked_delta_deg"),
            event="fit_handoff",
            expected_source="manual/background observed + simulation prediction",
            actual_source=prediction_source,
            expected_value="stable predicted-observed residual",
            actual_value=row.get("residual_caked_deg"),
            suggested_fix_target="ra_sim/gui/geometry_fit.py:build_geometry_fit_qr_handoff_audit_rows",
            failure_reason="fit_handoff_inconsistent",
        )

    _QR_POINT_CONSISTENCY_RUNG_CACHE["payload"] = rows
    if print_table:
        _diag_print_point_consistency_rungs(rows)
    return rows


def test_minus_1_0_10_point_consistency_rungs(tmp_path, monkeypatch) -> None:
    rows = _diag_minus_1_0_10_point_consistency_rungs(
        tmp_path,
        monkeypatch,
        print_table=True,
    )
    assert rows
    assert not [row for row in rows if row["status"] != "PASS"]


def test_minus_1_0_10_detector_origin_rungs_pass(tmp_path, monkeypatch) -> None:
    rows = _diag_minus_1_0_10_point_consistency_rungs(
        tmp_path,
        monkeypatch,
        print_table=False,
    )
    for rung in (1, 2, 3, 4, 5, 6):
        assert not _diag_point_consistency_rung_status(rows, rung)


def test_minus_1_0_10_caked_origin_rungs_still_pass(tmp_path, monkeypatch) -> None:
    rows = _diag_minus_1_0_10_point_consistency_rungs(
        tmp_path,
        monkeypatch,
        print_table=False,
    )
    for rung in (7, 9):
        assert not _diag_point_consistency_rung_status(rows, rung)


def test_minus_1_0_10_no_caked_angles_as_detector_pixels(tmp_path, monkeypatch) -> None:
    rows = _diag_minus_1_0_10_point_consistency_rungs(
        tmp_path,
        monkeypatch,
        print_table=False,
    )
    assert not _diag_point_consistency_rung_status(rows, 8)


def test_minus_1_0_10_fit_handoff_rung_still_passes(tmp_path, monkeypatch) -> None:
    rows = _diag_minus_1_0_10_point_consistency_rungs(
        tmp_path,
        monkeypatch,
        print_table=False,
    )
    assert not _diag_point_consistency_rung_status(rows, 10)


def _diag_headless_comparison_rows(entries, visual_map, cache_current, previews):
    branch_entries = _diag_branch_map(entries)
    rows = []
    for branch in (0, 1):
        entry = branch_entries[branch]
        observed, _observed_source = _diag_observed_caked(entry)
        visual = visual_map.get(branch, {}).get("caked")
        cache = cache_current.get(branch, {}).get("caked")
        visual_delta = _diag_angle_delta(observed, visual)
        cache_delta = _diag_angle_delta(observed, cache)
        preview = previews[branch]
        rows.append(
            {
                "branch": branch,
                "observed_caked": observed,
                "sim_visual_caked": visual,
                "sim_cache_current_caked": cache,
                "visual_delta": visual_delta,
                "cache_current_delta": cache_delta,
                "preview_distance": preview.get("status_distance_value", preview.get("sim_dist")),
                "assignment_distance": entry.get("assignment_distance_to_sim"),
            }
        )
    return rows


def _diag_print_headless_comparison_table(rows, run_id):
    print(
        "[ra-sim] Qr/Qz completed group comparison "
        + _diag_cmd_stamp(
            run_id,
            emitter="_diag_print_headless_comparison_table",
            event="completed_group_comparison",
        )
    )
    print(
        "branch | observed_caked | sim_visual_caked | sim_cache_current_caked | "
        "visual_delta | cache_current_delta | preview_distance | assignment_distance "
        + _diag_cmd_stamp(
            run_id,
            emitter="_diag_print_headless_comparison_table",
            event="completed_group_comparison_header",
        )
    )
    for row in rows:
        print(
            (
                " | ".join(
                    (
                        str(row["branch"]),
                        _diag_point_text(row["observed_caked"]),
                        _diag_point_text(row["sim_visual_caked"]),
                        _diag_point_text(row["sim_cache_current_caked"]),
                        _diag_point_text(row["visual_delta"]),
                        _diag_point_text(row["cache_current_delta"]),
                        _diag_float_text(row["preview_distance"]),
                        _diag_float_text(row["assignment_distance"]),
                    )
                )
                + " "
                + _diag_cmd_stamp(
                    run_id,
                    emitter="_diag_print_headless_comparison_table",
                    event="completed_group_comparison_row",
                    branch=row["branch"],
                    actual_source="sim_visual_caked_deg",
                    expected_source="sim_visual_caked_deg",
                )
            )
        )
        print(
            "legacy geometry_minus_sim_caked_delta_deg branch={branch} value={delta} {stamp}".format(
                branch=row["branch"],
                delta=_diag_point_text(row["visual_delta"]),
                stamp=_diag_cmd_stamp(
                    run_id,
                    emitter="_diag_print_headless_comparison_table",
                    event="completed_group_legacy_delta",
                    branch=row["branch"],
                    actual_source="legacy_geometry_minus_sim_caked_delta_deg",
                    expected_source="sim_visual_caked_deg",
                ),
            )
        )


def test_minus_1_0_10_live_cmd_log_validator(tmp_path) -> None:
    importlib.import_module("ra_sim.gui._runtime.runtime_session")
    context, rows = _diag_startup_context_and_rows(tmp_path)
    context["rows"] = rows
    profile_cache = rows["profile_cache"]
    display_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["current_background_display"])
    )
    caked_background = np.asarray(
        _diag_runtime_value(context["projection_kwargs"]["last_caked_background_image_unscaled"])
    )
    radial_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_radial_values"])
    azimuth_axis = _diag_runtime_value(context["projection_kwargs"]["last_caked_azimuth_values"])
    detector_callbacks = context["projection_callbacks"]
    detector_cache = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    detector_grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        detector_cache,
        display_background=display_background,
        profile_cache=profile_cache,
    )
    detector_targets = _diag_target_rows_by_branch(detector_grouped)
    assert set(detector_targets) == {0, 1}
    caked_callbacks, caked_cache, caked_grouped = _diag_build_caked_qr_cache(
        context,
        rows,
        detector_cache,
    )
    caked_targets = _diag_target_rows_by_branch(caked_grouped)
    assert set(caked_targets) == {0, 1}
    visual_map = _diag_visual_map(detector_targets)
    cache_current = _diag_cache_current_map(context, detector_targets, caked_targets)

    detector_statuses = []
    caked_statuses = []
    transcript = {}
    detector_seed0 = _diag_detector_display_point(detector_targets[0])
    assert detector_seed0 is not None
    detector_target_points = {
        0: (1083.818, 1083.270),
        1: (1846.620, 1083.734),
    }
    caked_target_points = {
        0: (40.1425, 35.5668),
        1: (40.8530, -37.5659),
    }

    with contextlib.redirect_stdout(io.StringIO()) as out:
        detector_workflow, detector_state = _diag_headless_workflow(
            cache_data=detector_cache,
            display_background=display_background,
            use_caked_space=False,
            callbacks=detector_callbacks,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            profile_cache=profile_cache,
            status_sink=detector_statuses,
        )
        detector_workflow.toggle_selection_at(*detector_seed0)
        detector_run_id = detector_state["pick_session"].get("manual_geometry_run_id")
        assert detector_run_id
        picker_resolved = _diag_headless_detector_picker_resolution(
            context,
            _diag_flatten_grouped(detector_grouped),
            detector_targets,
            detector_run_id,
        )
        for branch in (0, 1):
            detector_workflow.update_pick_preview(*detector_target_points[branch], force=True)
            detector_workflow.place_selection_at(*detector_target_points[branch])
        print(
            "[ra-sim] Qr/Qz detector geometry selection "
            + _diag_cmd_stamp(
                detector_run_id,
                emitter="test_minus_1_0_10_live_cmd_log_validator",
                event="detector_geometry_selection",
            )
        )
        for entry in _diag_sorted(detector_state["pairs"]):
            print(
                "branch={branch} q_group_key={q_group} hkl={hkl} "
                "geometry_detector_display_px={display} geometry_caked_deg={caked} "
                "assignment_distance={assignment} source={source} {stamp}".format(
                    branch=_diag_source_branch(entry),
                    q_group=_diag_q_group_key(entry),
                    hkl=_diag_hkl(entry),
                    display=_diag_point_text(_diag_observed_detector_display(entry)[0]),
                    caked=_diag_point_text(_diag_observed_caked(entry)[0]),
                    assignment=_diag_float_text(entry.get("assignment_distance_to_sim")),
                    source=entry.get("assignment_distance_source", "<unavailable>"),
                    stamp=_diag_cmd_stamp(
                        detector_run_id,
                        emitter="test_minus_1_0_10_live_cmd_log_validator",
                        event="detector_geometry_selection_row",
                        branch=_diag_source_branch(entry),
                        actual_source=entry.get("assignment_distance_source", "<unavailable>"),
                        expected_source="sim_visual_detector_display_px",
                    ),
                )
            )
    transcript["detector"] = out.getvalue().strip()
    detector_saved = [dict(entry) for entry in detector_state["pairs"]]

    with contextlib.redirect_stdout(io.StringIO()) as out:
        detector_to_caked = _diag_refresh_entries(caked_callbacks, detector_saved)
        _diag_print_projection_block(
            "[ra-sim] detector -> caked Qr/Qz projection",
            detector_to_caked,
            visual_map,
            detector_run_id,
            emitter="_diag_print_projection_block",
            event="detector_to_caked_projection",
        )
    transcript["detector_to_caked"] = out.getvalue().strip()

    with contextlib.redirect_stdout(io.StringIO()) as out:
        caked_workflow, caked_state = _diag_headless_workflow(
            cache_data=caked_cache,
            display_background=caked_background,
            use_caked_space=True,
            callbacks=caked_callbacks,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            profile_cache=profile_cache,
            status_sink=caked_statuses,
        )
        caked_select_point = _diag_target_caked(caked_targets[0])
        assert caked_select_point is not None
        caked_workflow.toggle_selection_at(*caked_select_point)
        caked_run_id = caked_state["pick_session"].get("manual_geometry_run_id")
        assert caked_run_id
        print(
            "[geometry] Clear saved manual geometry. "
            + _diag_cmd_stamp(
                caked_run_id,
                emitter="test_minus_1_0_10_live_cmd_log_validator",
                event="clear_saved_manual_geometry",
            )
        )
        print(
            "[ra-sim] Qr/Qz caked simulation selection "
            + _diag_cmd_stamp(
                caked_run_id,
                emitter="test_minus_1_0_10_live_cmd_log_validator",
                event="caked_simulation_selection",
            )
        )
        print(
            "q_group_key={q_group} hkl={hkl} click_caked_deg={click} {stamp}".format(
                q_group=_QR_PICKER_TARGET_Q_GROUP_KEY,
                hkl=_QR_PICKER_TARGET_HKL,
                click=_diag_point_text(caked_select_point),
                stamp=_diag_cmd_stamp(
                    caked_run_id,
                    emitter="test_minus_1_0_10_live_cmd_log_validator",
                    event="caked_simulation_selection_row",
                    branch=0,
                    actual_source="sim_visual_caked_deg",
                    expected_source="sim_visual_caked_deg",
                ),
            )
        )
        caked_previews = {}
        for branch in (0, 1):
            caked_workflow.update_pick_preview(*caked_target_points[branch], force=True)
            caked_previews[branch] = dict(
                mg.geometry_manual_pick_preview_state(
                    *caked_target_points[branch],
                    pick_session=caked_state["pick_session"],
                    current_background_index=0,
                    force=True,
                    remaining_candidates=mg.geometry_manual_unassigned_group_candidates(
                        caked_state["pick_session"],
                        current_background_index=0,
                    ),
                    display_background=caked_background,
                    cache_data=dict(caked_cache),
                    refine_preview_point=_diag_build_refine_preview(
                        True,
                        radial_axis,
                        azimuth_axis,
                    ),
                    use_caked_space=True,
                    caked_angles_to_background_display_coords=getattr(
                        caked_callbacks,
                        "caked_angles_to_background_display_coords",
                        None,
                    ),
                    radial_axis=radial_axis,
                    azimuth_axis=azimuth_axis,
                    profile_cache=profile_cache,
                )
            )
            caked_workflow.place_selection_at(*caked_target_points[branch])
        print(
            "[ra-sim] Qr/Qz caked geometry selection "
            + _diag_cmd_stamp(
                caked_run_id,
                emitter="test_minus_1_0_10_live_cmd_log_validator",
                event="caked_geometry_selection",
            )
        )
        for entry in _diag_sorted(caked_state["pairs"]):
            print(
                "branch={branch} q_group_key={q_group} hkl={hkl} "
                "geometry_caked_deg={caked} sim_visual_deg={visual} "
                "assignment_distance={assignment} source={source} {stamp}".format(
                    branch=_diag_source_branch(entry),
                    q_group=_diag_q_group_key(entry),
                    hkl=_diag_hkl(entry),
                    caked=_diag_point_text(_diag_observed_caked(entry)[0]),
                    visual=_diag_point_text(
                        visual_map.get(_diag_source_branch(entry), {}).get("caked")
                    ),
                    assignment=_diag_float_text(entry.get("assignment_distance_to_sim")),
                    source=entry.get("assignment_distance_source", "<unavailable>"),
                    stamp=_diag_cmd_stamp(
                        caked_run_id,
                        emitter="test_minus_1_0_10_live_cmd_log_validator",
                        event="caked_geometry_selection_row",
                        branch=_diag_source_branch(entry),
                        actual_source=entry.get("assignment_distance_source", "<unavailable>"),
                        expected_source="sim_visual_caked_deg",
                    ),
                )
            )
        comparison_rows = _diag_headless_comparison_rows(
            caked_state["pairs"],
            visual_map,
            cache_current,
            caked_previews,
        )
        _diag_print_headless_comparison_table(comparison_rows, caked_run_id)
    transcript["caked"] = out.getvalue().strip()
    caked_saved = [dict(entry) for entry in caked_state["pairs"]]

    with contextlib.redirect_stdout(io.StringIO()) as out:
        caked_to_detector = _diag_refresh_entries(detector_callbacks, caked_saved)
        _diag_print_projection_block(
            "[ra-sim] caked-origin caked -> detector projection",
            caked_to_detector,
            visual_map,
            caked_run_id,
            emitter="_diag_print_projection_block",
            event="caked_to_detector_projection",
        )
    transcript["caked_to_detector"] = out.getvalue().strip()

    full_transcript = (
        "[ra-sim] headless_minus_1_0_10_manual_picker_replay "
        + _diag_cmd_stamp(
            caked_run_id,
            emitter="test_minus_1_0_10_live_cmd_log_validator",
            event="headless_replay_transcript",
        )
        + "\n"
        "--- detector workflow transcript ---\n"
        f"{transcript['detector']}\n"
        "--- detector -> caked transcript ---\n"
        f"{transcript['detector_to_caked']}\n"
        "--- caked workflow transcript ---\n"
        f"{transcript['caked']}\n"
        "--- caked -> detector transcript ---\n"
        f"{transcript['caked_to_detector']}"
    )
    print("\n" + full_transcript)

    forbidden = {
        "tagged sim ... (77.85 deg)": "tagged sim [-1,0,10] (77.85 deg)",
        "nearest sim ... (80.32 deg)": "nearest sim [-1,0,10] (80.32 deg)",
        "Assigned ... (77.85 deg from sim)": "77.85 deg from sim",
        "Assigned ... (77.85 deg from sim bracketed)": (
            "Assigned to [-1,0,10] (77.85 deg from sim)"
        ),
        "legacy current_full_reflection_prediction": (
            "sim_caked_semantics=current_full_reflection_prediction"
        ),
        "legacy cache_fresh_row": "active_row_policy=cache_fresh_row",
        "caked labeled detector display px": "geometry_detector_display_px=(39.",
        "caked labeled detector px": "geometry_detector_px=(39.",
        "valid detector->caked outside detector": (
            "raw_caked_deg=<unavailable reason=outside detector>"
        ),
        "missing saved sim visual branch row": (
            "sim_caked_deg=<unavailable reason=no matching branch row>"
        ),
    }
    trace_marker = f"manual_trace_version={mg.MANUAL_GEOMETRY_TRACE_VERSION}"
    current_run_marker = f"manual_geometry_run_id={caked_run_id}"
    all_lines = full_transcript.splitlines()
    current_lines = [
        line for line in all_lines if current_run_marker in line and trace_marker in line
    ]

    def _line_emitter(line):
        for token in str(line).split():
            if token.startswith("emitter="):
                return token.split("=", 1)[1]
        return "<missing>"

    def _requires_manual_stamp(line):
        if not (line.startswith("[geometry]") or line.startswith("[ra-sim]")):
            return False
        manual_tokens = (
            "manual_geometry",
            "Manual pick preview",
            "Placed peak",
            "Saved ",
            "Selected ",
            "Qr/Qz",
            "live_caked_visual_source",
            "headless_minus_1_0_10",
            "Clear saved manual geometry",
            "caked-origin",
            "detector -> caked",
        )
        return any(token in line for token in manual_tokens)

    unmarked_lines = [
        line
        for line in all_lines
        if _requires_manual_stamp(line)
        and not (
            "manual_geometry_run_id=" in line
            and "manual_trace_version=" in line
            and "emitter=" in line
        )
    ]
    for line in unmarked_lines[:5]:
        print("unmarked_manual_geometry_line=yes")
        print(f"unmarked_line={line}")

    current_bad_hits = []
    stale_bad_hits = []
    for label, needle in forbidden.items():
        for line in all_lines:
            if needle not in line:
                continue
            hit = (label, needle, line)
            if line in current_lines:
                current_bad_hits.append(hit)
            else:
                stale_bad_hits.append(hit)

    print("\nlive_cmd_log_validator")
    print(f"current_manual_geometry_run_id={caked_run_id}")
    print(f"current_run_line_count={len(current_lines)}")
    print(
        "headless_replay_bypasses_live_emitter="
        + (
            "no"
            if any("emitter=geometry_manual_pick_preview_state" in line for line in current_lines)
            and any("emitter=geometry_manual_place_selection_at" in line for line in current_lines)
            else "yes"
        )
    )
    if current_bad_hits:
        for label, needle, line in current_bad_hits[:5]:
            print("bad_line_classification=current_live_path_failure")
            print(f"bad_label={label}")
            print(f"bad_needle={needle}")
            print(f"bad_emitter={_line_emitter(line)}")
            print(f"bad_line={line}")
    if stale_bad_hits:
        for label, needle, line in stale_bad_hits[:5]:
            print("bad_line_classification=stale_or_previous_run")
            print(f"stale_bad_label={label}")
            print(f"stale_bad_needle={needle}")
            print(f"stale_bad_line={line}")

    caked_preview_sources = {
        str(caked_previews[branch].get("status_distance_source")) for branch in (0, 1)
    }
    caked_saved_by_branch = _diag_branch_map(caked_saved)
    caked_assignment_sources = {
        str(caked_saved_by_branch[branch].get("assignment_distance_source"))
        for branch in (0, 1)
    }
    print(f"current_run_bad_77_80_lines={'yes' if current_bad_hits else 'no'}")
    print(f"current_run_unmarked_manual_lines={'yes' if unmarked_lines else 'no'}")
    print(
        "caked_preview_distance_source="
        + (
            "sim_visual_caked_deg"
            if caked_preview_sources == {"sim_visual_caked_deg"}
            else ",".join(sorted(caked_preview_sources))
        )
    )
    print(
        "caked_assignment_distance_source="
        + (
            "sim_visual_caked_deg"
            if caked_assignment_sources == {"sim_visual_caked_deg"}
            else ",".join(sorted(caked_assignment_sources))
        )
    )
    print(f"stale_bad_lines_detected={'yes' if stale_bad_hits else 'no'}")
    print("branch | preview_source | preview_distance | assignment_source | assignment_distance")
    for branch in (0, 1):
        print(
            " | ".join(
                (
                    str(branch),
                    str(caked_previews[branch].get("status_distance_source")),
                    _diag_float_text(caked_previews[branch].get("status_distance_value")),
                    str(caked_saved_by_branch[branch].get("assignment_distance_source")),
                    _diag_float_text(caked_saved_by_branch[branch].get("assignment_distance_to_sim")),
                )
            )
        )

    assert "No simulated Qr/Qz groups are available" not in full_transcript
    assert "q_group_key=('q_group', 'primary', 1, 10)" in full_transcript
    assert picker_resolved[0]["source_table_index"] == 160
    assert picker_resolved[0]["source_row_index"] == 24
    assert picker_resolved[0]["source_branch_index"] == 0
    assert picker_resolved[1]["source_table_index"] == 167
    assert picker_resolved[1]["source_row_index"] == 24
    assert picker_resolved[1]["source_branch_index"] == 1
    assert set(_diag_branch_map(detector_saved)) == {0, 1}
    assert set(_diag_branch_map(caked_saved)) == {0, 1}
    assert current_lines
    assert not unmarked_lines
    assert any("emitter=geometry_manual_pick_preview_state" in line for line in current_lines)
    assert any("emitter=geometry_manual_place_selection_at" in line for line in current_lines)
    assert caked_preview_sources == {"sim_visual_caked_deg"}
    assert caked_assignment_sources == {"sim_visual_caked_deg"}
    assert "status_distance_source=sim_visual_caked_deg" in transcript["caked"]
    assert all(float(row["preview_distance"]) < 2.0 for row in comparison_rows)
    assert all(float(row["assignment_distance"]) < 2.0 for row in comparison_rows)
    for row in comparison_rows:
        assert _diag_matches_distance(row["preview_distance"], np.hypot(*row["visual_delta"]))
        assert _diag_matches_distance(row["assignment_distance"], np.hypot(*row["visual_delta"]))
        assert np.hypot(*row["visual_delta"]) < 2.0
    assert not current_bad_hits


def test_detector_mode_qr_picker_has_candidates_after_startup_sim_ready(tmp_path) -> None:
    context, rows = _diag_startup_context_and_rows(tmp_path)
    projected_fresh_rows = [
        _diag_project_detector_entry(context, entry)
        for entry in rows["fresh_rows"]
        if isinstance(entry, Mapping)
    ]
    cache_data = _diag_detector_picker_cache(projected_fresh_rows, overlay_grouped={})
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        profile_cache=rows["profile_cache"],
    )
    trace = mg.geometry_manual_detector_picker_input_trace(
        cache_data,
        background_index=0,
        display_background=np.zeros((3000, 3000), dtype=float),
        grouped_candidates=grouped,
        profile_cache=rows["profile_cache"],
    )
    picker_rows = _diag_flatten_grouped(grouped)
    print("\nDetector-mode startup trace")
    print(trace)
    assert trace["simulation_ready"] is True
    assert trace["source_rows_count"] > 0
    assert trace["qr_group_rows_count"] > 0
    assert trace["picker_candidate_count"] > 0
    assert picker_rows

    click = _diag_detector_display_point(picker_rows[0])
    assert click is not None
    _session, statuses = _diag_toggle_detector_click(cache_data, click, rows["profile_cache"])
    assert not any("No simulated Qr/Qz groups are available" in text for text in statuses)


def test_detector_mode_qr_picker_clean_start_without_saved_pairs(tmp_path) -> None:
    _context, rows = _diag_startup_context_and_rows(tmp_path)
    fresh_rows = [dict(entry) for entry in rows["fresh_rows"] if isinstance(entry, Mapping)]
    previous_empty_cache = {
        "signature": ("clean-start-detector-picker-empty-before-fallback",),
        "simulated_peaks": [],
        "active_simulated_peaks": [],
        "grouped_candidates": {},
        "caked_qr_projection_grouped_candidates": {},
    }
    previous_trace = mg.geometry_manual_detector_picker_input_trace(
        previous_empty_cache,
        view_mode="detector",
        background_index=0,
        display_background=np.zeros((3000, 3000), dtype=float),
        qr_overlay_visible=False,
        grouped_candidates={},
        profile_cache=rows["profile_cache"],
    )
    assert previous_trace["reason_candidates_are_empty"] == "no_detector_picker_source_rows"

    cache_data = {
        "signature": ("clean-start-detector-picker-fresh-source-rows",),
        "simulated_peaks": [],
        "active_simulated_peaks": [],
        "detector_picker_source_rows": [],
        "detector_picker_rows": [],
        "detector_picker_grouped_candidates": {},
        "fresh_source_rows": fresh_rows,
        "grouped_candidates": {},
        "caked_qr_projection_entries": [],
        "caked_qr_projection_grouped_candidates": {},
        "caked_qr_projection_lookup": {},
        "cache_metadata": {
            "cache_source": "clean_start_fresh_source_rows",
            "simulated_peak_count": len(fresh_rows),
        },
    }
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        display_background=np.zeros((3000, 3000), dtype=float),
        profile_cache=rows["profile_cache"],
    )
    trace = mg.geometry_manual_detector_picker_input_trace(
        cache_data,
        view_mode="detector",
        background_index=0,
        display_background=np.zeros((3000, 3000), dtype=float),
        qr_overlay_visible=False,
        grouped_candidates=grouped,
        profile_cache=rows["profile_cache"],
    )
    picker_rows = _diag_flatten_grouped(grouped)
    source_breakdown = trace["detector_picker_candidate_count_by_source"]

    print("\nDetector-mode clean-start trace")
    print(f"previous_empty_reason={previous_trace['reason_candidates_are_empty']}")
    print(f"live_clean_start_candidate_count={trace['detector_picker_candidate_count']}")
    print(f"live_clean_start_source_breakdown={source_breakdown}")
    print(trace)

    assert trace["simulation_ready"] is True
    assert trace["caked_ready"] is False
    assert trace["qr_overlay_visible"] is False
    assert trace["manual_saved_pair_count"] == 0
    assert trace["fresh_source_row_count"] > 0
    assert trace["detector_picker_candidate_count"] > 0
    assert trace["overlay_drawn_count"] == 0
    assert trace["reason_candidates_are_empty"] == ""
    assert picker_rows
    assert not any("manual_saved_pair" in str(key) for key in source_breakdown)
    assert any("fresh" in str(key) or "source" in str(key) for key in source_breakdown)

    fresh_target_rows = [
        row
        for row in picker_rows
        if _diag_q_group_key(row) == _QR_PICKER_TARGET_Q_GROUP_KEY
        and _diag_hkl(row) == _QR_PICKER_TARGET_HKL
    ]
    if fresh_target_rows:
        assert {int(row["source_branch_index"]) for row in fresh_target_rows} >= {0, 1}
    else:
        available_target_hkls = sorted(
            {
                _diag_hkl(row)
                for row in picker_rows
                if _diag_q_group_key(row) == _QR_PICKER_TARGET_Q_GROUP_KEY
            },
            key=repr,
        )
        print(
            "clean-start (-1,0,10) fresh rows unavailable; "
            f"available target q_group hkls={available_target_hkls}"
        )

    click = _diag_detector_display_point(picker_rows[0])
    assert click is not None
    live_traces = []
    _session, statuses = _diag_toggle_detector_click(
        cache_data,
        click,
        rows["profile_cache"],
        trace_output=live_traces,
    )
    assert not any("No simulated Qr/Qz groups are available" in text for text in statuses)
    assert live_traces
    assert live_traces[-1]["detector_picker_candidate_count"] > 0
    assert live_traces[-1]["manual_saved_pair_count"] == 0
    assert live_traces[-1]["caked_ready"] is False


def test_detector_mode_qr_picker_selects_minus_1_0_10_branch_clicks(tmp_path) -> None:
    _context, rows = _diag_startup_context_and_rows(tmp_path)
    cache_data = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        profile_cache=rows["profile_cache"],
    )
    target_by_branch = _diag_target_rows_by_branch(grouped)
    assert set(target_by_branch) == {0, 1}

    expected = {
        0: ((1074.878, 1085.949), 160, 24),
        1: ((1834.555, 1083.733), 167, 24),
    }
    for branch, (click, table_index, row_index) in expected.items():
        session, _statuses = _diag_toggle_detector_click(
            cache_data,
            click,
            rows["profile_cache"],
        )
        selected = session.get("tagged_candidate")
        assert isinstance(selected, Mapping)
        selected_display = _diag_detector_display_point(selected)
        assert selected_display is not None
        assert np.allclose(selected_display, click, atol=1.0e-3, rtol=0.0)
        assert _diag_q_group_key(selected) == _QR_PICKER_TARGET_Q_GROUP_KEY
        assert _diag_hkl(selected) == _QR_PICKER_TARGET_HKL
        assert int(selected["source_branch_index"]) == branch
        assert int(selected["source_table_index"]) == table_index
        assert int(selected["source_row_index"]) == row_index
        print(
            "detector click "
            f"{click} -> q_group={_diag_q_group_key(selected)} "
            f"branch={selected.get('source_branch_index')} "
            f"table={selected.get('source_table_index')} "
            f"row={selected.get('source_row_index')}"
        )


def test_detector_mode_qr_picker_not_blocked_by_caked_unavailable(tmp_path) -> None:
    _context, rows = _diag_startup_context_and_rows(tmp_path)
    cache_data = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    cache_data["caked_qr_projection_entries"] = []
    cache_data["caked_qr_projection_grouped_candidates"] = {}
    cache_data["caked_qr_projection_lookup"] = {}
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        profile_cache=rows["profile_cache"],
    )
    target_by_branch = _diag_target_rows_by_branch(grouped)
    display = _diag_detector_display_point(target_by_branch[0])
    assert display is not None
    session, statuses = _diag_toggle_detector_click(cache_data, display, rows["profile_cache"])
    assert session["group_key"] == _QR_PICKER_TARGET_Q_GROUP_KEY
    assert not any("No simulated Qr/Qz groups are available" in text for text in statuses)


def test_qr_overlay_hidden_does_not_disable_manual_picker_candidates(tmp_path) -> None:
    _context, rows = _diag_startup_context_and_rows(tmp_path)
    cache_data = _diag_detector_picker_cache(rows["overlay_rows"], overlay_grouped={})
    trace = mg.geometry_manual_detector_picker_input_trace(
        cache_data,
        background_index=0,
        display_background=np.zeros((3000, 3000), dtype=float),
        profile_cache=rows["profile_cache"],
    )
    assert trace["overlay_drawn_count"] == 0
    assert trace["picker_candidate_count"] > 0
    grouped = mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        cache_data,
        profile_cache=rows["profile_cache"],
    )
    target_by_branch = _diag_target_rows_by_branch(grouped)
    display = _diag_detector_display_point(target_by_branch[1])
    assert display is not None
    session, _statuses = _diag_toggle_detector_click(cache_data, display, rows["profile_cache"])
    selected = session.get("tagged_candidate")
    assert isinstance(selected, Mapping)
    assert int(selected["source_branch_index"]) == 1


def test_detector_mode_qr_picker_overlay_hidden_does_not_disable_manual_picker_candidates(
    tmp_path,
) -> None:
    test_qr_overlay_hidden_does_not_disable_manual_picker_candidates(tmp_path)


_QR_FIT_AUDIT_CACHE = {}
_QR_FIT_STEP_CACHE = {}


def _diag_caked_view_payload(context):
    kwargs = context["projection_kwargs"]
    return {
        "background_image": _diag_runtime_value(kwargs["last_caked_background_image_unscaled"]),
        "background": _diag_runtime_value(kwargs["last_caked_background_image_unscaled"]),
        "radial_axis": _diag_runtime_value(kwargs["last_caked_radial_values"]),
        "azimuth_axis": _diag_runtime_value(kwargs["last_caked_azimuth_values"]),
        "raw_azimuth_axis": _diag_runtime_value(kwargs["last_caked_azimuth_values"]),
        "transform_bundle": _diag_runtime_value(kwargs["caked_transform_bundle"]),
    }


def _diag_fit_handoff_dataset(tmp_path):
    if "payload" in _QR_FIT_AUDIT_CACHE:
        return _QR_FIT_AUDIT_CACHE["payload"]
    context, rows = _diag_startup_context_and_rows(tmp_path)
    kwargs = context["projection_kwargs"]
    callbacks = context["projection_callbacks"]
    saved_entries = _diag_manual_entries_for_active_background(context["saved_state"])
    native_background = np.asarray(_diag_runtime_value(kwargs["current_background_native"]))
    display_background = np.asarray(_diag_runtime_value(kwargs["current_background_display"]))
    params = dict(_diag_runtime_value(kwargs["current_geometry_fit_params"]))
    image_size = int(_diag_runtime_value(kwargs["image_size"]))
    source_rows = [dict(entry) for entry in rows["overlay_rows"] if isinstance(entry, Mapping)]
    events = []

    def _source_rows_for_background(_idx, _params=None, **_kwargs):
        return [dict(entry) for entry in source_rows]

    bindings = gf.GeometryFitRuntimeManualDatasetBindings(
        osc_files=("Bi2Se3_5m_5d.osc",),
        current_background_index=0,
        image_size=image_size,
        display_rotate_k=int(kwargs.get("display_rotate_k", hgf.DISPLAY_ROTATE_K)),
        geometry_manual_pairs_for_index=lambda idx: [dict(entry) for entry in saved_entries]
        if int(idx) == 0
        else [],
        load_background_by_index=lambda _idx: (native_background, display_background),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [
            dict(entry) for entry in source_rows
        ],
        geometry_manual_simulated_lookup=callbacks.simulated_lookup,
        geometry_manual_entry_display_coords=callbacks.entry_display_coords,
        unrotate_display_peaks=lambda measured, rotated_shape, *, k=None: (
            geometry_overlay.unrotate_display_peaks(
                measured,
                rotated_shape,
                k=k,
                default_display_rotate_k=hgf.DISPLAY_ROTATE_K,
            )
        ),
        display_to_native_sim_coords=lambda col, row, image_shape: (
            geometry_overlay.display_to_native_sim_coords(
                col,
                row,
                image_shape,
                sim_display_rotate_k=hgf.SIM_DISPLAY_ROTATE_K,
            )
        ),
        select_fit_orientation=geometry_overlay.select_fit_orientation,
        apply_orientation_to_entries=geometry_overlay.apply_orientation_to_entries,
        orient_image_for_fit=geometry_overlay.orient_image_for_fit,
        geometry_manual_project_peaks_to_current_view=lambda projected_rows: [
            dict(entry) for entry in (projected_rows or []) if isinstance(entry, Mapping)
        ],
        geometry_manual_project_peaks_for_background_view=lambda _idx, projected_rows: [
            dict(entry) for entry in (projected_rows or []) if isinstance(entry, Mapping)
        ],
        native_detector_coords_to_bundle_detector_coords=kwargs.get(
            "native_detector_coords_to_bundle_detector_coords"
        ),
        native_detector_coords_to_detector_display_coords=kwargs.get(
            "native_detector_coords_to_detector_display_coords"
        ),
        geometry_manual_source_rows_for_background=_source_rows_for_background,
        geometry_manual_rebuild_source_rows_for_background=_source_rows_for_background,
        geometry_manual_last_source_snapshot_diagnostics=lambda: {"status": "test_ready"},
        geometry_manual_last_simulation_diagnostics=callbacks.last_simulation_diagnostics,
        geometry_manual_match_config=lambda: {},
        pick_uses_caked_space=lambda: False,
        geometry_manual_caked_view_for_index=lambda idx: _diag_caked_view_payload(context)
        if int(idx) == 0
        else None,
        geometry_manual_refresh_pair_entry=callbacks.refresh_entry_geometry,
    )
    dataset = gf.build_geometry_manual_fit_dataset(
        0,
        theta_base=0.0,
        base_fit_params=params,
        manual_dataset_bindings=bindings,
        orientation_cfg={"enabled": False},
        stage_callback=lambda stage, payload: events.append((stage, dict(payload))),
    )
    _QR_FIT_AUDIT_CACHE["payload"] = (context, dataset, events)
    return context, dataset, events


def _diag_fit_audit_rows(dataset):
    rows = [
        dict(row)
        for row in dataset.get("fit_handoff_audit_rows", []) or []
        if isinstance(row, Mapping)
    ]
    return {
        int(row["source_branch_index"]): row
        for row in rows
        if row.get("source_branch_index") is not None
    }


def _diag_entry_is_minus_1_0_10_branch(entry):
    if not isinstance(entry, Mapping):
        return False
    branch = entry.get("source_branch_index")
    try:
        branch = int(branch)
    except Exception:
        return False
    return (
        _diag_q_group_key(entry) == _QR_PICKER_TARGET_Q_GROUP_KEY
        and _diag_hkl(entry) == _QR_PICKER_TARGET_HKL
        and branch in {0, 1}
    )


def _diag_filter_minus_1_0_10_dataset(dataset):
    filtered = dict(dataset)
    pair_fields = (
        "provider_pairs",
        "manual_point_pairs",
        "measured_display",
        "measured_for_fit",
        "initial_pairs_display",
    )
    pair_lists = {
        key: [dict(entry) for entry in dataset.get(key, ()) or () if isinstance(entry, Mapping)]
        for key in pair_fields
    }
    pair_count = max((len(entries) for entries in pair_lists.values()), default=0)
    keep_indices = []
    for index in range(pair_count):
        entries = [
            entries[index]
            for entries in pair_lists.values()
            if index < len(entries) and isinstance(entries[index], Mapping)
        ]
        if any(_diag_entry_is_minus_1_0_10_branch(entry) for entry in entries):
            keep_indices.append(index)
    for key, entries in pair_lists.items():
        filtered[key] = [
            dict(entries[index]) for index in keep_indices if index < len(entries)
        ]
    filtered["pair_count"] = int(len(keep_indices))
    for key in ("source_rows_for_trace", "source_rows", "simulated_peaks"):
        entries = [
            dict(entry)
            for entry in dataset.get(key, ()) or ()
            if isinstance(entry, Mapping)
        ]
        if entries:
            filtered[key] = entries
    spec = dict(dataset.get("spec", {}) or {})
    spec["manual_point_pairs"] = [dict(entry) for entry in filtered["manual_point_pairs"]]
    spec["measured_peaks"] = [dict(entry) for entry in filtered["measured_for_fit"]]
    filtered["spec"] = spec
    return filtered


def _diag_float_pair(value):
    assert isinstance(value, (list, tuple)) and len(value) >= 2
    return (float(value[0]), float(value[1]))


def _diag_residual_record(
    branch,
    *,
    observed_detector_native,
    observed_caked,
    predicted_detector_native,
    predicted_caked,
):
    observed_detector_native = _diag_float_pair(observed_detector_native)
    observed_caked = _diag_float_pair(observed_caked)
    predicted_detector_native = _diag_float_pair(predicted_detector_native)
    predicted_caked = _diag_float_pair(predicted_caked)
    residual_detector = gf._qr_residual_detector_native_px(
        observed_detector_native,
        predicted_detector_native,
    )
    residual_caked = gf._qr_residual_caked_deg(observed_caked, predicted_caked)
    assert residual_detector is not None
    assert residual_caked is not None
    residual_detector_norm = gf._qr_residual_norm(residual_detector)
    residual_caked_norm = gf._qr_residual_norm(residual_caked)
    assert residual_detector_norm is not None
    assert residual_caked_norm is not None
    return {
        "branch": int(branch),
        "observed_detector_native": observed_detector_native,
        "observed_caked": observed_caked,
        "predicted_detector_native": predicted_detector_native,
        "predicted_caked": predicted_caked,
        "residual_detector": residual_detector,
        "residual_caked": residual_caked,
        "residual_detector_norm": float(residual_detector_norm),
        "residual_norm": float(residual_caked_norm),
    }


def _diag_total_residual_norm(records):
    values = []
    for record in records.values():
        residual = record["residual_caked"]
        values.extend([float(residual[0]), float(residual[1])])
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


def _diag_fmt_pair(value):
    return f"({float(value[0]):.6f}, {float(value[1]):.6f})"


def _diag_print_fit_step_table(before_records, after_records):
    print(
        "branch | observed_caked | prediction_before | residual_before | "
        "prediction_after | residual_after | residual_norm_before | "
        "residual_norm_after"
    )
    for branch in sorted(before_records):
        before = before_records[branch]
        after = after_records[branch]
        print(
            f"{branch} | "
            f"{_diag_fmt_pair(before['observed_caked'])} | "
            f"{_diag_fmt_pair(before['predicted_caked'])} | "
            f"{_diag_fmt_pair(before['residual_caked'])} | "
            f"{_diag_fmt_pair(after['predicted_caked'])} | "
            f"{_diag_fmt_pair(after['residual_caked'])} | "
            f"{before['residual_norm']:.9f} | "
            f"{after['residual_norm']:.9f}"
        )


def _diag_print_residual_table(title, records):
    print(title)
    print(
        "branch | observed_detector_native_px | predicted_detector_native_px | "
        "residual_detector_native_px | observed_caked_deg | predicted_caked_deg | "
        "residual_caked_deg | residual_detector_norm_px | residual_caked_norm_deg"
    )
    for branch in sorted(records):
        record = records[branch]
        print(
            f"{branch} | "
            f"{_diag_fmt_pair(record['observed_detector_native'])} | "
            f"{_diag_fmt_pair(record['predicted_detector_native'])} | "
            f"{_diag_fmt_pair(record['residual_detector'])} | "
            f"{_diag_fmt_pair(record['observed_caked'])} | "
            f"{_diag_fmt_pair(record['predicted_caked'])} | "
            f"{_diag_fmt_pair(record['residual_caked'])} | "
            f"{record['residual_detector_norm']:.9f} | "
            f"{record['residual_norm']:.9f}"
        )


def _diag_records_from_audit_rows(rows_by_branch):
    return {
        int(branch): _diag_residual_record(
            branch,
            observed_detector_native=row.get(
                "observed_detector_native_px",
                row["fit_observed_detector_native_px"],
            ),
            observed_caked=row.get("observed_caked_deg", row["fit_observed_caked_deg"]),
            predicted_detector_native=row.get(
                "predicted_detector_native_px",
                row["fit_prediction_detector_native_px"],
            ),
            predicted_caked=row.get("predicted_caked_deg", row["fit_prediction_caked_deg"]),
        )
        for branch, row in rows_by_branch.items()
    }


def _diag_records_close(left, right, *, atol=1.0e-9):
    assert set(left) == set(right)
    for branch in left:
        for key in (
            "predicted_detector_native",
            "predicted_caked",
            "residual_detector",
            "residual_caked",
        ):
            assert np.allclose(left[branch][key], right[branch][key], atol=atol, rtol=0.0), (
                branch,
                key,
                left[branch][key],
                right[branch][key],
            )


def _diag_saved_bool(saved_state, name, default=False):
    variables = saved_state.get("variables", {}) if isinstance(saved_state, Mapping) else {}
    value = variables.get(name, default) if isinstance(variables, Mapping) else default
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _diag_geometry_fit_var_names(saved_state):
    return gf.current_geometry_fit_var_names(
        fit_zb=_diag_saved_bool(saved_state, "fit_zb_var"),
        fit_zs=_diag_saved_bool(saved_state, "fit_zs_var"),
        fit_theta=_diag_saved_bool(saved_state, "fit_theta_var"),
        fit_psi_z=_diag_saved_bool(saved_state, "fit_psi_z_var"),
        fit_chi=_diag_saved_bool(saved_state, "fit_chi_var"),
        fit_cor=_diag_saved_bool(saved_state, "fit_cor_var"),
        fit_gamma=_diag_saved_bool(saved_state, "fit_gamma_var"),
        fit_Gamma=_diag_saved_bool(saved_state, "fit_Gamma_var"),
        fit_dist=_diag_saved_bool(saved_state, "fit_dist_var"),
        fit_a=_diag_saved_bool(saved_state, "fit_a_var"),
        fit_c=_diag_saved_bool(saved_state, "fit_c_var"),
        fit_center_x=_diag_saved_bool(saved_state, "fit_center_x_var"),
        fit_center_y=_diag_saved_bool(saved_state, "fit_center_y_var"),
        use_shared_theta_offset=False,
    )


def _diag_build_minus_1_0_10_fit_request(
    context,
    dataset,
    *,
    seed_multistart_enabled=True,
    objective_trace_enabled=False,
    qr_only_objective=False,
    objective_dry_run_only=False,
    optimizer_overrides=None,
    bounds_overrides=None,
    x_scale_overrides=None,
    params_overrides=None,
):
    kwargs = context["projection_kwargs"]
    saved_state = context["saved_state"]
    params = dict(_diag_runtime_value(kwargs["current_geometry_fit_params"]))
    if isinstance(params_overrides, Mapping):
        params.update(dict(params_overrides))
    fit_dataset = (
        _diag_filter_minus_1_0_10_dataset(dataset)
        if bool(qr_only_objective)
        else dataset
    )
    image_size = int(_diag_runtime_value(kwargs["image_size"]))
    defaults = hgf._build_runtime_defaults(saved_state)
    var_names = _diag_geometry_fit_var_names(saved_state)
    domains = hgf._headless_runtime_geometry_fit_parameter_domains(
        fit_config=defaults.fit_config,
        current_params=params,
        image_size=image_size,
        names=var_names,
        use_shared_theta_offset=False,
    )
    runtime_cfg = gf.build_geometry_fit_runtime_config(
        defaults.fit_config.get("geometry", {})
        if isinstance(defaults.fit_config, Mapping)
        else {},
        params,
        {},
        domains,
        candidate_param_names=var_names,
        caked_roi_enabled=False,
    )
    runtime_cfg = gf.apply_manual_caked_point_geometry_fit_runtime_overrides(
        runtime_cfg,
        joint_background_mode=False,
    )
    optimizer_cfg = dict(runtime_cfg.get("optimizer", {}) or {})
    optimizer_cfg.update(
        {
            "max_nfev": 1,
            "restarts": 0,
            "workers": 1,
            "parallel_mode": "off",
            "worker_numba_threads": 0,
            "loss": "linear",
            "seed_multistart_enabled": bool(seed_multistart_enabled),
            "seed_multistart": bool(seed_multistart_enabled),
            "objective_trace_enabled": bool(objective_trace_enabled),
            "objective_dry_run_only": bool(objective_dry_run_only),
            "objective_trace_max_evals": 512,
        }
    )
    if isinstance(optimizer_overrides, Mapping):
        optimizer_cfg.update(dict(optimizer_overrides))
    if qr_only_objective:
        optimizer_cfg.update(
            {
                "q_group_line_constraints": False,
                "q_group_line_constraints_enabled": False,
                "q_group_line_angle_weight": 0.0,
                "q_group_line_offset_weight": 0.0,
            }
        )
    seed_search_cfg = dict(runtime_cfg.get("seed_search", {}) or {})
    seed_search_cfg.update(
        {
            "enabled": bool(seed_multistart_enabled),
            "prescore_top_k": 1,
            "n_global": 0,
            "n_jitter": 0,
        }
    )
    runtime_cfg["optimizer"] = optimizer_cfg
    runtime_cfg["solver"] = dict(optimizer_cfg)
    runtime_cfg["seed_search"] = seed_search_cfg
    if isinstance(bounds_overrides, Mapping):
        runtime_cfg["bounds"] = dict(bounds_overrides)
    if isinstance(x_scale_overrides, Mapping):
        runtime_cfg["x_scale"] = dict(x_scale_overrides)
    if qr_only_objective:
        runtime_cfg["priors"] = {}
    runtime_cfg["use_numba"] = False
    runtime_cfg["allow_unsafe_runtime"] = False

    dataset_spec = dict(fit_dataset.get("spec", {}) or {})
    prepared_run = gf.GeometryFitPreparedRun(
        fit_params=params,
        selected_background_indices=[0],
        background_theta_values=[float(params.get("theta_initial", 0.0))],
        joint_background_mode=False,
        current_dataset=dict(fit_dataset),
        dataset_infos=[{"dataset_index": 0, "background_index": 0}],
        dataset_specs=[dataset_spec],
        start_cmd_line="[ra-sim] minus_1_0_10 diagnostic fit step",
        start_log_sections=[],
        max_display_markers=100,
        geometry_runtime_cfg=runtime_cfg,
        stage_timing_s={},
    )
    solver_inputs = gf.GeometryFitRuntimeSolverInputs(
        miller=np.asarray(_diag_runtime_value(kwargs["miller"]), dtype=float),
        intensities=np.asarray(_diag_runtime_value(kwargs["intensities"]), dtype=float),
        image_size=image_size,
    )
    request = gf.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=var_names,
        solver_inputs=solver_inputs,
    )
    return request, var_names


def _diag_parameter_values_from_result(request, var_names, result):
    before = {name: float(request.params[name]) for name in var_names}
    after = dict(before)
    raw_x = getattr(result, "x", None)
    if raw_x is not None:
        try:
            x_values = np.asarray(raw_x, dtype=float).ravel()
        except Exception:
            x_values = np.asarray([], dtype=float)
        if x_values.size >= len(var_names):
            after = {
                str(name): float(value)
                for name, value in zip(var_names, x_values[: len(var_names)])
            }
    return before, after


def _diag_result_target_predictions(result):
    predictions = {}
    diagnostics = getattr(result, "point_match_diagnostics", None) or []
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            continue
        branch = raw.get("source_branch_index")
        if branch is None:
            continue
        if _diag_q_group_key(raw) != _QR_PICKER_TARGET_Q_GROUP_KEY:
            continue
        if _diag_hkl(raw) != _QR_PICKER_TARGET_HKL:
            continue
        if int(branch) not in {0, 1}:
            continue
        if raw.get("simulated_two_theta_deg") is None or raw.get("simulated_phi_deg") is None:
            continue
        predictions[int(branch)] = {
            "predicted_caked": (
                float(raw["simulated_two_theta_deg"]),
                float(raw["simulated_phi_deg"]),
            ),
            "predicted_detector_native": (
                float(raw.get("simulated_x", np.nan)),
                float(raw.get("simulated_y", np.nan)),
            ),
            "measured_caked": (
                float(raw.get("measured_two_theta_deg", np.nan)),
                float(raw.get("measured_phi_deg", np.nan)),
            ),
            }
    return predictions


def _diag_after_records_from_fit_result(baseline_records, result):
    result_predictions = _diag_result_target_predictions(result)
    after_records = {}
    for branch, before in baseline_records.items():
        assert branch in result_predictions
        prediction = result_predictions[branch]
        after_records[branch] = _diag_residual_record(
            branch,
            observed_detector_native=before["observed_detector_native"],
            observed_caked=before["observed_caked"],
            predicted_detector_native=prediction["predicted_detector_native"],
            predicted_caked=prediction["predicted_caked"],
        )
        assert after_records[branch]["observed_detector_native"] == before[
            "observed_detector_native"
        ]
        assert after_records[branch]["observed_caked"] == before["observed_caked"]
    return after_records


def _diag_predictions_changed(before_records, after_records):
    return any(
        not np.allclose(
            after_records[branch]["predicted_caked"],
            before_records[branch]["predicted_caked"],
            atol=1.0e-12,
            rtol=0.0,
        )
        for branch in before_records
    )


def _diag_fixed_manual_pair_no_decrease_reason(
    *,
    q_residual_count,
    valid_evaluation,
    params_changed,
    prediction_changed,
    result,
):
    trace = getattr(result, "objective_trace", None) or []
    for record in trace:
        for row in _diag_target_objective_rows(record):
            if row.get("resolution_reason") == "prediction_branch_source_switched":
                return "prediction_branch_source_switched"
    for row in getattr(result, "point_match_diagnostics", None) or []:
        if (
            _diag_entry_is_minus_1_0_10_branch(row)
            and row.get("resolution_reason") == "prediction_branch_source_switched"
        ):
            return "prediction_branch_source_switched"
    summary = getattr(result, "point_match_summary", None) or {}
    if int(summary.get("prediction_branch_source_switched_count", 0) or 0) > 0:
        return "prediction_branch_source_switched"
    if int(q_residual_count) < 2:
        return "objective_excludes_qr_residual"
    if not valid_evaluation:
        return "optimizer_step_rejected"
    if not params_changed:
        bound_summary = getattr(result, "bound_proximity_summary", {}) or {}
        hugging = []
        if isinstance(bound_summary, Mapping):
            hugging = list(bound_summary.get("hugging_parameters", []) or [])
        return "bounds_block_update" if hugging else "parameterization_cannot_move_qr_prediction"
    if not prediction_changed:
        return "parameterization_cannot_move_qr_prediction"
    if not bool(getattr(result, "success", False)):
        return "optimizer_step_rejected"
    if len(trace) >= 2:
        before = trace[0]
        after = trace[-1]
        before_total = float(before.get("residual_norm", np.nan))
        after_total = float(after.get("residual_norm", np.nan))
        if np.isfinite(before_total) and np.isfinite(after_total) and after_total < before_total:
            return "total_objective_improved_but_qr_worsened"
    return "optimizer_residual_vector_mismatch"


def _diag_objective_trace(result):
    trace = getattr(result, "objective_trace", None) or []
    return [dict(record) for record in trace if isinstance(record, Mapping)]


def _diag_target_objective_components(record):
    components = []
    for raw in record.get("point_components", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("component")) not in {
            "delta_two_theta_deg",
            "wrapped_delta_phi_deg",
        }:
            continue
        if (
            _diag_q_group_key(raw) == _QR_PICKER_TARGET_Q_GROUP_KEY
            and _diag_hkl(raw) == _QR_PICKER_TARGET_HKL
        ):
            try:
                if int(raw.get("source_branch_index")) in {0, 1}:
                    components.append(dict(raw))
            except Exception:
                continue
    return components


def _diag_target_objective_rows(record):
    rows = []
    for raw in record.get("point_rows", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if (
            _diag_q_group_key(raw) == _QR_PICKER_TARGET_Q_GROUP_KEY
            and _diag_hkl(raw) == _QR_PICKER_TARGET_HKL
        ):
            try:
                if int(raw.get("source_branch_index")) in {0, 1}:
                    rows.append(dict(raw))
            except Exception:
                continue
    return rows


def _diag_qr_norm_from_objective_record(record):
    values = [
        float(component["unweighted_value"])
        for component in _diag_target_objective_components(record)
    ]
    return float(np.linalg.norm(np.asarray(values, dtype=float))) if values else float("nan")


def _diag_non_qr_point_norm_from_objective_record(record):
    values = []
    for raw in record.get("point_components", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if (
            _diag_q_group_key(raw) == _QR_PICKER_TARGET_Q_GROUP_KEY
            and _diag_hkl(raw) == _QR_PICKER_TARGET_HKL
        ):
            try:
                if int(raw.get("source_branch_index")) in {0, 1}:
                    continue
            except Exception:
                pass
        values.append(float(raw.get("weighted_value", np.nan)))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.linalg.norm(arr)) if arr.size else 0.0


def _diag_print_optimizer_residual_vector_table(record):
    print("optimizer_residual_vector_table")
    print(
        "component_index | branch | q_group_key | hkl | coordinate_space | units | "
        "observed_source | predicted_source | observed_value | predicted_value | "
        "residual_unweighted | weight | residual_weighted"
    )
    for component_index, component in enumerate(record.get("point_components", []) or []):
        if not isinstance(component, Mapping):
            continue
        component_name = str(component.get("component", ""))
        axis_index = 1 if "phi" in component_name else 0
        observed_pair = component.get("observed_caked_deg", (np.nan, np.nan))
        predicted_pair = component.get("predicted_caked_deg", (np.nan, np.nan))
        try:
            observed_value = float(observed_pair[axis_index])
        except Exception:
            observed_value = float("nan")
        try:
            predicted_value = float(predicted_pair[axis_index])
        except Exception:
            predicted_value = float("nan")
        print(
            f"{int(component_index)} | "
            f"{component.get('source_branch_index')} | "
            f"{component.get('q_group_key')} | "
            f"{component.get('hkl')} | "
            f"{component.get('coordinate_space')} | "
            f"{component.get('units')} | "
            f"{component.get('observed_source')} | "
            f"{component.get('predicted_source')} | "
            f"{observed_value:.9f} | "
            f"{predicted_value:.9f} | "
            f"{float(component.get('unweighted_value', np.nan)):.9f} | "
            f"{float(component.get('weight', np.nan)):.9f} | "
            f"{float(component.get('weighted_value', np.nan)):.9f}"
        )


def _diag_optimizer_rows_by_branch(record):
    rows = {}
    for row in _diag_target_objective_rows(record):
        try:
            branch = int(row.get("source_branch_index"))
        except Exception:
            continue
        rows[branch] = dict(row)
    return rows


def _diag_pair_or_nan(value):
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            pass
    return (float("nan"), float("nan"))


def _diag_row_pair(row, key, fallback_keys=()):
    value = row.get(key)
    point = _diag_pair_or_nan(value)
    if np.isfinite(point[0]) and np.isfinite(point[1]):
        return point
    for left_key, right_key in fallback_keys:
        if left_key in row or right_key in row:
            point = _diag_pair_or_nan((row.get(left_key), row.get(right_key)))
            if np.isfinite(point[0]) and np.isfinite(point[1]):
                return point
    return point


def _diag_project_native_for_test(dataset, native_point):
    point = _diag_pair_or_nan(native_point)
    if not (np.isfinite(point[0]) and np.isfinite(point[1])):
        return (float("nan"), float("nan"))
    spec = dict(dataset.get("spec", {}) or {})
    projector = spec.get("fit_space_projector")
    if not callable(projector):
        return (float("nan"), float("nan"))
    projected, _reason = gf._geometry_fit_audit_project_native_to_caked(
        point,
        fit_space_projector=projector,
        base_fit_params={"theta_initial": 0.0},
    )
    if projected is None:
        return (float("nan"), float("nan"))
    return (float(projected[0]), float(projected[1]))


def _diag_print_handoff_optimizer_prediction_table(rows_by_branch, optimizer_rows):
    print("handoff_optimizer_prediction_table")
    print(
        "branch | observed_caked | handoff_prediction_source | "
        "handoff_prediction_detector_display_px | handoff_prediction_detector_native_px | "
        "handoff_prediction_caked_deg | optimizer_prediction_source | "
        "optimizer_prediction_detector_display_px | optimizer_prediction_detector_native_px | "
        "optimizer_prediction_caked_deg | optimizer_minus_handoff_prediction_delta_deg | "
        "source_field_used_by_resolver | projection_callback_bundle_identity"
    )
    for branch in sorted(rows_by_branch):
        handoff = rows_by_branch[branch]
        optimizer = optimizer_rows.get(branch, {})
        handoff_pred = _diag_row_pair(handoff, "fit_prediction_caked_deg")
        optimizer_pred = _diag_row_pair(optimizer, "predicted_caked_deg")
        delta = (
            float(optimizer_pred[0]) - float(handoff_pred[0]),
            gf._geometry_fit_audit_phi_delta(float(optimizer_pred[1]), float(handoff_pred[1])),
        )
        optimizer_native = _diag_row_pair(
            optimizer,
            "display_to_native_saved_sim_detector_display_px",
            (("simulated_native_col", "simulated_native_row"),),
        )
        projection_identity = (
            f"{optimizer.get('fit_space_projector_kind')}:"
            f"{optimizer.get('cake_bundle_signature')}"
        )
        print(
            f"{branch} | "
            f"{_diag_fmt_pair(_diag_row_pair(handoff, 'fit_observed_caked_deg'))} | "
            f"{handoff.get('fit_prediction_source')} | "
            f"{_diag_fmt_pair(_diag_row_pair(handoff, 'fit_prediction_detector_display_px'))} | "
            f"{_diag_fmt_pair(_diag_row_pair(handoff, 'fit_prediction_detector_native_px'))} | "
            f"{_diag_fmt_pair(handoff_pred)} | "
            f"{optimizer.get('predicted_source')} | "
            f"{_diag_fmt_pair(_diag_row_pair(optimizer, 'saved_sim_detector_display_px'))} | "
            f"{_diag_fmt_pair(optimizer_native)} | "
            f"{_diag_fmt_pair(optimizer_pred)} | "
            f"{_diag_fmt_pair(delta)} | "
            f"{optimizer.get('provider_local_saved_sim_detector_source_field')} | "
            f"{projection_identity}"
        )


def _diag_print_saved_sim_native_diagnostics(dataset, rows_by_branch, optimizer_rows):
    print("provider_local_saved_sim_detector_native_px_diagnostics")
    print(
        "branch | saved_sim_detector_display_px | saved_sim_detector_native_px | "
        "display_to_native(saved_sim_detector_display_px) | caked_from_saved_native | "
        "caked_from_display_to_native | handoff_prediction_caked | "
        "saved_sim_detector_native_rejected_reason"
    )
    for branch in sorted(rows_by_branch):
        row = optimizer_rows.get(branch, {})
        saved_native = _diag_row_pair(row, "saved_sim_detector_native_px")
        canonical_native = _diag_row_pair(
            row,
            "display_to_native_saved_sim_detector_display_px",
        )
        caked_from_saved_native = _diag_row_pair(row, "caked_from_saved_native")
        if not (np.isfinite(caked_from_saved_native[0]) and np.isfinite(caked_from_saved_native[1])):
            caked_from_saved_native = _diag_project_native_for_test(dataset, saved_native)
        caked_from_display_to_native = _diag_row_pair(row, "caked_from_display_to_native")
        if not (
            np.isfinite(caked_from_display_to_native[0])
            and np.isfinite(caked_from_display_to_native[1])
        ):
            caked_from_display_to_native = _diag_project_native_for_test(
                dataset,
                canonical_native,
            )
        print(
            f"{branch} | "
            f"{_diag_fmt_pair(_diag_row_pair(row, 'saved_sim_detector_display_px'))} | "
            f"{_diag_fmt_pair(saved_native)} | "
            f"{_diag_fmt_pair(canonical_native)} | "
            f"{_diag_fmt_pair(caked_from_saved_native)} | "
            f"{_diag_fmt_pair(caked_from_display_to_native)} | "
            f"{_diag_fmt_pair(_diag_row_pair(rows_by_branch[branch], 'fit_prediction_caked_deg'))} | "
            f"{row.get('saved_sim_detector_native_rejected_reason')}"
        )


def _diag_print_branch_identity_stability_table(trace):
    print("branch_identity_stability_table")
    print(
        "eval | branch | q_group_key | hkl | source_table_index | source_row_index | "
        "source_branch_index | source_peak_index | branch_id | prediction_source | "
        "resolution_reason | source_row_rejection_reason | predicted_caked_deg"
    )
    for record in trace:
        for row in _diag_target_objective_rows(record):
            predicted = row.get("predicted_caked_deg", (np.nan, np.nan))
            print(
                f"{int(record.get('eval_index', -1))} | "
                f"{row.get('source_branch_index')} | "
                f"{row.get('q_group_key')} | "
                f"{row.get('hkl')} | "
                f"{row.get('source_table_index')} | "
                f"{row.get('source_row_index')} | "
                f"{row.get('source_branch_index')} | "
                f"{row.get('source_peak_index')} | "
                f"{row.get('branch_id')} | "
                f"{row.get('predicted_source')} | "
                f"{row.get('resolution_reason')} | "
                f"{row.get('source_row_rejection_reason')} | "
                f"{_diag_fmt_pair(predicted)}"
            )


def _diag_pick_fit_failure_reason(
    *,
    var_names,
    q_residual_present,
    params_changed,
    prediction_changed,
    result_success,
    result_message,
    before_total_norm,
    after_total_norm,
):
    if after_total_norm < before_total_norm - 1.0e-9:
        return None
    if not q_residual_present:
        return "qr_residual_not_in_fit_objective"
    if not var_names:
        return "optimizer_parameters_locked"
    if params_changed and not prediction_changed:
        return "dynamic_prediction_not_recomputed"
    if not params_changed and not result_success:
        return "step_rejected_by_optimizer"
    if params_changed and after_total_norm >= before_total_norm - 1.0e-9:
        return "fit_not_sensitive_to_qr_residual"
    if not params_changed:
        return "optimizer_parameters_locked"
    if result_message:
        return f"other={result_message}"
    return "other=residual_not_decreased"


def _diag_recompute_fit_audit_rows(context, dataset, params=None):
    if params is None:
        params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    rows = gf.build_geometry_fit_qr_handoff_audit_rows(
        dataset,
        base_fit_params=dict(params),
    )
    return _diag_fit_audit_rows({"fit_handoff_audit_rows": rows})


def _diag_run_controlled_minus_1_0_10_fit(
    context,
    dataset,
    *,
    seed_multistart_enabled=True,
    objective_trace_enabled=False,
    qr_only_objective=False,
    objective_dry_run_only=False,
    optimizer_overrides=None,
    bounds_overrides=None,
    x_scale_overrides=None,
    params_overrides=None,
):
    from ra_sim.fitting.optimization import fit_geometry_parameters

    use_cache = not any(
        isinstance(value, Mapping)
        for value in (
            optimizer_overrides,
            bounds_overrides,
            x_scale_overrides,
            params_overrides,
        )
    )
    cache_key = (
        bool(seed_multistart_enabled),
        bool(objective_trace_enabled),
        bool(qr_only_objective),
        bool(objective_dry_run_only),
    )
    if use_cache:
        cached = _QR_FIT_STEP_CACHE.get(cache_key)
        if cached is not None:
            return cached

    request, var_names = _diag_build_minus_1_0_10_fit_request(
        context,
        dataset,
        seed_multistart_enabled=seed_multistart_enabled,
        objective_trace_enabled=objective_trace_enabled,
        qr_only_objective=qr_only_objective,
        objective_dry_run_only=objective_dry_run_only,
        optimizer_overrides=optimizer_overrides,
        bounds_overrides=bounds_overrides,
        x_scale_overrides=x_scale_overrides,
        params_overrides=params_overrides,
    )
    result = gf.solve_geometry_fit_request(
        request,
        solve_fit=fit_geometry_parameters,
        status_callback=lambda _message: None,
        live_update_callback=lambda _payload: None,
    )
    before_params, after_params = _diag_parameter_values_from_result(
        request,
        var_names,
        result,
    )
    params_changed = any(
        not np.isclose(before_params[name], after_params.get(name, before_params[name]))
        for name in before_params
    )
    nfev = int(getattr(result, "nfev", 0) or 0)
    step_executed = bool(nfev > 0)
    payload = {
        "request": request,
        "var_names": var_names,
        "result": result,
        "before_params": before_params,
        "after_params": after_params,
        "params_changed": bool(params_changed),
        "step_executed": bool(step_executed),
        "valid_evaluation": bool(nfev > 0),
        "params_accepted": bool(nfev > 0 and getattr(result, "x", None) is not None),
    }
    if use_cache:
        _QR_FIT_STEP_CACHE[cache_key] = payload
    return payload


def _diag_delta_pair_close(delta, *, theta_tol=0.25, phi_tol=0.5):
    assert isinstance(delta, (list, tuple)) and len(delta) >= 2
    assert abs(float(delta[0])) <= float(theta_tol)
    assert abs(float(delta[1])) <= float(phi_tol)


def test_minus_1_0_10_fit_handoff_audit_prints_visual_and_fit_values(tmp_path) -> None:
    _context, dataset, events = _diag_fit_handoff_dataset(tmp_path)
    audit_lines = list(dataset.get("fit_handoff_audit_lines", []))
    print("\n".join(str(line) for line in audit_lines))
    assert audit_lines
    assert audit_lines[0] == "[ra-sim] Qr/Qz fit handoff audit"
    assert any(
        stage == "cmd_line" and payload.get("text") == "[ra-sim] Qr/Qz fit handoff audit"
        for stage, payload in events
    )
    rows_by_branch = _diag_fit_audit_rows(dataset)
    assert set(rows_by_branch) == {0, 1}
    for branch, row in rows_by_branch.items():
        assert row["q_group_key"] == _QR_PICKER_TARGET_Q_GROUP_KEY
        assert row["hkl"] == _QR_PICKER_TARGET_HKL
        assert row["source_branch_index"] == branch
        for field in (
            "observed_refined_detector_display_px",
            "observed_refined_detector_native_px",
            "observed_refined_caked_deg",
            "sim_nominal_detector_display_px",
            "sim_nominal_detector_native_px",
            "fit_observed_detector_display_px",
            "fit_observed_detector_native_px",
            "fit_observed_caked_deg",
            "fit_prediction_source",
        ):
            assert field in row


def test_minus_1_0_10_fit_step_reduces_qr_residual(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    assert set(rows_by_branch) == {0, 1}

    before_records = _diag_records_from_audit_rows(rows_by_branch)
    for branch, row in rows_by_branch.items():
        assert row["q_group_key"] == _QR_PICKER_TARGET_Q_GROUP_KEY
        assert row["hkl"] == _QR_PICKER_TARGET_HKL
        assert row["source_branch_index"] == branch
        assert row["fit_prediction_source"] == "dynamic_current_simulation"
        assert "cache_fresh_row" not in str(row.get("fit_prediction_source", ""))
        assert row["fit_observed_caked_deg"] is not row["fit_prediction_caked_deg"]
        assert row["fit_observed_detector_native_px"] is not row[
            "fit_prediction_detector_native_px"
        ]
        assert not np.allclose(
            _diag_float_pair(row["fit_observed_caked_deg"]),
            _diag_float_pair(row["fit_prediction_caked_deg"]),
            atol=1.0e-12,
            rtol=0.0,
        )

    fit_run = _diag_run_controlled_minus_1_0_10_fit(context, dataset)
    request = fit_run["request"]
    var_names = fit_run["var_names"]
    result = fit_run["result"]
    before_params = fit_run["before_params"]
    after_params = fit_run["after_params"]
    params_changed = fit_run["params_changed"]
    step_executed = fit_run["step_executed"]
    handoff_summary = request.refinement_config.get("optimizer_request_handoff_summary", {})
    assert int(handoff_summary.get("fallback_row_count", 0)) == 0
    assert int(handoff_summary.get("fixed_source_pair_count", 0)) >= 2

    after_records = {}
    target_predictions_available = False
    if step_executed:
        result_predictions = _diag_result_target_predictions(result)
        target_predictions_available = all(
            branch in result_predictions for branch in before_records
        )
        for branch, before in before_records.items():
            prediction = result_predictions.get(branch, {})
            predicted_detector_native = prediction.get(
                "predicted_detector_native",
                before["predicted_detector_native"],
            )
            predicted_caked = prediction.get("predicted_caked", before["predicted_caked"])
            after_records[branch] = _diag_residual_record(
                branch,
                observed_detector_native=before["observed_detector_native"],
                observed_caked=before["observed_caked"],
                predicted_detector_native=predicted_detector_native,
                predicted_caked=predicted_caked,
            )
            assert after_records[branch]["observed_caked"] == before["observed_caked"]
            assert after_records[branch]["observed_detector_native"] == before[
                "observed_detector_native"
            ]
    else:
        after_records = {branch: dict(record) for branch, record in before_records.items()}
        repeated_rows = _diag_recompute_fit_audit_rows(context, dataset)
        repeated_records = _diag_records_from_audit_rows(repeated_rows)
        _diag_records_close(before_records, repeated_records)

    prediction_changed = any(
        not np.allclose(
            after_records[branch]["predicted_caked"],
            before_records[branch]["predicted_caked"],
            atol=1.0e-12,
            rtol=0.0,
        )
        for branch in before_records
    )
    if params_changed and target_predictions_available:
        assert prediction_changed, "dynamic_prediction_not_recomputed"

    point_match_summary = getattr(result, "point_match_summary", {}) or {}
    q_residual_count = int(point_match_summary.get("manual_caked_residual_row_count", 0))
    trace = _diag_objective_trace(result)
    q_residual_present = bool(trace and _diag_target_objective_components(trace[0]))
    target_source_switched = any(
        row.get("resolution_reason") == "prediction_branch_source_switched"
        for record in trace
        for row in _diag_target_objective_rows(record)
    )
    assert q_residual_present or target_source_switched, "qr_residual_not_in_fit_objective"

    before_total_norm = _diag_total_residual_norm(before_records)
    after_total_norm = _diag_total_residual_norm(after_records)
    result_success = bool(getattr(result, "success", False))
    result_message = str(getattr(result, "message", "") or "")
    first_reason = _diag_pick_fit_failure_reason(
        var_names=var_names,
        q_residual_present=q_residual_present,
        params_changed=params_changed,
        prediction_changed=prediction_changed,
        result_success=result_success,
        result_message=result_message,
        before_total_norm=before_total_norm,
        after_total_norm=after_total_norm,
    )
    if target_source_switched:
        first_reason = "prediction_branch_source_switched"

    optimizer_cfg = request.refinement_config.get("optimizer", {})
    optimizer_method = str(
        optimizer_cfg.get("method")
        or optimizer_cfg.get("least_squares_method")
        or "least_squares"
    )
    print("[ra-sim] (-1,0,10) Qr/Qz fit-step diagnostic")
    _diag_print_residual_table("baseline_residual_table", before_records)
    if step_executed:
        _diag_print_fit_step_table(before_records, after_records)
    else:
        print("fit_step_executed=no")
        print("post_fit_residual=<unavailable reason=no accepted optimizer step>")
    print("fit_metadata")
    print(f"parameters_varied={list(var_names)}")
    print(f"parameter_values_before={before_params}")
    print(f"parameter_values_after={after_params}")
    print(f"optimizer_method={optimizer_method}")
    print(f"number_of_evaluations={int(getattr(result, 'nfev', 0) or 0)}")
    print(f"success={result_success}")
    print(f"message={result_message}")
    print(f"manual_caked_residual_row_count={q_residual_count}")
    print(f"residual_norm_before={before_total_norm:.9f}")
    print(f"residual_norm_after={after_total_norm:.9f}")
    print(
        "qr_residual_reduced="
        f"{str(after_total_norm < before_total_norm - 1.0e-9).lower()}"
    )
    if first_reason is not None:
        print(f"first_failing_reason={first_reason}")

    if result_success and target_predictions_available:
        assert (
            after_total_norm <= before_total_norm + 1.0e-9
        ), "fitter silently reported success while residual norm increased"


def test_minus_1_0_10_residual_definition_consistent_between_cmd_and_fit(
    tmp_path,
) -> None:
    _context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    audit_lines = [str(line) for line in dataset.get("fit_handoff_audit_lines", [])]
    records = _diag_records_from_audit_rows(rows_by_branch)
    _diag_print_residual_table("baseline_residual_table", records)
    assert set(rows_by_branch) == {0, 1}
    for branch, row in rows_by_branch.items():
        detector_residual = gf._qr_residual_detector_native_px(
            row["observed_detector_native_px"],
            row["predicted_detector_native_px"],
        )
        caked_residual = gf._qr_residual_caked_deg(
            row["observed_caked_deg"],
            row["predicted_caked_deg"],
        )
        assert detector_residual is not None
        assert caked_residual is not None
        assert np.allclose(row["residual_detector_native_px"], detector_residual)
        assert np.allclose(row["fit_residual_detector_native_px"], detector_residual)
        assert np.allclose(row["residual_caked_deg"], caked_residual)
        assert np.allclose(row["fit_residual_caked_deg"], caked_residual)
        assert row["residual_sign_convention"] == "predicted - observed"
        assert row["residual_detector_native_units"] == "px"
        assert row["residual_caked_units"] == "deg"
        assert np.allclose(
            row["geometry_minus_sim_detector_native_px"],
            -np.asarray(detector_residual, dtype=float),
        )
        assert np.allclose(
            row["geometry_minus_sim_caked_deg"],
            -np.asarray(caked_residual, dtype=float),
        )
        assert np.allclose(
            row["fit_observed_minus_fit_prediction_caked_delta_deg"],
            row["geometry_minus_sim_caked_deg"],
        )
        for field in (
            "observed_detector_native_px",
            "predicted_detector_native_px",
            "residual_detector_native_px",
            "observed_caked_deg",
            "predicted_caked_deg",
            "residual_caked_deg",
            "residual_sign_convention",
            "objective_space",
        ):
            expected = (
                f"{field}="
                f"{gf._geometry_fit_audit_value_text(row.get(field))}"
            )
            assert any(expected in line for line in audit_lines), (branch, expected)


def test_minus_1_0_10_observed_is_background_predicted_is_simulation(tmp_path) -> None:
    _context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    assert set(rows_by_branch) == {0, 1}
    for row in rows_by_branch.values():
        assert row["observed_source"] == "background/manual"
        assert row["predicted_source"] == "simulation"
        assert row["fit_prediction_source"] == "dynamic_current_simulation"
        assert row["fit_prediction_is_dynamic"] == "yes"
        assert np.allclose(
            row["observed_detector_native_px"],
            row["observed_refined_detector_native_px"],
        )
        assert np.allclose(row["observed_caked_deg"], row["observed_refined_caked_deg"])
        assert np.allclose(
            row["predicted_detector_native_px"],
            row["fit_prediction_detector_native_px"],
        )
        assert np.allclose(row["predicted_caked_deg"], row["fit_prediction_caked_deg"])
        assert row["observed_source"] != row["predicted_source"]
        assert row["observed_detector_native_px"] is not row["predicted_detector_native_px"]
        assert not np.allclose(
            row["observed_detector_native_px"],
            row["predicted_detector_native_px"],
            atol=1.0e-12,
            rtol=0.0,
        )


def test_minus_1_0_10_same_params_recompute_same_prediction_and_residual(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_first = _diag_recompute_fit_audit_rows(context, dataset)
    rows_second = _diag_recompute_fit_audit_rows(context, dataset)
    records_first = _diag_records_from_audit_rows(rows_first)
    records_second = _diag_records_from_audit_rows(rows_second)
    _diag_print_residual_table("baseline_residual_table", records_first)
    _diag_print_residual_table("repeated_same_params_residual_table", records_second)
    _diag_records_close(records_first, records_second)


def test_minus_1_0_10_failed_fit_does_not_report_fake_after_state(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(context, dataset)
    result = fit_run["result"]
    nfev = int(getattr(result, "nfev", 0) or 0)
    success = bool(getattr(result, "success", False))
    params_changed = bool(fit_run["params_changed"])
    if nfev == 0 and not success and not params_changed:
        print("fit_step_executed=no")
        print("post_fit_residual=<unavailable reason=no accepted optimizer step>")
        repeated_rows = _diag_recompute_fit_audit_rows(context, dataset)
        repeated_records = _diag_records_from_audit_rows(repeated_rows)
        _diag_print_residual_table("baseline_residual_table", baseline_records)
        _diag_print_residual_table("repeated_same_params_residual_table", repeated_records)
        _diag_records_close(baseline_records, repeated_records)
        result_predictions = _diag_result_target_predictions(result)
        for branch, prediction in result_predictions.items():
            if branch not in baseline_records:
                continue
            assert not np.allclose(
                prediction["predicted_caked"],
                baseline_records[branch]["predicted_caked"],
                atol=1.0e-12,
                rtol=0.0,
            ) or str(getattr(result, "message", "") or "")
        return
    print("fit_step_executed=yes")


def test_minus_1_0_10_accepted_fit_step_recomputes_from_accepted_params(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
    )
    result = fit_run["result"]
    if not fit_run["step_executed"]:
        reason = str(getattr(result, "message", "") or "no_compatible_fit_mode")
        print(f"fit_not_testable_reason={reason}")
        assert reason == "seed_multistart_incompatible_with_fixed_manual_pairs"
        return

    print("fit_step_executed=yes")
    print(f"nfev={int(getattr(result, 'nfev', 0) or 0)}")
    print(f"accepted_params={fit_run['after_params']}")
    try:
        after_records = _diag_after_records_from_fit_result(baseline_records, result)
    except AssertionError:
        after_records = None
    if after_records is None:
        print(
            "fit_not_testable_reason="
            "prediction_branch_source_switched"
        )
    else:
        _diag_print_fit_step_table(baseline_records, after_records)
    assert int(getattr(result, "nfev", 0) or 0) > 0


def test_minus_1_0_10_fixed_manual_pairs_fit_step_runs_without_seed_multistart(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
    )
    request = fit_run["request"]
    result = fit_run["result"]
    trace = getattr(result, "seed_multistart_trace", {}) or {}
    summary = getattr(result, "point_match_summary", {}) or {}
    q_residual_count = int(summary.get("manual_caked_residual_row_count", 0))
    nfev = int(getattr(result, "nfev", 0) or 0)
    try:
        after_records = _diag_after_records_from_fit_result(baseline_records, result)
    except AssertionError:
        after_records = None
    prediction_changed = (
        _diag_predictions_changed(baseline_records, after_records)
        if after_records is not None
        else False
    )

    print("[ra-sim] (-1,0,10) fixed manual-pair one-step trace")
    print(f"seed_multistart_enabled={bool(trace.get('enabled', True))}")
    print(
        "fixed_manual_pairs_enabled="
        f"{bool(trace.get('fixed_manual_pair_integrity_enabled', False))}"
    )
    print(f"active_fit_mode={trace.get('active_fit_mode')}")
    print(f"manual_pair_count={int(trace.get('manual_pair_count', 0) or 0)}")
    print(f"qr_residual_count={q_residual_count}")
    print(f"optimizer_method={trace.get('optimizer_method')}")
    print(f"parameter_list={trace.get('parameter_list')}")
    print(f"guard_rejection_reason={trace.get('guard_rejection_reason')}")
    print(f"bypassed_guard={trace.get('bypassed_guard')}")
    print(f"fit_step_executed={'yes' if nfev > 0 else 'no'}")
    print(f"nfev={nfev}")
    print(f"success={bool(getattr(result, 'success', False))}")
    print(f"message={str(getattr(result, 'message', '') or '')}")
    print(f"parameter_values_before={fit_run['before_params']}")
    print(f"parameter_values_after={fit_run['after_params']}")
    _diag_print_residual_table("baseline_residual_table", baseline_records)
    if after_records is not None:
        _diag_print_fit_step_table(baseline_records, after_records)
    else:
        print("post_fit_residual=<unavailable reason=no accepted optimizer step>")

    optimizer_cfg = request.refinement_config.get("optimizer", {})
    seed_search_cfg = request.refinement_config.get("seed_search", {})
    assert bool(optimizer_cfg.get("seed_multistart_enabled")) is False
    assert bool(seed_search_cfg.get("enabled")) is False
    assert bool(trace.get("enabled", True)) is False
    assert trace.get("disabled_reason") == "disabled_by_config"
    assert trace.get("active_fit_mode") == "fixed_manual_pair_direct_least_squares"
    assert bool(trace.get("fixed_manual_pair_integrity_enabled", False))
    assert q_residual_count >= 2
    assert nfev > 0
    if after_records is not None:
        assert fit_run["params_accepted"]
    if fit_run["params_changed"] and after_records is not None:
        assert prediction_changed, "parameterization_cannot_move_qr_prediction"


def test_minus_1_0_10_fit_step_reduces_or_reports_qr_residual(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
    )
    result = fit_run["result"]
    summary = getattr(result, "point_match_summary", {}) or {}
    q_residual_count = int(summary.get("manual_caked_residual_row_count", 0))
    before_norm = _diag_total_residual_norm(baseline_records)
    nfev = int(getattr(result, "nfev", 0) or 0)
    valid_evaluation = bool(fit_run["valid_evaluation"])

    print("[ra-sim] (-1,0,10) Qr residual reduce-or-report")
    _diag_print_residual_table("baseline_residual_table", baseline_records)
    print(f"baseline_residual_norm={before_norm:.9f}")
    print(f"fit_step_executed={'yes' if valid_evaluation else 'no'}")
    print(f"nfev={nfev}")

    after_records = None
    after_norm = float("nan")
    prediction_changed = False
    if valid_evaluation and fit_run["params_accepted"]:
        try:
            after_records = _diag_after_records_from_fit_result(baseline_records, result)
        except AssertionError:
            after_records = None
        if after_records is not None:
            after_norm = _diag_total_residual_norm(after_records)
            prediction_changed = _diag_predictions_changed(baseline_records, after_records)
            _diag_print_fit_step_table(baseline_records, after_records)
            print(f"trial_or_after_residual_norm={after_norm:.9f}")
        else:
            print("post_fit_residual=<unavailable reason=no accepted optimizer step>")
    else:
        print("post_fit_residual=<unavailable reason=no accepted optimizer step>")

    reduced = bool(np.isfinite(after_norm) and after_norm < before_norm - 1.0e-9)
    print(f"qr_residual_reduced={'yes' if reduced else 'no'}")
    if reduced:
        return

    reason = _diag_fixed_manual_pair_no_decrease_reason(
        q_residual_count=q_residual_count,
        valid_evaluation=valid_evaluation,
        params_changed=bool(fit_run["params_changed"]),
        prediction_changed=bool(prediction_changed),
        result=result,
    )
    print(f"first_failing_reason={reason}")
    assert reason in {
        "optimizer_step_rejected",
        "bounds_block_update",
        "parameterization_cannot_move_qr_prediction",
        "objective_excludes_qr_residual",
        "prediction_branch_source_switched",
        "total_objective_improved_but_qr_worsened",
        "optimizer_residual_vector_mismatch",
    }


def test_minus_1_0_10_optimizer_residual_vector_matches_audit(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    baseline_record = trace[0]
    _diag_print_residual_table("baseline_audit_residual_table", baseline_records)
    _diag_print_optimizer_residual_vector_table(baseline_record)

    components = _diag_target_objective_components(baseline_record)
    if len(components) != 4:
        target_rows = _diag_target_objective_rows(baseline_record)
        print("optimizer_residual_vector_mismatch=prediction_branch_source_switched")
        for row in target_rows:
            print(
                "target_rejection | "
                f"branch={row.get('source_branch_index')} | "
                f"prediction_source={row.get('predicted_source')} | "
                f"resolution_reason={row.get('resolution_reason')} | "
                f"resolution_subreason={row.get('resolution_subreason')} | "
                f"source_row_rejection_reason={row.get('source_row_rejection_reason')}"
            )
        assert target_rows, "qr_residual_absent_from_objective"
        assert all(
            row.get("predicted_source") == "rejected:prediction_branch_source_switched"
            for row in target_rows
        ), "optimizer_residual_vector_mismatch"
        assert all(
            row.get("resolution_reason") == "prediction_branch_source_switched"
            for row in target_rows
        ), "optimizer_residual_vector_mismatch"
        return
    assert len(components) == 4, "qr_residual_absent_from_objective"
    by_branch_component = {
        (int(component["source_branch_index"]), str(component["component"])): component
        for component in components
    }
    for branch, audit_record in baseline_records.items():
        theta_component = by_branch_component[(branch, "delta_two_theta_deg")]
        phi_component = by_branch_component[(branch, "wrapped_delta_phi_deg")]
        expected = audit_record["residual_caked"]
        assert theta_component["predicted_source"] == (
            "dynamic_current_simulation:q_group_hkl_source_row_provenance"
        )
        assert phi_component["predicted_source"] == (
            "dynamic_current_simulation:q_group_hkl_source_row_provenance"
        )
        assert theta_component["coordinate_space"] == "caked_deg"
        assert phi_component["coordinate_space"] == "caked_deg"
        assert theta_component["units"] == "deg"
        assert phi_component["units"] == "deg"
        assert np.isclose(
            float(theta_component["unweighted_value"]),
            float(expected[0]),
            atol=1.0e-9,
            rtol=0.0,
        )
        assert np.isclose(
            float(phi_component["unweighted_value"]),
            float(expected[1]),
            atol=1.0e-9,
            rtol=0.0,
        )
        assert np.isclose(
            float(theta_component["weighted_value"]),
            float(theta_component["unweighted_value"]) * float(theta_component["weight"]),
            atol=1.0e-9,
            rtol=0.0,
        )
        assert np.isclose(
            float(phi_component["weighted_value"]),
            float(phi_component["unweighted_value"]) * float(phi_component["weight"]),
            atol=1.0e-9,
            rtol=0.0,
        )


def test_minus_1_0_10_qr_only_objective_does_not_accept_worse_solution(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    baseline_norm = _diag_total_residual_norm(baseline_records)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    result = fit_run["result"]
    success = bool(getattr(result, "success", False))
    nfev = int(getattr(result, "nfev", 0) or 0)
    trace = _diag_objective_trace(result)
    rejected_trial_norm = (
        _diag_qr_norm_from_objective_record(trace[-1]) if trace else float("nan")
    )
    after_records = None
    accepted_norm = float("nan")
    accepted_predictions_available = False
    try:
        candidate_after_records = _diag_after_records_from_fit_result(
            baseline_records,
            result,
        )
    except AssertionError:
        candidate_after_records = None
    if candidate_after_records is not None:
        after_records = candidate_after_records
        accepted_norm = _diag_total_residual_norm(after_records)
        accepted_predictions_available = True

    print("[ra-sim] (-1,0,10) Qr-only fit result")
    print(f"baseline_qr_only_norm={baseline_norm:.9f}")
    if accepted_predictions_available:
        print("fit_step_executed=yes")
        print(f"accepted_qr_only_norm={accepted_norm:.9f}")
    else:
        print("fit_step_executed=no")
        print(f"rejected_trial_norm={rejected_trial_norm:.9f}")
        print("post_fit_residual=<unavailable reason=no accepted optimizer step>")
    print(f"success={success}")
    print(f"nfev={nfev}")
    print(f"parameter_values_before={fit_run['before_params']}")
    print(f"parameter_values_after={fit_run['after_params']}")
    if after_records is not None:
        _diag_print_fit_step_table(baseline_records, after_records)

    assert nfev > 0
    assert (not success) or accepted_predictions_available, (
        "optimizer_accepted_without_target_qr_prediction",
        baseline_norm,
        rejected_trial_norm,
    )
    assert (not success) or accepted_norm <= baseline_norm + 1.0e-9, (
        "optimizer_accepted_worse_qr_only_solution",
        baseline_norm,
        accepted_norm,
    )


def _diag_param_delta_text(var_names, delta):
    return {
        str(name): float(delta[idx])
        for idx, name in enumerate(var_names)
        if idx < len(delta)
    }


def _diag_branch_identity_tuple(row):
    return (
        _diag_q_group_key(row),
        _diag_hkl(row),
        row.get("source_table_index"),
        row.get("source_row_index"),
        row.get("source_branch_index"),
        row.get("source_peak_index"),
        row.get("branch_id"),
    )


def _diag_trace_branch_identity_stable(record, expected_by_branch):
    rows = _diag_target_prediction_rows_by_branch(record)
    if set(rows) != set(expected_by_branch):
        return False
    for branch, row in rows.items():
        if _diag_branch_identity_tuple(row) != expected_by_branch[branch]:
            return False
        if row.get("predicted_source") == "rejected:prediction_branch_source_switched":
            return False
    return True


def _diag_trace_pair(record, branch, key):
    row = _diag_target_prediction_rows_by_branch(record).get(int(branch), {})
    return _diag_pair_or_nan(row.get(key))


def _diag_trace_eval_status(record, result, baseline_x):
    x = np.asarray(record.get("x", ()), dtype=float).reshape(-1)
    if x.size == baseline_x.size and np.allclose(x, baseline_x, atol=1.0e-12, rtol=0.0):
        return "baseline"
    final_x = np.asarray(getattr(result, "x", ()), dtype=float).reshape(-1)
    if (
        bool(getattr(result, "success", False))
        and final_x.size == x.size
        and np.allclose(x, final_x, atol=1.0e-12, rtol=0.0)
    ):
        return "accepted"
    return "rejected"


def _diag_first_bad_qr_eval(trace, baseline_norm=None):
    if not trace:
        return None
    reference_norm = (
        float(baseline_norm)
        if baseline_norm is not None and np.isfinite(float(baseline_norm))
        else _diag_qr_norm_from_objective_record(trace[0])
    )
    first_norm = _diag_qr_norm_from_objective_record(trace[0])
    if np.isfinite(reference_norm) and np.isfinite(first_norm):
        if reference_norm < 5.0 and first_norm > max(10.0, 10.0 * reference_norm):
            return trace[0]
    previous_norm = first_norm
    for record in trace[1:]:
        norm = _diag_qr_norm_from_objective_record(record)
        if (
            np.isfinite(previous_norm)
            and np.isfinite(norm)
            and previous_norm < 5.0
            and norm > max(10.0, 10.0 * reference_norm)
        ):
            return record
        previous_norm = norm
    return None


def _diag_caked_delta(after, before):
    return (
        float(after[0]) - float(before[0]),
        float(((float(after[1]) - float(before[1]) + 180.0) % 360.0) - 180.0),
    )


def _diag_parameter_sensitivity_epsilon(name, value):
    if name in {"zb", "zs"}:
        return 1.0e-6
    if name in {"theta_initial", "theta_offset", "psi_z", "chi", "cor_angle", "gamma", "Gamma"}:
        return 1.0e-4
    if name == "corto_detector":
        return 1.0e-6
    if name in {"a", "c"}:
        return 1.0e-5
    if name in {"center_x", "center_y"}:
        return 1.0e-3
    return max(1.0e-8 * max(abs(float(value)), 1.0), 1.0e-8)


def _diag_micro_step_epsilon(name, value):
    return 0.05 * _diag_parameter_sensitivity_epsilon(name, value)


def _diag_prediction_delta_max_norm(baseline_records, trial_records):
    max_norm = 0.0
    for branch in sorted(baseline_records):
        delta = _diag_caked_delta(
            trial_records[branch]["predicted_caked"],
            baseline_records[branch]["predicted_caked"],
        )
        max_norm = max(max_norm, float(np.linalg.norm(np.asarray(delta, dtype=float))))
    return float(max_norm)


def _diag_print_qr_trial_history(trace, result, var_names):
    assert trace
    baseline_x = np.asarray(trace[0].get("x", ()), dtype=float).reshape(-1)
    expected_by_branch = {
        branch: _diag_branch_identity_tuple(row)
        for branch, row in _diag_target_prediction_rows_by_branch(trace[0]).items()
    }
    print("qr_only_trial_history")
    print(
        "eval | status | x | delta_from_baseline | pred_b0 | pred_b1 | "
        "residual_b0 | residual_b1 | total_qr_norm | branch_identity_stable"
    )
    for record in trace:
        x = np.asarray(record.get("x", ()), dtype=float).reshape(-1)
        delta = x - baseline_x if x.size == baseline_x.size else np.asarray([], dtype=float)
        print(
            f"{int(record.get('eval_index', -1))} | "
            f"{_diag_trace_eval_status(record, result, baseline_x)} | "
            f"{_diag_param_delta_text(var_names, x)} | "
            f"{_diag_param_delta_text(var_names, delta)} | "
            f"{_diag_fmt_pair(_diag_trace_pair(record, 0, 'predicted_caked_deg'))} | "
            f"{_diag_fmt_pair(_diag_trace_pair(record, 1, 'predicted_caked_deg'))} | "
            f"{_diag_fmt_pair(_diag_trace_pair(record, 0, 'residual_caked_deg'))} | "
            f"{_diag_fmt_pair(_diag_trace_pair(record, 1, 'residual_caked_deg'))} | "
            f"{_diag_qr_norm_from_objective_record(record):.9f} | "
            f"{'yes' if _diag_trace_branch_identity_stable(record, expected_by_branch) else 'no'}"
        )


def test_minus_1_0_10_qr_only_trial_history(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    baseline_records = _diag_records_from_audit_rows(_diag_fit_audit_rows(dataset))
    baseline_norm = _diag_total_residual_norm(baseline_records)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    result = fit_run["result"]
    trace = _diag_objective_trace(result)
    assert trace
    print(f"handoff_baseline_qr_norm={baseline_norm:.9f}")
    _diag_print_qr_trial_history(trace, result, fit_run["var_names"])
    first_bad = _diag_first_bad_qr_eval(trace, baseline_norm)
    assert first_bad is not None, "first_bad_eval_not_found"
    baseline_x = np.asarray(trace[0].get("x", ()), dtype=float).reshape(-1)
    bad_x = np.asarray(first_bad.get("x", ()), dtype=float).reshape(-1)
    bad_delta = bad_x - baseline_x
    print(f"first_bad_eval_index={int(first_bad.get('eval_index', -1))}")
    print(f"first_bad_parameter_delta={_diag_param_delta_text(fit_run['var_names'], bad_delta)}")


def test_minus_1_0_10_qr_parameter_sensitivity_scale(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    var_names = _diag_geometry_fit_var_names(context["saved_state"])
    baseline_rows = _diag_recompute_fit_audit_rows(context, dataset, params)
    baseline_records = _diag_records_from_audit_rows(baseline_rows)
    baseline_norm = _diag_total_residual_norm(baseline_records)
    print("qr_parameter_sensitivity_scale")
    print(
        "param | sign | epsilon | pred_delta_b0 | pred_delta_b1 | "
        "residual_norm_delta | derivative_estimate"
    )
    suspects = []
    for name in var_names:
        if name not in params:
            continue
        value = float(params[name])
        epsilon = _diag_parameter_sensitivity_epsilon(name, value)
        records_by_sign = {}
        norms_by_sign = {}
        for sign in (1.0, -1.0):
            trial_params = dict(params)
            trial_params[name] = value + sign * epsilon
            trial_rows = _diag_recompute_fit_audit_rows(context, dataset, trial_params)
            trial_records = _diag_records_from_audit_rows(trial_rows)
            records_by_sign[sign] = trial_records
            norms_by_sign[sign] = _diag_total_residual_norm(trial_records)
        derivative = (norms_by_sign[1.0] - norms_by_sign[-1.0]) / (2.0 * epsilon)
        for sign in (1.0, -1.0):
            trial_records = records_by_sign[sign]
            pred_delta_b0 = _diag_caked_delta(
                trial_records[0]["predicted_caked"],
                baseline_records[0]["predicted_caked"],
            )
            pred_delta_b1 = _diag_caked_delta(
                trial_records[1]["predicted_caked"],
                baseline_records[1]["predicted_caked"],
            )
            max_move = max(
                float(np.linalg.norm(np.asarray(pred_delta_b0, dtype=float))),
                float(np.linalg.norm(np.asarray(pred_delta_b1, dtype=float))),
            )
            if max_move > 10.0:
                suspects.append(str(name))
            print(
                f"{name} | {'+' if sign > 0 else '-'} | {epsilon:.9g} | "
                f"{_diag_fmt_pair(pred_delta_b0)} | {_diag_fmt_pair(pred_delta_b1)} | "
                f"{(norms_by_sign[sign] - baseline_norm):.9f} | {derivative:.9f}"
            )
    suspects = sorted(set(suspects))
    print(f"extreme_sensitivity_parameters={suspects}")
    print(f"parameter_scaling_suspect={'yes' if suspects else 'no'}")


def test_minus_1_0_10_qr_only_bounded_micro_step(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    var_names = _diag_geometry_fit_var_names(context["saved_state"])
    bounds = {
        str(name): {
            "mode": "relative",
            "min": -_diag_micro_step_epsilon(str(name), params.get(str(name), 0.0)),
            "max": _diag_micro_step_epsilon(str(name), params.get(str(name), 0.0)),
        }
        for name in var_names
    }
    baseline_records = _diag_records_from_audit_rows(_diag_fit_audit_rows(dataset))
    baseline_norm = _diag_total_residual_norm(baseline_records)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
        bounds_overrides=bounds,
        optimizer_overrides={"max_nfev": 20},
    )
    result = fit_run["result"]
    nfev = int(getattr(result, "nfev", 0) or 0)
    trace = _diag_objective_trace(result)
    after_records = None
    after_unavailable_reason = ""
    try:
        after_records = _diag_after_records_from_fit_result(baseline_records, result)
        after_norm = _diag_total_residual_norm(after_records)
    except AssertionError:
        after_unavailable_reason = "nonfinite_or_missing_target_prediction"
        after_norm = _diag_qr_norm_from_objective_record(trace[-1]) if trace else float("nan")
    optimizer_before_norm = (
        _diag_qr_norm_from_objective_record(trace[0]) if trace else float("nan")
    )
    comparison_before_norm = (
        optimizer_before_norm if np.isfinite(optimizer_before_norm) else baseline_norm
    )
    prediction_changed = (
        _diag_predictions_changed(baseline_records, after_records)
        if after_records is not None
        else bool(
            len(trace) >= 2
            and not np.allclose(
                np.asarray(trace[-1].get("x", ()), dtype=float),
                np.asarray(trace[0].get("x", ()), dtype=float),
                atol=1.0e-12,
                rtol=0.0,
            )
        )
    )
    print("qr_only_bounded_micro_step")
    print(f"micro_bounds={bounds}")
    print(f"nfev={nfev}")
    print(f"prediction_recomputed={'yes' if trace else 'no'}")
    print(f"prediction_changed={'yes' if prediction_changed else 'no'}")
    print(f"handoff_residual_before={baseline_norm:.9f}")
    print(f"optimizer_residual_before={optimizer_before_norm:.9f}")
    print(f"residual_after={after_norm:.9f}")
    if after_records is not None:
        _diag_print_fit_step_table(baseline_records, after_records)
    else:
        print(f"final_residual_table=<unavailable reason={after_unavailable_reason}>")
    if after_norm <= comparison_before_norm + 1.0e-9:
        print("bounded_micro_step_reduces_qr_residual=yes")
        if after_norm <= baseline_norm + 1.0e-9:
            print("qr_remaining_issue=step_scale_or_trust_region")
        else:
            print("qr_remaining_issue=optimizer_baseline_prediction_mismatch")
    else:
        print("bounded_micro_step_reduces_qr_residual=no")
        print("parameterization_cannot_reduce_qr_residual")
    assert nfev > 0
    assert trace
    assert prediction_changed
    assert after_norm <= comparison_before_norm + 1.0e-9


def test_minus_1_0_10_qr_residual_phi_wrap_continuity(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    var_names = _diag_geometry_fit_var_names(context["saved_state"])
    baseline_rows = _diag_recompute_fit_audit_rows(context, dataset, params)
    baseline_records = _diag_records_from_audit_rows(baseline_rows)
    baseline_identity = {
        branch: _diag_branch_identity_tuple(row) for branch, row in baseline_rows.items()
    }
    print("qr_residual_phi_wrap_continuity")
    print(
        "param | sign | epsilon | residual_phi_delta_b0 | residual_phi_delta_b1 | "
        "branch_identity_stable"
    )
    failures = []
    for name in var_names:
        if name not in params:
            continue
        value = float(params[name])
        epsilon = _diag_micro_step_epsilon(name, value)
        for sign in (1.0, -1.0):
            trial_params = dict(params)
            trial_params[name] = value + sign * epsilon
            trial_rows = _diag_recompute_fit_audit_rows(context, dataset, trial_params)
            trial_records = _diag_records_from_audit_rows(trial_rows)
            identity_stable = set(trial_rows) == set(baseline_identity)
            if identity_stable:
                identity_stable = all(
                    _diag_branch_identity_tuple(trial_rows[branch])
                    == baseline_identity[branch]
                    for branch in baseline_identity
                )
            phi_deltas = {}
            for branch in sorted(baseline_records):
                phi_deltas[branch] = float(
                    (
                        trial_records[branch]["residual_caked"][1]
                        - baseline_records[branch]["residual_caked"][1]
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
                if abs(phi_deltas[branch]) > 90.0:
                    failures.append((name, sign, branch, phi_deltas[branch], "phi_jump"))
            if not identity_stable:
                failures.append((name, sign, "all", float("nan"), "branch_identity_flip"))
            print(
                f"{name} | {'+' if sign > 0 else '-'} | {epsilon:.9g} | "
                f"{phi_deltas.get(0, np.nan):.9f} | {phi_deltas.get(1, np.nan):.9f} | "
                f"{'yes' if identity_stable else 'no'}"
            )
    print(f"phi_wrap_discontinuity_failures={failures}")
    assert not failures, "qr_residual_phi_wrap_discontinuity"


def test_minus_1_0_10_qr_only_solver_inputs_are_scaled(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    baseline_records = _diag_records_from_audit_rows(_diag_fit_audit_rows(dataset))
    baseline_norm = _diag_total_residual_norm(baseline_records)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    request = fit_run["request"]
    result = fit_run["result"]
    trace = _diag_objective_trace(result)
    assert trace
    debug = getattr(result, "geometry_fit_debug_summary", {}) or {}
    parameter_entries = {
        str(entry.get("name")): dict(entry)
        for entry in debug.get("parameter_entries", []) or []
        if isinstance(entry, Mapping)
    }
    var_names = [str(name) for name in fit_run["var_names"]]
    x0 = [float(fit_run["before_params"][name]) for name in var_names]
    x_scale = [float(parameter_entries.get(name, {}).get("scale", np.nan)) for name in var_names]
    lower_bounds = [
        float(parameter_entries.get(name, {}).get("lower_bound", np.nan))
        for name in var_names
    ]
    upper_bounds = [
        float(parameter_entries.get(name, {}).get("upper_bound", np.nan))
        for name in var_names
    ]
    jacobian = getattr(result, "jac", None)
    first_bad = _diag_first_bad_qr_eval(trace, baseline_norm)
    scaling_suspect = False
    baseline_prediction_mismatch = False
    if first_bad is not None:
        baseline_x = np.asarray(trace[0].get("x", ()), dtype=float).reshape(-1)
        bad_x = np.asarray(first_bad.get("x", ()), dtype=float).reshape(-1)
        scale = np.asarray(x_scale, dtype=float)
        if bad_x.size == baseline_x.size and scale.size == bad_x.size:
            delta = bad_x - baseline_x
            finite_scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
            scaled_delta_norm = float(np.linalg.norm(delta / finite_scale))
            raw_delta_norm = float(np.linalg.norm(delta))
            baseline_prediction_mismatch = bool(raw_delta_norm <= 1.0e-15)
            scaling_suspect = bool(raw_delta_norm > 1.0e-15 and scaled_delta_norm < 1.0e-3)
            print(f"first_bad_scaled_delta_norm={scaled_delta_norm:.12f}")
    unbounded = any(
        not (np.isfinite(lo) and np.isfinite(hi))
        for lo, hi in zip(lower_bounds, upper_bounds)
    )
    scaling_suspect = bool(scaling_suspect or unbounded)
    optimizer_cfg = request.refinement_config.get("optimizer", {})
    print("qr_only_solver_inputs")
    print(f"x0={dict(zip(var_names, x0))}")
    print(f"x_scale={dict(zip(var_names, x_scale))}")
    print(f"bounds={dict(zip(var_names, zip(lower_bounds, upper_bounds)))}")
    print(f"residual_vector={trace[0].get('residual_vector')}")
    if jacobian is None:
        print("jacobian=<unavailable>")
    else:
        print(f"jacobian={np.array2string(np.asarray(jacobian, dtype=float), precision=9)}")
    print(f"method={optimizer_cfg.get('method', 'trf')}")
    print(f"loss={optimizer_cfg.get('loss', 'linear')}")
    print("ftol=default(1e-8)")
    print("xtol=default(1e-8)")
    print("gtol=default(1e-8)")
    print(f"max_nfev={max(20, int(optimizer_cfg.get('max_nfev', 120)))}")
    print(
        "optimizer_baseline_prediction_mismatch="
        f"{'yes' if baseline_prediction_mismatch else 'no'}"
    )
    print(f"optimizer_scaling_suspect={'yes' if scaling_suspect else 'no'}")
    assert x0
    assert len(x_scale) == len(x0)
    assert trace[0].get("residual_vector")


def _diag_source_identity(row):
    return {
        "q_group_key": _diag_q_group_key(row),
        "hkl": _diag_hkl(row),
        "source_table_index": row.get("source_table_index"),
        "source_row_index": row.get("source_row_index"),
        "source_branch_index": row.get("source_branch_index"),
        "source_peak_index": row.get("source_peak_index"),
        "branch_id": row.get("branch_id"),
    }


def _diag_source_identity_key(row):
    identity = _diag_source_identity(row)
    return (
        identity["q_group_key"],
        identity["hkl"],
        identity["source_table_index"],
        identity["source_row_index"],
        identity["source_branch_index"],
        identity["source_peak_index"],
        identity["branch_id"],
    )


def _diag_locked_correspondence_key(row):
    identity = _diag_source_identity(row)
    return (
        identity["q_group_key"],
        identity["hkl"],
        identity["source_table_index"],
        identity["source_row_index"],
        identity["source_branch_index"],
        identity["source_peak_index"],
    )


def _diag_prediction_source_text(row):
    source = str(row.get("fit_prediction_source", row.get("predicted_source", "")) or "")
    reason = str(row.get("resolution_reason", row.get("correspondence_resolution_reason", "")) or "")
    subreason = str(row.get("resolution_subreason", "") or "")
    return source, reason, subreason


def _diag_resolver_function_text(row):
    source, reason, subreason = _diag_prediction_source_text(row)
    if "q_group_hkl_source_row_provenance" in source or reason == "resolved_source_row":
        return "q_group_hkl_source_row_provenance"
    if "branch_representative" in source or reason == "resolved_source_peak":
        return "branch_representative_fallback"
    if reason:
        return reason
    return subreason or "unknown"


def _diag_cache_id_text(row):
    for key in (
        "source_rows_cache_id",
        "source_row_cache_id",
        "simulation_cache_id",
        "source_snapshot_id",
        "source_rows_hash",
    ):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "<unavailable>"


def _diag_handoff_predicted_caked(row):
    return _diag_row_pair(row, "fit_prediction_caked_deg")


def _diag_solver_predicted_caked(row):
    return _diag_row_pair(row, "predicted_caked_deg")


def _diag_handoff_residual_vector(records):
    values = []
    for branch in sorted(records):
        residual = records[branch]["residual_caked"]
        values.extend([float(residual[0]), float(residual[1])])
    return np.asarray(values, dtype=float)


def _diag_trace_residual_vector(record):
    values = []
    rows = _diag_target_prediction_rows_by_branch(record)
    for branch in sorted(rows):
        residual = _diag_pair_or_nan(rows[branch].get("residual_caked_deg"))
        values.extend([float(residual[0]), float(residual[1])])
    return np.asarray(values, dtype=float)


def _diag_source_rows_hash(rows):
    normalized = []
    for row in rows or ():
        if not isinstance(row, Mapping) or not _diag_entry_is_minus_1_0_10_branch(row):
            continue
        normalized.append(
            {
                "q_group_key": repr(_diag_q_group_key(row)),
                "hkl": repr(_diag_hkl(row)),
                "source_table_index": row.get("source_table_index"),
                "source_row_index": row.get("source_row_index"),
                "source_branch_index": row.get("source_branch_index"),
                "source_peak_index": row.get("source_peak_index"),
                "predicted_caked": _diag_pair_or_nan(
                    row.get(
                        "fit_prediction_caked_deg",
                        row.get(
                            "predicted_caked_deg",
                            (
                                row.get("simulated_two_theta_deg"),
                                row.get("simulated_phi_deg"),
                            ),
                        ),
                    )
                ),
            }
        )
    text = repr(sorted(normalized, key=repr))
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest(), normalized


def _diag_print_x0_prediction_comparison(handoff_rows, dry_rows, solver_rows):
    print("handoff_dry_run_solver_x0_prediction_table")
    print(
        "branch | q_group_key | hkl | source_table_index | source_row_index | "
        "source_branch_index | source_peak_index | branch_id | handoff_predicted_caked | "
        "dry_run_predicted_caked | solver_x0_predicted_caked | "
        "handoff_minus_dry_run_delta | handoff_minus_solver_x0_delta | "
        "resolver_function | prediction_source | source_match_reason | cache_id"
    )
    for branch in (0, 1):
        handoff = handoff_rows.get(branch, {})
        dry = dry_rows.get(branch, {})
        solver = solver_rows.get(branch, {})
        handoff_pred = _diag_handoff_predicted_caked(handoff)
        dry_pred = _diag_solver_predicted_caked(dry)
        solver_pred = _diag_solver_predicted_caked(solver)
        handoff_minus_dry = _diag_caked_delta(handoff_pred, dry_pred)
        handoff_minus_solver = _diag_caked_delta(handoff_pred, solver_pred)
        source, reason, subreason = _diag_prediction_source_text(solver)
        print(
            f"{branch} | {handoff.get('q_group_key')} | {handoff.get('hkl')} | "
            f"{handoff.get('source_table_index')} | {handoff.get('source_row_index')} | "
            f"{handoff.get('source_branch_index')} | {handoff.get('source_peak_index')} | "
            f"{handoff.get('branch_id')} | {_diag_fmt_pair(handoff_pred)} | "
            f"{_diag_fmt_pair(dry_pred)} | {_diag_fmt_pair(solver_pred)} | "
            f"{_diag_fmt_pair(handoff_minus_dry)} | {_diag_fmt_pair(handoff_minus_solver)} | "
            f"{_diag_resolver_function_text(solver)} | {source} | "
            f"{reason or subreason} | {_diag_cache_id_text(solver)}"
        )


def _diag_solver_x0_bundle(context, dataset):
    handoff_rows = _diag_fit_audit_rows(dataset)
    handoff_records = _diag_records_from_audit_rows(handoff_rows)
    dry_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
        objective_dry_run_only=True,
    )
    solver_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    dry_trace = _diag_objective_trace(dry_run["result"])
    solver_trace = _diag_objective_trace(solver_run["result"])
    assert dry_trace
    assert solver_trace
    return {
        "handoff_rows": handoff_rows,
        "handoff_records": handoff_records,
        "dry_run": dry_run,
        "solver_run": solver_run,
        "dry_record": dry_trace[0],
        "solver_record": solver_trace[0],
    }


def test_minus_1_0_10_solver_x0_matches_handoff_and_dry_run(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    bundle = _diag_solver_x0_bundle(context, dataset)
    handoff_rows = bundle["handoff_rows"]
    dry_rows = _diag_target_prediction_rows_by_branch(bundle["dry_record"])
    solver_rows = _diag_target_prediction_rows_by_branch(bundle["solver_record"])
    _diag_print_x0_prediction_comparison(handoff_rows, dry_rows, solver_rows)

    handoff_vec = _diag_handoff_residual_vector(bundle["handoff_records"])
    dry_vec = _diag_trace_residual_vector(bundle["dry_record"])
    solver_vec = _diag_trace_residual_vector(bundle["solver_record"])
    print(f"handoff_residual_vector={handoff_vec.tolist()}")
    print(f"dry_run_residual_vector={dry_vec.tolist()}")
    print(f"solver_x0_residual_vector={solver_vec.tolist()}")

    first_bad = None
    for branch in (0, 1):
        handoff_pred = _diag_handoff_predicted_caked(handoff_rows.get(branch, {}))
        dry_pred = _diag_solver_predicted_caked(dry_rows.get(branch, {}))
        solver_pred = _diag_solver_predicted_caked(solver_rows.get(branch, {}))
        if not (
            np.allclose(handoff_pred, dry_pred, atol=1.0e-9, rtol=0.0)
            and np.allclose(handoff_pred, solver_pred, atol=1.0e-9, rtol=0.0)
        ):
            first_bad = branch
            print("first_failure=solver_callback_x0_prediction_mismatch")
            print(f"first_bad_branch={branch}")
            print(
                "expected_source="
                f"{_diag_prediction_source_text(handoff_rows.get(branch, {}))[0]}"
            )
            print(
                "actual_source="
                f"{_diag_prediction_source_text(solver_rows.get(branch, {}))[0]}"
            )
            break
    if first_bad is None and not (
        np.allclose(handoff_vec, dry_vec, atol=1.0e-9, rtol=0.0)
        and np.allclose(handoff_vec, solver_vec, atol=1.0e-9, rtol=0.0)
    ):
        first_bad = -1
        print("first_failure=solver_callback_x0_residual_vector_mismatch")
        print("first_bad_branch=<residual_vector>")
    assert first_bad is None


def test_minus_1_0_10_solver_small_perturbations_keep_same_prediction_source(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    var_names = _diag_geometry_fit_var_names(context["saved_state"])
    baseline = _diag_solver_x0_bundle(context, dataset)
    baseline_rows = _diag_target_prediction_rows_by_branch(baseline["dry_record"])
    print("solver_small_perturbation_prediction_source_table")
    print(
        "param | epsilon | branch | baseline_predicted_caked | plus_predicted_caked | "
        "minus_predicted_caked | baseline_source_identity | plus_source_identity | "
        "minus_source_identity | delta_plus | delta_minus | derivative_estimate | "
        "source_changed | discontinuity"
    )
    failures = []
    for name in var_names:
        if name not in params:
            continue
        epsilon = _diag_micro_step_epsilon(str(name), params[name])
        records_by_sign = {}
        for sign in (1.0, -1.0):
            trial_params = dict(params)
            trial_params[str(name)] = float(params[str(name)]) + sign * epsilon
            fit_run = _diag_run_controlled_minus_1_0_10_fit(
                context,
                dataset,
                seed_multistart_enabled=False,
                objective_trace_enabled=True,
                qr_only_objective=True,
                objective_dry_run_only=True,
                params_overrides=trial_params,
            )
            trace = _diag_objective_trace(fit_run["result"])
            assert trace
            records_by_sign[sign] = _diag_target_prediction_rows_by_branch(trace[0])
        for branch in (0, 1):
            baseline_row = baseline_rows.get(branch, {})
            plus_row = records_by_sign[1.0].get(branch, {})
            minus_row = records_by_sign[-1.0].get(branch, {})
            baseline_pred = _diag_solver_predicted_caked(baseline_row)
            plus_pred = _diag_solver_predicted_caked(plus_row)
            minus_pred = _diag_solver_predicted_caked(minus_row)
            delta_plus = _diag_caked_delta(plus_pred, baseline_pred)
            delta_minus = _diag_caked_delta(minus_pred, baseline_pred)
            derivative = (
                (np.asarray(plus_pred, dtype=float) - np.asarray(minus_pred, dtype=float))
                / (2.0 * epsilon)
            )
            baseline_id = _diag_source_identity_key(baseline_row)
            plus_id = _diag_source_identity_key(plus_row)
            minus_id = _diag_source_identity_key(minus_row)
            source_changed = bool(plus_id != baseline_id or minus_id != baseline_id)
            discontinuity = bool(
                max(
                    float(np.linalg.norm(np.asarray(delta_plus, dtype=float))),
                    float(np.linalg.norm(np.asarray(delta_minus, dtype=float))),
                )
                > 10.0
            )
            if source_changed or discontinuity:
                failures.append((str(name), branch, source_changed, discontinuity))
            print(
                f"{name} | {epsilon:.9g} | {branch} | {_diag_fmt_pair(baseline_pred)} | "
                f"{_diag_fmt_pair(plus_pred)} | {_diag_fmt_pair(minus_pred)} | "
                f"{baseline_id} | {plus_id} | {minus_id} | {_diag_fmt_pair(delta_plus)} | "
                f"{_diag_fmt_pair(delta_minus)} | {_diag_fmt_pair(derivative)} | "
                f"{'yes' if source_changed else 'no'} | "
                f"{'yes' if discontinuity else 'no'}"
            )
    if failures:
        print("first_failure=prediction_source_or_phi_discontinuity")
        print(f"first_bad_perturbation={failures[0]}")
    assert not failures


def test_minus_1_0_10_solver_callback_uses_locked_correspondence(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    bundle = _diag_solver_x0_bundle(context, dataset)
    locked = {
        branch: _diag_locked_correspondence_key(row)
        for branch, row in bundle["handoff_rows"].items()
        if branch in {0, 1}
    }
    solver_rows = _diag_target_prediction_rows_by_branch(bundle["solver_record"])
    callback = {
        branch: _diag_locked_correspondence_key(row)
        for branch, row in solver_rows.items()
        if branch in {0, 1}
    }
    print(f"locked_branch_0_source={locked.get(0)}")
    print(f"locked_branch_1_source={locked.get(1)}")
    print(f"solver_callback_branch_0_source={callback.get(0)}")
    print(f"solver_callback_branch_1_source={callback.get(1)}")
    locked_equals_callback = bool(locked == callback)
    print(f"locked_equals_callback={'yes' if locked_equals_callback else 'no'}")
    if not locked_equals_callback:
        print("first_failure=fixed_correspondence_lost_inside_solver_callback")
    assert locked_equals_callback


def test_minus_1_0_10_x0_params_reconstruct_exact_baseline(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    var_names = [str(name) for name in fit_run["var_names"]]
    baseline_params = {
        name: float(
            dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))[
                name
            ]
        )
        for name in var_names
    }
    x0 = np.asarray(
        [float(fit_run["request"].params[name]) for name in var_names],
        dtype=float,
    ).reshape(-1)
    reconstructed = {name: float(x0[idx]) for idx, name in enumerate(var_names)}
    diffs = {
        name: float(reconstructed[name] - baseline_params[name])
        for name in var_names
    }
    print(f"baseline_params={baseline_params}")
    print(f"x0={x0.tolist()}")
    print(f"reconstructed_params_from_x0={reconstructed}")
    print(f"per_param_diff={diffs}")
    ok = all(abs(value) <= 1.0e-12 for value in diffs.values())
    if not ok:
        print("first_failure=x0_param_reconstruction_not_identity")
    assert ok


def test_minus_1_0_10_solver_source_rows_not_rebuilt_differently_at_x0(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    bundle = _diag_solver_x0_bundle(context, dataset)
    before_hash, before_rows = _diag_source_rows_hash(dataset.get("source_rows_for_trace", ()))
    solver_rows = _diag_target_prediction_rows_by_branch(bundle["solver_record"])
    callback_hash, callback_rows = _diag_source_rows_hash(solver_rows.values())
    print("solver_source_rows_x0")
    print(f"before_solver_source_rows_hash={before_hash}")
    print(f"inside_solver_eval1_rows_hash={callback_hash}")
    print(f"before_solver_rows={before_rows}")
    print(f"inside_solver_eval1_rows={callback_rows}")
    for branch in (0, 1):
        row = solver_rows.get(branch, {})
        print(
            "solver_eval1_row | "
            f"branch={branch} | q_group_key={row.get('q_group_key')} | "
            f"hkl={row.get('hkl')} | source_branch_index={row.get('source_branch_index')} | "
            f"source_table_index={row.get('source_table_index')} | "
            f"source_row_index={row.get('source_row_index')} | "
            f"predicted_caked={_diag_fmt_pair(_diag_solver_predicted_caked(row))}"
        )
    locked = {
        branch: _diag_locked_correspondence_key(row)
        for branch, row in bundle["handoff_rows"].items()
        if branch in {0, 1}
    }
    callback = {
        branch: _diag_locked_correspondence_key(row)
        for branch, row in solver_rows.items()
        if branch in {0, 1}
    }
    source_ok = bool(locked == callback)
    print(f"source_rows_locked_branch_match={'yes' if source_ok else 'no'}")
    if not source_ok:
        print("first_failure=source_rows_changed_between_dry_run_and_solver")
    assert source_ok


def test_minus_1_0_10_fit_prediction_identity_stable_during_step(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    _diag_print_branch_identity_stability_table(trace)

    expected_by_branch = {}
    for row in _diag_target_objective_rows(trace[0]):
        branch = int(row["source_branch_index"])
        expected_by_branch[branch] = (
            _diag_q_group_key(row),
            _diag_hkl(row),
            row.get("source_table_index"),
            row.get("source_row_index"),
            row.get("source_branch_index"),
            row.get("source_peak_index"),
            row.get("branch_id"),
        )
    assert set(expected_by_branch) == {0, 1}
    for record in trace:
        rows = {int(row["source_branch_index"]): row for row in _diag_target_objective_rows(record)}
        assert set(rows) == {0, 1}, "prediction_branch_source_switched"
        for branch, row in rows.items():
            identity = (
                _diag_q_group_key(row),
                _diag_hkl(row),
                row.get("source_table_index"),
                row.get("source_row_index"),
                row.get("source_branch_index"),
                row.get("source_peak_index"),
                row.get("branch_id"),
            )
            assert identity == expected_by_branch[branch], "prediction_branch_source_switched"
            assert row["predicted_source"] in {
                "dynamic_current_simulation:q_group_hkl_source_row_provenance",
                "rejected:prediction_branch_source_switched",
            }
            if row["predicted_source"] == "rejected:prediction_branch_source_switched":
                assert row.get("resolution_reason") == "prediction_branch_source_switched"


def test_minus_1_0_10_total_objective_reports_qr_contribution(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    result = fit_run["result"]
    trace = _diag_objective_trace(result)
    assert len(trace) >= 2
    before = trace[0]
    after = trace[-1]
    qr_before = _diag_qr_norm_from_objective_record(before)
    qr_after = _diag_qr_norm_from_objective_record(after)
    total_before = float(before.get("residual_norm", np.nan))
    total_after = float(after.get("residual_norm", np.nan))
    non_qr_before = _diag_non_qr_point_norm_from_objective_record(before)
    non_qr_after = _diag_non_qr_point_norm_from_objective_record(after)
    line_before = float(before.get("line_residual_norm", np.nan))
    line_after = float(after.get("line_residual_norm", np.nan))
    prior_before = float(before.get("prior_residual_norm", np.nan))
    prior_after = float(after.get("prior_residual_norm", np.nan))
    weights = sorted(
        {
            float(component.get("weight", np.nan))
            for component in _diag_target_objective_components(before)
        }
    )

    print("production_fit_objective_decomposition")
    print(f"total_objective_norm_before={total_before:.9f}")
    print(f"total_objective_norm_after={total_after:.9f}")
    print(f"qr_objective_norm_before={qr_before:.9f}")
    print(f"qr_objective_norm_after={qr_after:.9f}")
    print(f"non_qr_point_objective_norm_before={non_qr_before:.9f}")
    print(f"non_qr_point_objective_norm_after={non_qr_after:.9f}")
    print(f"line_objective_norm_before={line_before:.9f}")
    print(f"line_objective_norm_after={line_after:.9f}")
    print(f"prior_objective_norm_before={prior_before:.9f}")
    print(f"prior_objective_norm_after={prior_after:.9f}")
    print(f"qr_weights={weights}")

    if not np.isfinite(qr_before) or not _diag_target_objective_components(before):
        print("qr_residual_absent_from_objective=yes")
        target_rows = _diag_target_objective_rows(before)
        assert target_rows, "qr_residual_absent_from_objective"
        assert all(
            row.get("resolution_reason") == "prediction_branch_source_switched"
            for row in target_rows
        ), "qr_residual_absent_from_objective"
        print("prediction_branch_source_switched=yes")
        return

    assert np.isfinite(qr_before), "qr_residual_absent_from_objective"
    assert _diag_target_objective_components(before), "qr_residual_absent_from_objective"
    if total_after < total_before - 1.0e-9 and qr_after > qr_before + 1.0e-9:
        print("qr_residual_sacrificed_to_other_terms=yes")
    if weights:
        print(f"qr_weight={weights[0]:.9f}")
        if max(weights) < 1.0:
            print("qr_weight_too_low=yes")


def _diag_target_prediction_rows_by_branch(record):
    rows = {}
    for row in _diag_target_objective_rows(record):
        try:
            branch = int(row.get("source_branch_index"))
        except Exception:
            continue
        rows[branch] = dict(row)
    return rows


def _diag_request_pair_counts(request, result, baseline_record):
    measured = [entry for entry in request.measured_peaks if isinstance(entry, Mapping)]
    handoff_summary = request.refinement_config.get("optimizer_request_handoff_summary", {})
    if not isinstance(handoff_summary, Mapping):
        handoff_summary = {}
    point_summary = getattr(result, "point_match_summary", {}) or {}
    if not isinstance(point_summary, Mapping):
        point_summary = {}
    diagnostics = getattr(result, "point_match_diagnostics", []) or []
    fixed_source_resolution_fallback_count = 0
    branch_mismatch_count = int(point_summary.get("branch_mismatch_count", 0) or 0)
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            continue
        if bool(raw.get("optimizer_request_fallback_row", False)):
            fixed_source_resolution_fallback_count += 1
        if str(raw.get("resolution_kind", "")).strip().lower() not in {
            "",
            "fixed_source",
        }:
            fixed_source_resolution_fallback_count += 1
        try:
            source_branch = int(raw.get("source_branch_index"))
            resolved_peak = int(raw.get("resolved_peak_index"))
        except Exception:
            continue
        if source_branch in {0, 1} and resolved_peak in {0, 1} and source_branch != resolved_peak:
            branch_mismatch_count += 1
    components = _diag_target_objective_components(baseline_record)
    return {
        "provider_pair_count": sum(
            1 for entry in measured if entry.get("optimizer_request_source") == "provider_pair"
        ),
        "dataset_pair_count": len(measured),
        "optimizer_request_pair_count": len(measured),
        "fixed_source_pair_count": int(handoff_summary.get("fixed_source_pair_count", 0) or 0),
        "fallback_row_count": int(handoff_summary.get("fallback_row_count", 0) or 0),
        "missing_fixed_source_count": sum(
            1 for entry in measured if not bool(entry.get("optimizer_request_has_fixed_source"))
        ),
        "fixed_source_resolution_fallback_count": int(fixed_source_resolution_fallback_count),
        "matched_pair_count": int(point_summary.get("matched_pair_count", 0) or 0),
        "missing_pair_count": int(point_summary.get("missing_pair_count", 0) or 0),
        "branch_mismatch_count": int(branch_mismatch_count),
        "qr_residual_block_absent": "no" if components else "yes",
        "qr_weights": sorted({float(component.get("weight", np.nan)) for component in components}),
        "objective_eval_called": bool(getattr(result, "objective_eval_called", False)),
        "objective_dry_run_residual_finite": bool(
            getattr(result, "objective_dry_run_residual_finite", False)
        ),
        "least_squares_called": bool(getattr(result, "least_squares_called", True)),
        "optimizer_solve_called": bool(getattr(result, "optimizer_solve_called", True)),
    }


def test_minus_1_0_10_rung1_objective_dry_run_uses_qr_residuals(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        objective_dry_run_only=True,
    )
    result = fit_run["result"]
    trace = _diag_objective_trace(result)
    assert trace
    baseline_record = trace[0]
    counts = _diag_request_pair_counts(fit_run["request"], result, baseline_record)
    for key in (
        "provider_pair_count",
        "dataset_pair_count",
        "optimizer_request_pair_count",
        "fixed_source_pair_count",
        "fallback_row_count",
        "missing_fixed_source_count",
        "fixed_source_resolution_fallback_count",
        "matched_pair_count",
        "missing_pair_count",
        "branch_mismatch_count",
        "qr_residual_block_absent",
        "qr_weights",
        "objective_eval_called",
        "objective_dry_run_residual_finite",
        "least_squares_called",
        "optimizer_solve_called",
    ):
        print(f"{key}={counts[key]}")
    _diag_print_optimizer_residual_vector_table(baseline_record)

    if counts["qr_residual_block_absent"] == "yes":
        for row in _diag_target_objective_rows(baseline_record):
            print(
                "failing_row | "
                f"pair_id={row.get('manual_pair_id', row.get('row_index'))} | "
                f"branch={row.get('source_branch_index')} | "
                f"reason={row.get('resolution_reason')} | "
                f"subreason={row.get('resolution_subreason')} | "
                f"source_row_reason={row.get('source_row_rejection_reason')}"
            )
    assert counts["least_squares_called"] is False
    assert counts["optimizer_solve_called"] is False
    assert counts["objective_eval_called"] is True
    assert counts["objective_dry_run_residual_finite"] is True
    assert counts["fixed_source_pair_count"] == 7
    assert counts["fallback_row_count"] == 0
    assert counts["missing_fixed_source_count"] == 0
    assert counts["fixed_source_resolution_fallback_count"] == 0
    assert counts["matched_pair_count"] == 7
    assert counts["missing_pair_count"] == 0
    assert counts["branch_mismatch_count"] == 0
    assert counts["qr_residual_block_absent"] == "no"
    assert counts["qr_weights"]


def test_minus_1_0_10_fitter_objective_matches_residual_audit(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    baseline_record = trace[0]
    _diag_print_residual_table("baseline_residual_table", baseline_records)
    _diag_print_optimizer_residual_vector_table(baseline_record)

    components = _diag_target_objective_components(baseline_record)
    target_rows = _diag_target_objective_rows(baseline_record)
    if len(components) != 4:
        print("first_failure=objective_not_using_qr_residual")
        for row in target_rows:
            print(
                "target_rejection | "
                f"branch={row.get('source_branch_index')} | "
                f"q_group_key={row.get('q_group_key')} | "
                f"hkl={row.get('hkl')} | "
                f"predicted_source={row.get('predicted_source')} | "
                f"resolution_reason={row.get('resolution_reason')} | "
                f"resolution_subreason={row.get('resolution_subreason')} | "
                f"source_row_rejection_reason={row.get('source_row_rejection_reason')}"
            )
    assert len(components) == 4, "objective_not_using_qr_residual"

    by_key = {
        (int(component["source_branch_index"]), str(component["component"])): component
        for component in components
    }
    for branch, audit_record in baseline_records.items():
        for component_name, axis_index in (
            ("delta_two_theta_deg", 0),
            ("wrapped_delta_phi_deg", 1),
        ):
            component = by_key[(branch, component_name)]
            assert component["coordinate_space"] == "caked_deg"
            assert component["units"] == "deg"
            assert component["observed_source"] == "background/manual"
            assert str(component["predicted_source"]).startswith("dynamic_current_simulation")
            assert np.isclose(
                float(component["unweighted_value"]),
                float(audit_record["residual_caked"][axis_index]),
                atol=1.0e-9,
                rtol=0.0,
            )
            assert np.isclose(
                float(component["weighted_value"]),
                float(component["unweighted_value"]) * float(component["weight"]),
                atol=1.0e-9,
                rtol=0.0,
            )


def test_minus_1_0_10_optimizer_prediction_matches_fit_handoff_prediction(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    optimizer_rows = _diag_optimizer_rows_by_branch(trace[0])
    _diag_print_handoff_optimizer_prediction_table(rows_by_branch, optimizer_rows)
    _diag_print_saved_sim_native_diagnostics(dataset, rows_by_branch, optimizer_rows)

    failures = []
    for branch in (0, 1):
        handoff = rows_by_branch[branch]
        optimizer = optimizer_rows.get(branch)
        if optimizer is None:
            failures.append((branch, "missing_optimizer_prediction"))
            continue
        handoff_prediction = _diag_row_pair(handoff, "fit_prediction_caked_deg")
        optimizer_prediction = _diag_row_pair(optimizer, "predicted_caked_deg")
        if not np.allclose(
            optimizer_prediction,
            handoff_prediction,
            atol=1.0e-9,
            rtol=0.0,
        ):
            failures.append(
                (
                    branch,
                    "optimizer_prediction_source_mismatch",
                    handoff_prediction,
                    optimizer_prediction,
                )
            )
    if failures:
        print("first_failure=optimizer_prediction_source_mismatch")
        print(f"prediction_failures={failures}")
    assert not failures


def test_minus_1_0_10_optimizer_rejects_noncanonical_saved_sim_native(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    callbacks = context["projection_callbacks"]
    from ra_sim.fitting import optimization as opt

    resolver_rows = {}
    measured_rows = [
        dict(row) for row in dataset.get("measured_for_fit", ()) or [] if isinstance(row, Mapping)
    ]
    initial_rows = [
        dict(row)
        for row in dataset.get("initial_pairs_display", ()) or []
        if isinstance(row, Mapping)
    ]
    for index, measured_row in enumerate(measured_rows):
        if not _diag_entry_is_minus_1_0_10_branch(measured_row):
            continue
        branch = int(measured_row["source_branch_index"])
        initial_row = initial_rows[index] if index < len(initial_rows) else {}
        handoff_row = rows_by_branch[branch]
        entry = dict(measured_row)
        entry.update(
            {
                "fit_source_resolution_kind": "provider_fixed_source_local",
                "optimizer_request_source": "provider_pair",
                "optimizer_request_has_fixed_source": True,
                "optimizer_request_fallback_row": False,
                "provider_local_subset_provenance": True,
                "provider_local_subset_assignment": "provider_local_duplicate_hkl_unproven",
                "resolved_table_index": entry.get("source_table_index"),
                "resolved_peak_index": branch,
            }
        )
        if "sim_display" in initial_row:
            entry["fit_prediction_detector_display_px"] = initial_row["sim_display"]
        if "sim_native" in initial_row:
            entry["fit_prediction_detector_native_px"] = initial_row["sim_native"]
            entry["fit_prediction_detector_native_px_source"] = (
                "display_to_native_sim_coords(sim_display)"
            )
            entry["sim_visual_detector_canonical_native_px"] = initial_row["sim_native"]
            entry["sim_visual_detector_canonical_native_source"] = (
                "display_to_native_sim_coords(sim_display)"
            )
        entry["fit_prediction_detector_display_px"] = handoff_row[
            "fit_prediction_detector_display_px"
        ]
        entry["sim_visual_detector_display_px"] = handoff_row[
            "fit_prediction_detector_display_px"
        ]
        entry["fit_prediction_detector_native_px"] = handoff_row[
            "fit_prediction_detector_native_px"
        ]
        entry["fit_prediction_detector_native_px_source"] = (
            "display_to_native_sim_coords(sim_display)"
        )
        entry["sim_visual_detector_canonical_native_px"] = handoff_row[
            "fit_prediction_detector_native_px"
        ]
        entry["sim_visual_detector_canonical_native_source"] = (
            "display_to_native_sim_coords(sim_display)"
        )
        display_point = entry["sim_visual_detector_display_px"]
        wrong_native = callbacks.background_display_to_native_detector_coords(
            float(display_point[0]),
            float(display_point[1]),
        )
        if wrong_native is not None:
            entry["sim_visual_detector_native_px"] = (
                float(wrong_native[0]),
                float(wrong_native[1]),
            )
        point, payload, reason = opt._provider_local_saved_sim_detector_point(entry)
        assert point is not None, (branch, reason, payload)
        raw_only_entry = dict(entry)
        for key in (
            "sim_visual_detector_canonical_native_px",
            "sim_visual_detector_canonical_native_source",
            "fit_prediction_detector_native_px",
            "fit_prediction_detector_native_px_source",
            "sim_native",
            "sim_native_source",
        ):
            raw_only_entry.pop(key, None)
        raw_only_point, raw_only_payload, raw_only_reason = (
            opt._provider_local_saved_sim_detector_point(raw_only_entry)
        )
        assert raw_only_point is None, (branch, raw_only_reason, raw_only_payload)
        assert raw_only_payload.get("saved_sim_detector_native_rejected_reason") == (
            "display_native_unproven"
        )
        row = dict(payload)
        row["caked_from_saved_native"] = entry.get("sim_visual_caked_deg")
        row["caked_from_display_to_native"] = handoff_row["fit_prediction_caked_deg"]
        row["predicted_caked_deg"] = handoff_row["fit_prediction_caked_deg"]
        resolver_rows[branch] = row
    _diag_print_saved_sim_native_diagnostics(dataset, rows_by_branch, resolver_rows)

    for branch in (0, 1):
        row = resolver_rows[branch]
        saved_native = _diag_row_pair(row, "saved_sim_detector_native_px")
        canonical_native = _diag_row_pair(
            row,
            "display_to_native_saved_sim_detector_display_px",
        )
        handoff_prediction = _diag_row_pair(
            rows_by_branch[branch],
            "fit_prediction_caked_deg",
        )
        optimizer_prediction = _diag_row_pair(row, "predicted_caked_deg")
        assert not np.allclose(saved_native, canonical_native, atol=1.0e-6, rtol=0.0)
        assert not np.allclose(
            _diag_row_pair(row, "caked_from_saved_native"),
            handoff_prediction,
            atol=1.0e-6,
            rtol=0.0,
        )
        assert row.get("saved_sim_detector_native_rejected_reason") == (
            "display_native_mismatch"
        )
        assert np.allclose(
            _diag_row_pair(row, "caked_from_display_to_native"),
            handoff_prediction,
            atol=1.0e-9,
            rtol=0.0,
        )
        assert np.allclose(
            optimizer_prediction,
            handoff_prediction,
            atol=1.0e-9,
            rtol=0.0,
        )


def test_minus_1_0_10_fitter_same_params_reproduce_same_residual(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_first = _diag_recompute_fit_audit_rows(context, dataset)
    rows_second = _diag_recompute_fit_audit_rows(context, dataset)
    records_first = _diag_records_from_audit_rows(rows_first)
    records_second = _diag_records_from_audit_rows(rows_second)
    _diag_print_residual_table("baseline_residual_table", records_first)
    _diag_print_residual_table("same_params_repeated_residual_table", records_second)
    _diag_records_close(records_first, records_second)

    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    first_vec = np.asarray(trace[0].get("residual_vector", ()), dtype=float)
    second_vec = np.asarray(trace[0].get("residual_vector", ()), dtype=float)
    assert np.allclose(first_vec, second_vec, atol=1.0e-12, rtol=0.0)


def test_minus_1_0_10_fitter_trial_param_changes_prediction(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    params = dict(_diag_runtime_value(context["projection_kwargs"]["current_geometry_fit_params"]))
    base_rows = _diag_recompute_fit_audit_rows(context, dataset, params)
    base_records = _diag_records_from_audit_rows(base_rows)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    var_names = list(fit_run["var_names"])
    print("parameter_sensitivity_table")
    print(
        "param | before | after | branch0_prediction_before | branch0_prediction_after | "
        "branch1_prediction_before | branch1_prediction_after | residual_norm_before | "
        "residual_norm_after | sensitivity_norm"
    )
    any_sensitive = False
    before_norm = _diag_total_residual_norm(base_records)
    for name in var_names:
        if name not in params:
            continue
        before_value = float(params[name])
        step = 0.1 if name not in {"zb", "zs", "corto_detector"} else 1.0
        trial_params = dict(params)
        trial_params[name] = before_value + float(step)
        trial_rows = _diag_recompute_fit_audit_rows(context, dataset, trial_params)
        trial_records = _diag_records_from_audit_rows(trial_rows)
        after_norm = _diag_total_residual_norm(trial_records)
        deltas = []
        for branch in (0, 1):
            before_pred = np.asarray(base_records[branch]["predicted_caked"], dtype=float)
            after_pred = np.asarray(trial_records[branch]["predicted_caked"], dtype=float)
            deltas.extend((after_pred - before_pred).tolist())
        sensitivity_norm = float(np.linalg.norm(np.asarray(deltas, dtype=float)))
        any_sensitive = any_sensitive or bool(sensitivity_norm > 1.0e-12)
        print(
            f"{name} | {before_value:.9f} | {trial_params[name]:.9f} | "
            f"{_diag_fmt_pair(base_records[0]['predicted_caked'])} | "
            f"{_diag_fmt_pair(trial_records[0]['predicted_caked'])} | "
            f"{_diag_fmt_pair(base_records[1]['predicted_caked'])} | "
            f"{_diag_fmt_pair(trial_records[1]['predicted_caked'])} | "
            f"{before_norm:.9f} | {after_norm:.9f} | {sensitivity_norm:.9f}"
        )
    if not any_sensitive:
        print("first_failure=fit_not_sensitive_to_qr_prediction")
    assert any_sensitive, "fit_not_sensitive_to_qr_prediction"


def test_minus_1_0_10_qr_only_fit_reduces_residual_after_correspondence_fix(
    tmp_path,
) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    baseline_records = _diag_records_from_audit_rows(rows_by_branch)
    baseline_norm = _diag_total_residual_norm(baseline_records)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
        qr_only_objective=True,
    )
    result = fit_run["result"]
    trace = _diag_objective_trace(result)
    after_records = None
    after_norm = float("nan")
    try:
        after_records = _diag_after_records_from_fit_result(baseline_records, result)
        after_norm = _diag_total_residual_norm(after_records)
    except AssertionError:
        after_records = None

    print("qr_only_fit")
    print(f"initial_params={fit_run['before_params']}")
    print(f"final_params={fit_run['after_params']}")
    print(f"nfev={int(getattr(result, 'nfev', 0) or 0)}")
    print(f"success={bool(getattr(result, 'success', False))}")
    _diag_print_residual_table("baseline_residual_table", baseline_records)
    if after_records is None:
        print("final_residual_table=<unavailable reason=no accepted optimizer step>")
    else:
        _diag_print_fit_step_table(baseline_records, after_records)
    print(f"total_norm_before={baseline_norm:.9f}")
    print(f"total_norm_after={after_norm:.9f}")
    assert int(getattr(result, "nfev", 0) or 0) > 0
    components = _diag_target_objective_components(trace[0]) if trace else []
    if len(components) != 4:
        print("first_failure=objective_not_using_qr_residual")
    assert len(components) == 4, "objective_not_using_qr_residual"
    if not (np.isfinite(after_norm) and after_norm <= baseline_norm + 1.0e-9):
        reason = _diag_fixed_manual_pair_no_decrease_reason(
            q_residual_count=len(components),
            valid_evaluation=bool(np.isfinite(after_norm)),
            params_changed=fit_run["before_params"] != fit_run["after_params"],
            prediction_changed=bool(
                after_records is not None
                and _diag_predictions_changed(baseline_records, after_records)
            ),
            result=result,
        )
        print(f"qr_only_after_norm_not_reduced_reason={reason}")
        assert reason
        accepted_worse_step = bool(
            getattr(result, "success", False)
            and np.isfinite(after_norm)
            and after_norm > baseline_norm + 1.0e-9
        )
        assert not accepted_worse_step, f"accepted_worse_qr_norm:{reason}"
    else:
        assert np.isfinite(after_norm)


def test_minus_1_0_10_full_fit_reports_qr_contribution(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert len(trace) >= 1
    before = trace[0]
    after = trace[-1]
    total_before = float(before.get("residual_norm", np.nan))
    total_after = float(after.get("residual_norm", np.nan))
    qr_before = _diag_qr_norm_from_objective_record(before)
    qr_after = _diag_qr_norm_from_objective_record(after)
    non_qr_before = _diag_non_qr_point_norm_from_objective_record(before)
    non_qr_after = _diag_non_qr_point_norm_from_objective_record(after)
    weights = sorted(
        {
            float(component.get("weight", np.nan))
            for component in _diag_target_objective_components(before)
        }
    )
    print("full_fit_objective_decomposition")
    print(f"total_objective_norm_before={total_before:.9f}")
    print(f"total_objective_norm_after={total_after:.9f}")
    print(f"qr_residual_block_norm_before={qr_before:.9f}")
    print(f"qr_residual_block_norm_after={qr_after:.9f}")
    print(f"non_qr_block_norm_before={non_qr_before:.9f}")
    print(f"non_qr_block_norm_after={non_qr_after:.9f}")
    print(f"line_block_norm_before={float(before.get('line_residual_norm', np.nan)):.9f}")
    print(f"line_block_norm_after={float(after.get('line_residual_norm', np.nan)):.9f}")
    print(f"prior_block_norm_before={float(before.get('prior_residual_norm', np.nan)):.9f}")
    print(f"prior_block_norm_after={float(after.get('prior_residual_norm', np.nan)):.9f}")
    print(f"qr_weights={weights}")
    print(f"accepted_params_before={fit_run['before_params']}")
    print(f"accepted_params_after={fit_run['after_params']}")
    if not _diag_target_objective_components(before):
        print("qr_residual_block_absent=yes")
    assert _diag_target_objective_components(before), "qr_residual_block_absent"
    if total_after < total_before - 1.0e-9 and qr_after > qr_before + 1.0e-9:
        print("qr_residual_sacrificed_to_other_terms=yes")
    if weights and max(weights) < 1.0:
        print("qr_weight_too_low=yes")


def test_minus_1_0_10_prediction_identity_stable_during_fit(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(
        context,
        dataset,
        seed_multistart_enabled=False,
        objective_trace_enabled=True,
    )
    trace = _diag_objective_trace(fit_run["result"])
    assert trace
    _diag_print_branch_identity_stability_table(trace)
    baseline_rows = _diag_target_prediction_rows_by_branch(trace[0])
    assert set(baseline_rows) == {0, 1}
    for record in trace:
        rows = _diag_target_prediction_rows_by_branch(record)
        assert set(rows) == {0, 1}, "branch_identity_switched"
        for branch, row in rows.items():
            expected = baseline_rows[branch]
            identity = (
                row.get("q_group_key"),
                row.get("hkl"),
                row.get("source_branch_index"),
                row.get("source_peak_index"),
                row.get("branch_id"),
            )
            expected_identity = (
                expected.get("q_group_key"),
                expected.get("hkl"),
                expected.get("source_branch_index"),
                expected.get("source_peak_index"),
                expected.get("branch_id"),
            )
            assert identity == expected_identity, "branch_identity_switched"
            assert row.get("predicted_source") != "rejected:prediction_branch_source_switched"


def test_qr_residual_objective_units_are_not_mixed_unweighted(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    fit_run = _diag_run_controlled_minus_1_0_10_fit(context, dataset)
    request = fit_run["request"]
    result = fit_run["result"]
    optimizer_cfg = request.refinement_config.get("optimizer", {})
    assert bool(optimizer_cfg.get("dynamic_point_geometry_fit", False))
    for row in rows_by_branch.values():
        assert row["objective_space"] in {"detector_native_px", "caked_deg"}
        assert row["objective_mixes_detector_px_and_caked_deg"] == "no"
        if row["objective_space"] == "caked_deg":
            assert row["objective_residual_units"] == "deg"
        else:
            assert row["objective_residual_units"] == "px"
    summary = getattr(result, "point_match_summary", {}) or {}
    assert int(summary.get("manual_caked_residual_row_count", 0)) >= 2
    assert summary.get("metric_unit") == "deg"
    assert summary.get("weighted_metric_unit") == "weighted_deg"
    for diag in getattr(result, "point_match_diagnostics", []) or []:
        if not isinstance(diag, Mapping) or not diag.get("valid"):
            continue
        vector = np.asarray(diag.get("solver_residual_vector", ()), dtype=float)
        weighted = np.asarray(
            [
                float(diag.get("weighted_delta_two_theta_deg", np.nan)),
                float(diag.get("weighted_delta_phi_deg", np.nan)),
            ],
            dtype=float,
        )
        assert vector.shape == weighted.shape
        assert np.allclose(vector, weighted, atol=1.0e-9, rtol=0.0)
        offset_source = str(
            diag.get("provider_local_saved_sim_fit_space_offset_source", "") or ""
        )
        if offset_source:
            assert offset_source != "first_eval_unprimed"
            if diag.get("provider_local_saved_sim_fit_space_reference_aligned"):
                assert (
                    diag.get("provider_local_saved_sim_fit_space_offset_baseline_primed")
                    is True
                )
            else:
                assert (
                    diag.get(
                        "provider_local_saved_sim_fit_space_offset_unavailable_reason"
                    )
                    == "baseline_offset_not_primed"
                )


def test_minus_1_0_10_fit_observed_matches_manual_refined_visual(tmp_path) -> None:
    _context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    assert set(rows_by_branch) == {0, 1}
    for row in rows_by_branch.values():
        assert row["observed_visual_to_fit_observed_match"] == "yes"
        assert float(row["fit_observed_minus_observed_refined_detector_delta_px"]) <= 1.0
        _diag_delta_pair_close(row["fit_observed_minus_observed_refined_caked_delta_deg"])


def test_minus_1_0_10_sim_refined_caked_uses_real_projection(tmp_path) -> None:
    context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    callbacks = context["projection_callbacks"]
    assert set(rows_by_branch) == {0, 1}
    for row in rows_by_branch.values():
        caked = row.get("sim_refined_caked_deg")
        native = row.get("sim_refined_detector_native_px")
        assert caked is not None, (
            row.get("source_branch_index"),
            row.get("sim_refined_caked_deg_unavailable_reason"),
        )
        assert native is not None, row.get("source_branch_index")
        assert row["caked_values_from_real_projection_callback"] == "yes"
        assert row["sim_refined_caked_projection_status"] == "real_projection_callback"
        assert not np.allclose(
            caked,
            (float(native[0]) / 100.0, float(native[1]) / 100.0),
            atol=1.0e-3,
            rtol=0.0,
        )
        display_back = callbacks.caked_angles_to_background_display_coords(
            float(caked[0]),
            float(caked[1]),
        )
        assert display_back is not None
        native_back = callbacks.background_display_to_native_detector_coords(
            float(display_back[0]),
            float(display_back[1]),
        )
        assert native_back is not None
        assert np.hypot(
            float(native_back[0]) - float(native[0]),
            float(native_back[1]) - float(native[1]),
        ) <= 2.0


def test_minus_1_0_10_fit_prediction_source_is_explicit(tmp_path) -> None:
    _context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    rows_by_branch = _diag_fit_audit_rows(dataset)
    allowed = {
        "dynamic_current_simulation",
        "saved_visual_sim_refined",
    }
    for row in rows_by_branch.values():
        source = str(row.get("fit_prediction_source"))
        assert source in allowed or source.startswith("unavailable_reason:")
        assert "cache_fresh_row" not in source
        if source == "dynamic_current_simulation":
            assert row["fit_prediction_is_dynamic"] == "yes"
            assert row["sim_visual_to_fit_prediction_match"] == "not-applicable"


def test_minus_1_0_10_no_caked_values_printed_as_detector_px(tmp_path) -> None:
    _context, dataset, _events = _diag_fit_handoff_dataset(tmp_path)
    audit_lines = [str(line) for line in dataset.get("fit_handoff_audit_lines", [])]
    suspicious = ("(39.827, 35.250)", "(10.880, 19.210)", "(10.810, 11.680)")
    for line in audit_lines:
        if "_detector" not in line or "_px=" not in line:
            continue
        assert not any(token in line for token in suspicious), line


def test_minus_1_0_10_fit_handoff_audit_uses_bound_native_to_display_callback() -> None:
    dataset = {
        "native_background": np.zeros((20, 20), dtype=float),
        "display_rotate_k": 0,
        "native_detector_coords_to_detector_display_coords": lambda col, row: (
            float(col) + 1000.0,
            float(row) + 2000.0,
        ),
    }
    assert gf._geometry_fit_audit_native_to_display((3.0, 4.0), dataset) == (
        1003.0,
        2004.0,
    )


def test_minus_1_0_10_fit_handoff_prefers_background_bound_native_to_display_callback() -> None:
    bindings = SimpleNamespace(
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 1000.0,
            float(row) + 2000.0,
        ),
        native_detector_coords_to_detector_display_coords_for_background=lambda idx: (
            lambda col, row: (
                float(col) + 10.0 * int(idx),
                float(row) + 100.0 * int(idx),
            )
        ),
    )

    callback, reason, source = gf._geometry_fit_dataset_native_to_display_callback(bindings, 2)

    assert callable(callback)
    assert reason is None
    assert source == "background_bound_callback"
    assert callback(3.0, 4.0) == (23.0, 204.0)


def test_minus_1_0_10_fit_handoff_bound_transform_failure_no_rotate_fallback() -> None:
    bindings = SimpleNamespace(
        native_detector_coords_to_detector_display_coords=lambda col, row: (
            float(col) + 1000.0,
            float(row) + 2000.0,
        ),
        native_detector_coords_to_detector_display_coords_for_background=lambda _idx: None,
    )
    callback, reason, source = gf._geometry_fit_dataset_native_to_display_callback(bindings, 0)
    dataset = {
        "native_background": np.zeros((20, 20), dtype=float),
        "display_rotate_k": 0,
        "native_detector_coords_to_detector_display_coords": callback,
        "native_detector_coords_to_detector_display_coords_unavailable_reason": reason,
    }

    point, unavailable = gf._geometry_fit_audit_native_to_display_result((3.0, 4.0), dataset)

    assert callback is None
    assert source == "background_bound_callback"
    assert point is None
    assert unavailable == "background-bound native->display callback unavailable"


def test_minus_1_0_10_fit_handoff_audit_native_to_display_callback_failure_no_fallback() -> None:
    dataset = {
        "native_background": np.zeros((20, 20), dtype=float),
        "display_rotate_k": 0,
        "native_detector_coords_to_detector_display_coords": lambda _col, _row: None,
    }
    assert gf._geometry_fit_audit_native_to_display((3.0, 4.0), dataset) is None
    lines = gf.build_geometry_fit_qr_handoff_audit_lines(
        [
            {
                "source_branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
                "hkl": (-1, 0, 10),
                "source_table_index": 160,
                "source_row_index": 24,
                "source_peak_index": 0,
                "branch_id": "branch-0",
                "fit_prediction_detector_display_px": None,
                "fit_prediction_detector_display_px_unavailable_reason": (
                    "live native->display callback returned unavailable"
                ),
            }
        ]
    )
    assert (
        "    fit_prediction_detector_display_px="
        "<unavailable reason=live native->display callback returned unavailable>"
    ) in lines


def test_qr_fit_audit_caked_axes_do_not_claim_real_projection_callback() -> None:
    caked, meta = gf._geometry_fit_audit_sim_refined_caked(
        [
            {
                "sim_refined_caked_deg": (7.0, 4.0),
                "sim_refined_caked_projection_status": "caked_simulation_image_axes",
                "sim_refined_caked_projection_real_callback": False,
            }
        ],
        (107.0, 204.0),
        fit_space_projector=None,
        base_fit_params={},
    )

    assert caked == (7.0, 4.0)
    assert meta["sim_refined_caked_projection_status"] == "caked_simulation_image_axes"
    assert meta["sim_refined_caked_projection_real_callback"] is False
    assert meta["fit_prediction_uses_fake_or_test_transform"] is False


def test_minus_1_0_10_fit_handoff_audit_lines_print_missing_target_block() -> None:
    lines = gf.build_geometry_fit_qr_handoff_audit_lines([])
    assert lines == [
        "[ra-sim] Qr/Qz fit handoff audit",
        "  <unavailable reason=target_q_group_not_in_fit_dataset>",
    ]


def test_minus_1_0_10_fit_handoff_audit_fit_observed_matches_manual_refined_visual(
    tmp_path,
) -> None:
    test_minus_1_0_10_fit_observed_matches_manual_refined_visual(tmp_path)


def test_minus_1_0_10_fit_handoff_audit_sim_refined_caked_uses_real_projection(
    tmp_path,
) -> None:
    test_minus_1_0_10_sim_refined_caked_uses_real_projection(tmp_path)


def test_minus_1_0_10_fit_handoff_audit_fit_prediction_source_is_explicit(
    tmp_path,
) -> None:
    test_minus_1_0_10_fit_prediction_source_is_explicit(tmp_path)


def test_minus_1_0_10_fit_handoff_audit_no_caked_values_printed_as_detector_px(
    tmp_path,
) -> None:
    test_minus_1_0_10_no_caked_values_printed_as_detector_px(tmp_path)
