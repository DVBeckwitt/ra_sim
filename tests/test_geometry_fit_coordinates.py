from __future__ import annotations

import math

import numpy as np

from ra_sim.gui.geometry_fit_coordinates import (
    background_detector_pair_for_frame,
    caked_angle_pair,
    entry_frame,
    finite_float,
    finite_pair,
    native_detector_anchor,
    native_detector_anchor_with_provenance,
)


def test_finite_float_accepts_scalar_numbers_and_rejects_nonfinite_values() -> None:
    assert finite_float("1.25") == 1.25
    assert finite_float(np.float64(2.5)) == 2.5
    assert finite_float(math.nan) is None
    assert finite_float(math.inf) is None
    assert finite_float("not-a-number") is None


def test_finite_pair_reads_two_finite_mapping_values() -> None:
    assert finite_pair({"x": "3.0", "y": np.float64(4.0)}, "x", "y") == (3.0, 4.0)
    assert finite_pair({"x": 3.0, "y": math.nan}, "x", "y") is None
    assert finite_pair(None, "x", "y") is None


def test_entry_frame_normalizes_missing_and_present_frame_values() -> None:
    assert entry_frame({"frame": " Native_Detector "}, "frame") == "native_detector"
    assert entry_frame({"frame": None}, "frame") == ""
    assert entry_frame(None, "frame") == ""


def test_background_detector_pair_for_frame_requires_matching_frame() -> None:
    entry = {
        "background_detector_input_frame": "native_detector",
        "background_detector_x": "10.5",
        "background_detector_y": "11.5",
    }

    assert background_detector_pair_for_frame(entry, "native_detector") == (10.5, 11.5)
    assert background_detector_pair_for_frame(entry, "detector_display") is None


def test_native_detector_anchor_prefers_background_detector_native_frame() -> None:
    entry = {
        "background_detector_input_frame": "native_detector",
        "background_detector_x": 1.0,
        "background_detector_y": 2.0,
        "detector_native_x": 3.0,
        "detector_native_y": 4.0,
    }

    assert native_detector_anchor(entry) == (1.0, 2.0)
    assert native_detector_anchor_with_provenance(entry) == (
        (1.0, 2.0),
        "saved_background_detector_native",
    )


def test_native_detector_anchor_falls_back_to_saved_native_aliases() -> None:
    assert native_detector_anchor({"native_col": 5.0, "native_row": 6.0}) == (5.0, 6.0)
    assert native_detector_anchor_with_provenance(
        {"detector_native_x": 7.0, "detector_native_y": 8.0}
    ) == ((7.0, 8.0), "saved_detector_native_xy")


def test_caked_angle_pair_uses_first_finite_x_and_y_aliases() -> None:
    entry = {
        "caked_x": math.nan,
        "background_two_theta_deg": "22.25",
        "caked_y": math.inf,
        "background_phi_deg": "-35.5",
    }

    assert caked_angle_pair(
        entry,
        x_keys=("caked_x", "background_two_theta_deg"),
        y_keys=("caked_y", "background_phi_deg"),
    ) == (22.25, -35.5)
    assert (
        caked_angle_pair(
            {"caked_x": 1.0},
            x_keys=("caked_x",),
            y_keys=("caked_y",),
        )
        is None
    )
