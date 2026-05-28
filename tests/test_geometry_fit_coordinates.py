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
    observed_detector_anchor_for_caked_projection,
    project_detector_anchor_to_caked_fit_space,
    resolve_fit_space_anchor,
    simulated_detector_anchor_for_caked_projection,
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


def test_observed_detector_anchor_for_caked_projection_prefers_native_detector() -> None:
    entry = {
        "background_detector_input_frame": "native_detector",
        "background_detector_x": 10.0,
        "background_detector_y": 20.0,
        "fit_detector_x": 30.0,
        "fit_detector_y": 40.0,
        "caked_x": -777.0,
        "caked_y": -888.0,
    }

    assert observed_detector_anchor_for_caked_projection(entry) == {
        "point": (10.0, 20.0),
        "space": "detector_px",
        "frame": "native_detector",
        "authority": "saved_native",
        "provenance": "saved_background_detector_native",
        "fresh": True,
        "input_frame": "native_detector",
    }


def test_observed_detector_anchor_for_caked_projection_rejects_missing_nonfinite_and_caked_aliases() -> None:
    assert observed_detector_anchor_for_caked_projection({}) is None
    assert (
        observed_detector_anchor_for_caked_projection(
            {
                "background_detector_input_frame": "native_detector",
                "background_detector_x": math.nan,
                "background_detector_y": 20.0,
            }
        )
        is None
    )
    assert (
        observed_detector_anchor_for_caked_projection(
            {"caked_x": 12.5, "caked_y": -4.0, "background_two_theta_deg": 12.5}
        )
        is None
    )


def test_observed_detector_anchor_for_caked_projection_uses_fit_detector_fallback() -> None:
    assert observed_detector_anchor_for_caked_projection(
        {"fit_detector_x": 30.0, "fit_detector_y": 40.0}
    ) == {
        "point": (30.0, 40.0),
        "space": "detector_px",
        "frame": "fit_detector",
        "authority": "display_cache",
        "provenance": "fit_detector_coords",
        "fresh": True,
        "input_frame": "fit_detector",
    }
    assert observed_detector_anchor_for_caked_projection(
        {
            "detector_x": 31.0,
            "detector_y": 41.0,
            "detector_input_frame": "fit_detector",
        }
    ) == {
        "point": (31.0, 41.0),
        "space": "detector_px",
        "frame": "fit_detector",
        "authority": "display_cache",
        "provenance": "detector_fit_frame",
        "fresh": True,
        "input_frame": "fit_detector",
    }


def test_simulated_detector_anchor_for_caked_projection_uses_native_source_rows_without_mutation() -> None:
    entry = {"caked_x": -777.0, "caked_y": -888.0}
    source_row = {"sim_native": (12.0, 34.0), "sim_native_source": "provider_detector_native"}

    assert simulated_detector_anchor_for_caked_projection(entry, source_row) == {
        "point": (12.0, 34.0),
        "space": "detector_px",
        "frame": "native_detector",
        "authority": "saved_native",
        "provenance": "provider_detector_native",
        "fresh": True,
        "input_frame": "native_detector",
    }
    assert source_row == {
        "sim_native": (12.0, 34.0),
        "sim_native_source": "provider_detector_native",
    }


def test_simulated_detector_anchor_for_caked_projection_rejects_missing_nonfinite_and_caked_aliases() -> None:
    assert simulated_detector_anchor_for_caked_projection({}, {}) is None
    assert (
        simulated_detector_anchor_for_caked_projection(
            {"sim_native": (1.0, math.inf)},
            None,
        )
        is None
    )
    assert (
        simulated_detector_anchor_for_caked_projection(
            {"predicted_caked_deg": (1.0, 2.0), "caked_x": 1.0, "caked_y": 2.0},
            None,
        )
        is None
    )


def test_project_detector_anchor_to_caked_fit_space_uses_exact_projector_output() -> None:
    calls: list[dict[str, object]] = []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        calls.append(
            {
                "cols": list(cols),
                "rows": list(rows),
                "local_params": local_params,
                "anchor_kind": anchor_kind,
                "input_frame": input_frame,
            }
        )
        return {
            "two_theta_deg": [12.5],
            "phi_deg": [-4.25],
            "caked_projection_source": "fit_space_projector_native_detector",
            "valid": True,
        }

    assert project_detector_anchor_to_caked_fit_space(
        (100.0, 200.0),
        projector,
        local_params={"theta_initial": 2.0},
        anchor_kind="measured",
        input_frame="native_detector",
    ) == {
        "point": (12.5, -4.25),
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": "fit_space_projector_native_detector",
        "fresh": True,
        "source": "fit_space_projector_native_detector",
        "unavailable_reason": None,
    }
    assert calls == [
        {
            "cols": [100.0],
            "rows": [200.0],
            "local_params": {"theta_initial": 2.0},
            "anchor_kind": "measured",
            "input_frame": "native_detector",
        }
    ]


def test_project_detector_anchor_to_caked_fit_space_rejects_nonfinite_output() -> None:
    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        return {"two_theta_deg": [math.nan], "phi_deg": [1.0], "valid": True}

    assert project_detector_anchor_to_caked_fit_space((100.0, 200.0), projector) == {
        "point": None,
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": "",
        "fresh": False,
        "source": None,
        "unavailable_reason": "nonfinite_measured_caked_projection",
    }


def test_project_detector_anchor_to_caked_fit_space_rejects_missing_output() -> None:
    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        return {"two_theta_deg": [], "phi_deg": []}

    assert project_detector_anchor_to_caked_fit_space((100.0, 200.0), projector)[
        "unavailable_reason"
    ] == "fit_space_projector_returned_no_caked_point"


def test_resolve_fit_space_anchor_keeps_detector_objective_projector_free() -> None:
    entry = {"native_col": 10.0, "native_row": 20.0}

    assert resolve_fit_space_anchor(entry, None, "detector_px", None) == {
        "point": (10.0, 20.0),
        "space": "detector_px",
        "frame": "native_detector",
        "authority": "saved_native",
        "provenance": "saved_native_col_row",
        "fresh": True,
        "input_frame": "native_detector",
    }


def test_resolve_fit_space_anchor_requires_projector_for_caked_objective() -> None:
    entry = {"native_col": 10.0, "native_row": 20.0, "caked_x": -777.0, "caked_y": -888.0}

    assert resolve_fit_space_anchor(entry, None, "caked_deg", None) == {
        "point": None,
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": "",
        "fresh": False,
        "source": None,
        "unavailable_reason": "fit_space_projector_unavailable",
    }


def test_resolve_fit_space_anchor_projects_detector_native_for_caked_objective() -> None:
    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        return {
            "two_theta_deg": [float(cols[0]) + 0.5],
            "phi_deg": [float(rows[0]) - 0.25],
            "caked_projection_source": "fit_space_projector_native_detector",
            "valid": True,
        }

    assert resolve_fit_space_anchor(
        {"native_col": 10.0, "native_row": 20.0, "caked_x": -777.0, "caked_y": -888.0},
        None,
        "caked_deg",
        projector,
    ) == {
        "point": (10.5, 19.75),
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": "fit_space_projector_native_detector",
        "fresh": True,
        "source": "fit_space_projector_native_detector",
        "unavailable_reason": None,
    }
