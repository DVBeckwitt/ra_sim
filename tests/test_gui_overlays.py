from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ra_sim.gui import geometry_fit, manual_geometry, overlays


def test_clear_artists_removes_existing_plot_artists() -> None:
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0.0, 1.0], [0.0, 1.0])
        text = ax.text(0.5, 0.5, "marker")
        artists = [line, text]
        draws: list[str] = []

        overlays.clear_artists(
            artists,
            draw_idle=lambda: draws.append("draw"),
            redraw=True,
        )

        assert artists == []
        assert line not in ax.lines
        assert text not in ax.texts
        assert draws == ["draw"]
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_renders_markers_labels_and_residual_arrow() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                geometry_pick_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "hkl": (1, 0, 0),
                    "initial_sim_display": (10.0, 12.0),
                    "initial_bg_display": (14.0, 16.0),
                    "final_sim_display": (11.0, 13.0),
                    "final_bg_display": (15.0, 17.0),
                    "overlay_distance_px": 2.5,
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
        )

        labels = [artist.get_text() for artist in ax.texts]

        assert len(geometry_pick_artists) >= 5
        assert any("fit sim" in label for label in labels)
        assert any("|Δ|=5.7px" in label for label in labels)
        assert draws == ["draw"]
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_labels_caked_view_distance_from_drawn_points() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                geometry_pick_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "hkl": (0, 0, 6),
                    "initial_sim_display": (900.0, 900.0),
                    "initial_bg_display": (910.0, 910.0),
                    "final_sim_display": (950.0, 950.0),
                    "final_bg_display": (10.0, 10.0),
                    "initial_sim_caked_display": (40.0, -10.0),
                    "initial_bg_caked_display": (41.0, -11.0),
                    "final_sim_caked_display": (40.7, -10.4),
                    "final_bg_caked_display": (40.6, -10.5),
                    "overlay_distance_px": 1234.0,
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
            show_caked_2d=True,
        )

        labels = [artist.get_text() for artist in ax.texts]

        assert any("|Δ|=0.1deg" in label for label in labels)
        assert not any("|Δ|=1234.0px" in label for label in labels)
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_captures_fit_sim_artist_point() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []
        visual_probe_records: list[dict[str, object]] = []

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "dataset_index": 2,
                    "pair_id": "pair-4",
                    "q_group_key": ("q_group", "primary", 1, 10),
                    "hkl": (1, 0, 1),
                    "overlay_match_index": 4,
                    "source_branch_index": 1,
                    "source_table_index": 160,
                    "source_row_index": 158,
                    "source_peak_index": 1,
                    "match_status": "matched",
                    "initial_sim_display": (8.0, 9.0),
                    "initial_bg_display": (8.5, 9.5),
                    "final_sim_display": (11.0, 13.0),
                    "final_sim_display_source": "fit_prediction_detector_display_px",
                    "final_bg_display": (12.0, 14.0),
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=lambda *, redraw=True: None,
            draw_idle=lambda: None,
            visual_probe_records=visual_probe_records,
        )

        assert len(visual_probe_records) == 1
        assert visual_probe_records[0]["overlay_match_index"] == 4
        assert visual_probe_records[0]["dataset_index"] == 2
        assert visual_probe_records[0]["pair_id"] == "pair-4"
        assert visual_probe_records[0]["q_group_key"] == ("q_group", "primary", 1, 10)
        assert visual_probe_records[0]["hkl"] == (1, 0, 1)
        assert visual_probe_records[0]["source_branch_index"] == 1
        assert visual_probe_records[0]["source_table_index"] == 160
        assert visual_probe_records[0]["source_row_index"] == 158
        assert visual_probe_records[0]["source_peak_index"] == 1
        assert visual_probe_records[0]["record_point"] == pytest.approx((11.0, 13.0))
        assert visual_probe_records[0]["artist_point"] == pytest.approx((11.0, 13.0))
        assert visual_probe_records[0]["record_source"] == "fit_prediction_detector_display_px"
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_links_sim_and_background_points() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                geometry_pick_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            [
                {
                    "hkl": (1, 2, 3),
                    "sim_display": np.array([8.0, 9.0]),
                    "bg_display": np.array([10.0, 11.0]),
                    "qr": 1.234567890123,
                    "qz": -2.345678901234,
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
        )

        labels = [artist.get_text() for artist in ax.texts]

        assert len(geometry_pick_artists) == 3
        assert any("(1, 2, 3)" in label for label in labels)
        assert any("Qr=1.23" in label for label in labels)
        assert any("Qz=-2.35" in label for label in labels)
        assert draws == ["draw"]
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_can_hide_pair_connectors() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            [
                {
                    "hkl": (1, 2, 3),
                    "sim_display": np.array([8.0, 9.0]),
                    "bg_display": np.array([10.0, 11.0]),
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
            show_pair_connectors=False,
        )

        line_segments = [
            line for line in ax.lines if len(line.get_xdata()) == 2 and len(line.get_ydata()) == 2
        ]
        labels = [artist.get_text() for artist in ax.texts]

        assert line_segments == []
        assert any("(1, 2, 3)" in label for label in labels)
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_draws_q_group_line_segments() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            [
                {
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "sim_display": (10.0, 20.0),
                    "bg_display": (12.0, 22.0),
                },
                {
                    "hkl": (-1, 1, 0),
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "sim_display": (30.0, 24.0),
                    "bg_display": (32.0, 26.0),
                },
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        line_segments = [
            line for line in ax.lines if len(line.get_xdata()) == 2 and len(line.get_ydata()) == 2
        ]

        assert len(line_segments) == 2
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_uses_one_faint_q_group_label() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            [
                {
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "sim_display": (10.0, 20.0),
                    "bg_display": (12.0, 22.0),
                },
                {
                    "hkl": (-1, 1, 0),
                    "q_group_key": ("q_group", "primary", 1, 0),
                    "sim_display": (30.0, 24.0),
                    "bg_display": (32.0, 26.0),
                },
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        labels = [artist.get_text() for artist in ax.texts if artist.get_text()]
        q_group_labels = [label for label in labels if label.startswith("Qr set")]
        label_artist = next(artist for artist in ax.texts if artist.get_text().startswith("Qr set"))

        assert q_group_labels == ["Qr set 1/0"]
        assert all("(1, 1, 0)" not in label for label in labels)
        assert all("(-1, 1, 0)" not in label for label in labels)
        assert float(label_artist.get_fontsize()) < 8.0
        assert float(label_artist.get_alpha()) < 0.7
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_draws_shared_qz_axis_line_for_hk0() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            [
                {
                    "hkl": (0, 0, 2),
                    "q_group_key": ("q_group", "primary", 0, 2),
                    "sim_display": (40.0, 50.0),
                    "bg_display": (42.0, 52.0),
                },
                {
                    "hkl": (0, 0, 4),
                    "q_group_key": ("q_group", "primary", 0, 4),
                    "sim_display": (44.0, 60.0),
                    "bg_display": (46.0, 62.0),
                },
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        line_segments = [
            line for line in ax.lines if len(line.get_xdata()) == 2 and len(line.get_ydata()) == 2
        ]

        assert len(line_segments) == 2
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_uses_detector_reprojected_sim_point() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=False,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "x": 12.0,
                    "y": 14.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 400.0,
                    "display_row": -500.0,
                    "sim_col": 11.0,
                    "sim_row": 13.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.0, 13.0) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (400.0, -500.0) not in marker_points
        assert np.hypot(11.0 - 12.0, 13.0 - 14.0) < 2.0
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_prefers_raw_detector_truth_over_live_sim_alias() -> (
    None
):
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=False,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "x": 12.0,
                    "y": 14.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 300.0,
                    "display_row": -200.0,
                    "sim_col": 400.0,
                    "sim_row": -500.0,
                    "sim_col_raw": 11.0,
                    "sim_row_raw": 13.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.0, 13.0) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (400.0, -500.0) not in marker_points
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_ignores_stale_detector_sim_aliases() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=False,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "x": 12.0,
                    "y": 14.0,
                    "refined_sim_x": 11.0,
                    "refined_sim_y": 13.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 300.0,
                    "display_row": -200.0,
                    "sim_col": 400.0,
                    "sim_row": -500.0,
                    "sim_col_raw": 9.0,
                    "sim_row_raw": 8.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.0, 13.0) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (400.0, -500.0) not in marker_points
        assert np.hypot(11.0 - 12.0, 13.0 - 14.0) < 2.0
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_moves_sim_marker_with_live_theta_source() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        live_theta_candidate = {
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "theta_initial": 7.5,
            "sim_col": 190.0,
            "sim_row": 96.0,
        }

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={"theta_initial": 7.5},
            current_background_index=0,
            prefer_cache=True,
            use_caked_display=False,
            pairs_for_index=lambda _idx: [
                {
                    "q_group_key": ("q_group", "primary", 1, 5),
                    "source_table_index": 9,
                    "source_row_index": 0,
                    "source_reflection_index": 203,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "hkl": (-1, 0, 5),
                    "x": 182.0,
                    "y": 138.0,
                    "refined_sim_x": 30.25,
                    "refined_sim_y": -57.5,
                    "theta_initial": 5.0,
                }
            ],
            get_cache_data=lambda **_kwargs: {
                "simulated_lookup": {
                    manual_geometry.geometry_manual_candidate_source_key(
                        live_theta_candidate,
                    ): dict(live_theta_candidate),
                }
            },
            simulated_peaks_for_params=lambda _params, *, prefer_cache: [],
            build_simulated_lookup=lambda _rows: {},
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (190.0, 96.0) in marker_points
        assert (182.0, 138.0) in marker_points
        assert (30.25, -57.5) not in marker_points
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_prefers_saved_refined_native_detector_coords() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=False,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "x": 12.0,
                    "y": 14.0,
                    "refined_sim_native_x": 11.0,
                    "refined_sim_native_y": 13.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 300.0,
                    "display_row": -200.0,
                    "sim_col": 400.0,
                    "sim_row": -500.0,
                    "sim_col_raw": 9.0,
                    "sim_row_raw": 8.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.0, 13.0) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (400.0, -500.0) not in marker_points
        assert (9.0, 8.0) not in marker_points
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_projects_raw_sim_image_detector_rows_into_background_frame() -> (
    None
):
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((5, 5), dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((5, 5), dtype=float),
        current_background_native=lambda: np.ones((5, 5), dtype=float),
        image_size=lambda: 5,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )
    expected_sim = manual_geometry._default_rotate_point(1.0, 2.0, (5, 5), 3)
    background_point = (float(expected_sim[0]) + 0.25, float(expected_sim[1]) + 0.25)

    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=False,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "x": background_point[0],
                    "y": background_point[1],
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 400.0,
                    "display_row": -500.0,
                    "sim_col": 300.0,
                    "sim_row": -200.0,
                    "sim_col_raw": 1.0,
                    "sim_row_raw": 2.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
            entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert expected_sim in marker_points
        assert background_point in marker_points
        assert (300.0, -200.0) not in marker_points
        assert (400.0, -500.0) not in marker_points
        assert (
            np.hypot(
                float(expected_sim[0]) - background_point[0],
                float(expected_sim[1]) - background_point[1],
            )
            < 1.0
        )
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_uses_caked_reprojected_sim_point() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=True,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "caked_x": 12.0,
                    "caked_y": 14.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 400.0,
                    "display_row": -500.0,
                    "sim_col": 11.0,
                    "sim_row": 13.0,
                    "sim_col_raw": 9.0,
                    "sim_row_raw": 8.0,
                    "caked_x": 11.5,
                    "caked_y": 13.5,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (
                float(entry["caked_x"]),
                float(entry["caked_y"]),
            ),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.5, 13.5) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (400.0, -500.0) not in marker_points
        assert np.hypot(11.5 - 12.0, 13.5 - 14.0) < 1.0
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_projects_raw_detector_rows_into_caked_angles(
    monkeypatch,
) -> None:
    bundle = geometry_fit.CakeTransformBundle(
        detector_shape=(8, 8),
        radial_deg=np.linspace(10.0, 17.0, 8),
        raw_azimuth_deg=np.linspace(13.0, 20.0, 8),
        gui_azimuth_deg=np.linspace(13.0, 20.0, 8),
        lut=object(),
    )
    monkeypatch.setattr(
        manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (11.5, 13.5)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (None, None)
        ),
    )
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(13.0, 20.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
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

    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=True,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "caked_x": 12.0,
                    "caked_y": 14.0,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 400.0,
                    "display_row": -500.0,
                    "sim_col": 300.0,
                    "sim_row": -200.0,
                    "sim_col_raw": 3.0,
                    "sim_row_raw": 4.0,
                    "caked_x": 401.0,
                    "caked_y": -499.0,
                    "raw_caked_x": 402.0,
                    "raw_caked_y": -498.0,
                    "two_theta_deg": 403.0,
                    "phi_deg": -497.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
            entry_display_coords=lambda entry: (
                float(entry["caked_x"]),
                float(entry["caked_y"]),
            ),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.5, 13.5) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (300.0, -200.0) not in marker_points
        assert (401.0, -499.0) not in marker_points
        assert np.hypot(11.5 - 12.0, 13.5 - 14.0) < 1.0
    finally:
        plt.close(fig)


def test_draw_initial_geometry_pairs_overlay_ignores_stale_caked_sim_aliases() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        _measured, initial_pairs = manual_geometry.build_geometry_manual_initial_pairs_display(
            0,
            param_set={},
            current_background_index=0,
            prefer_cache=False,
            use_caked_display=True,
            pairs_for_index=lambda idx: [
                {
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "hkl": (1, 1, 0),
                    "caked_x": 12.0,
                    "caked_y": 14.0,
                    "refined_sim_caked_x": 11.5,
                    "refined_sim_caked_y": 13.5,
                }
            ],
            get_cache_data=lambda **kwargs: {},
            simulated_peaks_for_params=lambda params, *, prefer_cache: [
                {
                    "display_col": 400.0,
                    "display_row": -500.0,
                    "sim_col": 11.0,
                    "sim_row": 13.0,
                    "sim_col_raw": 9.0,
                    "sim_row_raw": 8.0,
                    "caked_x": 401.0,
                    "caked_y": -499.0,
                    "raw_caked_x": 402.0,
                    "raw_caked_y": -498.0,
                    "two_theta_deg": 403.0,
                    "phi_deg": -497.0,
                    "hkl": (1, 1, 0),
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            build_simulated_lookup=lambda rows: {(1, 2): dict(rows[0])},
            entry_display_coords=lambda entry: (
                float(entry["caked_x"]),
                float(entry["caked_y"]),
            ),
        )

        overlays.draw_initial_geometry_pairs_overlay(
            ax,
            initial_pairs,
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (11.5, 13.5) in marker_points
        assert (12.0, 14.0) in marker_points
        assert (401.0, -499.0) not in marker_points
        assert np.hypot(11.5 - 12.0, 13.5 - 14.0) < 1.0
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_projects_native_points_in_caked_view() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "hkl": (1, 0, 0),
                    "initial_sim_display": (999.0, 999.0),
                    "initial_bg_display": (888.0, 888.0),
                    "final_sim_display": (11.0, 13.0),
                    "final_bg_display": (15.0, 17.0),
                    "initial_sim_native": (1.0, 2.0),
                    "initial_bg_native": (3.0, 4.0),
                    "final_sim_native": (5.0, 6.0),
                    "final_bg_native": (7.0, 8.0),
                    "overlay_distance_px": 2.5,
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
            show_caked_2d=True,
            native_detector_coords_to_caked_display_coords=(
                lambda col, row: (100.0 + float(col), 200.0 + float(row))
            ),
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (101.0, 202.0) in marker_points
        assert (103.0, 204.0) in marker_points
        assert (105.0, 206.0) in marker_points
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_prefers_locked_caked_manual_points() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_pick_artists, redraw=redraw)

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "hkl": (1, 0, 0),
                    "initial_sim_display": (999.0, 999.0),
                    "initial_bg_display": (888.0, 888.0),
                    "initial_sim_caked_display": (10.0, 20.0),
                    "initial_bg_caked_display": (30.0, 40.0),
                    "final_bg_caked_display": (30.0, 40.0),
                    "initial_sim_native": (1.0, 2.0),
                    "initial_bg_native": (3.0, 4.0),
                    "final_sim_native": (5.0, 6.0),
                    "final_bg_native": (7.0, 8.0),
                    "overlay_distance_px": 2.5,
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: None,
            show_caked_2d=True,
            native_detector_coords_to_caked_display_coords=(
                lambda col, row: (100.0 + float(col), 200.0 + float(row))
            ),
        )

        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (10.0, 20.0) in marker_points
        assert (30.0, 40.0) in marker_points
        assert (105.0, 206.0) in marker_points
        assert (101.0, 202.0) not in marker_points
        assert (103.0, 204.0) not in marker_points
    finally:
        plt.close(fig)


def test_draw_geometry_fit_overlay_keeps_initial_only_unmatched_pairs_visible() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_pick_artists: list[object] = []
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                geometry_pick_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_geometry_fit_overlay(
            ax,
            [
                {
                    "hkl": (2, 0, 0),
                    "match_status": "missing_pair",
                    "initial_sim_display": (20.0, 22.0),
                    "initial_bg_display": (24.0, 26.0),
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
        )

        labels = [artist.get_text() for artist in ax.texts]
        marker_points = {
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.lines
            if len(line.get_xdata()) == 1
        }

        assert (20.0, 22.0) in marker_points
        assert (24.0, 26.0) in marker_points
        assert any("(2, 0, 0)" in label for label in labels)
        assert draws == ["draw"]
    finally:
        plt.close(fig)


def test_draw_live_geometry_preview_overlay_marks_excluded_matches() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_preview_artists: list[object] = []
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                geometry_preview_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_live_geometry_preview_overlay(
            ax,
            [
                {"hkl": (1, 0, 0), "sim_x": 5.0, "sim_y": 6.0, "x": 7.0, "y": 8.0},
                {
                    "hkl": (2, 0, 0),
                    "sim_x": 9.0,
                    "sim_y": 10.0,
                    "x": 11.0,
                    "y": 12.0,
                    "excluded": True,
                },
            ],
            geometry_preview_artists=geometry_preview_artists,
            clear_geometry_preview_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
            normalize_hkl_key=(lambda value: tuple(value) if isinstance(value, tuple) else None),
            live_preview_match_is_excluded=lambda entry: bool(entry.get("excluded")),
        )

        labels = [artist.get_text() for artist in ax.texts]
        line_styles = [line.get_linestyle() for line in ax.lines if len(line.get_xdata()) == 2]

        assert len(geometry_preview_artists) == 8
        assert any("excluded" in label for label in labels)
        assert ":" in line_styles
        assert "--" in line_styles
        assert draws == ["draw"]
    finally:
        plt.close(fig)


def test_draw_live_geometry_preview_overlay_can_hide_pair_connectors() -> None:
    fig, ax = plt.subplots()
    try:
        geometry_preview_artists: list[object] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(geometry_preview_artists, redraw=redraw)

        overlays.draw_live_geometry_preview_overlay(
            ax,
            [
                {"hkl": (1, 0, 0), "sim_x": 5.0, "sim_y": 6.0, "x": 7.0, "y": 8.0},
            ],
            geometry_preview_artists=geometry_preview_artists,
            clear_geometry_preview_artists=_clear,
            draw_idle=lambda: None,
            normalize_hkl_key=(lambda value: tuple(value) if isinstance(value, tuple) else None),
            live_preview_match_is_excluded=lambda _entry: False,
            show_pair_connectors=False,
        )

        line_segments = [
            line for line in ax.lines if len(line.get_xdata()) == 2 and len(line.get_ydata()) == 2
        ]
        labels = [artist.get_text() for artist in ax.texts]

        assert line_segments == []
        assert any("(1, 0, 0)" in label for label in labels)
    finally:
        plt.close(fig)


def test_draw_qr_cylinder_overlay_paths_replaces_previous_lines() -> None:
    fig, ax = plt.subplots()
    try:
        (old_line,) = ax.plot([0.0, 1.0], [1.0, 0.0])
        qr_cylinder_overlay_artists: list[object] = [old_line]
        draws: list[str] = []

        def _clear(*, redraw: bool = True) -> None:
            overlays.clear_artists(
                qr_cylinder_overlay_artists,
                draw_idle=lambda: draws.append("clear"),
                redraw=redraw,
            )

        overlays.draw_qr_cylinder_overlay_paths(
            ax,
            [
                {
                    "source": "primary",
                    "cols": np.asarray([0.0, 1.0, 2.0], dtype=float),
                    "rows": np.asarray([2.0, 1.5, 1.0], dtype=float),
                },
                {
                    "source": "secondary",
                    "cols": np.asarray([0.0, 1.0, 2.0], dtype=float),
                    "rows": np.asarray([1.0, 0.5, 0.0], dtype=float),
                },
            ],
            qr_cylinder_overlay_artists=qr_cylinder_overlay_artists,
            clear_qr_cylinder_overlay_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
            redraw=True,
        )

        colors = [line.get_color() for line in ax.lines]

        assert old_line not in ax.lines
        assert len(qr_cylinder_overlay_artists) == 2
        assert "#fff06a" in colors
        assert "#78d7ff" in colors
        assert draws == ["draw"]
    finally:
        plt.close(fig)
