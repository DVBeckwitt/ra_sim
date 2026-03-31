from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ra_sim.gui import overlays


def test_clear_artists_removes_existing_plot_artists() -> None:
    fig, ax = plt.subplots()
    try:
        line, = ax.plot([0.0, 1.0], [0.0, 1.0])
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
        assert any("|Δ|=2.5px" in label for label in labels)
        assert draws == ["draw"]
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
                }
            ],
            geometry_pick_artists=geometry_pick_artists,
            clear_geometry_pick_artists=_clear,
            draw_idle=lambda: draws.append("draw"),
        )

        labels = [artist.get_text() for artist in ax.texts]

        assert len(geometry_pick_artists) == 3
        assert any("(1, 2, 3)" in label for label in labels)
        assert draws == ["draw"]
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
            normalize_hkl_key=lambda value: tuple(value) if isinstance(value, tuple) else None,
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


def test_draw_qr_cylinder_overlay_paths_replaces_previous_lines() -> None:
    fig, ax = plt.subplots()
    try:
        old_line, = ax.plot([0.0, 1.0], [1.0, 0.0])
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
