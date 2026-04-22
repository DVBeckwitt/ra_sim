"""Coordinate parity diagnostics for geometry-fit visual/backend handoffs."""

from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ra_sim.gui import geometry_fit, manual_geometry
from ra_sim.gui.geometry_overlay import normalize_initial_geometry_pairs_display
from ra_sim.gui.overlays import draw_initial_geometry_pairs_overlay

SURFACE_ORDER = (
    "provider_pairs",
    "manual_point_pairs",
    "initial_pairs_display",
    "measured_for_fit",
    'spec["measured_peaks"]',
    "optimizer_request.measured_peaks",
)
REQUIRED_SURFACES = SURFACE_ORDER[:-1]
STORED_POINT_ABS_TOLERANCE_PX = 1.0e-6


def _load_new4_geometry_fit_ladder_module():
    module_name = "scripts.debug.run_new4_geometry_fit_ladder"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or "")
        if missing == "scripts" or missing.startswith("scripts."):
            raise RuntimeError(
                "Optimizer request capture requires "
                "scripts/debug/run_new4_geometry_fit_ladder.py in a source checkout."
            ) from exc
        raise


CSV_COLUMNS = [
    "pair_index",
    "pair_id",
    "hkl",
    "source_branch_index",
    "q_group_key",
    "visual_background_x",
    "visual_background_y",
    "visual_simulated_x",
    "visual_simulated_y",
    "visual_dx",
    "visual_dy",
    "visual_distance_px",
    "surface_name",
    "surface_background_x",
    "surface_background_y",
    "surface_simulated_x",
    "surface_simulated_y",
    "surface_dx",
    "surface_dy",
    "surface_distance_px",
    "background_delta_x",
    "background_delta_y",
    "background_delta_px",
    "simulated_delta_x",
    "simulated_delta_y",
    "simulated_delta_px",
    "vector_delta_x",
    "vector_delta_y",
    "vector_delta_px",
    "background_frame_match",
    "simulated_frame_match",
]


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if np.isfinite(out) else None


def _point(value: object) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    x_val = _finite_float(value[0])
    y_val = _finite_float(value[1])
    if x_val is None or y_val is None:
        return None
    return [float(x_val), float(y_val)]


def _point_from_keys(row: Mapping[str, object], keys: tuple[str, str]) -> list[float] | None:
    return _point((row.get(keys[0]), row.get(keys[1])))


def _identity_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row.get("pair_id"),
        _jsonable(row.get("hkl", row.get("normalized_hkl"))),
        row.get("source_branch_index"),
        _jsonable(row.get("q_group_key", row.get("branch_group_key"))),
    )


def _identity_key_text(row: Mapping[str, object]) -> str:
    return json.dumps(_identity_key(row), sort_keys=True, separators=(",", ":"))


def _pair_id(row: Mapping[str, object], pair_index: int) -> str:
    if row.get("pair_id") is not None:
        return str(row["pair_id"])
    order_key = row.get("manual_pair_order_key")
    if isinstance(order_key, Sequence) and not isinstance(order_key, (str, bytes)):
        if len(order_key) >= 2:
            return f"bg{int(order_key[0])}:pair{int(order_key[1])}"
        return ":".join(str(part) for part in order_key)
    if row.get("background_index") is not None:
        return f"bg{int(row['background_index'])}:pair{int(pair_index)}"
    return f"pair:{int(pair_index)}"


def _hkl(row: Mapping[str, object]) -> object:
    return _jsonable(row.get("hkl", row.get("normalized_hkl", row.get("label"))))


def _q_group_key(row: Mapping[str, object]) -> object:
    return _jsonable(row.get("q_group_key", row.get("branch_group_key")))


def _source_branch_index(row: Mapping[str, object]) -> int | None:
    try:
        value = row.get("source_branch_index")
        if value is None:
            identity = row.get("selected_source_identity_canonical")
            if isinstance(identity, Mapping):
                value = identity.get("source_branch_index")
        return int(value) if value is not None else None
    except Exception:
        return None


def _normalize_frame(value: object) -> str:
    text = str(value or "unknown").strip()
    return text if text else "unknown"


def _distance(point_a: Sequence[float] | None, point_b: Sequence[float] | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    return float(
        math.hypot(float(point_b[0]) - float(point_a[0]), float(point_b[1]) - float(point_a[1]))
    )


def _vector(
    background: Sequence[float] | None, simulated: Sequence[float] | None
) -> list[float] | None:
    if background is None or simulated is None:
        return None
    return [
        float(simulated[0]) - float(background[0]),
        float(simulated[1]) - float(background[1]),
    ]


def _delta(lhs: Sequence[float] | None, rhs: Sequence[float] | None) -> list[float] | None:
    if lhs is None or rhs is None:
        return None
    return [float(lhs[0]) - float(rhs[0]), float(lhs[1]) - float(rhs[1])]


def _norm(vec: Sequence[float] | None) -> float | None:
    if vec is None:
        return None
    return float(math.hypot(float(vec[0]), float(vec[1])))


def _matrix_json(matrix: object) -> object:
    try:
        return np.asarray(matrix, dtype=float).tolist()
    except Exception:
        return None


class _AxesDrawSpy:
    def __init__(self, ax: Any):
        self._ax = ax
        self.plot_calls: list[dict[str, object]] = []
        self.scatter_calls: list[dict[str, object]] = []
        self.annotate_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._ax, name)

    def plot(self, x_values: object, y_values: object, *args: object, **kwargs: object) -> object:
        x_list = (
            list(x_values)
            if isinstance(x_values, Sequence) and not isinstance(x_values, (str, bytes))
            else [x_values]
        )
        y_list = (
            list(y_values)
            if isinstance(y_values, Sequence) and not isinstance(y_values, (str, bytes))
            else [y_values]
        )
        marker = args[0] if args else kwargs.get("marker")
        self.plot_calls.append(
            {
                "x": [_finite_float(value) for value in x_list],
                "y": [_finite_float(value) for value in y_list],
                "args": list(args),
                "kwargs": dict(kwargs),
                "marker": str(marker) if marker is not None else None,
            }
        )
        return self._ax.plot(x_values, y_values, *args, **kwargs)

    def scatter(
        self, x_values: object, y_values: object, *args: object, **kwargs: object
    ) -> object:
        self.scatter_calls.append(
            {
                "x": _jsonable(x_values),
                "y": _jsonable(y_values),
                "args": list(args),
                "kwargs": dict(kwargs),
            }
        )
        return self._ax.scatter(x_values, y_values, *args, **kwargs)

    def annotate(self, *args: object, **kwargs: object) -> object:
        self.annotate_calls.append({"args": list(args), "kwargs": dict(kwargs)})
        return self._ax.annotate(*args, **kwargs)

    def text(self, *args: object, **kwargs: object) -> object:
        self.text_calls.append({"args": list(args), "kwargs": dict(kwargs)})
        return self._ax.text(*args, **kwargs)


def _canvas_point(ax: Any, point: Sequence[float] | None) -> list[float] | None:
    if point is None:
        return None
    try:
        transformed = ax.transData.transform((float(point[0]), float(point[1])))
    except Exception:
        return None
    return [float(transformed[0]), float(transformed[1])]


def _plotting_metadata(ax: Any) -> dict[str, object]:
    image_extent = None
    image_shape = None
    origin = None
    try:
        images = list(getattr(ax, "images", []) or [])
    except Exception:
        images = []
    if images:
        image = images[0]
        try:
            image_extent = _jsonable(list(image.get_extent()))
        except Exception:
            image_extent = None
        try:
            image_shape = _jsonable(tuple(np.asarray(image.get_array()).shape))
        except Exception:
            image_shape = None
        try:
            origin = str(image.origin)
        except Exception:
            try:
                origin = str(image.get_origin())
            except Exception:
                origin = None

    try:
        fig = ax.figure
    except Exception:
        fig = None
    return {
        "axes_xlim": _jsonable(list(ax.get_xlim())) if hasattr(ax, "get_xlim") else None,
        "axes_ylim": _jsonable(list(ax.get_ylim())) if hasattr(ax, "get_ylim") else None,
        "image_extent": image_extent,
        "image_shape": image_shape,
        "origin": origin,
        "y_axis_inverted": bool(ax.yaxis_inverted()) if hasattr(ax, "yaxis_inverted") else None,
        "display_transform_matrix": _matrix_json(
            getattr(ax.transData, "get_matrix", lambda: None)()
        ),
        "canvas_pixel_transform": (
            _matrix_json(getattr(fig.dpi_scale_trans, "get_matrix", lambda: None)())
            if fig is not None and hasattr(fig, "dpi_scale_trans")
            else None
        ),
    }


def build_visual_overlay_records_from_saved_entries(
    saved_entries: Sequence[Mapping[str, object]] | None,
    *,
    background_index: int = 0,
) -> list[dict[str, object]]:
    """Build overlay input from saved GUI manual-pair fields, not backend surfaces."""

    records: list[dict[str, object]] = []
    for pair_index, raw_entry in enumerate(saved_entries or ()):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        bg_point = _point_from_keys(entry, ("x", "y"))
        sim_point = _point_from_keys(entry, ("refined_sim_x", "refined_sim_y"))
        if sim_point is None:
            sim_point = _point_from_keys(entry, ("sim_col", "sim_row"))
        record = {
            "pair_index": int(pair_index),
            "pair_id": str(
                entry.get("pair_id") or f"bg{int(background_index)}:pair{int(pair_index)}"
            ),
            "background_index": int(background_index),
            "hkl": _hkl(entry),
            "label": entry.get("label"),
            "source_branch_index": _source_branch_index(entry),
            "q_group_key": _q_group_key(entry),
            "branch_group_key": _jsonable(entry.get("branch_group_key")),
            "bg_display": bg_point,
            "sim_display": sim_point,
            "visual_overlay_input_source": "saved_gui_manual_pair_fields",
        }
        records.append(record)
    return records


def _manual_render_entry_display_coords(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    for keys in (
        ("x", "y"),
        ("display_col", "display_row"),
        ("background_x", "background_y"),
        ("background_col", "background_row"),
        ("caked_x", "caked_y"),
    ):
        point = _point_from_keys(entry, keys)
        if point is not None:
            return float(point[0]), float(point[1])
    return None


def _manual_render_simulated_lookup(
    rows: Sequence[dict[str, object]] | None,
) -> dict[tuple[object, ...], object]:
    lookup: dict[tuple[object, ...], object] = {}
    for raw_row in rows or ():
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        key = manual_geometry.geometry_manual_candidate_source_key(row)
        if key is None:
            continue
        existing = lookup.get(tuple(key))
        if isinstance(existing, Mapping):
            lookup[tuple(key)] = [dict(existing), row]
        elif isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
            lookup[tuple(key)] = [dict(item) for item in existing if isinstance(item, Mapping)] + [
                row
            ]
        else:
            lookup[tuple(key)] = row
    return lookup


def _enrich_manual_overlay_identity(
    overlay_rows: Sequence[Mapping[str, object]],
    saved_entries: Sequence[Mapping[str, object]],
    *,
    background_index: int,
) -> list[dict[str, object]]:
    enriched_rows: list[dict[str, object]] = []
    saved = [dict(entry) for entry in saved_entries if isinstance(entry, Mapping)]
    for fallback_index, raw_row in enumerate(overlay_rows):
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        try:
            pair_index = int(row.get("overlay_match_index", fallback_index) or fallback_index)
        except Exception:
            pair_index = int(fallback_index)
        source = saved[pair_index] if 0 <= pair_index < len(saved) else {}
        row["pair_index"] = int(pair_index)
        row.setdefault("background_index", int(background_index))
        row.setdefault(
            "pair_id",
            source.get("pair_id") or f"bg{int(background_index)}:pair{int(pair_index)}",
        )
        for key in (
            "hkl",
            "label",
            "source_branch_index",
            "q_group_key",
            "branch_group_key",
            "selected_source_identity_canonical",
        ):
            if row.get(key) is None and source.get(key) is not None:
                row[key] = source.get(key)
        row["visual_overlay_input_source"] = "manual_geometry.render_current_geometry_manual_pairs"
        enriched_rows.append(row)
    return enriched_rows


def capture_manual_geometry_overlay_input_from_render_path(
    saved_entries: Sequence[Mapping[str, object]] | None,
    *,
    background_index: int,
) -> dict[str, object]:
    """Capture the exact overlay input handed to the real manual render path."""

    saved = [dict(entry) for entry in saved_entries or () if isinstance(entry, Mapping)]
    captured_rows: list[dict[str, object]] | None = None
    captured_marker_limit: int | None = None

    def _pairs_for_index(index: int) -> list[dict[str, object]]:
        return [dict(entry) for entry in saved] if int(index) == int(background_index) else []

    def _build_initial_pairs_display(
        index: int,
        *,
        param_set: dict[str, object] | None = None,
        prefer_cache: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return manual_geometry.build_geometry_manual_initial_pairs_display(
            int(index),
            param_set=param_set,
            current_background_index=int(background_index),
            prefer_cache=prefer_cache,
            use_caked_display=False,
            pairs_for_index=_pairs_for_index,
            current_geometry_fit_params=lambda: {},
            get_cache_data=lambda **_kwargs: {},
            source_rows_for_background=lambda bg_index, _params=None, **_kwargs: (
                [dict(entry) for entry in saved] if int(bg_index) == int(background_index) else []
            ),
            simulated_peaks_for_params=lambda **_kwargs: [dict(entry) for entry in saved],
            build_simulated_lookup=_manual_render_simulated_lookup,
            project_peaks_to_current_view=lambda rows: [
                dict(row) for row in rows or () if isinstance(row, Mapping)
            ],
            entry_display_coords=_manual_render_entry_display_coords,
            filter_active_rows=lambda rows: [
                dict(row) for row in rows or () if isinstance(row, Mapping)
            ],
        )

    def _capture_overlay_input(
        combined_pairs_display: Sequence[Mapping[str, object]] | None,
        *,
        max_display_markers: int = 120,
        **_kwargs: object,
    ) -> None:
        nonlocal captured_rows, captured_marker_limit
        raw_rows = [dict(row) for row in combined_pairs_display or () if isinstance(row, Mapping)]
        captured_rows = _enrich_manual_overlay_identity(
            raw_rows,
            saved,
            background_index=int(background_index),
        )
        captured_marker_limit = int(max_display_markers)

    try:
        render_source = inspect.getsource(manual_geometry.render_current_geometry_manual_pairs)
    except Exception:
        render_source = ""
    path_confirmed = bool(
        "combined_pairs_display" in render_source
        and "draw_initial_geometry_pairs_overlay" in render_source
    )
    rendered = manual_geometry.render_current_geometry_manual_pairs(
        background_visible=True,
        current_background_index=int(background_index),
        current_background_image=object(),
        pick_session=None,
        build_initial_pairs_display=_build_initial_pairs_display,
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda **_kwargs: None,
        draw_initial_geometry_pairs_overlay=_capture_overlay_input,
        update_button_label_fn=lambda: None,
        set_background_file_status_text_fn=lambda: None,
        pair_group_count=lambda index: len(_pairs_for_index(int(index))),
        update_status=False,
    )
    rows = captured_rows or []
    return {
        "visual_capture_path_confirmed": bool(rendered and path_confirmed and rows),
        "visual_capture_path": (
            "ra_sim.gui.manual_geometry.render_current_geometry_manual_pairs"
            " -> ra_sim.gui.overlays.draw_initial_geometry_pairs_overlay"
        ),
        "visual_capture_function": (
            "ra_sim.gui.manual_geometry.render_current_geometry_manual_pairs"
        ),
        "overlay_draw_function": "ra_sim.gui.overlays.draw_initial_geometry_pairs_overlay",
        "overlay_source": inspect.getsourcefile(draw_initial_geometry_pairs_overlay),
        "overlay_input_pair_count": int(len(rows)),
        "max_display_markers": int(captured_marker_limit or max(1, len(rows))),
        "pairs": rows,
    }


def collect_geometry_visual_pair_positions(
    initial_pairs_display: Sequence[Mapping[str, object]] | None,
    *,
    ax: Any | None = None,
    max_display_markers: int = 120,
    show_pair_connectors: bool = True,
) -> dict[str, object]:
    """Capture actual data coordinates passed to the GUI initial-pair overlay."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.zeros((10, 10)), origin="upper")
        created_figure = True

    normalized = normalize_initial_geometry_pairs_display(initial_pairs_display)
    geometry_pick_artists: list[object] = []
    spy = _AxesDrawSpy(ax)
    draw_initial_geometry_pairs_overlay(
        spy,
        normalized,
        geometry_pick_artists=geometry_pick_artists,
        clear_geometry_pick_artists=lambda **_kwargs: geometry_pick_artists.clear(),
        draw_idle=lambda: None,
        max_display_markers=max_display_markers,
        show_pair_connectors=show_pair_connectors,
    )

    marker_calls = [
        call
        for call in spy.plot_calls
        if len(call.get("x", [])) == 1
        and len(call.get("y", [])) == 1
        and call.get("marker") in {"s", "^"}
    ]
    marker_cursor = 0
    visual_pairs: list[dict[str, object]] = []
    missing_identity = False
    missing_points = False
    limit = max(1, int(max_display_markers))

    for draw_index, entry in enumerate(normalized[:limit]):
        row = dict(entry)
        pair_index = int(row.get("pair_index", draw_index) or draw_index)
        pair_id = _pair_id(row, pair_index)
        sim_record = None
        bg_record = None

        if row.get("sim_display") is not None:
            if (
                marker_cursor < len(marker_calls)
                and marker_calls[marker_cursor].get("marker") == "s"
            ):
                sim_record = marker_calls[marker_cursor]
                marker_cursor += 1
            else:
                missing_points = True
        if row.get("bg_display") is not None:
            if (
                marker_cursor < len(marker_calls)
                and marker_calls[marker_cursor].get("marker") == "^"
            ):
                bg_record = marker_calls[marker_cursor]
                marker_cursor += 1
            else:
                missing_points = True

        sim_point = (
            [float(sim_record["x"][0]), float(sim_record["y"][0])]
            if sim_record is not None
            and sim_record.get("x", [None])[0] is not None
            and sim_record.get("y", [None])[0] is not None
            else None
        )
        bg_point = (
            [float(bg_record["x"][0]), float(bg_record["y"][0])]
            if bg_record is not None
            and bg_record.get("x", [None])[0] is not None
            and bg_record.get("y", [None])[0] is not None
            else None
        )
        if pair_id is None or _hkl(row) is None:
            missing_identity = True
        if bg_point is None or sim_point is None:
            missing_points = True

        visual_pairs.append(
            {
                "pair_index": pair_index,
                "pair_id": pair_id,
                "hkl": _hkl(row),
                "source_branch_index": _source_branch_index(row),
                "q_group_key": _q_group_key(row),
                "branch_group_key": _jsonable(row.get("branch_group_key")),
                "visual_background_point": bg_point,
                "visual_background_frame": "display",
                "visual_background_data_point": bg_point,
                "visual_background_canvas_point": _canvas_point(ax, bg_point),
                "visual_simulated_point": sim_point,
                "visual_simulated_frame": "display",
                "visual_simulated_data_point": sim_point,
                "visual_simulated_canvas_point": _canvas_point(ax, sim_point),
                "visual_background_artist_source": (
                    "draw_initial_geometry_pairs_overlay:ax.plot:^"
                    if bg_record is not None
                    else None
                ),
                "visual_simulated_artist_source": (
                    "draw_initial_geometry_pairs_overlay:ax.plot:s"
                    if sim_record is not None
                    else None
                ),
            }
        )

    source_file = inspect.getsourcefile(draw_initial_geometry_pairs_overlay)
    result = {
        "visual_truth_available": bool(
            visual_pairs and not missing_identity and not missing_points
        ),
        "visual_capture_path": "ra_sim.gui.overlays.draw_initial_geometry_pairs_overlay",
        "visual_capture_path_confirmed": False,
        "visual_capture_function": "ra_sim.gui.overlays.draw_initial_geometry_pairs_overlay",
        "overlay_source": str(source_file) if source_file else None,
        "plotting_metadata": _plotting_metadata(ax),
        "artist_call_counts": {
            "plot": int(len(spy.plot_calls)),
            "scatter": int(len(spy.scatter_calls)),
            "annotate": int(len(spy.annotate_calls)),
            "text": int(len(spy.text_calls)),
        },
        "pairs": visual_pairs,
    }
    if created_figure:
        plt.close(ax.figure)
    return _jsonable(result)  # type: ignore[return-value]


def _base_surface_record(row: Mapping[str, object], pair_index: int) -> dict[str, object]:
    return {
        "pair_index": int(
            row.get("pair_index", row.get("dataset_pair_index", pair_index)) or pair_index
        ),
        "pair_id": _pair_id(row, pair_index),
        "hkl": _hkl(row),
        "source_branch_index": _source_branch_index(row),
        "q_group_key": _q_group_key(row),
        "branch_group_key": _jsonable(row.get("branch_group_key")),
    }


def _provider_surface_record(row: Mapping[str, object], pair_index: int) -> dict[str, object]:
    record = _base_surface_record(row, pair_index)
    record.update(
        {
            "background_point": _point(
                row.get("background_point") or row.get("provider_background_point")
            ),
            "background_frame": _normalize_frame(
                row.get("background_frame", row.get("provider_background_frame"))
            ),
            "simulated_point": _point(
                row.get("simulated_point") or row.get("provider_selected_simulated_point")
            ),
            "simulated_frame": _normalize_frame(
                row.get("simulated_frame", row.get("provider_selected_simulated_frame"))
            ),
        }
    )
    return record


def _manual_surface_record(row: Mapping[str, object], pair_index: int) -> dict[str, object]:
    record = _base_surface_record(row, pair_index)
    record.update(
        {
            "background_point": _point(
                row.get("background_point") or row.get("manual_background_point")
            ),
            "background_frame": _normalize_frame(
                row.get("background_frame", row.get("manual_background_frame"))
            ),
            "simulated_point": _point(
                row.get("simulated_point") or row.get("manual_selected_simulated_point")
            ),
            "simulated_frame": _normalize_frame(
                row.get("simulated_frame", row.get("manual_selected_simulated_frame"))
            ),
        }
    )
    return record


def _initial_surface_record(row: Mapping[str, object], pair_index: int) -> dict[str, object]:
    record = _base_surface_record(row, pair_index)
    record.update(
        {
            "background_point": _point(row.get("bg_display") or row.get("background_point")),
            "background_frame": _normalize_frame(row.get("provider_background_frame", "display")),
            "simulated_point": _point(row.get("sim_display") or row.get("simulated_point")),
            "simulated_frame": _normalize_frame(row.get("provider_simulated_frame", "display")),
        }
    )
    return record


def _measured_surface_record(row: Mapping[str, object], pair_index: int) -> dict[str, object]:
    record = _base_surface_record(row, pair_index)
    bg_point = _point_from_keys(row, ("x", "y")) or _point_from_keys(
        row, ("display_col", "display_row")
    )
    sim_point = _point(row.get("sim_display")) or _point_from_keys(row, ("sim_col", "sim_row"))
    record.update(
        {
            "background_point": bg_point,
            "background_frame": _normalize_frame(
                row.get("provider_background_frame", row.get("detector_input_frame", "display"))
            ),
            "simulated_point": sim_point,
            "simulated_frame": _normalize_frame(row.get("provider_simulated_frame", "display")),
        }
    )
    return record


def _optimizer_surface_record(
    row: Mapping[str, object],
    pair_index: int,
    provider_row: Mapping[str, object] | None = None,
) -> dict[str, object]:
    provider = provider_row if isinstance(provider_row, Mapping) else {}
    merged = {**provider, **dict(row)}
    record = _base_surface_record(merged, pair_index)
    sim_point = (
        _point(row.get("simulated_point"))
        or _point(row.get("sim_display"))
        or _point(provider.get("simulated_point"))
        or _point(provider.get("provider_selected_simulated_point"))
    )
    record.update(
        {
            "background_point": _point_from_keys(row, ("x", "y"))
            or _point(row.get("background_point")),
            "background_frame": _normalize_frame(
                row.get(
                    "provider_background_frame",
                    row.get("detector_input_frame", provider.get("background_frame", "display")),
                )
            ),
            "simulated_point": sim_point,
            "simulated_frame": _normalize_frame(
                row.get("provider_simulated_frame", provider.get("simulated_frame", "display"))
            ),
        }
    )
    return record


def _optimizer_request_missing_fields(row: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    if row.get("pair_index", row.get("dataset_pair_index")) is None:
        missing.append("pair_index")
    if _hkl(row) is None:
        missing.append("hkl")
    if _source_branch_index(row) is None:
        missing.append("source_branch_index")
    if (
        _point_from_keys(row, ("x", "y")) is None
        and _point(row.get("background_point")) is None
        and _point(row.get("measured_point")) is None
        and _point_from_keys(row, ("measured_x", "measured_y")) is None
        and _point_from_keys(row, ("background_x", "background_y")) is None
    ):
        missing.append("background_point")
    if (
        _point(row.get("simulated_point")) is None
        and _point(row.get("sim_display")) is None
        and _point_from_keys(row, ("simulated_x", "simulated_y")) is None
        and _point_from_keys(row, ("sim_x", "sim_y")) is None
    ):
        missing.append("simulated_point")
    if (
        row.get("background_frame") is None
        and row.get("provider_background_frame") is None
        and row.get("detector_input_frame") is None
    ):
        missing.append("background_frame")
    if row.get("simulated_frame") is None and row.get("provider_simulated_frame") is None:
        missing.append("simulated_frame")
    return missing


def _normalize_optimizer_request_row(
    row: Mapping[str, object],
    pair_index: int,
) -> dict[str, object]:
    normalized = _normalize_optimizer_rung_row(row)
    normalized["surface_name"] = "optimizer_request.measured_peaks"
    if normalized.get("pair_index") is None:
        if normalized.get("dataset_pair_index") is not None:
            normalized["pair_index"] = normalized.get("dataset_pair_index")
        elif normalized.get("optimizer_request_pair_index") is not None:
            normalized["pair_index"] = normalized.get("optimizer_request_pair_index")
        else:
            normalized["pair_index"] = int(pair_index)
    if _point(normalized.get("background_point")) is None:
        background = (
            _point_from_keys(normalized, ("x", "y"))
            or _point(normalized.get("measured_point"))
            or _point_from_keys(normalized, ("measured_x", "measured_y"))
            or _point_from_keys(normalized, ("background_x", "background_y"))
        )
        if background is not None:
            normalized["background_point"] = background
    missing_fields = _optimizer_request_missing_fields(normalized)
    if normalized.get("background_frame") is None:
        normalized["background_frame"] = _normalize_frame(
            normalized.get(
                "provider_background_frame",
                normalized.get("detector_input_frame", "display"),
            )
        )
    if normalized.get("simulated_frame") is None:
        normalized["simulated_frame"] = _normalize_frame(
            normalized.get("provider_simulated_frame", "display")
        )
    if missing_fields:
        normalized["missing_optimizer_request_fields"] = missing_fields
    return normalized


def build_backend_surfaces_from_dataset(
    dataset: Mapping[str, object],
    *,
    provider_report: Mapping[str, object] | None = None,
    optimizer_request_rows: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Extract comparable backend surfaces in the required diagnostic order."""

    report_pairs = [
        dict(row)
        for row in (provider_report or {}).get("pairs", ()) or ()
        if isinstance(row, Mapping)
    ]
    provider_rows = report_pairs or [
        dict(row) for row in dataset.get("provider_pairs", ()) or () if isinstance(row, Mapping)
    ]
    manual_rows = report_pairs or [
        dict(row) for row in dataset.get("manual_point_pairs", ()) or () if isinstance(row, Mapping)
    ]
    initial_rows = [
        dict(row)
        for row in dataset.get("initial_pairs_display", ()) or ()
        if isinstance(row, Mapping)
    ]
    measured_rows = [
        dict(row) for row in dataset.get("measured_for_fit", ()) or () if isinstance(row, Mapping)
    ]
    spec = dataset.get("spec") if isinstance(dataset.get("spec"), Mapping) else {}
    spec_rows = [
        dict(row) for row in spec.get("measured_peaks", ()) or () if isinstance(row, Mapping)
    ]

    surfaces = {
        "provider_pairs": [
            _provider_surface_record(row, idx) for idx, row in enumerate(provider_rows)
        ],
        "manual_point_pairs": [
            _manual_surface_record(row, idx) for idx, row in enumerate(manual_rows)
        ],
        "initial_pairs_display": [
            _initial_surface_record(row, idx) for idx, row in enumerate(initial_rows)
        ],
        "measured_for_fit": [
            _measured_surface_record(row, idx) for idx, row in enumerate(measured_rows)
        ],
        'spec["measured_peaks"]': [
            _measured_surface_record(row, idx) for idx, row in enumerate(spec_rows)
        ],
    }

    if optimizer_request_rows is not None:
        rows = [dict(row) for row in optimizer_request_rows if isinstance(row, Mapping)]
        surfaces["optimizer_request.measured_peaks"] = [
            _optimizer_surface_record(
                row,
                idx,
                None,
            )
            for idx, row in enumerate(rows)
        ]
    return surfaces


def _errors(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None, None
    arr = np.asarray(finite, dtype=np.float64)
    return float(math.sqrt(float(np.mean(arr * arr)))), float(np.max(arr))


def _surface_metrics(pair_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    bg_errors = [row.get("background_delta_px") for row in pair_rows]
    sim_errors = [row.get("simulated_delta_px") for row in pair_rows]
    vector_errors = [row.get("vector_delta_px") for row in pair_rows]
    all_point_errors = [
        value
        for row in pair_rows
        for value in (row.get("background_delta_px"), row.get("simulated_delta_px"))
    ]
    bg_rms, bg_max = _errors(bg_errors)  # type: ignore[arg-type]
    sim_rms, sim_max = _errors(sim_errors)  # type: ignore[arg-type]
    vec_rms, vec_max = _errors(vector_errors)  # type: ignore[arg-type]
    all_rms, all_max = _errors(all_point_errors)  # type: ignore[arg-type]
    return {
        "background_rms_error_px": bg_rms,
        "background_max_error_px": bg_max,
        "simulated_rms_error_px": sim_rms,
        "simulated_max_error_px": sim_max,
        "vector_rms_error_px": vec_rms,
        "vector_max_error_px": vec_max,
        "all_points_rms_error_px": all_rms,
        "all_points_max_error_px": all_max,
    }


def _compare_surface(
    visual_pairs: Sequence[Mapping[str, object]],
    surface_name: str,
    surface_rows: Sequence[Mapping[str, object]],
    *,
    tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pair_rows: list[dict[str, object]] = []
    visual_order = [_identity_key_text(row) for row in visual_pairs]
    surface_order = [_identity_key_text(row) for row in surface_rows]
    ordered_pairs_match = bool(visual_order == surface_order)
    unordered_pairs_match = bool(Counter(visual_order) == Counter(surface_order))
    frame_mismatch_count = 0
    background_mismatch_count = 0
    simulated_mismatch_count = 0
    vector_mismatch_count = 0

    max_len = max(len(visual_pairs), len(surface_rows))
    for idx in range(max_len):
        visual = visual_pairs[idx] if idx < len(visual_pairs) else {}
        surface = surface_rows[idx] if idx < len(surface_rows) else {}
        visual_bg = _point(visual.get("visual_background_point"))
        visual_sim = _point(visual.get("visual_simulated_point"))
        surface_bg = _point(surface.get("background_point"))
        surface_sim = _point(surface.get("simulated_point"))
        visual_vec = _vector(visual_bg, visual_sim)
        surface_vec = _vector(surface_bg, surface_sim)
        bg_delta = _delta(surface_bg, visual_bg)
        sim_delta = _delta(surface_sim, visual_sim)
        vector_delta = _delta(surface_vec, visual_vec)
        bg_delta_px = _norm(bg_delta)
        sim_delta_px = _norm(sim_delta)
        vector_delta_px = _norm(vector_delta)
        bg_frame_match = bool(
            _normalize_frame(surface.get("background_frame"))
            == _normalize_frame(visual.get("visual_background_frame"))
        )
        sim_frame_match = bool(
            _normalize_frame(surface.get("simulated_frame"))
            == _normalize_frame(visual.get("visual_simulated_frame"))
        )
        if not bg_frame_match or not sim_frame_match:
            frame_mismatch_count += 1
        if bg_delta_px is None or bg_delta_px > tolerance:
            background_mismatch_count += 1
        if sim_delta_px is None or sim_delta_px > tolerance:
            simulated_mismatch_count += 1
        if vector_delta_px is None or vector_delta_px > tolerance:
            vector_mismatch_count += 1

        pair_rows.append(
            {
                "pair_index": int(visual.get("pair_index", idx) or idx),
                "pair_id": visual.get("pair_id"),
                "hkl": _jsonable(visual.get("hkl")),
                "source_branch_index": visual.get("source_branch_index"),
                "q_group_key": _jsonable(visual.get("q_group_key")),
                "visual_background_point": visual_bg,
                "visual_background_frame": _normalize_frame(visual.get("visual_background_frame")),
                "visual_simulated_point": visual_sim,
                "visual_simulated_frame": _normalize_frame(visual.get("visual_simulated_frame")),
                "visual_vector": visual_vec,
                "visual_distance_px": _distance(visual_bg, visual_sim),
                "surface_name": surface_name,
                "surface_background_point": surface_bg,
                "surface_background_frame": _normalize_frame(surface.get("background_frame")),
                "surface_simulated_point": surface_sim,
                "surface_simulated_frame": _normalize_frame(surface.get("simulated_frame")),
                "surface_vector": surface_vec,
                "surface_distance_px": _distance(surface_bg, surface_sim),
                "background_delta": bg_delta,
                "background_delta_px": bg_delta_px,
                "simulated_delta": sim_delta,
                "simulated_delta_px": sim_delta_px,
                "vector_delta": vector_delta,
                "vector_delta_px": vector_delta_px,
                "background_frame_match": bg_frame_match,
                "simulated_frame_match": sim_frame_match,
            }
        )

    metrics = _surface_metrics(pair_rows)
    bg_max = metrics["background_max_error_px"]
    sim_max = metrics["simulated_max_error_px"]
    vec_max = metrics["vector_max_error_px"]
    endpoint_state = "both_endpoints_mismatch"
    if background_mismatch_count and not simulated_mismatch_count:
        endpoint_state = "background_endpoint_mismatch"
    elif simulated_mismatch_count and not background_mismatch_count:
        endpoint_state = "simulated_endpoint_mismatch"
    elif not background_mismatch_count and not simulated_mismatch_count and vector_mismatch_count:
        endpoint_state = "vector_only_mismatch"
    elif frame_mismatch_count:
        endpoint_state = "frame_mismatch"
    elif not background_mismatch_count and not simulated_mismatch_count:
        endpoint_state = "points_match"
    pass_visual_parity = bool(
        ordered_pairs_match
        and bg_max is not None
        and sim_max is not None
        and vec_max is not None
        and float(bg_max) <= tolerance
        and float(sim_max) <= tolerance
        and float(vec_max) <= tolerance
        and frame_mismatch_count == 0
    )
    result = {
        "surface_name": surface_name,
        "available": True,
        "pair_count": int(len(surface_rows)),
        "ordered_pairs_match": ordered_pairs_match,
        "unordered_pairs_match": unordered_pairs_match,
        "endpoint_diagnosis": endpoint_state,
        "background_endpoint_mismatch_count": int(background_mismatch_count),
        "simulated_endpoint_mismatch_count": int(simulated_mismatch_count),
        "both_endpoints_mismatch_count": int(
            sum(
                1
                for row in pair_rows
                if (
                    row.get("background_delta_px") is None
                    or float(row["background_delta_px"]) > tolerance
                )
                and (
                    row.get("simulated_delta_px") is None
                    or float(row["simulated_delta_px"]) > tolerance
                )
            )
        ),
        "vector_only_mismatch_count": int(
            0 if background_mismatch_count or simulated_mismatch_count else vector_mismatch_count
        ),
        "frame_mismatch_count": int(frame_mismatch_count),
        "passes_visual_parity": pass_visual_parity,
        **metrics,
    }
    return result, pair_rows


def _missing_surface_result(
    surface_name: str,
    *,
    visual_pair_count: int,
) -> dict[str, object]:
    return {
        "surface_name": surface_name,
        "available": False,
        "pair_count": 0,
        "ordered_pairs_match": False,
        "unordered_pairs_match": False,
        "endpoint_diagnosis": "missing_required_surface",
        "background_endpoint_mismatch_count": int(visual_pair_count),
        "simulated_endpoint_mismatch_count": int(visual_pair_count),
        "both_endpoints_mismatch_count": int(visual_pair_count),
        "vector_only_mismatch_count": 0,
        "frame_mismatch_count": 0,
        "passes_visual_parity": False,
        "background_rms_error_px": None,
        "background_max_error_px": None,
        "simulated_rms_error_px": None,
        "simulated_max_error_px": None,
        "vector_rms_error_px": None,
        "vector_max_error_px": None,
        "all_points_rms_error_px": None,
        "all_points_max_error_px": None,
    }


def _extent_limits(
    metadata: Mapping[str, object] | None,
) -> tuple[float, float, float, float] | None:
    metadata = metadata or {}
    extent = metadata.get("image_extent")
    if isinstance(extent, Sequence) and not isinstance(extent, (str, bytes)) and len(extent) >= 4:
        values = [_finite_float(value) for value in extent[:4]]
        if all(value is not None for value in values):
            return float(values[0]), float(values[1]), float(values[2]), float(values[3])
    shape = metadata.get("image_shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and len(shape) >= 2:
        height = _finite_float(shape[0])
        width = _finite_float(shape[1])
        if height is not None and width is not None:
            return 0.0, float(width), 0.0, float(height)
    return None


def _fixed_transform(
    name: str,
    points: np.ndarray,
    metadata: Mapping[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(2, dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    limits = _extent_limits(metadata)
    x_mid = 0.0
    y_mid = 0.0
    if limits is not None:
        x_mid = float(limits[0] + limits[1])
        y_mid = float(limits[2] + limits[3])
    elif points.size:
        x_mid = float(np.min(points[:, 0]) + np.max(points[:, 0]))
        y_mid = float(np.min(points[:, 1]) + np.max(points[:, 1]))

    if name in {"swap_axes", "row_col_swap"}:
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    elif name == "flip_x":
        matrix = np.array([[-1.0, 0.0], [0.0, 1.0]])
        translation = np.array([x_mid, 0.0])
    elif name in {"flip_y", "top_left_vs_bottom_left_origin"}:
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
        translation = np.array([0.0, y_mid])
    elif name == "swap_plus_flip_x":
        matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
        translation = np.array([x_mid, 0.0])
    elif name == "swap_plus_flip_y":
        matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        translation = np.array([0.0, y_mid])
    elif name == "flip_xy_180":
        matrix = np.array([[-1.0, 0.0], [0.0, -1.0]])
        translation = np.array([x_mid, y_mid])
    return matrix, translation


def _fit_translation(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(2, dtype=np.float64)
    translation = np.mean(dst - src, axis=0) if src.size else np.zeros(2, dtype=np.float64)
    return matrix, translation


def _fit_axis_scale_translation(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(2, dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        a = np.column_stack([src[:, axis], np.ones(src.shape[0])])
        try:
            scale, offset = np.linalg.lstsq(a, dst[:, axis], rcond=None)[0]
        except Exception:
            scale, offset = 1.0, 0.0
        matrix[axis, axis] = float(scale)
        translation[axis] = float(offset)
    return matrix, translation


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.column_stack([src, np.ones(src.shape[0])])
    try:
        coeff, *_ = np.linalg.lstsq(a, dst, rcond=None)
        matrix = coeff[:2, :].T
        translation = coeff[2, :]
    except Exception:
        matrix = np.eye(2, dtype=np.float64)
        translation = np.zeros(2, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64), np.asarray(translation, dtype=np.float64)


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if src.shape[0] < 2:
        return _fit_translation(src, dst)
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    variance = float(np.sum(src_centered * src_centered))
    if variance <= 1.0e-12:
        return _fit_translation(src, dst)
    try:
        u, _, vh = np.linalg.svd(src_centered.T @ dst_centered)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            vh[-1, :] *= -1
            rotation = u @ vh
        scale = float(np.trace((src_centered @ rotation).T @ dst_centered) / variance)
        matrix = scale * rotation
        translation = dst_mean - src_mean @ matrix.T
    except Exception:
        return _fit_affine(src, dst)
    return np.asarray(matrix, dtype=np.float64), np.asarray(translation, dtype=np.float64)


def _apply_transform(points: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ matrix.T + translation


def _score_transform(
    visual_pairs: Sequence[Mapping[str, object]],
    surface_rows: Sequence[Mapping[str, object]],
    matrix: np.ndarray,
    translation: np.ndarray,
) -> dict[str, object]:
    transformed_rows: list[dict[str, object]] = []
    for idx in range(min(len(visual_pairs), len(surface_rows))):
        visual = visual_pairs[idx]
        surface = surface_rows[idx]
        bg = _point(surface.get("background_point"))
        sim = _point(surface.get("simulated_point"))
        bg_t = (
            _apply_transform(np.asarray([bg], dtype=np.float64), matrix, translation)[0].tolist()
            if bg is not None
            else None
        )
        sim_t = (
            _apply_transform(np.asarray([sim], dtype=np.float64), matrix, translation)[0].tolist()
            if sim is not None
            else None
        )
        visual_bg = _point(visual.get("visual_background_point"))
        visual_sim = _point(visual.get("visual_simulated_point"))
        visual_vec = _vector(visual_bg, visual_sim)
        surface_vec = _vector(bg, sim)
        vector_t = (
            (np.asarray(surface_vec, dtype=np.float64) @ matrix.T).tolist()
            if surface_vec is not None
            else None
        )
        transformed_rows.append(
            {
                "background_delta_px": _norm(_delta(bg_t, visual_bg)),
                "simulated_delta_px": _norm(_delta(sim_t, visual_sim)),
                "vector_delta_px": _norm(_delta(vector_t, visual_vec)),
            }
        )
    return _surface_metrics(transformed_rows)


def _classification_from_metrics(metrics: Mapping[str, object], tolerance: float) -> str:
    max_values = [
        metrics.get("background_max_error_px"),
        metrics.get("simulated_max_error_px"),
        metrics.get("vector_max_error_px"),
        metrics.get("all_points_max_error_px"),
    ]
    finite = [float(value) for value in max_values if value is not None and np.isfinite(value)]
    if not finite:
        return "rejected"
    max_error = max(finite)
    if max_error <= tolerance:
        return "exact_visual_match"
    if max_error <= 0.1:
        return "subpixel_visual_match"
    if max_error <= 5.0:
        return "approximate_only"
    return "rejected"


def score_coordinate_transforms(
    visual_pairs: Sequence[Mapping[str, object]],
    surface_rows: Sequence[Mapping[str, object]],
    *,
    plotting_metadata: Mapping[str, object] | None = None,
    tolerance: float = STORED_POINT_ABS_TOLERANCE_PX,
) -> dict[str, object]:
    all_src: list[list[float]] = []
    all_dst: list[list[float]] = []
    for visual, surface in zip(visual_pairs, surface_rows, strict=False):
        for surface_key, visual_key in (
            ("background_point", "visual_background_point"),
            ("simulated_point", "visual_simulated_point"),
        ):
            src = _point(surface.get(surface_key))
            dst = _point(visual.get(visual_key))
            if src is not None and dst is not None:
                all_src.append(src)
                all_dst.append(dst)
    src_arr = np.asarray(all_src, dtype=np.float64).reshape(-1, 2)
    dst_arr = np.asarray(all_dst, dtype=np.float64).reshape(-1, 2)
    if src_arr.shape[0] == 0:
        return {"best_transform_name": None, "transform_name": None, "candidates": []}

    fixed_names = [
        "identity",
        "swap_axes",
        "flip_x",
        "flip_y",
        "swap_plus_flip_x",
        "swap_plus_flip_y",
        "flip_xy_180",
        "row_col_swap",
        "top_left_vs_bottom_left_origin",
    ]
    fit_names = [
        "constant_translation",
        "per_axis_scale_translation",
        "similarity_transform",
        "full_affine",
    ]
    candidates: list[dict[str, object]] = []
    for name in fixed_names + fit_names:
        if name in fixed_names:
            matrix, translation = _fixed_transform(name, src_arr, plotting_metadata)
        elif name == "constant_translation":
            matrix, translation = _fit_translation(src_arr, dst_arr)
        elif name == "per_axis_scale_translation":
            matrix, translation = _fit_axis_scale_translation(src_arr, dst_arr)
        elif name == "similarity_transform":
            matrix, translation = _fit_similarity(src_arr, dst_arr)
        else:
            matrix, translation = _fit_affine(src_arr, dst_arr)
        metrics = _score_transform(visual_pairs, surface_rows, matrix, translation)
        transform_classification = _classification_from_metrics(metrics, tolerance)
        candidates.append(
            {
                "transform_name": name,
                "matrix": matrix.tolist(),
                "translation": translation.tolist(),
                "classification": transform_classification,
                **metrics,
            }
        )

    def _rank(candidate: Mapping[str, object]) -> tuple[float, int]:
        max_error = candidate.get("all_points_max_error_px")
        if max_error is None or not np.isfinite(float(max_error)):
            return float("inf"), 999
        return float(max_error), (fixed_names + fit_names).index(str(candidate["transform_name"]))

    candidates.sort(key=_rank)
    best = dict(candidates[0]) if candidates else {}
    best["best_transform_name"] = best.get("transform_name")
    simple_exact = [
        candidate
        for candidate in candidates
        if candidate.get("transform_name") not in {"similarity_transform", "full_affine"}
        and candidate.get("classification") == "exact_visual_match"
    ]
    arbitrary_affine = bool(
        not simple_exact
        and best.get("transform_name") in {"similarity_transform", "full_affine"}
        and best.get("classification") == "exact_visual_match"
    )
    best["arbitrary_affine_detected"] = arbitrary_affine
    best["candidates"] = candidates
    return best


def _recommended_fix_location(surface_name: str | None) -> str | None:
    return {
        "provider_pairs": "provider_pair_construction",
        "manual_point_pairs": "build_geometry_manual_fit_dataset",
        "initial_pairs_display": "initial_pairs_display_construction",
        "measured_for_fit": "measured_for_fit_construction",
        'spec["measured_peaks"]': "spec_measured_peaks_construction",
        "optimizer_request.measured_peaks": "GeometryFitSolverRequest_construction",
    }.get(str(surface_name) if surface_name is not None else "")


def _top_level_classification(
    *,
    visual_truth_available: bool,
    first_surface: str | None,
    surface_results: Mapping[str, Mapping[str, object]],
    best_transform: Mapping[str, object] | None,
) -> str:
    if not visual_truth_available:
        return "diagnostic_incomplete_missing_visual_truth"
    if first_surface is None:
        return "visual_backend_parity_ok"
    first_result = surface_results.get(first_surface, {})
    if int(first_result.get("frame_mismatch_count", 0) or 0) > 0:
        return "frame_mismatch_detected"
    if not bool(first_result.get("ordered_pairs_match", True)) and bool(
        first_result.get("unordered_pairs_match", False)
    ):
        return "pair_order_mismatch"
    best_name = str(
        (best_transform or {}).get("best_transform_name")
        or (best_transform or {}).get("transform_name")
        or ""
    )
    best_class = str((best_transform or {}).get("classification") or "")
    if bool((best_transform or {}).get("arbitrary_affine_detected", False)):
        return "arbitrary_affine_detected"
    if best_class in {"exact_visual_match", "subpixel_visual_match"}:
        if best_name in {"swap_axes", "row_col_swap"}:
            return "axis_swap_detected"
        if best_name in {"flip_y", "top_left_vs_bottom_left_origin"}:
            return "origin_flip_detected"
        if best_name == "constant_translation":
            return "constant_translation_detected"
    if first_surface == "provider_pairs":
        return "provider_mismatch_visual"
    if first_surface == "optimizer_request.measured_peaks":
        return "optimizer_request_mismatch_visual"
    return "dataset_mismatch_visual"


def build_coordinate_parity_diagnosis(
    visual_capture: Mapping[str, object],
    backend_surfaces: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    tolerance: float = STORED_POINT_ABS_TOLERANCE_PX,
    optimizer_called: bool = False,
    least_squares_called: bool = False,
    optimizer_entrypoints_called: Sequence[object] | None = None,
) -> dict[str, object]:
    visual_pairs = [
        dict(row) for row in visual_capture.get("pairs", ()) or () if isinstance(row, Mapping)
    ]
    visual_pairs.sort(key=lambda row: int(row.get("pair_index", 0) or 0))
    visual_truth_available = bool(visual_capture.get("visual_truth_available", False))
    plotting_metadata = (
        dict(visual_capture.get("plotting_metadata", {}) or {})
        if isinstance(visual_capture.get("plotting_metadata"), Mapping)
        else {}
    )
    compared_surfaces = [
        name
        for name in SURFACE_ORDER
        if name in REQUIRED_SURFACES
        or (name == "optimizer_request.measured_peaks" and name in backend_surfaces)
    ]
    optimizer_request_compared = bool("optimizer_request.measured_peaks" in compared_surfaces)
    surface_results: dict[str, dict[str, object]] = {}
    pair_rows_by_surface: dict[str, list[dict[str, object]]] = {}
    best_transform_by_surface: dict[str, dict[str, object]] = {}
    missing_required_surfaces: list[str] = []
    first_mismatch = None

    for surface_name in compared_surfaces:
        surface_rows = [dict(row) for row in backend_surfaces.get(surface_name, [])]
        surface_available = bool(surface_name in backend_surfaces and surface_rows)
        if surface_name in REQUIRED_SURFACES and not surface_available:
            result = _missing_surface_result(
                surface_name,
                visual_pair_count=int(len(visual_pairs)),
            )
            surface_results[surface_name] = result
            pair_rows_by_surface[surface_name] = []
            best_transform_by_surface[surface_name] = {
                "best_transform_name": None,
                "transform_name": None,
                "classification": "not_scored_missing_required_surface",
                "candidates": [],
            }
            missing_required_surfaces.append(surface_name)
            if first_mismatch is None:
                first_mismatch = surface_name
            continue

        result, pair_rows = _compare_surface(
            visual_pairs,
            surface_name,
            surface_rows,
            tolerance=float(tolerance),
        )
        surface_results[surface_name] = result
        pair_rows_by_surface[surface_name] = pair_rows
        if not bool(result.get("passes_visual_parity", False)) and first_mismatch is None:
            first_mismatch = surface_name
        if not (
            surface_name == first_mismatch
            and not bool(result.get("ordered_pairs_match", True))
            and bool(result.get("unordered_pairs_match", False))
        ):
            best_transform_by_surface[surface_name] = score_coordinate_transforms(
                visual_pairs,
                surface_rows,
                plotting_metadata=plotting_metadata,
                tolerance=float(tolerance),
            )
        else:
            best_transform_by_surface[surface_name] = {
                "best_transform_name": None,
                "transform_name": None,
                "classification": "not_scored_pair_order_mismatch",
                "candidates": [],
            }

    required_ok = not missing_required_surfaces and all(
        bool(surface_results.get(surface, {}).get("passes_visual_parity", False))
        for surface in REQUIRED_SURFACES
    )
    if optimizer_request_compared:
        required_ok = required_ok and bool(
            surface_results.get("optimizer_request.measured_peaks", {}).get(
                "passes_visual_parity",
                False,
            )
        )
    ok = bool(visual_truth_available and required_ok)
    optimizer_request_result = surface_results.get(
        "optimizer_request.measured_peaks",
        {},
    )
    optimizer_request_pair_count = (
        int(optimizer_request_result.get("pair_count", 0) or 0) if optimizer_request_compared else 0
    )
    optimizer_request_visual_parity_ok = (
        bool(optimizer_request_result.get("passes_visual_parity", False))
        if optimizer_request_compared
        else False
    )
    optimizer_entrypoints = [str(name) for name in optimizer_entrypoints_called or ()]
    first_transform = (
        best_transform_by_surface.get(first_mismatch, {}) if first_mismatch is not None else {}
    )
    classification = _top_level_classification(
        visual_truth_available=visual_truth_available,
        first_surface=first_mismatch,
        surface_results=surface_results,
        best_transform=first_transform,
    )

    pairs: list[dict[str, object]] = []
    for visual in visual_pairs:
        pair_index = int(visual.get("pair_index", len(pairs)) or len(pairs))
        pair_entry: dict[str, object] = {
            "pair_index": pair_index,
            "pair_id": visual.get("pair_id"),
            "hkl": _jsonable(visual.get("hkl")),
            "source_branch_index": visual.get("source_branch_index"),
            "q_group_key": _jsonable(visual.get("q_group_key")),
            "branch_group_key": _jsonable(visual.get("branch_group_key")),
            "visual_background_point": _point(visual.get("visual_background_point")),
            "visual_background_data_point": _point(
                visual.get("visual_background_data_point") or visual.get("visual_background_point")
            ),
            "visual_background_canvas_point": _point(visual.get("visual_background_canvas_point")),
            "visual_background_frame": visual.get("visual_background_frame"),
            "visual_simulated_point": _point(visual.get("visual_simulated_point")),
            "visual_simulated_data_point": _point(
                visual.get("visual_simulated_data_point") or visual.get("visual_simulated_point")
            ),
            "visual_simulated_canvas_point": _point(visual.get("visual_simulated_canvas_point")),
            "visual_simulated_frame": visual.get("visual_simulated_frame"),
            "visual_background_artist_source": visual.get("visual_background_artist_source"),
            "visual_simulated_artist_source": visual.get("visual_simulated_artist_source"),
            "surfaces": {},
        }
        for surface_name in compared_surfaces:
            for row in pair_rows_by_surface.get(surface_name, []):
                row_pair_index = row.get("pair_index", -1)
                if row_pair_index is None:
                    row_pair_index = -1
                if int(row_pair_index) == pair_index:
                    pair_entry["surfaces"][surface_name] = row
                    break
        pairs.append(pair_entry)

    report = {
        "ok": ok,
        "classification": classification,
        "visual_truth_available": visual_truth_available,
        "visual_capture_path": visual_capture.get("visual_capture_path"),
        "visual_capture_path_confirmed": bool(
            visual_capture.get("visual_capture_path_confirmed", False)
        ),
        "visual_capture_function": visual_capture.get("visual_capture_function"),
        "overlay_source": visual_capture.get("overlay_source"),
        "pair_count": int(len(visual_pairs)),
        "stored_point_abs_tolerance_px": float(tolerance),
        "surfaces_compared": compared_surfaces,
        "optimizer_request_compared": optimizer_request_compared,
        "optimizer_request_pair_count": optimizer_request_pair_count,
        "optimizer_request_visual_parity_ok": optimizer_request_visual_parity_ok,
        "missing_required_surfaces": missing_required_surfaces,
        "surface_results": surface_results,
        "first_mismatching_surface": first_mismatch,
        "best_transform_by_surface": best_transform_by_surface,
        "recommended_fix_location": _recommended_fix_location(first_mismatch),
        "optimizer_called": bool(optimizer_called or optimizer_entrypoints),
        "least_squares_called": bool(least_squares_called),
        "optimizer_entrypoints_called": optimizer_entrypoints,
        "plotting_metadata": plotting_metadata,
        "pairs": pairs,
    }
    return _jsonable(report)  # type: ignore[return-value]


def _csv_row(pair_row: Mapping[str, object]) -> dict[str, object]:
    visual_bg = _point(pair_row.get("visual_background_point"))
    visual_sim = _point(pair_row.get("visual_simulated_point"))
    surface_bg = _point(pair_row.get("surface_background_point"))
    surface_sim = _point(pair_row.get("surface_simulated_point"))
    visual_vec = _vector(visual_bg, visual_sim)
    surface_vec = _vector(surface_bg, surface_sim)
    bg_delta = _delta(surface_bg, visual_bg)
    sim_delta = _delta(surface_sim, visual_sim)
    vector_delta = _delta(surface_vec, visual_vec)
    return {
        "pair_index": pair_row.get("pair_index"),
        "pair_id": pair_row.get("pair_id"),
        "hkl": json.dumps(_jsonable(pair_row.get("hkl")), sort_keys=True),
        "source_branch_index": pair_row.get("source_branch_index"),
        "q_group_key": json.dumps(_jsonable(pair_row.get("q_group_key")), sort_keys=True),
        "visual_background_x": visual_bg[0] if visual_bg is not None else None,
        "visual_background_y": visual_bg[1] if visual_bg is not None else None,
        "visual_simulated_x": visual_sim[0] if visual_sim is not None else None,
        "visual_simulated_y": visual_sim[1] if visual_sim is not None else None,
        "visual_dx": visual_vec[0] if visual_vec is not None else None,
        "visual_dy": visual_vec[1] if visual_vec is not None else None,
        "visual_distance_px": _distance(visual_bg, visual_sim),
        "surface_name": pair_row.get("surface_name"),
        "surface_background_x": surface_bg[0] if surface_bg is not None else None,
        "surface_background_y": surface_bg[1] if surface_bg is not None else None,
        "surface_simulated_x": surface_sim[0] if surface_sim is not None else None,
        "surface_simulated_y": surface_sim[1] if surface_sim is not None else None,
        "surface_dx": surface_vec[0] if surface_vec is not None else None,
        "surface_dy": surface_vec[1] if surface_vec is not None else None,
        "surface_distance_px": _distance(surface_bg, surface_sim),
        "background_delta_x": bg_delta[0] if bg_delta is not None else None,
        "background_delta_y": bg_delta[1] if bg_delta is not None else None,
        "background_delta_px": _norm(bg_delta),
        "simulated_delta_x": sim_delta[0] if sim_delta is not None else None,
        "simulated_delta_y": sim_delta[1] if sim_delta is not None else None,
        "simulated_delta_px": _norm(sim_delta),
        "vector_delta_x": vector_delta[0] if vector_delta is not None else None,
        "vector_delta_y": vector_delta[1] if vector_delta is not None else None,
        "vector_delta_px": _norm(vector_delta),
        "background_frame_match": pair_row.get("background_frame_match"),
        "simulated_frame_match": pair_row.get("simulated_frame_match"),
    }


def write_coordinate_pairs_csv(report: Mapping[str, object], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for pair in sorted(
        [dict(row) for row in report.get("pairs", []) if isinstance(row, Mapping)],
        key=lambda row: int(row.get("pair_index", 0) or 0),
    ):
        surfaces = pair.get("surfaces") if isinstance(pair.get("surfaces"), Mapping) else {}
        for surface_name in SURFACE_ORDER:
            surface_row = surfaces.get(surface_name) if isinstance(surfaces, Mapping) else None
            if not isinstance(surface_row, Mapping):
                continue
            rows.append(_csv_row(surface_row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def write_coordinate_diagnosis_json(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _plot_axes_style(ax: Any, metadata: Mapping[str, object]) -> None:
    try:
        if bool(metadata.get("y_axis_inverted", False)) and not ax.yaxis_inverted():
            ax.invert_yaxis()
    except Exception:
        pass
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass


def write_coordinate_plots(report: Mapping[str, object], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = [dict(row) for row in report.get("pairs", []) if isinstance(row, Mapping)]
    metadata = (
        report.get("plotting_metadata")
        if isinstance(report.get("plotting_metadata"), Mapping)
        else {}
    )
    colors = {
        "provider_pairs": "#d62728",
        "manual_point_pairs": "#9467bd",
        "initial_pairs_display": "#8c564b",
        "measured_for_fit": "#2ca02c",
        'spec["measured_peaks"]': "#ff7f0e",
        "optimizer_request.measured_peaks": "#17becf",
    }

    fig, ax = plt.subplots(figsize=(8, 8))
    for pair in pairs:
        bg = _point(pair.get("visual_background_point"))
        sim = _point(pair.get("visual_simulated_point"))
        if bg is not None:
            ax.plot([bg[0]], [bg[1]], "^", color="black", markersize=8, label="visual bg")
        if sim is not None:
            ax.plot([sim[0]], [sim[1]], "s", color="black", markersize=8, label="visual sim")
        surfaces = pair.get("surfaces") if isinstance(pair.get("surfaces"), Mapping) else {}
        for surface_name in SURFACE_ORDER:
            surface = surfaces.get(surface_name) if isinstance(surfaces, Mapping) else None
            if not isinstance(surface, Mapping):
                continue
            color = colors.get(surface_name, "#666666")
            s_bg = _point(surface.get("surface_background_point"))
            s_sim = _point(surface.get("surface_simulated_point"))
            if bg is not None and s_bg is not None:
                ax.plot([s_bg[0]], [s_bg[1]], "^", color=color, alpha=0.65)
                ax.plot([bg[0], s_bg[0]], [bg[1], s_bg[1]], "-", color=color, alpha=0.25)
            if sim is not None and s_sim is not None:
                ax.plot([s_sim[0]], [s_sim[1]], "s", color=color, alpha=0.65)
                ax.plot([sim[0], s_sim[0]], [sim[1], s_sim[1]], "-", color=color, alpha=0.25)
    ax.set_title("Visual truth vs backend raw points")
    _plot_axes_style(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_dir / "coordinate_transform_overlay.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    first_surface = report.get("first_mismatching_surface")
    best_by_surface = report.get("best_transform_by_surface")
    best = (
        best_by_surface.get(first_surface, {})
        if isinstance(best_by_surface, Mapping) and first_surface in best_by_surface
        else {}
    )
    matrix = (
        np.asarray(best.get("matrix", np.eye(2)), dtype=np.float64).reshape(2, 2)
        if isinstance(best, Mapping) and best.get("matrix") is not None
        else np.eye(2)
    )
    for pair in pairs:
        bg = _point(pair.get("visual_background_point"))
        sim = _point(pair.get("visual_simulated_point"))
        if bg is not None and sim is not None:
            ax.arrow(
                bg[0],
                bg[1],
                sim[0] - bg[0],
                sim[1] - bg[1],
                color="black",
                width=0.15,
                length_includes_head=True,
                alpha=0.85,
            )
        surfaces = pair.get("surfaces") if isinstance(pair.get("surfaces"), Mapping) else {}
        for surface_name in SURFACE_ORDER:
            surface = surfaces.get(surface_name) if isinstance(surfaces, Mapping) else None
            if not isinstance(surface, Mapping):
                continue
            s_bg = _point(surface.get("surface_background_point"))
            s_sim = _point(surface.get("surface_simulated_point"))
            if s_bg is None or s_sim is None:
                continue
            color = colors.get(surface_name, "#666666")
            vec = np.asarray([s_sim[0] - s_bg[0], s_sim[1] - s_bg[1]], dtype=np.float64)
            vec_t = vec @ matrix.T
            ax.arrow(
                s_bg[0],
                s_bg[1],
                vec[0],
                vec[1],
                color=color,
                width=0.08,
                length_includes_head=True,
                alpha=0.35,
            )
            ax.arrow(
                s_bg[0],
                s_bg[1],
                vec_t[0],
                vec_t[1],
                color=color,
                width=0.04,
                linestyle="--",
                length_includes_head=True,
                alpha=0.65,
            )
    ax.set_title("Pair vectors: visual, backend, transformed backend")
    _plot_axes_style(ax, metadata)
    fig.tight_layout()
    fig.savefig(output_dir / "coordinate_transform_vectors.png", dpi=160)
    plt.close(fig)


def write_coordinate_diagnostic_outputs(report: Mapping[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_coordinate_diagnosis_json(report, output_dir / "coordinate_transform_diagnosis.json")
    write_coordinate_pairs_csv(report, output_dir / "coordinate_transform_pairs.csv")
    write_coordinate_plots(report, output_dir)


def saved_entries_for_background(
    saved_state: Mapping[str, object],
    *,
    background_index: int,
) -> list[dict[str, object]]:
    geometry = saved_state.get("geometry")
    if not isinstance(geometry, Mapping):
        return []
    manual_pairs = geometry.get("manual_pairs", [])
    for raw_group in manual_pairs if isinstance(manual_pairs, Sequence) else []:
        if not isinstance(raw_group, Mapping):
            continue
        try:
            group_background_index = int(raw_group.get("background_index", 0) or 0)
        except Exception:
            group_background_index = 0
        if group_background_index != int(background_index):
            continue
        entries = raw_group.get("entries", [])
        return [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    return []


def load_gui_state_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("state"), Mapping):
        return dict(payload["state"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"GUI state JSON is not an object: {path}")


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def capture_optimizer_request_rows_from_solver_request(
    *,
    state_path: Path,
    background_index: int,
) -> dict[str, object]:
    """Build request rows through the ladder request builder without solving."""

    optimizer_entrypoints_called: list[str] = []
    patched: list[tuple[object, str, object]] = []
    patched_keys: set[tuple[int, str]] = set()

    def _record_and_fail(name: str):
        def _fail(*_args, **_kwargs):
            optimizer_entrypoints_called.append(name)
            raise AssertionError(f"{name} must not run during coordinate diagnostic")

        return _fail

    def _patch(obj: object, attr: str, name: str) -> None:
        if not hasattr(obj, attr):
            return
        key = (id(obj), attr)
        if key in patched_keys:
            return
        patched_keys.add(key)
        patched.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, _record_and_fail(name))

    try:
        ladder = _load_new4_geometry_fit_ladder_module()
        _patch(ladder.opt, "least_squares", "least_squares")
        _patch(geometry_fit, "solve_geometry_fit_request", "solve_geometry_fit_request")
        _patch(
            ladder.gui_geometry_fit,
            "solve_geometry_fit_request",
            "solve_geometry_fit_request",
        )
        _patch(ladder, "run_objective_dry_run", "run_objective_dry_run")
        _patch(ladder, "_run_solver_rung_with_timeout", "_run_solver_rung_with_timeout")
        _patch(ladder, "_run_with_probe_least_squares", "_run_with_probe_least_squares")

        context = ladder._capture_solver_context(
            Path(state_path),
            int(background_index),
        )
        request = ladder.build_solver_request(
            context,
            ["center_x"],
            max_nfev=1,
        )
        rows = [
            _normalize_optimizer_request_row(row, idx)
            for idx, row in enumerate(request.measured_peaks or ())
            if isinstance(row, Mapping)
        ]
        missing_by_row = [
            {
                "row_index": int(idx),
                "pair_index": row.get("pair_index"),
                "missing_fields": list(row.get("missing_optimizer_request_fields", [])),
            }
            for idx, row in enumerate(rows)
            if row.get("missing_optimizer_request_fields")
        ]
        missing_fields = sorted(
            {
                str(field)
                for row in rows
                for field in row.get("missing_optimizer_request_fields", []) or []
            }
        )
        return {
            "rows": rows,
            "optimizer_entrypoints_called": optimizer_entrypoints_called,
            "optimizer_request_missing_fields": missing_fields,
            "optimizer_request_missing_fields_by_row": missing_by_row,
            "optimizer_request_capture_error": None,
        }
    except Exception as exc:
        if optimizer_entrypoints_called:
            raise
        return {
            "rows": [],
            "optimizer_entrypoints_called": optimizer_entrypoints_called,
            "optimizer_request_missing_fields": [],
            "optimizer_request_missing_fields_by_row": [],
            "optimizer_request_capture_error": str(exc),
        }
    finally:
        for obj, attr, original in reversed(patched):
            setattr(obj, attr, original)


def _optimizer_rung_candidate_lists(
    payload: Mapping[str, object],
) -> list[Sequence[object]]:
    candidates: list[Sequence[object]] = []
    for key in (
        "optimizer_request_measured_peaks",
        "solver_request_measured_peaks",
        "measured_peaks",
        "optimizer_request_rows",
    ):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates.append(value)
    solver_request = payload.get("solver_request")
    if isinstance(solver_request, Mapping):
        value = solver_request.get("measured_peaks")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates.append(value)
    request = payload.get("request")
    if isinstance(request, Mapping):
        value = request.get("measured_peaks")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidates.append(value)
    return candidates


def _optimizer_request_row_has_coordinates(row: Mapping[str, object]) -> bool:
    return bool(
        _point_from_keys(row, ("x", "y")) is not None
        or _point(row.get("background_point")) is not None
        or _point(row.get("measured_point")) is not None
        or _point_from_keys(row, ("measured_x", "measured_y")) is not None
        or _point_from_keys(row, ("background_x", "background_y")) is not None
    )


def _normalize_optimizer_rung_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    if _point_from_keys(normalized, ("x", "y")) is None:
        background = (
            _point(normalized.get("background_point"))
            or _point(normalized.get("measured_point"))
            or _point_from_keys(normalized, ("measured_x", "measured_y"))
            or _point_from_keys(normalized, ("background_x", "background_y"))
        )
        if background is not None:
            normalized["x"] = float(background[0])
            normalized["y"] = float(background[1])
    if _point(normalized.get("simulated_point")) is None:
        simulated = (
            _point(normalized.get("sim_display"))
            or _point_from_keys(normalized, ("simulated_x", "simulated_y"))
            or _point_from_keys(normalized, ("sim_x", "sim_y"))
        )
        if simulated is not None:
            normalized["simulated_point"] = simulated
    return normalized


def extract_optimizer_request_rows_from_rung_report(
    rung_report: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], str | None]:
    if not isinstance(rung_report, Mapping):
        return [], "rung_01_missing"
    for candidate_list in _optimizer_rung_candidate_lists(rung_report):
        rows = [
            _normalize_optimizer_rung_row(row)
            for row in candidate_list
            if isinstance(row, Mapping) and _optimizer_request_row_has_coordinates(row)
        ]
        if rows:
            rows.sort(key=lambda row: int(row.get("pair_index", len(rows)) or 0))
            return rows, None
    return [], "rung_01_lacks_measured_peak_coordinates"


def run_new4_visual_backend_coordinate_diagnostic(
    *,
    state_path: Path,
    provider_report_path: Path | None,
    background_index: int,
    output_dir: Path,
    include_optimizer_request: bool = False,
    rung_report_path: Path | None = None,
) -> dict[str, object]:
    saved_state = load_gui_state_payload(state_path)
    saved_entries = saved_entries_for_background(
        saved_state,
        background_index=int(background_index),
    )
    overlay_input_capture = capture_manual_geometry_overlay_input_from_render_path(
        saved_entries,
        background_index=int(background_index),
    )
    visual_capture = collect_geometry_visual_pair_positions(
        overlay_input_capture.get("pairs", []),
        max_display_markers=int(
            overlay_input_capture.get(
                "max_display_markers",
                max(1, len(overlay_input_capture.get("pairs", []))),
            )
        ),
    )
    visual_capture.update(
        {
            "visual_capture_path": overlay_input_capture.get("visual_capture_path"),
            "visual_capture_path_confirmed": bool(
                overlay_input_capture.get("visual_capture_path_confirmed", False)
            ),
            "visual_capture_function": overlay_input_capture.get("visual_capture_function"),
            "overlay_draw_function": overlay_input_capture.get("overlay_draw_function"),
            "overlay_source": overlay_input_capture.get("overlay_source"),
            "overlay_input_pair_count": overlay_input_capture.get("overlay_input_pair_count"),
        }
    )
    if not bool(visual_capture.get("visual_capture_path_confirmed", False)):
        visual_capture["visual_truth_available"] = False
    dataset = geometry_fit.build_geometry_fit_saved_state_point_provider_dataset(
        int(background_index),
        saved_entries,
    )
    provider_report = None
    if provider_report_path is not None and provider_report_path.exists():
        provider_report = json.loads(provider_report_path.read_text(encoding="utf-8"))
    optimizer_rows: list[dict[str, object]] | None = None
    optimizer_request_source = None
    optimizer_request_unavailable_reason = None
    optimizer_entrypoints_called: list[str] = []
    optimizer_request_missing_fields: list[str] = []
    optimizer_request_missing_fields_by_row: list[dict[str, object]] = []
    optimizer_request_capture_error = None
    optimizer_request_capture_failed = False
    state_hash_before = _file_sha256(state_path) if include_optimizer_request else None
    state_hash_after = None
    state_hash_unchanged = None
    if include_optimizer_request:
        capture = capture_optimizer_request_rows_from_solver_request(
            state_path=state_path,
            background_index=int(background_index),
        )
        optimizer_rows = [
            dict(row) for row in capture.get("rows", []) or [] if isinstance(row, Mapping)
        ]
        optimizer_entrypoints_called = [
            str(name) for name in capture.get("optimizer_entrypoints_called", []) or []
        ]
        optimizer_request_missing_fields = [
            str(field) for field in capture.get("optimizer_request_missing_fields", []) or []
        ]
        optimizer_request_missing_fields_by_row = [
            dict(row)
            for row in capture.get("optimizer_request_missing_fields_by_row", []) or []
            if isinstance(row, Mapping)
        ]
        optimizer_request_capture_error = capture.get("optimizer_request_capture_error")
        optimizer_request_source = "GeometryFitSolverRequest.measured_peaks"
        if optimizer_request_capture_error:
            optimizer_request_unavailable_reason = "solver_request_capture_failed"
            optimizer_request_capture_failed = True
            optimizer_rows = None
        elif not optimizer_rows:
            optimizer_request_unavailable_reason = "solver_request_no_measured_peak_rows"
        state_hash_after = _file_sha256(state_path)
        state_hash_unchanged = bool(
            state_hash_before is not None
            and state_hash_after is not None
            and state_hash_before == state_hash_after
        )
    else:
        optimizer_request_unavailable_reason = "not_requested"
    surfaces = build_backend_surfaces_from_dataset(
        dataset,
        provider_report=provider_report if isinstance(provider_report, Mapping) else None,
        optimizer_request_rows=optimizer_rows,
    )
    report = build_coordinate_parity_diagnosis(
        visual_capture,
        surfaces,
        tolerance=STORED_POINT_ABS_TOLERANCE_PX,
        optimizer_called=bool(optimizer_entrypoints_called),
        least_squares_called=False,
        optimizer_entrypoints_called=optimizer_entrypoints_called,
    )
    if optimizer_request_capture_failed:
        report.update(
            {
                "ok": False,
                "classification": ("diagnostic_incomplete_optimizer_request_unavailable"),
                "optimizer_request_compared": False,
                "optimizer_request_pair_count": 0,
                "optimizer_request_visual_parity_ok": False,
                "first_mismatching_surface": None,
                "recommended_fix_location": "optimizer_request_capture",
            }
        )
    elif include_optimizer_request and state_hash_unchanged is not True:
        report["ok"] = False
        report["classification"] = "state_mutated_during_optimizer_request_capture"
    report["state_path"] = str(state_path)
    report["provider_report_path"] = str(provider_report_path) if provider_report_path else None
    report["rung_report_detected"] = bool(
        rung_report_path is not None and rung_report_path.exists()
    )
    report["rung_report_filename"] = rung_report_path.name if rung_report_path is not None else None
    report["optimizer_request_source"] = optimizer_request_source
    report["optimizer_request_unavailable_reason"] = optimizer_request_unavailable_reason
    report["optimizer_request_capture_error"] = optimizer_request_capture_error
    report["optimizer_request_missing_fields"] = optimizer_request_missing_fields
    report["optimizer_request_missing_fields_by_row"] = optimizer_request_missing_fields_by_row
    report["state_sha256_before"] = state_hash_before
    report["state_sha256_after"] = state_hash_after
    report["state_hash_unchanged"] = state_hash_unchanged
    write_coordinate_diagnostic_outputs(report, output_dir)
    return _jsonable(report)  # type: ignore[return-value]
