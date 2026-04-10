"""Tk-native primary 2D viewport for the main diffraction image surface."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Any

import time

import numpy as np

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - Pillow is optional at import time.
    Image = None
    ImageTk = None

from ra_sim.gui import fast_plot_viewer

try:
    import matplotlib.colors as _matplotlib_colors
except Exception:  # pragma: no cover - Matplotlib is always present in the GUI runtime.
    _matplotlib_colors = None


_PIL_RESAMPLING = getattr(Image, "Resampling", Image) if Image is not None else None
_PIL_RESAMPLE_NEAREST = getattr(_PIL_RESAMPLING, "NEAREST", 0)
_PIL_RESAMPLE_BILINEAR = getattr(_PIL_RESAMPLING, "BILINEAR", 2)
_PIL_TRANSPOSE_LEFT_RIGHT = getattr(Image, "FLIP_LEFT_RIGHT", 0) if Image is not None else 0
_PIL_TRANSPOSE_TOP_BOTTOM = getattr(Image, "FLIP_TOP_BOTTOM", 1) if Image is not None else 1
_EPSILON = 1.0e-9


def parse_primary_viewport_backend(raw_value: object) -> str:
    """Normalize the runtime backend selection for the primary 2D viewport."""

    normalized = str(raw_value or "matplotlib").strip().lower().replace("-", "_")
    if normalized in {"", "matplotlib", "mpl", "tkagg"}:
        return "matplotlib"
    if normalized in {"tk", "tk_canvas", "tkcanvas", "canvas"}:
        return "tk_canvas"
    return "matplotlib"


def primary_viewport_runtime_available() -> bool:
    """Return whether the Tk-canvas viewport can render raster layers."""

    return Image is not None and ImageTk is not None


def primary_viewport_unavailable_reason() -> str | None:
    """Return a short unavailability note for the Tk-canvas viewport."""

    if primary_viewport_runtime_available():
        return None
    return "Pillow ImageTk is unavailable"


@dataclass(frozen=True)
class ViewportViewState:
    width: int
    height: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass(frozen=True)
class ViewportEvent:
    name: str
    x: float | None
    y: float | None
    xdata: float | None
    ydata: float | None
    button: int | str | None
    dblclick: bool
    inaxes: object | None
    step: float = 0.0
    canvas: object | None = None


@dataclass(frozen=True)
class ViewportPeakCache:
    positions: tuple[tuple[float, float], ...] = ()
    visible_positions: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ViewportQGroupCache:
    visible_entries: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class _ViewportImageLayer:
    name: str
    visible: bool
    extent: tuple[float, float, float, float] | None
    interpolation: str
    source_rgba: np.ndarray | None
    origin: str = "upper"


@dataclass(frozen=True)
class _ViewportTextSpec:
    x: float
    y: float
    text: str
    fill_rgba: tuple[int, int, int, int]
    font_size: int
    zorder: float


@dataclass(frozen=True)
class ViewportScene:
    view_state: ViewportViewState
    view_mode: str
    background_layer: _ViewportImageLayer | None = None
    simulation_layer: _ViewportImageLayer | None = None
    overlay_layer: _ViewportImageLayer | None = None
    overlay_model: fast_plot_viewer.FastViewerOverlayModel = fast_plot_viewer.FastViewerOverlayModel()
    text_specs: tuple[_ViewportTextSpec, ...] = ()
    peak_cache: ViewportPeakCache = ViewportPeakCache()
    q_group_cache: ViewportQGroupCache = ViewportQGroupCache()
    raster_signature: object = None
    overlay_signature: object = None


@dataclass(frozen=True)
class PrimaryViewportBackend:
    name: str
    canvas_proxy: object
    activate: Callable[[], None]
    deactivate: Callable[[], None]
    sync_from_matplotlib: Callable[..., None]
    shutdown: Callable[[], None]


def _resolve_runtime_value(value_or_factory: object) -> object:
    if callable(value_or_factory):
        try:
            return value_or_factory()
        except Exception:
            return None
    return value_or_factory


def _safe_float(value: object) -> float | None:
    try:
        converted = float(value)
    except Exception:
        return None
    return converted if isfinite(converted) else None


def _safe_widget_dimension(widget: object, name: str, fallback: int) -> int:
    getter = getattr(widget, name, None)
    if not callable(getter):
        return int(max(1, fallback))
    try:
        value = int(getter())
    except Exception:
        return int(max(1, fallback))
    return int(max(1, value))


def _coerce_2d_array(array_like: object) -> np.ndarray | None:
    if array_like is None:
        return None
    arr = np.asarray(array_like)
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, 0.0)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return None
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr)


def _normalize_image_origin(raw_value: object) -> str:
    origin = str(raw_value or "upper").strip().lower()
    return origin if origin in {"upper", "lower"} else "upper"


def _normalize_view_mode(layer_versions: Mapping[str, object] | None) -> str:
    if not isinstance(layer_versions, Mapping):
        return "detector"
    simulation_version = layer_versions.get("simulation")
    if isinstance(simulation_version, Sequence) and simulation_version:
        first = str(simulation_version[0] or "").strip().lower()
        if first == "caked":
            return "caked"
    return "detector"


def _view_bounds(view_state: ViewportViewState) -> tuple[float, float, float, float]:
    x0, x1 = view_state.xlim
    y0, y1 = view_state.ylim
    return (
        float(min(x0, x1)),
        float(max(x0, x1)),
        float(min(y0, y1)),
        float(max(y0, y1)),
    )


def world_to_screen(
    view_state: ViewportViewState,
    world_x: float,
    world_y: float,
) -> tuple[float, float] | None:
    width = max(int(view_state.width), 1)
    height = max(int(view_state.height), 1)
    x0, x1 = view_state.xlim
    y0, y1 = view_state.ylim
    span_x = float(x1) - float(x0)
    span_y = float(y1) - float(y0)
    if abs(span_x) <= _EPSILON or abs(span_y) <= _EPSILON:
        return None
    sx = ((float(world_x) - float(x0)) / span_x) * float(width)
    sy = ((float(world_y) - float(y0)) / span_y) * float(height)
    if not (isfinite(sx) and isfinite(sy)):
        return None
    return (float(sx), float(sy))


def screen_to_world(
    view_state: ViewportViewState,
    screen_x: float,
    screen_y: float,
) -> tuple[float, float] | None:
    width = max(int(view_state.width), 1)
    height = max(int(view_state.height), 1)
    x0, x1 = view_state.xlim
    y0, y1 = view_state.ylim
    span_x = float(x1) - float(x0)
    span_y = float(y1) - float(y0)
    if abs(span_x) <= _EPSILON or abs(span_y) <= _EPSILON:
        return None
    world_x = float(x0) + (float(screen_x) / float(width)) * span_x
    world_y = float(y0) + (float(screen_y) / float(height)) * span_y
    if not (isfinite(world_x) and isfinite(world_y)):
        return None
    return (float(world_x), float(world_y))


def _point_within_view(view_state: ViewportViewState, x: float, y: float) -> bool:
    x_min, x_max, y_min, y_max = _view_bounds(view_state)
    return (
        x_min - _EPSILON <= float(x) <= x_max + _EPSILON
        and y_min - _EPSILON <= float(y) <= y_max + _EPSILON
    )


def _build_peak_cache(
    peak_positions: object,
    view_state: ViewportViewState,
) -> ViewportPeakCache:
    all_positions: list[tuple[float, float]] = []
    visible_positions: list[tuple[float, float]] = []
    for raw_position in peak_positions or ():
        if not isinstance(raw_position, Sequence) or len(raw_position) < 2:
            continue
        x_value = _safe_float(raw_position[0])
        y_value = _safe_float(raw_position[1])
        if x_value is None or y_value is None:
            continue
        point = (float(x_value), float(y_value))
        all_positions.append(point)
        if _point_within_view(view_state, *point):
            visible_positions.append(point)
    return ViewportPeakCache(
        positions=tuple(all_positions),
        visible_positions=tuple(visible_positions),
    )


def _extract_qgroup_display_point(
    candidate: Mapping[str, object],
    *,
    view_mode: str,
) -> tuple[float, float] | None:
    ordered_keys: tuple[tuple[str, str], ...]
    if str(view_mode) == "caked":
        ordered_keys = (
            ("two_theta_deg", "phi_deg"),
            ("display_col", "display_row"),
            ("sim_col_local", "sim_row_local"),
            ("sim_col", "sim_row"),
        )
    else:
        ordered_keys = (
            ("display_col", "display_row"),
            ("sim_col_local", "sim_row_local"),
            ("sim_col", "sim_row"),
            ("two_theta_deg", "phi_deg"),
        )
    for x_key, y_key in ordered_keys:
        x_value = _safe_float(candidate.get(x_key))
        y_value = _safe_float(candidate.get(y_key))
        if x_value is not None and y_value is not None:
            return (float(x_value), float(y_value))
    return None


def _build_q_group_cache(
    grouped_candidates: object,
    view_state: ViewportViewState,
    *,
    view_mode: str,
) -> ViewportQGroupCache:
    if not isinstance(grouped_candidates, Mapping):
        return ViewportQGroupCache()
    visible_entries: list[dict[str, object]] = []
    for group_key, entries in grouped_candidates.items():
        for raw_entry in entries or ():
            if not isinstance(raw_entry, Mapping):
                continue
            point = _extract_qgroup_display_point(raw_entry, view_mode=view_mode)
            if point is None or not _point_within_view(view_state, *point):
                continue
            record = dict(raw_entry)
            record.setdefault("q_group_key", group_key)
            record["_viewport_point"] = point
            visible_entries.append(record)
    return ViewportQGroupCache(visible_entries=tuple(visible_entries))


def _flatten_artists(values: object) -> Iterable[object]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return ()
    if isinstance(values, Iterable):
        flattened: list[object] = []
        for entry in values:
            flattened.extend(list(_flatten_artists(entry)))
        return tuple(flattened)
    return (values,)


def _to_rgba_uint8(color: object) -> tuple[int, int, int, int]:
    if _matplotlib_colors is None:
        return (255, 255, 255, 255)
    try:
        rgba = _matplotlib_colors.to_rgba(color)
    except Exception:
        return (255, 255, 255, 255)
    return tuple(
        int(round(max(0.0, min(1.0, float(channel))) * 255.0))
        for channel in rgba
    )


def _rgba_to_hex(rgba: tuple[int, int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(255, rgba[0]))),
        int(max(0, min(255, rgba[1]))),
        int(max(0, min(255, rgba[2]))),
    )


def _dash_pattern(linestyle: object) -> tuple[int, ...] | None:
    normalized = str(linestyle or "-").strip().lower()
    if normalized in {"-", "solid", "none", ""}:
        return None
    if normalized in {"--", "dashed"}:
        return (6, 4)
    if normalized in {":", "dotted"}:
        return (2, 3)
    if normalized in {"-.", "dashdot"}:
        return (6, 3, 2, 3)
    return None


def _interpolation_to_resample(interpolation: str) -> int:
    normalized = str(interpolation or "").strip().lower()
    if normalized in {"", "nearest", "none"}:
        return _PIL_RESAMPLE_NEAREST
    return _PIL_RESAMPLE_BILINEAR


def _extract_text_specs(artist_groups: object) -> tuple[_ViewportTextSpec, ...]:
    text_specs: list[_ViewportTextSpec] = []
    for artist in _flatten_artists(artist_groups):
        get_text = getattr(artist, "get_text", None)
        get_position = getattr(artist, "get_position", None)
        if not callable(get_text) or not callable(get_position):
            continue
        try:
            visible = bool(getattr(artist, "get_visible", lambda: True)())
        except Exception:
            visible = True
        if not visible:
            continue
        try:
            text_value = str(get_text() or "")
        except Exception:
            continue
        if not text_value:
            continue
        try:
            position = get_position()
        except Exception:
            continue
        if not isinstance(position, Sequence) or len(position) < 2:
            continue
        x_value = _safe_float(position[0])
        y_value = _safe_float(position[1])
        if x_value is None or y_value is None:
            continue
        try:
            font_size = int(round(float(getattr(artist, "get_fontsize", lambda: 10)())))
        except Exception:
            font_size = 10
        try:
            zorder = float(getattr(artist, "get_zorder", lambda: 10.0)())
        except Exception:
            zorder = 10.0
        fill_rgba = _to_rgba_uint8(getattr(artist, "get_color", lambda: "white")())
        text_specs.append(
            _ViewportTextSpec(
                x=float(x_value),
                y=float(y_value),
                text=text_value,
                fill_rgba=fill_rgba,
                font_size=max(8, font_size),
                zorder=float(zorder),
            )
        )
    text_specs.sort(key=lambda spec: spec.zorder)
    return tuple(text_specs)


def _layer_signature(layer: _ViewportImageLayer | None) -> object:
    if layer is None:
        return None
    source_rgba = layer.source_rgba
    source_signature = None
    if source_rgba is not None:
        source_signature = (
            id(source_rgba),
            tuple(int(value) for value in source_rgba.shape),
            str(source_rgba.dtype),
        )
    return (
        str(layer.name),
        bool(layer.visible),
        (
            None
            if layer.extent is None
            else tuple(float(value) for value in layer.extent)
        ),
        str(layer.interpolation),
        str(layer.origin),
        source_signature,
    )


def _scene_view_signature(
    view_state: ViewportViewState,
    *,
    view_mode: str,
) -> tuple[object, ...]:
    return (
        int(view_state.width),
        int(view_state.height),
        tuple(float(value) for value in view_state.xlim),
        tuple(float(value) for value in view_state.ylim),
        str(view_mode),
    )


def _scene_raster_signature(
    view_state: ViewportViewState,
    *,
    view_mode: str,
    background_layer: _ViewportImageLayer | None,
    simulation_layer: _ViewportImageLayer | None,
) -> tuple[object, ...]:
    return (
        _scene_view_signature(view_state, view_mode=view_mode),
        _layer_signature(background_layer),
        _layer_signature(simulation_layer),
    )


def _scene_overlay_signature(
    view_state: ViewportViewState,
    *,
    view_mode: str,
    overlay_layer: _ViewportImageLayer | None,
    overlay_model: fast_plot_viewer.FastViewerOverlayModel,
    text_specs: tuple[_ViewportTextSpec, ...],
) -> tuple[object, ...]:
    return (
        _scene_view_signature(view_state, view_mode=view_mode),
        _layer_signature(overlay_layer),
        overlay_model,
        tuple(text_specs),
    )


class _TkPrimaryViewport:
    def __init__(
        self,
        *,
        tk_module,
        parent,
        ax,
        marker_artist_factory: object = None,
        overlay_model_factory: object = None,
        overlay_artist_groups_factory: object = None,
        layer_versions_factory: object = None,
        peak_cache_factory: object = None,
        qgroup_cache_factory: object = None,
        initial_width: int = 800,
        initial_height: int = 800,
    ) -> None:
        self._tk = tk_module
        self._ax = ax
        self._marker_artist_factory = marker_artist_factory
        self._overlay_model_factory = overlay_model_factory
        self._overlay_artist_groups_factory = overlay_artist_groups_factory
        self._layer_versions_factory = layer_versions_factory
        self._peak_cache_factory = peak_cache_factory
        self._qgroup_cache_factory = qgroup_cache_factory
        self._widget = tk_module.Canvas(
            parent,
            background="white",
            highlightthickness=0,
            borderwidth=0,
            width=max(int(initial_width), 1),
            height=max(int(initial_height), 1),
        )
        self._photo_image = None
        self._photo_item = None
        self._overlay_photo_image = None
        self._overlay_photo_item = None
        self._last_scene: ViewportScene | None = None
        self._last_raster_signature = None
        self._last_overlay_signature = None
        self._layer_rgba_cache: dict[str, tuple[object, np.ndarray]] = {}
        self._widget.bind("<Configure>", self._on_configure, add="+")

    @property
    def widget(self):
        return self._widget

    @property
    def view_state(self) -> ViewportViewState | None:
        if self._last_scene is None:
            return None
        return self._last_scene.view_state

    def _on_configure(self, _event=None) -> None:
        scene = self._last_scene
        if scene is None:
            return
        view_state = self._current_view_state()
        refreshed_scene = ViewportScene(
            view_state=view_state,
            view_mode=scene.view_mode,
            background_layer=scene.background_layer,
            simulation_layer=scene.simulation_layer,
            overlay_layer=scene.overlay_layer,
            overlay_model=scene.overlay_model,
            text_specs=scene.text_specs,
            peak_cache=scene.peak_cache,
            q_group_cache=scene.q_group_cache,
            raster_signature=_scene_raster_signature(
                view_state,
                view_mode=scene.view_mode,
                background_layer=scene.background_layer,
                simulation_layer=scene.simulation_layer,
            ),
            overlay_signature=_scene_overlay_signature(
                view_state,
                view_mode=scene.view_mode,
                overlay_layer=scene.overlay_layer,
                overlay_model=scene.overlay_model,
                text_specs=scene.text_specs,
            ),
        )
        self.render_scene(refreshed_scene)

    def _current_view_state(self) -> ViewportViewState:
        width = _safe_widget_dimension(self._widget, "winfo_width", 1)
        height = _safe_widget_dimension(self._widget, "winfo_height", 1)
        xlim = tuple(float(value) for value in self._ax.get_xlim())
        ylim = tuple(float(value) for value in self._ax.get_ylim())
        return ViewportViewState(
            width=width,
            height=height,
            xlim=(float(xlim[0]), float(xlim[1])),
            ylim=(float(ylim[0]), float(ylim[1])),
        )

    def contains_screen_point(self, x_pixel: float, y_pixel: float) -> bool:
        view_state = self.view_state
        if view_state is None:
            return False
        return (
            0.0 <= float(x_pixel) <= float(view_state.width)
            and 0.0 <= float(y_pixel) <= float(view_state.height)
        )

    def screen_to_world(self, x_pixel: float, y_pixel: float) -> tuple[float, float] | None:
        view_state = self.view_state
        if view_state is None:
            return None
        return screen_to_world(view_state, float(x_pixel), float(y_pixel))

    def sync_from_matplotlib(
        self,
        *,
        image_artist,
        background_artist,
        overlay_artist,
        force_view_range: bool = False,
    ) -> None:
        del force_view_range
        layer_versions = _resolve_runtime_value(self._layer_versions_factory)
        view_state = self._current_view_state()
        view_mode = _normalize_view_mode(
            layer_versions if isinstance(layer_versions, Mapping) else None
        )
        overlay_model = _resolve_runtime_value(self._overlay_model_factory)
        if not isinstance(overlay_model, fast_plot_viewer.FastViewerOverlayModel):
            overlay_model = fast_plot_viewer.FastViewerOverlayModel()
        marker_overlay_model = fast_plot_viewer.build_artist_overlay_model(
            transient_artists=_resolve_runtime_value(self._marker_artist_factory),
        )
        combined_overlay_model = fast_plot_viewer.FastViewerOverlayModel(
            rectangle_specs=tuple(overlay_model.rectangle_specs),
            transient_rectangle_specs=tuple(overlay_model.transient_rectangle_specs),
            transient_marker_specs=tuple(overlay_model.transient_marker_specs)
            + tuple(marker_overlay_model.transient_marker_specs),
            transient_curve_specs=tuple(overlay_model.transient_curve_specs),
            suppress_overlay_image=bool(overlay_model.suppress_overlay_image),
        )
        background_layer = self._extract_image_layer(
            "background",
            background_artist,
            version=(
                layer_versions.get("background")
                if isinstance(layer_versions, Mapping)
                else None
            ),
        )
        simulation_layer = self._extract_image_layer(
            "simulation",
            image_artist,
            version=(
                layer_versions.get("simulation")
                if isinstance(layer_versions, Mapping)
                else None
            ),
        )
        overlay_layer = self._extract_image_layer(
            "overlay",
            overlay_artist,
            version=(
                layer_versions.get("overlay") if isinstance(layer_versions, Mapping) else None
            ),
            force_invisible=bool(combined_overlay_model.suppress_overlay_image),
        )
        peak_cache = _build_peak_cache(
            _resolve_runtime_value(self._peak_cache_factory),
            view_state,
        )
        qgroup_cache = _build_q_group_cache(
            _resolve_runtime_value(self._qgroup_cache_factory),
            view_state,
            view_mode=view_mode,
        )
        text_specs = _extract_text_specs(_resolve_runtime_value(self._overlay_artist_groups_factory))
        scene = ViewportScene(
            view_state=view_state,
            view_mode=view_mode,
            background_layer=background_layer,
            simulation_layer=simulation_layer,
            overlay_layer=overlay_layer,
            overlay_model=combined_overlay_model,
            text_specs=text_specs,
            peak_cache=peak_cache,
            q_group_cache=qgroup_cache,
            raster_signature=_scene_raster_signature(
                view_state,
                view_mode=view_mode,
                background_layer=background_layer,
                simulation_layer=simulation_layer,
            ),
            overlay_signature=_scene_overlay_signature(
                view_state,
                view_mode=view_mode,
                overlay_layer=overlay_layer,
                overlay_model=combined_overlay_model,
                text_specs=text_specs,
            ),
        )
        self.render_scene(scene)

    def _extract_image_layer(
        self,
        layer_name: str,
        artist: object,
        *,
        version: object,
        force_invisible: bool = False,
    ) -> _ViewportImageLayer | None:
        if artist is None:
            return None
        try:
            visible = bool(getattr(artist, "get_visible", lambda: True)())
        except Exception:
            visible = True
        if force_invisible:
            visible = False
        array = _coerce_2d_array(getattr(artist, "get_array", lambda: None)())
        if array is None:
            visible = False
        try:
            extent = tuple(float(value) for value in getattr(artist, "get_extent")())
        except Exception:
            extent = None
        origin = _normalize_image_origin(
            getattr(artist, "get_origin", lambda: "upper")()
        )
        interpolation = str(
            getattr(artist, "get_interpolation", lambda: "nearest")() or "nearest"
        )
        if not visible or extent is None or array is None:
            return _ViewportImageLayer(
                name=layer_name,
                visible=False,
                extent=extent,
                interpolation=interpolation,
                source_rgba=None,
                origin=origin,
            )
        try:
            cmap = getattr(getattr(artist, "get_cmap", lambda: None)(), "name", None)
        except Exception:
            cmap = None
        try:
            clim = tuple(float(value) for value in getattr(artist, "get_clim")())
        except Exception:
            clim = None
        try:
            alpha_value = getattr(artist, "get_alpha", lambda: None)()
            alpha = 1.0 if alpha_value is None else float(alpha_value)
        except Exception:
            alpha = 1.0
        cache_key = (
            version,
            id(array),
            tuple(int(value) for value in array.shape),
            cmap,
            clim,
            round(float(alpha), 6),
        )
        cached = self._layer_rgba_cache.get(layer_name)
        if cached is not None and cached[0] == cache_key:
            rgba = cached[1]
        else:
            rgba = np.asarray(artist.to_rgba(array, bytes=True, norm=True), dtype=np.uint8)
            if rgba.ndim != 3 or rgba.shape[-1] != 4:
                return None
            if alpha < 1.0:
                rgba = rgba.copy()
                rgba[..., 3] = np.asarray(
                    np.clip(rgba[..., 3].astype(np.float32) * float(alpha), 0.0, 255.0),
                    dtype=np.uint8,
                )
            self._layer_rgba_cache[layer_name] = (cache_key, rgba)
        return _ViewportImageLayer(
            name=layer_name,
            visible=True,
            extent=(
                float(extent[0]),
                float(extent[1]),
                float(extent[2]),
                float(extent[3]),
            ),
            interpolation=interpolation,
            source_rgba=rgba,
            origin=origin,
        )

    def render_scene(self, scene: ViewportScene) -> None:
        if Image is None or ImageTk is None:
            return
        raster_signature = (
            scene.raster_signature
            if scene.raster_signature is not None
            else _scene_raster_signature(
                scene.view_state,
                view_mode=scene.view_mode,
                background_layer=scene.background_layer,
                simulation_layer=scene.simulation_layer,
            )
        )
        overlay_signature = (
            scene.overlay_signature
            if scene.overlay_signature is not None
            else _scene_overlay_signature(
                scene.view_state,
                view_mode=scene.view_mode,
                overlay_layer=scene.overlay_layer,
                overlay_model=scene.overlay_model,
                text_specs=scene.text_specs,
            )
        )
        if (
            self._last_scene is not None
            and self._photo_item is not None
            and raster_signature == self._last_raster_signature
            and overlay_signature == self._last_overlay_signature
        ):
            self._last_scene = scene
            return
        if raster_signature != self._last_raster_signature:
            photo = self._render_raster_photo(scene)
            if self._photo_item is None:
                self._photo_item = self._widget.create_image(
                    0,
                    0,
                    anchor=self._tk.NW,
                    image=photo,
                    tags=("viewport_base",),
                )
            else:
                self._widget.itemconfigure(self._photo_item, image=photo)
            self._photo_image = photo
            self._last_raster_signature = raster_signature
        if overlay_signature != self._last_overlay_signature:
            overlay_photo = self._render_overlay_photo(scene)
            if overlay_photo is None:
                if self._overlay_photo_item is not None:
                    self._widget.delete(self._overlay_photo_item)
                    self._overlay_photo_item = None
                self._overlay_photo_image = None
            else:
                if self._overlay_photo_item is None:
                    self._overlay_photo_item = self._widget.create_image(
                        0,
                        0,
                        anchor=self._tk.NW,
                        image=overlay_photo,
                        tags=("viewport_overlay_image",),
                    )
                else:
                    self._widget.itemconfigure(
                        self._overlay_photo_item,
                        image=overlay_photo,
                    )
                self._overlay_photo_image = overlay_photo
            self._widget.delete("viewport_overlay")
            self._render_overlay_items(scene)
            self._last_overlay_signature = overlay_signature
        self._last_scene = scene

    def _render_raster_photo(self, scene: ViewportScene):
        base_image = Image.new(
            "RGBA",
            (max(int(scene.view_state.width), 1), max(int(scene.view_state.height), 1)),
            (255, 255, 255, 255),
        )
        for layer in (
            scene.background_layer,
            scene.simulation_layer,
        ):
            base_image = self._composite_layer(base_image, layer, scene.view_state)
        return ImageTk.PhotoImage(base_image)

    def _render_overlay_photo(self, scene: ViewportScene):
        patch = self._render_layer_patch(scene.overlay_layer, scene.view_state)
        if patch is None:
            return None
        overlay_image = Image.new(
            "RGBA",
            (max(int(scene.view_state.width), 1), max(int(scene.view_state.height), 1)),
            (0, 0, 0, 0),
        )
        layer_image, left, top = patch
        overlay_image.paste(layer_image, (int(left), int(top)))
        return ImageTk.PhotoImage(overlay_image)

    def _composite_layer(
        self,
        base_image,
        layer: _ViewportImageLayer | None,
        view_state: ViewportViewState,
    ):
        if Image is None:
            return base_image
        patch = self._render_layer_patch(layer, view_state)
        if patch is None:
            return base_image
        layer_image, left, top = patch
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        overlay.paste(layer_image, (int(left), int(top)))
        return Image.alpha_composite(base_image, overlay)

    def _render_layer_patch(
        self,
        layer: _ViewportImageLayer | None,
        view_state: ViewportViewState,
    ):
        if Image is None or layer is None or not layer.visible or layer.source_rgba is None:
            return None
        if layer.extent is None:
            return None
        src_rgba = layer.source_rgba
        src_height, src_width = (int(src_rgba.shape[0]), int(src_rgba.shape[1]))
        if src_height <= 0 or src_width <= 0:
            return None
        extent_x0, extent_x1, extent_y0, extent_y1 = layer.extent
        view_x_min, view_x_max, view_y_min, view_y_max = _view_bounds(view_state)
        extent_x_min, extent_x_max = (min(extent_x0, extent_x1), max(extent_x0, extent_x1))
        extent_y_min, extent_y_max = (min(extent_y0, extent_y1), max(extent_y0, extent_y1))
        overlap_x0 = max(view_x_min, extent_x_min)
        overlap_x1 = min(view_x_max, extent_x_max)
        overlap_y0 = max(view_y_min, extent_y_min)
        overlap_y1 = min(view_y_max, extent_y_max)
        if overlap_x1 - overlap_x0 <= _EPSILON or overlap_y1 - overlap_y0 <= _EPSILON:
            return None
        src_edge_x0 = self._world_to_source_edge(overlap_x0, extent_x0, extent_x1, src_width)
        src_edge_x1 = self._world_to_source_edge(overlap_x1, extent_x0, extent_x1, src_width)
        origin = _normalize_image_origin(layer.origin)
        source_y_start, source_y_end = (
            (extent_y1, extent_y0) if origin == "upper" else (extent_y0, extent_y1)
        )
        src_edge_y0 = self._world_to_source_edge(overlap_y0, source_y_start, source_y_end, src_height)
        src_edge_y1 = self._world_to_source_edge(overlap_y1, source_y_start, source_y_end, src_height)
        if None in {src_edge_x0, src_edge_x1, src_edge_y0, src_edge_y1}:
            return None
        flip_x = bool(src_edge_x1 < src_edge_x0)
        flip_y = origin == "lower"
        src_left = max(0, int(floor(min(src_edge_x0, src_edge_x1))))
        src_right = min(src_width, int(ceil(max(src_edge_x0, src_edge_x1))))
        src_top = max(0, int(floor(min(src_edge_y0, src_edge_y1))))
        src_bottom = min(src_height, int(ceil(max(src_edge_y0, src_edge_y1))))
        if src_right <= src_left or src_bottom <= src_top:
            return None
        patch_rgba = src_rgba[src_top:src_bottom, src_left:src_right]
        if patch_rgba.size == 0:
            return None
        top_left = world_to_screen(view_state, overlap_x0, overlap_y0)
        bottom_right = world_to_screen(view_state, overlap_x1, overlap_y1)
        if top_left is None or bottom_right is None:
            return None
        dst_left = max(0, int(round(min(top_left[0], bottom_right[0]))))
        dst_right = min(view_state.width, int(round(max(top_left[0], bottom_right[0]))))
        dst_top = max(0, int(round(min(top_left[1], bottom_right[1]))))
        dst_bottom = min(view_state.height, int(round(max(top_left[1], bottom_right[1]))))
        if dst_right <= dst_left:
            dst_right = min(view_state.width, dst_left + 1)
        if dst_bottom <= dst_top:
            dst_bottom = min(view_state.height, dst_top + 1)
        if dst_right <= dst_left or dst_bottom <= dst_top:
            return None
        patch_image = Image.fromarray(np.ascontiguousarray(patch_rgba))
        if flip_x:
            patch_image = patch_image.transpose(_PIL_TRANSPOSE_LEFT_RIGHT)
        if flip_y:
            patch_image = patch_image.transpose(_PIL_TRANSPOSE_TOP_BOTTOM)
        patch_image = patch_image.resize(
            (int(max(1, dst_right - dst_left)), int(max(1, dst_bottom - dst_top))),
            resample=_interpolation_to_resample(layer.interpolation),
        )
        return patch_image, dst_left, dst_top

    def _world_to_source_edge(
        self,
        world_value: float,
        extent_start: float,
        extent_end: float,
        size: int,
    ) -> float | None:
        span = float(extent_end) - float(extent_start)
        if abs(span) <= _EPSILON or int(size) <= 0:
            return None
        src_edge = ((float(world_value) - float(extent_start)) / span) * float(size)
        return float(src_edge) if isfinite(src_edge) else None

    def _render_overlay_items(self, scene: ViewportScene) -> None:
        draw_records: list[tuple[float, str, object]] = []
        for spec in scene.overlay_model.rectangle_specs:
            draw_records.append((float(spec.zorder), "rect", spec))
        for spec in scene.overlay_model.transient_rectangle_specs:
            draw_records.append((float(spec.zorder), "rect", spec))
        for spec in scene.overlay_model.transient_curve_specs:
            draw_records.append((float(spec.zorder), "curve", spec))
        for spec in scene.overlay_model.transient_marker_specs:
            draw_records.append((float(spec.zorder), "marker", spec))
        for spec in scene.text_specs:
            draw_records.append((float(spec.zorder), "text", spec))
        draw_records.sort(key=lambda item: item[0])
        for _, kind, spec in draw_records:
            if kind == "rect":
                self._draw_rectangle(spec, scene.view_state)
            elif kind == "curve":
                self._draw_curve(spec, scene.view_state)
            elif kind == "marker":
                self._draw_marker(spec, scene.view_state)
            elif kind == "text":
                self._draw_text(spec, scene.view_state)

    def _draw_rectangle(self, spec, view_state: ViewportViewState) -> None:
        start = world_to_screen(view_state, float(spec.x), float(spec.y))
        end = world_to_screen(
            view_state,
            float(spec.x) + float(spec.width),
            float(spec.y) + float(spec.height),
        )
        if start is None or end is None:
            return
        x0, y0 = start
        x1, y1 = end
        self._widget.create_rectangle(
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            outline=_rgba_to_hex(tuple(spec.edge_rgba)),
            width=max(1.0, float(spec.linewidth)),
            dash=_dash_pattern(spec.linestyle),
            fill="",
            tags=("viewport_overlay",),
        )

    def _draw_curve(self, spec, view_state: ViewportViewState) -> None:
        current_segment: list[float] = []
        for x_value, y_value in zip(spec.x_values, spec.y_values, strict=False):
            if not (isfinite(float(x_value)) and isfinite(float(y_value))):
                self._flush_curve_segment(current_segment, spec)
                current_segment = []
                continue
            screen_point = world_to_screen(view_state, float(x_value), float(y_value))
            if screen_point is None:
                continue
            current_segment.extend([float(screen_point[0]), float(screen_point[1])])
        self._flush_curve_segment(current_segment, spec)

    def _flush_curve_segment(self, coords: list[float], spec) -> None:
        if len(coords) < 4:
            return
        self._widget.create_line(
            *coords,
            fill=_rgba_to_hex(tuple(spec.edge_rgba)),
            width=max(1.0, float(spec.linewidth)),
            dash=_dash_pattern(spec.linestyle),
            tags=("viewport_overlay",),
        )

    def _draw_marker(self, spec, view_state: ViewportViewState) -> None:
        radius = max(2.0, float(spec.size) / 2.0)
        outline = _rgba_to_hex(tuple(spec.edge_rgba))
        fill = ""
        if len(spec.face_rgba) >= 4 and int(spec.face_rgba[3]) > 0:
            fill = _rgba_to_hex(tuple(spec.face_rgba))
        for x_value, y_value in zip(spec.x_values, spec.y_values, strict=False):
            screen_point = world_to_screen(view_state, float(x_value), float(y_value))
            if screen_point is None:
                continue
            sx, sy = screen_point
            symbol = str(spec.symbol or "o")
            if symbol == "s":
                self._widget.create_rectangle(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    outline=outline,
                    fill=fill,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )
            elif symbol == "x":
                self._widget.create_line(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    fill=outline,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )
                self._widget.create_line(
                    sx - radius,
                    sy + radius,
                    sx + radius,
                    sy - radius,
                    fill=outline,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )
            elif symbol == "+":
                self._widget.create_line(
                    sx - radius,
                    sy,
                    sx + radius,
                    sy,
                    fill=outline,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )
                self._widget.create_line(
                    sx,
                    sy - radius,
                    sx,
                    sy + radius,
                    fill=outline,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )
            else:
                self._widget.create_oval(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    outline=outline,
                    fill=fill,
                    width=max(1.0, float(spec.pen_width)),
                    tags=("viewport_overlay",),
                )

    def _draw_text(self, spec: _ViewportTextSpec, view_state: ViewportViewState) -> None:
        screen_point = world_to_screen(view_state, float(spec.x), float(spec.y))
        if screen_point is None:
            return
        self._widget.create_text(
            float(screen_point[0]),
            float(screen_point[1]),
            text=spec.text,
            fill=_rgba_to_hex(spec.fill_rgba),
            font=("TkDefaultFont", max(8, int(spec.font_size))),
            anchor=self._tk.CENTER,
            tags=("viewport_overlay",),
        )


class TkCanvasViewportProxy:
    """Drop-in canvas proxy that dispatches Matplotlib-style events from Tk."""

    def __init__(
        self,
        viewport: _TkPrimaryViewport,
        *,
        event_axes=None,
        draw_interval_s: float = 1.0 / 60.0,
        sync_callback: Callable[[], None] | None = None,
    ) -> None:
        self._viewport = viewport
        self._event_axes = event_axes
        self._draw_interval_s = float(max(0.0, draw_interval_s))
        self._sync_callback = sync_callback
        self._callbacks: dict[str, dict[int, Callable]] = {}
        self._next_cid = 1
        self._draw_idle_token = None
        self._last_draw_ts = -float("inf")
        self._bind_widget_events()

    def set_sync_callback(self, callback: Callable[[], None] | None) -> None:
        self._sync_callback = callback

    def get_tk_widget(self):
        return self._viewport.widget

    def mpl_connect(self, event_name: str, callback: Callable):
        cid = self._next_cid
        self._next_cid += 1
        bucket = self._callbacks.setdefault(str(event_name), {})
        bucket[cid] = callback
        return cid

    def mpl_disconnect(self, cid: int):
        for bucket in self._callbacks.values():
            if cid in bucket:
                del bucket[cid]
                break
        return None

    def draw(self, *args, **kwargs):
        del args, kwargs
        self._cancel_draw_idle()
        self._sync(force=True)
        return None

    def draw_idle(self, *args, **kwargs):
        del args, kwargs
        widget = self.get_tk_widget()
        after = getattr(widget, "after", None)
        if not callable(after):
            self._sync(force=False)
            return None
        if self._draw_idle_token is not None:
            return None
        delay_ms = 0
        if self._draw_interval_s > 0.0 and isfinite(self._last_draw_ts):
            elapsed = time.perf_counter() - self._last_draw_ts
            delay_s = max(0.0, self._draw_interval_s - elapsed)
            delay_ms = max(0, int(round(delay_s * 1000.0)))
        self._draw_idle_token = after(delay_ms, self._flush_draw_idle)
        return None

    def _flush_draw_idle(self) -> None:
        self._draw_idle_token = None
        self._sync(force=False)

    def _cancel_draw_idle(self) -> None:
        token = self._draw_idle_token
        self._draw_idle_token = None
        if token is None:
            return
        widget = self.get_tk_widget()
        after_cancel = getattr(widget, "after_cancel", None)
        if callable(after_cancel):
            try:
                after_cancel(token)
            except Exception:
                pass

    def _sync(self, *, force: bool) -> None:
        if not callable(self._sync_callback):
            return
        self._last_draw_ts = time.perf_counter()
        try:
            self._sync_callback()
        except Exception:
            if force:
                raise

    def _bind_widget_events(self) -> None:
        widget = self.get_tk_widget()
        binder = getattr(widget, "bind", None)
        if not callable(binder):
            return
        binder(
            "<ButtonPress-1>",
            lambda event: self.dispatch_tk_event(
                "button_press_event",
                event,
                button=1,
            ),
            add="+",
        )
        binder(
            "<Double-Button-1>",
            lambda event: self.dispatch_tk_event(
                "button_press_event",
                event,
                button=1,
                dblclick=True,
            ),
            add="+",
        )
        binder(
            "<ButtonRelease-1>",
            lambda event: self.dispatch_tk_event(
                "button_release_event",
                event,
                button=1,
            ),
            add="+",
        )
        binder(
            "<ButtonPress-3>",
            lambda event: self.dispatch_tk_event(
                "button_press_event",
                event,
                button=3,
            ),
            add="+",
        )
        binder(
            "<ButtonRelease-3>",
            lambda event: self.dispatch_tk_event(
                "button_release_event",
                event,
                button=3,
            ),
            add="+",
        )
        binder(
            "<B1-Motion>",
            lambda event: self.dispatch_tk_event("motion_notify_event", event, button=None),
            add="+",
        )
        binder(
            "<B3-Motion>",
            lambda event: self.dispatch_tk_event("motion_notify_event", event, button=None),
            add="+",
        )
        binder(
            "<Motion>",
            lambda event: self.dispatch_tk_event("motion_notify_event", event, button=None),
            add="+",
        )
        binder(
            "<MouseWheel>",
            self._dispatch_mousewheel,
            add="+",
        )
        binder(
            "<Button-4>",
            lambda event: self.dispatch_tk_event(
                "scroll_event",
                event,
                button="up",
                step=1.0,
            ),
            add="+",
        )
        binder(
            "<Button-5>",
            lambda event: self.dispatch_tk_event(
                "scroll_event",
                event,
                button="down",
                step=-1.0,
            ),
            add="+",
        )

    def _dispatch_mousewheel(self, event) -> None:
        try:
            delta = float(getattr(event, "delta", 0.0))
        except Exception:
            delta = 0.0
        step = 0.0 if not isfinite(delta) or abs(delta) <= _EPSILON else delta / 120.0
        button = "up" if step > 0.0 else "down"
        self.dispatch_tk_event(
            "scroll_event",
            event,
            button=button,
            step=float(step),
        )

    def dispatch_tk_event(
        self,
        event_name: str,
        event: object,
        *,
        button: int | str | None,
        dblclick: bool = False,
        step: float = 0.0,
    ) -> None:
        callbacks = list(self._callbacks.get(str(event_name), {}).values())
        if not callbacks:
            return
        event_obj = self._build_viewport_event(
            event_name,
            event,
            button=button,
            dblclick=dblclick,
            step=step,
        )
        for callback in callbacks:
            try:
                callback(event_obj)
            except Exception:
                pass

    def _build_viewport_event(
        self,
        event_name: str,
        event: object,
        *,
        button: int | str | None,
        dblclick: bool,
        step: float,
    ) -> ViewportEvent:
        x_pixel = _safe_float(getattr(event, "x", 0.0))
        y_pixel = _safe_float(getattr(event, "y", 0.0))
        inaxes = None
        xdata = None
        ydata = None
        if x_pixel is not None and y_pixel is not None and self._viewport.contains_screen_point(
            x_pixel,
            y_pixel,
        ):
            world = self._viewport.screen_to_world(x_pixel, y_pixel)
            if world is not None:
                xdata = float(world[0])
                ydata = float(world[1])
                inaxes = self._event_axes
        return ViewportEvent(
            name=str(event_name),
            x=x_pixel,
            y=y_pixel,
            xdata=xdata,
            ydata=ydata,
            button=button,
            dblclick=bool(dblclick),
            inaxes=inaxes,
            step=float(step),
            canvas=self,
        )


def build_tk_primary_viewport_backend(
    *,
    tk_module,
    canvas_frame,
    matplotlib_canvas,
    ax,
    image_artist,
    background_artist,
    overlay_artist,
    marker_artist_factory: object = None,
    overlay_model_factory: object = None,
    overlay_artist_groups_factory: object = None,
    layer_versions_factory: object = None,
    peak_cache_factory: object = None,
    qgroup_cache_factory: object = None,
    draw_interval_s: float = 1.0 / 60.0,
) -> PrimaryViewportBackend:
    """Build a Tk-native primary viewport backed by current Matplotlib artists."""

    matplotlib_widget = matplotlib_canvas.get_tk_widget()
    initial_width = _safe_widget_dimension(matplotlib_widget, "winfo_reqwidth", 800)
    initial_height = _safe_widget_dimension(matplotlib_widget, "winfo_reqheight", 800)
    viewport = _TkPrimaryViewport(
        tk_module=tk_module,
        parent=canvas_frame,
        ax=ax,
        marker_artist_factory=marker_artist_factory,
        overlay_model_factory=overlay_model_factory,
        overlay_artist_groups_factory=overlay_artist_groups_factory,
        layer_versions_factory=layer_versions_factory,
        peak_cache_factory=peak_cache_factory,
        qgroup_cache_factory=qgroup_cache_factory,
        initial_width=initial_width,
        initial_height=initial_height,
    )

    def _sync_from_matplotlib(*, force_view_range: bool = False) -> None:
        viewport.sync_from_matplotlib(
            image_artist=image_artist,
            background_artist=background_artist,
            overlay_artist=overlay_artist,
            force_view_range=bool(force_view_range),
        )

    proxy = TkCanvasViewportProxy(
        viewport,
        event_axes=ax,
        draw_interval_s=float(draw_interval_s),
        sync_callback=lambda: _sync_from_matplotlib(force_view_range=False),
    )

    def _activate() -> None:
        try:
            matplotlib_widget.pack_forget()
        except Exception:
            pass
        viewport.widget.pack(
            side=tk_module.TOP,
            fill=tk_module.BOTH,
            expand=True,
        )
        _sync_from_matplotlib(force_view_range=True)

    def _deactivate() -> None:
        try:
            viewport.widget.pack_forget()
        except Exception:
            pass
        try:
            matplotlib_widget.pack(
                side=tk_module.TOP,
                fill=tk_module.BOTH,
                expand=True,
            )
        except Exception:
            pass

    def _shutdown() -> None:
        try:
            viewport.widget.destroy()
        except Exception:
            pass

    return PrimaryViewportBackend(
        name="tk_canvas",
        canvas_proxy=proxy,
        activate=_activate,
        deactivate=_deactivate,
        sync_from_matplotlib=_sync_from_matplotlib,
        shutdown=_shutdown,
    )
