from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ra_sim.debug_controls import retain_optional_cache
from ra_sim.simulation.exact_cake_portable import (
    CakeTransformBundle,
    detector_pixel_to_caked_bin,
    resolve_cake_transform_bundle,
)
from . import overlays as gui_overlays
from ra_sim.simulation.intersection_analysis import (
    IntersectionGeometry,
    project_qr_cylinder_to_detector,
)


@dataclass(frozen=True)
class QrCylinderOverlayRenderConfig:
    """Normalized runtime inputs needed to render analytic Qr-cylinder traces."""

    render_in_caked_space: bool
    image_size: int
    display_rotate_k: int
    center_col: float
    center_row: float
    distance_cor_to_detector: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    theta_initial_deg: float
    cor_angle_deg: float
    pixel_size_m: float
    wavelength: float
    n2: complex
    phi_samples: int = 721
    two_theta_limits: tuple[float, float] = (0.0, 90.0)


@dataclass
class QrCylinderOverlayRuntimeBindings:
    """Runtime callbacks and shared state used by the live overlay workflow."""

    ax: Any
    overlay_artists: list[object]
    overlay_cache: dict[str, object]
    overlay_enabled_factory: object
    get_active_entries: Callable[[], Sequence[dict[str, object]]]
    render_config_factory: object
    ai_factory: object
    get_detector_angular_maps: Callable[[Any], tuple[object, object]]
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ]
    get_caked_projection_context: Callable[[], Mapping[str, object] | None] | None = None
    draw_idle: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _set_status_text(
    bindings: QrCylinderOverlayRuntimeBindings,
    text: str,
) -> None:
    if callable(bindings.set_status_text):
        bindings.set_status_text(str(text))


def _overlay_enabled(bindings: QrCylinderOverlayRuntimeBindings) -> bool:
    return bool(_resolve_runtime_value(bindings.overlay_enabled_factory))


def _runtime_render_config(
    bindings: QrCylinderOverlayRuntimeBindings,
) -> QrCylinderOverlayRenderConfig | None:
    config = _resolve_runtime_value(bindings.render_config_factory)
    if isinstance(config, QrCylinderOverlayRenderConfig):
        return config
    return None


def _runtime_ai(bindings: QrCylinderOverlayRuntimeBindings):
    return _resolve_runtime_value(bindings.ai_factory)


def _retain_qr_cylinder_overlay_cache(bindings: QrCylinderOverlayRuntimeBindings) -> bool:
    return retain_optional_cache(
        "qr_cylinder_overlay",
        feature_needed=_overlay_enabled(bindings),
    )


def build_qr_cylinder_overlay_render_config(
    *,
    render_in_caked_space: object,
    image_size: object,
    display_rotate_k: object,
    center_col: object,
    center_row: object,
    distance_cor_to_detector: object,
    gamma_deg: object,
    Gamma_deg: object,
    chi_deg: object,
    psi_deg: object,
    psi_z_deg: object,
    zs: object,
    zb: object,
    theta_initial_deg: object,
    cor_angle_deg: object,
    pixel_size_m: object,
    wavelength: object,
    n2: object,
    phi_samples: object = 721,
    two_theta_limits: tuple[float, float] = (0.0, 90.0),
) -> QrCylinderOverlayRenderConfig:
    """Build one validated overlay render config from runtime scalar values."""

    return QrCylinderOverlayRenderConfig(
        render_in_caked_space=bool(render_in_caked_space),
        image_size=int(image_size),
        display_rotate_k=int(display_rotate_k),
        center_col=float(center_col),
        center_row=float(center_row),
        distance_cor_to_detector=float(distance_cor_to_detector),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
        chi_deg=float(chi_deg),
        psi_deg=float(psi_deg),
        psi_z_deg=float(psi_z_deg),
        zs=float(zs),
        zb=float(zb),
        theta_initial_deg=float(theta_initial_deg),
        cor_angle_deg=float(cor_angle_deg),
        pixel_size_m=float(pixel_size_m),
        wavelength=float(wavelength),
        n2=complex(n2),
        phi_samples=int(phi_samples),
        two_theta_limits=(
            float(two_theta_limits[0]),
            float(two_theta_limits[1]),
        ),
    )


def make_runtime_qr_cylinder_overlay_render_config_factory(
    *,
    render_in_caked_space_factory: object,
    image_size: object,
    display_rotate_k: object,
    center_col_factory: object,
    center_row_factory: object,
    distance_cor_to_detector_factory: object,
    gamma_deg_factory: object,
    Gamma_deg_factory: object,
    chi_deg_factory: object,
    psi_deg_factory: object,
    psi_z_deg_factory: object,
    zs_factory: object,
    zb_factory: object,
    theta_initial_deg_factory: object,
    cor_angle_deg_factory: object,
    pixel_size_m: object,
    wavelength: object,
    n2: object,
    phi_samples: object = 721,
    two_theta_limits: tuple[float, float] = (0.0, 90.0),
) -> Callable[[], QrCylinderOverlayRenderConfig]:
    """Return a zero-arg factory for the live analytic overlay config."""

    return lambda: build_qr_cylinder_overlay_render_config(
        render_in_caked_space=_resolve_runtime_value(render_in_caked_space_factory),
        image_size=_resolve_runtime_value(image_size),
        display_rotate_k=_resolve_runtime_value(display_rotate_k),
        center_col=_resolve_runtime_value(center_col_factory),
        center_row=_resolve_runtime_value(center_row_factory),
        distance_cor_to_detector=_resolve_runtime_value(
            distance_cor_to_detector_factory
        ),
        gamma_deg=_resolve_runtime_value(gamma_deg_factory),
        Gamma_deg=_resolve_runtime_value(Gamma_deg_factory),
        chi_deg=_resolve_runtime_value(chi_deg_factory),
        psi_deg=_resolve_runtime_value(psi_deg_factory),
        psi_z_deg=_resolve_runtime_value(psi_z_deg_factory),
        zs=_resolve_runtime_value(zs_factory),
        zb=_resolve_runtime_value(zb_factory),
        theta_initial_deg=_resolve_runtime_value(theta_initial_deg_factory),
        cor_angle_deg=_resolve_runtime_value(cor_angle_deg_factory),
        pixel_size_m=_resolve_runtime_value(pixel_size_m),
        wavelength=_resolve_runtime_value(wavelength),
        n2=_resolve_runtime_value(n2),
        phi_samples=_resolve_runtime_value(phi_samples),
        two_theta_limits=two_theta_limits,
    )


def build_qr_cylinder_overlay_signature(
    entries: Sequence[dict[str, object]],
    *,
    config: QrCylinderOverlayRenderConfig,
    projection_context: Mapping[str, object] | None = None,
) -> tuple[object, ...]:
    """Return a cache signature for analytic detector-trace overlay inputs."""

    qr_keys = tuple(
        (str(entry["source"]), int(entry["m"]), round(float(entry["qr"]), 10))
        for entry in entries
    )
    projection_signature: object = None
    if bool(config.render_in_caked_space):
        projection_signature = _projection_context_signature(projection_context)
    return (
        tuple(qr_keys),
        bool(config.render_in_caked_space),
        int(config.image_size),
        int(config.display_rotate_k),
        float(config.center_col),
        float(config.center_row),
        float(config.distance_cor_to_detector),
        float(config.gamma_deg),
        float(config.Gamma_deg),
        float(config.chi_deg),
        float(config.psi_deg),
        float(config.psi_z_deg),
        float(config.zs),
        float(config.zb),
        float(config.theta_initial_deg),
        float(config.cor_angle_deg),
        float(config.pixel_size_m),
        float(config.wavelength),
        round(float(np.real(config.n2)), 12),
        round(float(np.imag(config.n2)), 12),
        projection_signature,
    )


def _trace_geometry(
    config: QrCylinderOverlayRenderConfig,
) -> IntersectionGeometry:
    return IntersectionGeometry(
        image_size=int(config.image_size),
        center_col=float(config.center_col),
        center_row=float(config.center_row),
        distance_cor_to_detector=float(config.distance_cor_to_detector),
        gamma_deg=float(config.gamma_deg),
        Gamma_deg=float(config.Gamma_deg),
        chi_deg=float(config.chi_deg),
        psi_deg=float(config.psi_deg),
        psi_z_deg=float(config.psi_z_deg),
        zs=float(config.zs),
        zb=float(config.zb),
        theta_initial_deg=float(config.theta_initial_deg),
        cor_angle_deg=float(config.cor_angle_deg),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        pixel_size_m=float(config.pixel_size_m),
    )


def _float64_axis_digest(values: object) -> tuple[tuple[int, ...], str] | None:
    try:
        axis = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if axis.size <= 0 or not np.all(np.isfinite(axis)):
        return None
    return tuple(int(v) for v in axis.shape), hashlib.blake2b(
        axis.tobytes(order="C"),
        digest_size=16,
    ).hexdigest()


def _resolve_caked_projection_context(
    projection_context: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(projection_context, Mapping):
        return None
    bundle = projection_context.get("transform_bundle")
    if not isinstance(bundle, CakeTransformBundle):
        return None
    try:
        azimuth_axis = np.asarray(
            projection_context.get("azimuth_axis"),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return None
    if (
        azimuth_axis.size != int(np.asarray(bundle.raw_azimuth_deg).size)
        or not np.all(np.isfinite(azimuth_axis))
    ):
        return None
    try:
        radial_axis = np.asarray(
            projection_context.get("radial_axis", bundle.radial_deg),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return None
    if (
        radial_axis.size != int(np.asarray(bundle.radial_deg).size)
        or not np.all(np.isfinite(radial_axis))
    ):
        return None
    raw_azimuth_axis = projection_context.get("raw_azimuth_axis")
    radial_digest = _float64_axis_digest(
        radial_axis
    )
    azimuth_digest = _float64_axis_digest(azimuth_axis)
    if radial_digest is None or azimuth_digest is None:
        return None
    try:
        detector_shape = tuple(
            int(v)
            for v in tuple(
                projection_context.get("detector_shape", bundle.detector_shape)
            )[:2]
        )
    except Exception:
        return None
    if (
        len(detector_shape) < 2
    ):
        return None
    resolved_bundle = resolve_cake_transform_bundle(
        None,
        detector_shape,
        radial_axis,
        gui_azimuth_deg=azimuth_axis,
        raw_azimuth_deg=raw_azimuth_axis,
        transform_bundle=bundle,
        require_gui_display_match=True,
    )
    if not isinstance(resolved_bundle, CakeTransformBundle):
        return None
    return {
        "transform_bundle": resolved_bundle,
        "detector_shape": detector_shape,
        "radial_signature": radial_digest,
        "azimuth_signature": azimuth_digest,
    }


def _projection_context_signature(
    projection_context: Mapping[str, object] | None,
) -> object:
    resolved_projection = _resolve_caked_projection_context(projection_context)
    if resolved_projection is None:
        return None
    return (
        tuple(int(v) for v in resolved_projection["detector_shape"]),
        resolved_projection["radial_signature"],
        resolved_projection["azimuth_signature"],
    )


def _display_trace_from_detector_trace(
    trace: Any,
    *,
    config: QrCylinderOverlayRenderConfig,
    projection_context: Mapping[str, object] | None,
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
) -> tuple[np.ndarray, np.ndarray]:
    detector_cols = np.asarray(trace.detector_col, dtype=np.float64)
    detector_rows = np.asarray(trace.detector_row, dtype=np.float64)
    valid_mask = np.asarray(trace.valid_mask, dtype=bool)

    if config.render_in_caked_space:
        return interpolate_trace_to_caked_coords(
            detector_cols=detector_cols,
            detector_rows=detector_rows,
            valid_mask=valid_mask,
            projection_context=projection_context,
            two_theta_limits=config.two_theta_limits,
        )

    native_shape = (int(config.image_size), int(config.image_size))
    display_cols = np.full(detector_cols.shape, np.nan, dtype=np.float64)
    display_rows = np.full(detector_rows.shape, np.nan, dtype=np.float64)
    for idx in np.nonzero(valid_mask)[0]:
        try:
            dcol, drow = native_sim_to_display_coords(
                float(detector_cols[idx]),
                float(detector_rows[idx]),
                native_shape,
            )
        except Exception:
            continue
        display_cols[idx] = float(dcol)
        display_rows[idx] = float(drow)
    return display_cols, display_rows


def build_qr_cylinder_overlay_paths(
    entries: Sequence[dict[str, object]],
    *,
    config: QrCylinderOverlayRenderConfig,
    projection_context: Mapping[str, object] | None,
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    project_traces: Callable[..., Sequence[Any]] = project_qr_cylinder_to_detector,
) -> list[dict[str, object]]:
    """Build visible detector or caked overlay paths for the active Qr groups."""

    geometry = _trace_geometry(config)
    paths: list[dict[str, object]] = []

    for entry in entries:
        try:
            traces = project_traces(
                qr_value=float(entry["qr"]),
                geometry=geometry,
                wavelength=float(config.wavelength),
                n2=config.n2,
                phi_samples=int(config.phi_samples),
            )
        except Exception:
            continue
        for trace in traces:
            display_cols, display_rows = _display_trace_from_detector_trace(
                trace,
                config=config,
                projection_context=projection_context,
                native_sim_to_display_coords=native_sim_to_display_coords,
            )
            visible = np.isfinite(display_cols) & np.isfinite(display_rows)
            if int(np.count_nonzero(visible)) < 2:
                continue
            paths.append(
                {
                    "source": str(entry.get("source", "primary")),
                    "qr": float(entry["qr"]),
                    "cols": display_cols,
                    "rows": display_rows,
                }
            )

    return paths


def clear_runtime_qr_cylinder_overlay_artists(
    bindings: QrCylinderOverlayRuntimeBindings,
    *,
    redraw: bool = True,
) -> None:
    """Remove live Qr-cylinder overlay artists from the current axes."""

    gui_overlays.clear_artists(
        bindings.overlay_artists,
        draw_idle=bindings.draw_idle,
        redraw=redraw,
    )


def invalidate_runtime_qr_cylinder_overlay_cache(
    bindings: QrCylinderOverlayRuntimeBindings,
    *,
    clear_artists: bool = False,
    redraw: bool = False,
) -> None:
    """Drop cached overlay paths so the next refresh rebuilds them."""

    bindings.overlay_cache["signature"] = None
    bindings.overlay_cache["paths"] = []
    if clear_artists:
        clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=redraw)


def refresh_runtime_qr_cylinder_overlay(
    bindings: QrCylinderOverlayRuntimeBindings,
    *,
    redraw: bool = True,
    update_status: bool = False,
) -> None:
    """Refresh the live detector/caked Qr-cylinder overlay from runtime inputs."""

    if not _overlay_enabled(bindings):
        clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=redraw)
        return

    entries = list(bindings.get_active_entries() or [])
    if not entries:
        invalidate_runtime_qr_cylinder_overlay_cache(bindings)
        clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=redraw)
        if update_status:
            _set_status_text(
                bindings,
                "Qr cylinder overlay unavailable: no active Bragg Qr groups.",
            )
        return

    render_config = _runtime_render_config(bindings)
    if render_config is None:
        clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=redraw)
        return

    projection_context = None
    if bool(render_config.render_in_caked_space) and callable(
        bindings.get_caked_projection_context
    ):
        try:
            projection_context = bindings.get_caked_projection_context()
        except Exception:
            projection_context = None

    signature = build_qr_cylinder_overlay_signature(
        entries,
        config=render_config,
        projection_context=projection_context,
    )
    retain_cache = _retain_qr_cylinder_overlay_cache(bindings)
    cached_signature = bindings.overlay_cache.get("signature") if retain_cache else None
    if cached_signature != signature:
        paths = build_qr_cylinder_overlay_paths(
            entries,
            config=render_config,
            projection_context=projection_context,
            native_sim_to_display_coords=bindings.native_sim_to_display_coords,
        )
        if retain_cache:
            bindings.overlay_cache["signature"] = signature
            bindings.overlay_cache["paths"] = paths
        else:
            bindings.overlay_cache["signature"] = None
            bindings.overlay_cache["paths"] = []

    paths = (
        list(bindings.overlay_cache.get("paths", []) or [])
        if retain_cache
        else list(paths)
    )
    if not paths:
        clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=False)
        if redraw and callable(bindings.draw_idle):
            bindings.draw_idle()
        if update_status:
            _set_status_text(
                bindings,
                "Qr cylinder overlay found no visible traces in the current view.",
            )
        return

    gui_overlays.draw_qr_cylinder_overlay_paths(
        bindings.ax,
        paths,
        qr_cylinder_overlay_artists=bindings.overlay_artists,
        clear_qr_cylinder_overlay_artists=(
            lambda *, redraw=False: clear_runtime_qr_cylinder_overlay_artists(
                bindings,
                redraw=redraw,
            )
        ),
        draw_idle=bindings.draw_idle or (lambda: None),
        redraw=redraw,
    )
    if update_status:
        _set_status_text(
            bindings,
            f"Showing analytic Qr-cylinder traces for {len(entries)} active Qr groups.",
        )


def toggle_runtime_qr_cylinder_overlay(
    bindings: QrCylinderOverlayRuntimeBindings,
) -> None:
    """Handle one user toggle of the live Qr-cylinder overlay."""

    if _overlay_enabled(bindings):
        refresh_runtime_qr_cylinder_overlay(
            bindings,
            redraw=True,
            update_status=True,
        )
        return

    clear_runtime_qr_cylinder_overlay_artists(bindings, redraw=True)
    _set_status_text(bindings, "Qr cylinder overlay hidden.")


def wrap_caked_phi_degrees(phi_values: np.ndarray | float) -> np.ndarray:
    """Wrap azimuth values into the ``[-180, 180)`` interval."""

    return ((np.asarray(phi_values, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _bilinear_sample(
    image: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Sample one 2D image at fractional detector coordinates."""

    arr = np.asarray(image, dtype=np.float64)
    col_arr = np.asarray(cols, dtype=np.float64)
    row_arr = np.asarray(rows, dtype=np.float64)
    out = np.full(col_arr.shape, np.nan, dtype=np.float64)

    if arr.ndim != 2 or col_arr.shape != row_arr.shape or arr.size == 0:
        return out

    height, width = arr.shape
    if height <= 0 or width <= 0:
        return out

    valid = (
        np.isfinite(col_arr)
        & np.isfinite(row_arr)
        & (col_arr >= 0.0)
        & (row_arr >= 0.0)
        & (col_arr <= float(width - 1))
        & (row_arr <= float(height - 1))
    )
    if not np.any(valid):
        return out

    cols_valid = col_arr[valid]
    rows_valid = row_arr[valid]

    col0 = np.floor(cols_valid).astype(np.int64, copy=False)
    row0 = np.floor(rows_valid).astype(np.int64, copy=False)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row1 = np.clip(row0 + 1, 0, height - 1)

    dcol = cols_valid - col0
    drow = rows_valid - row0

    top = (1.0 - dcol) * arr[row0, col0] + dcol * arr[row0, col1]
    bottom = (1.0 - dcol) * arr[row1, col0] + dcol * arr[row1, col1]
    out[valid] = (1.0 - drow) * top + drow * bottom
    return out


def _sample_wrapped_phi_degrees(
    phi_map_deg: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Interpolate wrapped azimuths without introducing 180-degree artifacts."""

    phi_wrapped = wrap_caked_phi_degrees(phi_map_deg)
    phi_rad = np.deg2rad(phi_wrapped)
    sin_sample = _bilinear_sample(np.sin(phi_rad), cols, rows)
    cos_sample = _bilinear_sample(np.cos(phi_rad), cols, rows)

    phi_deg = np.full_like(sin_sample, np.nan, dtype=np.float64)
    finite = np.isfinite(sin_sample) & np.isfinite(cos_sample)
    if not np.any(finite):
        return phi_deg

    magnitude = np.hypot(sin_sample[finite], cos_sample[finite])
    good = magnitude > 1e-12
    if not np.any(good):
        return phi_deg

    finite_idx = np.flatnonzero(finite)
    good_idx = finite_idx[good]
    phi_deg[good_idx] = wrap_caked_phi_degrees(
        np.rad2deg(np.arctan2(sin_sample[good_idx], cos_sample[good_idx]))
    )
    return phi_deg


def interpolate_trace_to_caked_coords(
    *,
    detector_cols: np.ndarray,
    detector_rows: np.ndarray,
    valid_mask: np.ndarray | None,
    projection_context: Mapping[str, object] | None,
    two_theta_limits: tuple[float, float] = (0.0, 90.0),
    discontinuity_threshold_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one detector-space trace into caked ``(2theta, phi)`` coordinates."""

    cols = np.asarray(detector_cols, dtype=np.float64)
    rows = np.asarray(detector_rows, dtype=np.float64)
    tth = np.full(cols.shape, np.nan, dtype=np.float64)
    phi = np.full(cols.shape, np.nan, dtype=np.float64)

    if cols.shape != rows.shape:
        return tth, phi

    base_valid = np.isfinite(cols) & np.isfinite(rows)
    if valid_mask is not None:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        if mask_arr.shape != cols.shape:
            return tth, phi
        base_valid &= mask_arr
    if not np.any(base_valid):
        return tth, phi

    resolved_projection = _resolve_caked_projection_context(projection_context)
    if resolved_projection is None:
        return tth, phi
    tth_sample = np.full(cols.shape, np.nan, dtype=np.float64)
    phi_sample = np.full(cols.shape, np.nan, dtype=np.float64)
    bundle = resolved_projection["transform_bundle"]
    for idx in np.flatnonzero(base_valid):
        two_theta_value, phi_value = detector_pixel_to_caked_bin(
            bundle,
            float(cols[idx]),
            float(rows[idx]),
        )
        if (
            two_theta_value is None
            or phi_value is None
            or not np.isfinite(float(two_theta_value))
            or not np.isfinite(float(phi_value))
        ):
            continue
        tth_sample[idx] = float(two_theta_value)
        phi_sample[idx] = float(phi_value)

    visible = base_valid & np.isfinite(tth_sample) & np.isfinite(phi_sample)
    tth_min, tth_max = sorted((float(two_theta_limits[0]), float(two_theta_limits[1])))
    visible &= (tth_sample >= tth_min) & (tth_sample <= tth_max)
    if not np.any(visible):
        return tth, phi

    tth[visible] = tth_sample[visible]
    phi[visible] = phi_sample[visible]

    jump_limit = float(discontinuity_threshold_deg)
    if math.isfinite(jump_limit) and jump_limit > 0.0 and phi.size > 1:
        finite_pairs = (
            np.isfinite(tth[:-1])
            & np.isfinite(tth[1:])
            & np.isfinite(phi[:-1])
            & np.isfinite(phi[1:])
        )
        jumps = finite_pairs & (np.abs(phi[1:] - phi[:-1]) > jump_limit)
        if np.any(jumps):
            jump_idx = np.flatnonzero(jumps) + 1
            tth[jump_idx] = np.nan
            phi[jump_idx] = np.nan

    return tth, phi


def make_runtime_qr_cylinder_overlay_bindings_factory(
    *,
    ax: Any,
    overlay_artists: list[object],
    overlay_cache: dict[str, object],
    overlay_enabled_factory: object,
    get_active_entries: Callable[[], Sequence[dict[str, object]]],
    render_config_factory: object,
    ai_factory: object,
    get_detector_angular_maps: Callable[[Any], tuple[object, object]],
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, int]],
        tuple[float, float],
    ],
    get_caked_projection_context: Callable[[], Mapping[str, object] | None] | None = None,
    draw_idle_factory: object = None,
    set_status_text_factory: object = None,
) -> Callable[[], QrCylinderOverlayRuntimeBindings]:
    """Return a zero-arg factory for live Qr-cylinder overlay bindings."""

    def _build() -> QrCylinderOverlayRuntimeBindings:
        return QrCylinderOverlayRuntimeBindings(
            ax=ax,
            overlay_artists=overlay_artists,
            overlay_cache=overlay_cache,
            overlay_enabled_factory=overlay_enabled_factory,
            get_active_entries=get_active_entries,
            render_config_factory=render_config_factory,
            ai_factory=ai_factory,
            get_detector_angular_maps=get_detector_angular_maps,
            native_sim_to_display_coords=native_sim_to_display_coords,
            get_caked_projection_context=get_caked_projection_context,
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
        )

    return _build


def make_runtime_qr_cylinder_overlay_refresh_callback(
    bindings_factory: Callable[[], QrCylinderOverlayRuntimeBindings],
) -> Callable[..., None]:
    """Return one callback that refreshes the live Qr-cylinder overlay."""

    return lambda *, redraw=True, update_status=False: refresh_runtime_qr_cylinder_overlay(
        bindings_factory(),
        redraw=redraw,
        update_status=update_status,
    )


def make_runtime_qr_cylinder_overlay_toggle_callback(
    bindings_factory: Callable[[], QrCylinderOverlayRuntimeBindings],
) -> Callable[[], None]:
    """Return one callback that handles overlay checkbox toggles."""

    return lambda: toggle_runtime_qr_cylinder_overlay(bindings_factory())
