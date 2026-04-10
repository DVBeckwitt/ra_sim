"""Workflow helpers for GUI background-file management."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import background as gui_background
from . import controllers as gui_controllers
from . import views as gui_views


@dataclass
class BackgroundRuntimeBindings:
    """Runtime callbacks and shared state used by background-file workflows."""

    view_state: object
    background_state: object
    image_size: int
    display_rotate_k: int
    read_osc: Callable[[str], object]
    current_background_theta_values: Callable[[], Sequence[object]]
    background_theta_for_index: Callable[[int], object]
    geometry_fit_uses_shared_theta_offset: Callable[[], bool]
    geometry_manual_pairs_for_index: Callable[[int], Sequence[object]]
    geometry_manual_pair_group_count: Callable[[int], int]
    current_geometry_fit_background_indices: Callable[[], Sequence[object]]
    sync_background_runtime_state: Callable[[], None]
    replace_geometry_manual_pairs_by_background: Callable[[dict], None]
    invalidate_geometry_manual_pick_cache: Callable[[], None]
    clear_geometry_manual_undo_stack: Callable[[], None]
    clear_geometry_fit_undo_stack: Callable[[], None]
    set_geometry_manual_pick_mode: Callable[[bool], None]
    set_background_display_data: Callable[[object], None]
    update_background_slider_defaults: Callable[[object], None]
    sync_background_theta_controls: Callable[[], None]
    sync_geometry_fit_background_selection: Callable[[], None]
    clear_geometry_pick_artists: Callable[[], None]
    set_background_alpha: Callable[[float], None] | None = None
    sync_theta_initial_to_background: Callable[[int], None] | None = None
    render_current_geometry_manual_pairs: Callable[[], None] | None = None
    caked_view_active: Callable[[], bool] | None = None
    background_backend_debug_view_state: object | None = None
    mark_chi_square_dirty: Callable[[], None] | None = None
    refresh_chi_square_display: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    preempt_simulation_update: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None
    file_dialog_dir: object | None = None
    askopenfilenames: Callable[..., object] | None = None


@dataclass(frozen=True)
class BackgroundRuntimeCallbacks:
    """Bound callbacks for runtime background-file and backend debug workflows."""

    refresh_status: Callable[[], str]
    toggle_visibility: Callable[[], bool]
    load_files: Callable[[Sequence[object], int], dict[str, object]]
    browse_files: Callable[[], bool]
    switch_background: Callable[[], bool]
    refresh_backend_status: Callable[[], str]
    rotate_backend_minus_90: Callable[[], str]
    rotate_backend_plus_90: Callable[[], str]
    flip_backend_x: Callable[[], str]
    flip_backend_y: Callable[[], str]
    reset_backend_orientation: Callable[[], str]


@dataclass(frozen=True)
class BackgroundDisplayDefaults:
    """Derived defaults and slider ranges for background display controls."""

    min_candidate: float
    vmin_default: float
    vmax_default: float
    slider_min: float
    slider_max: float
    slider_step: float


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _set_status_text(set_status_text: Callable[[str], None] | None, text: str) -> None:
    if callable(set_status_text):
        set_status_text(str(text))


def _caked_view_active(caked_view_active: Callable[[], bool] | None) -> bool:
    if not callable(caked_view_active):
        return False
    try:
        return bool(caked_view_active())
    except Exception:
        return False


def _replace_list(target: list, values: Sequence[object] | None) -> list:
    target.clear()
    target.extend(list(values or ()))
    return target


def _normalize_runtime_list_attr(background_state, attr_name: str) -> list:
    values = list(getattr(background_state, attr_name, ()) or ())
    target = getattr(background_state, attr_name, None)
    if isinstance(target, list):
        target.clear()
        target.extend(values)
        return target
    setattr(background_state, attr_name, values)
    return values


def _finite_percentile(array, percentile: float, fallback: float) -> float:
    if array is None:
        return float(fallback)
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.nanpercentile(finite, percentile))


def apply_background_state_update(
    background_state,
    payload: dict[str, object],
):
    """Apply one background payload to the shared runtime state in place."""

    if "osc_files" in payload:
        _replace_list(background_state.osc_files, payload.get("osc_files"))
    if "background_images" in payload:
        _replace_list(
            background_state.background_images,
            payload.get("background_images"),
        )
    if "background_images_native" in payload:
        _replace_list(
            background_state.background_images_native,
            payload.get("background_images_native"),
        )
    if "background_images_display" in payload:
        _replace_list(
            background_state.background_images_display,
            payload.get("background_images_display"),
        )
    if "current_background_index" in payload:
        background_state.current_background_index = int(
            payload.get("current_background_index", 0)
        )
    if "current_background_image" in payload:
        background_state.current_background_image = payload.get(
            "current_background_image"
        )
    if "current_background_display" in payload:
        background_state.current_background_display = payload.get(
            "current_background_display"
        )
    return background_state


def normalize_background_runtime_state(
    background_state,
    *,
    file_paths_state: dict[str, object] | None = None,
):
    """Normalize shared background-runtime state after one mutation path."""

    _normalize_runtime_list_attr(background_state, "osc_files")
    _normalize_runtime_list_attr(background_state, "background_images")
    _normalize_runtime_list_attr(background_state, "background_images_native")
    _normalize_runtime_list_attr(background_state, "background_images_display")
    background_state.current_background_index = int(
        getattr(background_state, "current_background_index", 0)
    )
    background_state.visible = bool(getattr(background_state, "visible", True))
    background_state.backend_rotation_k = int(
        getattr(background_state, "backend_rotation_k", 0)
    )
    background_state.backend_flip_x = bool(
        getattr(background_state, "backend_flip_x", False)
    )
    background_state.backend_flip_y = bool(
        getattr(background_state, "backend_flip_y", False)
    )
    if isinstance(file_paths_state, dict):
        file_paths_state["simulation_background_osc_files"] = list(
            background_state.osc_files
        )
    return background_state


def _apply_background_cache_state_update(
    background_state,
    payload: dict[str, object],
):
    """Apply only background cache arrays from one lazy-read payload in place."""

    apply_background_state_update(
        background_state,
        {
            key: payload[key]
            for key in (
                "background_images",
                "background_images_native",
                "background_images_display",
            )
            if key in payload
        },
    )
    return background_state


def initialize_background_runtime_state(
    background_state,
    first_path: str,
    *,
    total_count: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    file_paths_state: dict[str, object] | None = None,
    visible: bool = True,
    backend_rotation_k: int = 3,
    backend_flip_x: bool = False,
    backend_flip_y: bool = False,
) -> dict[str, object]:
    """Initialize lazy background cache state and apply default runtime flags."""

    updated = gui_background.initialize_background_cache(
        str(first_path),
        total_count=int(total_count),
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    apply_background_state_update(background_state, updated)
    background_state.visible = bool(visible)
    background_state.backend_rotation_k = int(backend_rotation_k)
    background_state.backend_flip_x = bool(backend_flip_x)
    background_state.backend_flip_y = bool(backend_flip_y)
    normalize_background_runtime_state(
        background_state,
        file_paths_state=file_paths_state,
    )
    return updated


def resolve_background_display_defaults(image) -> BackgroundDisplayDefaults:
    """Return initial background display defaults from one image."""

    min_candidate = _finite_percentile(image, 1.0, 0.0)
    vmin_default = 0.0
    _, vmax_default = gui_controllers.ensure_display_intensity_range(
        vmin_default,
        _finite_percentile(image, 99.0, 1.0),
    )
    slider_min = min(min_candidate, 0.0)
    slider_max = max(
        _finite_percentile(image, 100.0, vmax_default),
        vmax_default,
        slider_min + 1.0,
    )
    slider_step = max(
        (max(vmax_default * 5.0, slider_min + 1.0) - slider_min) / 500.0,
        0.01,
    )
    return BackgroundDisplayDefaults(
        min_candidate=float(min_candidate),
        vmin_default=float(vmin_default),
        vmax_default=float(vmax_default),
        slider_min=float(slider_min),
        slider_max=float(slider_max),
        slider_step=float(slider_step),
    )


def apply_background_transparency(
    view_state,
    *,
    background_display,
) -> float | None:
    """Apply the current background transparency control to the display artist."""

    background_transparency_var = getattr(view_state, "background_transparency_var", None)
    if background_transparency_var is None:
        return None
    transparency = max(0.0, min(1.0, float(background_transparency_var.get())))
    alpha = 1.0 - transparency
    background_display.set_alpha(alpha)
    return float(alpha)


def apply_background_limits(
    display_controls_state,
    display_controls_view_state,
    *,
    background_display,
    draw_idle: Callable[[], None] | None = None,
) -> bool:
    """Apply the live background display range from the current control values."""

    background_min_var = getattr(display_controls_view_state, "background_min_var", None)
    background_max_var = getattr(display_controls_view_state, "background_max_var", None)
    if background_min_var is None or background_max_var is None:
        return False
    min_val = background_min_var.get()
    max_val = background_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1.0e-6, 1.0e-6)
        display_controls_state.suppress_background_limit_callback = True
        try:
            background_min_var.set(max_val - adjustment)
        finally:
            display_controls_state.suppress_background_limit_callback = False
        return False
    display_controls_state.background_limits_user_override = True
    background_display.set_clim(min_val, max_val)
    apply_background_transparency(
        display_controls_view_state,
        background_display=background_display,
    )
    if callable(draw_idle):
        draw_idle()
    return True


def update_background_slider_defaults(
    display_controls_state,
    display_controls_view_state,
    *,
    background_display,
    image,
    reset_override: bool = False,
) -> tuple[float, float] | None:
    """Refresh background slider bounds/defaults from one image."""

    background_max_var = getattr(display_controls_view_state, "background_max_var", None)
    background_min_var = getattr(display_controls_view_state, "background_min_var", None)
    background_min_slider = getattr(
        display_controls_view_state,
        "background_min_slider",
        None,
    )
    background_max_slider = getattr(
        display_controls_view_state,
        "background_max_slider",
        None,
    )
    if (
        image is None
        or background_max_var is None
        or background_min_var is None
        or background_min_slider is None
        or background_max_slider is None
    ):
        return None
    min_candidate = _finite_percentile(image, 1.0, 0.0)
    max_candidate = _finite_percentile(image, 99.0, background_max_var.get())
    min_candidate, max_candidate = gui_controllers.ensure_display_intensity_range(
        min_candidate,
        max_candidate,
    )
    max_limit = max(
        _finite_percentile(image, 100.0, max_candidate),
        max_candidate,
        1.0,
    )
    slider_from = min(float(background_min_slider.cget("from")), min_candidate, 0.0)
    slider_to = float(max_limit)
    if (
        display_controls_state.background_limits_user_override
        and not reset_override
    ):
        slider_to = max(slider_to, float(background_max_var.get()))
    background_min_slider.configure(from_=slider_from, to=slider_to)
    background_max_slider.configure(from_=slider_from, to=slider_to)
    display_controls_state.suppress_background_limit_callback = True
    try:
        if reset_override or not display_controls_state.background_limits_user_override:
            min_value = 0.0
            max_value = max_candidate
        else:
            min_value = float(background_min_var.get())
            max_value = float(background_max_var.get())
            min_value = min(max(min_value, slider_from), slider_to)
            max_value = min(max(max_value, slider_from), slider_to)
        min_value, max_value = gui_controllers.ensure_display_intensity_range(
            min_value,
            max_value,
        )
        background_min_var.set(min_value)
        background_max_var.set(max_value)
    finally:
        display_controls_state.suppress_background_limit_callback = False
    background_display.set_clim(min_value, max_value)
    if reset_override:
        display_controls_state.background_limits_user_override = False
    return float(min_value), float(max_value)


def load_background_files(
    background_state,
    file_paths: Sequence[object],
    *,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    select_index: int = 0,
) -> dict[str, object]:
    """Load background files and apply the resulting state payload in place."""

    updated = gui_background.load_background_files(
        file_paths,
        image_size=image_size,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
        select_index=select_index,
    )
    apply_background_state_update(background_state, updated)
    return updated


def load_background_image_by_index(
    background_state,
    index: int,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Load one background image lazily and apply cache updates in place."""

    updated = gui_background.load_background_image_by_index(
        int(index),
        osc_files=background_state.osc_files,
        background_images=background_state.background_images,
        background_images_native=background_state.background_images_native,
        background_images_display=background_state.background_images_display,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    _apply_background_cache_state_update(background_state, updated)
    return updated


def get_current_background_native(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Resolve the current native background and apply cache updates in place."""

    updated = gui_background.get_current_background_native(
        osc_files=background_state.osc_files,
        background_images=background_state.background_images,
        background_images_native=background_state.background_images_native,
        background_images_display=background_state.background_images_display,
        current_background_index=background_state.current_background_index,
        current_background_image=background_state.current_background_image,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    _apply_background_cache_state_update(background_state, updated)
    return updated


def get_current_background_display(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Resolve the current display background and apply cache updates in place."""

    updated = gui_background.get_current_background_display(
        osc_files=background_state.osc_files,
        background_images=background_state.background_images,
        background_images_native=background_state.background_images_native,
        background_images_display=background_state.background_images_display,
        current_background_index=background_state.current_background_index,
        current_background_image=background_state.current_background_image,
        current_background_display=background_state.current_background_display,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    _apply_background_cache_state_update(background_state, updated)
    return updated


def switch_background(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Advance to the next background image and apply it in place."""

    updated = gui_background.switch_background(
        osc_files=background_state.osc_files,
        background_images=background_state.background_images,
        background_images_native=background_state.background_images_native,
        background_images_display=background_state.background_images_display,
        current_background_index=background_state.current_background_index,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    apply_background_state_update(background_state, updated)
    return updated


def background_file_dialog_initial_dir(
    osc_files: Sequence[object],
    current_background_index: int,
    default_dir: object,
) -> str:
    """Return the preferred starting directory for the background file dialog."""

    try:
        if osc_files:
            current_path = Path(
                str(osc_files[int(current_background_index)])
            ).expanduser()
            return str(current_path.parent)
    except Exception:
        pass
    return str(default_dir)


def build_background_file_status_text(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_base: float | None = None,
    theta_effective: float | None = None,
    use_shared_theta_offset: bool = False,
    pair_count: int = 0,
    group_count: int = 0,
    sigma_values: Sequence[object] | None = None,
    fit_indices: Sequence[object] | None = None,
) -> str:
    """Build the background-file status line shown in the GUI."""

    if not osc_files:
        return "Background: no files loaded"

    idx = int(current_background_index) % len(osc_files)
    current_name = Path(str(osc_files[idx])).name
    status_text = f"Background {idx + 1}/{len(osc_files)}: {current_name}"

    theta_base_finite = theta_base is not None and np.isfinite(float(theta_base))
    theta_effective_finite = theta_effective is not None and np.isfinite(
        float(theta_effective)
    )
    if theta_base_finite:
        if use_shared_theta_offset and theta_effective_finite:
            status_text += (
                f" | theta_i={float(theta_base):.4f}°"
                f" | theta={float(theta_effective):.4f}°"
            )
        else:
            status_text += f" | theta={float(theta_base):.4f}°"

    pair_count_i = max(0, int(pair_count))
    group_count_i = max(0, int(group_count))
    if pair_count_i > 0:
        status_text += f" | manual={group_count_i} groups/{pair_count_i} pts"
        finite_sigma_values = [
            float(value)
            for value in sigma_values or ()
            if np.isfinite(float(value))
        ]
        if finite_sigma_values:
            status_text += (
                f" | sigma~{float(np.median(np.asarray(finite_sigma_values, dtype=float))):.2f}px"
            )

    normalized_fit_indices: list[int] = []
    for raw_idx in fit_indices or ():
        try:
            normalized_fit_indices.append(int(raw_idx))
        except Exception:
            continue
    if normalized_fit_indices:
        if len(normalized_fit_indices) > 1:
            status_text += f" | fit={len(normalized_fit_indices)} backgrounds"
        elif len(osc_files) > 1:
            status_text += f" | fit=bg {normalized_fit_indices[0] + 1}"

    return status_text


def background_backend_status_text(background_state) -> str:
    """Build the backend-background debug status string from shared state."""

    return (
        f"k={int(getattr(background_state, 'backend_rotation_k', 0)) % 4} "
        f"flip_x={bool(getattr(background_state, 'backend_flip_x', False))} "
        f"flip_y={bool(getattr(background_state, 'backend_flip_y', False))}"
    )


def background_alpha_for_visibility(
    visible: bool,
    *,
    visible_alpha: float = 0.5,
    hidden_alpha: float = 1.0,
) -> float:
    """Return the display alpha used for the current background visibility."""

    return float(visible_alpha if bool(visible) else hidden_alpha)


def set_background_backend_status_from_state(
    *,
    view_state,
    background_state,
) -> str:
    """Refresh the backend-background debug status label from shared state."""

    status_text = background_backend_status_text(background_state)
    gui_views.set_background_backend_status_text(view_state, status_text)
    return status_text


def set_background_file_status_from_state(
    *,
    view_state,
    background_state,
    current_background_theta_values: Callable[[], Sequence[object]],
    background_theta_for_index: Callable[[int], object],
    geometry_fit_uses_shared_theta_offset: Callable[[], bool],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[object]],
    geometry_manual_pair_group_count: Callable[[int], int],
    current_geometry_fit_background_indices: Callable[[], Sequence[object]],
) -> str:
    """Refresh the GUI background-file status line from shared runtime state."""

    if getattr(view_state, "background_file_status_var", None) is None:
        return ""

    theta_base = None
    theta_effective = None
    use_shared_theta_offset = False
    try:
        theta_values = list(current_background_theta_values())
        if theta_values:
            idx = int(background_state.current_background_index) % max(
                1,
                len(background_state.osc_files),
            )
            theta_base = float(theta_values[idx])
            theta_effective = float(background_theta_for_index(idx))
            use_shared_theta_offset = bool(geometry_fit_uses_shared_theta_offset())
    except Exception:
        pass

    pair_count = 0
    group_count = 0
    sigma_values: list[float] = []
    try:
        idx = int(background_state.current_background_index) % max(
            1,
            len(background_state.osc_files),
        )
        pair_rows = list(geometry_manual_pairs_for_index(idx) or ())
        pair_count = len(pair_rows)
        group_count = int(geometry_manual_pair_group_count(idx))
        for entry in pair_rows:
            if not isinstance(entry, Mapping):
                continue
            try:
                sigma_value = float(entry.get("sigma_px", np.nan))
            except Exception:
                continue
            if np.isfinite(sigma_value):
                sigma_values.append(sigma_value)
    except Exception:
        pass

    fit_indices: list[int] = []
    try:
        fit_indices = [
            int(raw_idx)
            for raw_idx in current_geometry_fit_background_indices() or ()
        ]
    except Exception:
        fit_indices = []

    status_text = build_background_file_status_text(
        osc_files=background_state.osc_files,
        current_background_index=background_state.current_background_index,
        theta_base=theta_base,
        theta_effective=theta_effective,
        use_shared_theta_offset=use_shared_theta_offset,
        pair_count=pair_count,
        group_count=group_count,
        sigma_values=sigma_values,
        fit_indices=fit_indices,
    )
    gui_views.set_background_file_status_text(view_state, status_text)
    return status_text


def toggle_background_visibility_with_side_effects(
    background_state,
    *,
    sync_background_runtime_state: Callable[[], None],
    set_background_alpha: Callable[[float], None] | None,
    schedule_update: Callable[[], None],
    visible_alpha: float = 0.5,
    hidden_alpha: float = 1.0,
) -> bool:
    """Flip background visibility and apply the dependent redraw side effects."""

    background_state.visible = not bool(getattr(background_state, "visible", True))
    sync_background_runtime_state()
    if callable(set_background_alpha):
        set_background_alpha(
            background_alpha_for_visibility(
                background_state.visible,
                visible_alpha=visible_alpha,
                hidden_alpha=hidden_alpha,
            )
        )
    schedule_update()
    return bool(background_state.visible)


def make_runtime_background_bindings_factory(
    *,
    view_state,
    background_state,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    current_background_theta_values: Callable[[], Sequence[object]],
    background_theta_for_index: Callable[[int], object],
    geometry_fit_uses_shared_theta_offset: Callable[[], bool],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[object]],
    geometry_manual_pair_group_count: Callable[[int], int],
    current_geometry_fit_background_indices: Callable[[], Sequence[object]],
    sync_background_runtime_state: Callable[[], None],
    replace_geometry_manual_pairs_by_background: Callable[[dict], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    set_geometry_manual_pick_mode: Callable[[bool], None],
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    sync_background_theta_controls: Callable[[], None],
    sync_geometry_fit_background_selection: Callable[[], None],
    clear_geometry_pick_artists: Callable[[], None],
    set_background_alpha: Callable[[float], None] | None = None,
    sync_theta_initial_to_background: Callable[[int], None] | None = None,
    render_current_geometry_manual_pairs: Callable[[], None] | None = None,
    caked_view_active: Callable[[], bool] | None = None,
    background_backend_debug_view_state: object | None = None,
    mark_chi_square_dirty: Callable[[], None] | None = None,
    refresh_chi_square_display: Callable[[], None] | None = None,
    schedule_update_factory: object | None = None,
    preempt_simulation_update_factory: object | None = None,
    set_status_text_factory: object | None = None,
    file_dialog_dir_factory: object | None = None,
    askopenfilenames: Callable[..., object] | None = None,
) -> Callable[[], BackgroundRuntimeBindings]:
    """Return a zero-arg factory for live background runtime bindings."""

    def _build_bindings() -> BackgroundRuntimeBindings:
        return BackgroundRuntimeBindings(
            view_state=view_state,
            background_state=background_state,
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            read_osc=read_osc,
            current_background_theta_values=current_background_theta_values,
            background_theta_for_index=background_theta_for_index,
            geometry_fit_uses_shared_theta_offset=geometry_fit_uses_shared_theta_offset,
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            geometry_manual_pair_group_count=geometry_manual_pair_group_count,
            current_geometry_fit_background_indices=current_geometry_fit_background_indices,
            sync_background_runtime_state=sync_background_runtime_state,
            replace_geometry_manual_pairs_by_background=replace_geometry_manual_pairs_by_background,
            invalidate_geometry_manual_pick_cache=invalidate_geometry_manual_pick_cache,
            clear_geometry_manual_undo_stack=clear_geometry_manual_undo_stack,
            clear_geometry_fit_undo_stack=clear_geometry_fit_undo_stack,
            set_geometry_manual_pick_mode=set_geometry_manual_pick_mode,
            set_background_display_data=set_background_display_data,
            set_background_alpha=set_background_alpha,
            update_background_slider_defaults=update_background_slider_defaults,
            sync_background_theta_controls=sync_background_theta_controls,
            sync_geometry_fit_background_selection=sync_geometry_fit_background_selection,
            clear_geometry_pick_artists=clear_geometry_pick_artists,
            sync_theta_initial_to_background=sync_theta_initial_to_background,
            render_current_geometry_manual_pairs=render_current_geometry_manual_pairs,
            caked_view_active=caked_view_active,
            background_backend_debug_view_state=background_backend_debug_view_state,
            mark_chi_square_dirty=mark_chi_square_dirty,
            refresh_chi_square_display=refresh_chi_square_display,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            preempt_simulation_update=_resolve_runtime_value(
                preempt_simulation_update_factory
            ),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
            file_dialog_dir=_resolve_runtime_value(file_dialog_dir_factory),
            askopenfilenames=askopenfilenames,
        )

    return _build_bindings


def refresh_runtime_background_file_status(
    bindings: BackgroundRuntimeBindings,
) -> str:
    """Refresh the runtime background-file status line from live bindings."""

    return set_background_file_status_from_state(
        view_state=bindings.view_state,
        background_state=bindings.background_state,
        current_background_theta_values=bindings.current_background_theta_values,
        background_theta_for_index=bindings.background_theta_for_index,
        geometry_fit_uses_shared_theta_offset=bindings.geometry_fit_uses_shared_theta_offset,
        geometry_manual_pairs_for_index=bindings.geometry_manual_pairs_for_index,
        geometry_manual_pair_group_count=bindings.geometry_manual_pair_group_count,
        current_geometry_fit_background_indices=bindings.current_geometry_fit_background_indices,
    )


def refresh_runtime_background_backend_status(
    bindings: BackgroundRuntimeBindings,
) -> str:
    """Refresh the runtime backend-background debug status from live bindings."""

    if bindings.background_backend_debug_view_state is None:
        return background_backend_status_text(bindings.background_state)
    return set_background_backend_status_from_state(
        view_state=bindings.background_backend_debug_view_state,
        background_state=bindings.background_state,
    )


def apply_background_payload_with_side_effects(
    background_state,
    payload: dict[str, object],
    *,
    sync_background_runtime_state: Callable[[], None],
    replace_geometry_manual_pairs_by_background: Callable[[dict], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    set_geometry_manual_pick_mode: Callable[[bool], None],
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    sync_background_theta_controls: Callable[[], None],
    sync_geometry_fit_background_selection: Callable[[], None],
    clear_geometry_pick_artists: Callable[[], None],
    refresh_background_file_status: Callable[[], None],
    schedule_update: Callable[[], None] | None,
) -> dict[str, object]:
    """Apply a loaded background payload and run the dependent GUI side effects."""

    apply_background_state_update(background_state, payload)
    sync_background_runtime_state()
    replace_geometry_manual_pairs_by_background({})
    invalidate_geometry_manual_pick_cache()
    clear_geometry_manual_undo_stack()
    clear_geometry_fit_undo_stack()
    set_geometry_manual_pick_mode(False)

    current_display = background_state.current_background_display
    set_background_display_data(current_display)
    update_background_slider_defaults(current_display)
    sync_background_theta_controls()
    sync_geometry_fit_background_selection()
    clear_geometry_pick_artists()
    refresh_background_file_status()
    if callable(schedule_update):
        schedule_update()
    return payload


def load_background_files_with_side_effects(
    background_state,
    file_paths: Sequence[object],
    *,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    sync_background_runtime_state: Callable[[], None],
    replace_geometry_manual_pairs_by_background: Callable[[dict], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    set_geometry_manual_pick_mode: Callable[[bool], None],
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    sync_background_theta_controls: Callable[[], None],
    sync_geometry_fit_background_selection: Callable[[], None],
    clear_geometry_pick_artists: Callable[[], None],
    refresh_background_file_status: Callable[[], None],
    schedule_update: Callable[[], None],
    select_index: int = 0,
) -> dict[str, object]:
    """Load background files and apply the dependent GUI side effects."""

    updated = load_background_files(
        background_state,
        file_paths,
        image_size=image_size,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
        select_index=select_index,
    )
    return apply_background_payload_with_side_effects(
        background_state,
        updated,
        sync_background_runtime_state=sync_background_runtime_state,
        replace_geometry_manual_pairs_by_background=replace_geometry_manual_pairs_by_background,
        invalidate_geometry_manual_pick_cache=invalidate_geometry_manual_pick_cache,
        clear_geometry_manual_undo_stack=clear_geometry_manual_undo_stack,
        clear_geometry_fit_undo_stack=clear_geometry_fit_undo_stack,
        set_geometry_manual_pick_mode=set_geometry_manual_pick_mode,
        set_background_display_data=set_background_display_data,
        update_background_slider_defaults=update_background_slider_defaults,
        sync_background_theta_controls=sync_background_theta_controls,
        sync_geometry_fit_background_selection=sync_geometry_fit_background_selection,
        clear_geometry_pick_artists=clear_geometry_pick_artists,
        refresh_background_file_status=refresh_background_file_status,
        schedule_update=schedule_update,
    )


def toggle_runtime_background_visibility(
    bindings: BackgroundRuntimeBindings,
    *,
    visible_alpha: float = 0.5,
    hidden_alpha: float = 1.0,
) -> bool:
    """Toggle the runtime background visibility from live bindings."""

    return toggle_background_visibility_with_side_effects(
        bindings.background_state,
        sync_background_runtime_state=bindings.sync_background_runtime_state,
        set_background_alpha=bindings.set_background_alpha,
        schedule_update=bindings.schedule_update or (lambda: None),
        visible_alpha=float(visible_alpha),
        hidden_alpha=float(hidden_alpha),
    )


def rotate_background_backend_with_side_effects(
    background_state,
    *,
    delta_k: int,
    sync_background_runtime_state: Callable[[], None],
    refresh_background_backend_status: Callable[[], str],
    mark_chi_square_dirty: Callable[[], None],
    refresh_chi_square_display: Callable[[], None],
    schedule_update: Callable[[], None],
) -> str:
    """Rotate the backend background orientation and apply dependent side effects."""

    background_state.backend_rotation_k = (
        int(getattr(background_state, "backend_rotation_k", 0)) + int(delta_k)
    ) % 4
    sync_background_runtime_state()
    status_text = refresh_background_backend_status()
    mark_chi_square_dirty()
    refresh_chi_square_display()
    schedule_update()
    return status_text


def toggle_background_backend_flip_with_side_effects(
    background_state,
    *,
    axis: str,
    sync_background_runtime_state: Callable[[], None],
    refresh_background_backend_status: Callable[[], str],
    mark_chi_square_dirty: Callable[[], None],
    refresh_chi_square_display: Callable[[], None],
    schedule_update: Callable[[], None],
) -> str:
    """Flip one backend background axis and apply dependent side effects."""

    normalized_axis = str(axis or "").lower()
    if normalized_axis == "x":
        background_state.backend_flip_x = not bool(
            getattr(background_state, "backend_flip_x", False)
        )
    elif normalized_axis == "y":
        background_state.backend_flip_y = not bool(
            getattr(background_state, "backend_flip_y", False)
        )
    sync_background_runtime_state()
    status_text = refresh_background_backend_status()
    mark_chi_square_dirty()
    refresh_chi_square_display()
    schedule_update()
    return status_text


def reset_background_backend_orientation_with_side_effects(
    background_state,
    *,
    sync_background_runtime_state: Callable[[], None],
    refresh_background_backend_status: Callable[[], str],
    mark_chi_square_dirty: Callable[[], None],
    refresh_chi_square_display: Callable[[], None],
    schedule_update: Callable[[], None],
) -> str:
    """Reset backend background orientation state and apply dependent side effects."""

    background_state.backend_rotation_k = 0
    background_state.backend_flip_x = False
    background_state.backend_flip_y = False
    sync_background_runtime_state()
    status_text = refresh_background_backend_status()
    mark_chi_square_dirty()
    refresh_chi_square_display()
    schedule_update()
    return status_text


def load_runtime_background_files(
    bindings: BackgroundRuntimeBindings,
    file_paths: Sequence[object],
    *,
    select_index: int = 0,
) -> dict[str, object]:
    """Load runtime background files and apply the configured side effects."""

    return load_background_files_with_side_effects(
        bindings.background_state,
        file_paths,
        image_size=int(bindings.image_size),
        display_rotate_k=int(bindings.display_rotate_k),
        read_osc=bindings.read_osc,
        sync_background_runtime_state=bindings.sync_background_runtime_state,
        replace_geometry_manual_pairs_by_background=bindings.replace_geometry_manual_pairs_by_background,
        invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
        clear_geometry_manual_undo_stack=bindings.clear_geometry_manual_undo_stack,
        clear_geometry_fit_undo_stack=bindings.clear_geometry_fit_undo_stack,
        set_geometry_manual_pick_mode=bindings.set_geometry_manual_pick_mode,
        set_background_display_data=bindings.set_background_display_data,
        update_background_slider_defaults=bindings.update_background_slider_defaults,
        sync_background_theta_controls=bindings.sync_background_theta_controls,
        sync_geometry_fit_background_selection=bindings.sync_geometry_fit_background_selection,
        clear_geometry_pick_artists=bindings.clear_geometry_pick_artists,
        refresh_background_file_status=lambda: refresh_runtime_background_file_status(
            bindings
        ),
        schedule_update=bindings.schedule_update or (lambda: None),
        select_index=int(select_index),
    )


def switch_background_with_side_effects(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    sync_background_runtime_state: Callable[[], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    sync_background_theta_controls: Callable[[], None],
    sync_geometry_fit_background_selection: Callable[[], None],
    sync_theta_initial_to_background: Callable[[int], None] | None,
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    refresh_background_file_status: Callable[[], None],
    render_current_geometry_manual_pairs: Callable[[], None],
    schedule_update: Callable[[], None],
    preempt_simulation_update: Callable[[], None] | None = None,
    caked_view_active: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Switch backgrounds and apply the dependent GUI side effects."""

    if callable(preempt_simulation_update):
        preempt_simulation_update()
    updated = switch_background(
        background_state,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    sync_background_runtime_state()
    invalidate_geometry_manual_pick_cache()
    clear_geometry_manual_undo_stack()
    clear_geometry_fit_undo_stack()

    # In caked mode, keep the existing caked raster onscreen until the
    # recomputed caked background is ready.
    defer_immediate_canvas_refresh = _caked_view_active(caked_view_active)
    current_display = background_state.current_background_display
    if not defer_immediate_canvas_refresh:
        set_background_display_data(current_display)
        update_background_slider_defaults(current_display)
    if callable(sync_background_theta_controls):
        sync_background_theta_controls()
    elif sync_theta_initial_to_background is not None:
        sync_theta_initial_to_background(int(background_state.current_background_index))
    if callable(sync_geometry_fit_background_selection):
        sync_geometry_fit_background_selection()
    if not defer_immediate_canvas_refresh:
        render_current_geometry_manual_pairs()
    refresh_background_file_status()
    schedule_update()
    return updated


def switch_runtime_background(
    bindings: BackgroundRuntimeBindings,
) -> bool:
    """Advance the runtime background image and report status text on failure."""

    if not bindings.background_state.osc_files:
        _set_status_text(bindings.set_status_text, "No background images loaded.")
        return False

    try:
        switch_background_with_side_effects(
            bindings.background_state,
            display_rotate_k=int(bindings.display_rotate_k),
            read_osc=bindings.read_osc,
            sync_background_runtime_state=bindings.sync_background_runtime_state,
            invalidate_geometry_manual_pick_cache=bindings.invalidate_geometry_manual_pick_cache,
            clear_geometry_manual_undo_stack=bindings.clear_geometry_manual_undo_stack,
            clear_geometry_fit_undo_stack=bindings.clear_geometry_fit_undo_stack,
            sync_background_theta_controls=bindings.sync_background_theta_controls,
            sync_geometry_fit_background_selection=(
                bindings.sync_geometry_fit_background_selection
            ),
            sync_theta_initial_to_background=bindings.sync_theta_initial_to_background,
            set_background_display_data=bindings.set_background_display_data,
            update_background_slider_defaults=bindings.update_background_slider_defaults,
            refresh_background_file_status=lambda: refresh_runtime_background_file_status(
                bindings
            ),
            render_current_geometry_manual_pairs=(
                bindings.render_current_geometry_manual_pairs or (lambda: None)
            ),
            schedule_update=bindings.schedule_update or (lambda: None),
            preempt_simulation_update=bindings.preempt_simulation_update,
            caked_view_active=bindings.caked_view_active,
        )
    except Exception as exc:
        _set_status_text(
            bindings.set_status_text,
            f"Failed to switch background: {exc}",
        )
        return False
    return True


def rotate_runtime_background_backend(
    bindings: BackgroundRuntimeBindings,
    *,
    delta_k: int,
) -> str:
    """Rotate the runtime backend background orientation from live bindings."""

    return rotate_background_backend_with_side_effects(
        bindings.background_state,
        delta_k=int(delta_k),
        sync_background_runtime_state=bindings.sync_background_runtime_state,
        refresh_background_backend_status=lambda: refresh_runtime_background_backend_status(
            bindings
        ),
        mark_chi_square_dirty=bindings.mark_chi_square_dirty or (lambda: None),
        refresh_chi_square_display=bindings.refresh_chi_square_display or (lambda: None),
        schedule_update=bindings.schedule_update or (lambda: None),
    )


def toggle_runtime_background_backend_flip(
    bindings: BackgroundRuntimeBindings,
    *,
    axis: str,
) -> str:
    """Flip one runtime backend background axis from live bindings."""

    return toggle_background_backend_flip_with_side_effects(
        bindings.background_state,
        axis=axis,
        sync_background_runtime_state=bindings.sync_background_runtime_state,
        refresh_background_backend_status=lambda: refresh_runtime_background_backend_status(
            bindings
        ),
        mark_chi_square_dirty=bindings.mark_chi_square_dirty or (lambda: None),
        refresh_chi_square_display=bindings.refresh_chi_square_display or (lambda: None),
        schedule_update=bindings.schedule_update or (lambda: None),
    )


def reset_runtime_background_backend_orientation(
    bindings: BackgroundRuntimeBindings,
) -> str:
    """Reset the runtime backend background orientation from live bindings."""

    return reset_background_backend_orientation_with_side_effects(
        bindings.background_state,
        sync_background_runtime_state=bindings.sync_background_runtime_state,
        refresh_background_backend_status=lambda: refresh_runtime_background_backend_status(
            bindings
        ),
        mark_chi_square_dirty=bindings.mark_chi_square_dirty or (lambda: None),
        refresh_chi_square_display=bindings.refresh_chi_square_display or (lambda: None),
        schedule_update=bindings.schedule_update or (lambda: None),
    )


def browse_runtime_background_files(
    bindings: BackgroundRuntimeBindings,
) -> bool:
    """Open the runtime background-file picker and load the chosen files."""

    if not callable(bindings.askopenfilenames):
        _set_status_text(bindings.set_status_text, "Background file picker unavailable.")
        return False

    initial_dir = background_file_dialog_initial_dir(
        bindings.background_state.osc_files,
        bindings.background_state.current_background_index,
        bindings.file_dialog_dir or Path.cwd(),
    )
    selected = bindings.askopenfilenames(
        title="Select Background OSC Files",
        initialdir=initial_dir,
        filetypes=[("OSC files", "*.osc *.OSC"), ("All files", "*.*")],
    )
    if not selected:
        return False

    try:
        load_runtime_background_files(bindings, selected, select_index=0)
        _set_status_text(
            bindings.set_status_text,
            f"Loaded {len(bindings.background_state.osc_files)} background file(s).",
        )
    except Exception as exc:
        _set_status_text(
            bindings.set_status_text,
            f"Failed to load background files: {exc}",
        )
        return False
    return True


def make_runtime_background_callbacks(
    bindings_factory: Callable[[], BackgroundRuntimeBindings],
) -> BackgroundRuntimeCallbacks:
    """Return bound callbacks for the runtime background and backend debug workflow."""

    return BackgroundRuntimeCallbacks(
        refresh_status=lambda: refresh_runtime_background_file_status(
            bindings_factory()
        ),
        toggle_visibility=lambda: toggle_runtime_background_visibility(
            bindings_factory()
        ),
        load_files=lambda file_paths, select_index=0: load_runtime_background_files(
            bindings_factory(),
            file_paths,
            select_index=int(select_index),
        ),
        browse_files=lambda: browse_runtime_background_files(bindings_factory()),
        switch_background=lambda: switch_runtime_background(bindings_factory()),
        refresh_backend_status=lambda: refresh_runtime_background_backend_status(
            bindings_factory()
        ),
        rotate_backend_minus_90=lambda: rotate_runtime_background_backend(
            bindings_factory(),
            delta_k=-1,
        ),
        rotate_backend_plus_90=lambda: rotate_runtime_background_backend(
            bindings_factory(),
            delta_k=1,
        ),
        flip_backend_x=lambda: toggle_runtime_background_backend_flip(
            bindings_factory(),
            axis="x",
        ),
        flip_backend_y=lambda: toggle_runtime_background_backend_flip(
            bindings_factory(),
            axis="y",
        ),
        reset_backend_orientation=lambda: reset_runtime_background_backend_orientation(
            bindings_factory()
        ),
    )
