from pathlib import Path
import numpy as np
from types import SimpleNamespace

from ra_sim.gui import integration_range_drag, state
from ra_sim.simulation import exact_cake_portable


SOURCE_PATH = Path(integration_range_drag.__file__)


def _qr_drag_config() -> (
    integration_range_drag.gui_qr_cylinder_overlay.QrCylinderOverlayRenderConfig
):
    return integration_range_drag.gui_qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=True,
        image_size=64,
        display_rotate_k=0,
        center_col=10.0,
        center_row=11.0,
        distance_cor_to_detector=123.0,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        pixel_size_m=1.0e-4,
        wavelength=1.0,
        n2=1.0 + 0.0j,
    )


def _qr_drag_projection_context(
    *,
    matrix: np.ndarray,
    detector_shape: tuple[int, int] = (2, 3),
    radial_axis: np.ndarray | None = None,
    azimuth_axis: np.ndarray | None = None,
) -> dict[str, object]:
    radial = (
        np.asarray(radial_axis, dtype=float)
        if radial_axis is not None
        else np.asarray([1.0, 2.0, 3.0], dtype=float)
    )
    azimuth = (
        np.asarray(azimuth_axis, dtype=float)
        if azimuth_axis is not None
        else np.asarray([-80.0, 0.0, 80.0], dtype=float)
    )
    raw_azimuth = np.asarray(
        exact_cake_portable.gui_phi_to_raw_phi(azimuth),
        dtype=float,
    )
    bundle = integration_range_drag.gui_qr_cylinder_overlay.CakeTransformBundle(
        detector_shape=detector_shape,
        radial_deg=radial,
        raw_azimuth_deg=raw_azimuth,
        gui_azimuth_deg=np.asarray(
            exact_cake_portable.raw_phi_to_gui_phi(raw_azimuth),
            dtype=float,
        ),
        lut=SimpleNamespace(matrix=np.asarray(matrix, dtype=np.float32)),
    )
    return {
        "detector_shape": detector_shape,
        "radial_axis": radial,
        "azimuth_axis": azimuth,
        "raw_azimuth_axis": raw_azimuth,
        "transform_bundle": bundle,
    }


def test_drag_module_removes_fast_viewer_overlay_state() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "_set_fast_viewer_overlay_state" not in source
    assert "_fast_viewer_curve_specs" not in source
    assert "_fast_viewer_suppress_overlay_image" not in source
    assert "_fast_viewer_overlay_version" not in source


class _FakeVar:
    def __init__(self, value=0.0):
        self._value = value
        self.trace_calls = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def trace_add(self, mode, callback):
        self.trace_calls.append((mode, callback))
        return f"trace-{len(self.trace_calls)}"


class _FakeSlider:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.state = "normal"

    def cget(self, key):
        if key == "from":
            return self.minimum
        if key == "to":
            return self.maximum
        raise KeyError(key)

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)

    def config(self, **kwargs):
        self.configure(**kwargs)


class _FakeEntry:
    def __init__(self, *, textvariable=None, state="normal"):
        self.textvariable = textvariable
        self.state = state
        self.selected = []
        self.seen_index = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)

    def config(self, **kwargs):
        self.configure(**kwargs)

    def selection_clear(self, _start, _end):
        self.selected = []

    def selection_set(self, idx):
        if int(idx) not in self.selected:
            self.selected.append(int(idx))

    def curselection(self):
        return tuple(self.selected)

    def see(self, idx):
        self.seen_index = int(idx)


class _FakeDebouncedListbox(_FakeEntry):
    def __init__(self, root, *, state="normal"):
        super().__init__(state=state)
        self.root = root
        self.exists = True

    def after(self, delay, callback):
        return self.root.after(delay, callback)

    def after_cancel(self, token):
        self.root.after_cancel(token)

    def winfo_exists(self):
        return bool(self.exists)


class _FakeRect:
    def __init__(self, xy=None, width=None, height=None, **kwargs):
        self.init_xy = tuple(xy) if xy is not None else None
        self.init_width = None if width is None else float(width)
        self.init_height = None if height is None else float(height)
        self.init_kwargs = dict(kwargs)
        self.xy = self.init_xy
        self.width = self.init_width
        self.height = self.init_height
        self.visible = False

    def set_xy(self, xy):
        self.xy = tuple(xy)

    def set_width(self, width):
        self.width = float(width)

    def set_height(self, height):
        self.height = float(height)

    def set_visible(self, visible):
        self.visible = bool(visible)


class _FakeOverlay:
    def __init__(self):
        self.data = None
        self.extent = None
        self.visible = False

    def set_data(self, data):
        self.data = np.asarray(data, dtype=float)

    def set_extent(self, extent):
        self.extent = tuple(extent)

    def set_visible(self, visible):
        self.visible = bool(visible)


class _FakeAxis:
    def __init__(self, *, xlim=(0.0, 40.0), ylim=(-20.0, 20.0)):
        self._xlim = tuple(xlim)
        self._ylim = tuple(ylim)
        self.patches = []

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def add_patch(self, patch):
        self.patches.append(patch)


class _FakeImageDisplay:
    def __init__(
        self,
        extent=(0.0, 3.0, 2.0, 0.0),
        *,
        source_extent=None,
        source_origin=None,
    ):
        self._extent = tuple(extent)
        if source_extent is not None:
            self._ra_sim_source_extent = tuple(source_extent)
        if source_origin is not None:
            self._ra_sim_source_origin = str(source_origin)

    def get_extent(self):
        return self._extent


class _FakeEvent:
    def __init__(self, *, button=1, inaxes=None, xdata=None, ydata=None):
        self.button = button
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata


class _FakeRoot:
    def __init__(self):
        self.next_token = 0
        self.after_calls = []
        self.after_cancel_calls = []

    def after(self, delay, callback):
        self.next_token += 1
        token = f"after-{self.next_token}"
        self.after_calls.append((delay, callback, token))
        return token

    def after_cancel(self, token):
        self.after_cancel_calls.append(token)


def _range_view_state() -> state.IntegrationRangeControlsViewState:
    return state.IntegrationRangeControlsViewState(
        tth_min_var=_FakeVar(0.0),
        tth_max_var=_FakeVar(0.0),
        phi_min_var=_FakeVar(0.0),
        phi_max_var=_FakeVar(0.0),
        tth_min_slider=_FakeSlider(0.0, 40.0),
        phi_min_slider=_FakeSlider(-20.0, 20.0),
    )


def test_create_runtime_drag_rectangles_attach_hidden_overlays() -> None:
    axis = _FakeAxis()

    drag_rect = integration_range_drag.create_drag_select_rectangle(
        axis,
        rectangle_cls=_FakeRect,
    )
    region_rect = integration_range_drag.create_integration_region_rectangle(
        axis,
        rectangle_cls=_FakeRect,
    )

    assert axis.patches == [drag_rect, region_rect]
    assert drag_rect.init_xy == (0.0, 0.0)
    assert drag_rect.init_width == 0.0
    assert drag_rect.init_height == 0.0
    assert drag_rect.init_kwargs["edgecolor"] == integration_range_drag._ACTIVE_DRAG_EDGE_COLOR
    assert drag_rect.init_kwargs["facecolor"] == integration_range_drag._ACTIVE_DRAG_FACE_RGBA
    assert drag_rect.init_kwargs["linewidth"] == integration_range_drag._ACTIVE_DRAG_LINEWIDTH
    assert drag_rect.init_kwargs["zorder"] == 8
    assert drag_rect.visible is False
    assert region_rect.init_xy == (0.0, 0.0)
    assert region_rect.init_width == 0.0
    assert region_rect.init_height == 0.0
    assert (
        region_rect.init_kwargs["edgecolor"] == integration_range_drag._SELECTED_REGION_EDGE_COLOR
    )
    assert region_rect.init_kwargs["facecolor"] == integration_range_drag._SELECTED_REGION_FACE_RGBA
    assert region_rect.init_kwargs["linewidth"] == integration_range_drag._SELECTED_REGION_LINEWIDTH
    assert region_rect.init_kwargs["linestyle"] == "-"
    assert region_rect.visible is False


def test_create_integration_region_highlight_cmap_uses_bold_selection_color() -> None:
    colors = integration_range_drag.create_integration_region_highlight_cmap(
        listed_colormap_cls=lambda values: tuple(values),
    )

    assert colors == (
        (0.0, 0.0, 0.0, 0.0),
        integration_range_drag._SELECTED_REGION_OVERLAY_RGBA,
        integration_range_drag._SELECTED_QR_ROD_BAND_FILL_RGBA,
        integration_range_drag._SELECTED_QR_ROD_BAND_BOUNDARY_RGBA,
        integration_range_drag._SELECTED_QR_ROD_CENTERLINE_RGBA,
        integration_range_drag._SELECTED_QR_ROD_MUTED_BAND_FILL_RGBA,
        integration_range_drag._SELECTED_QR_ROD_MUTED_BAND_BOUNDARY_RGBA,
        integration_range_drag._SELECTED_QR_ROD_MUTED_CENTERLINE_RGBA,
    )


def test_range_refresh_requires_pending_analysis_result_only_without_cache() -> None:
    assert (
        integration_range_drag.range_refresh_requires_pending_analysis_result(
            active_job=object(),
            queued_job=None,
            cached_result=None,
        )
        is True
    )
    assert (
        integration_range_drag.range_refresh_requires_pending_analysis_result(
            active_job=None,
            queued_job=object(),
            cached_result=None,
        )
        is True
    )
    assert (
        integration_range_drag.range_refresh_requires_pending_analysis_result(
            active_job=object(),
            queued_job=None,
            cached_result=object(),
        )
        is False
    )
    assert (
        integration_range_drag.range_refresh_requires_pending_analysis_result(
            active_job=None,
            queued_job=None,
            cached_result=None,
        )
        is False
    )


def test_integration_range_drag_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"view": 0, "schedule": 0, "draw": 0, "status": 0}

    monkeypatch.setattr(
        integration_range_drag,
        "IntegrationRangeDragBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_view():
        counters["view"] += 1
        return f"view-{counters['view']}"

    def build_schedule():
        counters["schedule"] += 1
        idx = counters["schedule"]
        return lambda: f"schedule-{idx}"

    def build_draw():
        counters["draw"] += 1
        idx = counters["draw"]
        return lambda: f"draw-{idx}"

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    factory = integration_range_drag.make_runtime_integration_range_drag_bindings_factory(
        drag_state="drag-state",
        peak_selection_state="peak-state",
        range_view_state_factory=build_view,
        ax="axis",
        drag_select_rect="drag-rect",
        integration_region_overlay="overlay",
        integration_region_rect="overlay-rect",
        image_display="image-display",
        get_detector_angular_maps=lambda ai: ai,
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: "ai",
        sync_peak_selection_state=lambda: None,
        schedule_range_update_factory=build_schedule,
        last_sim_res2_factory=lambda: "res2",
        draw_idle_factory=build_draw,
        set_status_text_factory=build_status,
        caked_qr_rod_payload_factory=lambda: {"signature": ("mask-sig", 1)},
        detector_geometry_signature_factory=lambda: ("geom-sig", 1),
        caked_qr_rod_drag_context_factory=lambda: {"qr": 1.25},
    )

    assert factory()["drag_state"] == "drag-state"
    assert factory()["drag_state"] == "drag-state"
    assert calls[0]["range_view_state"] == "view-1"
    assert calls[1]["range_view_state"] == "view-2"
    assert calls[0]["ax"] == "axis"
    assert calls[0]["range_visible_factory"]() is True
    assert callable(calls[0]["schedule_range_update"])
    assert callable(calls[0]["draw_idle"])
    assert callable(calls[0]["set_status_text"])
    assert calls[0]["caked_qr_rod_payload_factory"]() == {"signature": ("mask-sig", 1)}
    assert calls[0]["detector_geometry_signature_factory"]() == ("geom-sig", 1)
    assert calls[0]["caked_qr_rod_drag_context_factory"]() == {"qr": 1.25}
    assert "update_integration_region_visuals" not in calls[0]
    assert calls[0]["schedule_range_update"] is not calls[1]["schedule_range_update"]
    assert calls[0]["draw_idle"] is not calls[1]["draw_idle"]
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]


def test_integration_range_update_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {
        "analysis": 0,
        "range": 0,
        "drag": 0,
        "lookup": 0,
        "refresh": 0,
        "schedule": 0,
        "debounce": 0,
    }

    monkeypatch.setattr(
        integration_range_drag,
        "IntegrationRangeUpdateBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def _bump(key: str, prefix: str) -> str:
        counters[key] += 1
        return f"{prefix}-{counters[key]}"

    def _next_debounce() -> int:
        counters["debounce"] += 1
        return 90 + counters["debounce"]

    factory = integration_range_drag.make_runtime_integration_range_update_bindings_factory(
        root="root",
        simulation_runtime_state="simulation-state",
        analysis_view_state_factory=lambda: _bump("analysis", "analysis"),
        range_view_state_factory=lambda: _bump("range", "range"),
        display_controls_state="display-state",
        hkl_lookup_controls_factory=lambda: _bump("lookup", "lookup"),
        integration_range_drag_callbacks_factory=lambda: _bump("drag", "drag"),
        refresh_integration_from_cached_results_factory=lambda: _bump(
            "refresh",
            "refresh",
        ),
        schedule_update_factory=lambda: _bump("schedule", "schedule"),
        range_update_debounce_ms_factory=_next_debounce,
    )

    first = factory()
    second = factory()

    assert first["root"] == "root"
    assert first["simulation_runtime_state"] == "simulation-state"
    assert first["analysis_view_state"] == "analysis-1"
    assert second["analysis_view_state"] == "analysis-2"
    assert first["range_view_state"] == "range-1"
    assert second["range_view_state"] == "range-2"
    assert first["display_controls_state"] == "display-state"
    assert first["hkl_lookup_controls"] == "lookup-1"
    assert second["hkl_lookup_controls"] == "lookup-2"
    assert first["integration_range_drag_callbacks"] == "drag-1"
    assert second["integration_range_drag_callbacks"] == "drag-2"
    assert first["refresh_integration_from_cached_results"] == "refresh-1"
    assert second["schedule_update"] == "schedule-2"
    assert first["range_update_debounce_ms"] == 91
    assert second["range_update_debounce_ms"] == 92


def test_create_runtime_integration_range_controls_wires_callbacks_and_text_sync() -> None:
    schedule_calls = []
    refresh_calls = []
    disable_peak_pick_calls = []
    callback_refs = {}
    view_state = state.IntegrationRangeControlsViewState()
    show_1d_var = _FakeVar(False)

    def _create_controls(**kwargs):
        callback_refs.update(kwargs)
        kwargs["view_state"].tth_min_var = _FakeVar(kwargs["tth_min"])
        kwargs["view_state"].tth_max_var = _FakeVar(kwargs["tth_max"])
        kwargs["view_state"].phi_min_var = _FakeVar(kwargs["phi_min"])
        kwargs["view_state"].phi_max_var = _FakeVar(kwargs["phi_max"])
        kwargs["view_state"].qz_min_var = _FakeVar(kwargs["qz_min"])
        kwargs["view_state"].qz_max_var = _FakeVar(kwargs["qz_max"])
        kwargs["view_state"].delta_qr_var = _FakeVar(kwargs["delta_qr"])
        kwargs["view_state"].tth_min_label_var = _FakeVar("")
        kwargs["view_state"].tth_max_label_var = _FakeVar("")
        kwargs["view_state"].phi_min_label_var = _FakeVar("")
        kwargs["view_state"].phi_max_label_var = _FakeVar("")
        kwargs["view_state"].qz_min_label_var = _FakeVar("")
        kwargs["view_state"].qz_max_label_var = _FakeVar("")
        kwargs["view_state"].delta_qr_label_var = _FakeVar("")
        kwargs["view_state"].integrate_selected_qr_rod_var = _FakeVar(
            kwargs["integrate_selected_qr_rod"]
        )
        kwargs["view_state"].mirror_selected_qr_phi_var = _FakeVar(kwargs["mirror_selected_qr_phi"])
        kwargs["view_state"].include_selected_qr_rod_shape_var = _FakeVar(
            kwargs["include_selected_qr_rod_shape"]
        )
        kwargs["view_state"].caked_intensity_mode_var = _FakeVar(kwargs["caked_intensity_mode"])
        kwargs["view_state"].caked_intensity_mode_buttons = {
            "density": _FakeEntry(state="normal"),
            "raw_sum": _FakeEntry(state="normal"),
        }
        kwargs["view_state"].rod_profile_intensity_mode_var = _FakeVar(
            kwargs["rod_profile_intensity_mode"]
        )
        kwargs["view_state"].rod_profile_intensity_mode_buttons = {
            "density": _FakeEntry(state="disabled"),
            "raw_sum": _FakeEntry(state="disabled"),
        }
        kwargs["view_state"].selected_qr_rod_key_var = _FakeVar(kwargs["selected_qr_rod_key"])
        kwargs["view_state"].selected_qr_rod_display_var = _FakeVar("")
        kwargs["view_state"].selected_qr_rod_option_labels = {
            str(key): str(label) for key, label in kwargs["selected_qr_rod_options"]
        }
        kwargs["view_state"].selected_qr_rod_key_by_label = {
            str(label): str(key) for key, label in kwargs["selected_qr_rod_options"]
        }
        kwargs["view_state"].selected_qr_rod_listbox = _FakeEntry(state="disabled")
        kwargs["view_state"].selected_qr_rod_checkbox_vars = {}
        kwargs["view_state"].selected_qr_rod_checkbuttons = {}
        kwargs["view_state"].tth_min_entry_var = _FakeVar("")
        kwargs["view_state"].tth_max_entry_var = _FakeVar("")
        kwargs["view_state"].phi_min_entry_var = _FakeVar("")
        kwargs["view_state"].phi_max_entry_var = _FakeVar("")
        kwargs["view_state"].qz_min_entry_var = _FakeVar("")
        kwargs["view_state"].qz_max_entry_var = _FakeVar("")
        kwargs["view_state"].delta_qr_entry_var = _FakeVar("")
        kwargs["view_state"].delta_qr_cue_var = _FakeVar("")
        kwargs["view_state"].tth_min_slider = _FakeSlider(0.0, 60.0)
        kwargs["view_state"].tth_max_slider = _FakeSlider(0.0, 60.0)
        kwargs["view_state"].phi_min_slider = _FakeSlider(-15.0, 15.0)
        kwargs["view_state"].phi_max_slider = _FakeSlider(-15.0, 15.0)
        kwargs["view_state"].mirror_selected_qr_phi_checkbutton = _FakeEntry(state="disabled")
        kwargs["view_state"].include_selected_qr_rod_shape_checkbutton = _FakeEntry(
            state="disabled"
        )
        kwargs["view_state"].qz_min_slider = _FakeSlider(-2.0, 2.0)
        kwargs["view_state"].qz_max_slider = _FakeSlider(-2.0, 2.0)
        kwargs["view_state"].delta_qr_slider = _FakeSlider(0.001, 1.0)
        kwargs["view_state"].qz_min_entry = _FakeEntry(state="disabled")
        kwargs["view_state"].qz_max_entry = _FakeEntry(state="disabled")
        kwargs["view_state"].delta_qr_entry = _FakeEntry(state="disabled")

    integration_range_drag.create_runtime_integration_range_controls(
        parent="parent",
        views_module=SimpleNamespace(create_integration_range_controls=_create_controls),
        view_state=view_state,
        show_1d_var=show_1d_var,
        tth_min=0.0,
        tth_max=60.0,
        phi_min=-15.0,
        phi_max=15.0,
        selected_qr_rod_key="phase-a|1",
        selected_qr_rod_options=[("phase-a|1", "phase-a m=1 | Qr=1.2500 A^-1")],
        schedule_range_update=lambda: schedule_calls.append("range"),
        disable_peak_pick=lambda: disable_peak_pick_calls.append("disable"),
        refresh_region_visuals=lambda: refresh_calls.append("refresh"),
    )

    assert callback_refs["parent"] == "parent"
    assert view_state.tth_min_label_var.get() == "0.0"
    assert view_state.tth_max_entry_var.get() == "60.0000"
    assert view_state.phi_min_entry_var.get() == "-15.0000"
    assert view_state.qz_min_label_var.get() == "0.0000"
    assert view_state.qz_max_entry_var.get() == "5.0000"
    assert view_state.delta_qr_label_var.get() == "0.2500"
    assert view_state.delta_qr_entry_var.get() == "0.2500"
    assert view_state.delta_qr_cue_var.get() == "ΔQr = 0.2500 A^-1"
    assert view_state.selected_qr_rod_display_var.get() == "phase-a m=1 | Qr=1.2500 A^-1"
    assert view_state.selected_qr_rod_listbox.state == "disabled"
    assert view_state.mirror_selected_qr_phi_var.get() is False
    assert view_state.mirror_selected_qr_phi_checkbutton.state == "disabled"
    assert view_state.caked_intensity_mode_var.get() == "density"
    assert view_state.caked_intensity_mode_buttons["density"].state == "normal"
    assert view_state.rod_profile_intensity_mode_var.get() == "density"
    assert view_state.rod_profile_intensity_mode_buttons["density"].state == "disabled"
    assert view_state.qz_min_slider.state == "disabled"
    assert view_state.qz_max_entry.state == "disabled"
    assert view_state.delta_qr_slider.state == "disabled"
    assert len(view_state.tth_min_var.trace_calls) == 1
    assert len(view_state.phi_max_var.trace_calls) == 1
    assert len(view_state.qz_min_var.trace_calls) == 1
    assert len(view_state.qz_max_var.trace_calls) == 1
    assert len(view_state.delta_qr_var.trace_calls) == 1
    assert len(view_state.integrate_selected_qr_rod_var.trace_calls) == 1

    callback_refs["on_tth_min_changed"]("12.5")
    assert view_state.tth_min_var.get() == 12.5
    assert show_1d_var.get() is True
    assert view_state.tth_min_label_var.get() == "12.5"
    assert view_state.tth_min_entry_var.get() == "12.5000"
    assert refresh_calls == []

    show_1d_var.set(False)
    callback_refs["on_delta_qr_changed"]("0.125")
    assert view_state.delta_qr_var.get() == 0.125
    assert show_1d_var.get() is True
    assert view_state.delta_qr_label_var.get() == "0.1250"
    assert view_state.delta_qr_entry_var.get() == "0.1250"
    assert view_state.delta_qr_cue_var.get() == "ΔQr = 0.1250 A^-1"
    assert refresh_calls == ["refresh"]

    show_1d_var.set(False)
    view_state.integrate_selected_qr_rod_var.set(True)
    callback_refs["on_toggle_integrate_selected_qr_rod"]()
    assert show_1d_var.get() is True
    assert view_state.selected_qr_rod_listbox.state == "normal"
    assert view_state.mirror_selected_qr_phi_checkbutton.state == "normal"
    assert view_state.rod_profile_intensity_mode_buttons["density"].state == "normal"
    assert view_state.rod_profile_intensity_mode_var.get() == "density"
    assert view_state.qz_min_slider.state == "normal"
    assert view_state.qz_max_entry.state == "normal"
    assert view_state.delta_qr_slider.state == "normal"
    assert view_state.tth_min_slider.state == "disabled"
    assert view_state.phi_min_slider.state == "normal"
    assert view_state.phi_max_slider.state == "normal"
    assert refresh_calls == ["refresh", "refresh"]
    assert disable_peak_pick_calls == ["disable"]

    show_1d_var.set(False)
    view_state.mirror_selected_qr_phi_var.set(True)
    callback_refs["on_toggle_mirror_selected_qr_phi"]()
    assert show_1d_var.get() is True
    assert refresh_calls == ["refresh", "refresh", "refresh"]

    show_1d_var.set(False)
    callback_refs["on_rod_profile_intensity_mode_changed"]("raw_sum")
    assert view_state.rod_profile_intensity_mode_var.get() == "raw_sum"
    assert show_1d_var.get() is True

    show_1d_var.set(False)
    callback_refs["on_caked_intensity_mode_changed"]("raw_sum")
    assert view_state.caked_intensity_mode_var.get() == "raw_sum"
    assert show_1d_var.get() is True

    show_1d_var.set(False)
    view_state.integrate_selected_qr_rod_var.set(False)
    callback_refs["on_toggle_integrate_selected_qr_rod"]()
    assert disable_peak_pick_calls == ["disable"]

    show_1d_var.set(False)
    view_state.phi_max_entry_var.set("45.0")
    callback_refs["on_apply_entry"](
        view_state.phi_max_entry_var,
        view_state.phi_max_var,
        view_state.phi_max_slider,
    )
    assert view_state.phi_max_var.get() == 15.0
    assert show_1d_var.get() is True
    assert view_state.phi_max_label_var.get() == "15.0"
    assert view_state.phi_max_entry_var.get() == "15.0000"
    assert refresh_calls == ["refresh", "refresh", "refresh", "refresh", "refresh"]

    show_1d_var.set(False)
    view_state.delta_qr_entry_var.set("9.0")
    callback_refs["on_apply_entry"](
        view_state.delta_qr_entry_var,
        view_state.delta_qr_var,
        view_state.delta_qr_slider,
    )
    assert view_state.delta_qr_var.get() == 1.0
    assert show_1d_var.get() is True
    assert view_state.delta_qr_label_var.get() == "1.0000"
    assert view_state.delta_qr_entry_var.get() == "1.0000"
    assert view_state.delta_qr_cue_var.get() == "ΔQr = 1.0000 A^-1"
    show_1d_var.set(False)
    view_state.delta_qr_entry_var.set("0.0")
    callback_refs["on_apply_entry"](
        view_state.delta_qr_entry_var,
        view_state.delta_qr_var,
        view_state.delta_qr_slider,
    )
    assert view_state.delta_qr_var.get() == 0.001
    assert show_1d_var.get() is True
    assert view_state.delta_qr_label_var.get() == "0.0010"
    assert view_state.delta_qr_entry_var.get() == "0.0010"
    assert view_state.delta_qr_cue_var.get() == "ΔQr = 0.0010 A^-1"
    show_1d_var.set(False)
    callback_refs["on_selected_qr_rod_changed"]("phase-a m=1 | Qr=1.2500 A^-1")
    assert view_state.selected_qr_rod_key_var.get() == "phase-a|1"
    assert show_1d_var.get() is True
    assert schedule_calls == [
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
        "range",
    ]
    assert refresh_calls == [
        "refresh",
        "refresh",
        "refresh",
        "refresh",
        "refresh",
        "refresh",
        "refresh",
        "refresh",
    ]

    view_state.phi_min_var.set(-7.25)
    trace_callback = view_state.phi_min_var.trace_calls[0][1]
    trace_callback()
    assert view_state.phi_min_label_var.get() == "-7.2"
    assert view_state.phi_min_entry_var.get() == "-7.2500"


def test_selected_qr_rod_toggle_applies_phi_defaults_only_when_not_custom() -> None:
    schedule_calls = []
    refresh_calls = []
    disable_calls = []
    view_state = state.IntegrationRangeControlsViewState(
        phi_min_var=_FakeVar(-15.0),
        phi_max_var=_FakeVar(15.0),
        phi_min_label_var=_FakeVar(""),
        phi_max_label_var=_FakeVar(""),
        phi_min_entry_var=_FakeVar(""),
        phi_max_entry_var=_FakeVar(""),
        phi_min_slider=_FakeSlider(-180.0, 180.0),
        phi_max_slider=_FakeSlider(-180.0, 180.0),
        integrate_selected_qr_rod_var=_FakeVar(True),
        integrate_selected_qr_rod_value=False,
        selected_qr_rod_phi_customized=False,
    )
    show_1d_var = _FakeVar(False)

    integration_range_drag._toggle_runtime_integrate_selected_qr_rod(
        view_state=view_state,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        disable_peak_pick=lambda: disable_calls.append(True),
        refresh_region_visuals=lambda: refresh_calls.append(True),
    )

    assert view_state.phi_min_var.get() == -90.0
    assert view_state.phi_max_var.get() == 90.0
    assert view_state.phi_min_entry_var.get() == "-90.0000"
    assert view_state.phi_max_entry_var.get() == "90.0000"
    assert schedule_calls == [True]
    assert refresh_calls == [True]
    assert disable_calls == [True]

    view_state.integrate_selected_qr_rod_value = False
    view_state.integrate_selected_qr_rod_var.set(True)
    view_state.phi_min_var.set(-15.0)
    view_state.phi_max_var.set(15.0)
    view_state.selected_qr_rod_phi_customized = True

    integration_range_drag._toggle_runtime_integrate_selected_qr_rod(
        view_state=view_state,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
    )

    assert view_state.phi_min_var.get() == -15.0
    assert view_state.phi_max_var.get() == 15.0


def test_selected_qr_rod_toggle_detector_default_raw_sum_when_fresh() -> None:
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        integrate_selected_qr_rod_value=False,
        rod_profile_intensity_mode_var=_FakeVar("density"),
        selected_qr_rod_default_intensity_mode="raw_sum",
        rod_profile_intensity_mode_customized=False,
        phi_min_var=_FakeVar(-15.0),
        phi_max_var=_FakeVar(15.0),
    )
    show_1d_var = _FakeVar(False)

    integration_range_drag._toggle_runtime_integrate_selected_qr_rod(
        view_state=view_state,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: None,
    )

    assert view_state.rod_profile_intensity_mode_var.get() == "raw_sum"
    assert view_state.rod_profile_intensity_mode_value == "raw_sum"
    assert show_1d_var.get() is True


def test_selected_qr_rod_toggle_caked_default_keeps_density_when_fresh() -> None:
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        integrate_selected_qr_rod_value=False,
        rod_profile_intensity_mode_var=_FakeVar("density"),
        selected_qr_rod_default_intensity_mode="density",
        rod_profile_intensity_mode_customized=False,
        phi_min_var=_FakeVar(-15.0),
        phi_max_var=_FakeVar(15.0),
    )

    integration_range_drag._toggle_runtime_integrate_selected_qr_rod(
        view_state=view_state,
        show_1d_var=_FakeVar(False),
        schedule_range_update=lambda: None,
    )

    assert view_state.rod_profile_intensity_mode_var.get() == "density"


def test_selected_qr_rod_toggle_preserves_custom_intensity_mode() -> None:
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        integrate_selected_qr_rod_value=False,
        rod_profile_intensity_mode_var=_FakeVar("density"),
        selected_qr_rod_default_intensity_mode="raw_sum",
        rod_profile_intensity_mode_customized=True,
        phi_min_var=_FakeVar(-15.0),
        phi_max_var=_FakeVar(15.0),
    )

    integration_range_drag._toggle_runtime_integrate_selected_qr_rod(
        view_state=view_state,
        show_1d_var=_FakeVar(False),
        schedule_range_update=lambda: None,
    )

    assert view_state.rod_profile_intensity_mode_var.get() == "density"


def test_select_runtime_selected_qr_rod_empty_selection_promotes_first_option() -> None:
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        selected_qr_rod_options=["rod-1", "rod-2"],
        selected_qr_rod_key_var=_FakeVar(""),
        selected_qr_rod_display_var=_FakeVar(""),
        selected_qr_rod_option_labels={"rod-1": "Rod 1", "rod-2": "Rod 2"},
        selected_qr_rod_listbox=_FakeEntry(),
    )
    schedule_calls = []

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=[],
        show_1d_var=_FakeVar(False),
        schedule_range_update=lambda: schedule_calls.append(True),
    )

    assert view_state.selected_qr_rod_keys_value == ["rod-1"]
    assert view_state.selected_qr_rod_key_var.get() == "rod-1"
    assert view_state.selected_qr_rod_listbox.selected == [0]
    assert schedule_calls == [True]


def test_select_runtime_selected_qr_rod_debounces_expensive_refresh() -> None:
    root = _FakeRoot()
    listbox = _FakeDebouncedListbox(root)
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        selected_qr_rod_options=["rod-1", "rod-2", "rod-3"],
        selected_qr_rod_key_var=_FakeVar(""),
        selected_qr_rod_display_var=_FakeVar(""),
        selected_qr_rod_option_labels={
            "rod-1": "Rod 1",
            "rod-2": "Rod 2",
            "rod-3": "Rod 3",
        },
        selected_qr_rod_listbox=listbox,
    )
    show_1d_var = _FakeVar(False)
    schedule_calls = []
    refresh_calls = []

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-2"],
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(tuple(view_state.selected_qr_rod_keys)),
        refresh_region_visuals=lambda: refresh_calls.append(tuple(view_state.selected_qr_rod_keys)),
    )

    assert view_state.selected_qr_rod_keys_value == ["rod-2"]
    assert view_state.selected_qr_rod_key_var.get() == "rod-2"
    assert show_1d_var.get() is True
    assert schedule_calls == []
    assert refresh_calls == []
    assert root.after_calls[0][0] == 125
    first_token = root.after_calls[0][2]

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-3", "rod-1"],
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(tuple(view_state.selected_qr_rod_keys)),
        refresh_region_visuals=lambda: refresh_calls.append(tuple(view_state.selected_qr_rod_keys)),
    )

    assert view_state.selected_qr_rod_keys_value == ["rod-1", "rod-3"]
    assert view_state.selected_qr_rod_key_var.get() == "rod-1"
    assert view_state.selected_qr_rod_listbox.selected == [0, 2]
    assert schedule_calls == []
    assert refresh_calls == []
    assert root.after_cancel_calls == [first_token]
    assert len(root.after_calls) == 2

    root.after_calls[-1][1]()

    assert schedule_calls == [("rod-1", "rod-3")]
    assert refresh_calls == [("rod-1", "rod-3")]
    assert view_state.selected_qr_rod_refresh_after_id is None
    assert view_state.selected_qr_rod_refresh_after_widget is None

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-2"],
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(tuple(view_state.selected_qr_rod_keys)),
        refresh_region_visuals=lambda: refresh_calls.append(tuple(view_state.selected_qr_rod_keys)),
    )
    assert len(root.after_calls) == 3

    root.after_calls[-1][1]()

    assert schedule_calls == [("rod-1", "rod-3"), ("rod-2",)]
    assert refresh_calls == [("rod-1", "rod-3"), ("rod-2",)]
    assert view_state.selected_qr_rod_refresh_after_id is None
    assert view_state.selected_qr_rod_refresh_after_widget is None


def test_delta_qr_slider_uses_single_debounced_selected_rod_refresh_path() -> None:
    root = _FakeRoot()
    view_state = state.IntegrationRangeControlsViewState(
        selected_qr_rod_listbox=_FakeDebouncedListbox(root),
        delta_qr_var=_FakeVar(0.25),
        delta_qr_label_var=_FakeVar(""),
        delta_qr_entry_var=_FakeVar(""),
        delta_qr_cue_var=_FakeVar(""),
    )
    show_1d_var = _FakeVar(False)
    schedule_calls = []
    refresh_calls = []
    callback = integration_range_drag._make_runtime_range_slider_callback(
        view_state=view_state,
        value_var_name="delta_qr_var",
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(view_state.delta_qr_var.get()),
        refresh_region_visuals=lambda: refresh_calls.append(view_state.delta_qr_var.get()),
    )

    callback("0.125")
    assert show_1d_var.get() is True
    assert view_state.delta_qr_var.get() == 0.125
    assert view_state.delta_qr_entry_var.get() == "0.1250"
    assert view_state.delta_qr_cue_var.get() == "ΔQr = 0.1250 A^-1"
    assert schedule_calls == []
    assert refresh_calls == []
    assert len(root.after_calls) == 1
    first_token = root.after_calls[0][2]

    callback("0.250")
    assert view_state.delta_qr_var.get() == 0.25
    assert view_state.delta_qr_entry_var.get() == "0.2500"
    assert root.after_cancel_calls == [first_token]
    assert len(root.after_calls) == 2
    assert schedule_calls == []
    assert refresh_calls == []

    root.after_calls[-1][1]()

    assert schedule_calls == [0.25]
    assert refresh_calls == [0.25]
    assert view_state.selected_qr_rod_refresh_after_id is None
    assert view_state.selected_qr_rod_refresh_after_widget is None


def test_select_runtime_selected_qr_rod_skips_debounce_after_widget_destroyed() -> None:
    root = _FakeRoot()
    listbox = _FakeDebouncedListbox(root)
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        selected_qr_rod_options=["rod-1", "rod-2"],
        selected_qr_rod_key_var=_FakeVar(""),
        selected_qr_rod_display_var=_FakeVar(""),
        selected_qr_rod_option_labels={"rod-1": "Rod 1", "rod-2": "Rod 2"},
        selected_qr_rod_listbox=listbox,
    )
    schedule_calls = []
    refresh_calls = []

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-2"],
        show_1d_var=_FakeVar(False),
        schedule_range_update=lambda: schedule_calls.append(True),
        refresh_region_visuals=lambda: refresh_calls.append(True),
    )
    listbox.exists = False
    root.after_calls[-1][1]()

    assert schedule_calls == []
    assert refresh_calls == []
    assert view_state.selected_qr_rod_refresh_after_id is None
    assert view_state.selected_qr_rod_refresh_after_widget is None


def test_select_runtime_selected_qr_rod_debounce_reads_latest_state_at_execution() -> None:
    root = _FakeRoot()
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        selected_qr_rod_options=["rod-1", "rod-2", "rod-3"],
        selected_qr_rod_key_var=_FakeVar(""),
        selected_qr_rod_display_var=_FakeVar(""),
        selected_qr_rod_option_labels={
            "rod-1": "Rod 1",
            "rod-2": "Rod 2",
            "rod-3": "Rod 3",
        },
        selected_qr_rod_listbox=_FakeDebouncedListbox(root),
    )
    observed_keys = []

    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-1"],
        show_1d_var=_FakeVar(False),
        schedule_range_update=lambda: observed_keys.append(tuple(view_state.selected_qr_rod_keys)),
    )

    integration_range_drag._set_runtime_selected_qr_rod_keys(
        view_state,
        ["rod-3", "rod-2"],
    )
    root.after_calls[-1][1]()

    assert observed_keys == [("rod-2", "rod-3")]


def test_qr_rod_profiling_hooks_are_env_gated_and_keep_debounce(
    monkeypatch,
) -> None:
    root = _FakeRoot()
    view_state = state.IntegrationRangeControlsViewState(
        integrate_selected_qr_rod_var=_FakeVar(True),
        selected_qr_rod_options=["rod-1", "rod-2", "rod-3"],
        selected_qr_rod_key_var=_FakeVar(""),
        selected_qr_rod_display_var=_FakeVar(""),
        selected_qr_rod_option_labels={
            "rod-1": "Rod 1",
            "rod-2": "Rod 2",
            "rod-3": "Rod 3",
        },
        selected_qr_rod_listbox=_FakeDebouncedListbox(root),
    )
    show_1d_var = _FakeVar(False)
    schedule_calls = []
    refresh_calls = []

    monkeypatch.delenv("RA_SIM_PROFILE_QR_ROD", raising=False)
    integration_range_drag.clear_qr_rod_profile_timings()
    integration_range_drag._set_runtime_selected_qr_rod_options(
        view_state,
        [("rod-1", "Rod 1"), ("rod-2", "Rod 2")],
    )
    assert integration_range_drag._QR_ROD_PROFILE_TIMINGS == []

    monkeypatch.setenv("RA_SIM_PROFILE_QR_ROD", "1")
    integration_range_drag.clear_qr_rod_profile_timings()
    integration_range_drag._set_runtime_selected_qr_rod_options(
        view_state,
        [("rod-1", "Rod 1"), ("rod-2", "Rod 2"), ("rod-3", "Rod 3")],
    )
    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-2"],
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(tuple(view_state.selected_qr_rod_keys)),
        refresh_region_visuals=lambda: refresh_calls.append(tuple(view_state.selected_qr_rod_keys)),
    )
    first_token = root.after_calls[-1][2]
    integration_range_drag._select_runtime_selected_qr_rod(
        view_state=view_state,
        display_value=["rod-3", "rod-1"],
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(tuple(view_state.selected_qr_rod_keys)),
        refresh_region_visuals=lambda: refresh_calls.append(tuple(view_state.selected_qr_rod_keys)),
    )
    root.after_calls[-1][1]()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda _ai: (None, None),
        range_visible_factory=lambda: False,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=object(),
        sim_res2=None,
    )

    assert root.after_cancel_calls == [first_token]
    assert schedule_calls == [("rod-1", "rod-3")]
    assert refresh_calls == [("rod-1", "rod-3")]
    assert {
        "rod_option_sync",
        "selection_debounce_callback",
        "overlay_update",
    }.issubset({entry["stage"] for entry in integration_range_drag._QR_ROD_PROFILE_TIMINGS})
    assert all(
        entry["elapsed_ms"] >= 0.0 for entry in integration_range_drag._QR_ROD_PROFILE_TIMINGS
    )
    integration_range_drag.clear_qr_rod_profile_timings()
    for index in range(integration_range_drag._QR_ROD_PROFILE_TIMINGS_MAX_RECORDS + 5):
        integration_range_drag._qr_rod_profile_end(f"cap-{index}", 0.0)
    assert len(integration_range_drag._QR_ROD_PROFILE_TIMINGS) == (
        integration_range_drag._QR_ROD_PROFILE_TIMINGS_MAX_RECORDS
    )
    assert integration_range_drag._QR_ROD_PROFILE_TIMINGS[0]["stage"] == "cap-5"


def test_integration_range_drag_bindings_default_qr_rod_payload_factories_are_none() -> None:
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    assert bindings.caked_qr_rod_payload_factory is None
    assert bindings.detector_qr_rod_payload_factory is None
    assert bindings.detector_geometry_signature_factory is None
    assert bindings.caked_qr_rod_drag_context_factory is None


def test_integration_range_update_callbacks_schedule_reschedule_and_toggle_modes() -> None:
    root = _FakeRoot()
    schedule_update_calls = []
    refresh_display_calls = []
    hkl_pick_calls = []
    drag_reset_calls = []
    refresh_results = [False]
    sim_state = SimpleNamespace(
        integration_update_pending=None,
        update_running=False,
        caked_limits_user_override=True,
    )
    analysis_view_state = SimpleNamespace(
        show_1d_var=_FakeVar(False),
        show_caked_2d_var=_FakeVar(False),
    )
    display_controls_state = SimpleNamespace(simulation_limits_user_override=True)
    bindings = integration_range_drag.IntegrationRangeUpdateBindings(
        root=root,
        simulation_runtime_state=sim_state,
        analysis_view_state=analysis_view_state,
        range_view_state=state.IntegrationRangeControlsViewState(),
        display_controls_state=display_controls_state,
        hkl_lookup_controls=SimpleNamespace(
            set_hkl_pick_mode=lambda enabled: hkl_pick_calls.append(enabled)
        ),
        integration_range_drag_callbacks=SimpleNamespace(
            reset=lambda: drag_reset_calls.append(True)
        ),
        refresh_integration_from_cached_results=lambda: refresh_results[-1],
        refresh_display_from_controls=lambda: refresh_display_calls.append(True),
        schedule_update=lambda: schedule_update_calls.append(True),
        range_update_debounce_ms=120,
    )
    callbacks = integration_range_drag.make_runtime_integration_range_update_callbacks(
        lambda: bindings
    )

    callbacks.schedule_range_update(delay_ms=50)
    assert root.after_calls[0][0] == 120
    assert sim_state.integration_update_pending == "after-1"

    sim_state.update_running = True
    root.after_calls[0][1]()
    assert sim_state.integration_update_pending == "after-2"
    assert root.after_cancel_calls == []
    assert root.after_calls[1][0] == 120
    assert schedule_update_calls == []

    sim_state.update_running = False
    root.after_calls[1][1]()
    assert sim_state.integration_update_pending is None
    assert schedule_update_calls == [True]

    callbacks.toggle_1d_plots()
    callbacks.toggle_log_display()
    assert root.after_cancel_calls == []
    assert root.after_calls[-1][0] == 120
    assert refresh_display_calls == [True]

    analysis_view_state.show_caked_2d_var.set(True)
    display_controls_state.simulation_limits_user_override = True
    callbacks.toggle_caked_2d()
    assert analysis_view_state.show_1d_var.get() is False
    assert display_controls_state.simulation_limits_user_override is False
    assert hkl_pick_calls == [False]
    assert drag_reset_calls == [True]
    assert schedule_update_calls == [True, True]

    sim_state.caked_limits_user_override = True
    analysis_view_state.show_caked_2d_var.set(False)
    callbacks.toggle_caked_2d()
    assert analysis_view_state.show_1d_var.get() is False
    assert sim_state.caked_limits_user_override is False
    assert drag_reset_calls == [True, True]
    assert schedule_update_calls == [True, True, True]


def test_integration_range_update_callbacks_use_direct_toggle_callback_when_bound() -> None:
    direct_toggle_calls = []
    schedule_update_calls = []
    drag_reset_calls = []
    hkl_pick_calls = []
    sim_state = SimpleNamespace(
        integration_update_pending=None,
        update_running=False,
        caked_limits_user_override=True,
    )
    analysis_view_state = SimpleNamespace(
        show_1d_var=_FakeVar(False),
        show_caked_2d_var=_FakeVar(False),
    )
    display_controls_state = SimpleNamespace(simulation_limits_user_override=True)
    bindings = integration_range_drag.IntegrationRangeUpdateBindings(
        root=_FakeRoot(),
        simulation_runtime_state=sim_state,
        analysis_view_state=analysis_view_state,
        range_view_state=state.IntegrationRangeControlsViewState(),
        display_controls_state=display_controls_state,
        hkl_lookup_controls=SimpleNamespace(
            set_hkl_pick_mode=lambda enabled: hkl_pick_calls.append(enabled)
        ),
        integration_range_drag_callbacks=SimpleNamespace(
            reset=lambda: drag_reset_calls.append(True)
        ),
        toggle_caked_2d=lambda: direct_toggle_calls.append(True),
        schedule_update=lambda: schedule_update_calls.append(True),
        range_update_debounce_ms=120,
    )
    callbacks = integration_range_drag.make_runtime_integration_range_update_callbacks(
        lambda: bindings
    )

    analysis_view_state.show_caked_2d_var.set(True)
    callbacks.toggle_caked_2d()

    assert analysis_view_state.show_1d_var.get() is False
    assert display_controls_state.simulation_limits_user_override is False
    assert hkl_pick_calls == [False]
    assert drag_reset_calls == [True]
    assert direct_toggle_calls == [True]
    assert schedule_update_calls == []


def test_update_runtime_integration_region_visuals_hides_when_range_hidden() -> None:
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()
    overlay_rect.visible = True

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: False,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=object(),
        sim_res2=object(),
    )

    assert overlay.visible is False
    assert overlay_rect.visible is False


def test_update_runtime_integration_region_visuals_updates_caked_rectangle() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(8.0)
    view_state.tth_max_var.set(3.0)
    view_state.phi_min_var.set(-1.0)
    view_state.phi_max_var.set(7.0)
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=object(),
    )

    assert overlay.visible is False
    assert overlay_rect.visible is True
    assert overlay_rect.xy == (3.0, -1.0)
    assert overlay_rect.width == 5.0
    assert overlay_rect.height == 8.0


def test_update_runtime_integration_region_visuals_uses_caked_qr_rod_payload_once() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(1.0)
    view_state.tth_max_var.set(3.0)
    view_state.phi_min_var.set(-180.0)
    view_state.phi_max_var.set(180.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    sim_res2 = SimpleNamespace(
        radial=np.asarray([1.0, 2.0, 3.0], dtype=float),
        azimuthal=np.asarray([10.0, 0.0, -10.0], dtype=float),
    )
    custom_mask = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    payload_calls = []

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        caked_qr_rod_payload_factory=lambda: (
            payload_calls.append(True)
            or {
                "mask": custom_mask,
                "signature": ("caked-mask", 1),
                "selected_qr_rod_keys": ["rod-1"],
                "metadata": {"source": "test"},
            }
        ),
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )

    assert overlay.visible is True
    assert overlay_rect.visible is False
    np.testing.assert_allclose(overlay.data, custom_mask.astype(float))
    assert len(payload_calls) == 1


def test_update_runtime_integration_region_visuals_emits_caked_overlay_source_signature() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(1.0)
    view_state.tth_max_var.set(3.0)
    view_state.phi_min_var.set(-180.0)
    view_state.phi_max_var.set(180.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    sim_res2 = SimpleNamespace(
        radial=np.asarray([1.0, 2.0, 3.0], dtype=float),
        azimuthal=np.asarray([10.0, 0.0, -10.0], dtype=float),
    )
    custom_mask = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    custom_mask_signature = [("mask-sig", 1)]
    projected: list[tuple[np.ndarray, object | None]] = []

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        set_integration_overlay_image=lambda image, *, source_signature=None: projected.append(
            (np.asarray(image, dtype=float), source_signature)
        ),
        caked_qr_rod_payload_factory=lambda: {
            "mask": custom_mask,
            "signature": custom_mask_signature[0],
            "selected_qr_rod_keys": ["rod-1"],
            "metadata": {},
        },
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )
    custom_mask_signature[0] = ("mask-sig", 2)
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )

    assert overlay.visible is True
    assert overlay_rect.visible is False
    assert len(projected) == 3
    np.testing.assert_allclose(projected[0][0], custom_mask.astype(float))
    assert projected[0][1] == ("mask-sig", 1)
    assert projected[1][1] == projected[0][1]
    assert projected[2][1] == ("mask-sig", 2)


def test_update_runtime_integration_region_visuals_uses_qr_rod_visual_overlay() -> None:
    view_state = _range_view_state()
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    sim_res2 = SimpleNamespace(
        radial=np.asarray([1.0, 2.0], dtype=float),
        azimuthal=np.asarray([0.0, 10.0], dtype=float),
    )
    custom_mask = np.asarray([[True, True], [False, True]], dtype=bool)
    visual_overlay = np.asarray([[2.0, 4.0], [0.0, 3.0]], dtype=np.float32)
    projected: list[tuple[np.ndarray, object | None]] = []

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        set_integration_overlay_image=lambda image, *, source_signature=None: projected.append(
            (np.asarray(image, dtype=float), source_signature)
        ),
        caked_qr_rod_payload_factory=lambda: {
            "mask": custom_mask,
            "signature": ("mask-sig", 1),
            "visual_overlay": visual_overlay,
            "visual_signature": ("visual-sig", 1),
        },
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )

    assert overlay.visible is True
    assert overlay_rect.visible is False
    assert len(projected) == 1
    np.testing.assert_allclose(projected[0][0], visual_overlay)
    assert projected[0][1] == ("visual-sig", 1)


def test_update_runtime_integration_region_visuals_payload_factory_reads_latest_selection_settings() -> (
    None
):
    view_state = _range_view_state()
    view_state.tth_min_var.set(1.0)
    view_state.tth_max_var.set(3.0)
    view_state.phi_min_var.set(-180.0)
    view_state.phi_max_var.set(180.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.selected_qr_rod_options = ["rod-1", "rod-2"]
    view_state.selected_qr_rod_option_labels = {
        "rod-1": "Rod 1",
        "rod-2": "Rod 2",
    }
    view_state.selected_qr_rod_display_var = _FakeVar("")
    view_state.selected_qr_rod_key_var = _FakeVar("")
    view_state.qz_min_var = _FakeVar(0.0)
    view_state.qz_max_var = _FakeVar(2.0)
    view_state.delta_qr_var = _FakeVar(0.25)
    integration_range_drag._set_runtime_selected_qr_rod_keys(view_state, ["rod-1"])

    sim_res2 = SimpleNamespace(
        radial=np.asarray([1.0, 2.0], dtype=float),
        azimuthal=np.asarray([0.0, 10.0], dtype=float),
    )
    observed: list[tuple[tuple[str, ...], tuple[float, float, float]]] = []
    projected: list[tuple[np.ndarray, object | None]] = []

    def payload_factory() -> dict[str, object]:
        keys = tuple(str(key) for key in getattr(view_state, "selected_qr_rod_keys", []))
        settings = (
            float(view_state.qz_min_var.get()),
            float(view_state.qz_max_var.get()),
            float(view_state.delta_qr_var.get()),
        )
        observed.append((keys, settings))
        mask = (
            np.asarray([[True, False], [False, False]], dtype=bool)
            if keys == ("rod-2",)
            else np.asarray([[False, False], [False, True]], dtype=bool)
        )
        return {
            "mask": mask,
            "signature": ("mask-sig", keys, settings),
            "selected_qr_rod_keys": list(keys),
            "metadata": {"settings": settings},
        }

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        set_integration_overlay_image=lambda image, *, source_signature=None: projected.append(
            (np.asarray(image, dtype=float), source_signature)
        ),
        caked_qr_rod_payload_factory=payload_factory,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )
    view_state.qz_min_var.set(1.5)
    view_state.qz_max_var.set(4.5)
    view_state.delta_qr_var.set(0.5)
    integration_range_drag._set_runtime_selected_qr_rod_keys(view_state, ["rod-2"])
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=sim_res2,
    )

    assert observed == [
        (("rod-1",), (0.0, 2.0, 0.25)),
        (("rod-2",), (1.5, 4.5, 0.5)),
    ]
    assert projected[0][1] == ("mask-sig", ("rod-1",), (0.0, 2.0, 0.25))
    assert projected[1][1] == ("mask-sig", ("rod-2",), (1.5, 4.5, 0.5))
    np.testing.assert_allclose(projected[1][0], [[1.0, 0.0], [0.0, 0.0]])


def test_update_runtime_integration_region_visuals_uses_cached_values_without_controls() -> None:
    view_state = state.IntegrationRangeControlsViewState(
        tth_min_value=8.0,
        tth_max_value=3.0,
        phi_min_value=-1.0,
        phi_max_value=7.0,
    )
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=object(),
    )

    assert overlay.visible is False
    assert overlay_rect.visible is True
    assert overlay_rect.xy == (3.0, -1.0)
    assert overlay_rect.width == 5.0
    assert overlay_rect.height == 8.0


def test_update_runtime_integration_region_visuals_updates_raw_overlay() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert overlay.extent == (0.0, 2.0, 2.0, 0.0)
    assert int(np.sum(overlay.data)) == 6


def test_update_runtime_integration_region_visuals_uses_detector_qr_rod_payload_once(
    monkeypatch,
) -> None:
    view_state = _range_view_state()
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    detector_mask = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    angular_calls = []
    payload_calls = []

    monkeypatch.setattr(
        integration_range_drag,
        "_update_detector_integration_overlay",
        lambda *args, **kwargs: angular_calls.append((args, kwargs)) or True,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai: (_ for _ in ()).throw(AssertionError()),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
        detector_qr_rod_payload_factory=lambda: (
            payload_calls.append(True)
            or {
                "mask": detector_mask,
                "signature": ("detector-mask", 1),
                "selected_qr_rod_keys": ["rod-1"],
                "metadata": {"source": "test"},
            }
        ),
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=object(),
        sim_res2=None,
    )

    assert angular_calls == []
    assert overlay.visible is True
    assert overlay_rect.visible is False
    np.testing.assert_allclose(overlay.data, detector_mask.astype(float))
    assert len(payload_calls) == 1


def test_update_runtime_integration_region_visuals_hides_invalid_detector_qr_rod_payload(
    monkeypatch,
) -> None:
    view_state = _range_view_state()
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()
    overlay_rect.visible = True
    angular_calls = []
    payload_calls = []
    monkeypatch.setattr(
        integration_range_drag,
        "_update_detector_integration_overlay",
        lambda *args, **kwargs: angular_calls.append((args, kwargs)) or True,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai: (_ for _ in ()).throw(AssertionError()),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
        detector_qr_rod_payload_factory=lambda: (
            payload_calls.append(True)
            or {
                "mask": np.zeros((3,), dtype=bool),
                "signature": ("detector-mask", 1),
                "selected_qr_rod_keys": ["rod-1"],
                "metadata": {},
            }
        ),
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=object(),
        sim_res2=None,
    )

    assert angular_calls == []
    assert overlay.visible is False
    assert overlay_rect.visible is False
    assert len(payload_calls) == 1


def test_update_runtime_integration_region_visuals_keeps_detector_fallback_for_active_rod_mask() -> (
    None
):
    view_state = _range_view_state()
    view_state.integrate_qz_rods_var = _FakeVar(True)
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    sim_res2 = SimpleNamespace(
        radial=np.asarray([10.0, 20.0, 30.0], dtype=float),
        azimuthal=np.asarray([-10.0, 0.0, 10.0], dtype=float),
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        last_sim_res2_factory=lambda: sim_res2,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 6


def test_update_runtime_integration_region_visuals_keeps_detector_fallback_with_mismatched_rod_mask() -> (
    None
):
    view_state = _range_view_state()
    view_state.integrate_qz_rods_var = _FakeVar(True)
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    sim_res2 = SimpleNamespace(
        radial=np.asarray([10.0, 20.0, 30.0], dtype=float),
        azimuthal=np.asarray([-10.0, 0.0, 10.0], dtype=float),
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        last_sim_res2_factory=lambda: sim_res2,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 6


def test_update_runtime_integration_region_visuals_keeps_detector_fallback_without_rod_mask() -> (
    None
):
    view_state = _range_view_state()
    view_state.integrate_qz_rods_var = _FakeVar(True)
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 6


def test_update_runtime_integration_region_visuals_uses_overlay_projection_callback() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    projected: list[tuple[np.ndarray, object | None]] = []
    detector_geometry_signature = [("geom", 1)]

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        detector_geometry_signature_factory=lambda: detector_geometry_signature[0],
        set_integration_overlay_image=lambda image, *, source_signature=None: projected.append(
            (np.asarray(image, dtype=float), source_signature)
        ),
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )
    detector_geometry_signature[0] = ("geom", 2)
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )
    view_state.tth_max_var.set(12.0)
    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert len(projected) == 4
    assert int(np.sum(projected[0][0])) == 6
    assert projected[0][1] == (
        "detector_integration_overlay",
        (3, 3),
        (10.0, 22.0, -10.0, 2.0),
        ("geom", 1),
    )
    assert projected[1][1] == projected[0][1]
    assert projected[2][1] == (
        "detector_integration_overlay",
        (3, 3),
        (10.0, 22.0, -10.0, 2.0),
        ("geom", 2),
    )
    assert projected[3][1] == (
        "detector_integration_overlay",
        (3, 3),
        (10.0, 12.0, -10.0, 2.0),
        ("geom", 2),
    )
    assert overlay.data is None


def test_update_runtime_raw_drag_preview_uses_detector_overlay_shape() -> None:
    overlay = _FakeOverlay()
    drag_rect = _FakeRect()
    overlay_rect = _FakeRect()
    draw_calls = []
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="raw",
        x0=0.2,
        y0=1.1,
        x1=1.8,
        y1=2.0,
        tth0=20.0,
        phi0=0.0,
        tth1=32.0,
        phi1=12.0,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        draw_idle=lambda: draw_calls.append(True),
    )

    updated = integration_range_drag.update_runtime_raw_drag_preview(bindings, ai)

    assert updated is True
    assert drag_rect.visible is False
    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert overlay.extent == (0.0, 2.0, 2.0, 0.0)
    assert int(np.sum(overlay.data)) > 0
    assert draw_calls == [True]


def test_update_runtime_raw_drag_preview_reuses_buffer_with_new_source_signature() -> None:
    overlay = _FakeOverlay()
    drag_rect = _FakeRect()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="raw",
        x0=0.2,
        y0=1.1,
        x1=1.8,
        y1=2.0,
        tth0=20.0,
        phi0=0.0,
        tth1=32.0,
        phi1=12.0,
    )
    projected: list[tuple[np.ndarray, object | None]] = []
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        set_integration_overlay_image=lambda image, *, source_signature=None: projected.append(
            (image, source_signature)
        ),
    )

    assert integration_range_drag.update_runtime_raw_drag_preview(bindings, ai) is True

    drag_state.x1 = 1.5
    drag_state.y1 = 0.5
    drag_state.tth1 = 22.0
    drag_state.phi1 = 2.0

    assert integration_range_drag.update_runtime_raw_drag_preview(bindings, ai) is True
    assert len(projected) == 2
    assert projected[0][0] is projected[1][0]
    assert projected[0][1] == ("raw_drag_preview", (3, 3), 1)
    assert projected[1][1] == ("raw_drag_preview", (3, 3), 2)

    integration_range_drag.reset_runtime_integration_drag(bindings, redraw=False)
    assert getattr(drag_state, "_raw_drag_preview_revision", None) == 2

    drag_state.active = True
    drag_state.mode = "raw"
    drag_state.x0 = 0.2
    drag_state.y0 = 1.1
    drag_state.x1 = 1.8
    drag_state.y1 = 2.0
    drag_state.tth0 = 20.0
    drag_state.phi0 = 0.0
    drag_state.tth1 = 32.0
    drag_state.phi1 = 12.0

    assert integration_range_drag.update_runtime_raw_drag_preview(bindings, ai) is True
    assert projected[2][1] == ("raw_drag_preview", (3, 3), 3)


def test_update_runtime_raw_drag_preview_prefers_source_extent_for_overlay_geometry() -> None:
    overlay = _FakeOverlay()
    drag_rect = _FakeRect()
    overlay_rect = _FakeRect()
    draw_calls = []
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="raw",
        x0=0.2,
        y0=1.1,
        x1=1.8,
        y1=2.0,
        tth0=20.0,
        phi0=0.0,
        tth1=32.0,
        phi1=12.0,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(
            extent=(100.0, 103.0, 203.0, 200.0),
            source_extent=(0.0, 3.0, 3.0, 0.0),
        ),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        draw_idle=lambda: draw_calls.append(True),
    )

    updated = integration_range_drag.update_runtime_raw_drag_preview(bindings, ai)

    assert updated is True
    assert overlay.visible is True
    assert overlay.extent == (0.0, 3.0, 3.0, 0.0)
    assert int(np.sum(overlay.data)) > 0
    assert draw_calls == [True]


def test_display_to_detector_angles_respects_inverted_detector_extent() -> None:
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [170.0, 175.0, -179.0],
            [160.0, 165.0, -175.0],
            [150.0, 155.0, -170.0],
        ],
        dtype=float,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0)),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 3.0, 3.0, 0.0)),
        get_detector_angular_maps=lambda ai: (two_theta, phi_vals),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )

    top_angles = integration_range_drag.display_to_detector_angles(
        bindings,
        0.2,
        2.8,
        object(),
    )
    bottom_angles = integration_range_drag.display_to_detector_angles(
        bindings,
        0.2,
        0.2,
        object(),
    )

    assert top_angles == (10.0, 170.0)
    assert bottom_angles == (30.0, 150.0)


def test_display_to_detector_angles_prefers_source_extent_over_live_extent() -> None:
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [170.0, 175.0, -179.0],
            [160.0, 165.0, -175.0],
            [150.0, 155.0, -170.0],
        ],
        dtype=float,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0)),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(
            extent=(100.0, 103.0, 203.0, 200.0),
            source_extent=(0.0, 3.0, 3.0, 0.0),
        ),
        get_detector_angular_maps=lambda ai: (two_theta, phi_vals),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )

    top_angles = integration_range_drag.display_to_detector_angles(
        bindings,
        0.2,
        2.8,
        object(),
    )
    bottom_angles = integration_range_drag.display_to_detector_angles(
        bindings,
        0.2,
        0.2,
        object(),
    )

    assert top_angles == (10.0, 170.0)
    assert bottom_angles == (30.0, 150.0)


def test_detector_preview_center_respects_inverted_detector_extent() -> None:
    two_theta = np.asarray(
        [
            [5.0, 1.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0)),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 3.0, 3.0, 0.0)),
        get_detector_angular_maps=lambda ai: (two_theta, two_theta),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )

    center = integration_range_drag._detector_preview_center(bindings, two_theta)

    assert center == (1.5, 2.5)


def test_detector_preview_center_prefers_source_extent_over_live_extent() -> None:
    two_theta = np.asarray(
        [
            [5.0, 1.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0)),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(
            extent=(100.0, 103.0, 203.0, 200.0),
            source_extent=(0.0, 3.0, 3.0, 0.0),
        ),
        get_detector_angular_maps=lambda ai: (two_theta, two_theta),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )

    center = integration_range_drag._detector_preview_center(bindings, two_theta)

    assert center == (1.5, 2.5)


def test_runtime_integration_region_visuals_callback_uses_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    bindings = object()

    monkeypatch.setattr(
        integration_range_drag,
        "refresh_runtime_integration_region_visuals",
        lambda bound: calls.append(bound),
    )

    callback = integration_range_drag.make_runtime_integration_region_visuals_callback(
        lambda: bindings
    )
    callback()

    assert calls == [bindings]


def test_integration_range_drag_runtime_helpers_handle_suppress_and_caked_drag() -> None:
    axis = _FakeAxis()
    drag_state = state.IntegrationRangeDragState()
    peak_state = state.PeakSelectionState(suppress_drag_press_once=True)
    view_state = _range_view_state()
    drag_rect = _FakeRect()
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    overlay.visible = True
    overlay_rect.visible = True
    status_messages = []
    sync_calls = []
    draw_calls = []
    schedule_calls = []
    show_1d_var = _FakeVar(False)

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=peak_state,
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: ai,
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        show_1d_var=show_1d_var,
        sync_peak_selection_state=lambda: sync_calls.append(True),
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: None,
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    suppressed = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=5.0, ydata=-10.0),
    )
    assert suppressed is True
    assert peak_state.suppress_drag_press_once is False
    assert sync_calls == [True]

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=5.0, ydata=-10.0),
    )
    assert started is True
    assert drag_state.active is True
    assert drag_state.mode == "caked"
    assert drag_rect.visible is True
    assert drag_rect.xy == (5.0, -10.0)
    assert overlay.visible is False
    assert overlay_rect.visible is False

    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=8.0, ydata=-4.0),
    )
    assert moved is True
    assert drag_state.x1 == 8.0
    assert drag_state.y1 == -4.0
    assert drag_rect.width == 3.0
    assert drag_rect.height == 6.0

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=8.0, ydata=-4.0),
    )
    assert released is True
    assert view_state.tth_min_var.get() == 5.0
    assert view_state.tth_max_var.get() == 8.0
    assert view_state.phi_min_var.get() == -10.0
    assert view_state.phi_max_var.get() == -4.0
    assert show_1d_var.get() is True
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[5.00, 8.00]°, φ=[-10.00, -4.00]°"
    assert drag_state.active is False
    assert drag_state.mode is None
    assert drag_rect.visible is False
    assert len(draw_calls) >= 3


def test_qz_bounds_from_caked_drag_for_qr_rod_uses_projected_trace_samples() -> None:
    projected_samples = {
        "two_theta": np.asarray([1.0, 2.0, 3.0, 3.8], dtype=float),
        "phi": np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=float),
        "qz": np.asarray([0.0, 1.0, 3.0, 8.0], dtype=float),
    }

    assert integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
        projected_samples,
        x0=1.5,
        y0=-0.5,
        x1=3.5,
        y1=1.5,
    ) == (1.0, 3.0)
    assert integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
        projected_samples,
        x0=1.5,
        y0=-0.5,
        x1=3.5,
        y1=1.5,
        phi_min=0.5,
        phi_max=1.5,
    ) == (3.0, 3.0)
    assert (
        integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
            projected_samples,
            x0=1.5,
            y0=5.0,
            x1=3.5,
            y1=6.0,
        )
        is None
    )


def test_qz_bounds_from_caked_drag_for_qr_rod_supports_phi_windows() -> None:
    projected_samples = {
        "two_theta": np.asarray([2.0, 2.0, 2.0], dtype=float),
        "phi": np.asarray([-80.0, 0.0, 80.0], dtype=float),
        "qz": np.asarray([-1.0, 0.0, 1.0], dtype=float),
    }
    phi_windows = ((-85.0, -72.5), (72.5, 85.0))

    assert (
        integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
            projected_samples,
            x0=1.5,
            y0=-5.0,
            x1=2.5,
            y1=5.0,
            phi_windows=phi_windows,
        )
        is None
    )
    assert integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
        projected_samples,
        x0=1.5,
        y0=75.0,
        x1=2.5,
        y1=85.0,
        phi_windows=phi_windows,
    ) == (1.0, 1.0)
    assert integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod(
        projected_samples,
        x0=1.5,
        y0=-85.0,
        x1=2.5,
        y1=-75.0,
        phi_windows=phi_windows,
    ) == (-1.0, -1.0)


def test_qz_bounds_from_caked_drag_for_qr_rod_bins_uses_lut_transpose(
    monkeypatch,
) -> None:
    radial_axis = np.asarray([1.0, 2.0, 3.0], dtype=float)
    azimuth_axis = np.asarray([-80.0, 0.0, 80.0], dtype=float)
    matrix = np.zeros((azimuth_axis.size * radial_axis.size, 6), dtype=np.float32)
    matrix[0 * radial_axis.size + 1, 1] = 1.0
    matrix[2 * radial_axis.size + 1, 4] = 1.0
    projection_context = _qr_drag_projection_context(
        matrix=matrix,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )
    qr_map = np.zeros((2, 3), dtype=float)
    qz_map = np.full((2, 3), 99.0, dtype=float)
    valid = np.zeros((2, 3), dtype=bool)
    qr_map[0, 1] = 0.25
    qz_map[0, 1] = -1.5
    valid[0, 1] = True
    qr_map[1, 1] = 0.25
    qz_map[1, 1] = 1.5
    valid[1, 1] = True
    monkeypatch.setattr(
        integration_range_drag.gui_qr_cylinder_overlay,
        "detector_qr_qz_maps_for_projection",
        lambda **_kwargs: (qr_map, qz_map, valid),
    )

    assert integration_range_drag.qz_bounds_from_caked_drag_for_qr_rod_bins(
        selected_entry={"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25},
        config=_qr_drag_config(),
        projection_context=projection_context,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        delta_qr=0.01,
        x0=1.5,
        x1=2.5,
        y0=-85.0,
        y1=-72.5,
        phi_windows=((-85.0, -72.5), (72.5, 85.0)),
    ) == (-1.5, -1.5)


def test_update_runtime_qr_rod_drag_preview_passes_mirrored_phi_windows(monkeypatch) -> None:
    axis = _FakeAxis(xlim=(0.0, 4.0), ylim=(-90.0, 90.0))
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="caked_qr_rod",
        x0=1.5,
        y0=75.0,
        x1=2.5,
        y1=85.0,
    )
    view_state = _range_view_state()
    view_state.phi_min_var.set(72.5)
    view_state.phi_max_var.set(85.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.mirror_selected_qr_phi_var = _FakeVar(True)
    view_state.delta_qr_var = _FakeVar(0.3)
    radial_axis = np.asarray([1.0, 2.0, 3.0], dtype=float)
    azimuth_axis = np.asarray([-80.0, 0.0, 80.0], dtype=float)
    captured_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        integration_range_drag.gui_qr_cylinder_overlay,
        "build_selected_qr_rod_qz_caked_mask",
        lambda **kwargs: (
            captured_calls.append(dict(kwargs))
            or {
                "mask": np.ones((azimuth_axis.size, radial_axis.size), dtype=bool),
                "signature": ("preview", kwargs["phi_windows"]),
            }
        ),
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        caked_qr_rod_drag_context_factory=lambda: {
            "selected_entry": {"key": ("phase-a", 1), "qr": 1.25},
            "config": object(),
            "projection_context": {"ctx": True},
            "projected_samples": {
                "two_theta": np.asarray([2.0], dtype=float),
                "phi": np.asarray([80.0], dtype=float),
                "qz": np.asarray([0.5], dtype=float),
            },
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
        },
    )

    assert integration_range_drag.update_runtime_qr_rod_drag_preview(bindings) is True
    assert captured_calls
    assert captured_calls[0]["phi_windows"] == ((-85.0, -72.5), (72.5, 85.0))


def test_selected_qr_rod_caked_drag_updates_only_qz_bounds(monkeypatch) -> None:
    axis = _FakeAxis(xlim=(0.0, 4.0), ylim=(-2.0, 2.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(20.0)
    view_state.phi_min_var.set(-5.0)
    view_state.phi_max_var.set(5.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.qz_min_var = _FakeVar(-9.0)
    view_state.qz_max_var = _FakeVar(9.0)
    view_state.delta_qr_var = _FakeVar(0.3)
    view_state.qz_min_label_var = _FakeVar("")
    view_state.qz_min_entry_var = _FakeVar("")
    view_state.qz_max_label_var = _FakeVar("")
    view_state.qz_max_entry_var = _FakeVar("")
    view_state.delta_qr_label_var = _FakeVar("")
    view_state.delta_qr_entry_var = _FakeVar("")
    radial_axis = np.asarray([1.0, 2.0, 3.0], dtype=float)
    azimuth_axis = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    projected_samples = {
        "two_theta": np.asarray([1.0, 2.0, 3.0, 3.8], dtype=float),
        "phi": np.asarray([-1.0, 0.0, 1.0, 0.0], dtype=float),
        "qz": np.asarray([0.0, 1.0, 3.0, 8.0], dtype=float),
    }
    monkeypatch.setattr(
        integration_range_drag.gui_qr_cylinder_overlay,
        "build_selected_qr_rod_qz_caked_mask",
        lambda **kwargs: {
            "mask": np.ones((azimuth_axis.size, radial_axis.size), dtype=bool),
            "signature": (
                "preview",
                round(float(kwargs["qz_min"]), 4),
                round(float(kwargs["qz_max"]), 4),
            ),
        },
    )
    overlay = _FakeOverlay()
    schedule_calls = []
    status_messages = []
    show_1d_var = _FakeVar(False)

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: None,
        set_status_text=status_messages.append,
        caked_qr_rod_drag_context_factory=lambda: {
            "qr": 1.25,
            "selected_entry": {"key": ("phase-a", 1), "qr": 1.25},
            "config": object(),
            "projection_context": {"ctx": True},
            "projected_samples": projected_samples,
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
        },
    )

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.5, ydata=-0.5),
    )
    assert started is True
    assert drag_state.active is True
    assert drag_state.mode == "caked_qr_rod"
    assert "Rectangle drag is disabled" not in " ".join(status_messages)

    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=3.5, ydata=1.5),
    )
    assert moved is True
    assert overlay.data is not None

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=3.5, ydata=1.5),
    )

    assert released is True
    assert drag_state.active is False
    assert drag_state.mode is None
    assert view_state.qz_min_var.get() == 1.0
    assert view_state.qz_max_var.get() == 3.0
    assert view_state.qz_min_entry_var.get() == "1.0000"
    assert view_state.qz_max_entry_var.get() == "3.0000"
    assert view_state.tth_min_var.get() == 10.0
    assert view_state.tth_max_var.get() == 20.0
    assert view_state.phi_min_var.get() == -5.0
    assert view_state.phi_max_var.get() == 5.0
    assert show_1d_var.get() is True
    assert schedule_calls == [True]
    assert status_messages[-1] == "Selected Qr rod Qz range set: Qz=[1.0000, 3.0000] A^-1"


def test_selected_qr_rod_caked_drag_without_qr_intersection_leaves_controls(
    monkeypatch,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 4.0), ylim=(-2.0, 2.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(20.0)
    view_state.phi_min_var.set(-5.0)
    view_state.phi_max_var.set(5.0)
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.qz_min_var = _FakeVar(-9.0)
    view_state.qz_max_var = _FakeVar(9.0)
    view_state.delta_qr_var = _FakeVar(0.3)
    radial_axis = np.asarray([1.0, 2.0, 3.0], dtype=float)
    azimuth_axis = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    projected_samples = {
        "two_theta": np.asarray([0.2, 3.8], dtype=float),
        "phi": np.asarray([-1.8, 1.8], dtype=float),
        "qz": np.asarray([-2.0, 2.0], dtype=float),
    }
    schedule_calls = []
    status_messages = []

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: None,
        set_status_text=status_messages.append,
        caked_qr_rod_drag_context_factory=lambda: {
            "qr": 1.25,
            "selected_entry": {"key": ("phase-a", 1), "qr": 1.25},
            "config": object(),
            "projection_context": {"ctx": True},
            "projected_samples": projected_samples,
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
        },
    )

    assert integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.5, ydata=-0.5),
    )
    assert integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=3.5, ydata=1.5),
    )
    assert integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=3.5, ydata=1.5),
    )

    assert drag_state.active is False
    assert drag_state.mode is None
    assert view_state.qz_min_var.get() == -9.0
    assert view_state.qz_max_var.get() == 9.0
    assert view_state.tth_min_var.get() == 10.0
    assert view_state.tth_max_var.get() == 20.0
    assert view_state.phi_min_var.get() == -5.0
    assert view_state.phi_max_var.get() == 5.0
    assert schedule_calls == []
    assert status_messages[-1] == "Drag across the selected Qr rod to set a Qz range."


def test_raw_release_with_incomplete_drag_restores_current_region_visuals() -> None:
    axis = _FakeAxis(xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="raw",
        tth0=10.0,
        phi0=-10.0,
        tth1=None,
        phi1=None,
    )
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    draw_calls = []
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
    )

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=None, xdata=None, ydata=None),
    )

    assert released is True
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 6
    assert drag_state.active is False
    assert drag_state.mode is None
    assert len(draw_calls) >= 1


def test_reset_runtime_integration_drag_hides_raw_preview_overlay() -> None:
    drag_state = state.IntegrationRangeDragState(active=True, mode="raw")
    drag_rect = _FakeRect()
    drag_rect.visible = True
    overlay = _FakeOverlay()
    overlay.visible = True
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.reset_runtime_integration_drag(bindings, redraw=False)

    assert drag_state.active is False
    assert drag_state.mode is None
    assert drag_rect.visible is False
    assert overlay.visible is False


def test_integration_range_drag_runtime_helpers_handle_raw_drag_and_callback_bundle(
    monkeypatch,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 2.0), ylim=(2.0, 0.0))
    drag_state = state.IntegrationRangeDragState()
    peak_state = state.PeakSelectionState()
    view_state = _range_view_state()
    drag_rect = _FakeRect()
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    status_messages = []
    schedule_calls = []
    draw_calls = []

    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    ai = object()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=peak_state,
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        sync_peak_selection_state=lambda: None,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=1.1),
    )
    assert started is True
    assert drag_state.active is True
    assert drag_state.mode == "raw"
    assert np.isclose(drag_state.tth0, 20.0)
    assert np.isclose(drag_state.phi0, 0.0)
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) > 0
    assert overlay_rect.visible is False
    assert drag_rect.visible is False

    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=0.2),
    )
    assert moved is True
    assert np.isclose(drag_state.tth1, 32.0)
    assert np.isclose(drag_state.phi1, 12.0)
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) > 0
    assert drag_rect.visible is False

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=0.2),
    )
    assert released is True
    assert view_state.tth_min_var.get() == 20.0
    assert view_state.tth_max_var.get() == 32.0
    assert view_state.phi_min_var.get() == 0.0
    assert view_state.phi_max_var.get() == 12.0
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[20.00, 32.00]°, φ=[0.00, 12.00]°"
    assert drag_state.active is False
    assert drag_state.mode is None
    assert drag_rect.visible is False
    assert overlay.visible is True
    assert overlay.extent == (0.0, 2.0, 2.0, 0.0)
    assert int(np.sum(overlay.data)) > 0
    assert len(draw_calls) >= 3

    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_press",
        lambda bindings_arg, event: callback_calls.append(("press", bindings_arg, event)) or True,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_motion",
        lambda bindings_arg, event: callback_calls.append(("motion", bindings_arg, event)) or False,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_release",
        lambda bindings_arg, event: callback_calls.append(("release", bindings_arg, event)) or True,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "reset_runtime_integration_drag",
        lambda bindings_arg, *, redraw=True: callback_calls.append(("reset", bindings_arg, redraw)),
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = integration_range_drag.make_runtime_integration_range_drag_callbacks(build_bindings)

    press_event = _FakeEvent(button=1)
    motion_event = _FakeEvent(button=1)
    release_event = _FakeEvent(button=1)
    assert callbacks.on_press(press_event) is True
    assert callbacks.on_motion(motion_event) is False
    assert callbacks.on_release(release_event) is True
    callbacks.reset()

    assert callback_calls == [
        ("press", "bindings-1", press_event),
        ("motion", "bindings-2", motion_event),
        ("release", "bindings-3", release_event),
        ("reset", "bindings-4", True),
    ]


def test_integration_range_drag_callbacks_cancel_runtime_errors(monkeypatch) -> None:
    status_messages = []
    draw_calls = []
    reset_calls = []
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(active=True, mode="raw"),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=lambda text: status_messages.append(str(text)),
    )

    def _raise_runtime_error(_bindings_arg, _event):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_motion",
        _raise_runtime_error,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "reset_runtime_integration_drag",
        lambda bindings_arg, *, redraw=True: reset_calls.append((bindings_arg, redraw)),
    )
    monkeypatch.setattr(integration_range_drag.traceback, "print_exc", lambda: None)

    callbacks = integration_range_drag.make_runtime_integration_range_drag_callbacks(
        lambda: bindings
    )

    assert callbacks.on_motion(_FakeEvent(button=1)) is False
    assert reset_calls == [(bindings, False)]
    assert status_messages == ["Integration drag canceled after an internal error."]
    assert draw_calls == [True]


def test_detector_phi_mask_and_bounds_support_wrapped_short_arc() -> None:
    phi_vals = np.asarray(
        [-179.0, -175.0, -170.0, -10.0, 0.0, 10.0, 170.0, 175.0, 179.0],
        dtype=float,
    )

    mask = integration_range_drag.detector_phi_mask(phi_vals, 170.0, -170.0)
    bounds = integration_range_drag._sorted_detector_angle_bounds(
        12.0,
        -170.0,
        4.0,
        170.0,
    )

    assert np.allclose(phi_vals[mask], [-179.0, -175.0, -170.0, 170.0, 175.0, 179.0])
    assert bounds == (4.0, 12.0, 170.0, -170.0)


def test_update_detector_integration_overlay_uses_wrapped_phi_interval() -> None:
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [170.0, 175.0, -179.0],
            [150.0, 0.0, -150.0],
            [165.0, 169.0, -170.0],
        ],
        dtype=float,
    )
    overlay = _FakeOverlay()
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai: (two_theta, phi_vals),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
    )

    updated = integration_range_drag._update_detector_integration_overlay(
        bindings,
        ai=object(),
        tth_min=0.0,
        tth_max=40.0,
        phi_min=170.0,
        phi_max=-170.0,
    )

    assert updated is True
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 4


def test_raw_drag_release_preserves_wrapped_short_arc_interval() -> None:
    axis = _FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = state.IntegrationRangeControlsViewState(
        tth_min_var=_FakeVar(0.0),
        tth_max_var=_FakeVar(0.0),
        phi_min_var=_FakeVar(0.0),
        phi_max_var=_FakeVar(0.0),
        tth_min_slider=_FakeSlider(0.0, 40.0),
        phi_min_slider=_FakeSlider(-180.0, 180.0),
    )
    overlay = _FakeOverlay()
    schedule_calls = []
    status_messages = []
    draw_calls = []
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [170.0, 175.0, -179.0],
            [160.0, 165.0, -175.0],
            [150.0, 155.0, -170.0],
        ],
        dtype=float,
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 3.0, 3.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=2.8),
    )
    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=2.8, ydata=2.8),
    )
    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=2.8, ydata=2.8),
    )

    assert started is True
    assert moved is True
    assert released is True
    assert view_state.tth_min_var.get() == 10.0
    assert view_state.tth_max_var.get() == 12.0
    assert view_state.phi_min_var.get() == 170.0
    assert view_state.phi_max_var.get() == -179.0
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[10.00, 12.00]°, φ=[170.00, -179.00]°"
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 3
    assert len(draw_calls) >= 3


def test_detector_qr_rod_drag_sets_qz_bounds_without_detector_angles(
    monkeypatch,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = _range_view_state()
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.qz_min_var = _FakeVar(0.0)
    view_state.qz_max_var = _FakeVar(0.0)
    view_state.delta_qr_var = _FakeVar(0.05)
    show_1d_var = _FakeVar(False)
    schedule_calls = []
    status_messages = []
    draw_calls = []
    context_calls = []
    qz_map = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    context = {
        "detector_shape": (3, 3),
        "qr_map": np.ones((3, 3), dtype=float),
        "qz_map": qz_map,
        "valid_q": np.ones((3, 3), dtype=bool),
        "detector_phi_deg": None,
        "qr_center": 1.0,
        "delta_qr": 0.05,
        "phi_windows": ((-180.0, 180.0),),
        "shape_mask": None,
        "signature": ("drag-context", 1),
    }

    angle_calls = []
    monkeypatch.setattr(
        integration_range_drag,
        "display_to_detector_angles",
        lambda *args, **kwargs: (
            angle_calls.append((args, kwargs))
            or (_ for _ in ()).throw(AssertionError("raw detector angle path used"))
        ),
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 3.0, 3.0, 0.0)),
        get_detector_angular_maps=lambda ai: (_ for _ in ()).throw(AssertionError()),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
        detector_qr_rod_drag_context_factory=lambda: context_calls.append("shared") or context,
    )

    assert integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=2.8),
    )
    assert drag_state.mode == "detector_qr_rod"
    assert integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=1.1),
    )
    assert integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=1.1),
    )

    assert angle_calls == []
    assert view_state.qz_min_var.get() == 0.0
    assert view_state.qz_max_var.get() == 8.0
    assert show_1d_var.get() is True
    assert schedule_calls == [True]
    assert status_messages[-1] == "Selected Qr rod detector Qz range set: Qz=[0.0000, 8.0000] A^-1"
    assert context_calls == ["shared", "shared", "shared"]
    assert draw_calls


def test_detector_qr_rod_drag_no_hit_preserves_qz_bounds_and_emits_status(
    monkeypatch,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 3.0), ylim=(3.0, 0.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = _range_view_state()
    view_state.integrate_selected_qr_rod_var = _FakeVar(True)
    view_state.qz_min_var = _FakeVar(1.0)
    view_state.qz_max_var = _FakeVar(3.0)
    show_1d_var = _FakeVar(False)
    schedule_calls = []
    status_messages = []
    draw_calls = []
    context_calls = []
    qz_map = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=float,
    )
    context = {
        "detector_shape": (3, 3),
        "qr_map": np.ones((3, 3), dtype=float),
        "qz_map": qz_map,
        "valid_q": np.ones((3, 3), dtype=bool),
        "detector_phi_deg": None,
        "qr_center": 1.0,
        "delta_qr": 0.05,
        "phi_windows": ((-180.0, 180.0),),
        "shape_mask": None,
        "signature": ("drag-context", "empty"),
        "union_support_mask": np.zeros((3, 3), dtype=bool),
    }
    monkeypatch.setattr(
        integration_range_drag,
        "display_to_detector_angles",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("raw detector angle path used")
        ),
    )
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(extent=(0.0, 3.0, 3.0, 0.0)),
        get_detector_angular_maps=lambda ai: (_ for _ in ()).throw(AssertionError()),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: object(),
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
        detector_qr_rod_drag_context_factory=lambda: context_calls.append("shared") or context,
    )

    assert integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=2.8),
    )
    assert integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=1.1),
    )
    assert integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=1.1),
    )

    assert view_state.qz_min_var.get() == 1.0
    assert view_state.qz_max_var.get() == 3.0
    assert show_1d_var.get() is False
    assert schedule_calls == []
    assert (
        status_messages[-1] == "Drag across the selected Qr rod detector support to set a Qz range."
    )
    assert context_calls == ["shared", "shared", "shared"]
    assert draw_calls


def test_set_runtime_integration_range_from_drag_without_controls_uses_cached_state() -> None:
    view_state = state.IntegrationRangeControlsViewState()
    schedule_calls = []
    status_messages = []
    show_1d_var = _FakeVar(False)
    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=_FakeOverlay(),
        integration_region_rect=_FakeRect(),
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        set_status_text=status_messages.append,
    )

    applied = integration_range_drag.set_runtime_integration_range_from_drag(
        bindings,
        18.0,
        -5.0,
        12.0,
        4.0,
    )

    assert applied is True
    assert view_state.tth_min_value == 12.0
    assert view_state.tth_max_value == 18.0
    assert view_state.phi_min_value == -5.0
    assert view_state.phi_max_value == 4.0
    assert show_1d_var.get() is True
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[12.00, 18.00]°, φ=[-5.00, 4.00]°"


def test_raw_drag_release_without_controls_caches_range_values() -> None:
    axis = _FakeAxis(xlim=(0.0, 2.0), ylim=(2.0, 0.0))
    drag_state = state.IntegrationRangeDragState()
    view_state = state.IntegrationRangeControlsViewState()
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    schedule_calls = []
    status_messages = []
    draw_calls = []
    show_1d_var = _FakeVar(False)
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    ai = object()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        show_1d_var=show_1d_var,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    assert integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=1.1),
    )
    assert integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=0.2),
    )
    assert integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=0.2),
    )

    assert view_state.tth_min_value == 20.0
    assert view_state.tth_max_value == 32.0
    assert view_state.phi_min_value == 0.0
    assert view_state.phi_max_value == 12.0
    assert show_1d_var.get() is True
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[20.00, 32.00]°, φ=[0.00, 12.00]°"
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) > 0
    assert len(draw_calls) >= 3
