"""Shared GUI startup helpers for launcher and runtime entrypoints."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


LAUNCHER_COMMANDS = {"gui", "calibrant"}


@dataclass(frozen=True)
class RuntimeBindingsCallbacksBootstrap:
    """Zero-arg runtime bindings factory plus one bound callback bundle."""

    bindings_factory: Callable[[], Any]
    callbacks: Any


@dataclass(frozen=True)
class RuntimeBindingsRefreshToggleBootstrap:
    """Zero-arg runtime bindings factory plus refresh/toggle callbacks."""

    bindings_factory: Callable[[], Any]
    refresh: Callable[..., Any]
    toggle: Callable[..., Any]


@dataclass(frozen=True)
class BraggQrRuntimeBootstrap:
    """Runtime Bragg-Qr bindings plus the shared window callbacks."""

    bindings_factory: Callable[[], Any]
    refresh_window: Callable[..., Any]
    open_window: Callable[..., Any]


@dataclass(frozen=True)
class StructureFactorPruningRuntimeBootstrap:
    """Runtime pruning bindings plus the derived callback helpers."""

    bindings_factory: Callable[[], Any]
    current_sf_prune_bias: Callable[[], Any]
    current_solve_q_values: Callable[[], Any]
    update_status_label: Callable[[], Any]
    apply_filters: Callable[..., Any]
    on_sf_prune_bias_change: Callable[..., Any]
    on_solve_q_steps_change: Callable[..., Any]
    on_solve_q_rel_tol_change: Callable[..., Any]
    set_solve_q_control_states: Callable[[], Any]
    on_solve_q_mode_change: Callable[..., Any]


@dataclass(frozen=True)
class PeakSelectionRuntimeBootstrap:
    """Selected-peak runtime callbacks plus the shared overlay-data callback."""

    bindings_factory: Callable[[], Any]
    callbacks: Any
    ensure_peak_overlay_data: Callable[..., object]
    maintenance_callbacks: Any


@dataclass(frozen=True)
class GeometryFitActionRuntimeBootstrap:
    """Geometry-fit action bindings plus the shared fit-button callback."""

    bindings_factory: Callable[[], Any]
    callback: Callable[..., Any]


@dataclass(frozen=True)
class HklLookupControlsRuntimeBootstrap:
    """HKL lookup control wiring plus shared refresh/pick-mode callbacks."""

    create_controls: Callable[[Any], None]
    refresh_controls: Callable[[], None]
    set_hkl_pick_mode: Callable[[bool, str | None], None]


@dataclass(frozen=True)
class GeometryToolActionsRuntimeBootstrap:
    """Geometry tool action control wiring extracted from the runtime."""

    create_controls: Callable[[Any], None]


@dataclass(frozen=True)
class IntegrationRangeDragRuntimeBootstrap:
    """Integration-range callbacks plus the shared visual-refresh callback."""

    bindings_factory: Callable[[], Any]
    callbacks: Any
    refresh_visuals: Callable[[], None]


@dataclass(frozen=True)
class BraggQrWorkflowRuntimeBootstrap:
    """Live pruning and Bragg-Qr manager callbacks wired as one workflow."""

    pruning_bindings_factory: Callable[[], Any]
    manager_bindings_factory: Callable[[], Any]
    current_sf_prune_bias: Callable[[], Any]
    current_solve_q_values: Callable[[], Any]
    update_status_label: Callable[[], Any]
    apply_filters: Callable[..., Any]
    on_sf_prune_bias_change: Callable[..., Any]
    on_solve_q_steps_change: Callable[..., Any]
    on_solve_q_rel_tol_change: Callable[..., Any]
    set_solve_q_control_states: Callable[[], Any]
    on_solve_q_mode_change: Callable[..., Any]
    refresh_window: Callable[..., Any]
    open_window: Callable[..., Any]


def extract_bundle_arg(argv: Sequence[str]) -> str | None:
    """Extract an optional ``--bundle`` value from CLI args."""

    for idx, token in enumerate(argv):
        if token.startswith("--bundle="):
            value = token.split("=", 1)[1].strip()
            return value or None
        if token == "--bundle":
            if idx + 1 < len(argv):
                value = str(argv[idx + 1]).strip()
                if value and not value.startswith("-"):
                    return value
            return None
    return None


def quick_startup_mode_dialog() -> str | None:
    """Ask startup mode before heavy simulation initialization."""

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        # Headless or Tk unavailable: default to simulation behavior.
        return "simulation"

    choice = {"mode": None}

    launcher = tk.Tk()
    launcher.title("RA-SIM Launcher")
    launcher.configure(bg="#e8eef5")
    launcher.resizable(False, False)
    launcher.attributes("-topmost", True)

    try:
        style = ttk.Style(launcher)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    panel = tk.Frame(
        launcher,
        bg="#ffffff",
        highlightbackground="#d4deeb",
        highlightthickness=1,
    )
    panel.pack(fill="both", expand=True, padx=18, pady=18)

    tk.Label(
        panel,
        text="Choose Startup Mode",
        font=("Segoe UI", 16, "bold"),
        fg="#0f172a",
        bg="#ffffff",
    ).pack(anchor="w", padx=18, pady=(18, 4))
    tk.Label(
        panel,
        text="Select what you want to run right now.",
        font=("Segoe UI", 10),
        fg="#475569",
        bg="#ffffff",
    ).pack(anchor="w", padx=18, pady=(0, 14))

    def _set_mode(mode_name: str | None) -> None:
        choice["mode"] = mode_name
        launcher.destroy()

    sim_btn = tk.Button(
        panel,
        text="Run Simulation GUI",
        font=("Segoe UI", 11, "bold"),
        bg="#2563eb",
        fg="#ffffff",
        activebackground="#1d4ed8",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=14,
        pady=10,
        cursor="hand2",
        command=lambda: _set_mode("simulation"),
    )
    sim_btn.pack(fill="x", padx=18, pady=(0, 6))
    tk.Label(
        panel,
        text="Full RA-SIM simulation workspace and controls.",
        font=("Segoe UI", 9),
        fg="#64748b",
        bg="#ffffff",
    ).pack(anchor="w", padx=20, pady=(0, 12))

    cal_btn = tk.Button(
        panel,
        text="Fit Calibrant (hBN Fitter)",
        font=("Segoe UI", 11, "bold"),
        bg="#0f766e",
        fg="#ffffff",
        activebackground="#0f5f59",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        padx=14,
        pady=10,
        cursor="hand2",
        command=lambda: _set_mode("calibrant"),
    )
    cal_btn.pack(fill="x", padx=18, pady=(0, 6))
    tk.Label(
        panel,
        text="Open the calibrant fitting tool directly.",
        font=("Segoe UI", 9),
        fg="#64748b",
        bg="#ffffff",
    ).pack(anchor="w", padx=20, pady=(0, 16))

    footer = tk.Frame(panel, bg="#ffffff")
    footer.pack(fill="x", padx=18, pady=(0, 16))
    tk.Label(
        footer,
        text="Keyboard: 1 = Simulation, 2 = Calibrant, Esc = Cancel",
        font=("Segoe UI", 8),
        fg="#64748b",
        bg="#ffffff",
    ).pack(side="left")
    tk.Button(
        footer,
        text="Cancel",
        font=("Segoe UI", 9),
        bg="#e2e8f0",
        fg="#1e293b",
        activebackground="#cbd5e1",
        activeforeground="#1e293b",
        relief="flat",
        bd=0,
        padx=12,
        pady=6,
        cursor="hand2",
        command=lambda: _set_mode(None),
    ).pack(side="right")

    launcher.bind("<Escape>", lambda _event: _set_mode(None))
    launcher.bind("<Return>", lambda _event: _set_mode("simulation"))
    launcher.bind("1", lambda _event: _set_mode("simulation"))
    launcher.bind("2", lambda _event: _set_mode("calibrant"))
    launcher.protocol("WM_DELETE_WINDOW", lambda: _set_mode(None))

    launcher.update_idletasks()
    width = max(launcher.winfo_width(), 480)
    height = max(launcher.winfo_height(), 330)
    x = launcher.winfo_screenwidth() // 2 - width // 2
    y = launcher.winfo_screenheight() // 2 - height // 2
    launcher.geometry(f"{width}x{height}+{x}+{y}")
    sim_btn.focus_set()
    launcher.mainloop()

    mode = choice["mode"]
    if mode in {"simulation", "calibrant"}:
        return mode
    return None


def choose_startup_mode_dialog(root: Any) -> str | None:
    """Ask whether to start the simulation GUI or calibrant fitter."""

    import tkinter as tk
    from tkinter import ttk

    selection = {"mode": None}

    try:
        root.withdraw()
    except tk.TclError:
        pass

    chooser = tk.Toplevel(root)
    chooser.title("RA-SIM Startup")
    chooser.resizable(False, False)
    chooser.transient(root)
    chooser.grab_set()

    frame = ttk.Frame(chooser, padding=14)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frame,
        text="Choose startup mode:",
        font=("Segoe UI", 11, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        frame,
        text="Select calibrant fitting or full simulation GUI.",
    ).pack(anchor="w", pady=(4, 10))

    def _set_mode(mode_name: str | None) -> None:
        selection["mode"] = mode_name
        try:
            chooser.grab_release()
        except tk.TclError:
            pass
        chooser.destroy()

    simulation_btn = ttk.Button(
        frame,
        text="Run Simulation GUI",
        command=lambda: _set_mode("simulation"),
    )
    simulation_btn.pack(fill=tk.X, pady=2)

    ttk.Button(
        frame,
        text="Fit Calibrant (hBN Fitter)",
        command=lambda: _set_mode("calibrant"),
    ).pack(fill=tk.X, pady=2)

    ttk.Button(
        frame,
        text="Cancel",
        command=lambda: _set_mode(None),
    ).pack(fill=tk.X, pady=(8, 0))

    chooser.protocol("WM_DELETE_WINDOW", lambda: _set_mode(None))
    chooser.bind("<Escape>", lambda _event: _set_mode(None))
    chooser.bind("<Return>", lambda _event: _set_mode("simulation"))

    chooser.update_idletasks()
    width = chooser.winfo_width()
    height = chooser.winfo_height()
    x = chooser.winfo_screenwidth() // 2 - width // 2
    y = chooser.winfo_screenheight() // 2 - height // 2
    chooser.geometry(f"{width}x{height}+{x}+{y}")

    simulation_btn.focus_set()
    root.wait_window(chooser)
    return selection["mode"]


def launch_calibrant_gui(*, bundle: str | None = None) -> None:
    """Launch the calibrant fitter GUI."""

    import tkinter as tk

    try:
        from hbn_fitter.fitter import HBNFitterGUI
    except Exception as exc:
        raise RuntimeError(
            "Unable to import hbn_fitter GUI. Ensure `hbn_fitter/fitter.py` exists."
        ) from exc

    root = tk.Tk()
    _ = HBNFitterGUI(root, startup_bundle=bundle)
    root.mainloop()


def should_forward_to_cli(argv: Sequence[str]) -> bool:
    """Return whether argv targets the shared CLI instead of the GUI launcher."""

    if not argv:
        return False
    first_arg = argv[0]
    return (
        first_arg not in LAUNCHER_COMMANDS.union({"-h", "--help"})
        and not first_arg.startswith("--")
    )


def build_launch_arg_parser() -> argparse.ArgumentParser:
    """Build the shared GUI launcher argument parser."""

    parser = argparse.ArgumentParser(description="RA Simulation launcher")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(LAUNCHER_COMMANDS),
        help=(
            "Optional startup mode: 'gui' runs simulation directly; "
            "'calibrant' launches the hBN fitter directly."
        ),
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write the initial intensity Excel file",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional NPZ bundle path to preload in calibrant mode.",
    )
    return parser


def parse_launch_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse shared GUI launcher arguments."""

    return build_launch_arg_parser().parse_args(list(argv))


def resolve_startup_mode(
    command: str | None,
    *,
    early_mode: str | None = None,
) -> str:
    """Resolve the launcher command and early bootstrap state to a startup mode."""

    if command == "gui":
        return "simulation"
    if command == "calibrant":
        return "calibrant"
    if early_mode in {"simulation", "calibrant"}:
        return early_mode
    return "prompt"


def build_runtime_structure_factor_pruning_bootstrap(
    *,
    structure_factor_pruning_module: Any,
    uniform_flag: int,
    adaptive_flag: int,
    **bindings_kwargs: Any,
) -> StructureFactorPruningRuntimeBootstrap:
    """Build the live pruning/solve-q runtime callback surface."""

    bindings_factory = (
        structure_factor_pruning_module.make_runtime_structure_factor_pruning_bindings_factory(
            **bindings_kwargs
        )
    )
    return StructureFactorPruningRuntimeBootstrap(
        bindings_factory=bindings_factory,
        current_sf_prune_bias=(
            structure_factor_pruning_module.make_runtime_current_sf_prune_bias_callback(
                bindings_factory
            )
        ),
        current_solve_q_values=(
            structure_factor_pruning_module.make_runtime_current_solve_q_values_callback(
                bindings_factory,
                uniform_flag=uniform_flag,
                adaptive_flag=adaptive_flag,
            )
        ),
        update_status_label=(
            structure_factor_pruning_module.make_runtime_structure_factor_pruning_status_callback(
                bindings_factory
            )
        ),
        apply_filters=structure_factor_pruning_module.make_runtime_bragg_qr_filter_callback(
            bindings_factory
        ),
        on_sf_prune_bias_change=(
            structure_factor_pruning_module.make_runtime_sf_prune_bias_change_callback(
                bindings_factory
            )
        ),
        on_solve_q_steps_change=(
            structure_factor_pruning_module.make_runtime_solve_q_steps_change_callback(
                bindings_factory
            )
        ),
        on_solve_q_rel_tol_change=(
            structure_factor_pruning_module.make_runtime_solve_q_rel_tol_change_callback(
                bindings_factory
            )
        ),
        set_solve_q_control_states=(
            structure_factor_pruning_module.make_runtime_solve_q_control_states_callback(
                bindings_factory
            )
        ),
        on_solve_q_mode_change=(
            structure_factor_pruning_module.make_runtime_solve_q_mode_change_callback(
                bindings_factory
            )
        ),
    )


def build_runtime_bragg_qr_bootstrap(
    *,
    bragg_qr_manager_module: Any,
    root: Any,
    **bindings_kwargs: Any,
) -> BraggQrRuntimeBootstrap:
    """Build the live Bragg-Qr manager binding and window callbacks."""

    bindings_factory = bragg_qr_manager_module.make_runtime_bragg_qr_bindings_factory(
        **bindings_kwargs
    )
    refresh_window = bragg_qr_manager_module.make_runtime_bragg_qr_refresh_callback(
        bindings_factory
    )
    return BraggQrRuntimeBootstrap(
        bindings_factory=bindings_factory,
        refresh_window=refresh_window,
        open_window=bragg_qr_manager_module.make_runtime_bragg_qr_open_callback(
            root=root,
            bindings_factory=bindings_factory,
        ),
    )


def build_runtime_bragg_qr_workflow_bootstrap(
    *,
    structure_factor_pruning_module: Any,
    bragg_qr_manager_module: Any,
    root: Any,
    uniform_flag: int,
    adaptive_flag: int,
    structure_factor_pruning_view_state_factory: object,
    bragg_qr_view_state: Any,
    simulation_runtime_state: Any,
    bragg_qr_manager_state: Any,
    clip_prune_bias: Callable[[object], float],
    clip_solve_q_steps: Callable[[object], int],
    clip_solve_q_rel_tol: Callable[[object], float],
    normalize_solve_q_mode_label: Callable[[object], str],
    schedule_update_factory: object | None = None,
    primary_candidate: object = None,
    primary_fallback: object = None,
    secondary_candidate: object | None = None,
    set_progress_text_factory: Callable[[], Callable[[str], None] | None] | None = None,
    invalid_key: int = 0,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> BraggQrWorkflowRuntimeBootstrap:
    """Build the coupled pruning and Bragg-Qr manager workflow callbacks."""

    manager_refresh_window: Callable[..., Any] | None = None
    pruning_runtime = build_runtime_structure_factor_pruning_bootstrap(
        structure_factor_pruning_module=structure_factor_pruning_module,
        uniform_flag=uniform_flag,
        adaptive_flag=adaptive_flag,
        view_state_factory=structure_factor_pruning_view_state_factory,
        simulation_runtime_state=simulation_runtime_state,
        bragg_qr_manager_state=bragg_qr_manager_state,
        clip_prune_bias=clip_prune_bias,
        clip_solve_q_steps=clip_solve_q_steps,
        clip_solve_q_rel_tol=clip_solve_q_rel_tol,
        normalize_solve_q_mode_label=normalize_solve_q_mode_label,
        schedule_update_factory=schedule_update_factory,
        refresh_window_factory=lambda: manager_refresh_window,
    )
    bragg_qr_runtime = build_runtime_bragg_qr_bootstrap(
        bragg_qr_manager_module=bragg_qr_manager_module,
        root=root,
        view_state=bragg_qr_view_state,
        manager_state=bragg_qr_manager_state,
        simulation_runtime_state=simulation_runtime_state,
        primary_candidate=primary_candidate,
        primary_fallback=primary_fallback,
        secondary_candidate=secondary_candidate,
        apply_filters=lambda: pruning_runtime.apply_filters(trigger_update=True),
        set_progress_text_factory=set_progress_text_factory,
        invalid_key=invalid_key,
        tcl_error_types=tcl_error_types,
    )
    manager_refresh_window = bragg_qr_runtime.refresh_window
    return BraggQrWorkflowRuntimeBootstrap(
        pruning_bindings_factory=pruning_runtime.bindings_factory,
        manager_bindings_factory=bragg_qr_runtime.bindings_factory,
        current_sf_prune_bias=pruning_runtime.current_sf_prune_bias,
        current_solve_q_values=pruning_runtime.current_solve_q_values,
        update_status_label=pruning_runtime.update_status_label,
        apply_filters=pruning_runtime.apply_filters,
        on_sf_prune_bias_change=pruning_runtime.on_sf_prune_bias_change,
        on_solve_q_steps_change=pruning_runtime.on_solve_q_steps_change,
        on_solve_q_rel_tol_change=pruning_runtime.on_solve_q_rel_tol_change,
        set_solve_q_control_states=pruning_runtime.set_solve_q_control_states,
        on_solve_q_mode_change=pruning_runtime.on_solve_q_mode_change,
        refresh_window=bragg_qr_runtime.refresh_window,
        open_window=bragg_qr_runtime.open_window,
    )


def build_runtime_qr_cylinder_overlay_bootstrap(
    *,
    qr_cylinder_overlay_module: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsRefreshToggleBootstrap:
    """Build the live Qr-cylinder overlay runtime callback surface."""

    bindings_factory = (
        qr_cylinder_overlay_module.make_runtime_qr_cylinder_overlay_bindings_factory(
            **bindings_kwargs
        )
    )
    return RuntimeBindingsRefreshToggleBootstrap(
        bindings_factory=bindings_factory,
        refresh=qr_cylinder_overlay_module.make_runtime_qr_cylinder_overlay_refresh_callback(
            bindings_factory
        ),
        toggle=qr_cylinder_overlay_module.make_runtime_qr_cylinder_overlay_toggle_callback(
            bindings_factory
        ),
    )


def build_runtime_peak_selection_bootstrap(
    *,
    peak_selection_module: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsCallbacksBootstrap:
    """Build the live selected-peak runtime binding/callback bundle."""

    bindings_factory = peak_selection_module.make_runtime_peak_selection_bindings_factory(
        **bindings_kwargs
    )
    return RuntimeBindingsCallbacksBootstrap(
        bindings_factory=bindings_factory,
        callbacks=peak_selection_module.make_runtime_peak_selection_callbacks(
            bindings_factory
        ),
    )


def build_runtime_selected_peak_bootstrap(
    *,
    peak_selection_module: Any,
    simulation_runtime_state: Any,
    peak_selection_state: Any,
    hkl_lookup_view_state_factory: object,
    selected_peak_marker_factory: object,
    current_primary_a_factory: object,
    caked_view_enabled_factory: object,
    sync_peak_selection_state: Callable[[], None] | None,
    image_size: int,
    primary_a_factory: object,
    primary_c_factory: object,
    max_distance_px: object,
    min_separation_px: object,
    image_shape_factory: object,
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
    sigma_mosaic_deg_factory: object,
    gamma_mosaic_deg_factory: object,
    eta_factory: object,
    wavelength_factory: object,
    debye_x_factory: object,
    debye_y_factory: object,
    detector_center_factory: object,
    optics_mode_factory: object,
    solve_q_values_factory: object,
    overlay_primary_a_factory: object | None = None,
    overlay_primary_c_factory: object | None = None,
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None = None,
    reflection_q_group_metadata: Callable[..., object] | None = None,
    max_hits_per_reflection: object = None,
    schedule_update_factory: object = None,
    set_status_text_factory: object = None,
    draw_idle_factory: object = None,
    display_to_native_sim_coords: Callable[..., tuple[float, float]] | None = None,
    deactivate_conflicting_modes_factory: object = None,
    n2: Any = None,
    process_peaks_parallel: Callable[..., object] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> PeakSelectionRuntimeBootstrap:
    """Build the live selected-peak workflow bootstrap plus overlay callback."""

    config_factories = peak_selection_module.make_runtime_selected_peak_config_factories(
        simulation_runtime_state=simulation_runtime_state,
        image_size=int(image_size),
        primary_a_factory=primary_a_factory,
        primary_c_factory=primary_c_factory,
        max_distance_px=max_distance_px,
        min_separation_px=min_separation_px,
        image_shape_factory=image_shape_factory,
        center_col_factory=center_col_factory,
        center_row_factory=center_row_factory,
        distance_cor_to_detector_factory=distance_cor_to_detector_factory,
        gamma_deg_factory=gamma_deg_factory,
        Gamma_deg_factory=Gamma_deg_factory,
        chi_deg_factory=chi_deg_factory,
        psi_deg_factory=psi_deg_factory,
        psi_z_deg_factory=psi_z_deg_factory,
        zs_factory=zs_factory,
        zb_factory=zb_factory,
        theta_initial_deg_factory=theta_initial_deg_factory,
        cor_angle_deg_factory=cor_angle_deg_factory,
        sigma_mosaic_deg_factory=sigma_mosaic_deg_factory,
        gamma_mosaic_deg_factory=gamma_mosaic_deg_factory,
        eta_factory=eta_factory,
        wavelength_factory=wavelength_factory,
        debye_x_factory=debye_x_factory,
        debye_y_factory=debye_y_factory,
        detector_center_factory=detector_center_factory,
        optics_mode_factory=optics_mode_factory,
        solve_q_values_factory=solve_q_values_factory,
        n2=n2,
        process_peaks_parallel=process_peaks_parallel,
    )
    ensure_peak_overlay_data = peak_selection_module.make_runtime_peak_overlay_data_callback(
        simulation_runtime_state=simulation_runtime_state,
        primary_a_factory=(
            overlay_primary_a_factory
            if overlay_primary_a_factory is not None
            else primary_a_factory
        ),
        primary_c_factory=(
            overlay_primary_c_factory
            if overlay_primary_c_factory is not None
            else primary_c_factory
        ),
        native_sim_to_display_coords=native_sim_to_display_coords,
        reflection_q_group_metadata=reflection_q_group_metadata,
        max_hits_per_reflection=max_hits_per_reflection,
        min_separation_px=min_separation_px,
    )
    runtime_bootstrap = build_runtime_peak_selection_bootstrap(
        peak_selection_module=peak_selection_module,
        simulation_runtime_state=simulation_runtime_state,
        peak_selection_state=peak_selection_state,
        hkl_lookup_view_state_factory=hkl_lookup_view_state_factory,
        selected_peak_marker_factory=selected_peak_marker_factory,
        current_primary_a_factory=current_primary_a_factory,
        caked_view_enabled_factory=caked_view_enabled_factory,
        current_canvas_pick_config_factory=config_factories.canvas_pick,
        current_intersection_config_factory=config_factories.intersection,
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        sync_peak_selection_state=sync_peak_selection_state,
        schedule_update_factory=schedule_update_factory,
        set_status_text_factory=set_status_text_factory,
        draw_idle_factory=draw_idle_factory,
        display_to_native_sim_coords=display_to_native_sim_coords,
        native_sim_to_display_coords=native_sim_to_display_coords,
        simulate_ideal_hkl_native_center=config_factories.ideal_center,
        deactivate_conflicting_modes_factory=deactivate_conflicting_modes_factory,
        n2=n2,
        tcl_error_types=tcl_error_types,
    )
    return PeakSelectionRuntimeBootstrap(
        bindings_factory=runtime_bootstrap.bindings_factory,
        callbacks=runtime_bootstrap.callbacks,
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        maintenance_callbacks=(
            peak_selection_module.make_runtime_peak_selection_maintenance_callbacks(
                runtime_bootstrap.bindings_factory
            )
        ),
    )


def build_runtime_hkl_lookup_controls_bootstrap(
    *,
    views_module: Any,
    view_state: Any,
    peak_selection_callbacks: Any,
    open_bragg_qr_groups: Callable[[], None],
) -> HklLookupControlsRuntimeBootstrap:
    """Build the HKL lookup control wiring shared by peak-selection and Bragg-Qr."""

    def _refresh_controls() -> None:
        refresh = getattr(
            peak_selection_callbacks,
            "update_hkl_pick_button_label",
            None,
        )
        if callable(refresh):
            refresh()

    def _set_hkl_pick_mode(
        enabled: bool,
        message: str | None = None,
    ) -> None:
        set_pick_mode = getattr(peak_selection_callbacks, "set_hkl_pick_mode", None)
        if callable(set_pick_mode):
            set_pick_mode(bool(enabled), message=message)

    def _create_controls(parent: Any) -> None:
        views_module.create_hkl_lookup_controls(
            parent=parent,
            view_state=view_state,
            on_select_hkl=peak_selection_callbacks.select_peak_from_hkl_controls,
            on_toggle_hkl_pick=peak_selection_callbacks.toggle_hkl_pick_mode,
            on_clear_selected_peak=peak_selection_callbacks.clear_selected_peak,
            on_show_bragg_ewald=(
                peak_selection_callbacks.open_selected_peak_intersection_figure
            ),
            on_open_bragg_qr_groups=open_bragg_qr_groups,
        )
        _refresh_controls()

    return HklLookupControlsRuntimeBootstrap(
        create_controls=_create_controls,
        refresh_controls=_refresh_controls,
        set_hkl_pick_mode=_set_hkl_pick_mode,
    )


def build_runtime_geometry_tool_action_controls_bootstrap(
    *,
    views_module: Any,
    view_state: Any,
    on_undo_fit: Callable[[], None],
    on_redo_fit: Callable[[], None],
    on_toggle_manual_pick: Callable[[], None],
    on_undo_manual_placement: Callable[[], None],
    on_export_manual_pairs: Callable[[], None],
    on_import_manual_pairs: Callable[[], None],
    on_toggle_preview_exclude: Callable[[], None],
    on_clear_manual_pairs: Callable[[], None],
) -> GeometryToolActionsRuntimeBootstrap:
    """Build the geometry tool action control wiring through shared views."""

    def _create_controls(parent: Any) -> None:
        views_module.create_geometry_tool_action_controls(
            parent=parent,
            view_state=view_state,
            on_undo_fit=on_undo_fit,
            on_redo_fit=on_redo_fit,
            on_toggle_manual_pick=on_toggle_manual_pick,
            on_undo_manual_placement=on_undo_manual_placement,
            on_export_manual_pairs=on_export_manual_pairs,
            on_import_manual_pairs=on_import_manual_pairs,
            on_toggle_preview_exclude=on_toggle_preview_exclude,
            on_clear_manual_pairs=on_clear_manual_pairs,
        )

    return GeometryToolActionsRuntimeBootstrap(create_controls=_create_controls)


def build_runtime_geometry_fit_action_bootstrap(
    *,
    geometry_fit_module: Any,
    before_run: Callable[[], None] | None = None,
    **bindings_kwargs: Any,
) -> GeometryFitActionRuntimeBootstrap:
    """Build the live geometry-fit action bindings and fit-button callback."""

    bindings_factory = (
        geometry_fit_module.make_runtime_geometry_fit_action_bindings_factory(
            **bindings_kwargs
        )
    )
    return GeometryFitActionRuntimeBootstrap(
        bindings_factory=bindings_factory,
        callback=geometry_fit_module.make_runtime_geometry_fit_action_callback(
            bindings_factory=bindings_factory,
            before_run=before_run,
        ),
    )


def build_runtime_integration_range_drag_bootstrap(
    *,
    integration_range_drag_module: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsCallbacksBootstrap:
    """Build the live integration-range drag binding/callback bundle."""

    bindings_factory = (
        integration_range_drag_module.make_runtime_integration_range_drag_bindings_factory(
            **bindings_kwargs
        )
    )
    return RuntimeBindingsCallbacksBootstrap(
        bindings_factory=bindings_factory,
        callbacks=integration_range_drag_module.make_runtime_integration_range_drag_callbacks(
            bindings_factory
        ),
    )


def build_runtime_integration_range_workflow_bootstrap(
    *,
    integration_range_drag_module: Any,
    ax: Any,
    **bindings_kwargs: Any,
) -> IntegrationRangeDragRuntimeBootstrap:
    """Build the live integration-range drag workflow and visual refresh hooks."""

    integration_region_rect = (
        integration_range_drag_module.create_integration_region_rectangle(ax)
    )
    drag_select_rect = integration_range_drag_module.create_drag_select_rectangle(ax)
    runtime_bootstrap = build_runtime_integration_range_drag_bootstrap(
        integration_range_drag_module=integration_range_drag_module,
        ax=ax,
        drag_select_rect=drag_select_rect,
        integration_region_rect=integration_region_rect,
        **bindings_kwargs,
    )
    return IntegrationRangeDragRuntimeBootstrap(
        bindings_factory=runtime_bootstrap.bindings_factory,
        callbacks=runtime_bootstrap.callbacks,
        refresh_visuals=(
            integration_range_drag_module.make_runtime_integration_region_visuals_callback(
                runtime_bootstrap.bindings_factory
            )
        ),
    )


def build_runtime_canvas_interaction_bootstrap(
    *,
    canvas_interactions_module: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsCallbacksBootstrap:
    """Build the live canvas-interaction binding/callback bundle."""

    bindings_factory = (
        canvas_interactions_module.make_runtime_canvas_interaction_bindings_factory(
            **bindings_kwargs
        )
    )
    return RuntimeBindingsCallbacksBootstrap(
        bindings_factory=bindings_factory,
        callbacks=canvas_interactions_module.make_runtime_canvas_interaction_callbacks(
            bindings_factory
        ),
    )


def build_runtime_background_bootstrap(
    *,
    background_manager_module: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsCallbacksBootstrap:
    """Build the live background-manager binding/callback bundle."""

    bindings_factory = background_manager_module.make_runtime_background_bindings_factory(
        **bindings_kwargs
    )
    return RuntimeBindingsCallbacksBootstrap(
        bindings_factory=bindings_factory,
        callbacks=background_manager_module.make_runtime_background_callbacks(
            bindings_factory
        ),
    )


def build_runtime_geometry_q_group_bootstrap(
    *,
    geometry_q_group_manager_module: Any,
    root: Any,
    **bindings_kwargs: Any,
) -> RuntimeBindingsCallbacksBootstrap:
    """Build the live geometry Q-group binding/callback bundle."""

    bindings_factory = (
        geometry_q_group_manager_module.make_runtime_geometry_q_group_bindings_factory(
            **bindings_kwargs
        )
    )
    return RuntimeBindingsCallbacksBootstrap(
        bindings_factory=bindings_factory,
        callbacks=geometry_q_group_manager_module.make_runtime_geometry_q_group_callbacks(
            root=root,
            bindings_factory=bindings_factory,
        ),
    )


def early_main_bootstrap(
    module_name: str,
    argv: Sequence[str] | None = None,
    *,
    cli_main: Callable[[list[str]], None] | None = None,
) -> None:
    """Handle startup routing before importing heavy simulation dependencies."""

    if module_name != "__main__":
        return

    cli_argv = list(sys.argv[1:] if argv is None else argv)
    if cli_argv and cli_argv[0] in {"-h", "--help"}:
        return

    bundle_path = extract_bundle_arg(cli_argv)

    if cli_argv and cli_argv[0] == "calibrant":
        launch_calibrant_gui(bundle=bundle_path)
        raise SystemExit(0)

    if cli_argv and cli_argv[0] == "gui":
        os.environ["RA_SIM_EARLY_STARTUP_MODE"] = "simulation"
        return

    if should_forward_to_cli(cli_argv):
        if cli_main is None:
            from ra_sim.cli import main as cli_main

        cli_main(list(cli_argv))
        raise SystemExit(0)

    mode = quick_startup_mode_dialog()
    if mode == "calibrant":
        launch_calibrant_gui(bundle=bundle_path)
        raise SystemExit(0)
    if mode == "simulation":
        os.environ["RA_SIM_EARLY_STARTUP_MODE"] = "simulation"
        return
    raise SystemExit(0)
