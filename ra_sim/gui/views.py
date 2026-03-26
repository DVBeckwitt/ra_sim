"""GUI view helpers used by RA-SIM Tk applications."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
import tkinter as tk
from tkinter import ttk

from .collapsible import CollapsibleFrame
from .state import (
    AnalysisViewControlsViewState,
    AnalysisExportControlsViewState,
    BackgroundThetaControlsViewState,
    BackgroundBackendDebugViewState,
    BraggQrManagerViewState,
    FiniteStackControlsViewState,
    GeometryOverlayActionsViewState,
    GeometryToolActionsViewState,
    GeometryFitConstraintsViewState,
    HklLookupViewState,
    GeometryQGroupViewState,
    HbnGeometryDebugViewState,
    SamplingOpticsControlsViewState,
    WorkspacePanelsViewState,
)


_GEOMETRY_Q_GROUP_EMPTY_TEXT = (
    "No Qr/Qz groups are listed yet. "
    'Press "Update Listed Peaks" to snapshot the current simulation.'
)
_GEOMETRY_FIT_CONSTRAINTS_HELP_TEXT = (
    "Each window is applied as current value +/- deviation during geometry fitting. "
    "Stay-close adds a soft pull back to the starting guess."
)
_BACKGROUND_THETA_HELP_TEXT = "Per-background theta_i values (deg, in load order)"
_GEOMETRY_FIT_BACKGROUND_HELP_TEXT = (
    "Use 'current', 'all', or 1-based indices/ranges like 1,3-5"
)


def create_root_window(title: str = "RA Simulation") -> tk.Tk:
    """Create and return a Tk root window with the provided title."""

    root = tk.Tk()
    root.title(title)
    return root


def create_workspace_panels(
    *,
    parent: tk.Misc,
    view_state: WorkspacePanelsViewState,
) -> None:
    """Create and store the workspace action/background/session panel frames."""

    workspace_actions_frame = ttk.LabelFrame(parent, text="Workspace Actions")
    workspace_actions_frame.pack(fill=tk.X, padx=5, pady=5)

    workspace_backgrounds_frame = ttk.Frame(parent)
    workspace_backgrounds_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    workspace_session_frame = ttk.LabelFrame(parent, text="Session")
    workspace_session_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    view_state.workspace_actions_frame = workspace_actions_frame
    view_state.workspace_backgrounds_frame = workspace_backgrounds_frame
    view_state.workspace_session_frame = workspace_session_frame


def populate_stacked_button_group(
    parent: tk.Misc,
    button_specs: Sequence[tuple[str, Callable[[], None]]],
) -> None:
    """Append one vertical stack of standard buttons to a panel."""

    for text, command in button_specs:
        ttk.Button(
            parent,
            text=str(text),
            command=command,
        ).pack(side=tk.TOP, padx=5, pady=2)


def create_background_file_controls(
    *,
    parent: tk.Misc,
    view_state: WorkspacePanelsViewState,
    on_load_backgrounds: Callable[[], None],
    status_text: str = "",
) -> None:
    """Create the background-file load button and status label."""

    ttk.Button(
        parent,
        text="Load Background Files...",
        command=on_load_backgrounds,
    ).pack(side=tk.TOP, padx=5, pady=2)

    background_file_status_var = tk.StringVar(value=str(status_text))
    background_file_status_label = ttk.Label(
        parent,
        textvariable=background_file_status_var,
        wraplength=520,
        justify=tk.LEFT,
    )
    background_file_status_label.pack(side=tk.TOP, padx=5, pady=(0, 2))

    view_state.background_file_status_var = background_file_status_var
    view_state.background_file_status_label = background_file_status_label


def create_sampling_optics_controls(
    *,
    parent: tk.Misc,
    view_state: SamplingOpticsControlsViewState,
    resolution_options: Sequence[object],
    initial_resolution: str,
    custom_samples_text: str,
    resolution_count_text: str,
    optics_mode_text: str,
    on_apply_custom_samples: Callable[[], None],
) -> None:
    """Create the sampling-resolution / optics controls and store their refs."""

    resolution_selector_frame = ttk.Frame(parent)
    resolution_selector_frame.pack(fill=tk.X, pady=5)
    ttk.Label(resolution_selector_frame, text="Sampling Resolution").pack(
        anchor=tk.W,
        padx=5,
    )

    resolution_var = tk.StringVar(value=str(initial_resolution))
    resolution_menu = ttk.OptionMenu(
        resolution_selector_frame,
        resolution_var,
        resolution_var.get(),
        *[str(option) for option in resolution_options],
    )
    resolution_menu.pack(fill=tk.X, padx=5, pady=(2, 0))

    custom_samples_row = ttk.Frame(resolution_selector_frame)
    custom_samples_row.pack(fill=tk.X, padx=5, pady=(2, 0))
    ttk.Label(custom_samples_row, text="Custom Samples").pack(side=tk.LEFT)

    custom_samples_var = tk.StringVar(value=str(custom_samples_text))
    custom_samples_entry = ttk.Entry(
        custom_samples_row,
        textvariable=custom_samples_var,
        width=10,
        justify="right",
    )
    custom_samples_entry.pack(side=tk.LEFT, padx=(6, 4))
    custom_samples_entry.bind("<Return>", lambda _event: on_apply_custom_samples())

    custom_samples_apply_button = ttk.Button(
        custom_samples_row,
        text="Apply",
        command=on_apply_custom_samples,
    )
    custom_samples_apply_button.pack(side=tk.LEFT)

    resolution_count_var = tk.StringVar(value=str(resolution_count_text))
    resolution_count_label = ttk.Label(
        resolution_selector_frame,
        textvariable=resolution_count_var,
    )
    resolution_count_label.pack(anchor=tk.W, padx=5, pady=(2, 0))

    optics_mode_frame = ttk.Frame(parent)
    optics_mode_frame.pack(fill=tk.X, pady=(6, 2))
    ttk.Label(optics_mode_frame, text="Optics Transport").pack(anchor=tk.W, padx=5)

    optics_mode_var = tk.StringVar(value=str(optics_mode_text))
    fast_optics_button = ttk.Radiobutton(
        optics_mode_frame,
        text="Original Fast Approx (Fresnel + Beer-Lambert)",
        variable=optics_mode_var,
        value="fast",
    )
    fast_optics_button.pack(anchor=tk.W, padx=12)

    exact_optics_button = ttk.Radiobutton(
        optics_mode_frame,
        text="Complex-k DWBA slab optics (Precise)",
        variable=optics_mode_var,
        value="exact",
    )
    exact_optics_button.pack(anchor=tk.W, padx=12)

    view_state.resolution_selector_frame = resolution_selector_frame
    view_state.resolution_var = resolution_var
    view_state.resolution_menu = resolution_menu
    view_state.resolution_count_var = resolution_count_var
    view_state.resolution_count_label = resolution_count_label
    view_state.custom_samples_var = custom_samples_var
    view_state.custom_samples_row = custom_samples_row
    view_state.custom_samples_entry = custom_samples_entry
    view_state.custom_samples_apply_button = custom_samples_apply_button
    view_state.optics_mode_frame = optics_mode_frame
    view_state.optics_mode_var = optics_mode_var
    view_state.fast_optics_button = fast_optics_button
    view_state.exact_optics_button = exact_optics_button


def set_sampling_resolution_summary_text(
    view_state: SamplingOpticsControlsViewState,
    text: str,
) -> None:
    """Update the sampling-resolution summary label."""

    setter = getattr(view_state.resolution_count_var, "set", None)
    if callable(setter):
        setter(str(text))


def _configure_widget_state(widget: object | None, state: str) -> None:
    configure = getattr(widget, "configure", None)
    if configure is None:
        configure = getattr(widget, "config", None)
    if callable(configure):
        configure(state=state)


def set_sampling_custom_controls_enabled(
    view_state: SamplingOpticsControlsViewState,
    *,
    enabled: bool,
) -> None:
    """Enable or disable the custom-sample entry and apply button."""

    state_value = tk.NORMAL if enabled else tk.DISABLED
    _configure_widget_state(view_state.custom_samples_entry, state_value)
    _configure_widget_state(view_state.custom_samples_apply_button, state_value)


def create_finite_stack_controls(
    *,
    parent: tk.Misc,
    view_state: FiniteStackControlsViewState,
    finite_stack: bool,
    stack_layers: int,
    phi_l_divisor: float,
    phase_delta_expression: str,
    on_toggle_finite_stack: Callable[[], None],
    on_layer_slider: Callable[[object], None],
    on_commit_layer_entry: Callable[[object], None],
    on_commit_phi_l_divisor_entry: Callable[[object], None],
    on_commit_phase_delta_expression_entry: Callable[[object], None],
) -> None:
    """Create the finite-stack controls and store their widget refs/vars."""

    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, padx=5, pady=5)

    finite_stack_var = tk.BooleanVar(value=bool(finite_stack))
    stack_layers_var = tk.IntVar(value=int(stack_layers))
    phi_l_divisor_var = tk.DoubleVar(value=float(phi_l_divisor))
    phase_delta_expr_var = tk.StringVar(value=str(phase_delta_expression))

    finite_stack_checkbutton = ttk.Checkbutton(
        frame,
        text="Finite Stack",
        variable=finite_stack_var,
        command=on_toggle_finite_stack,
    )
    finite_stack_checkbutton.pack(anchor=tk.W, padx=5, pady=2)

    layers_row = ttk.Frame(frame)
    layers_row.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(layers_row, text="Layers:").grid(row=0, column=0, sticky="w")

    layers_entry_var = tk.StringVar(value=str(int(stack_layers)))
    layers_entry = ttk.Entry(
        layers_row,
        textvariable=layers_entry_var,
        width=8,
        justify="right",
    )
    layers_entry.grid(row=0, column=2, sticky="e", padx=(5, 0))
    layers_entry.bind("<Return>", on_commit_layer_entry)
    layers_entry.bind("<FocusOut>", on_commit_layer_entry)

    layers_scale = tk.Scale(
        layers_row,
        from_=1,
        to=1000,
        orient=tk.HORIZONTAL,
        resolution=1,
        showvalue=False,
        variable=stack_layers_var,
        command=on_layer_slider,
    )
    layers_scale.grid(row=0, column=1, sticky="ew", padx=(5, 5))
    layers_row.columnconfigure(1, weight=1)

    phi_div_row = ttk.Frame(frame)
    phi_div_row.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(phi_div_row, text="Phi L divisor:").pack(side=tk.LEFT)
    phi_l_divisor_entry_var = tk.StringVar(value=f"{float(phi_l_divisor):.6g}")
    phi_l_divisor_entry = ttk.Entry(
        phi_div_row,
        textvariable=phi_l_divisor_entry_var,
        width=12,
        justify="right",
    )
    phi_l_divisor_entry.pack(side=tk.RIGHT)
    phi_l_divisor_entry.bind("<Return>", on_commit_phi_l_divisor_entry)
    phi_l_divisor_entry.bind("<FocusOut>", on_commit_phi_l_divisor_entry)

    phase_row = ttk.Frame(frame)
    phase_row.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(phase_row, text="Phase delta equation:").pack(side=tk.LEFT)
    phase_delta_entry_var = tk.StringVar(value=str(phase_delta_expression))
    phase_delta_entry = ttk.Entry(
        phase_row,
        textvariable=phase_delta_entry_var,
        width=36,
    )
    phase_delta_entry.pack(side=tk.RIGHT)
    phase_delta_entry.bind("<Return>", on_commit_phase_delta_expression_entry)
    phase_delta_entry.bind("<FocusOut>", on_commit_phase_delta_expression_entry)

    view_state.frame = frame
    view_state.finite_stack_var = finite_stack_var
    view_state.finite_stack_checkbutton = finite_stack_checkbutton
    view_state.layers_row = layers_row
    view_state.stack_layers_var = stack_layers_var
    view_state.layers_scale = layers_scale
    view_state.layers_entry_var = layers_entry_var
    view_state.layers_entry = layers_entry
    view_state.phi_l_divisor_var = phi_l_divisor_var
    view_state.phi_l_divisor_entry_var = phi_l_divisor_entry_var
    view_state.phi_l_divisor_entry = phi_l_divisor_entry
    view_state.phase_delta_expr_var = phase_delta_expr_var
    view_state.phase_delta_entry_var = phase_delta_entry_var
    view_state.phase_delta_entry = phase_delta_entry


def set_finite_stack_layer_controls_enabled(
    view_state: FiniteStackControlsViewState,
    *,
    enabled: bool,
) -> None:
    """Enable or disable the finite-stack layer slider and entry."""

    state_value = tk.NORMAL if enabled else tk.DISABLED
    _configure_widget_state(view_state.layers_scale, state_value)
    _configure_widget_state(view_state.layers_entry, state_value)


def ensure_finite_stack_layer_scale_max(
    view_state: FiniteStackControlsViewState,
    minimum_to: int,
) -> None:
    """Ensure the finite-stack layer slider can reach at least ``minimum_to``."""

    if view_state.layers_scale is None:
        return
    try:
        current_to = int(round(float(view_state.layers_scale.cget("to"))))
    except Exception:
        current_to = int(minimum_to)
    if int(minimum_to) > current_to:
        view_state.layers_scale.configure(to=int(minimum_to))


def set_finite_stack_layer_entry_text(
    view_state: FiniteStackControlsViewState,
    text: str,
) -> None:
    """Update the finite-stack layer entry text."""

    setter = getattr(view_state.layers_entry_var, "set", None)
    if callable(setter):
        setter(str(text))


def set_finite_stack_phi_l_divisor_entry_text(
    view_state: FiniteStackControlsViewState,
    text: str,
) -> None:
    """Update the finite-stack phi-L divisor entry text."""

    setter = getattr(view_state.phi_l_divisor_entry_var, "set", None)
    if callable(setter):
        setter(str(text))


def set_finite_stack_phase_delta_entry_text(
    view_state: FiniteStackControlsViewState,
    text: str,
) -> None:
    """Update the finite-stack phase-delta entry text."""

    setter = getattr(view_state.phase_delta_entry_var, "set", None)
    if callable(setter):
        setter(str(text))


def create_geometry_tool_action_controls(
    *,
    parent: tk.Misc,
    view_state: GeometryToolActionsViewState,
    on_undo_fit: Callable[[], None],
    on_redo_fit: Callable[[], None],
    on_toggle_manual_pick: Callable[[], None],
    on_undo_manual_placement: Callable[[], None],
    on_export_manual_pairs: Callable[[], None],
    on_import_manual_pairs: Callable[[], None],
    on_toggle_preview_exclude: Callable[[], None],
    on_clear_manual_pairs: Callable[[], None],
    manual_pick_text: str = "Pick Qr Sets on Image",
    preview_exclude_text: str = "Select Qr/Qz Peaks",
) -> None:
    """Create fit-history/manual-geometry action controls and store their refs."""

    undo_geometry_fit_button = ttk.Button(
        parent,
        text="Undo Fit",
        command=on_undo_fit,
    )
    undo_geometry_fit_button.pack(side=tk.TOP, padx=5, pady=2)

    redo_geometry_fit_button = ttk.Button(
        parent,
        text="Redo Fit",
        command=on_redo_fit,
    )
    redo_geometry_fit_button.pack(side=tk.TOP, padx=5, pady=2)

    geometry_manual_pick_button_var = tk.StringVar(value=str(manual_pick_text))
    geometry_manual_pick_button = ttk.Button(
        parent,
        textvariable=geometry_manual_pick_button_var,
        command=on_toggle_manual_pick,
    )
    geometry_manual_pick_button.pack(side=tk.TOP, padx=5, pady=2)

    geometry_manual_undo_button = ttk.Button(
        parent,
        text="Undo Placement",
        command=on_undo_manual_placement,
    )
    geometry_manual_undo_button.pack(side=tk.TOP, padx=5, pady=2)

    geometry_manual_export_button = ttk.Button(
        parent,
        text="Export Placements...",
        command=on_export_manual_pairs,
    )
    geometry_manual_export_button.pack(side=tk.TOP, padx=5, pady=2)

    geometry_manual_import_button = ttk.Button(
        parent,
        text="Import Placements...",
        command=on_import_manual_pairs,
    )
    geometry_manual_import_button.pack(side=tk.TOP, padx=5, pady=2)

    geometry_preview_exclude_button_var = tk.StringVar(value=str(preview_exclude_text))
    geometry_preview_exclude_button = ttk.Button(
        parent,
        textvariable=geometry_preview_exclude_button_var,
        command=on_toggle_preview_exclude,
    )
    geometry_preview_exclude_button.pack(side=tk.TOP, padx=5, pady=2)

    clear_geometry_preview_exclusions_button = ttk.Button(
        parent,
        text="Clear Current Image Pairs",
        command=on_clear_manual_pairs,
    )
    clear_geometry_preview_exclusions_button.pack(side=tk.TOP, padx=5, pady=2)

    view_state.undo_geometry_fit_button = undo_geometry_fit_button
    view_state.redo_geometry_fit_button = redo_geometry_fit_button
    view_state.geometry_manual_pick_button_var = geometry_manual_pick_button_var
    view_state.geometry_manual_pick_button = geometry_manual_pick_button
    view_state.geometry_manual_undo_button = geometry_manual_undo_button
    view_state.geometry_manual_export_button = geometry_manual_export_button
    view_state.geometry_manual_import_button = geometry_manual_import_button
    view_state.geometry_preview_exclude_button_var = geometry_preview_exclude_button_var
    view_state.geometry_preview_exclude_button = geometry_preview_exclude_button
    view_state.clear_geometry_preview_exclusions_button = (
        clear_geometry_preview_exclusions_button
    )


def set_geometry_tool_action_texts(
    view_state: GeometryToolActionsViewState,
    *,
    manual_pick_text: str | None = None,
    preview_exclude_text: str | None = None,
) -> None:
    """Update geometry-tool action button labels backed by Tk vars."""

    if manual_pick_text is not None and view_state.geometry_manual_pick_button_var is not None:
        setter = getattr(view_state.geometry_manual_pick_button_var, "set", None)
        if callable(setter):
            setter(str(manual_pick_text))
    if (
        preview_exclude_text is not None
        and view_state.geometry_preview_exclude_button_var is not None
    ):
        setter = getattr(view_state.geometry_preview_exclude_button_var, "set", None)
        if callable(setter):
            setter(str(preview_exclude_text))


def set_geometry_fit_history_button_state(
    view_state: GeometryToolActionsViewState,
    *,
    can_undo: bool,
    can_redo: bool,
) -> None:
    """Enable or disable the fit-history action buttons."""

    if view_state.undo_geometry_fit_button is not None:
        configure = getattr(view_state.undo_geometry_fit_button, "config", None)
        if configure is None:
            configure = getattr(view_state.undo_geometry_fit_button, "configure", None)
        if callable(configure):
            configure(state=("normal" if can_undo else "disabled"))
    if view_state.redo_geometry_fit_button is not None:
        configure = getattr(view_state.redo_geometry_fit_button, "config", None)
        if configure is None:
            configure = getattr(view_state.redo_geometry_fit_button, "configure", None)
        if callable(configure):
            configure(state=("normal" if can_redo else "disabled"))


def create_hkl_lookup_controls(
    *,
    parent: tk.Misc,
    view_state: HklLookupViewState,
    on_select_hkl: Callable[[], None],
    on_toggle_hkl_pick: Callable[[], None],
    on_clear_selected_peak: Callable[[], None],
    on_show_bragg_ewald: Callable[[], None],
    on_open_bragg_qr_groups: Callable[[], None],
    selected_h_text: str = "0",
    selected_k_text: str = "0",
    selected_l_text: str = "0",
    hkl_pick_text: str = "Pick HKL on Image",
) -> None:
    """Create the HKL lookup / peak-selection control cluster."""

    frame = ttk.LabelFrame(parent, text="Peak Lookup (HKL)")
    frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4)

    selected_h_var = tk.StringVar(value=str(selected_h_text))
    selected_k_var = tk.StringVar(value=str(selected_k_text))
    selected_l_var = tk.StringVar(value=str(selected_l_text))
    hkl_pick_button_var = tk.StringVar(value=str(hkl_pick_text))

    for col in range(8):
        frame.columnconfigure(col, weight=1 if col in {1, 3, 5} else 0)

    ttk.Label(frame, text="H").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(6, 2),
        pady=(4, 2),
    )
    h_entry = ttk.Entry(frame, width=5, textvariable=selected_h_var)
    h_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(4, 2))

    ttk.Label(frame, text="K").grid(
        row=0,
        column=2,
        sticky="w",
        padx=(0, 2),
        pady=(4, 2),
    )
    k_entry = ttk.Entry(frame, width=5, textvariable=selected_k_var)
    k_entry.grid(row=0, column=3, sticky="ew", padx=(0, 6), pady=(4, 2))

    ttk.Label(frame, text="L").grid(
        row=0,
        column=4,
        sticky="w",
        padx=(0, 2),
        pady=(4, 2),
    )
    l_entry = ttk.Entry(frame, width=5, textvariable=selected_l_var)
    l_entry.grid(row=0, column=5, sticky="ew", padx=(0, 8), pady=(4, 2))

    select_button = ttk.Button(
        frame,
        text="Select HKL",
        command=on_select_hkl,
    )
    select_button.grid(row=0, column=6, sticky="ew", padx=(0, 4), pady=(4, 2))

    hkl_pick_button = ttk.Button(
        frame,
        textvariable=hkl_pick_button_var,
        command=on_toggle_hkl_pick,
    )
    hkl_pick_button.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=(6, 4),
        pady=(2, 6),
    )

    clear_button = ttk.Button(
        frame,
        text="Clear",
        command=on_clear_selected_peak,
    )
    clear_button.grid(row=0, column=7, sticky="ew", padx=(0, 6), pady=(4, 2))

    show_bragg_ewald_button = ttk.Button(
        frame,
        text="Show Bragg/Ewald",
        command=on_show_bragg_ewald,
    )
    show_bragg_ewald_button.grid(
        row=1,
        column=2,
        columnspan=3,
        sticky="ew",
        padx=(0, 4),
        pady=(2, 6),
    )

    bragg_qr_groups_button = ttk.Button(
        frame,
        text="Bragg Qr Groups",
        command=on_open_bragg_qr_groups,
    )
    bragg_qr_groups_button.grid(
        row=1,
        column=5,
        columnspan=3,
        sticky="ew",
        padx=(0, 6),
        pady=(2, 6),
    )

    for entry in (h_entry, k_entry, l_entry):
        entry.bind("<Return>", lambda _event: on_select_hkl())

    view_state.frame = frame
    view_state.selected_h_var = selected_h_var
    view_state.selected_k_var = selected_k_var
    view_state.selected_l_var = selected_l_var
    view_state.h_entry = h_entry
    view_state.k_entry = k_entry
    view_state.l_entry = l_entry
    view_state.select_button = select_button
    view_state.hkl_pick_button_var = hkl_pick_button_var
    view_state.hkl_pick_button = hkl_pick_button
    view_state.clear_button = clear_button
    view_state.show_bragg_ewald_button = show_bragg_ewald_button
    view_state.bragg_qr_groups_button = bragg_qr_groups_button


def set_hkl_lookup_values(
    view_state: HklLookupViewState,
    *,
    h_text: str | None = None,
    k_text: str | None = None,
    l_text: str | None = None,
) -> None:
    """Update the HKL entry vars for the lookup controls."""

    for var, value in (
        (view_state.selected_h_var, h_text),
        (view_state.selected_k_var, k_text),
        (view_state.selected_l_var, l_text),
    ):
        if value is None or var is None:
            continue
        setter = getattr(var, "set", None)
        if callable(setter):
            setter(str(value))


def set_hkl_pick_button_text(
    view_state: HklLookupViewState,
    text: str,
) -> None:
    """Update the HKL image-pick button label."""

    if view_state.hkl_pick_button_var is None:
        return
    setter = getattr(view_state.hkl_pick_button_var, "set", None)
    if callable(setter):
        setter(str(text))


def create_geometry_overlay_action_controls(
    *,
    parent: tk.Misc,
    view_state: GeometryOverlayActionsViewState,
    on_toggle_qr_cylinder_overlay: Callable[[], None],
    on_clear_geometry_overlays: Callable[[], None],
    on_fit_mosaic: Callable[[], None],
    show_qr_cylinder_overlay: bool = False,
) -> None:
    """Create the overlay/mosaic action controls for the fit-actions column."""

    show_qr_cylinder_overlay_var = tk.BooleanVar(value=bool(show_qr_cylinder_overlay))
    show_qr_cylinder_overlay_checkbutton = ttk.Checkbutton(
        parent,
        text="Show Qr Cylinder Lines",
        variable=show_qr_cylinder_overlay_var,
        command=on_toggle_qr_cylinder_overlay,
    )
    show_qr_cylinder_overlay_checkbutton.pack(side=tk.TOP, padx=5, pady=2)

    clear_geometry_markers_button = ttk.Button(
        parent,
        text="Clear Geometry Overlays",
        command=on_clear_geometry_overlays,
    )
    clear_geometry_markers_button.pack(side=tk.TOP, padx=5, pady=2)

    fit_button_mosaic = ttk.Button(
        parent,
        text="Fit Mosaic Widths",
        command=on_fit_mosaic,
    )
    fit_button_mosaic.pack(side=tk.TOP, padx=5, pady=2)

    view_state.show_qr_cylinder_overlay_var = show_qr_cylinder_overlay_var
    view_state.show_qr_cylinder_overlay_checkbutton = (
        show_qr_cylinder_overlay_checkbutton
    )
    view_state.clear_geometry_markers_button = clear_geometry_markers_button
    view_state.fit_button_mosaic = fit_button_mosaic


def create_analysis_view_controls(
    *,
    parent: tk.Misc,
    view_state: AnalysisViewControlsViewState,
    on_toggle_1d_plots: Callable[[], None],
    on_toggle_caked_2d: Callable[[], None],
    on_toggle_log_radial: Callable[[], None],
    on_toggle_log_azimuth: Callable[[], None],
    show_1d: bool = False,
    show_caked_2d: bool = False,
    log_radial: bool = False,
    log_azimuth: bool = False,
) -> None:
    """Create the analysis view toggle controls and store their vars."""

    show_1d_var = tk.BooleanVar(value=bool(show_1d))
    check_1d = ttk.Checkbutton(
        parent,
        text="Show 1D Integration",
        variable=show_1d_var,
        command=on_toggle_1d_plots,
    )
    check_1d.pack(side=tk.TOP, padx=5, pady=2)

    show_caked_2d_var = tk.BooleanVar(value=bool(show_caked_2d))
    check_2d = ttk.Checkbutton(
        parent,
        text="Show 2D Caked Integration",
        variable=show_caked_2d_var,
        command=on_toggle_caked_2d,
    )
    check_2d.pack(side=tk.TOP, padx=5, pady=2)

    log_radial_var = tk.BooleanVar(value=bool(log_radial))
    check_log_radial = ttk.Checkbutton(
        parent,
        text="Log Radial",
        variable=log_radial_var,
        command=on_toggle_log_radial,
    )
    check_log_radial.pack(side=tk.TOP, padx=5, pady=2)

    log_azimuth_var = tk.BooleanVar(value=bool(log_azimuth))
    check_log_azimuth = ttk.Checkbutton(
        parent,
        text="Log Azimuth",
        variable=log_azimuth_var,
        command=on_toggle_log_azimuth,
    )
    check_log_azimuth.pack(side=tk.TOP, padx=5, pady=2)

    view_state.show_1d_var = show_1d_var
    view_state.check_1d = check_1d
    view_state.show_caked_2d_var = show_caked_2d_var
    view_state.check_2d = check_2d
    view_state.log_radial_var = log_radial_var
    view_state.check_log_radial = check_log_radial
    view_state.log_azimuth_var = log_azimuth_var
    view_state.check_log_azimuth = check_log_azimuth


def create_analysis_export_controls(
    *,
    parent: tk.Misc,
    view_state: AnalysisExportControlsViewState,
    on_save_snapshot: Callable[[], None],
    on_save_q_space: Callable[[], None],
    on_save_1d_grid: Callable[[], None],
) -> None:
    """Create the analysis export buttons and store their widget refs."""

    snapshot_button = ttk.Button(
        parent,
        text="Save 1D Snapshot",
        command=on_save_snapshot,
    )
    snapshot_button.pack(side=tk.TOP, padx=5, pady=2)

    save_q_button = ttk.Button(
        parent,
        text="Save Q-Space Snapshot",
        command=on_save_q_space,
    )
    save_q_button.pack(side=tk.TOP, padx=5, pady=2)

    save_1d_grid_button = ttk.Button(
        parent,
        text="Save 1D Grid",
        command=on_save_1d_grid,
    )
    save_1d_grid_button.pack(side=tk.TOP, padx=5, pady=2)

    view_state.snapshot_button = snapshot_button
    view_state.save_q_button = save_q_button
    view_state.save_1d_grid_button = save_1d_grid_button


def set_background_file_status_text(
    view_state: WorkspacePanelsViewState,
    text: str,
) -> None:
    """Update the background-file status line in the workspace panel."""

    if view_state.background_file_status_var is None:
        return
    setter = getattr(view_state.background_file_status_var, "set", None)
    if callable(setter):
        setter(str(text))


def create_background_backend_debug_controls(
    *,
    parent: tk.Misc,
    view_state: BackgroundBackendDebugViewState,
    status_text: str,
    on_rotate_minus_90: Callable[[], None],
    on_rotate_plus_90: Callable[[], None],
    on_flip_x: Callable[[], None],
    on_flip_y: Callable[[], None],
    on_reset: Callable[[], None],
) -> None:
    """Create the backend background-orientation debug controls."""

    frame = ttk.LabelFrame(parent, text="Background Backend (debug)")
    frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    status_label = ttk.Label(frame, text=str(status_text))
    status_label.pack(side=tk.LEFT, padx=4)

    ttk.Button(
        frame,
        text="Rot -90",
        command=on_rotate_minus_90,
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        frame,
        text="Rot +90",
        command=on_rotate_plus_90,
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        frame,
        text="Flip X",
        command=on_flip_x,
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        frame,
        text="Flip Y",
        command=on_flip_y,
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        frame,
        text="Reset",
        command=on_reset,
    ).pack(side=tk.LEFT, padx=2)

    view_state.background_backend_frame = frame
    view_state.background_backend_status_label = status_label


def set_background_backend_status_text(
    view_state: BackgroundBackendDebugViewState,
    text: str,
) -> None:
    """Update the backend-background debug status text."""

    if view_state.background_backend_status_label is None:
        return
    configure = getattr(view_state.background_backend_status_label, "config", None)
    if configure is None:
        configure = getattr(view_state.background_backend_status_label, "configure", None)
    if callable(configure):
        configure(text=str(text))


def _set_var_value(var: object | None, value: object) -> None:
    setter = getattr(var, "set", None)
    if callable(setter):
        setter(value)


def reset_backend_orientation_debug_controls(
    view_state: BackgroundBackendDebugViewState,
) -> None:
    """Reset the backend-orientation debug vars to their default values."""

    _set_var_value(view_state.backend_rotation_var, 0)
    _set_var_value(view_state.backend_flip_y_axis_var, False)
    _set_var_value(view_state.backend_flip_x_axis_var, False)
    _set_var_value(view_state.backend_flip_order_var, "yx")


def create_backend_orientation_debug_controls(
    *,
    parent: tk.Misc,
    view_state: BackgroundBackendDebugViewState,
    rotation_value: int = 0,
    flip_y_axis: bool = False,
    flip_x_axis: bool = False,
    flip_order: str = "yx",
) -> None:
    """Create the backend-orientation debug controls and store their vars."""

    frame = ttk.LabelFrame(parent, text="Backend orientation (debug)")
    frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    rotation_var = tk.IntVar(value=int(rotation_value))
    flip_y_axis_var = tk.BooleanVar(value=bool(flip_y_axis))
    flip_x_axis_var = tk.BooleanVar(value=bool(flip_x_axis))
    flip_order_var = tk.StringVar(value=str(flip_order))

    ttk.Label(frame, text="Rotate ×90° (k):").pack(side=tk.LEFT, padx=2)
    tk.Spinbox(
        frame,
        from_=-3,
        to=3,
        width=4,
        textvariable=rotation_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Checkbutton(
        frame,
        text="Flip about y-axis",
        variable=flip_y_axis_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Checkbutton(
        frame,
        text="Flip about x-axis",
        variable=flip_x_axis_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Label(frame, text="Flip order:").pack(side=tk.LEFT, padx=2)
    tk.OptionMenu(
        frame,
        flip_order_var,
        str(flip_order),
        "yx",
        "xy",
    ).pack(side=tk.LEFT, padx=2)

    view_state.backend_orientation_frame = frame
    view_state.backend_rotation_var = rotation_var
    view_state.backend_flip_y_axis_var = flip_y_axis_var
    view_state.backend_flip_x_axis_var = flip_x_axis_var
    view_state.backend_flip_order_var = flip_order_var

    ttk.Button(
        frame,
        text="Reset",
        command=lambda: reset_backend_orientation_debug_controls(view_state),
    ).pack(side=tk.LEFT, padx=4)


def create_geometry_fit_constraints_panel(
    *,
    parent: tk.Misc,
    root: tk.Misc,
    view_state: GeometryFitConstraintsViewState,
    after: object | None = None,
    on_mousewheel: Callable[[object], object] | None = None,
) -> None:
    """Create the scrollable geometry-fit constraints panel and store its widgets."""

    panel = CollapsibleFrame(
        parent,
        text="Geometry Fit Constraints",
        expanded=True,
    )
    pack_kwargs = {
        "side": tk.TOP,
        "fill": tk.X,
        "padx": 5,
        "pady": 5,
    }
    if after is not None:
        pack_kwargs["after"] = after
    panel.pack(**pack_kwargs)

    ttk.Label(
        panel.frame,
        text=_GEOMETRY_FIT_CONSTRAINTS_HELP_TEXT,
        wraplength=420,
        justify="left",
    ).pack(fill=tk.X, padx=6, pady=(2, 6))

    list_frame = ttk.Frame(panel.frame)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    canvas_height = int(min(max(root.winfo_screenheight() * 0.35, 220), 420))
    canvas = tk.Canvas(
        list_frame,
        highlightthickness=0,
        borderwidth=0,
        height=canvas_height,
    )
    scrollbar = ttk.Scrollbar(
        list_frame,
        orient=tk.VERTICAL,
        command=canvas.yview,
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    body = ttk.Frame(canvas)
    body_window = canvas.create_window(
        (0, 0),
        window=body,
        anchor="nw",
    )

    view_state.panel = panel
    view_state.canvas = canvas
    view_state.body = body
    view_state.body_window = body_window
    view_state.controls.clear()

    body.bind(
        "<Configure>",
        lambda _event: refresh_geometry_fit_constraints_scrollregion(view_state),
    )
    canvas.bind(
        "<Configure>",
        lambda event: resize_geometry_fit_constraints_body(
            view_state,
            getattr(event, "width", 0),
        ),
    )

    if on_mousewheel is not None:
        root.bind_all("<MouseWheel>", on_mousewheel, add="+")
        root.bind_all("<Button-4>", on_mousewheel, add="+")
        root.bind_all("<Button-5>", on_mousewheel, add="+")


def refresh_geometry_fit_constraints_scrollregion(
    view_state: GeometryFitConstraintsViewState,
) -> None:
    """Update the constraints canvas scrollregion from its current content."""

    if view_state.canvas is None:
        return
    view_state.canvas.configure(scrollregion=view_state.canvas.bbox("all"))


def resize_geometry_fit_constraints_body(
    view_state: GeometryFitConstraintsViewState,
    width: int | float,
) -> None:
    """Match the constraints body width to the visible canvas width."""

    if view_state.canvas is None or view_state.body_window is None:
        return
    view_state.canvas.itemconfigure(
        view_state.body_window,
        width=width,
    )


def scroll_geometry_fit_constraints_canvas(
    view_state: GeometryFitConstraintsViewState,
    *,
    pointer_x: int,
    pointer_y: int,
    event: object,
) -> str | None:
    """Scroll the constraints canvas when the pointer is over the panel."""

    canvas = view_state.canvas
    if canvas is None:
        return None

    canvas_x = canvas.winfo_rootx()
    canvas_y = canvas.winfo_rooty()
    if not (
        canvas_x <= pointer_x <= canvas_x + canvas.winfo_width()
        and canvas_y <= pointer_y <= canvas_y + canvas.winfo_height()
    ):
        return None

    delta = 0
    if getattr(event, "delta", 0):
        delta = int(-event.delta / 120)
    elif getattr(event, "num", None) == 4:
        delta = -1
    elif getattr(event, "num", None) == 5:
        delta = 1
    if delta:
        canvas.yview_scroll(delta, "units")
        return "break"
    return None


def set_geometry_fit_constraint_control(
    view_state: GeometryFitConstraintsViewState,
    *,
    name: str,
    row: object,
    window_var: object,
    pull_var: object,
) -> None:
    """Register one geometry-fit constraint row in the shared view-state map."""

    view_state.controls[str(name)] = {
        "row": row,
        "window_var": window_var,
        "pull_var": pull_var,
        "_mapped": False,
    }


def create_background_theta_controls(
    *,
    parent: tk.Misc,
    view_state: BackgroundThetaControlsViewState,
    background_theta_values_text: str,
    geometry_theta_offset_text: str,
    on_apply: Callable[[], None],
) -> None:
    """Create the workspace background-theta controls and store their refs."""

    controls = ttk.LabelFrame(parent, text="Background Theta_i")
    controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    ttk.Label(
        controls,
        text=_BACKGROUND_THETA_HELP_TEXT,
    ).pack(anchor=tk.W, padx=5, pady=(4, 0))

    background_theta_list_var = tk.StringVar(value=str(background_theta_values_text))
    row = ttk.Frame(controls)
    row.pack(fill=tk.X, padx=5, pady=(2, 4))
    background_theta_entry = ttk.Entry(
        row,
        textvariable=background_theta_list_var,
    )
    background_theta_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    geometry_theta_offset_var = tk.StringVar(value=str(geometry_theta_offset_text))
    ttk.Label(row, text="shared offset").pack(side=tk.LEFT, padx=(8, 4))
    background_theta_offset_entry = ttk.Entry(
        row,
        textvariable=geometry_theta_offset_var,
        width=10,
        justify=tk.RIGHT,
    )
    background_theta_offset_entry.pack(side=tk.LEFT)
    ttk.Button(
        row,
        text="Apply",
        command=on_apply,
    ).pack(side=tk.LEFT, padx=(6, 0))
    background_theta_entry.bind("<Return>", lambda _event: on_apply())
    background_theta_offset_entry.bind("<Return>", lambda _event: on_apply())

    view_state.background_theta_controls = controls
    view_state.background_theta_list_var = background_theta_list_var
    view_state.background_theta_entry = background_theta_entry
    view_state.geometry_theta_offset_var = geometry_theta_offset_var
    view_state.background_theta_offset_entry = background_theta_offset_entry


def create_geometry_fit_background_controls(
    *,
    parent: tk.Misc,
    view_state: BackgroundThetaControlsViewState,
    selection_text: str,
    on_apply: Callable[[], None],
) -> None:
    """Create the geometry-fit background selector controls and store refs."""

    controls = ttk.LabelFrame(parent, text="Geometry Fit Backgrounds")
    controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    ttk.Label(
        controls,
        text=_GEOMETRY_FIT_BACKGROUND_HELP_TEXT,
    ).pack(anchor=tk.W, padx=5, pady=(4, 0))

    selection_var = tk.StringVar(value=str(selection_text))
    row = ttk.Frame(controls)
    row.pack(fill=tk.X, padx=5, pady=(2, 4))
    geometry_fit_background_entry = ttk.Entry(
        row,
        textvariable=selection_var,
    )
    geometry_fit_background_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(
        row,
        text="Apply",
        command=on_apply,
    ).pack(side=tk.LEFT, padx=(6, 0))
    geometry_fit_background_entry.bind("<Return>", lambda _event: on_apply())

    view_state.geometry_fit_background_controls = controls
    view_state.geometry_fit_background_selection_var = selection_var
    view_state.geometry_fit_background_entry = geometry_fit_background_entry


def geometry_q_group_window_open(view_state: GeometryQGroupViewState) -> bool:
    """Return whether the Qr/Qz selector window currently exists."""

    if view_state.window is None:
        return False
    try:
        return bool(view_state.window.winfo_exists())
    except tk.TclError:
        return False


def set_geometry_q_group_status_text(
    view_state: GeometryQGroupViewState,
    text: str,
) -> None:
    """Update the summary label shown above the Qr/Qz selector list."""

    if view_state.status_label is None:
        return
    view_state.status_label.config(text=text)


def close_geometry_q_group_window(view_state: GeometryQGroupViewState) -> None:
    """Destroy the Qr/Qz selector window and clear its widget references."""

    if view_state.window is not None:
        try:
            view_state.window.destroy()
        except tk.TclError:
            pass
    view_state.window = None
    view_state.canvas = None
    view_state.body = None
    view_state.status_label = None


def open_geometry_q_group_window(
    *,
    root: tk.Misc,
    view_state: GeometryQGroupViewState,
    on_include_all: Callable[[], None],
    on_exclude_all: Callable[[], None],
    on_update_listed_peaks: Callable[[], None],
    on_save: Callable[[], None],
    on_load: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open the scrollable Qr/Qz selector window if needed."""

    if geometry_q_group_window_open(view_state):
        try:
            view_state.window.lift()
            view_state.window.focus_force()
        except tk.TclError:
            pass
        return False

    window = tk.Toplevel(root)
    window.title("Geometry Fit Qr/Qz Selector")
    window.geometry("860x520")
    window.minsize(680, 320)
    window.transient(root)

    frame = ttk.Frame(window, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frame,
        text=(
            "Select which simulated Qr/Qz groups are allowed into the live preview "
            "and geometry fitting. Only integer Gz/L groups are listed. "
            'Unchecked rows are skipped, and the list only changes when you press "Update Listed Peaks".'
        ),
        justify=tk.LEFT,
        wraplength=820,
    ).pack(anchor=tk.W, pady=(0, 6))

    status_label = ttk.Label(frame, text="")
    status_label.pack(anchor=tk.W, pady=(0, 6))

    list_frame = ttk.Frame(frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(list_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    body = ttk.Frame(canvas)
    body_window = canvas.create_window((0, 0), window=body, anchor="nw")

    body.bind(
        "<Configure>",
        lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(body_window, width=event.width),
    )

    actions = ttk.Frame(frame)
    actions.pack(fill=tk.X, pady=(8, 0))
    ttk.Button(
        actions,
        text="Include All",
        command=on_include_all,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        actions,
        text="Exclude All",
        command=on_exclude_all,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        actions,
        text="Update Listed Peaks",
        command=on_update_listed_peaks,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        actions,
        text="Save List...",
        command=on_save,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        actions,
        text="Load List...",
        command=on_load,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        actions,
        text="Close",
        command=on_close,
    ).pack(side=tk.RIGHT, padx=(5, 0))

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.bind("<Escape>", lambda _event: on_close())

    view_state.window = window
    view_state.canvas = canvas
    view_state.body = body
    view_state.status_label = status_label
    return True


def _sync_geometry_q_group_canvas(canvas: tk.Canvas) -> None:
    """Recompute the Qr/Qz selector scroll region after one render pass."""

    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))


def refresh_geometry_q_group_window(
    *,
    view_state: GeometryQGroupViewState,
    entries: Sequence[dict[str, object]],
    excluded_q_groups: Sequence[object],
    status_text: str,
    format_line: Callable[[dict[str, object]], str],
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
    clear_row_vars: Callable[[], None],
    register_row_var: Callable[[tuple[object, ...] | None, object], None],
    boolean_var_factory: Callable[[bool], object] | None = None,
) -> bool:
    """Redraw the Qr/Qz selector rows from the stored manual snapshot."""

    if not geometry_q_group_window_open(view_state):
        return False
    if view_state.body is None or view_state.canvas is None:
        return False

    try:
        yview = view_state.canvas.yview()
    except Exception:
        yview = (0.0, 1.0)

    for child in view_state.body.winfo_children():
        child.destroy()
    clear_row_vars()
    set_geometry_q_group_status_text(view_state, status_text)

    if not entries:
        ttk.Label(
            view_state.body,
            text=_GEOMETRY_Q_GROUP_EMPTY_TEXT,
            justify=tk.LEFT,
            wraplength=760,
        ).pack(anchor=tk.W, padx=8, pady=8)
        _sync_geometry_q_group_canvas(view_state.canvas)
        return True

    make_bool_var = boolean_var_factory
    if make_bool_var is None:
        make_bool_var = lambda included: tk.BooleanVar(value=included)

    excluded_keys = {
        tuple(raw_key) if isinstance(raw_key, list) else raw_key
        for raw_key in excluded_q_groups
    }
    for entry in entries:
        key = entry.get("key")
        included = key not in excluded_keys
        row_var = make_bool_var(included)
        register_row_var(key, row_var)
        ttk.Checkbutton(
            view_state.body,
            text=format_line(entry),
            variable=row_var,
            command=lambda row_key=key, var=row_var: on_toggle(row_key, var),
        ).pack(anchor=tk.W, fill=tk.X, padx=6, pady=1)

    _sync_geometry_q_group_canvas(view_state.canvas)
    if yview:
        try:
            first = float(yview[0])
        except Exception:
            first = float("nan")
        if math.isfinite(first):
            view_state.canvas.yview_moveto(first)
    return True


def bragg_qr_manager_window_open(view_state: BraggQrManagerViewState) -> bool:
    """Return whether the Bragg Qr manager window currently exists."""

    if view_state.window is None:
        return False
    try:
        return bool(view_state.window.winfo_exists())
    except tk.TclError:
        return False


def set_bragg_qr_manager_status_text(
    view_state: BraggQrManagerViewState,
    *,
    qr_text: str | None = None,
    l_text: str | None = None,
) -> None:
    """Update one or both status labels in the Bragg Qr manager."""

    if qr_text is not None and view_state.qr_status_label is not None:
        view_state.qr_status_label.config(text=qr_text)
    if l_text is not None and view_state.l_status_label is not None:
        view_state.l_status_label.config(text=l_text)


def close_bragg_qr_manager_window(view_state: BraggQrManagerViewState) -> None:
    """Destroy the Bragg Qr manager window and clear its widget references."""

    if view_state.window is not None:
        try:
            view_state.window.destroy()
        except tk.TclError:
            pass
    view_state.window = None
    view_state.qr_listbox = None
    view_state.qr_status_label = None
    view_state.l_listbox = None
    view_state.l_status_label = None


def open_bragg_qr_manager_window(
    *,
    root: tk.Misc,
    view_state: BraggQrManagerViewState,
    on_qr_selection_changed: Callable[[object], None],
    on_toggle_qr: Callable[[object], None],
    on_toggle_l: Callable[[object], None],
    on_enable_selected_qr: Callable[[], None],
    on_disable_selected_qr: Callable[[], None],
    on_enable_all_qr: Callable[[], None],
    on_disable_all_qr: Callable[[], None],
    on_enable_selected_l: Callable[[], None],
    on_disable_selected_l: Callable[[], None],
    on_enable_all_l: Callable[[], None],
    on_disable_all_l: Callable[[], None],
    on_refresh: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open the Bragg Qr manager window if needed."""

    if bragg_qr_manager_window_open(view_state):
        try:
            view_state.window.lift()
            view_state.window.focus_force()
        except tk.TclError:
            pass
        return False

    window = tk.Toplevel(root)
    window.title("Bragg Qr Group Manager")
    window.geometry("1020x520")
    window.minsize(740, 360)
    window.transient(root)

    frame = ttk.Frame(window, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frame,
        text="Enable/disable Bragg peaks grouped by identical Qr (same m = h^2 + hk + k^2).",
        wraplength=900,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(0, 6))

    lists_container = ttk.Frame(frame)
    lists_container.pack(fill=tk.BOTH, expand=True)

    qr_frame = ttk.LabelFrame(lists_container, text="Qr Groups")
    qr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
    l_frame = ttk.LabelFrame(lists_container, text="L Values of Selected Qr")
    l_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    qr_status_label = ttk.Label(qr_frame, text="")
    qr_status_label.pack(anchor=tk.W, pady=(0, 6), padx=6)

    qr_list_frame = ttk.Frame(qr_frame)
    qr_list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
    qr_y_scroll = ttk.Scrollbar(qr_list_frame, orient=tk.VERTICAL)
    qr_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    qr_listbox = tk.Listbox(
        qr_list_frame,
        selectmode=tk.EXTENDED,
        exportselection=False,
        yscrollcommand=qr_y_scroll.set,
    )
    qr_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    qr_y_scroll.config(command=qr_listbox.yview)

    qr_listbox.bind("<Double-Button-1>", on_toggle_qr)
    qr_listbox.bind("<space>", on_toggle_qr)
    qr_listbox.bind("<<ListboxSelect>>", on_qr_selection_changed)

    l_status_label = ttk.Label(l_frame, text="")
    l_status_label.pack(anchor=tk.W, pady=(0, 6), padx=6)

    l_list_frame = ttk.Frame(l_frame)
    l_list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
    l_y_scroll = ttk.Scrollbar(l_list_frame, orient=tk.VERTICAL)
    l_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    l_listbox = tk.Listbox(
        l_list_frame,
        selectmode=tk.EXTENDED,
        exportselection=False,
        yscrollcommand=l_y_scroll.set,
    )
    l_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    l_y_scroll.config(command=l_listbox.yview)

    l_listbox.bind("<Double-Button-1>", on_toggle_l)
    l_listbox.bind("<space>", on_toggle_l)

    qr_actions = ttk.Frame(frame)
    qr_actions.pack(fill=tk.X, pady=(10, 0))
    ttk.Button(
        qr_actions,
        text="Enable Selected",
        command=on_enable_selected_qr,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        qr_actions,
        text="Disable Selected",
        command=on_disable_selected_qr,
    ).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(
        qr_actions,
        text="Enable All",
        command=on_enable_all_qr,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        qr_actions,
        text="Disable All",
        command=on_disable_all_qr,
    ).pack(side=tk.LEFT, padx=(0, 18))

    l_actions = ttk.Frame(frame)
    l_actions.pack(fill=tk.X, pady=(8, 0))
    ttk.Button(
        l_actions,
        text="Enable Selected L",
        command=on_enable_selected_l,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        l_actions,
        text="Disable Selected L",
        command=on_disable_selected_l,
    ).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(
        l_actions,
        text="Enable All L (Selected Qr)",
        command=on_enable_all_l,
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        l_actions,
        text="Disable All L (Selected Qr)",
        command=on_disable_all_l,
    ).pack(side=tk.LEFT, padx=(0, 18))
    ttk.Button(
        l_actions,
        text="Refresh",
        command=on_refresh,
    ).pack(side=tk.LEFT)
    ttk.Button(
        l_actions,
        text="Close",
        command=on_close,
    ).pack(side=tk.RIGHT)

    window.protocol("WM_DELETE_WINDOW", on_close)

    view_state.window = window
    view_state.qr_listbox = qr_listbox
    view_state.qr_status_label = qr_status_label
    view_state.l_listbox = l_listbox
    view_state.l_status_label = l_status_label
    return True


def _replace_listbox_lines(
    listbox: object | None,
    lines: Sequence[str],
    selected_indices: Sequence[int] | None,
    *,
    see_index: int | None = None,
) -> bool:
    """Replace one Tk listbox contents and restore any requested selection."""

    if listbox is None:
        return False
    listbox.delete(0, tk.END)
    for line in lines:
        listbox.insert(tk.END, str(line))
    for raw_idx in selected_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < len(lines):
            listbox.selection_set(idx)
    if see_index is not None and 0 <= int(see_index) < len(lines):
        listbox.see(int(see_index))
    return True


def refresh_bragg_qr_manager_qr_list(
    *,
    view_state: BraggQrManagerViewState,
    lines: Sequence[str],
    selected_indices: Sequence[int] | None,
    status_text: str,
    see_index: int | None = None,
) -> bool:
    """Redraw the Bragg-Qr group list and status text."""

    if not bragg_qr_manager_window_open(view_state):
        return False
    if not _replace_listbox_lines(
        view_state.qr_listbox,
        lines,
        selected_indices,
        see_index=see_index,
    ):
        return False
    set_bragg_qr_manager_status_text(view_state, qr_text=status_text)
    return True


def refresh_bragg_qr_manager_l_list(
    *,
    view_state: BraggQrManagerViewState,
    lines: Sequence[str],
    selected_indices: Sequence[int] | None,
    status_text: str,
) -> bool:
    """Redraw the Bragg-Qr L-value list and status text."""

    if not bragg_qr_manager_window_open(view_state):
        return False
    if not _replace_listbox_lines(
        view_state.l_listbox,
        lines,
        selected_indices,
    ):
        return False
    set_bragg_qr_manager_status_text(view_state, l_text=status_text)
    return True


def hbn_geometry_debug_window_open(view_state: HbnGeometryDebugViewState) -> bool:
    """Return whether the hBN geometry debug viewer currently exists."""

    if view_state.window is None:
        return False
    try:
        return bool(view_state.window.winfo_exists())
    except tk.TclError:
        return False


def set_hbn_geometry_debug_text(
    view_state: HbnGeometryDebugViewState,
    text: str,
) -> None:
    """Replace the read-only debug report text shown in the viewer."""

    if view_state.text_widget is None:
        return
    try:
        view_state.text_widget.configure(state=tk.NORMAL)
        view_state.text_widget.delete("1.0", tk.END)
        view_state.text_widget.insert("1.0", str(text))
        view_state.text_widget.configure(state=tk.DISABLED)
    except tk.TclError:
        return


def close_hbn_geometry_debug_window(view_state: HbnGeometryDebugViewState) -> None:
    """Destroy the hBN geometry debug viewer and clear its widget references."""

    if view_state.window is not None:
        try:
            view_state.window.destroy()
        except tk.TclError:
            pass
    view_state.window = None
    view_state.text_widget = None


def open_hbn_geometry_debug_window(
    *,
    root: tk.Misc,
    view_state: HbnGeometryDebugViewState,
    text: str,
    on_close: Callable[[], None] | None = None,
) -> bool:
    """Open the hBN geometry debug viewer if needed and populate its report."""

    if hbn_geometry_debug_window_open(view_state):
        set_hbn_geometry_debug_text(view_state, text)
        try:
            view_state.window.lift()
            view_state.window.focus_force()
        except tk.TclError:
            pass
        return False

    if on_close is None:
        on_close = lambda: close_hbn_geometry_debug_window(view_state)

    window = tk.Toplevel(root)
    window.title("hBN Geometry Debug")
    window.geometry("980x560")

    frame = ttk.Frame(window, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)

    text_widget = tk.Text(frame, wrap=tk.NONE)
    text_widget.grid(row=0, column=0, sticky="nsew")
    y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text_widget.xview)
    x_scroll.grid(row=1, column=0, sticky="ew")
    text_widget.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    window.protocol("WM_DELETE_WINDOW", on_close)

    view_state.window = window
    view_state.text_widget = text_widget
    set_hbn_geometry_debug_text(view_state, text)
    return True
