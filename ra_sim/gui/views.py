"""GUI view helpers used by RA-SIM Tk applications."""

from __future__ import annotations

import math
import re
import webbrowser
from importlib.metadata import PackageNotFoundError, version as get_package_version
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any
import tkinter as tk
from tkinter import font as tkfont, ttk

from ra_sim.config import get_config_dir

from .collapsible import CollapsibleFrame
from .sliders import create_slider
from . import window_affinity
from .state import (
    AppShellViewState,
    AnalysisViewControlsViewState,
    AnalysisExportControlsViewState,
    BeamMosaicParameterSlidersViewState,
    BackgroundThetaControlsViewState,
    BackgroundBackendDebugViewState,
    BraggQrManagerViewState,
    CifWeightControlsViewState,
    DisplayControlsViewState,
    FiniteStackControlsViewState,
    GeometryOverlayActionsViewState,
    GeometryFitParameterControlsViewState,
    GeometryToolActionsViewState,
    GeometryFitConstraintsViewState,
    HklLookupViewState,
    GeometryQGroupViewState,
    HbnGeometryDebugViewState,
    IntegrationRangeControlsViewState,
    OrderedStructureFitControlsViewState,
    PrimaryCifControlsViewState,
    SamplingOpticsControlsViewState,
    StackingParameterControlsViewState,
    StatusPanelViewState,
    StructureFactorPruningControlsViewState,
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
_PROJECT_GITHUB_URL = "https://github.com/DVBeckwitt/ra_sim"


def _configure_root_styles(root: tk.Misc) -> None:
    """Apply shared ttk styling that makes major GUI sections easier to scan."""

    try:
        style = ttk.Style(root)
    except Exception:
        return

    heading_font = None
    try:
        heading_font = tkfont.nametofont("TkDefaultFont").copy()
        heading_font.configure(weight="bold")
        setattr(root, "_ra_section_heading_font", heading_font)
    except Exception:
        heading_font = None

    style.configure(
        "TLabelframe",
        borderwidth=2,
        relief="groove",
        padding=(10, 8),
    )

    title_style_kwargs: dict[str, object] = {
        "padding": (4, 0, 4, 2),
    }
    if heading_font is not None:
        title_style_kwargs["font"] = heading_font
    style.configure("TLabelframe.Label", **title_style_kwargs)

    collapsible_style_kwargs: dict[str, object] = {
        "padding": (8, 6),
        "anchor": "w",
    }
    if heading_font is not None:
        collapsible_style_kwargs["font"] = heading_font
    style.configure("SectionHeader.Toolbutton", **collapsible_style_kwargs)

    style.configure("TNotebook.Tab", padding=(12, 8))


def create_root_window(title: str = "RA Simulation") -> tk.Tk:
    """Create and return a Tk root window with the provided title."""

    launch_context = window_affinity.capture_launch_window_context()
    root = tk.Tk()
    try:
        withdraw = getattr(root, "withdraw", None)
        if callable(withdraw):
            withdraw()
    except tk.TclError:
        pass
    root.title(title)
    try:
        setattr(root, "_ra_sim_launch_window_context", launch_context)
    except Exception:
        pass
    _configure_root_styles(root)
    window_affinity.apply_window_launch_context(root, context=launch_context)
    return root


def apply_launch_window_context(
    window: object,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """Reapply the launcher's desktop/monitor context to a Tk top-level."""

    context = getattr(window, "_ra_sim_launch_window_context", None)
    if context is None:
        context = window_affinity.capture_launch_window_context()
        try:
            setattr(window, "_ra_sim_launch_window_context", context)
        except Exception:
            pass
    return window_affinity.apply_window_launch_context(
        window,
        context=context,
        width=width,
        height=height,
    )


def _pointer_inside_widget(
    widget: object | None,
    *,
    pointer_x: int,
    pointer_y: int,
) -> bool:
    """Return whether one root-coordinate pointer lies inside a widget."""

    if widget is None:
        return False
    try:
        widget_x = int(widget.winfo_rootx())
        widget_y = int(widget.winfo_rooty())
        widget_width = int(widget.winfo_width())
        widget_height = int(widget.winfo_height())
    except Exception:
        return False
    return (
        widget_x <= int(pointer_x) <= widget_x + widget_width
        and widget_y <= int(pointer_y) <= widget_y + widget_height
    )


def _mousewheel_scroll_units(event: object) -> int:
    """Normalize mouse-wheel events into Tk canvas scroll units."""

    raw_delta = getattr(event, "delta", 0)
    try:
        raw_delta = float(raw_delta)
    except Exception:
        raw_delta = 0.0
    if raw_delta:
        steps = max(1, int(abs(raw_delta) / 120.0)) if abs(raw_delta) >= 120.0 else 1
        return -steps if raw_delta > 0 else steps
    event_num = getattr(event, "num", None)
    if event_num == 4:
        return -1
    if event_num == 5:
        return 1
    return 0


def _scroll_canvas_if_pointer_inside(
    canvas: object | None,
    *,
    pointer_x: int,
    pointer_y: int,
    event: object,
) -> str | None:
    """Scroll one canvas only when the pointer currently lies within it."""

    if not _pointer_inside_widget(
        canvas,
        pointer_x=pointer_x,
        pointer_y=pointer_y,
    ):
        return None
    delta = _mousewheel_scroll_units(event)
    if not delta:
        return None
    try:
        canvas.yview_scroll(delta, "units")
    except Exception:
        return None
    return "break"


def _dispatch_pointer_mousewheel(root: object, event: object) -> str | None:
    """Dispatch one wheel event to the most recently registered scroll target."""

    pointer_x = getattr(event, "x_root", None)
    pointer_y = getattr(event, "y_root", None)
    if pointer_x is None or pointer_y is None:
        pointer_x_fn = getattr(root, "winfo_pointerx", None)
        pointer_y_fn = getattr(root, "winfo_pointery", None)
        if not callable(pointer_x_fn) or not callable(pointer_y_fn):
            return None
        try:
            pointer_x = int(pointer_x_fn())
            pointer_y = int(pointer_y_fn())
        except Exception:
            return None

    registry = getattr(root, "_ra_pointer_mousewheel_registry", ())
    for _key, handler in reversed(tuple(registry)):
        try:
            result = handler(
                pointer_x=int(pointer_x),
                pointer_y=int(pointer_y),
                event=event,
            )
        except tk.TclError:
            continue
        if result == "break":
            return "break"
    return None


def _register_pointer_mousewheel_handler(
    root: object,
    *,
    key: object,
    handler: Callable[..., str | None],
) -> None:
    """Register one pointer-aware wheel handler on the shared Tk root."""

    bind_all = getattr(root, "bind_all", None)
    if not callable(bind_all):
        return

    registry = list(getattr(root, "_ra_pointer_mousewheel_registry", ()))
    registry = [
        (existing_key, existing_handler)
        for existing_key, existing_handler in registry
        if existing_key != key
    ]
    registry.append((key, handler))
    setattr(root, "_ra_pointer_mousewheel_registry", registry)

    if getattr(root, "_ra_pointer_mousewheel_bound", False):
        return

    def _dispatch(event: object) -> str | None:
        return _dispatch_pointer_mousewheel(root, event)

    bind_all("<MouseWheel>", _dispatch, add="+")
    bind_all("<Button-4>", _dispatch, add="+")
    bind_all("<Button-5>", _dispatch, add="+")
    setattr(root, "_ra_pointer_mousewheel_bound", True)


def _create_scrolled_frame(parent: tk.Misc) -> tuple[tk.Misc, tk.Misc, tk.Canvas]:
    """Return a vertically scrollable frame body for notebook/control panels."""

    container = ttk.Frame(parent)
    canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
    scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
    body = ttk.Frame(canvas)
    body_window = canvas.create_window((0, 0), window=body, anchor="nw")

    def _refresh_scrollregion(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _resize_body(event) -> None:
        canvas.itemconfigure(body_window, width=event.width)

    body.bind("<Configure>", _refresh_scrollregion)
    canvas.bind("<Configure>", _resize_body)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    return container, body, canvas


def _bind_notebook_state(
    notebook: Any,
    tab_var: Any,
    tab_frames: dict[str, Any],
) -> None:
    """Keep a notebook selection synchronized with a persisted ``StringVar``."""

    def _select_from_var(*_args) -> None:
        key = str(tab_var.get()).strip().lower()
        target = tab_frames.get(key)
        if target is None:
            return
        try:
            if str(notebook.select()) != str(target):
                notebook.select(target)
        except tk.TclError:
            return

    def _sync_from_notebook(_event=None) -> None:
        try:
            selected = notebook.select()
        except tk.TclError:
            return
        for key, tab in tab_frames.items():
            if str(tab) == str(selected):
                if tab_var.get() != key:
                    tab_var.set(key)
                break

    tab_var.trace_add("write", _select_from_var)
    notebook.bind("<<NotebookTabChanged>>", _sync_from_notebook, add="+")
    _select_from_var()


def _open_external_link(url: str) -> None:
    """Open one external URL with the system browser when possible."""

    try:
        webbrowser.open_new_tab(url)
    except Exception:
        try:
            webbrowser.open(url)
        except Exception:
            return


def _get_app_version_text() -> str:
    """Return the best available RA-SIM version string."""

    try:
        return str(get_package_version("ra_sim"))
    except PackageNotFoundError:
        pass
    except Exception:
        return "unknown"

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"

    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)
    if version_match is None:
        return "unknown"
    return str(version_match.group(1)).strip() or "unknown"


def create_app_shell(
    *,
    root: tk.Misc,
    view_state: AppShellViewState,
) -> None:
    """Create the shared top-level GUI shell and notebook layout."""

    main_pane = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

    controls_panel = ttk.Frame(main_pane)
    figure_panel = ttk.Frame(main_pane)
    main_pane.add(controls_panel, weight=1)
    main_pane.add(figure_panel, weight=3)

    session_summary_frame = ttk.LabelFrame(
        controls_panel,
        text="Current Session",
        padding=(8, 6),
    )
    session_summary_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))
    session_summary_var = tk.StringVar(
        value=(
            "Background: not loaded\n"
            "CIF: not loaded\n"
            "Fit backgrounds: current\n"
            "View: Detector\n"
            "Fit quality: waiting for fit"
        )
    )
    session_summary_label = ttk.Label(
        session_summary_frame,
        textvariable=session_summary_var,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=520,
    )
    session_summary_label.pack(fill=tk.X)

    # The current-mode banner was removed from the visible shell layout, but the
    # view-state fields remain available so existing runtime update hooks can
    # call the shared setter helpers without additional branching.
    mode_banner_frame = None
    mode_banner_title_var = None
    mode_banner_title_label = None
    mode_banner_detail_var = None
    mode_banner_detail_label = None

    run_status_frame = ttk.Frame(figure_panel, padding=(10, 8, 10, 4))
    run_status_frame.pack(side=tk.TOP, fill=tk.X)
    run_status_var = tk.StringVar(value="State: startup")
    run_status_label = ttk.Label(
        run_status_frame,
        textvariable=run_status_var,
        anchor=tk.W,
        justify=tk.LEFT,
    )
    run_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Separator(figure_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

    controls_notebook = ttk.Notebook(controls_panel)
    controls_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

    setup_tab = ttk.Frame(controls_notebook)
    match_tab = ttk.Frame(controls_notebook)
    refine_tab = ttk.Frame(controls_notebook)
    analyze_tab = ttk.Frame(controls_notebook)
    help_tab = ttk.Frame(controls_notebook)
    controls_notebook.add(setup_tab, text="Setup")
    controls_notebook.add(match_tab, text="Match")
    controls_notebook.add(refine_tab, text="Refine")
    controls_notebook.add(analyze_tab, text="Analyze")
    controls_notebook.add(help_tab, text="Help")

    setup_scroll_frame, setup_body, setup_canvas = _create_scrolled_frame(setup_tab)
    setup_scroll_frame.pack(fill=tk.BOTH, expand=True)
    match_scroll_frame, match_body, match_canvas = _create_scrolled_frame(match_tab)
    match_scroll_frame.pack(fill=tk.BOTH, expand=True)

    parameter_notebook = ttk.Notebook(refine_tab)
    parameter_notebook.pack(fill=tk.BOTH, expand=True)
    refine_basic_tab = ttk.Frame(parameter_notebook)
    refine_advanced_tab = ttk.Frame(parameter_notebook)
    parameter_notebook.add(refine_basic_tab, text="Basic")
    parameter_notebook.add(refine_advanced_tab, text="Advanced")
    refine_basic_scroll, refine_basic_body, refine_basic_canvas = (
        _create_scrolled_frame(refine_basic_tab)
    )
    refine_basic_scroll.pack(fill=tk.BOTH, expand=True)
    refine_advanced_scroll, refine_advanced_body, refine_advanced_canvas = (
        _create_scrolled_frame(refine_advanced_tab)
    )
    refine_advanced_scroll.pack(fill=tk.BOTH, expand=True)

    _register_pointer_mousewheel_handler(
        root,
        key=("app-shell-scroll", id(view_state), "setup"),
        handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
            setup_canvas,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            event=event,
        ),
    )
    _register_pointer_mousewheel_handler(
        root,
        key=("app-shell-scroll", id(view_state), "match"),
        handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
            match_canvas,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            event=event,
        ),
    )
    _register_pointer_mousewheel_handler(
        root,
        key=("app-shell-scroll", id(view_state), "refine-basic"),
        handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
            refine_basic_canvas,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            event=event,
        ),
    )
    _register_pointer_mousewheel_handler(
        root,
        key=("app-shell-scroll", id(view_state), "refine-advanced"),
        handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
            refine_advanced_canvas,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            event=event,
        ),
    )

    control_tab_var = tk.StringVar(value="setup")
    parameter_tab_var = tk.StringVar(value="basic")
    _bind_notebook_state(
        controls_notebook,
        control_tab_var,
        {
            "setup": setup_tab,
            "match": match_tab,
            "refine": refine_tab,
            "analyze": analyze_tab,
            "help": help_tab,
        },
    )
    _bind_notebook_state(
        parameter_notebook,
        parameter_tab_var,
        {
            "basic": refine_basic_tab,
            "advanced": refine_advanced_tab,
        },
    )

    def _add_match_guidance(parent: tk.Misc, text: str) -> None:
        ttk.Label(
            parent,
            text=text,
            justify=tk.LEFT,
            wraplength=520,
        ).pack(fill=tk.X, padx=6, pady=(4, 6))

    match_backgrounds_frame = ttk.LabelFrame(
        match_body,
        text="Step 1. Choose Fit Backgrounds",
    )
    match_backgrounds_frame.pack(fill=tk.X, padx=5, pady=5)
    _add_match_guidance(
        match_backgrounds_frame,
        "Choose which loaded backgrounds participate in geometry fitting. "
        "Confirm per-image theta values in Setup before running a fit.",
    )

    match_peak_tools_frame = ttk.LabelFrame(
        match_body,
        text="Step 2. Choose or Place Peaks",
    )
    match_peak_tools_frame.pack(fill=tk.X, padx=5, pady=5)
    _add_match_guidance(
        match_peak_tools_frame,
        "Use image-pick, HKL lookup, and manual placement tools to define "
        "peak correspondences before fitting geometry.",
    )

    match_parameter_frame = ttk.LabelFrame(
        match_body,
        text="Step 3. Choose What Can Move",
    )
    match_parameter_frame.pack(fill=tk.X, padx=5, pady=5)
    _add_match_guidance(
        match_parameter_frame,
        "Enable only the geometry parameters that should move during fitting, "
        "then tighten or relax the allowed deviation constraints.",
    )

    match_run_frame = ttk.LabelFrame(
        match_body,
        text="Step 4. Run Geometry Fit",
    )
    match_run_frame.pack(fill=tk.X, padx=5, pady=5)
    _add_match_guidance(
        match_run_frame,
        "Run the least-squares geometry fit after backgrounds, peaks, and "
        "movable parameters are set.",
    )

    match_results_frame = ttk.LabelFrame(
        match_body,
        text="Step 5. Review Results",
    )
    match_results_frame.pack(fill=tk.X, padx=5, pady=5)
    match_results_var = tk.StringVar(
        value="Fit results will appear here and in the status panel after a geometry fit."
    )
    match_results_label = ttk.Label(
        match_results_frame,
        textvariable=match_results_var,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=520,
    )
    match_results_label.pack(fill=tk.X, padx=6, pady=(4, 6))

    analysis_controls_frame = ttk.Frame(analyze_tab, padding=(10, 10, 10, 0))
    analysis_controls_frame.pack(side=tk.TOP, fill=tk.X)
    analysis_views_frame = ttk.LabelFrame(analysis_controls_frame, text="View Options")
    analysis_views_frame.pack(fill=tk.X, pady=(0, 5))
    analysis_exports_frame = ttk.LabelFrame(analysis_controls_frame, text="Exports")
    analysis_exports_frame.pack(fill=tk.X)

    app_version_text = _get_app_version_text()
    config_dir_text = str(get_config_dir())

    help_body = ttk.Frame(help_tab, padding=10)
    help_body.pack(fill=tk.BOTH, expand=True)
    help_quickstart_frame = ttk.LabelFrame(help_body, text="Quick Start", padding=10)
    help_quickstart_frame.pack(fill=tk.X)
    ttk.Label(
        help_quickstart_frame,
        text="1. Setup: load backgrounds and choose the primary CIF.",
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W)
    ttk.Label(
        help_quickstart_frame,
        text="2. Match: choose backgrounds, pick peaks, and run geometry fitting.",
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W, pady=(4, 0))
    ttk.Label(
        help_quickstart_frame,
        text="3. Refine: adjust core geometry and lattice values first, then open advanced controls only as needed.",
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W, pady=(4, 0))
    ttk.Label(
        help_quickstart_frame,
        text="4. Analyze: compare detector, caked, and 1D views to verify the model.",
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W, pady=(4, 0))
    ttk.Label(
        help_quickstart_frame,
        text="Modes: manual placement and HKL image-pick show their next action in the Current Mode panel.",
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W, pady=(8, 0))

    help_project_frame = ttk.LabelFrame(help_body, text="Project", padding=10)
    help_project_frame.pack(fill=tk.X, pady=(10, 0))
    ttk.Label(
        help_project_frame,
        text=f"Version: {app_version_text}",
    ).pack(anchor=tk.W, pady=(0, 8))
    ttk.Label(
        help_project_frame,
        text="GitHub repository:",
    ).pack(anchor=tk.W)
    ttk.Button(
        help_project_frame,
        text="Open GitHub Page",
        command=lambda: _open_external_link(_PROJECT_GITHUB_URL),
    ).pack(anchor=tk.W, pady=(6, 4))
    ttk.Label(
        help_project_frame,
        text=_PROJECT_GITHUB_URL,
    ).pack(anchor=tk.W)

    help_config_frame = ttk.LabelFrame(help_body, text="Configuration", padding=10)
    help_config_frame.pack(fill=tk.X, pady=(10, 0))
    ttk.Label(
        help_config_frame,
        text="Active config directory:",
    ).pack(anchor=tk.W)
    ttk.Label(
        help_config_frame,
        text=config_dir_text,
        justify=tk.LEFT,
        wraplength=420,
    ).pack(anchor=tk.W, pady=(4, 0))

    help_contact_frame = ttk.LabelFrame(help_body, text="Contact", padding=10)
    help_contact_frame.pack(fill=tk.X, pady=(10, 0))
    ttk.Label(help_contact_frame, text="David Beckwitt").pack(anchor=tk.W)
    ttk.Label(help_contact_frame, text="David.Beckwitt@proton.me").pack(
        anchor=tk.W,
        pady=(4, 0),
    )

    status_frame = ttk.LabelFrame(root, text="Status", padding=(6, 4))
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)

    fig_frame = ttk.Frame(figure_panel)
    fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas_frame = ttk.Frame(fig_frame)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left_col = ttk.Frame(refine_basic_body, padding=10)
    left_col.pack(fill=tk.BOTH, expand=True)

    right_col = ttk.Frame(refine_advanced_body, padding=10)
    right_col.pack(fill=tk.BOTH, expand=True)

    plot_frame_1d = ttk.Frame(analyze_tab, padding=(10, 8, 10, 10))
    plot_frame_1d.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    view_state.main_pane = main_pane
    view_state.controls_panel = controls_panel
    view_state.figure_panel = figure_panel
    view_state.session_summary_frame = session_summary_frame
    view_state.session_summary_var = session_summary_var
    view_state.session_summary_label = session_summary_label
    view_state.mode_banner_frame = mode_banner_frame
    view_state.mode_banner_title_var = mode_banner_title_var
    view_state.mode_banner_title_label = mode_banner_title_label
    view_state.mode_banner_detail_var = mode_banner_detail_var
    view_state.mode_banner_detail_label = mode_banner_detail_label
    view_state.run_status_frame = run_status_frame
    view_state.run_status_var = run_status_var
    view_state.run_status_label = run_status_label
    view_state.controls_notebook = controls_notebook
    view_state.setup_tab = setup_tab
    view_state.match_tab = match_tab
    view_state.refine_tab = refine_tab
    view_state.analyze_tab = analyze_tab
    view_state.workspace_tab = setup_tab
    view_state.fit_tab = match_tab
    view_state.parameters_tab = refine_tab
    view_state.analysis_tab = analyze_tab
    view_state.help_tab = help_tab
    view_state.setup_body = setup_body
    view_state.setup_canvas = setup_canvas
    view_state.match_body = match_body
    view_state.match_canvas = match_canvas
    view_state.refine_basic_tab = refine_basic_tab
    view_state.refine_advanced_tab = refine_advanced_tab
    view_state.refine_basic_body = refine_basic_body
    view_state.refine_basic_canvas = refine_basic_canvas
    view_state.refine_advanced_body = refine_advanced_body
    view_state.refine_advanced_canvas = refine_advanced_canvas
    view_state.workspace_body = setup_body
    view_state.workspace_canvas = setup_canvas
    view_state.fit_body = match_body
    view_state.fit_canvas = match_canvas
    view_state.parameter_notebook = parameter_notebook
    view_state.parameter_geometry_tab = refine_basic_tab
    view_state.parameter_structure_tab = refine_advanced_tab
    view_state.parameter_geometry_body = refine_basic_body
    view_state.parameter_geometry_canvas = refine_basic_canvas
    view_state.parameter_structure_body = refine_advanced_body
    view_state.parameter_structure_canvas = refine_advanced_canvas
    view_state.control_tab_var = control_tab_var
    view_state.parameter_tab_var = parameter_tab_var
    view_state.match_backgrounds_frame = match_backgrounds_frame
    view_state.match_peak_tools_frame = match_peak_tools_frame
    view_state.match_parameter_frame = match_parameter_frame
    view_state.match_run_frame = match_run_frame
    view_state.match_results_frame = match_results_frame
    view_state.match_results_var = match_results_var
    view_state.match_results_label = match_results_label
    view_state.fit_actions_frame = match_peak_tools_frame
    view_state.analysis_controls_frame = analysis_controls_frame
    view_state.analysis_views_frame = analysis_views_frame
    view_state.analysis_exports_frame = analysis_exports_frame
    view_state.status_frame = status_frame
    view_state.fig_frame = fig_frame
    view_state.canvas_frame = canvas_frame
    view_state.left_col = left_col
    view_state.right_col = right_col
    view_state.plot_frame_1d = plot_frame_1d


def set_app_shell_run_status_text(
    view_state: AppShellViewState,
    text: object,
) -> None:
    """Update the compact one-line runtime summary shown above the plot."""

    summary = " ".join(str(text).split())
    if view_state.run_status_var is not None:
        try:
            view_state.run_status_var.set(summary)
            return
        except tk.TclError:
            pass
    if view_state.run_status_label is not None:
        try:
            view_state.run_status_label.configure(text=summary)
        except tk.TclError:
            return


def set_app_shell_session_summary_text(
    view_state: AppShellViewState,
    text: object,
) -> None:
    """Update the always-visible workflow/session summary panel."""

    summary = "\n".join(str(text).splitlines())
    if view_state.session_summary_var is not None:
        try:
            view_state.session_summary_var.set(summary)
            return
        except tk.TclError:
            pass
    if view_state.session_summary_label is not None:
        try:
            view_state.session_summary_label.configure(text=summary)
        except tk.TclError:
            return


def set_app_shell_mode_banner_text(
    view_state: AppShellViewState,
    *,
    title: object,
    detail: object,
) -> None:
    """Update the current-mode banner shown above the workflow tabs."""

    title_text = " ".join(str(title).split())
    detail_text = " ".join(str(detail).split())

    if view_state.mode_banner_title_var is not None:
        try:
            view_state.mode_banner_title_var.set(title_text)
        except tk.TclError:
            pass
    elif view_state.mode_banner_title_label is not None:
        try:
            view_state.mode_banner_title_label.configure(text=title_text)
        except tk.TclError:
            pass

    if view_state.mode_banner_detail_var is not None:
        try:
            view_state.mode_banner_detail_var.set(detail_text)
            return
        except tk.TclError:
            pass
    if view_state.mode_banner_detail_label is not None:
        try:
            view_state.mode_banner_detail_label.configure(text=detail_text)
        except tk.TclError:
            return


def set_match_results_text(
    view_state: AppShellViewState,
    text: object,
) -> None:
    """Update the guided-fit results summary shown in the Match tab."""

    summary = " ".join(str(text).split())
    if view_state.match_results_var is not None:
        try:
            view_state.match_results_var.set(summary)
            return
        except tk.TclError:
            pass
    if view_state.match_results_label is not None:
        try:
            view_state.match_results_label.configure(text=summary)
        except tk.TclError:
            return


def set_collapsible_header_summary(
    frame: object | None,
    text: object,
) -> None:
    """Update one collapsible-frame header summary when supported."""

    setter = getattr(frame, "set_header_summary", None)
    if callable(setter):
        setter("" if text is None else str(text))


def _compact_status_text(text: object, *, max_chars: int = 120) -> str:
    summary = " ".join(str(text).split())
    if len(summary) > max_chars:
        return summary[: max_chars - 1] + "..."
    return summary


class ConsoleStatusLabel:
    """Mirror verbose GUI status messages to the terminal and keep GUI text compact."""

    def __init__(
        self,
        parent,
        *,
        name: str,
        max_gui_chars: int = 120,
        **label_kwargs,
    ) -> None:
        self._name = str(name)
        self._max_gui_chars = max(16, int(max_gui_chars))
        self._last_full_text = ""
        self._label = ttk.Label(
            parent,
            wraplength=0,
            justify=tk.LEFT,
            anchor=tk.W,
            **label_kwargs,
        )

    def config(self, cnf=None, **kwargs):
        options = {}
        if isinstance(cnf, dict):
            options.update(cnf)
        options.update(kwargs)
        if "text" in options:
            raw_text = "" if options["text"] is None else str(options["text"])
            if raw_text != self._last_full_text:
                if raw_text.strip():
                    print(f"[{self._name}] {raw_text}", flush=True)
                self._last_full_text = raw_text
            options["text"] = _compact_status_text(
                raw_text,
                max_chars=self._max_gui_chars,
            )
            options.pop("wraplength", None)
        return self._label.config(**options)

    configure = config

    def cget(self, key):
        return self._label.cget(key)

    def __getattr__(self, name):
        return getattr(self._label, name)


def create_status_panel(
    *,
    parent: tk.Misc,
    view_state: StatusPanelViewState,
) -> None:
    """Create the shared bottom status-panel labels and progress bar."""

    progress_label_positions = ConsoleStatusLabel(
        parent,
        name="positions",
        max_gui_chars=110,
    )
    progress_label_positions.pack(side=tk.BOTTOM, padx=5)

    progress_label_geometry = ConsoleStatusLabel(
        parent,
        name="geometry",
        max_gui_chars=110,
    )
    progress_label_geometry.pack(side=tk.BOTTOM, padx=5)

    ordered_structure_progressbar = ttk.Progressbar(
        parent,
        mode="indeterminate",
        length=240,
    )
    ordered_structure_progressbar.pack(side=tk.BOTTOM, padx=5, pady=(0, 2))

    progress_label_ordered_structure = ConsoleStatusLabel(
        parent,
        name="ordered-structure",
        max_gui_chars=110,
    )
    progress_label_ordered_structure.pack(side=tk.BOTTOM, padx=5)

    mosaic_progressbar = ttk.Progressbar(parent, mode="indeterminate", length=240)
    mosaic_progressbar.pack(side=tk.BOTTOM, padx=5, pady=(0, 2))

    progress_label_mosaic = ConsoleStatusLabel(
        parent,
        name="mosaic",
        max_gui_chars=110,
    )
    progress_label_mosaic.pack(side=tk.BOTTOM, padx=5)

    progress_label = ConsoleStatusLabel(
        parent,
        name="gui",
        max_gui_chars=110,
        font=("Helvetica", 8),
    )
    progress_label.pack(side=tk.BOTTOM, padx=5)

    update_timing_label = ttk.Label(
        parent,
        text="Timing | image generation: n/a | redraw/update: n/a | total: n/a",
        font=("Helvetica", 8),
    )
    update_timing_label.pack(side=tk.BOTTOM, padx=5)

    chi_square_label = ttk.Label(
        parent,
        text="Chi-Squared: ",
        font=("Helvetica", 8),
    )
    chi_square_label.pack(side=tk.BOTTOM, padx=5)

    view_state.progress_label_positions = progress_label_positions
    view_state.progress_label_geometry = progress_label_geometry
    view_state.ordered_structure_progressbar = ordered_structure_progressbar
    view_state.progress_label_ordered_structure = progress_label_ordered_structure
    view_state.mosaic_progressbar = mosaic_progressbar
    view_state.progress_label_mosaic = progress_label_mosaic
    view_state.progress_label = progress_label
    view_state.update_timing_label = update_timing_label
    view_state.chi_square_label = chi_square_label


def create_workspace_panels(
    *,
    parent: tk.Misc,
    view_state: WorkspacePanelsViewState,
) -> None:
    """Create and store the workspace action/background/session panel frames."""

    workspace_actions_frame = ttk.LabelFrame(parent, text="Setup Actions")
    workspace_actions_frame.pack(fill=tk.X, padx=5, pady=5)

    workspace_backgrounds_frame = ttk.LabelFrame(parent, text="Backgrounds")
    workspace_backgrounds_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    workspace_inputs_frame = ttk.LabelFrame(parent, text="Input Model")
    workspace_inputs_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    workspace_session_frame = ttk.LabelFrame(parent, text="Session")
    workspace_session_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    workspace_debug_frame = CollapsibleFrame(
        parent,
        text="Advanced / Debug",
        expanded=False,
    )
    workspace_debug_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    view_state.workspace_actions_frame = workspace_actions_frame
    view_state.workspace_backgrounds_frame = workspace_backgrounds_frame
    view_state.workspace_inputs_frame = workspace_inputs_frame
    view_state.workspace_session_frame = workspace_session_frame
    view_state.workspace_debug_frame = workspace_debug_frame


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


def create_primary_cif_controls(
    *,
    parent: tk.Misc,
    view_state: PrimaryCifControlsViewState,
    cif_path_text: str,
    on_apply_from_entry: Callable[[object | None], None],
    on_browse_primary_cif: Callable[[], None],
    on_open_diffuse_ht: Callable[[], None],
    on_export_diffuse_ht: Callable[[], None],
) -> None:
    """Create the primary-CIF path and diffuse-HT action controls."""

    cif_frame = CollapsibleFrame(parent, text="Primary CIF")
    cif_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(cif_frame.frame, text="Path").pack(anchor=tk.W, padx=5, pady=(2, 0))

    cif_file_var = tk.StringVar(value=str(cif_path_text))
    cif_entry = ttk.Entry(cif_frame.frame, textvariable=cif_file_var)
    cif_entry.pack(fill=tk.X, padx=5, pady=(2, 5))
    cif_entry.bind("<Return>", on_apply_from_entry)

    cif_actions_frame = ttk.Frame(cif_frame.frame)
    cif_actions_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    browse_button = ttk.Button(
        cif_actions_frame,
        text="Browse...",
        command=on_browse_primary_cif,
    )
    browse_button.pack(side=tk.LEFT, padx=(0, 5))

    apply_button = ttk.Button(
        cif_actions_frame,
        text="Apply",
        command=lambda: on_apply_from_entry(None),
    )
    apply_button.pack(side=tk.LEFT)

    diffuse_ht_button = ttk.Button(
        cif_actions_frame,
        text="Diffuse HT...",
        command=on_open_diffuse_ht,
    )
    diffuse_ht_button.pack(side=tk.LEFT, padx=(5, 0))

    export_diffuse_ht_button = ttk.Button(
        cif_actions_frame,
        text="Export HT .txt...",
        command=on_export_diffuse_ht,
    )
    export_diffuse_ht_button.pack(side=tk.LEFT, padx=(5, 0))

    view_state.cif_frame = cif_frame
    view_state.cif_file_var = cif_file_var
    view_state.cif_entry = cif_entry
    view_state.cif_actions_frame = cif_actions_frame
    view_state.browse_button = browse_button
    view_state.apply_button = apply_button
    view_state.diffuse_ht_button = diffuse_ht_button
    view_state.export_diffuse_ht_button = export_diffuse_ht_button


def create_geometry_fit_parameter_controls(
    *,
    parent: tk.Misc,
    view_state: GeometryFitParameterControlsViewState,
    initial_values: dict[str, bool] | None = None,
) -> None:
    """Create the fit-geometry parameter checklist and store its refs/vars."""

    values = dict(initial_values or {})
    frame = ttk.LabelFrame(parent, text="Fit geometry parameters")
    frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    specs = [
        ("zb", "fit_zb_var", "fit_zb_checkbutton", "z_b beam offset"),
        ("zs", "fit_zs_var", "fit_zs_checkbutton", "z_s sample offset"),
        ("theta_initial", "fit_theta_var", "fit_theta_checkbutton", "θ sample tilt"),
        ("psi_z", "fit_psi_z_var", "fit_psi_z_checkbutton", "Goniometer Axis Yaw (about z)"),
        ("chi", "fit_chi_var", "fit_chi_checkbutton", "χ sample pitch"),
        ("cor_angle", "fit_cor_var", "fit_cor_checkbutton", "Goniometer Axis Pitch (about y)"),
        ("gamma", "fit_gamma_var", "fit_gamma_checkbutton", "γ detector tilt"),
        ("Gamma", "fit_Gamma_var", "fit_Gamma_checkbutton", "Γ detector tilt"),
        ("corto_detector", "fit_dist_var", "fit_dist_checkbutton", "distance"),
        ("a", "fit_a_var", "fit_a_checkbutton", "a lattice"),
        ("c", "fit_c_var", "fit_c_checkbutton", "c lattice"),
        ("center_x", "fit_center_x_var", "fit_center_x_checkbutton", "center row"),
        ("center_y", "fit_center_y_var", "fit_center_y_checkbutton", "center col"),
    ]

    toggle_vars: dict[str, Any] = {}
    toggle_checkbuttons: dict[str, Any] = {}
    widgets: list[Any] = []
    for key, var_attr, widget_attr, label in specs:
        toggle_var = tk.BooleanVar(value=bool(values.get(key, False)))
        checkbutton = ttk.Checkbutton(frame, text=label, variable=toggle_var)
        setattr(view_state, var_attr, toggle_var)
        setattr(view_state, widget_attr, checkbutton)
        toggle_vars[key] = toggle_var
        toggle_checkbuttons[key] = checkbutton
        widgets.append(checkbutton)

    for col_idx in range(4):
        frame.columnconfigure(col_idx, weight=1)
    for idx, widget in enumerate(widgets):
        row_idx, col_idx = divmod(idx, 4)
        widget.grid(row=row_idx, column=col_idx, sticky="w", padx=4, pady=2)

    view_state.frame = frame
    view_state.toggle_vars = toggle_vars
    view_state.toggle_checkbuttons = toggle_checkbuttons


def create_cif_weight_controls(
    *,
    parent: tk.Misc,
    view_state: CifWeightControlsViewState,
    has_second_cif: bool,
    weight1: float,
    weight2: float,
) -> None:
    """Create the optional CIF-weight sliders and store their refs/vars."""

    weight1_var = tk.DoubleVar(value=float(weight1))
    weight2_var = tk.DoubleVar(value=float(weight2))
    weight1_scale = None
    weight2_scale = None
    frame = None

    if has_second_cif:
        frame = CollapsibleFrame(parent, text="CIF Weights")
        frame.pack(fill=tk.X, padx=5, pady=5)
        weight1_var, weight1_scale = create_slider(
            "CIF1 Weight",
            0.0,
            1.0,
            float(weight1),
            0.01,
            frame.frame,
        )
        weight2_var, weight2_scale = create_slider(
            "CIF2 Weight",
            0.0,
            1.0,
            float(weight2),
            0.01,
            frame.frame,
        )

    view_state.frame = frame
    view_state.weight1_var = weight1_var
    view_state.weight1_scale = weight1_scale
    view_state.weight2_var = weight2_var
    view_state.weight2_scale = weight2_scale


def _find_slider_entry(slider: object | None) -> object | None:
    """Return the entry widget paired with one slider row when discoverable."""

    master = getattr(slider, "master", None)
    children = getattr(master, "winfo_children", None)
    if not callable(children):
        return None
    for child in children():
        if child is slider:
            continue
        if callable(getattr(child, "bind", None)):
            return child
    return None


def _create_stored_slider(
    *,
    view_state: object,
    attr_prefix: str,
    label: str,
    min_val: float,
    max_val: float,
    initial_val: float,
    step_size: float,
    parent: tk.Misc,
    update_callback: Callable[[], None] | None,
    **slider_kwargs,
) -> None:
    """Create one slider and store its var/scale refs on ``view_state``."""

    slider_var, slider = create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback,
        **slider_kwargs,
    )
    setattr(view_state, f"{attr_prefix}_var", slider_var)
    setattr(view_state, f"{attr_prefix}_scale", slider)


def create_display_controls(
    *,
    parent: tk.Misc,
    view_state: DisplayControlsViewState,
    background_range: tuple[float, float],
    background_defaults: tuple[float, float],
    background_step: float,
    background_transparency: float,
    simulation_range: tuple[float, float],
    simulation_defaults: tuple[float, float],
    simulation_step: float,
    scale_factor_range: tuple[float, float],
    scale_factor_value: float,
    scale_factor_step: float,
    on_apply_background_limits: Callable[[], None],
    on_apply_simulation_limits: Callable[[], None],
    fast_viewer_enabled: bool = False,
    on_toggle_fast_viewer: Callable[[], None] | None = None,
    fast_viewer_status_text: str = "",
) -> None:
    """Create the background/simulation display-control panels."""

    frame = ttk.Frame(parent)
    frame.pack(side=tk.BOTTOM, fill=tk.X)

    background_controls = ttk.LabelFrame(frame, text="Background Display")
    background_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    simulation_controls = ttk.LabelFrame(frame, text="Simulation Display")
    simulation_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    background_min_var, background_min_slider = create_slider(
        "Background Min Intensity",
        float(background_range[0]),
        float(background_range[1]),
        float(background_defaults[0]),
        float(background_step),
        parent=background_controls,
        update_callback=on_apply_background_limits,
    )
    background_max_var, background_max_slider = create_slider(
        "Background Max Intensity",
        float(background_range[0]),
        float(background_range[1]),
        float(background_defaults[1]),
        float(background_step),
        parent=background_controls,
        update_callback=on_apply_background_limits,
    )
    background_transparency_var, background_transparency_slider = create_slider(
        "Background Transparency",
        0.0,
        1.0,
        float(background_transparency),
        0.01,
        parent=background_controls,
        update_callback=on_apply_background_limits,
    )

    simulation_min_var, simulation_min_slider = create_slider(
        "Simulation Min Intensity",
        float(simulation_range[0]),
        float(simulation_range[1]),
        float(simulation_defaults[0]),
        float(simulation_step),
        parent=simulation_controls,
        update_callback=on_apply_simulation_limits,
    )
    simulation_max_var, simulation_max_slider = create_slider(
        "Simulation Max Intensity",
        float(simulation_range[0]),
        float(simulation_range[1]),
        float(simulation_defaults[1]),
        float(simulation_step),
        parent=simulation_controls,
        update_callback=on_apply_simulation_limits,
    )
    simulation_scale_factor_var, scale_factor_slider = create_slider(
        "Simulation Scale Factor",
        float(scale_factor_range[0]),
        float(scale_factor_range[1]),
        float(scale_factor_value),
        float(scale_factor_step),
        parent=simulation_controls,
    )
    fast_viewer_var = tk.BooleanVar(value=bool(fast_viewer_enabled))
    fast_viewer_checkbutton = ttk.Checkbutton(
        simulation_controls,
        text="Fast Viewer Window",
        variable=fast_viewer_var,
        command=on_toggle_fast_viewer,
    )
    fast_viewer_checkbutton.pack(anchor=tk.W, padx=5, pady=(2, 0))
    fast_viewer_status_var = tk.StringVar(value=str(fast_viewer_status_text))
    fast_viewer_status_label = ttk.Label(
        simulation_controls,
        textvariable=fast_viewer_status_var,
        wraplength=320,
        justify=tk.LEFT,
    )
    fast_viewer_status_label.pack(anchor=tk.W, padx=5, pady=(2, 4))

    view_state.frame = frame
    view_state.background_controls_frame = background_controls
    view_state.simulation_controls_frame = simulation_controls
    view_state.background_min_var = background_min_var
    view_state.background_max_var = background_max_var
    view_state.background_transparency_var = background_transparency_var
    view_state.background_min_slider = background_min_slider
    view_state.background_max_slider = background_max_slider
    view_state.background_transparency_slider = background_transparency_slider
    view_state.simulation_min_var = simulation_min_var
    view_state.simulation_max_var = simulation_max_var
    view_state.simulation_scale_factor_var = simulation_scale_factor_var
    view_state.simulation_min_slider = simulation_min_slider
    view_state.simulation_max_slider = simulation_max_slider
    view_state.scale_factor_slider = scale_factor_slider
    view_state.scale_factor_entry = _find_slider_entry(scale_factor_slider)
    view_state.fast_viewer_var = fast_viewer_var
    view_state.fast_viewer_checkbutton = fast_viewer_checkbutton
    view_state.fast_viewer_status_var = fast_viewer_status_var
    view_state.fast_viewer_status_label = fast_viewer_status_label


def create_structure_factor_pruning_controls(
    *,
    parent: tk.Misc,
    view_state: StructureFactorPruningControlsViewState,
    sf_prune_bias_range: tuple[float, float],
    sf_prune_bias_value: float,
    solve_q_mode: str,
    solve_q_steps_range: tuple[float, float],
    solve_q_steps_value: float,
    solve_q_rel_tol_range: tuple[float, float],
    solve_q_rel_tol_value: float,
    status_text: str = "",
) -> None:
    """Create the SF-pruning / arc-integration controls and store refs."""

    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=(2, 2))
    ttk.Label(frame, text="Structure-Factor Pruning").pack(anchor=tk.W, padx=5)
    ttk.Label(
        frame,
        text="Bias: -2.0 keeps more, 0.0 recommended default, +2.0 prunes much harder.",
    ).pack(anchor=tk.W, padx=5)

    sf_prune_bias_var, sf_prune_bias_scale = create_slider(
        "SF Prune Bias",
        float(sf_prune_bias_range[0]),
        float(sf_prune_bias_range[1]),
        float(sf_prune_bias_value),
        0.01,
        frame,
        update_callback=None,
    )

    sf_prune_status_var = tk.StringVar(value=str(status_text))
    sf_prune_status_label = ttk.Label(
        frame,
        textvariable=sf_prune_status_var,
        wraplength=420,
        justify=tk.LEFT,
    )
    sf_prune_status_label.pack(anchor=tk.W, padx=5, pady=(0, 2))

    ttk.Label(
        frame,
        text=(
            "Uniform mode uses fixed arc sampling (fast). Adaptive mode uses "
            "max-interval budget + relative tolerance (robust tails, slower)."
        ),
    ).pack(anchor=tk.W, padx=5, pady=(4, 0))

    solve_q_mode_var = tk.StringVar(value=str(solve_q_mode))
    solve_q_mode_row = ttk.Frame(frame)
    solve_q_mode_row.pack(fill=tk.X, padx=5, pady=(2, 2))
    ttk.Label(solve_q_mode_row, text="Arc Integration Mode").pack(anchor=tk.W)

    solve_q_uniform_button = ttk.Radiobutton(
        solve_q_mode_row,
        text="Uniform Fast",
        variable=solve_q_mode_var,
        value="uniform",
    )
    solve_q_uniform_button.pack(anchor=tk.W, padx=10)

    solve_q_adaptive_button = ttk.Radiobutton(
        solve_q_mode_row,
        text="Adaptive Robust",
        variable=solve_q_mode_var,
        value="adaptive",
    )
    solve_q_adaptive_button.pack(anchor=tk.W, padx=10)

    solve_q_steps_var, solve_q_steps_scale = create_slider(
        "Arc Max Intervals",
        float(solve_q_steps_range[0]),
        float(solve_q_steps_range[1]),
        float(solve_q_steps_value),
        1.0,
        frame,
        update_callback=None,
    )
    solve_q_rel_tol_var, solve_q_rel_tol_scale = create_slider(
        "Arc Relative Tol",
        float(solve_q_rel_tol_range[0]),
        float(solve_q_rel_tol_range[1]),
        float(solve_q_rel_tol_value),
        1e-6,
        frame,
        update_callback=None,
    )

    view_state.frame = frame
    view_state.sf_prune_bias_var = sf_prune_bias_var
    view_state.sf_prune_bias_scale = sf_prune_bias_scale
    view_state.sf_prune_status_var = sf_prune_status_var
    view_state.sf_prune_status_label = sf_prune_status_label
    view_state.solve_q_mode_row = solve_q_mode_row
    view_state.solve_q_mode_var = solve_q_mode_var
    view_state.solve_q_uniform_button = solve_q_uniform_button
    view_state.solve_q_adaptive_button = solve_q_adaptive_button
    view_state.solve_q_steps_var = solve_q_steps_var
    view_state.solve_q_steps_scale = solve_q_steps_scale
    view_state.solve_q_rel_tol_var = solve_q_rel_tol_var
    view_state.solve_q_rel_tol_scale = solve_q_rel_tol_scale


def set_structure_factor_pruning_status_text(
    view_state: StructureFactorPruningControlsViewState,
    text: str,
) -> None:
    """Update the SF-pruning status label text."""

    setter = getattr(view_state.sf_prune_status_var, "set", None)
    if callable(setter):
        setter(str(text))


def set_structure_factor_pruning_rel_tol_enabled(
    view_state: StructureFactorPruningControlsViewState,
    *,
    enabled: bool,
) -> None:
    """Enable or disable the adaptive relative-tolerance slider."""

    state_value = "normal" if enabled else "disabled"
    _configure_widget_state(view_state.solve_q_rel_tol_scale, state_value)


def create_beam_mosaic_parameter_sliders(
    *,
    geometry_parent: tk.Misc,
    debye_parent: tk.Misc,
    detector_parent: tk.Misc,
    lattice_parent: tk.Misc,
    mosaic_parent: tk.Misc,
    beam_parent: tk.Misc,
    view_state: BeamMosaicParameterSlidersViewState,
    image_size: float,
    values: dict[str, float],
    on_standard_update: Callable[[], None],
    on_mosaic_update: Callable[[], None],
) -> None:
    """Create the main beam/geometry/mosaic parameter sliders."""

    def _standard(attr_prefix: str, label: str, min_val: float, max_val: float, initial_val: float, step_size: float, parent: tk.Misc, **slider_kwargs) -> None:
        _create_stored_slider(
            view_state=view_state,
            attr_prefix=attr_prefix,
            label=label,
            min_val=min_val,
            max_val=max_val,
            initial_val=initial_val,
            step_size=step_size,
            parent=parent,
            update_callback=on_standard_update,
            **slider_kwargs,
        )

    def _mosaic(attr_prefix: str, label: str, min_val: float, max_val: float, initial_val: float, step_size: float, parent: tk.Misc, **slider_kwargs) -> None:
        _create_stored_slider(
            view_state=view_state,
            attr_prefix=attr_prefix,
            label=label,
            min_val=min_val,
            max_val=max_val,
            initial_val=initial_val,
            step_size=step_size,
            parent=parent,
            update_callback=on_mosaic_update,
            **slider_kwargs,
        )

    _standard(
        "theta_initial",
        "θ sample tilt",
        0.5,
        30.0,
        float(values["theta_initial"]),
        0.01,
        geometry_parent,
    )
    _standard(
        "cor_angle",
        "Goniometer Axis Pitch (about y)",
        -5.0,
        5.0,
        float(values["cor_angle"]),
        0.01,
        geometry_parent,
    )
    _standard(
        "gamma",
        "γ detector pitch",
        -4.0,
        4.0,
        float(values["gamma"]),
        0.001,
        geometry_parent,
        allow_range_expand=True,
    )
    _standard(
        "Gamma",
        "Γ detector yaw",
        -4.0,
        4.0,
        float(values["Gamma"]),
        0.001,
        geometry_parent,
        allow_range_expand=True,
    )
    _standard(
        "chi",
        "χ sample pitch",
        -1.0,
        1.0,
        float(values["chi"]),
        0.001,
        geometry_parent,
    )
    _standard(
        "psi_z",
        "Goniometer Axis Yaw (about z)",
        -5.0,
        5.0,
        float(values["psi_z"]),
        0.01,
        geometry_parent,
    )
    _standard(
        "zs",
        "z_s sample offset",
        -2.0e-3,
        2.0e-3,
        float(values["zs"]),
        0.0001,
        geometry_parent,
    )
    _standard(
        "zb",
        "z_b beam offset",
        -2.0e-3,
        2.0e-3,
        float(values["zb"]),
        0.0001,
        geometry_parent,
    )
    _standard(
        "sample_width",
        "Sample Width (m)",
        0.0,
        20.0e-3,
        float(values["sample_width_m"]),
        0.1e-3,
        geometry_parent,
        allow_range_expand=True,
        range_expand_pad=1.0e-3,
    )
    _standard(
        "sample_length",
        "Sample Length (m)",
        0.0,
        20.0e-3,
        float(values["sample_length_m"]),
        0.1e-3,
        geometry_parent,
        allow_range_expand=True,
        range_expand_pad=1.0e-3,
    )
    _standard(
        "sample_depth",
        "Sample Depth (m)",
        0.0,
        1.0e-6,
        float(values["sample_depth_m"]),
        1.0e-9,
        geometry_parent,
        allow_range_expand=True,
        range_expand_pad=1.0e-7,
    )
    _standard(
        "debye_x",
        "Debye Qz",
        0.0,
        1.0,
        float(values["debye_x"]),
        0.001,
        debye_parent,
    )
    _standard(
        "debye_y",
        "Debye Qr",
        0.0,
        1.0,
        float(values["debye_y"]),
        0.001,
        debye_parent,
    )
    _standard(
        "corto_detector",
        "δ detector distance",
        0.0,
        100e-3,
        float(values["corto_detector"]),
        0.1e-3,
        detector_parent,
    )
    _standard(
        "a",
        "a (Å)",
        3.5,
        8.0,
        float(values["a"]),
        0.01,
        lattice_parent,
        allow_range_expand=True,
        range_expand_pad=0.1,
    )
    _standard(
        "c",
        "c (Å)",
        20.0,
        40.0,
        float(values["c"]),
        0.01,
        lattice_parent,
        allow_range_expand=True,
        range_expand_pad=0.5,
    )
    _mosaic(
        "sigma_mosaic",
        "σ Mosaic (deg)",
        0.0,
        5.0,
        float(values["sigma_mosaic_deg"]),
        0.01,
        mosaic_parent,
    )
    _mosaic(
        "gamma_mosaic",
        "γ Mosaic (deg)",
        0.0,
        5.0,
        float(values["gamma_mosaic_deg"]),
        0.01,
        mosaic_parent,
    )
    _mosaic(
        "eta",
        "η (fraction)",
        0.0,
        1.0,
        float(values["eta"]),
        0.001,
        mosaic_parent,
    )

    beam_center_slider_max = max(3000.0, float(image_size))
    _standard(
        "center_x",
        "Beam Center Row",
        0.0,
        beam_center_slider_max,
        float(values["center_x"]),
        1.0,
        beam_parent,
    )
    _mosaic(
        "bandwidth_percent",
        "Bandwidth (%)",
        0.0,
        5.0,
        float(values["bandwidth_percent"]),
        0.01,
        beam_parent,
    )
    _standard(
        "center_y",
        "Beam Center Col",
        0.0,
        beam_center_slider_max,
        float(values["center_y"]),
        1.0,
        beam_parent,
    )


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


def _destroy_children(parent: object | None) -> None:
    children = getattr(parent, "winfo_children", None)
    if not callable(children):
        return
    for child in children():
        destroy = getattr(child, "destroy", None)
        if callable(destroy):
            destroy()


def create_ordered_structure_fit_panel(
    *,
    parent: tk.Misc,
    view_state: OrderedStructureFitControlsViewState,
    ordered_scale_var: Any,
    coord_window_var: Any,
    fit_debye_x_var: Any,
    fit_debye_y_var: Any,
    result_var: Any,
    on_fit: Callable[[], None],
    on_revert: Callable[[], None],
    on_commit_ordered_scale: Callable[..., None],
    on_commit_coord_window: Callable[..., None],
) -> None:
    """Create the ordered-structure fit panel and store widget references."""

    frame = CollapsibleFrame(parent, text="Ordered Structure Fit", expanded=True)
    frame.pack(fill=tk.X, padx=5, pady=5)

    actions_frame = ttk.Frame(frame.frame)
    actions_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
    fit_button = ttk.Button(
        actions_frame,
        text="Fit Ordered Structure",
        command=on_fit,
    )
    fit_button.pack(side=tk.LEFT)
    revert_button = ttk.Button(
        actions_frame,
        text="Revert Last Ordered Fit",
        command=on_revert,
        state=tk.DISABLED,
    )
    revert_button.pack(side=tk.LEFT, padx=(6, 0))

    settings_frame = ttk.Frame(frame.frame)
    settings_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(settings_frame, text="Ordered Intensity Scale").grid(
        row=0,
        column=0,
        sticky=tk.W,
        padx=(0, 6),
        pady=(0, 4),
    )
    ordered_scale_entry = ttk.Entry(
        settings_frame,
        textvariable=ordered_scale_var,
        width=10,
    )
    ordered_scale_entry.grid(row=0, column=1, sticky=tk.W, pady=(0, 4))
    ordered_scale_entry.bind("<Return>", on_commit_ordered_scale)
    ordered_scale_entry.bind("<FocusOut>", on_commit_ordered_scale)

    ttk.Label(settings_frame, text="Coordinate Window").grid(
        row=1,
        column=0,
        sticky=tk.W,
        padx=(0, 6),
        pady=(0, 4),
    )
    coord_window_entry = ttk.Entry(
        settings_frame,
        textvariable=coord_window_var,
        width=10,
    )
    coord_window_entry.grid(row=1, column=1, sticky=tk.W, pady=(0, 4))
    coord_window_entry.bind("<Return>", on_commit_coord_window)
    coord_window_entry.bind("<FocusOut>", on_commit_coord_window)

    fit_debye_x_checkbutton = ttk.Checkbutton(
        settings_frame,
        text="Fit Debye Qz",
        variable=fit_debye_x_var,
    )
    fit_debye_x_checkbutton.grid(row=2, column=0, sticky=tk.W, pady=(0, 2))
    fit_debye_y_checkbutton = ttk.Checkbutton(
        settings_frame,
        text="Fit Debye Qr",
        variable=fit_debye_y_var,
    )
    fit_debye_y_checkbutton.grid(row=2, column=1, sticky=tk.W, pady=(0, 2))

    occupancy_toggle_frame = ttk.Frame(frame.frame)
    occupancy_toggle_frame.pack(fill=tk.X, padx=5, pady=(0, 4))

    atom_toggle_frame = ttk.Frame(frame.frame)
    atom_toggle_frame.pack(fill=tk.X, padx=5, pady=(0, 4))

    result_label = ttk.Label(
        frame.frame,
        textvariable=result_var,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=420,
    )
    result_label.pack(fill=tk.X, padx=5, pady=(0, 5))

    view_state.frame = frame
    view_state.actions_frame = actions_frame
    view_state.fit_button = fit_button
    view_state.revert_button = revert_button
    view_state.settings_frame = settings_frame
    view_state.ordered_scale_var = ordered_scale_var
    view_state.ordered_scale_entry = ordered_scale_entry
    view_state.coord_window_var = coord_window_var
    view_state.coord_window_entry = coord_window_entry
    view_state.fit_debye_x_var = fit_debye_x_var
    view_state.fit_debye_x_checkbutton = fit_debye_x_checkbutton
    view_state.fit_debye_y_var = fit_debye_y_var
    view_state.fit_debye_y_checkbutton = fit_debye_y_checkbutton
    view_state.occupancy_toggle_frame = occupancy_toggle_frame
    view_state.atom_toggle_frame = atom_toggle_frame
    view_state.result_var = result_var
    view_state.result_label = result_label


def create_stacking_parameter_panels(
    *,
    parent: tk.Misc,
    view_state: StackingParameterControlsViewState,
) -> None:
    """Create collapsible stacking/occupancy/atom-site panels and store refs."""

    stack_frame = CollapsibleFrame(parent, text="Stacking Probabilities")
    stack_frame.pack(fill=tk.X, padx=5, pady=5)

    occupancy_frame = CollapsibleFrame(parent, text="Site Occupancies")
    occupancy_frame.pack(fill=tk.X, padx=5, pady=5)
    occ_slider_frame = ttk.Frame(occupancy_frame.frame)
    occ_slider_frame.pack(fill=tk.X, padx=0, pady=0)
    occ_entry_frame = ttk.Frame(occupancy_frame.frame)
    occ_entry_frame.pack(fill=tk.X, padx=5, pady=5)

    atom_site_frame = CollapsibleFrame(parent, text="Atom Site Fractional Coordinates")
    atom_site_frame.pack(fill=tk.X, padx=5, pady=5)
    atom_site_table_frame = ttk.Frame(atom_site_frame.frame)
    atom_site_table_frame.pack(fill=tk.X, padx=5, pady=5)

    view_state.stack_frame = stack_frame
    view_state.occupancy_frame = occupancy_frame
    view_state.occ_slider_frame = occ_slider_frame
    view_state.occ_entry_frame = occ_entry_frame
    view_state.atom_site_frame = atom_site_frame
    view_state.atom_site_table_frame = atom_site_table_frame


def rebuild_ordered_structure_fit_occupancy_controls(
    *,
    view_state: OrderedStructureFitControlsViewState,
    occupancy_vars: Sequence[Any],
    occupancy_label_text: Callable[[int], str],
    empty_text: str = "No occupancy sites available in the active CIF.",
) -> None:
    """Rebuild the ordered-fit occupancy selection checkboxes."""

    frame = view_state.occupancy_toggle_frame
    _destroy_children(frame)
    view_state.occupancy_toggle_widgets = []

    if frame is None:
        return

    if not occupancy_vars:
        label = ttk.Label(frame, text=empty_text, justify=tk.LEFT, anchor=tk.W)
        label.pack(fill=tk.X)
        view_state.occupancy_toggle_widgets = [label]
        return

    header = ttk.Label(frame, text="Fit Occupancies", justify=tk.LEFT, anchor=tk.W)
    header.pack(fill=tk.X)
    view_state.occupancy_toggle_widgets.append(header)
    for idx, var in enumerate(occupancy_vars):
        check = ttk.Checkbutton(
            frame,
            text=str(occupancy_label_text(idx)),
            variable=var,
        )
        check.pack(anchor=tk.W)
        view_state.occupancy_toggle_widgets.append(check)


def rebuild_ordered_structure_fit_atom_coordinate_controls(
    *,
    view_state: OrderedStructureFitControlsViewState,
    atom_toggle_vars: Sequence[dict[str, Any]],
    atom_site_label_text: Callable[[int], str],
    empty_text: str = "No atom-site fractional coordinates found in the active CIF.",
) -> None:
    """Rebuild the ordered-fit atom-coordinate selection checkboxes."""

    frame = view_state.atom_toggle_frame
    _destroy_children(frame)
    view_state.atom_toggle_widgets = []

    if frame is None:
        return

    if not atom_toggle_vars:
        label = ttk.Label(frame, text=empty_text, justify=tk.LEFT, anchor=tk.W)
        label.pack(fill=tk.X)
        view_state.atom_toggle_widgets = [label]
        return

    header = ttk.Label(frame, text="Fit Atom Coordinates", justify=tk.LEFT, anchor=tk.W)
    header.pack(fill=tk.X)
    view_state.atom_toggle_widgets.append(header)
    for idx, axis_vars in enumerate(atom_toggle_vars):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(row, text=str(atom_site_label_text(idx))).pack(side=tk.LEFT, padx=(0, 8))
        for axis in ("x", "y", "z"):
            check = ttk.Checkbutton(
                row,
                text=axis,
                variable=axis_vars.get(axis),
            )
            check.pack(side=tk.LEFT, padx=(0, 6))
            view_state.atom_toggle_widgets.append(check)


def create_stacking_probability_sliders(
    *,
    parent: tk.Misc,
    view_state: StackingParameterControlsViewState,
    values: dict[str, float],
    on_update: Callable[..., None],
) -> None:
    """Create the stacking probability/weight sliders and store refs."""

    _create_stored_slider(
        view_state=view_state,
        attr_prefix="p0",
        label="p≈0",
        min_val=0.0,
        max_val=0.2,
        initial_val=float(values["p0"]),
        step_size=0.001,
        parent=parent,
        update_callback=on_update,
    )
    _create_stored_slider(
        view_state=view_state,
        attr_prefix="w0",
        label="w(p≈0)%",
        min_val=0.0,
        max_val=100.0,
        initial_val=float(values["w0"]),
        step_size=0.1,
        parent=parent,
        update_callback=on_update,
    )
    _create_stored_slider(
        view_state=view_state,
        attr_prefix="p1",
        label="p≈1",
        min_val=0.8,
        max_val=1.0,
        initial_val=float(values["p1"]),
        step_size=0.001,
        parent=parent,
        update_callback=on_update,
    )
    _create_stored_slider(
        view_state=view_state,
        attr_prefix="w1",
        label="w(p≈1)%",
        min_val=0.0,
        max_val=100.0,
        initial_val=float(values["w1"]),
        step_size=0.1,
        parent=parent,
        update_callback=on_update,
    )
    _create_stored_slider(
        view_state=view_state,
        attr_prefix="p2",
        label="p",
        min_val=0.0,
        max_val=1.0,
        initial_val=float(values["p2"]),
        step_size=0.001,
        parent=parent,
        update_callback=on_update,
    )
    _create_stored_slider(
        view_state=view_state,
        attr_prefix="w2",
        label="w(p)%",
        min_val=0.0,
        max_val=100.0,
        initial_val=float(values["w2"]),
        step_size=0.1,
        parent=parent,
        update_callback=on_update,
    )


def rebuild_occupancy_controls(
    *,
    view_state: StackingParameterControlsViewState,
    occ_vars: Sequence[object],
    occupancy_label_text: Callable[[int], str],
    occupancy_input_label_text: Callable[[int], str],
    on_update: Callable[..., None],
) -> None:
    """Recreate the occupancy sliders and entry rows for the current vars."""

    _destroy_children(view_state.occ_slider_frame)
    _destroy_children(view_state.occ_entry_frame)

    view_state.occ_scale_widgets.clear()
    view_state.occ_label_widgets.clear()
    view_state.occ_entry_widgets.clear()
    view_state.occ_entry_label_widgets.clear()

    if view_state.occ_slider_frame is None or view_state.occ_entry_frame is None:
        return

    for idx, occ_var in enumerate(occ_vars):
        occ_label = ttk.Label(
            view_state.occ_slider_frame,
            text=str(occupancy_label_text(int(idx))),
        )
        occ_label.pack(padx=5, pady=2)
        view_state.occ_label_widgets.append(occ_label)

        occ_scale = ttk.Scale(
            view_state.occ_slider_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=occ_var,
            command=on_update,
        )
        occ_scale.pack(fill=tk.X, padx=5, pady=2)
        occ_scale.bind(
            "<ButtonRelease-1>",
            lambda _event, occ_var=occ_var: on_update(occ_var.get()),
        )
        view_state.occ_scale_widgets.append(occ_scale)

    for idx, occ_var in enumerate(occ_vars):
        occ_entry_label = ttk.Label(
            view_state.occ_entry_frame,
            text=str(occupancy_input_label_text(int(idx))) + ":",
        )
        occ_entry_label.grid(row=idx, column=0, sticky="w", padx=5, pady=2)
        view_state.occ_entry_label_widgets.append(occ_entry_label)

        occ_entry = ttk.Entry(view_state.occ_entry_frame, textvariable=occ_var, width=7)
        occ_entry.grid(row=idx, column=1, padx=5, pady=2, sticky="ew")
        occ_entry.bind("<Return>", on_update)
        occ_entry.bind("<FocusOut>", on_update)
        view_state.occ_entry_widgets.append(occ_entry)

    view_state.occ_entry_frame.columnconfigure(1, weight=1)


def rebuild_atom_site_fractional_controls(
    *,
    view_state: StackingParameterControlsViewState,
    atom_site_fract_vars: Sequence[object],
    atom_site_label_text: Callable[[int], str],
    on_update: Callable[..., None],
    empty_text: str,
) -> None:
    """Recreate the atom-site fractional-coordinate entry table."""

    _destroy_children(view_state.atom_site_table_frame)
    view_state.atom_site_coord_entry_widgets.clear()

    if view_state.atom_site_table_frame is None:
        return

    if not atom_site_fract_vars:
        ttk.Label(
            view_state.atom_site_table_frame,
            text=str(empty_text),
        ).grid(row=0, column=0, sticky="w", padx=2, pady=2)
        return

    ttk.Label(view_state.atom_site_table_frame, text="Site").grid(
        row=0,
        column=0,
        sticky="w",
        padx=2,
        pady=2,
    )
    ttk.Label(view_state.atom_site_table_frame, text="x").grid(
        row=0,
        column=1,
        sticky="w",
        padx=2,
        pady=2,
    )
    ttk.Label(view_state.atom_site_table_frame, text="y").grid(
        row=0,
        column=2,
        sticky="w",
        padx=2,
        pady=2,
    )
    ttk.Label(view_state.atom_site_table_frame, text="z").grid(
        row=0,
        column=3,
        sticky="w",
        padx=2,
        pady=2,
    )

    for idx, axis_vars in enumerate(atom_site_fract_vars):
        ttk.Label(
            view_state.atom_site_table_frame,
            text=str(atom_site_label_text(int(idx))),
        ).grid(row=idx + 1, column=0, sticky="w", padx=2, pady=2)

        entry_x = ttk.Entry(
            view_state.atom_site_table_frame,
            textvariable=axis_vars["x"],
            width=10,
        )
        entry_y = ttk.Entry(
            view_state.atom_site_table_frame,
            textvariable=axis_vars["y"],
            width=10,
        )
        entry_z = ttk.Entry(
            view_state.atom_site_table_frame,
            textvariable=axis_vars["z"],
            width=10,
        )
        entry_x.grid(row=idx + 1, column=1, padx=2, pady=2, sticky="ew")
        entry_y.grid(row=idx + 1, column=2, padx=2, pady=2, sticky="ew")
        entry_z.grid(row=idx + 1, column=3, padx=2, pady=2, sticky="ew")
        entry_x.bind("<Return>", on_update)
        entry_y.bind("<Return>", on_update)
        entry_z.bind("<Return>", on_update)
        entry_x.bind("<FocusOut>", on_update)
        entry_y.bind("<FocusOut>", on_update)
        entry_z.bind("<FocusOut>", on_update)
        view_state.atom_site_coord_entry_widgets.extend([entry_x, entry_y, entry_z])

    view_state.atom_site_table_frame.columnconfigure(0, weight=1)
    view_state.atom_site_table_frame.columnconfigure(1, weight=1)
    view_state.atom_site_table_frame.columnconfigure(2, weight=1)
    view_state.atom_site_table_frame.columnconfigure(3, weight=1)


def create_geometry_tool_action_controls(
    *,
    parent: tk.Misc,
    view_state: GeometryToolActionsViewState,
    on_toggle_manual_pick: Callable[[], None],
    on_undo_manual_placement: Callable[[], None],
    on_export_manual_pairs: Callable[[], None],
    on_import_manual_pairs: Callable[[], None],
    on_toggle_preview_exclude: Callable[[], None],
    on_clear_manual_pairs: Callable[[], None],
    manual_pick_text: str = "Pick Qr Sets on Image",
    preview_exclude_text: str = "Select Qr/Qz Peaks",
) -> None:
    """Create manual-geometry action controls and store their refs."""

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


def create_geometry_fit_history_controls(
    *,
    parent: tk.Misc,
    view_state: GeometryToolActionsViewState,
    on_undo_fit: Callable[[], None],
    on_redo_fit: Callable[[], None],
) -> None:
    """Create geometry-fit history controls and store their refs."""

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

    view_state.undo_geometry_fit_button = undo_geometry_fit_button
    view_state.redo_geometry_fit_button = redo_geometry_fit_button


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
        text="Fit Mosaic Shapes",
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

    show_1d_var = tk.BooleanVar(value=bool(show_1d or show_caked_2d))

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
    view_state.check_1d = None
    view_state.show_caked_2d_var = show_caked_2d_var
    view_state.check_2d = check_2d
    view_state.log_radial_var = log_radial_var
    view_state.check_log_radial = check_log_radial
    view_state.log_azimuth_var = log_azimuth_var
    view_state.check_log_azimuth = check_log_azimuth


def create_integration_range_controls(
    *,
    parent: tk.Misc,
    view_state: IntegrationRangeControlsViewState,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
    on_tth_min_changed: Callable[[object], None],
    on_tth_max_changed: Callable[[object], None],
    on_phi_min_changed: Callable[[object], None],
    on_phi_max_changed: Callable[[object], None],
    on_apply_entry: Callable[[object, object, object], None],
) -> None:
    """Create the 1D integration-range controls and store their widget refs."""

    frame = CollapsibleFrame(parent, text="Integration Ranges", expanded=True)
    frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    range_frame = frame.frame

    def _create_range_row(
        *,
        prefix: str,
        label_text: str,
        initial_value: float,
        lower_bound: float,
        upper_bound: float,
        slider_command: Callable[[object], None],
    ) -> None:
        container = ttk.Frame(range_frame)
        container.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(container, text=label_text).pack(side=tk.LEFT, padx=5)

        value_var = tk.DoubleVar(value=float(initial_value))
        label_var = tk.StringVar(value=f"{value_var.get():.1f}")
        entry_var = tk.StringVar(value=f"{value_var.get():.4f}")
        setattr(view_state, f"{prefix}_container", container)
        setattr(view_state, f"{prefix}_var", value_var)
        setattr(view_state, f"{prefix}_label_var", label_var)
        setattr(view_state, f"{prefix}_entry_var", entry_var)

        slider = ttk.Scale(
            container,
            from_=float(lower_bound),
            to=float(upper_bound),
            orient=tk.HORIZONTAL,
            variable=value_var,
            command=slider_command,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        slider.bind(
            "<ButtonRelease-1>",
            lambda _event, value_var=value_var, slider_command=slider_command: slider_command(
                value_var.get()
            ),
        )

        label = ttk.Label(container, textvariable=label_var, width=6)
        label.pack(side=tk.LEFT, padx=4)

        entry = ttk.Entry(container, textvariable=entry_var, width=8)
        entry.pack(side=tk.LEFT, padx=(0, 5))
        entry.bind(
            "<Return>",
            lambda _event, entry_var=entry_var, value_var=value_var, slider=slider: on_apply_entry(
                entry_var,
                value_var,
                slider,
            ),
        )
        entry.bind(
            "<FocusOut>",
            lambda _event, entry_var=entry_var, value_var=value_var, slider=slider: on_apply_entry(
                entry_var,
                value_var,
                slider,
            ),
        )

        setattr(view_state, f"{prefix}_slider", slider)
        setattr(view_state, f"{prefix}_label", label)
        setattr(view_state, f"{prefix}_entry", entry)

    _create_range_row(
        prefix="tth_min",
        label_text="2θ Min (°):",
        initial_value=tth_min,
        lower_bound=0.0,
        upper_bound=90.0,
        slider_command=on_tth_min_changed,
    )
    _create_range_row(
        prefix="tth_max",
        label_text="2θ Max (°):",
        initial_value=tth_max,
        lower_bound=0.0,
        upper_bound=90.0,
        slider_command=on_tth_max_changed,
    )
    _create_range_row(
        prefix="phi_min",
        label_text="φ Min (°):",
        initial_value=phi_min,
        lower_bound=-90.0,
        upper_bound=90.0,
        slider_command=on_phi_min_changed,
    )
    _create_range_row(
        prefix="phi_max",
        label_text="φ Max (°):",
        initial_value=phi_max,
        lower_bound=-90.0,
        upper_bound=90.0,
        slider_command=on_phi_max_changed,
    )

    view_state.frame = frame
    view_state.range_frame = range_frame


def create_analysis_export_controls(
    *,
    parent: tk.Misc,
    view_state: AnalysisExportControlsViewState,
    on_save_snapshot: Callable[[], None],
    on_save_q_space: Callable[[], None],
    on_save_1d_grid: Callable[[], None],
    save_1d_grid_available: bool = True,
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
        text=(
            "Save 1D Grid"
            if save_1d_grid_available
            else "Save 1D Grid (Unavailable)"
        ),
        command=on_save_1d_grid,
        state=(tk.NORMAL if save_1d_grid_available else tk.DISABLED),
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

    if on_mousewheel is None:
        _register_pointer_mousewheel_handler(
            root,
            key=("geometry-fit-constraints-scroll", id(view_state)),
            handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
                canvas,
                pointer_x=pointer_x,
                pointer_y=pointer_y,
                event=event,
            ),
        )
    else:
        _register_pointer_mousewheel_handler(
            root,
            key=("geometry-fit-constraints-scroll", id(view_state)),
            handler=lambda *, pointer_x, pointer_y, event: on_mousewheel(event),
        )


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

    return _scroll_canvas_if_pointer_inside(
        canvas,
        pointer_x=pointer_x,
        pointer_y=pointer_y,
        event=event,
    )


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
    _register_pointer_mousewheel_handler(
        root,
        key=("geometry-q-group-scroll", id(view_state)),
        handler=lambda *, pointer_x, pointer_y, event: _scroll_canvas_if_pointer_inside(
            canvas,
            pointer_x=pointer_x,
            pointer_y=pointer_y,
            event=event,
        ),
    )
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
