"""GUI view helpers used by RA-SIM Tk applications."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
import tkinter as tk
from tkinter import ttk

from .collapsible import CollapsibleFrame
from .state import (
    BraggQrManagerViewState,
    GeometryFitConstraintsViewState,
    GeometryQGroupViewState,
    HbnGeometryDebugViewState,
)


_GEOMETRY_Q_GROUP_EMPTY_TEXT = (
    "No Qr/Qz groups are listed yet. "
    'Press "Update Listed Peaks" to snapshot the current simulation.'
)
_GEOMETRY_FIT_CONSTRAINTS_HELP_TEXT = (
    "Each window is applied as current value +/- deviation during geometry fitting. "
    "Stay-close adds a soft pull back to the starting guess."
)


def create_root_window(title: str = "RA Simulation") -> tk.Tk:
    """Create and return a Tk root window with the provided title."""

    root = tk.Tk()
    root.title(title)
    return root


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
