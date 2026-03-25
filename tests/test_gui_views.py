import tkinter as tk

from ra_sim.gui import state, views


class _FakeExistingChild:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeBody:
    def __init__(self) -> None:
        self.children = [_FakeExistingChild()]

    def winfo_children(self):
        return list(self.children)


class _FakeCanvas:
    def __init__(self) -> None:
        self.updated = False
        self.scrollregion = None
        self.y_moved_to = None

    def yview(self):
        return (0.25, 1.0)

    def update_idletasks(self) -> None:
        self.updated = True

    def configure(self, **kwargs) -> None:
        self.scrollregion = kwargs.get("scrollregion")

    def bbox(self, _value):
        return (0, 0, 100, 200)

    def yview_moveto(self, value: float) -> None:
        self.y_moved_to = value


class _FakeListbox:
    def __init__(self) -> None:
        self.items = []
        self.selected = []
        self.seen_index = None

    def delete(self, _start, _end) -> None:
        self.items = []
        self.selected = []

    def insert(self, _index, line) -> None:
        self.items.append(line)

    def selection_set(self, idx: int) -> None:
        self.selected.append(int(idx))

    def see(self, idx: int) -> None:
        self.seen_index = int(idx)


class _FakeWindow:
    def __init__(self, exists: bool = True) -> None:
        self.exists = exists
        self.destroyed = False
        self.title_text = None
        self.geometry_text = None
        self.protocols = {}
        self.lifted = False
        self.focused = False

    def winfo_exists(self) -> bool:
        if self.exists == "error":
            raise tk.TclError("gone")
        return bool(self.exists)

    def destroy(self) -> None:
        self.destroyed = True

    def title(self, text: str) -> None:
        self.title_text = text

    def geometry(self, text: str) -> None:
        self.geometry_text = text

    def protocol(self, name: str, callback) -> None:
        self.protocols[name] = callback

    def lift(self) -> None:
        self.lifted = True

    def focus_force(self) -> None:
        self.focused = True


class _FakeFrame:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.rowconfigure_calls = []
        self.columnconfigure_calls = []

    def pack(self, **_kwargs) -> None:
        pass

    def rowconfigure(self, index: int, weight: int) -> None:
        self.rowconfigure_calls.append((index, weight))

    def columnconfigure(self, index: int, weight: int) -> None:
        self.columnconfigure_calls.append((index, weight))


class _FakeScrollbar:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.set_calls = []
        _FakeScrollbar.created.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def set(self, *args) -> None:
        self.set_calls.append(args)


class _FakeText:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.content = ""
        self.state = None
        self.yscrollcommand = None
        self.xscrollcommand = None
        _FakeText.created.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def configure(self, **kwargs) -> None:
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "yscrollcommand" in kwargs:
            self.yscrollcommand = kwargs["yscrollcommand"]
        if "xscrollcommand" in kwargs:
            self.xscrollcommand = kwargs["xscrollcommand"]

    def delete(self, _start, _end) -> None:
        self.content = ""

    def insert(self, _index, text: str) -> None:
        self.content = str(text)

    def yview(self, *_args) -> None:
        pass

    def xview(self, *_args) -> None:
        pass


class _FakeLabel:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.text = kwargs.get("text", "")
        _FakeLabel.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass

    def config(self, **kwargs) -> None:
        self.text = kwargs.get("text", self.text)


class _FakeCheckbutton:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.variable = kwargs.get("variable")
        _FakeCheckbutton.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass


class _FakeVar:
    def __init__(self, value: bool) -> None:
        self._value = value

    def get(self) -> bool:
        return self._value


def test_geometry_q_group_window_state_helpers_close_and_report_open() -> None:
    view_state = state.GeometryQGroupViewState()

    assert views.geometry_q_group_window_open(view_state) is False

    window = _FakeWindow()
    view_state.window = window
    view_state.canvas = object()
    view_state.body = object()
    view_state.status_label = object()

    assert views.geometry_q_group_window_open(view_state) is True

    views.close_geometry_q_group_window(view_state)
    assert window.destroyed is True
    assert view_state.window is None
    assert view_state.canvas is None
    assert view_state.body is None
    assert view_state.status_label is None

    view_state.window = _FakeWindow(exists="error")
    assert views.geometry_q_group_window_open(view_state) is False


def test_refresh_geometry_q_group_window_updates_status_and_rows(monkeypatch) -> None:
    _FakeLabel.created = []
    _FakeCheckbutton.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)

    view_state = state.GeometryQGroupViewState(
        window=_FakeWindow(),
        canvas=_FakeCanvas(),
        body=_FakeBody(),
        status_label=_FakeLabel(None, text=""),
    )
    registered = []
    toggled = []
    cleared = []

    refreshed = views.refresh_geometry_q_group_window(
        view_state=view_state,
        entries=[
            {"key": ("q_group", "primary", 1, 0), "peak_count": 2},
            {"key": ("q_group", "secondary", 2, 1), "peak_count": 3},
        ],
        excluded_q_groups=[("q_group", "secondary", 2, 1)],
        status_text="status text",
        format_line=lambda entry: f"line:{entry['key']}",
        on_toggle=lambda key, row_var: toggled.append((key, row_var.get())),
        clear_row_vars=lambda: cleared.append(True),
        register_row_var=lambda key, row_var: registered.append((key, row_var.get())),
        boolean_var_factory=_FakeVar,
    )

    assert refreshed is True
    assert cleared == [True]
    assert registered == [
        (("q_group", "primary", 1, 0), True),
        (("q_group", "secondary", 2, 1), False),
    ]
    assert view_state.status_label.text == "status text"
    assert view_state.canvas.updated is True
    assert view_state.canvas.scrollregion == (0, 0, 100, 200)
    assert view_state.canvas.y_moved_to == 0.25
    assert view_state.body.children[0].destroyed is True
    assert _FakeCheckbutton.created[0].kwargs["text"] == "line:('q_group', 'primary', 1, 0)"

    _FakeCheckbutton.created[1].command()
    assert toggled == [(("q_group", "secondary", 2, 1), False)]


def test_bragg_qr_manager_view_helpers_close_and_report_open() -> None:
    view_state = state.BraggQrManagerViewState()

    assert views.bragg_qr_manager_window_open(view_state) is False

    window = _FakeWindow()
    view_state.window = window
    view_state.qr_listbox = object()
    view_state.qr_status_label = object()
    view_state.l_listbox = object()
    view_state.l_status_label = object()

    assert views.bragg_qr_manager_window_open(view_state) is True

    views.close_bragg_qr_manager_window(view_state)
    assert window.destroyed is True
    assert view_state.window is None
    assert view_state.qr_listbox is None
    assert view_state.qr_status_label is None
    assert view_state.l_listbox is None
    assert view_state.l_status_label is None

    view_state.window = _FakeWindow(exists="error")
    assert views.bragg_qr_manager_window_open(view_state) is False


def test_refresh_bragg_qr_manager_lists_update_status_and_selection() -> None:
    view_state = state.BraggQrManagerViewState(
        window=_FakeWindow(),
        qr_listbox=_FakeListbox(),
        qr_status_label=_FakeLabel(None, text=""),
        l_listbox=_FakeListbox(),
        l_status_label=_FakeLabel(None, text=""),
    )

    assert (
        views.refresh_bragg_qr_manager_qr_list(
            view_state=view_state,
            lines=["qr-a", "qr-b"],
            selected_indices=[1],
            status_text="qr-status",
            see_index=1,
        )
        is True
    )
    assert view_state.qr_listbox.items == ["qr-a", "qr-b"]
    assert view_state.qr_listbox.selected == [1]
    assert view_state.qr_listbox.seen_index == 1
    assert view_state.qr_status_label.text == "qr-status"

    assert (
        views.refresh_bragg_qr_manager_l_list(
            view_state=view_state,
            lines=["l-a", "l-b", "l-c"],
            selected_indices=[0, 2],
            status_text="l-status",
        )
        is True
    )
    assert view_state.l_listbox.items == ["l-a", "l-b", "l-c"]
    assert view_state.l_listbox.selected == [0, 2]
    assert view_state.l_status_label.text == "l-status"


def test_hbn_geometry_debug_view_helpers_close_and_report_open() -> None:
    view_state = state.HbnGeometryDebugViewState()

    assert views.hbn_geometry_debug_window_open(view_state) is False

    window = _FakeWindow()
    view_state.window = window
    view_state.text_widget = object()

    assert views.hbn_geometry_debug_window_open(view_state) is True

    views.close_hbn_geometry_debug_window(view_state)
    assert window.destroyed is True
    assert view_state.window is None
    assert view_state.text_widget is None

    view_state.window = _FakeWindow(exists="error")
    assert views.hbn_geometry_debug_window_open(view_state) is False


def test_open_hbn_geometry_debug_window_populates_and_reuses_existing_window(
    monkeypatch,
) -> None:
    _FakeText.created = []
    _FakeScrollbar.created = []
    created_windows = []

    monkeypatch.setattr(
        views.tk,
        "Toplevel",
        lambda _root: created_windows.append(_FakeWindow()) or created_windows[-1],
    )
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.tk, "Text", _FakeText)

    closed = []
    view_state = state.HbnGeometryDebugViewState()

    opened = views.open_hbn_geometry_debug_window(
        root=object(),
        view_state=view_state,
        text="first report",
        on_close=lambda: closed.append(True),
    )

    assert opened is True
    assert len(created_windows) == 1
    assert created_windows[0].title_text == "hBN Geometry Debug"
    assert created_windows[0].geometry_text == "980x560"
    assert created_windows[0].protocols["WM_DELETE_WINDOW"] is not None
    assert view_state.text_widget is _FakeText.created[0]
    assert view_state.text_widget.content == "first report"
    assert view_state.text_widget.state == tk.DISABLED

    reopened = views.open_hbn_geometry_debug_window(
        root=object(),
        view_state=view_state,
        text="updated report",
        on_close=lambda: closed.append(False),
    )

    assert reopened is False
    assert len(created_windows) == 1
    assert created_windows[0].lifted is True
    assert created_windows[0].focused is True
    assert view_state.text_widget.content == "updated report"

    created_windows[0].protocols["WM_DELETE_WINDOW"]()
    assert closed == [True]
