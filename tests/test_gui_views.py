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
    created = []

    def __init__(self, parent=None, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.updated = False
        self.scrollregion = None
        self.y_moved_to = None
        self.yscrollcommand = None
        self.created_window = None
        self.bindings = {}
        self.itemconfigure_calls = []
        self.scrolled = []
        self.rootx = 10
        self.rooty = 20
        self.width = 300
        self.height = 200
        _FakeCanvas.created.append(self)

    def yview(self):
        return (0.25, 1.0)

    def update_idletasks(self) -> None:
        self.updated = True

    def configure(self, **kwargs) -> None:
        self.scrollregion = kwargs.get("scrollregion")
        self.yscrollcommand = kwargs.get("yscrollcommand", self.yscrollcommand)

    def bbox(self, _value):
        return (0, 0, 100, 200)

    def yview_moveto(self, value: float) -> None:
        self.y_moved_to = value

    def pack(self, **_kwargs) -> None:
        pass

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback

    def create_window(self, coords, *, window, anchor):
        self.created_window = {
            "coords": coords,
            "window": window,
            "anchor": anchor,
        }
        return "body-window"

    def itemconfigure(self, item, **kwargs) -> None:
        self.itemconfigure_calls.append((item, kwargs))

    def yview_scroll(self, delta: int, units: str) -> None:
        self.scrolled.append((delta, units))

    def winfo_rootx(self) -> int:
        return self.rootx

    def winfo_rooty(self) -> int:
        return self.rooty

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height


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
        self.minsize_args = None
        self.transient_parent = None
        self.protocols = {}
        self.bindings = {}
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

    def minsize(self, width: int, height: int) -> None:
        self.minsize_args = (width, height)

    def transient(self, parent) -> None:
        self.transient_parent = parent

    def protocol(self, name: str, callback) -> None:
        self.protocols[name] = callback

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback

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
        self.bindings = {}

    def pack(self, **_kwargs) -> None:
        pass

    def grid(self, **_kwargs) -> None:
        pass

    def rowconfigure(self, index: int, weight: int) -> None:
        self.rowconfigure_calls.append((index, weight))

    def columnconfigure(self, index: int, weight: int) -> None:
        self.columnconfigure_calls.append((index, weight))

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback


class _FakePanedwindow:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.added = []

    def pack(self, **_kwargs) -> None:
        pass

    def add(self, child, *, weight: int) -> None:
        self.added.append((child, weight))


class _FakeNotebook:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.tabs = []
        self.bindings = {}
        self.selected_tab = None

    def pack(self, **_kwargs) -> None:
        pass

    def add(self, child, *, text: str) -> None:
        self.tabs.append((child, text))
        if self.selected_tab is None:
            self.selected_tab = child

    def bind(self, event: str, callback, add=None) -> None:
        self.bindings[event] = (callback, add)

    def select(self, value=None):
        if value is None:
            return self.selected_tab
        self.selected_tab = value
        return self.selected_tab


class _FakeScrollbar:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.set_calls = []
        _FakeScrollbar.created.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def pack(self, **_kwargs) -> None:
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


class _FakeRoot:
    def __init__(self, screenheight: int = 1000) -> None:
        self.screenheight = screenheight
        self.bind_all_calls = []

    def winfo_screenheight(self) -> int:
        return self.screenheight

    def bind_all(self, sequence: str, callback, add=None) -> None:
        self.bind_all_calls.append((sequence, callback, add))


def test_create_root_window_sets_title_and_configures_section_styles(monkeypatch) -> None:
    class _FakeTkRoot:
        def __init__(self) -> None:
            self.title_text = None
            self.withdrawn = False

        def title(self, text: str) -> None:
            self.title_text = text

        def withdraw(self) -> None:
            self.withdrawn = True

    class _FakeFont:
        def __init__(self) -> None:
            self.configured = {}

        def copy(self):
            return self

        def configure(self, **kwargs) -> None:
            self.configured.update(kwargs)

    class _FakeStyle:
        def __init__(self, root) -> None:
            self.root = root
            self.configure_calls = []

        def configure(self, name: str, **kwargs) -> None:
            self.configure_calls.append((name, kwargs))

    fake_root = _FakeTkRoot()
    fake_style = _FakeStyle(fake_root)
    fake_font = _FakeFont()
    launch_context = object()
    affinity_calls = []

    monkeypatch.setattr(views.tk, "Tk", lambda: fake_root)
    monkeypatch.setattr(views.ttk, "Style", lambda root: fake_style)
    monkeypatch.setattr(views.tkfont, "nametofont", lambda _name: fake_font)
    monkeypatch.setattr(
        views.window_affinity,
        "capture_launch_window_context",
        lambda: launch_context,
    )
    monkeypatch.setattr(
        views.window_affinity,
        "apply_window_launch_context",
        lambda window, *, context=None, width=None, height=None: affinity_calls.append(
            (window, context, width, height)
        )
        or True,
    )

    root = views.create_root_window("Readable Sections")

    assert root is fake_root
    assert fake_root.withdrawn is True
    assert fake_root.title_text == "Readable Sections"
    assert getattr(fake_root, "_ra_sim_launch_window_context") is launch_context
    assert fake_font.configured == {"weight": "bold"}
    assert (
        "TLabelframe",
        {"borderwidth": 2, "relief": "groove", "padding": (10, 8)},
    ) in fake_style.configure_calls
    assert any(name == "TLabelframe.Label" for name, _kwargs in fake_style.configure_calls)
    assert any(name == "SectionHeader.Toolbutton" for name, _kwargs in fake_style.configure_calls)
    assert ("TNotebook.Tab", {"padding": (12, 8)}) in fake_style.configure_calls
    assert affinity_calls == [(fake_root, launch_context, None, None)]


class _FakeThemeWidget:
    def __init__(self, class_name: str, **options) -> None:
        self._class_name = class_name
        self._options = dict(options)
        self._children = []

    def add_child(self, child) -> None:
        self._children.append(child)

    def winfo_class(self) -> str:
        return self._class_name

    def winfo_children(self):
        return list(self._children)

    def winfo_exists(self) -> bool:
        return True

    def cget(self, option: str):
        if option not in self._options:
            raise tk.TclError(option)
        return self._options[option]

    def configure(self, **kwargs) -> None:
        self._options.update(kwargs)

    def config(self, **kwargs) -> None:
        self.configure(**kwargs)


def test_bind_fit2d_theme_restyles_and_restores_widgets(monkeypatch) -> None:
    class _FakeStyle:
        def __init__(self, _root) -> None:
            self.configure_calls = []
            self.map_calls = []
            self.theme = "vista"

        def configure(self, name: str, **kwargs) -> None:
            self.configure_calls.append((name, kwargs))

        def map(self, name: str, **kwargs) -> None:
            self.map_calls.append((name, kwargs))

        def theme_names(self):
            return ("vista", "clam")

        def theme_use(self, name: str | None = None):
            if name is None:
                return self.theme
            self.theme = name
            return self.theme

    style = _FakeStyle(None)
    root = _FakeThemeWidget("Tk", background="#f0f0f0")
    root._ra_default_root_background = "#f0f0f0"
    canvas = _FakeThemeWidget(
        "Canvas",
        background="#ffffff",
        highlightbackground="#444444",
        highlightcolor="#444444",
    )
    text_widget = _FakeThemeWidget(
        "Text",
        background="#ffffff",
        foreground="#000000",
        insertbackground="#000000",
        selectbackground="#224488",
        selectforeground="#ffffff",
        highlightbackground="#444444",
        highlightcolor="#444444",
    )
    root.add_child(canvas)
    root.add_child(text_widget)

    popup = _FakeThemeWidget("Toplevel", background="#f0f0f0")
    listbox = _FakeThemeWidget(
        "Listbox",
        background="#ffffff",
        foreground="#000000",
        selectbackground="#224488",
        selectforeground="#ffffff",
        highlightbackground="#444444",
        highlightcolor="#444444",
    )
    popup.add_child(listbox)

    fit2d_var = _FakeStringVar(False)

    monkeypatch.setattr(views.ttk, "Style", lambda _root: style)

    views._bind_fit2d_theme(root, fit2d_var)
    views._sync_fit2d_theme_scope(root, popup)
    fit2d_var.set(True)

    assert getattr(root, "_ra_fit2d_theme_enabled") is True
    assert style.theme == "clam"
    assert any(name == "." for name, _kwargs in style.configure_calls)
    assert root._options["background"] == views._FIT2D_THEME_PALETTE["root_bg"]
    assert canvas._options["background"] == views._FIT2D_THEME_PALETTE["panel_bg"]
    assert text_widget._options["selectbackground"] == views._FIT2D_THEME_PALETTE["accent_alt"]
    assert listbox._options["background"] == views._FIT2D_THEME_PALETTE["field_bg"]

    fit2d_var.set(False)

    assert getattr(root, "_ra_fit2d_theme_enabled") is False
    assert style.theme == "vista"
    assert root._options["background"] == "#f0f0f0"
    assert canvas._options["background"] == "#ffffff"
    assert text_widget._options["foreground"] == "#000000"
    assert listbox._options["background"] == "#ffffff"


class _FakeCollapsibleFrame:
    def __init__(self, parent, text: str = "", expanded: bool = False) -> None:
        self.parent = parent
        self.text = text
        self.expanded = expanded
        self.summary_text = ""
        self.frame = _FakeFrame(self)
        self.pack_calls = []

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)

    def set_header_summary(self, text: str = "") -> None:
        self.summary_text = str(text)


class _FakeLabel:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.text = kwargs.get("text", "")
        _FakeLabel.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass

    def grid(self, **_kwargs) -> None:
        pass

    def config(self, **kwargs) -> None:
        self.text = kwargs.get("text", self.text)

    configure = config

    def cget(self, key: str):
        if key == "text":
            return self.text
        return self.kwargs.get(key)


class _FakeProgressbar:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs

    def pack(self, **_kwargs) -> None:
        pass


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

    def grid(self, **_kwargs) -> None:
        pass


class _FakeVar:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _FakeStringVar:
    def __init__(self, value="") -> None:
        self._value = value
        self._trace_callbacks = []

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value

        for callback in list(self._trace_callbacks):
            callback()

    def trace_add(self, _mode: str, callback) -> None:
        self._trace_callbacks.append(callback)


class _FakeEntry:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.textvariable = kwargs.get("textvariable")
        self.bindings = {}
        self.state = kwargs.get("state")
        self.value = ""

    def pack(self, **_kwargs) -> None:
        pass

    def grid(self, **_kwargs) -> None:
        pass

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback

    def unbind(self, event: str) -> None:
        self.bindings.pop(event, None)

    def configure(self, **kwargs) -> None:
        self.state = kwargs.get("state", self.state)

    def config(self, **kwargs) -> None:
        self.configure(**kwargs)

    def get(self) -> str:
        if self.textvariable is not None:
            return str(self.textvariable.get())
        return str(self.value)

    def delete(self, _start, _end=None) -> None:
        self.value = ""
        if self.textvariable is not None:
            self.textvariable.set("")

    def insert(self, _index, value) -> None:
        self.value = str(value)
        if self.textvariable is not None:
            self.textvariable.set(self.value)


class _FakeButton:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.state = kwargs.get("state")
        _FakeButton.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass

    def grid(self, **_kwargs) -> None:
        pass

    def config(self, **kwargs) -> None:
        self.state = kwargs.get("state", self.state)

    def configure(self, **kwargs) -> None:
        self.config(**kwargs)


class _FakeRadiobutton:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.variable = kwargs.get("variable")
        self.value = kwargs.get("value")
        _FakeRadiobutton.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass


class _FakeScale:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.variable = kwargs.get("variable")
        self.state = kwargs.get("state")
        self.bounds = {
            "from": kwargs.get("from_"),
            "to": kwargs.get("to"),
        }
        self.bindings = {}
        self.pack_calls = []
        _FakeScale.created.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)

    def bind(self, event: str, callback, add=None) -> None:
        self.bindings[event] = (callback, add)

    def configure(self, **kwargs) -> None:
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "from_" in kwargs:
            self.bounds["from"] = kwargs["from_"]
        if "to" in kwargs:
            self.bounds["to"] = kwargs["to"]

    def config(self, **kwargs) -> None:
        self.configure(**kwargs)

    def cget(self, key: str):
        return self.bounds[key]


class _FakeSpinbox:
    created = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        _FakeSpinbox.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass


class _FakeOptionMenu:
    created = []

    def __init__(self, parent, variable, default, *values) -> None:
        self.parent = parent
        self.variable = variable
        self.default = default
        self.values = values
        _FakeOptionMenu.created.append(self)

    def pack(self, **_kwargs) -> None:
        pass


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


def test_analysis_popout_view_helpers_close_and_report_open(monkeypatch) -> None:
    view_state = state.AnalysisPopoutViewState()

    assert views.analysis_popout_window_open(view_state) is False

    window = _FakeWindow()
    view_state.window = window
    view_state.exports_frame = object()
    view_state.peak_tools_frame = object()
    view_state.plot_frame = object()
    view_state.dock_button = object()

    assert views.analysis_popout_window_open(view_state) is True

    views.close_analysis_popout_window(view_state)
    assert window.destroyed is True
    assert view_state.window is None
    assert view_state.exports_frame is None
    assert view_state.peak_tools_frame is None
    assert view_state.plot_frame is None
    assert view_state.dock_button is None

    view_state.window = _FakeWindow(exists="error")
    assert views.analysis_popout_window_open(view_state) is False


def test_open_analysis_popout_window_populates_and_reuses_existing_window(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    created_windows = []

    monkeypatch.setattr(
        views.tk,
        "Toplevel",
        lambda _root: created_windows.append(_FakeWindow()) or created_windows[-1],
    )
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)

    closed = []
    root = object()
    view_state = state.AnalysisPopoutViewState()

    opened = views.open_analysis_popout_window(
        root=root,
        view_state=view_state,
        on_close=lambda: closed.append(True),
    )

    assert opened is True
    assert len(created_windows) == 1
    assert created_windows[0].title_text == "Analyze"
    assert created_windows[0].geometry_text == "900x820"
    assert created_windows[0].minsize_args == (620, 440)
    assert created_windows[0].transient_parent is root
    assert created_windows[0].protocols["WM_DELETE_WINDOW"] is not None
    assert "<Escape>" in created_windows[0].bindings
    assert isinstance(view_state.exports_frame, _FakeFrame)
    assert isinstance(view_state.peak_tools_frame, _FakeFrame)
    assert isinstance(view_state.plot_frame, _FakeFrame)
    assert view_state.dock_button is _FakeButton.created[0]
    assert view_state.dock_button.kwargs["text"] == "Dock Back"

    reopened = views.open_analysis_popout_window(
        root=root,
        view_state=view_state,
        on_close=lambda: closed.append(False),
    )

    assert reopened is False
    assert len(created_windows) == 1
    assert created_windows[0].lifted is True
    assert created_windows[0].focused is True

    created_windows[0].protocols["WM_DELETE_WINDOW"]()
    assert closed == [True]


def test_create_app_shell_stores_shared_shell_refs_and_notebook_state(
    monkeypatch,
) -> None:
    _FakeScrollbar.created = []
    _FakeCanvas.created = []
    monkeypatch.setattr(views.ttk, "Panedwindow", _FakePanedwindow)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Notebook", _FakeNotebook)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.ttk, "Separator", _FakeFrame)
    monkeypatch.setattr(views.tk, "Canvas", _FakeCanvas)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.AppShellViewState()
    views.create_app_shell(root=object(), view_state=view_state)

    assert isinstance(view_state.main_pane, _FakePanedwindow)
    assert view_state.main_pane.added == [
        (view_state.controls_panel, 1),
        (view_state.figure_panel, 3),
    ]
    assert view_state.session_summary_frame is None
    assert view_state.session_summary_var is None
    assert view_state.workflow_checklist_frame is None
    assert view_state.workflow_checklist_status_vars == {}
    assert view_state.mode_banner_frame is None
    assert view_state.mode_banner_title_var is None
    assert isinstance(view_state.view_switcher_frame, _FakeFrame)
    assert view_state.view_mode_var.get() == "detector"
    assert isinstance(view_state.canvas_context_frame, _FakeFrame)
    assert isinstance(view_state.canvas_context_left, _FakeFrame)
    assert isinstance(view_state.canvas_context_right, _FakeFrame)
    assert isinstance(view_state.dataset_summary_frame, _FakeFrame)
    assert isinstance(view_state.fit_health_frame, _FakeFrame)
    assert view_state.dataset_summary_frame.parent is view_state.canvas_context_left
    assert view_state.fit_health_frame.parent is view_state.canvas_context_right
    assert set(view_state.dataset_value_labels) == {
        "background",
        "theta_i",
        "theta",
        "fit",
        "model",
    }
    assert isinstance(view_state.controls_notebook, _FakeNotebook)
    assert [text for _, text in view_state.controls_notebook.tabs] == [
        "Setup",
        "Match",
        "Refine",
        "Analyze",
        "Help",
    ]
    assert [text for _, text in view_state.parameter_notebook.tabs] == [
        "Setup",
        "Sample Structure",
    ]
    assert isinstance(view_state.setup_body, _FakeFrame)
    assert isinstance(view_state.match_body, _FakeFrame)
    assert isinstance(view_state.refine_basic_body, _FakeFrame)
    assert isinstance(view_state.refine_advanced_body, _FakeFrame)
    assert isinstance(view_state.workspace_body, _FakeFrame)
    assert isinstance(view_state.fit_body, _FakeFrame)
    assert isinstance(view_state.parameter_geometry_body, _FakeFrame)
    assert isinstance(view_state.parameter_structure_body, _FakeFrame)
    assert isinstance(view_state.setup_canvas, _FakeCanvas)
    assert isinstance(view_state.match_canvas, _FakeCanvas)
    assert isinstance(view_state.refine_basic_canvas, _FakeCanvas)
    assert isinstance(view_state.refine_advanced_canvas, _FakeCanvas)
    assert isinstance(view_state.fit_actions_frame, _FakeFrame)
    assert isinstance(view_state.match_backgrounds_frame, _FakeFrame)
    assert isinstance(view_state.match_parameter_frame, _FakeFrame)
    assert isinstance(view_state.match_run_frame, _FakeFrame)
    assert isinstance(view_state.match_results_frame, _FakeFrame)
    assert "Fit results will appear here" in view_state.match_results_var.get()
    assert isinstance(view_state.analysis_views_frame, _FakeFrame)
    assert isinstance(view_state.analysis_exports_frame, _FakeFrame)
    assert isinstance(view_state.analysis_peak_tools_frame, _FakeFrame)
    assert isinstance(view_state.analysis_popout_button, _FakeButton)
    assert view_state.analysis_popout_button.kwargs["text"] == "Pop Out Analyze Window"
    assert isinstance(view_state.status_frame, _FakeFrame)
    assert isinstance(view_state.fig_frame, _FakeFrame)
    assert isinstance(view_state.figure_workspace_frame, _FakeFrame)
    assert isinstance(view_state.canvas_frame, _FakeFrame)
    assert isinstance(view_state.figure_controls_frame, _FakeFrame)
    assert isinstance(view_state.quick_controls_frame, _FakeFrame)
    assert isinstance(view_state.quick_controls_body, _FakeFrame)
    assert view_state.figure_workspace_frame.parent is view_state.fig_frame
    assert view_state.canvas_frame.parent is view_state.figure_workspace_frame
    assert view_state.figure_controls_frame.parent is view_state.fig_frame
    assert view_state.quick_controls_frame.parent is view_state.figure_workspace_frame
    assert isinstance(view_state.left_col, _FakeFrame)
    assert isinstance(view_state.right_col, _FakeFrame)
    assert isinstance(view_state.plot_frame_1d, _FakeFrame)
    assert view_state.control_tab_var.get() == "setup"
    assert view_state.parameter_tab_var.get() == "basic"
    assert view_state.controls_notebook.selected_tab is view_state.setup_tab
    assert view_state.parameter_notebook.selected_tab is view_state.refine_basic_tab

    view_state.control_tab_var.set("match")
    assert view_state.controls_notebook.selected_tab is view_state.match_tab

    callback, add = view_state.parameter_notebook.bindings["<<NotebookTabChanged>>"]
    assert add == "+"
    view_state.parameter_notebook.select(view_state.refine_advanced_tab)
    callback(None)
    assert view_state.parameter_tab_var.get() == "advanced"


def test_create_app_shell_binds_pointer_wheel_scrolling_when_root_supports_bind_all(
    monkeypatch,
) -> None:
    _FakeScrollbar.created = []
    _FakeCanvas.created = []
    monkeypatch.setattr(views.ttk, "Panedwindow", _FakePanedwindow)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Notebook", _FakeNotebook)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.ttk, "Separator", _FakeFrame)
    monkeypatch.setattr(views.tk, "Canvas", _FakeCanvas)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    root = _FakeRoot()
    view_state = state.AppShellViewState()
    views.create_app_shell(root=root, view_state=view_state)

    assert [call[0] for call in root.bind_all_calls] == [
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
    ]

    view_state.setup_canvas.rootx = 0
    view_state.setup_canvas.rooty = 0
    view_state.setup_canvas.width = 100
    view_state.setup_canvas.height = 100
    view_state.match_canvas.rootx = 150
    view_state.match_canvas.rooty = 0
    view_state.match_canvas.width = 100
    view_state.match_canvas.height = 100
    view_state.refine_basic_canvas.rootx = 300
    view_state.refine_basic_canvas.rooty = 0
    view_state.refine_basic_canvas.width = 100
    view_state.refine_basic_canvas.height = 100
    view_state.refine_advanced_canvas.rootx = 450
    view_state.refine_advanced_canvas.rooty = 0
    view_state.refine_advanced_canvas.width = 100
    view_state.refine_advanced_canvas.height = 100

    dispatch = next(
        callback
        for sequence, callback, _add in root.bind_all_calls
        if sequence == "<MouseWheel>"
    )
    event = type("Event", (), {"delta": 60, "num": None, "x_root": 175, "y_root": 25})()

    assert dispatch(event) == "break"
    assert view_state.setup_canvas.scrolled == []
    assert view_state.match_canvas.scrolled == [(-1, "units")]
    assert view_state.refine_basic_canvas.scrolled == []
    assert view_state.refine_advanced_canvas.scrolled == []


def test_create_app_shell_adds_fit2d_help_preference_when_var_is_supplied(
    monkeypatch,
) -> None:
    _FakeScrollbar.created = []
    _FakeCanvas.created = []
    _FakeCheckbutton.created = []
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "Panedwindow", _FakePanedwindow)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Notebook", _FakeNotebook)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.ttk, "Separator", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.tk, "Canvas", _FakeCanvas)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    fit2d_var = _FakeVar(False)
    view_state = state.AppShellViewState()

    views.create_app_shell(
        root=object(),
        view_state=view_state,
        fit2d_error_sound_var=fit2d_var,
    )

    assert isinstance(view_state.help_body, _FakeFrame)
    assert isinstance(view_state.help_preferences_frame, _FakeFrame)
    assert view_state.help_preferences_frame.kwargs["text"] == "Preferences"
    assert view_state.fit2d_error_sound_var is fit2d_var
    assert view_state.fit2d_error_sound_checkbutton is _FakeCheckbutton.created[0]
    assert view_state.fit2d_error_sound_checkbutton.variable is fit2d_var
    assert (
        view_state.fit2d_error_sound_checkbutton.kwargs["text"]
        == "Enable Fit2D mode (theme + error noise)"
    )
    assert any("Fit2D mode enables" in label.text for label in _FakeLabel.created)


def test_app_shell_summary_banner_and_match_result_setters_update_vars(monkeypatch) -> None:
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.AppShellViewState(
        session_summary_var=_FakeStringVar(""),
        mode_banner_title_var=_FakeStringVar(""),
        mode_banner_detail_var=_FakeStringVar(""),
        match_results_var=_FakeStringVar(""),
    )

    views.set_app_shell_session_summary_text(view_state, "Background: demo.osc")
    views.set_app_shell_mode_banner_text(
        view_state,
        title="Manual geometry picking armed",
        detail="Click a simulated Qr group on the image.",
    )
    views.set_match_results_text(view_state, "Chi-Squared: 1.23")

    assert view_state.session_summary_var.get() == "Background: demo.osc"
    assert view_state.mode_banner_title_var.get() == "Manual geometry picking armed"
    assert view_state.mode_banner_detail_var.get() == "Click a simulated Qr group on the image."
    assert view_state.match_results_var.get() == "Chi-Squared: 1.23"


def test_app_shell_context_helpers_update_status_labels(monkeypatch) -> None:
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    workflow_labels = {
        "backgrounds": _FakeLabel(object(), text=""),
        "geometry_fit": _FakeLabel(object(), text=""),
    }
    dataset_labels = {
        "background": _FakeLabel(object(), text=""),
        "theta_i": _FakeLabel(object(), text=""),
    }
    view_state = state.AppShellViewState(
        workflow_checklist_status_vars={
            "backgrounds": _FakeStringVar(""),
            "geometry_fit": _FakeStringVar(""),
        },
        workflow_checklist_status_labels=workflow_labels,
        dataset_value_labels=dataset_labels,
        fit_health_status_label=_FakeLabel(object(), text=""),
        fit_health_primary_label=_FakeLabel(object(), text=""),
        fit_health_secondary_label=_FakeLabel(object(), text=""),
        view_mode_var=_FakeStringVar("detector"),
    )

    views.set_app_shell_workflow_statuses(
        view_state,
        {"backgrounds": "ready", "geometry_fit": "not run"},
    )
    views.set_app_shell_dataset_values(
        view_state,
        {"background": "1/2 - bg.osc", "theta_i": "4.5000 deg"},
    )
    views.set_app_shell_fit_health_text(
        view_state,
        status="Stale",
        primary="Chi-Squared: 1.23",
        secondary="Peaks 10/12 | Gate 10/8",
    )
    views.set_app_shell_view_mode(view_state, "caked")

    assert view_state.workflow_checklist_status_vars["backgrounds"].get() == "Ready"
    assert view_state.workflow_checklist_status_vars["geometry_fit"].get() == "Not run"
    assert workflow_labels["backgrounds"].text == "Ready"
    assert workflow_labels["geometry_fit"].text == "Not run"
    assert dataset_labels["background"].text == "1/2 - bg.osc"
    assert dataset_labels["theta_i"].text == "4.5000 deg"
    assert view_state.fit_health_status_label.text == "Stale"
    assert view_state.fit_health_primary_label.text == "Chi-Squared: 1.23"
    assert view_state.fit_health_secondary_label.text == "Peaks 10/12 | Gate 10/8"
    assert view_state.view_mode_var.get() == "caked"


def test_bind_app_shell_view_mode_sync_tracks_analysis_view_changes() -> None:
    view_state = state.AppShellViewState(view_mode_var=_FakeStringVar("detector"))
    show_1d_var = _FakeStringVar(False)
    show_caked_2d_var = _FakeStringVar(False)

    def _resolve_mode() -> str:
        if show_caked_2d_var.get():
            return "caked"
        return "detector"

    views.bind_app_shell_view_mode_sync(
        view_state=view_state,
        show_1d_var=show_1d_var,
        show_caked_2d_var=show_caked_2d_var,
        resolve_mode=_resolve_mode,
    )

    assert view_state.view_mode_var.get() == "detector"

    show_caked_2d_var.set(True)
    assert view_state.view_mode_var.get() == "caked"

    show_caked_2d_var.set(False)
    assert view_state.view_mode_var.get() == "detector"

    show_1d_var.set(True)
    assert view_state.view_mode_var.get() == "detector"


def test_console_status_label_compacts_text_and_logs_once(monkeypatch) -> None:
    _FakeLabel.created = []
    printed = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )

    label = views.ConsoleStatusLabel(
        object(),
        name="gui",
        max_gui_chars=20,
    )
    label.config(text="This is a fairly long status line")

    assert printed == [
        (("[gui] This is a fairly long status line",), {"flush": True}),
    ]
    assert label.cget("text") == "This is a fairly lo..."

    label.config(text="This is a fairly long status line")
    assert len(printed) == 1


def test_create_status_panel_stores_console_labels_and_progressbar(monkeypatch) -> None:
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Progressbar", _FakeProgressbar)

    view_state = state.StatusPanelViewState()
    views.create_status_panel(parent=object(), view_state=view_state)

    assert isinstance(view_state.progress_label_positions, views.ConsoleStatusLabel)
    assert isinstance(view_state.progress_label_geometry, views.ConsoleStatusLabel)
    assert isinstance(view_state.ordered_structure_progressbar, _FakeProgressbar)
    assert isinstance(view_state.progress_label_ordered_structure, views.ConsoleStatusLabel)
    assert isinstance(view_state.mosaic_progressbar, _FakeProgressbar)
    assert isinstance(view_state.progress_label_mosaic, views.ConsoleStatusLabel)
    assert isinstance(view_state.progress_label, views.ConsoleStatusLabel)
    assert view_state.update_timing_label.text.startswith("Timing | image generation:")
    assert view_state.chi_square_label.text == "Chi-Squared: "


def test_create_workspace_panels_stores_panel_refs(monkeypatch) -> None:
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)

    view_state = state.WorkspacePanelsViewState()

    views.create_workspace_panels(
        parent=object(),
        view_state=view_state,
    )

    assert isinstance(view_state.workspace_actions_frame, _FakeFrame)
    assert view_state.workspace_actions_frame.kwargs["text"] == "Setup Actions"
    assert isinstance(view_state.workspace_backgrounds_frame, _FakeFrame)
    assert view_state.workspace_backgrounds_frame.kwargs["text"] == "Backgrounds"
    assert isinstance(view_state.workspace_inputs_frame, _FakeFrame)
    assert view_state.workspace_inputs_frame.kwargs["text"] == "Input Model"
    assert isinstance(view_state.workspace_session_frame, _FakeFrame)
    assert view_state.workspace_session_frame.kwargs["text"] == "Session"
    assert isinstance(view_state.workspace_debug_frame, _FakeCollapsibleFrame)
    assert view_state.workspace_debug_frame.text == "Advanced / Debug"


def test_populate_stacked_button_group_creates_buttons_in_order(monkeypatch) -> None:
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)

    called = []
    views.populate_stacked_button_group(
        object(),
        [
            ("First", lambda: called.append("first")),
            ("Second", lambda: called.append("second")),
        ],
    )

    assert [button.kwargs["text"] for button in _FakeButton.created] == ["First", "Second"]

    _FakeButton.created[0].command()
    _FakeButton.created[1].command()
    assert called == ["first", "second"]


def test_set_collapsible_header_summary_calls_frame_setter() -> None:
    frame = _FakeCollapsibleFrame(object(), text="Section")

    views.set_collapsible_header_summary(frame, "a 4.2 A | c 12.3 A")

    assert frame.summary_text == "a 4.2 A | c 12.3 A"


def test_background_file_controls_store_status_var_and_update_text(monkeypatch) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.WorkspacePanelsViewState()
    loaded = []

    views.create_background_file_controls(
        parent=object(),
        view_state=view_state,
        on_load_backgrounds=lambda: loaded.append(True),
        status_text="initial status",
    )

    assert view_state.background_file_status_var.get() == "initial status"
    assert view_state.background_file_status_label.kwargs["textvariable"] is (
        view_state.background_file_status_var
    )

    views.set_background_file_status_text(view_state, "updated status")
    assert view_state.background_file_status_var.get() == "updated status"

    _FakeButton.created[0].command()
    assert loaded == [True]


def test_geometry_fit_parameter_controls_store_toggle_vars_and_grid_checkbuttons(
    monkeypatch,
) -> None:
    class _GridCheckbutton:
        created = []

        def __init__(self, parent, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            self.variable = kwargs.get("variable")
            self.text = kwargs.get("text", "")
            self.grid_kwargs = None
            _GridCheckbutton.created.append(self)

        def grid(self, **kwargs) -> None:
            self.grid_kwargs = kwargs

        def config(self, **kwargs) -> None:
            self.text = kwargs.get("text", self.text)

        def configure(self, **kwargs) -> None:
            self.config(**kwargs)

    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Checkbutton", _GridCheckbutton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)

    view_state = state.GeometryFitParameterControlsViewState()

    views.create_geometry_fit_parameter_controls(
        parent=object(),
        view_state=view_state,
        initial_values={
            "zb": True,
            "zs": True,
            "theta_initial": True,
            "psi_z": False,
            "chi": True,
            "cor_angle": False,
            "gamma": True,
            "Gamma": True,
            "corto_detector": True,
            "a": False,
            "c": False,
            "center_x": True,
            "center_y": False,
        },
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert view_state.fit_zb_var.get() is True
    assert view_state.fit_psi_z_var.get() is False
    assert view_state.fit_center_x_var.get() is True
    assert view_state.fit_center_y_var.get() is False
    assert view_state.fit_theta_checkbutton is _GridCheckbutton.created[2]
    assert view_state.toggle_vars["theta_initial"] is view_state.fit_theta_var
    assert view_state.toggle_checkbuttons["theta_initial"] is view_state.fit_theta_checkbutton
    assert view_state.frame.columnconfigure_calls == [(0, 1), (1, 1), (2, 1), (3, 1)]
    assert _GridCheckbutton.created[0].text == "z_b beam offset"
    assert _GridCheckbutton.created[2].text == "θ sample tilt"
    assert _GridCheckbutton.created[-1].text == "center col"
    assert _GridCheckbutton.created[0].grid_kwargs == {
        "row": 0,
        "column": 0,
        "sticky": "w",
        "padx": 4,
        "pady": 2,
    }
    assert _GridCheckbutton.created[-1].grid_kwargs == {
        "row": 3,
        "column": 0,
        "sticky": "w",
        "padx": 4,
        "pady": 2,
    }


def test_primary_cif_controls_store_vars_and_bind_actions(monkeypatch) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.PrimaryCifControlsViewState()
    calls = []

    def _apply(event=None) -> None:
        calls.append(("apply", event))

    views.create_primary_cif_controls(
        parent=object(),
        view_state=view_state,
        cif_path_text="example.cif",
        on_apply_from_entry=_apply,
        on_browse_primary_cif=lambda: calls.append("browse"),
        on_open_diffuse_ht=lambda: calls.append("diffuse"),
        on_export_diffuse_ht=lambda: calls.append("export"),
    )

    assert isinstance(view_state.cif_frame, _FakeCollapsibleFrame)
    assert view_state.cif_file_var.get() == "example.cif"
    assert view_state.cif_entry.textvariable is view_state.cif_file_var
    assert isinstance(view_state.cif_actions_frame, _FakeFrame)
    assert view_state.browse_button is _FakeButton.created[0]
    assert view_state.apply_button is _FakeButton.created[1]
    assert view_state.diffuse_ht_button is _FakeButton.created[2]
    assert view_state.export_diffuse_ht_button is _FakeButton.created[3]
    assert _FakeLabel.created[0].text == "Path"

    view_state.cif_entry.bindings["<Return>"]("event")
    view_state.browse_button.command()
    view_state.apply_button.command()
    view_state.diffuse_ht_button.command()
    view_state.export_diffuse_ht_button.command()

    assert calls == [
        ("apply", "event"),
        "browse",
        ("apply", None),
        "diffuse",
        "export",
    ]


def test_cif_weight_controls_store_slider_refs_when_secondary_cif_is_present(
    monkeypatch,
) -> None:
    created = []

    def _fake_create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback=None,
        **_kwargs,
    ):
        var = _FakeVar(initial_val)
        slider = _FakeScale(parent, to=max_val)
        created.append(
            {
                "label": label,
                "var": var,
                "slider": slider,
                "step": step_size,
                "parent": parent,
                "update_callback": update_callback,
                "min": min_val,
                "max": max_val,
            }
        )
        return var, slider

    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views, "create_slider", _fake_create_slider)
    monkeypatch.setattr(views.tk, "DoubleVar", _FakeVar)

    view_state = state.CifWeightControlsViewState()

    views.create_cif_weight_controls(
        parent=object(),
        view_state=view_state,
        has_second_cif=True,
        weight1=0.6,
        weight2=0.4,
    )

    assert isinstance(view_state.frame, _FakeCollapsibleFrame)
    assert view_state.weight1_var.get() == 0.6
    assert view_state.weight2_var.get() == 0.4
    assert view_state.weight1_scale is created[0]["slider"]
    assert view_state.weight2_scale is created[1]["slider"]
    assert [item["label"] for item in created] == ["CIF1 Weight", "CIF2 Weight"]


def test_cif_weight_controls_create_only_vars_without_secondary_cif(monkeypatch) -> None:
    monkeypatch.setattr(views.tk, "DoubleVar", _FakeVar)

    view_state = state.CifWeightControlsViewState()

    views.create_cif_weight_controls(
        parent=object(),
        view_state=view_state,
        has_second_cif=False,
        weight1=1.0,
        weight2=0.0,
    )

    assert view_state.frame is None
    assert view_state.weight1_var.get() == 1.0
    assert view_state.weight2_var.get() == 0.0
    assert view_state.weight1_scale is None
    assert view_state.weight2_scale is None


def test_display_controls_store_refs_and_slider_vars(monkeypatch) -> None:
    class _FakeSliderRow:
        def __init__(self) -> None:
            self.children = []

        def winfo_children(self):
            return list(self.children)

    class _FakeSliderWidget:
        def __init__(self, entry, min_val, max_val) -> None:
            self.master = _FakeSliderRow()
            self.master.children = [self, entry]
            self.bounds = {"from": min_val, "to": max_val}

        def configure(self, **kwargs) -> None:
            if "from_" in kwargs:
                self.bounds["from"] = kwargs["from_"]
            if "to" in kwargs:
                self.bounds["to"] = kwargs["to"]

        def cget(self, key: str):
            return self.bounds[key]

    created = []

    def _fake_create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback=None,
        **_kwargs,
    ):
        var = _FakeVar(initial_val)
        entry = _FakeEntry(parent)
        slider = _FakeSliderWidget(entry, min_val, max_val)
        created.append(
            {
                "label": label,
                "var": var,
                "slider": slider,
                "step": step_size,
                "parent": parent,
                "update_callback": update_callback,
                "entry": entry,
            }
        )
        return var, slider

    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views, "create_slider", _fake_create_slider)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)
    _FakeCheckbutton.created = []

    view_state = state.DisplayControlsViewState()

    views.create_display_controls(
        parent=object(),
        view_state=view_state,
        background_range=(-1.0, 10.0),
        background_defaults=(0.0, 5.0),
        background_step=0.1,
        background_transparency=0.25,
        simulation_range=(0.0, 20.0),
        simulation_defaults=(0.0, 7.5),
        simulation_step=0.01,
        scale_factor_range=(0.0, 2.0),
        scale_factor_value=1.25,
        scale_factor_step=0.0001,
        on_apply_background_limits=lambda: None,
        on_apply_simulation_limits=lambda: None,
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert isinstance(view_state.background_controls_frame, _FakeFrame)
    assert isinstance(view_state.simulation_controls_frame, _FakeFrame)
    assert view_state.background_min_var.get() == 0.0
    assert view_state.background_max_var.get() == 5.0
    assert view_state.background_transparency_var.get() == 0.25
    assert view_state.simulation_min_var.get() == 0.0
    assert view_state.simulation_max_var.get() == 7.5
    assert view_state.simulation_scale_factor_var.get() == 1.25
    assert view_state.background_min_slider is created[0]["slider"]
    assert view_state.background_max_slider is created[1]["slider"]
    assert view_state.background_transparency_slider is created[2]["slider"]
    assert view_state.simulation_min_slider is created[3]["slider"]
    assert view_state.simulation_max_slider is created[4]["slider"]
    assert view_state.scale_factor_slider is created[5]["slider"]
    assert view_state.scale_factor_entry is created[5]["entry"]
    assert view_state.accumulate_intensity_var is None
    assert view_state.accumulate_intensity_checkbutton is None
    assert view_state.fast_viewer_var.get() is False
    assert view_state.fast_viewer_checkbutton is None
    assert view_state.fast_viewer_status_var.get() == ""
    assert len(_FakeCheckbutton.created) == 0
    assert [item["label"] for item in created] == [
        "Background Min Intensity",
        "Background Max Intensity",
        "Background Transparency",
        "Simulation Min Intensity",
        "Simulation Max Intensity",
        "Simulation Scale Factor",
    ]


def test_structure_factor_pruning_controls_store_refs_and_support_helpers(
    monkeypatch,
) -> None:
    created = []

    def _fake_create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback=None,
        **_kwargs,
    ):
        var = _FakeVar(initial_val)
        slider = _FakeScale(parent, to=max_val)
        created.append(
            {
                "label": label,
                "var": var,
                "slider": slider,
                "step": step_size,
                "parent": parent,
                "update_callback": update_callback,
                "min": min_val,
                "max": max_val,
            }
        )
        return var, slider

    _FakeLabel.created = []
    _FakeRadiobutton.created = []
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Radiobutton", _FakeRadiobutton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)
    monkeypatch.setattr(views, "create_slider", _fake_create_slider)

    view_state = state.StructureFactorPruningControlsViewState()

    views.create_structure_factor_pruning_controls(
        parent=object(),
        view_state=view_state,
        sf_prune_bias_range=(-2.0, 2.0),
        sf_prune_bias_value=0.25,
        solve_q_mode="adaptive",
        solve_q_steps_range=(4.0, 128.0),
        solve_q_steps_value=32.0,
        solve_q_rel_tol_range=(1e-6, 1e-2),
        solve_q_rel_tol_value=1e-4,
        status_text="ready",
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert view_state.sf_prune_bias_var.get() == 0.25
    assert view_state.sf_prune_status_var.get() == "ready"
    assert view_state.solve_q_mode_var.get() == "adaptive"
    assert view_state.solve_q_steps_var.get() == 32.0
    assert view_state.solve_q_rel_tol_var.get() == 1e-4
    assert view_state.sf_prune_bias_scale is created[0]["slider"]
    assert view_state.solve_q_steps_scale is created[1]["slider"]
    assert view_state.solve_q_rel_tol_scale is created[2]["slider"]
    assert [item["label"] for item in created] == [
        "SF Prune Bias",
        "Arc Max Intervals",
        "Arc Relative Tol",
    ]
    assert [radio.value for radio in _FakeRadiobutton.created] == [
        "uniform",
        "adaptive",
    ]

    views.set_structure_factor_pruning_status_text(view_state, "updated")
    assert view_state.sf_prune_status_var.get() == "updated"

    views.set_structure_factor_pruning_rel_tol_enabled(view_state, enabled=False)
    assert view_state.solve_q_rel_tol_scale.state == "disabled"

    views.set_structure_factor_pruning_rel_tol_enabled(view_state, enabled=True)
    assert view_state.solve_q_rel_tol_scale.state == "normal"


def test_beam_mosaic_parameter_sliders_store_refs_and_callbacks(monkeypatch) -> None:
    created = []

    def _fake_create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback=None,
        **kwargs,
    ):
        var = _FakeVar(initial_val)
        slider = _FakeScale(parent, to=max_val)
        created.append(
            {
                "label": label,
                "var": var,
                "slider": slider,
                "step": step_size,
                "parent": parent,
                "update_callback": update_callback,
                "kwargs": kwargs,
                "min": min_val,
                "max": max_val,
            }
        )
        return var, slider

    monkeypatch.setattr(views, "create_slider", _fake_create_slider)

    view_state = state.BeamMosaicParameterSlidersViewState()
    standard_calls = []
    mosaic_calls = []

    views.create_beam_mosaic_parameter_sliders(
        geometry_parent="geometry",
        debye_parent="debye",
        detector_parent="detector",
        lattice_parent="lattice",
        mosaic_parent="mosaic",
        beam_parent="beam",
        view_state=view_state,
        image_size=2048.0,
        values={
            "theta_initial": 5.0,
            "cor_angle": 0.5,
            "gamma": 1.0,
            "Gamma": -1.0,
            "chi": 0.2,
            "psi_z": 0.3,
            "zs": 1.0e-4,
            "zb": -1.0e-4,
            "sample_width_m": 2.0e-3,
            "sample_length_m": 3.0e-3,
            "sample_depth_m": 4.0e-7,
            "debye_x": 0.1,
            "debye_y": 0.2,
            "corto_detector": 0.03,
            "a": 4.2,
            "c": 25.0,
            "sigma_mosaic_deg": 0.4,
            "gamma_mosaic_deg": 0.5,
            "eta": 0.6,
            "center_x": 1200.0,
            "center_y": 900.0,
            "bandwidth_percent": 0.75,
        },
        on_standard_update=lambda: standard_calls.append("standard"),
        on_mosaic_update=lambda: mosaic_calls.append("mosaic"),
    )

    assert view_state.theta_initial_var.get() == 5.0
    assert view_state.gamma_var.get() == 1.0
    assert view_state.a_var.get() == 4.2
    assert view_state.sigma_mosaic_var.get() == 0.4
    assert view_state.center_x_var.get() == 1200.0
    assert view_state.bandwidth_percent_var.get() == 0.75
    assert view_state.center_y_var.get() == 900.0
    assert len(created) == 22
    assert created[0]["label"] == "θ sample tilt"
    assert created[-1]["label"] == "Beam Center Col"
    assert created[2]["kwargs"]["allow_range_expand"] is True
    assert created[8]["kwargs"]["allow_range_expand"] is True
    assert created[9]["kwargs"]["range_expand_pad"] == 1.0e-3
    assert created[10]["kwargs"]["range_expand_pad"] == 1.0e-7
    assert created[14]["kwargs"]["range_expand_pad"] == 0.1
    assert created[15]["kwargs"]["range_expand_pad"] == 0.5
    assert created[19]["max"] == 3000.0
    assert created[21]["max"] == 3000.0
    assert created[0]["update_callback"] is not None
    assert created[16]["update_callback"] is not None
    created[0]["update_callback"]()
    created[16]["update_callback"]()
    assert standard_calls == ["standard"]
    assert mosaic_calls == ["mosaic"]


def test_sampling_optics_controls_store_vars_bind_apply_and_toggle_custom_state(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    _FakeRadiobutton.created = []
    _FakeOptionMenu.created = []
    _FakeScale.created = []
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Radiobutton", _FakeRadiobutton)
    monkeypatch.setattr(views.ttk, "OptionMenu", _FakeOptionMenu)
    monkeypatch.setattr(views.tk, "IntVar", _FakeVar)
    monkeypatch.setattr(views.tk, "Scale", _FakeScale)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.SamplingOpticsControlsViewState()
    applied = []
    rod_commits = []

    views.create_sampling_optics_controls(
        parent=object(),
        view_state=view_state,
        resolution_options=["Low", "High", "Custom"],
        initial_resolution="High",
        custom_samples_text="2500",
        resolution_count_text="2,500 samples",
        rod_points_per_gz_value=480,
        rod_points_per_gz_min=10,
        rod_points_per_gz_max=2000,
        rod_points_per_gz_text="480 / Gz",
        rod_point_total_text="Longest rod: 960 points",
        optics_mode_text="exact",
        on_apply_custom_samples=lambda: applied.append("apply"),
        on_rod_points_per_gz_slide=lambda _value: rod_commits.append("slide"),
        on_commit_rod_points_per_gz=lambda _event: rod_commits.append("commit"),
    )

    assert isinstance(view_state.resolution_selector_frame, _FakeFrame)
    assert view_state.resolution_var.get() == "High"
    assert view_state.custom_samples_var.get() == "2500"
    assert view_state.resolution_count_var.get() == "2,500 samples"
    assert view_state.rod_points_per_gz_var.get() == 480
    assert view_state.rod_points_per_gz_value_var.get() == "480 / Gz"
    assert view_state.rod_point_total_var.get() == "Longest rod: 960 points"
    assert view_state.optics_mode_var.get() == "exact"
    assert view_state.custom_samples_entry.textvariable is view_state.custom_samples_var
    assert view_state.custom_samples_apply_button is _FakeButton.created[0]
    assert _FakeOptionMenu.created[0].variable is view_state.resolution_var
    assert _FakeOptionMenu.created[0].default == "High"
    assert _FakeOptionMenu.created[0].values == ("Low", "High", "Custom")
    assert [radio.value for radio in _FakeRadiobutton.created] == ["fast", "exact"]
    assert view_state.rod_points_per_gz_scale is _FakeScale.created[0]

    views.set_sampling_resolution_summary_text(view_state, "3,600 samples (custom)")
    assert view_state.resolution_count_var.get() == "3,600 samples (custom)"
    views.set_sampling_rod_points_per_gz_text(view_state, "512 / Gz")
    views.set_sampling_rod_point_total_text(view_state, "Longest rod: 1,024 points")
    assert view_state.rod_points_per_gz_value_var.get() == "512 / Gz"
    assert view_state.rod_point_total_var.get() == "Longest rod: 1,024 points"

    views.set_sampling_custom_controls_enabled(view_state, enabled=False)
    assert view_state.custom_samples_entry.state == tk.DISABLED
    assert view_state.custom_samples_apply_button.state == tk.DISABLED

    views.set_sampling_custom_controls_enabled(view_state, enabled=True)
    assert view_state.custom_samples_entry.state == tk.NORMAL
    assert view_state.custom_samples_apply_button.state == tk.NORMAL

    view_state.custom_samples_entry.bindings["<Return>"](None)
    _FakeButton.created[0].command()
    view_state.rod_points_per_gz_scale.command(512)
    view_state.rod_points_per_gz_scale.bindings["<ButtonRelease-1>"][0](None)
    assert applied == ["apply", "apply"]
    assert rod_commits == ["slide", "commit"]


def test_finite_stack_controls_store_vars_bindings_and_helper_updates(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    _FakeCheckbutton.created = []
    _FakeScale.created = []
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)
    monkeypatch.setattr(views.tk, "IntVar", _FakeVar)
    monkeypatch.setattr(views.tk, "DoubleVar", _FakeVar)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)
    monkeypatch.setattr(views.tk, "Scale", _FakeScale)

    view_state = state.FiniteStackControlsViewState()
    calls = []

    views.create_finite_stack_controls(
        parent=object(),
        view_state=view_state,
        finite_stack=True,
        stack_layers=64,
        phi_l_divisor=3.5,
        phase_delta_expression="2*pi*L/3",
        on_toggle_finite_stack=lambda: calls.append("toggle"),
        on_layer_slider=lambda value: calls.append(("slider", value)),
        on_commit_layer_entry=lambda _event: calls.append("layers"),
        on_commit_phi_l_divisor_entry=lambda _event: calls.append("phi"),
        on_commit_phase_delta_expression_entry=lambda _event: calls.append("phase"),
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert view_state.finite_stack_var.get() is True
    assert view_state.stack_layers_var.get() == 64
    assert view_state.layers_entry_var.get() == "64"
    assert view_state.phi_l_divisor_var.get() == 3.5
    assert view_state.phi_l_divisor_entry_var.get() == "3.5"
    assert view_state.phase_delta_expr_var.get() == "2*pi*L/3"
    assert view_state.phase_delta_entry_var.get() == "2*pi*L/3"
    assert _FakeCheckbutton.created[0].kwargs["text"] == "Finite Stack"
    assert _FakeScale.created[0].kwargs["variable"] is view_state.stack_layers_var

    views.set_finite_stack_layer_controls_enabled(view_state, enabled=False)
    assert view_state.layers_scale.state == tk.DISABLED
    assert view_state.layers_entry.state == tk.DISABLED

    views.set_finite_stack_layer_controls_enabled(view_state, enabled=True)
    assert view_state.layers_scale.state == tk.NORMAL
    assert view_state.layers_entry.state == tk.NORMAL

    views.ensure_finite_stack_layer_scale_max(view_state, 1200)
    assert view_state.layers_scale.cget("to") == 1200

    views.set_finite_stack_layer_entry_text(view_state, "72")
    views.set_finite_stack_phi_l_divisor_entry_text(view_state, "4")
    views.set_finite_stack_phase_delta_entry_text(view_state, "L/2")
    assert view_state.layers_entry_var.get() == "72"
    assert view_state.phi_l_divisor_entry_var.get() == "4"
    assert view_state.phase_delta_entry_var.get() == "L/2"

    _FakeCheckbutton.created[0].command()
    view_state.layers_entry.bindings["<Return>"](None)
    view_state.layers_entry.bindings["<FocusOut>"](None)
    view_state.phi_l_divisor_entry.bindings["<Return>"](None)
    view_state.phase_delta_entry.bindings["<FocusOut>"](None)
    assert calls == ["toggle", "layers", "layers", "phi", "phase"]


def test_stacking_parameter_panels_and_slider_refs_are_stored(monkeypatch) -> None:
    created = []

    def _fake_create_slider(
        label,
        min_val,
        max_val,
        initial_val,
        step_size,
        parent,
        update_callback=None,
        **_kwargs,
    ):
        var = _FakeVar(initial_val)
        slider = _FakeScale(parent, to=max_val)
        created.append(
            {
                "label": label,
                "var": var,
                "slider": slider,
                "step": step_size,
                "parent": parent,
                "update_callback": update_callback,
            }
        )
        return var, slider

    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views, "create_slider", _fake_create_slider)

    view_state = state.StackingParameterControlsViewState()

    views.create_stacking_parameter_panels(parent=object(), view_state=view_state)
    views.create_stacking_probability_sliders(
        parent=view_state.stack_frame.frame,
        view_state=view_state,
        values={
            "p0": 0.02,
            "w0": 10.0,
            "p1": 0.95,
            "w1": 20.0,
            "p2": 0.40,
            "w2": 70.0,
        },
        on_update=lambda *_args: None,
    )

    assert isinstance(view_state.stack_frame, _FakeCollapsibleFrame)
    assert isinstance(view_state.occupancy_frame, _FakeCollapsibleFrame)
    assert isinstance(view_state.atom_site_frame, _FakeCollapsibleFrame)
    assert isinstance(view_state.occ_slider_frame, _FakeFrame)
    assert isinstance(view_state.occ_entry_frame, _FakeFrame)
    assert isinstance(view_state.atom_site_table_frame, _FakeFrame)
    assert view_state.p0_var.get() == 0.02
    assert view_state.w0_var.get() == 10.0
    assert view_state.p1_var.get() == 0.95
    assert view_state.w1_var.get() == 20.0
    assert view_state.p2_var.get() == 0.40
    assert view_state.w2_var.get() == 70.0
    assert view_state.p0_scale is created[0]["slider"]
    assert view_state.w0_scale is created[1]["slider"]
    assert view_state.p1_scale is created[2]["slider"]
    assert view_state.w1_scale is created[3]["slider"]
    assert view_state.p2_scale is created[4]["slider"]
    assert view_state.w2_scale is created[5]["slider"]
    assert [item["label"] for item in created] == [
        "p≈0",
        "w(p≈0)%",
        "p≈1",
        "w(p≈1)%",
        "p",
        "w(p)%",
    ]


def test_stacking_parameter_rebuild_helpers_render_dynamic_controls(monkeypatch) -> None:
    class _TrackedFrame:
        def __init__(self, _parent=None, **_kwargs) -> None:
            self.children = []
            self.columnconfigure_calls = []

        def winfo_children(self):
            return list(self.children)

        def columnconfigure(self, index: int, weight: int) -> None:
            self.columnconfigure_calls.append((index, weight))

    class _TrackedWidget:
        def __init__(self, parent, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            self.bindings = {}
            self.destroyed = False
            if hasattr(parent, "children"):
                parent.children.append(self)

        def pack(self, **_kwargs) -> None:
            pass

        def grid(self, **_kwargs) -> None:
            pass

        def bind(self, event: str, callback) -> None:
            self.bindings[event] = callback

        def destroy(self) -> None:
            self.destroyed = True

    class _TrackedLabel(_TrackedWidget):
        pass

    class _TrackedEntry(_TrackedWidget):
        pass

    class _TrackedScale(_TrackedWidget):
        pass

    monkeypatch.setattr(views.ttk, "Label", _TrackedLabel)
    monkeypatch.setattr(views.ttk, "Entry", _TrackedEntry)
    monkeypatch.setattr(views.ttk, "Scale", _TrackedScale)

    view_state = state.StackingParameterControlsViewState(
        occ_slider_frame=_TrackedFrame(),
        occ_entry_frame=_TrackedFrame(),
        atom_site_table_frame=_TrackedFrame(),
    )
    old_occ_slider_child = _FakeExistingChild()
    old_occ_entry_child = _FakeExistingChild()
    old_atom_child = _FakeExistingChild()
    view_state.occ_slider_frame.children.append(old_occ_slider_child)
    view_state.occ_entry_frame.children.append(old_occ_entry_child)
    view_state.atom_site_table_frame.children.append(old_atom_child)

    occ_vars = [_FakeVar(0.4), _FakeVar(0.8)]
    atom_site_vars = [
        {
            "x": _FakeVar(0.1),
            "y": _FakeVar(0.2),
            "z": _FakeVar(0.3),
        }
    ]
    occupancy_updates = []

    views.rebuild_occupancy_controls(
        view_state=view_state,
        occ_vars=occ_vars,
        occupancy_label_text=lambda idx: f"Occupancy {idx + 1}",
        occupancy_input_label_text=lambda idx: f"Input {idx + 1}",
        on_update=lambda *args: occupancy_updates.append(args),
    )
    views.rebuild_atom_site_fractional_controls(
        view_state=view_state,
        atom_site_fract_vars=atom_site_vars,
        atom_site_label_text=lambda idx: f"site_{idx + 1}",
        on_update=lambda *_args: None,
        empty_text="empty",
    )

    assert old_occ_slider_child.destroyed is True
    assert old_occ_entry_child.destroyed is True
    assert old_atom_child.destroyed is True
    assert len(view_state.occ_label_widgets) == 2
    assert len(view_state.occ_scale_widgets) == 2
    assert len(view_state.occ_entry_label_widgets) == 2
    assert len(view_state.occ_entry_widgets) == 2
    assert view_state.occ_label_widgets[0].kwargs["text"] == "Occupancy 1"
    assert view_state.occ_entry_label_widgets[0].kwargs["text"] == "Input 1:"
    assert "<ButtonRelease-1>" in view_state.occ_scale_widgets[0].bindings
    assert "<Return>" in view_state.occ_entry_widgets[0].bindings
    assert "<FocusOut>" in view_state.occ_entry_widgets[0].bindings
    assert len(view_state.atom_site_coord_entry_widgets) == 3
    assert "<Return>" in view_state.atom_site_coord_entry_widgets[0].bindings
    assert "<FocusOut>" in view_state.atom_site_coord_entry_widgets[0].bindings
    occ_vars[0].set(0.6)
    view_state.occ_scale_widgets[0].bindings["<ButtonRelease-1>"](None)
    assert occupancy_updates[-1] == (0.6,)
    assert view_state.atom_site_table_frame.columnconfigure_calls == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]


def test_ordered_structure_fit_panel_helpers_build_and_rebuild_controls(monkeypatch) -> None:
    class _GridCheckbutton:
        created = []

        def __init__(self, parent, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            self.command = kwargs.get("command")
            self.variable = kwargs.get("variable")
            self.grid_kwargs = None
            self.pack_kwargs = None
            _GridCheckbutton.created.append(self)

        def grid(self, **kwargs) -> None:
            self.grid_kwargs = kwargs

        def pack(self, **kwargs) -> None:
            self.pack_kwargs = kwargs

    class _TrackedFrame:
        def __init__(self, _parent=None, **_kwargs) -> None:
            self.children = []

        def winfo_children(self):
            return list(self.children)

        def pack(self, **_kwargs) -> None:
            pass

    class _TrackedWidget:
        def __init__(self, parent, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            self.bindings = {}
            self.destroyed = False
            if hasattr(parent, "children"):
                parent.children.append(self)

        def pack(self, **_kwargs) -> None:
            pass

        def grid(self, **_kwargs) -> None:
            pass

        def bind(self, event: str, callback) -> None:
            self.bindings[event] = callback

        def destroy(self) -> None:
            self.destroyed = True

    class _TrackedLabel(_TrackedWidget):
        pass

    class _TrackedEntry(_TrackedWidget):
        pass

    _GridCheckbutton.created = []
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views.ttk, "Frame", _TrackedFrame)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Entry", _TrackedEntry)
    monkeypatch.setattr(views.ttk, "Label", _TrackedLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _GridCheckbutton)

    result_var = _FakeStringVar("idle")
    view_state = state.OrderedStructureFitControlsViewState()
    views.create_ordered_structure_fit_panel(
        parent=object(),
        view_state=view_state,
        ordered_scale_var=_FakeVar(1.0),
        coord_window_var=_FakeVar(0.02),
        fit_debye_x_var=_FakeVar(True),
        fit_debye_y_var=_FakeVar(True),
        result_var=result_var,
        on_fit=lambda: None,
        on_revert=lambda: None,
        on_commit_ordered_scale=lambda *_args: None,
        on_commit_coord_window=lambda *_args: None,
    )

    assert isinstance(view_state.frame, _FakeCollapsibleFrame)
    assert isinstance(view_state.fit_button, _FakeButton)
    assert isinstance(view_state.revert_button, _FakeButton)
    assert view_state.revert_button.state == tk.DISABLED
    assert isinstance(view_state.ordered_scale_entry, _TrackedEntry)
    assert "<Return>" in view_state.ordered_scale_entry.bindings
    assert "<FocusOut>" in view_state.coord_window_entry.bindings
    assert view_state.result_var is result_var

    old_occ_child = _FakeExistingChild()
    old_atom_child = _FakeExistingChild()
    view_state.occupancy_toggle_frame.children.append(old_occ_child)
    view_state.atom_toggle_frame.children.append(old_atom_child)
    occ_vars = [_FakeVar(True), _FakeVar(False)]
    atom_vars = [{"x": _FakeVar(False), "y": _FakeVar(False), "z": _FakeVar(True)}]

    views.rebuild_ordered_structure_fit_occupancy_controls(
        view_state=view_state,
        occupancy_vars=occ_vars,
        occupancy_label_text=lambda idx: f"Occ {idx + 1}",
    )
    views.rebuild_ordered_structure_fit_atom_coordinate_controls(
        view_state=view_state,
        atom_toggle_vars=atom_vars,
        atom_site_label_text=lambda idx: f"site_{idx + 1}",
    )

    assert old_occ_child.destroyed is True
    assert old_atom_child.destroyed is True
    assert len(view_state.occupancy_toggle_widgets) == 3
    assert len(view_state.atom_toggle_widgets) == 4


def test_geometry_tool_action_controls_store_refs_and_support_updates(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.GeometryToolActionsViewState()
    calls = []

    views.create_geometry_fit_history_controls(
        parent=object(),
        view_state=view_state,
        on_undo_fit=lambda: calls.append("undo-fit"),
        on_redo_fit=lambda: calls.append("redo-fit"),
    )

    views.create_geometry_tool_action_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_manual_pick=lambda: calls.append("toggle-pick"),
        on_undo_manual_placement=lambda: calls.append("undo-placement"),
        on_export_manual_pairs=lambda: calls.append("export"),
        on_import_manual_pairs=lambda: calls.append("import"),
        on_toggle_preview_exclude=lambda: calls.append("toggle-preview"),
        on_clear_manual_pairs=lambda: calls.append("clear"),
    )

    assert view_state.geometry_manual_pick_button_var.get() == "Pick Qr Sets on Image"
    assert view_state.geometry_preview_exclude_button_var.get() == "Choose Active Qr/Qz Groups"
    assert [button.kwargs.get("text") for button in _FakeButton.created[:4]] == [
        "Undo Fit",
        "Redo Fit",
        None,
        "Undo Placement",
    ]
    assert _FakeButton.created[2].kwargs["textvariable"] is view_state.geometry_manual_pick_button_var
    assert _FakeButton.created[6].kwargs["textvariable"] is (
        view_state.geometry_preview_exclude_button_var
    )

    views.set_geometry_tool_action_texts(
        view_state,
        manual_pick_text="manual-updated",
        preview_exclude_text="preview-updated",
    )
    assert view_state.geometry_manual_pick_button_var.get() == "manual-updated"
    assert view_state.geometry_preview_exclude_button_var.get() == "preview-updated"

    views.set_geometry_fit_history_button_state(
        view_state,
        can_undo=True,
        can_redo=False,
    )
    assert view_state.undo_geometry_fit_button.state == "normal"
    assert view_state.redo_geometry_fit_button.state == "disabled"

    for button in _FakeButton.created:
        if callable(button.command):
            button.command()
    assert calls == [
        "undo-fit",
        "redo-fit",
        "toggle-pick",
        "undo-placement",
        "export",
        "import",
        "toggle-preview",
        "clear",
    ]


def test_hkl_lookup_controls_store_vars_bind_entries_and_support_updates(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.HklLookupViewState()
    calls = []

    views.create_hkl_lookup_controls(
        parent=object(),
        view_state=view_state,
        on_select_hkl=lambda: calls.append("select"),
        on_toggle_hkl_pick=lambda: calls.append("pick"),
        on_clear_selected_peak=lambda: calls.append("clear"),
        on_show_bragg_ewald=lambda: calls.append("bragg-ewald"),
        on_open_bragg_qr_groups=lambda: calls.append("bragg-qr"),
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert view_state.frame.kwargs["text"] == "Peak Lookup (HKL)"
    assert view_state.selected_h_var.get() == "0"
    assert view_state.selected_k_var.get() == "0"
    assert view_state.selected_l_var.get() == "0"
    assert view_state.hkl_pick_button_var.get() == "Pick HKL on Image"
    assert view_state.h_entry.textvariable is view_state.selected_h_var
    assert view_state.k_entry.textvariable is view_state.selected_k_var
    assert view_state.l_entry.textvariable is view_state.selected_l_var
    assert [button.kwargs.get("text") for button in _FakeButton.created] == [
        "Select HKL",
        None,
        "Clear",
        "Open Specular View",
        "Bragg Qr Groups",
    ]
    assert _FakeButton.created[1].kwargs["textvariable"] is view_state.hkl_pick_button_var

    view_state.h_entry.bindings["<Return>"](None)
    view_state.k_entry.bindings["<Return>"](None)
    view_state.l_entry.bindings["<Return>"](None)
    assert calls == ["select", "select", "select"]

    views.set_hkl_lookup_values(
        view_state,
        h_text="1",
        k_text="2",
        l_text="-3",
    )
    assert view_state.selected_h_var.get() == "1"
    assert view_state.selected_k_var.get() == "2"
    assert view_state.selected_l_var.get() == "-3"

    views.set_hkl_pick_button_text(view_state, "Pick HKL on Image (Armed)")
    assert view_state.hkl_pick_button_var.get() == "Pick HKL on Image (Armed)"

    for button in _FakeButton.created:
        if callable(button.command):
            button.command()
    assert calls == [
        "select",
        "select",
        "select",
        "select",
        "pick",
        "clear",
        "bragg-ewald",
        "bragg-qr",
    ]


def test_geometry_overlay_action_controls_store_refs_and_commands(
    monkeypatch,
) -> None:
    _FakeCheckbutton.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)

    view_state = state.GeometryOverlayActionsViewState()
    calls = []

    views.create_geometry_overlay_action_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_geometry_overlays=lambda: calls.append("toggle-geometry-overlays"),
        on_fit_mosaic=lambda: calls.append("fit-mosaic"),
        mosaic_fit_initial_values={
            "fit_sigma_mosaic": True,
            "fit_gamma_mosaic": False,
            "fit_eta": True,
            "fit_theta_i": False,
        },
    )

    assert view_state.show_geometry_overlays_var.get() is True
    assert view_state.fit_sigma_mosaic_var.get() is True
    assert view_state.fit_gamma_mosaic_var.get() is False
    assert view_state.fit_eta_var.get() is True
    assert view_state.fit_theta_i_var.get() is False
    assert view_state.show_qr_cylinder_overlay_checkbutton is None
    assert view_state.show_geometry_overlays_checkbutton is _FakeCheckbutton.created[0]
    assert view_state.fit_button_mosaic is _FakeButton.created[0]
    assert view_state.mosaic_fit_toggle_checkbuttons["fit_gamma_mosaic"] is _FakeCheckbutton.created[2]
    assert view_state.mosaic_fit_toggle_checkbuttons["fit_theta_i"] is _FakeCheckbutton.created[4]
    assert _FakeCheckbutton.created[0].kwargs["text"] == "Show Geometry Overlays"
    assert [check.kwargs["text"] for check in _FakeCheckbutton.created] == [
        "Show Geometry Overlays",
        "Fit mosaic sigma",
        "Fit mosaic gamma",
        "Fit eta",
        "Fit theta_i",
    ]
    assert [button.kwargs["text"] for button in _FakeButton.created] == [
        "Fit Mosaic Shapes",
    ]

    _FakeCheckbutton.created[0].command()
    _FakeButton.created[0].command()
    assert calls == ["toggle-geometry-overlays", "fit-mosaic"]


def test_geometry_overlay_action_controls_can_build_split_sections(monkeypatch) -> None:
    _FakeCheckbutton.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)

    view_state = state.GeometryOverlayActionsViewState()

    views.create_geometry_overlay_action_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_geometry_overlays=lambda: None,
        on_fit_mosaic=lambda: None,
        include_fit_button=False,
    )
    views.create_geometry_overlay_action_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_geometry_overlays=lambda: None,
        on_fit_mosaic=lambda: None,
        include_geometry_toggle=False,
    )

    assert view_state.show_geometry_overlays_checkbutton is _FakeCheckbutton.created[0]
    assert view_state.fit_button_mosaic is _FakeButton.created[0]
    assert [check.kwargs["text"] for check in _FakeCheckbutton.created[1:]] == [
        "Fit mosaic sigma",
        "Fit mosaic gamma",
        "Fit eta",
        "Fit theta_i",
    ]


def test_analysis_view_controls_store_vars_and_commands(monkeypatch) -> None:
    _FakeCheckbutton.created = []
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)

    view_state = state.AnalysisViewControlsViewState()
    calls = []

    views.create_analysis_view_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_1d_plots=lambda: calls.append("toggle-1d"),
        on_toggle_caked_2d=lambda: calls.append("toggle-2d"),
        on_toggle_log_radial=lambda: calls.append("toggle-radial"),
        on_toggle_log_azimuth=lambda: calls.append("toggle-azimuth"),
    )

    assert view_state.show_1d_var.get() is False
    assert view_state.show_caked_2d_var.get() is False
    assert view_state.log_radial_var.get() is False
    assert view_state.log_azimuth_var.get() is False
    assert [check.kwargs["text"] for check in _FakeCheckbutton.created] == [
        "Log Radial",
        "Log Azimuth",
    ]
    assert view_state.check_1d is None
    assert view_state.check_2d is None
    assert view_state.check_log_radial is _FakeCheckbutton.created[0]
    assert view_state.check_log_azimuth is _FakeCheckbutton.created[1]

    for checkbutton in _FakeCheckbutton.created:
        checkbutton.command()
    assert calls == ["toggle-radial", "toggle-azimuth"]


def test_analysis_peak_tools_controls_store_vars_and_commands(monkeypatch) -> None:
    _FakeCheckbutton.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.AnalysisPeakToolsViewState()
    calls = []

    views.create_analysis_peak_tools_controls(
        parent=object(),
        view_state=view_state,
        on_toggle_pick_mode=lambda: calls.append("pick"),
        on_clear_selection=lambda: calls.append("clear"),
        on_fit_selected_peaks=lambda: calls.append("fit"),
        pick_enabled=True,
        fit_gaussian=True,
        fit_lorentzian=False,
        fit_pseudo_voigt=True,
        fit_radial=True,
        fit_azimuth=False,
        selection_status_text="Selected peaks: 2",
        fit_results_text="Radial fits ready.",
    )

    assert isinstance(view_state.frame, _FakeFrame)
    assert view_state.pick_button is _FakeButton.created[0]
    assert view_state.clear_button is _FakeButton.created[1]
    assert view_state.fit_button is _FakeButton.created[2]
    assert view_state.pick_button.kwargs["text"] == "Stop Picking Peaks"
    assert view_state.clear_button.kwargs["text"] == "Clear Peaks and Fits"
    assert view_state.fit_button.kwargs["text"] == "Fit Selected Peaks"
    assert [check.kwargs["text"] for check in _FakeCheckbutton.created] == [
        "Gaussian",
        "Lorentzian",
        "Pseudo-Voigt (eta)",
        "Radial (2θ)",
        "Azimuth (φ)",
    ]
    assert view_state.fit_gaussian_var.get() is True
    assert view_state.fit_lorentzian_var.get() is False
    assert view_state.fit_pseudo_voigt_var.get() is True
    assert view_state.fit_radial_var.get() is True
    assert view_state.fit_azimuth_var.get() is False
    assert view_state.selection_status_var.get() == "Selected peaks: 2"
    assert view_state.fit_results_var.get() == "Radial fits ready."

    view_state.pick_button.command()
    view_state.clear_button.command()
    view_state.fit_button.command()
    assert calls == ["pick", "clear", "fit"]


def test_create_integration_range_controls_store_vars_bindings_and_commands(
    monkeypatch,
) -> None:
    _FakeLabel.created = []
    _FakeScale.created = []
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Scale", _FakeScale)
    monkeypatch.setattr(views.tk, "DoubleVar", _FakeVar)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.IntegrationRangeControlsViewState()
    slider_calls = []
    apply_calls = []

    views.create_integration_range_controls(
        parent=object(),
        view_state=view_state,
        tth_min=1.5,
        tth_max=55.4,
        phi_min=-12.6,
        phi_max=18.7,
        on_tth_min_changed=lambda value: slider_calls.append(("tth-min", float(value))),
        on_tth_max_changed=lambda value: slider_calls.append(("tth-max", float(value))),
        on_phi_min_changed=lambda value: slider_calls.append(("phi-min", float(value))),
        on_phi_max_changed=lambda value: slider_calls.append(("phi-max", float(value))),
        on_apply_entry=lambda entry_var, value_var, slider: apply_calls.append(
            (
                entry_var.get(),
                value_var.get(),
                slider.cget("from"),
                slider.cget("to"),
            )
        ),
    )

    assert isinstance(view_state.frame, _FakeCollapsibleFrame)
    assert view_state.frame.text == "Integration Ranges"
    assert view_state.frame.expanded is True
    assert view_state.range_frame is view_state.frame.frame
    assert isinstance(view_state.tth_min_container, _FakeFrame)
    assert isinstance(view_state.phi_max_container, _FakeFrame)
    assert view_state.tth_min_var.get() == 1.5
    assert view_state.tth_max_var.get() == 55.4
    assert view_state.phi_min_var.get() == -12.6
    assert view_state.phi_max_var.get() == 18.7
    assert view_state.tth_min_label_var.get() == "1.5"
    assert view_state.tth_max_label_var.get() == "55.4"
    assert view_state.phi_min_label_var.get() == "-12.6"
    assert view_state.phi_max_label_var.get() == "18.7"
    assert view_state.tth_min_entry_var.get() == "1.5000"
    assert view_state.phi_max_entry_var.get() == "18.7000"
    assert view_state.tth_min_label.kwargs["textvariable"] is view_state.tth_min_label_var
    assert view_state.phi_max_label.kwargs["textvariable"] is view_state.phi_max_label_var
    assert view_state.tth_min_entry.textvariable is view_state.tth_min_entry_var
    assert view_state.phi_max_entry.textvariable is view_state.phi_max_entry_var
    assert view_state.tth_min_slider is _FakeScale.created[0]
    assert view_state.tth_max_slider is _FakeScale.created[1]
    assert view_state.phi_min_slider is _FakeScale.created[2]
    assert view_state.phi_max_slider is _FakeScale.created[3]
    assert view_state.tth_min_slider.cget("from") == 0.0
    assert view_state.tth_min_slider.cget("to") == 90.0
    assert view_state.phi_min_slider.cget("from") == -90.0
    assert view_state.phi_max_slider.cget("to") == 90.0
    assert "<ButtonRelease-1>" in view_state.tth_min_slider.bindings
    assert "<Return>" in view_state.tth_min_entry.bindings
    assert "<FocusOut>" in view_state.phi_max_entry.bindings

    view_state.tth_min_slider.command("3.0")
    view_state.phi_max_slider.command("17.5")
    assert slider_calls == [("tth-min", 3.0), ("phi-max", 17.5)]

    view_state.tth_min_entry.bindings["<Return>"](None)
    view_state.phi_max_entry.bindings["<FocusOut>"](None)
    assert apply_calls == [
        ("1.5000", 1.5, 0.0, 90.0),
        ("18.7000", 18.7, -90.0, 90.0),
    ]

    view_state.tth_min_var.set(4.0)
    view_state.tth_min_slider.bindings["<ButtonRelease-1>"][0](None)
    assert slider_calls[-1] == ("tth-min", 4.0)


def test_analysis_export_controls_store_refs_and_commands(monkeypatch) -> None:
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)

    view_state = state.AnalysisExportControlsViewState()
    calls = []

    views.create_analysis_export_controls(
        parent=object(),
        view_state=view_state,
        on_save_snapshot=lambda: calls.append("snapshot"),
        on_save_q_space=lambda: calls.append("q-space"),
        on_save_1d_grid=lambda: calls.append("grid"),
        save_1d_grid_available=False,
    )

    assert view_state.snapshot_button is _FakeButton.created[0]
    assert view_state.save_q_button is _FakeButton.created[1]
    assert view_state.save_1d_grid_button is _FakeButton.created[2]
    assert [button.kwargs["text"] for button in _FakeButton.created] == [
        "Save 1D Snapshot",
        "Save Q-Space Snapshot",
        "Save 1D Grid (Unavailable)",
    ]
    assert view_state.save_1d_grid_button.state == tk.DISABLED

    for button in _FakeButton.created:
        if button.state != tk.DISABLED:
            button.command()
    assert calls == ["snapshot", "q-space"]


def test_background_backend_debug_controls_store_status_label_and_commands(
    monkeypatch,
) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)

    view_state = state.BackgroundBackendDebugViewState()
    calls = []

    views.create_background_backend_debug_controls(
        parent=object(),
        view_state=view_state,
        status_text="k=3 flip_x=False flip_y=False",
        on_rotate_minus_90=lambda: calls.append("rot-"),
        on_rotate_plus_90=lambda: calls.append("rot+"),
        on_flip_x=lambda: calls.append("flip-x"),
        on_flip_y=lambda: calls.append("flip-y"),
        on_reset=lambda: calls.append("reset"),
    )

    assert isinstance(view_state.background_backend_frame, _FakeFrame)
    assert view_state.background_backend_frame.kwargs["text"] == "Background Backend (debug)"
    assert view_state.background_backend_status_label.text == "k=3 flip_x=False flip_y=False"

    views.set_background_backend_status_text(view_state, "updated")
    assert view_state.background_backend_status_label.text == "updated"

    for button in _FakeButton.created:
        button.command()
    assert calls == ["rot-", "rot+", "flip-x", "flip-y", "reset"]


def test_backend_orientation_debug_controls_store_vars_and_reset(monkeypatch) -> None:
    _FakeButton.created = []
    _FakeLabel.created = []
    _FakeCheckbutton.created = []
    _FakeSpinbox.created = []
    _FakeOptionMenu.created = []
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "Spinbox", _FakeSpinbox)
    monkeypatch.setattr(views.tk, "OptionMenu", _FakeOptionMenu)
    monkeypatch.setattr(views.tk, "IntVar", _FakeVar)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.BackgroundBackendDebugViewState()

    views.create_backend_orientation_debug_controls(
        parent=object(),
        view_state=view_state,
        rotation_value=2,
        flip_y_axis=True,
        flip_x_axis=True,
        flip_order="xy",
    )

    assert isinstance(view_state.backend_orientation_frame, _FakeFrame)
    assert view_state.backend_orientation_frame.kwargs["text"] == "Backend orientation (debug)"
    assert view_state.backend_rotation_var.get() == 2
    assert view_state.backend_flip_y_axis_var.get() is True
    assert view_state.backend_flip_x_axis_var.get() is True
    assert view_state.backend_flip_order_var.get() == "xy"
    assert _FakeSpinbox.created[0].kwargs["textvariable"] is view_state.backend_rotation_var
    assert _FakeOptionMenu.created[0].variable is view_state.backend_flip_order_var
    assert _FakeOptionMenu.created[0].default == "xy"

    views.reset_backend_orientation_debug_controls(view_state)
    assert view_state.backend_rotation_var.get() == 0
    assert view_state.backend_flip_y_axis_var.get() is False
    assert view_state.backend_flip_x_axis_var.get() is False
    assert view_state.backend_flip_order_var.get() == "yx"

    view_state.backend_rotation_var.set(3)
    view_state.backend_flip_y_axis_var.set(True)
    view_state.backend_flip_x_axis_var.set(True)
    view_state.backend_flip_order_var.set("xy")
    _FakeButton.created[-1].command()
    assert view_state.backend_rotation_var.get() == 0
    assert view_state.backend_flip_y_axis_var.get() is False
    assert view_state.backend_flip_x_axis_var.get() is False
    assert view_state.backend_flip_order_var.get() == "yx"


def test_create_geometry_fit_constraints_panel_stores_refs_and_binds_handlers(
    monkeypatch,
) -> None:
    _FakeLabel.created = []
    _FakeScrollbar.created = []
    _FakeCanvas.created = []
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.tk, "Canvas", _FakeCanvas)

    root = _FakeRoot()
    view_state = state.GeometryFitConstraintsViewState(
        controls={"old": {"row": object()}},
    )
    after_marker = object()
    wheel_events = []

    views.create_geometry_fit_constraints_panel(
        parent=object(),
        root=root,
        view_state=view_state,
        after=after_marker,
        on_mousewheel=lambda event: wheel_events.append(event),
    )

    assert isinstance(view_state.panel, _FakeCollapsibleFrame)
    assert view_state.panel.text == "Geometry Fit Constraints"
    assert view_state.panel.expanded is True
    assert view_state.panel.pack_calls == [
        {
            "side": tk.TOP,
            "fill": tk.X,
            "padx": 5,
            "pady": 5,
            "after": after_marker,
        }
    ]
    assert isinstance(view_state.canvas, _FakeCanvas)
    assert isinstance(view_state.body, _FakeFrame)
    assert view_state.body_window == "body-window"
    assert view_state.controls == {}
    assert _FakeLabel.created[0].text.startswith("Each window is applied")
    assert [call[0] for call in root.bind_all_calls] == [
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
    ]

    views.set_geometry_fit_constraint_control(
        view_state,
        name="theta_initial",
        row="row",
        window_var="window",
        pull_var="pull",
    )
    assert view_state.controls["theta_initial"]["window_var"] == "window"

    view_state.body.bindings["<Configure>"](None)
    assert view_state.canvas.scrollregion == (0, 0, 100, 200)

    resize_event = type("Event", (), {"width": 321})()
    view_state.canvas.bindings["<Configure>"](resize_event)
    assert view_state.canvas.itemconfigure_calls[-1] == ("body-window", {"width": 321})


def test_geometry_fit_constraints_scroll_registration_preempts_outer_fit_canvas(
    monkeypatch,
) -> None:
    _FakeScrollbar.created = []
    _FakeCanvas.created = []
    monkeypatch.setattr(views.ttk, "Panedwindow", _FakePanedwindow)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Notebook", _FakeNotebook)
    monkeypatch.setattr(views.ttk, "Scrollbar", _FakeScrollbar)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.ttk, "Separator", _FakeFrame)
    monkeypatch.setattr(views.tk, "Canvas", _FakeCanvas)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)
    monkeypatch.setattr(views, "CollapsibleFrame", _FakeCollapsibleFrame)

    root = _FakeRoot()
    shell_view_state = state.AppShellViewState()
    views.create_app_shell(root=root, view_state=shell_view_state)

    for canvas in (
        shell_view_state.workspace_canvas,
        shell_view_state.parameter_geometry_canvas,
        shell_view_state.parameter_structure_canvas,
    ):
        canvas.rootx = 1000
        canvas.rooty = 1000
        canvas.width = 50
        canvas.height = 50

    shell_view_state.fit_canvas.rootx = 0
    shell_view_state.fit_canvas.rooty = 0
    shell_view_state.fit_canvas.width = 400
    shell_view_state.fit_canvas.height = 300

    constraints_view_state = state.GeometryFitConstraintsViewState()
    views.create_geometry_fit_constraints_panel(
        parent=object(),
        root=root,
        view_state=constraints_view_state,
        on_mousewheel=lambda event: views.scroll_geometry_fit_constraints_canvas(
            constraints_view_state,
            pointer_x=event.x_root,
            pointer_y=event.y_root,
            event=event,
        ),
    )

    constraints_view_state.canvas.rootx = 40
    constraints_view_state.canvas.rooty = 30
    constraints_view_state.canvas.width = 180
    constraints_view_state.canvas.height = 120

    dispatch = next(
        callback
        for sequence, callback, _add in root.bind_all_calls
        if sequence == "<MouseWheel>"
    )
    event = type("Event", (), {"delta": 120, "num": None, "x_root": 80, "y_root": 60})()

    assert dispatch(event) == "break"
    assert constraints_view_state.canvas.scrolled == [(-1, "units")]
    assert shell_view_state.fit_canvas.scrolled == []


def test_scroll_geometry_fit_constraints_canvas_only_when_pointer_is_inside() -> None:
    canvas = _FakeCanvas()
    canvas.rootx = 50
    canvas.rooty = 75
    canvas.width = 200
    canvas.height = 120
    view_state = state.GeometryFitConstraintsViewState(canvas=canvas)

    inside_event = type("Event", (), {"delta": 60, "num": None})()
    assert (
        views.scroll_geometry_fit_constraints_canvas(
            view_state,
            pointer_x=100,
            pointer_y=100,
            event=inside_event,
        )
        == "break"
    )
    assert canvas.scrolled == [(-1, "units")]

    outside_event = type("Event", (), {"delta": 120, "num": None})()
    assert (
        views.scroll_geometry_fit_constraints_canvas(
            view_state,
            pointer_x=500,
            pointer_y=500,
            event=outside_event,
        )
        is None
    )
    assert canvas.scrolled == [(-1, "units")]


def test_create_background_theta_controls_stores_vars_and_binds_apply(monkeypatch) -> None:
    _FakeLabel.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.BackgroundThetaControlsViewState()
    applied = []

    views.create_background_theta_controls(
        parent=object(),
        view_state=view_state,
        background_theta_values_text="4, 7.5",
        geometry_theta_offset_text="0.25",
        on_apply=lambda: applied.append("apply"),
    )

    assert isinstance(view_state.background_theta_controls, _FakeFrame)
    assert view_state.background_theta_controls.kwargs["text"] == "Background Theta_i"
    assert isinstance(view_state.background_theta_entry, _FakeEntry)
    assert isinstance(view_state.background_theta_offset_entry, _FakeEntry)
    assert view_state.background_theta_list_var.get() == "4, 7.5"
    assert view_state.geometry_theta_offset_var.get() == "0.25"
    assert _FakeLabel.created[0].text == "Per-background theta_i values (deg, in load order)"

    view_state.background_theta_entry.bindings["<Return>"](None)
    view_state.background_theta_offset_entry.bindings["<Return>"](None)
    _FakeButton.created[-1].command()
    assert applied == ["apply", "apply", "apply"]


def test_create_geometry_fit_background_controls_stores_var_and_binds_apply(
    monkeypatch,
) -> None:
    _FakeLabel.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "LabelFrame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    view_state = state.BackgroundThetaControlsViewState()
    applied = []
    shortcut_events = []

    views.create_geometry_fit_background_controls(
        parent=object(),
        view_state=view_state,
        selection_text="all",
        on_apply=lambda: applied.append("apply"),
        on_select_current=lambda: shortcut_events.append("current"),
        on_select_all=lambda: shortcut_events.append("all"),
    )

    assert isinstance(view_state.geometry_fit_background_controls, _FakeFrame)
    assert (
        view_state.geometry_fit_background_controls.kwargs["text"]
        == "Geometry Fit Backgrounds"
    )
    assert isinstance(view_state.geometry_fit_background_entry, _FakeEntry)
    assert isinstance(view_state.geometry_fit_background_shortcuts_frame, _FakeFrame)
    assert isinstance(view_state.geometry_fit_background_rows_frame, _FakeFrame)
    assert view_state.geometry_fit_background_selection_var.get() == "all"
    assert (
        _FakeLabel.created[0].text
        == "Choose which loaded backgrounds participate in geometry fitting. The saved selection still uses the canonical current/all/index list format."
    )

    view_state.geometry_fit_background_entry.bindings["<Return>"](None)
    _FakeButton.created[0].command()
    _FakeButton.created[1].command()
    assert applied == ["apply"]
    assert shortcut_events == ["current", "all"]


def test_populate_geometry_fit_background_table_updates_rows(monkeypatch) -> None:
    _FakeLabel.created = []
    _FakeCheckbutton.created = []
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.tk, "BooleanVar", _FakeVar)

    toggled = []
    view_state = state.BackgroundThetaControlsViewState(
        geometry_fit_background_rows_frame=_FakeFrame(object())
    )
    views.populate_geometry_fit_background_table(
        view_state=view_state,
        row_count=2,
        on_toggle=lambda idx: toggled.append(idx),
    )
    views.update_geometry_fit_background_table(
        view_state=view_state,
        rows=[
            {"background": "bg0.osc", "theta_i": "4.0000 deg", "pairs": "3"},
            {"background": "bg1.osc", "theta_i": "7.5000 deg", "pairs": "0"},
        ],
        selected_indices=[1],
        current_index=0,
    )

    assert len(view_state.geometry_fit_background_include_vars) == 2
    assert view_state.geometry_fit_background_active_labels[0].text == ">"
    assert view_state.geometry_fit_background_active_labels[1].text == ""
    assert view_state.geometry_fit_background_include_vars[0].get() is False
    assert view_state.geometry_fit_background_include_vars[1].get() is True
    assert view_state.geometry_fit_background_name_labels[0].text == "bg0.osc"
    assert view_state.geometry_fit_background_theta_labels[1].text == "7.5000 deg"
    assert view_state.geometry_fit_background_pair_labels[0].text == "3"

    view_state.geometry_fit_background_include_checks[1].command()
    assert toggled == [1]


def test_populate_app_shell_view_switcher_creates_radios(monkeypatch) -> None:
    _FakeRadiobutton.created = []
    monkeypatch.setattr(views.ttk, "Radiobutton", _FakeRadiobutton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    events = []
    view_state = state.AppShellViewState(
        view_switcher_frame=_FakeFrame(object()),
        view_mode_var=_FakeStringVar("detector"),
    )
    views.populate_app_shell_view_switcher(
        view_state=view_state,
        on_select=lambda mode: events.append(mode),
    )

    assert [radio.value for radio in _FakeRadiobutton.created] == [
        "detector",
        "caked",
    ]
    _FakeRadiobutton.created[1].command()
    assert events == ["caked"]


def test_populate_app_shell_quick_controls_builds_linked_inputs(monkeypatch) -> None:
    _FakeLabel.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(views.ttk, "Scale", _FakeScale)
    monkeypatch.setattr(views.ttk, "Separator", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)
    monkeypatch.setattr(views.tk, "StringVar", _FakeStringVar)

    changed = []
    variable = _FakeVar(4.0)
    source_scale = _FakeScale(object(), from_=0.0, to=10.0)
    view_state = state.AppShellViewState(
        quick_controls_body=_FakeFrame(object()),
    )

    views.populate_app_shell_quick_controls(
        view_state=view_state,
        controls=[
            {
                "key": "theta_initial",
                "label": "theta",
                "variable": variable,
                "scale": source_scale,
                "command": lambda: changed.append(variable.get()),
                "step": 0.01,
            }
        ],
        on_more_controls=lambda: changed.append("more"),
    )

    assert "theta_initial" in view_state.quick_control_widgets
    control = view_state.quick_control_widgets["theta_initial"]
    assert isinstance(control["scale"], _FakeScale)
    control["entry_var"].set("5.25")
    control["entry"].bindings["<Return>"](None)
    assert variable.get() == 5.25
    assert changed == [5.25]
    assert isinstance(view_state.quick_controls_more_button, _FakeButton)
    view_state.quick_controls_more_button.command()
    assert changed == [5.25, "more"]


def test_populate_app_shell_quick_controls_supports_choice_controls(
    monkeypatch,
) -> None:
    _FakeLabel.created = []
    _FakeOptionMenu.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "OptionMenu", _FakeOptionMenu)

    variable = _FakeStringVar("High")
    view_state = state.AppShellViewState(
        quick_controls_body=_FakeFrame(object()),
    )

    views.populate_app_shell_quick_controls(
        view_state=view_state,
        controls=[
            {
                "key": "sampling_resolution",
                "label": "sampling resolution",
                "control_type": "choice",
                "variable": variable,
                "options": ("Low", "High", "Custom"),
            }
        ],
    )

    assert "sampling_resolution" in view_state.quick_control_widgets
    control = view_state.quick_control_widgets["sampling_resolution"]
    assert isinstance(control["menu"], _FakeOptionMenu)
    assert control["options"] == ("Low", "High", "Custom")
    assert control["variable"] is variable
    assert control["menu"].variable is variable
    assert control["menu"].default == "High"
    assert control["menu"].values == ("Low", "High", "Custom")


def test_populate_app_shell_quick_controls_supports_check_and_button_controls(
    monkeypatch,
) -> None:
    _FakeLabel.created = []
    _FakeCheckbutton.created = []
    _FakeButton.created = []
    monkeypatch.setattr(views.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(views.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(views.ttk, "Checkbutton", _FakeCheckbutton)
    monkeypatch.setattr(views.ttk, "Button", _FakeButton)

    events = []
    log_var = _FakeVar(False)
    view_state = state.AppShellViewState(
        quick_controls_body=_FakeFrame(object()),
    )

    views.populate_app_shell_quick_controls(
        view_state=view_state,
        controls=[
            {
                "key": "log_radial",
                "label": "Log radial",
                "control_type": "check",
                "variable": log_var,
                "command": lambda: events.append(("log", log_var.get())),
            },
            {
                "key": "auto_match_scale",
                "label": "Auto-Match Scale (Radial Peak)",
                "control_type": "button",
                "command": lambda: events.append("auto-match"),
            },
        ],
    )

    assert "log_radial" in view_state.quick_control_widgets
    log_control = view_state.quick_control_widgets["log_radial"]
    assert isinstance(log_control["checkbutton"], _FakeCheckbutton)
    assert log_control["variable"] is log_var

    log_var.set(True)
    log_control["checkbutton"].command()

    assert "auto_match_scale" in view_state.quick_control_widgets
    button_control = view_state.quick_control_widgets["auto_match_scale"]
    assert isinstance(button_control["button"], _FakeButton)
    button_control["button"].command()

    assert events == [("log", True), "auto-match"]
    assert view_state.quick_controls_more_button is None
