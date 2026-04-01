from ra_sim.gui import sliders as gui_sliders


class _FakeVar:
    def __init__(self, value=None) -> None:
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


class _FakeFrame:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.children = []
        if hasattr(parent, "children"):
            parent.children.append(self)

    def pack(self, **_kwargs) -> None:
        pass

    def columnconfigure(self, _index: int, weight: int) -> None:
        self.column_weight = weight

    def winfo_children(self):
        return list(self.children)


class _FakeLabel:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        if hasattr(parent, "children"):
            parent.children.append(self)

    def pack(self, **_kwargs) -> None:
        pass


class _FakeEntry:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.textvariable = kwargs.get("textvariable")
        self.bindings = {}
        if hasattr(parent, "children"):
            parent.children.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback

    def get(self) -> str:
        if self.textvariable is None:
            return ""
        return str(self.textvariable.get())


class _FakeScale:
    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.variable = kwargs.get("variable")
        self.command = kwargs.get("command")
        self.bindings = {}
        self.focused = False
        self.bounds = {
            "from": kwargs.get("from_"),
            "to": kwargs.get("to"),
        }
        if hasattr(parent, "children"):
            parent.children.append(self)

    def grid(self, **_kwargs) -> None:
        pass

    def bind(self, event: str, callback) -> None:
        self.bindings[event] = callback

    def focus_set(self) -> None:
        self.focused = True

    def cget(self, key: str):
        return self.bounds[key]

    def configure(self, **kwargs) -> None:
        if "from_" in kwargs:
            self.bounds["from"] = kwargs["from_"]
        if "to" in kwargs:
            self.bounds["to"] = kwargs["to"]

    def set(self, value) -> None:
        if self.variable is not None:
            self.variable.set(float(value))
        if self.command is not None:
            self.command(str(value))


def _patch_slider_widgets(monkeypatch) -> None:
    monkeypatch.setattr(gui_sliders.ttk, "Frame", _FakeFrame)
    monkeypatch.setattr(gui_sliders.ttk, "Label", _FakeLabel)
    monkeypatch.setattr(gui_sliders.ttk, "Scale", _FakeScale)
    monkeypatch.setattr(gui_sliders.ttk, "Entry", _FakeEntry)
    monkeypatch.setattr(gui_sliders.tk, "DoubleVar", _FakeVar)
    monkeypatch.setattr(gui_sliders.tk, "StringVar", _FakeVar)


def test_create_slider_supports_keyboard_and_wheel_stepping(monkeypatch) -> None:
    _patch_slider_widgets(monkeypatch)

    updates = []
    parent = _FakeFrame(None)
    slider_var, slider = gui_sliders.create_slider(
        "Test Slider",
        0.0,
        10.0,
        5.0,
        0.5,
        parent,
        update_callback=lambda: updates.append("update"),
    )

    assert slider.bindings["<Button-1>"](type("Event", (), {})()) is None
    assert slider.focused is True

    assert (
        slider.bindings["<KeyPress>"](
            type("Event", (), {"keysym": "Right", "state": 0})()
        )
        == "break"
    )
    assert slider_var.get() == 5.5
    assert updates == ["update"]

    assert (
        slider.bindings["<MouseWheel>"](
            type("Event", (), {"delta": 120, "num": None, "state": 1})()
        )
        == "break"
    )
    assert slider_var.get() == 6.0
    assert updates == ["update", "update"]


def test_create_slider_handles_small_wheel_delta_and_home_end_keys(monkeypatch) -> None:
    _patch_slider_widgets(monkeypatch)

    parent = _FakeFrame(None)
    slider_var, slider = gui_sliders.create_slider(
        "Bounded Slider",
        0.0,
        2.0,
        1.0,
        0.25,
        parent,
    )

    assert (
        slider.bindings["<MouseWheel>"](
            type("Event", (), {"delta": 60, "num": None, "state": 0})()
        )
        is None
    )
    assert slider_var.get() == 1.0

    assert (
        slider.bindings["<MouseWheel>"](
            type("Event", (), {"delta": 60, "num": None, "state": 1})()
        )
        == "break"
    )
    assert slider_var.get() == 1.25

    assert (
        slider.bindings["<KeyPress>"](
            type("Event", (), {"keysym": "End", "state": 0})()
        )
        == "break"
    )
    assert slider_var.get() == 2.0

    assert (
        slider.bindings["<KeyPress>"](
            type("Event", (), {"keysym": "Home", "state": 0})()
        )
        == "break"
    )
    assert slider_var.get() == 0.0


def test_create_slider_triggers_final_update_on_mouse_release(monkeypatch) -> None:
    _patch_slider_widgets(monkeypatch)

    updates = []
    parent = _FakeFrame(None)
    slider_var, slider = gui_sliders.create_slider(
        "Release Slider",
        0.0,
        10.0,
        5.0,
        0.5,
        parent,
        update_callback=lambda: updates.append("update"),
    )

    slider.bindings["<Button-1>"](type("Event", (), {})())
    slider.command("6.5")
    updates.clear()

    assert slider.bindings["<ButtonRelease-1>"](type("Event", (), {})()) is None
    assert slider_var.get() == 6.5
    assert updates == ["update"]
