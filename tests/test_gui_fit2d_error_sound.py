from ra_sim.gui import fit2d_error_sound


class _FakeVar:
    def __init__(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _FakeRoot:
    def __init__(self) -> None:
        self.bind_all_calls = []

    def bind_all(self, sequence: str, callback, add=None) -> None:
        self.bind_all_calls.append((sequence, callback, add))


class _FakeEntry:
    def __init__(
        self,
        *,
        insert_index: int = 0,
        selection_present: bool = False,
        state: str = "normal",
        class_name: str = "TEntry",
    ) -> None:
        self.insert_index = insert_index
        self._selection_present = selection_present
        self._state = state
        self._class_name = class_name

    def winfo_class(self) -> str:
        return self._class_name

    def cget(self, key: str):
        if key == "state":
            return self._state
        raise KeyError(key)

    def selection_present(self) -> bool:
        return self._selection_present

    def index(self, name: str):
        if name == "insert":
            return self.insert_index
        raise ValueError(name)


class _FakeText:
    def __init__(
        self,
        *,
        insert_index: str = "1.0",
        selection_present: bool = False,
        state: str = "normal",
    ) -> None:
        self.insert_index = insert_index
        self._selection_present = selection_present
        self._state = state

    def winfo_class(self) -> str:
        return "Text"

    def cget(self, key: str):
        if key == "state":
            return self._state
        raise KeyError(key)

    def tag_ranges(self, name: str):
        if name != "sel":
            return ()
        if self._selection_present:
            return ("1.0", "1.1")
        return ()

    def index(self, name: str):
        if name == "insert":
            return self.insert_index
        raise ValueError(name)


def test_widget_backspace_would_underflow_for_entry_only_at_start() -> None:
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeEntry(insert_index=0)
    ) is True
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeEntry(insert_index=1)
    ) is False
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeEntry(insert_index=0, selection_present=True)
    ) is False
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeEntry(insert_index=0, state="readonly")
    ) is False


def test_widget_backspace_would_underflow_for_text_only_at_absolute_start() -> None:
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeText(insert_index="1.0")
    ) is True
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeText(insert_index="2.0")
    ) is False
    assert fit2d_error_sound.widget_backspace_would_underflow(
        _FakeText(insert_index="1.0", selection_present=True)
    ) is False


def test_bind_fit2d_backspace_error_sound_queues_only_for_enabled_underflow(
    monkeypatch,
) -> None:
    queued = []
    root = _FakeRoot()
    enabled_var = _FakeVar(True)

    monkeypatch.setattr(
        fit2d_error_sound,
        "queue_fit2d_error_sound",
        lambda **kwargs: queued.append(kwargs) or 1,
    )

    assert (
        fit2d_error_sound.bind_fit2d_backspace_error_sound(
            root,
            enabled_var=enabled_var,
        )
        is True
    )
    assert root.bind_all_calls[0][0] == "<KeyPress-BackSpace>"
    assert root.bind_all_calls[0][2] == "+"

    callback = root.bind_all_calls[0][1]
    callback(type("Event", (), {"widget": _FakeEntry(insert_index=1)})())
    callback(type("Event", (), {"widget": _FakeEntry(insert_index=0)})())
    enabled_var.value = False
    callback(type("Event", (), {"widget": _FakeEntry(insert_index=0)})())

    assert queued == [{"repeat": 1, "bell_callback": None}]
