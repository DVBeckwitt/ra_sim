import numpy as np

from ra_sim.gui import bragg_qr_manager, controllers, state


class _FakeTclError(Exception):
    pass


class _FakeListbox:
    def __init__(self, selected=None, *, error=False):
        self._selected = list(selected or [])
        self._error = bool(error)

    def curselection(self):
        if self._error:
            raise _FakeTclError("selection failed")
        return tuple(self._selected)


def test_selected_bragg_qr_manager_helpers_map_selection_and_fallback() -> None:
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("primary", 1), ("secondary", 2)],
        l_index_keys=[10, 20],
        selected_group_key=("primary", 7),
    )

    assert bragg_qr_manager.selected_bragg_qr_window_keys(
        manager_state,
        _FakeListbox([1]),
    ) == [("secondary", 2)]
    assert bragg_qr_manager.selected_bragg_qr_l_window_keys(
        manager_state,
        _FakeListbox([0, 1]),
    ) == [10, 20]
    assert bragg_qr_manager.selected_primary_bragg_qr_window_key(
        manager_state,
        selected_keys=[("secondary", 2)],
    ) == ("secondary", 2)
    assert bragg_qr_manager.selected_primary_bragg_qr_window_key(manager_state) == (
        "primary",
        7,
    )

    assert bragg_qr_manager.selected_bragg_qr_window_keys(
        manager_state,
        _FakeListbox(error=True),
        tcl_error_types=(_FakeTclError,),
    ) == []
    assert bragg_qr_manager.selected_bragg_qr_l_window_keys(
        manager_state,
        _FakeListbox(error=True),
        tcl_error_types=(_FakeTclError,),
    ) == []


def test_refresh_bragg_qr_toggle_window_updates_view_and_index_keys(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(window=object(), qr_listbox=_FakeListbox([1]))
    manager_state = state.BraggQrManagerState(selected_group_key=("primary", 1))
    calls = {}

    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "bragg_qr_manager_window_open",
        lambda _view_state: True,
    )
    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "refresh_bragg_qr_manager_qr_list",
        lambda **kwargs: calls.setdefault("qr", kwargs),
    )

    ok = bragg_qr_manager.refresh_bragg_qr_toggle_window(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        entries=[
            {
                "key": ("primary", 1),
                "source": "primary",
                "m": 1,
                "qr": 1.2,
                "hk_preview": "(1 0)",
            },
            {
                "key": ("secondary", 2),
                "source": "secondary",
                "m": 2,
                "qr": 2.3,
                "hk_preview": "(1 1)",
            },
        ],
    )

    assert ok is True
    assert manager_state.qr_index_keys == [("primary", 1), ("secondary", 2)]
    assert calls["qr"]["selected_indices"] == [0]
    assert calls["qr"]["status_text"] == "Enabled: 2 / 2"
    assert calls["qr"]["see_index"] == 0


def test_refresh_bragg_qr_l_toggle_listbox_updates_view_and_selected_group(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(
        window=object(),
        qr_listbox=_FakeListbox([1]),
        l_listbox=_FakeListbox([0]),
    )
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("primary", 1), ("secondary", 2)],
        disabled_groups={("secondary", 2)},
    )
    calls = {}

    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "bragg_qr_manager_window_open",
        lambda _view_state: True,
    )
    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "refresh_bragg_qr_manager_l_list",
        lambda **kwargs: calls.setdefault("l", kwargs),
    )

    ok = bragg_qr_manager.refresh_bragg_qr_l_toggle_listbox(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        l_listbox=view_state.l_listbox,
        build_l_value_map=lambda source, m_idx: {
            controllers.bragg_qr_l_value_to_key(0.25): 0.25,
            controllers.bragg_qr_l_value_to_key(0.5): 0.5,
        }
        if (source, m_idx) == ("secondary", 2)
        else {},
    )

    assert ok is True
    assert manager_state.selected_group_key == ("secondary", 2)
    assert manager_state.l_index_keys == [
        controllers.bragg_qr_l_value_to_key(0.25),
        controllers.bragg_qr_l_value_to_key(0.5),
    ]
    assert calls["l"]["selected_indices"] == []
    assert calls["l"]["status_text"] == (
        "Selected: secondary m=2 | Enabled L: 0 / 2 | Qr group disabled"
    )


def test_bragg_qr_manager_selection_change_updates_selected_group_and_refreshes_l(
    monkeypatch,
) -> None:
    view_state = state.BraggQrManagerViewState(
        window=object(),
        qr_listbox=_FakeListbox([1]),
        l_listbox=_FakeListbox([0]),
    )
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("primary", 1), ("secondary", 2)],
        selected_group_key=("primary", 1),
    )
    calls = {}

    monkeypatch.setattr(
        bragg_qr_manager,
        "refresh_bragg_qr_l_toggle_listbox",
        lambda **kwargs: (calls.setdefault("refresh", kwargs), True)[1],
    )

    ok = bragg_qr_manager.on_bragg_qr_selection_changed(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        l_listbox=view_state.l_listbox,
        build_l_value_map=lambda _source, _m_idx: {},
    )

    assert ok is True
    assert manager_state.selected_group_key == ("secondary", 2)
    assert calls["refresh"]["build_l_value_map"]("primary", 1) == {}


def test_bragg_qr_manager_group_action_helpers_apply_filters_and_status(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(qr_listbox=_FakeListbox([1]))
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("primary", 1), ("secondary", 2)],
    )
    status_calls = []
    applied = []
    progress = []
    refreshed = []

    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "set_bragg_qr_manager_status_text",
        lambda _view_state, **kwargs: status_calls.append(kwargs),
    )

    ok = bragg_qr_manager.disable_selected_bragg_qr_groups(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        refresh_window=lambda: refreshed.append("refresh"),
        apply_filters=lambda: applied.append("filters"),
        set_progress_text=progress.append,
    )

    assert ok is True
    assert manager_state.disabled_groups == {("secondary", 2)}
    assert applied == ["filters"]
    assert progress == ["Disabled 1 Bragg Qr group(s)."]
    assert refreshed == []
    assert status_calls == []

    view_state.qr_listbox = _FakeListbox([])
    ok = bragg_qr_manager.enable_selected_bragg_qr_groups(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        refresh_window=lambda: refreshed.append("refresh"),
        apply_filters=lambda: applied.append("filters"),
        set_progress_text=progress.append,
    )

    assert ok is False
    assert status_calls[-1] == {"qr_text": "Select one or more Qr groups."}


def test_bragg_qr_manager_l_action_helpers_handle_selection_and_enable_all(
    monkeypatch,
) -> None:
    view_state = state.BraggQrManagerViewState(
        qr_listbox=_FakeListbox([0]),
        l_listbox=_FakeListbox([0, 1]),
    )
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("secondary", 2)],
        l_index_keys=[
            controllers.bragg_qr_l_value_to_key(0.25),
            controllers.bragg_qr_l_value_to_key(0.5),
        ],
        selected_group_key=("secondary", 2),
        disabled_l_values={("secondary", 2, controllers.bragg_qr_l_value_to_key(0.25))},
    )
    status_calls = []
    applied = []
    progress = []
    refreshed = []

    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "set_bragg_qr_manager_status_text",
        lambda _view_state, **kwargs: status_calls.append(kwargs),
    )

    ok = bragg_qr_manager.disable_selected_bragg_qr_l_values(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        l_listbox=view_state.l_listbox,
        invalid_key=-999,
        refresh_window=lambda: refreshed.append("refresh"),
        apply_filters=lambda: applied.append("filters"),
        set_progress_text=progress.append,
    )

    assert ok is True
    assert manager_state.disabled_l_values == {
        ("secondary", 2, controllers.bragg_qr_l_value_to_key(0.25)),
        ("secondary", 2, controllers.bragg_qr_l_value_to_key(0.5)),
    }
    assert progress[-1] == "Disabled 2 L value(s) for secondary m=2."

    ok = bragg_qr_manager.enable_all_bragg_qr_l_values_for_selected_qr(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=view_state.qr_listbox,
        refresh_window=lambda: refreshed.append("refresh"),
        apply_filters=lambda: applied.append("filters"),
        set_progress_text=progress.append,
    )

    assert ok is True
    assert manager_state.disabled_l_values == set()
    assert progress[-1] == "Enabled all L values for secondary m=2."
    assert len(applied) == 2
    assert refreshed == []
    assert status_calls == []


def test_bragg_qr_manager_open_and_close_helpers_wrap_view_and_state(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(window=object())
    manager_state = state.BraggQrManagerState(
        qr_index_keys=[("primary", 1)],
        l_index_keys=[0],
        selected_group_key=("primary", 1),
    )
    calls = []

    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "open_bragg_qr_manager_window",
        lambda **kwargs: calls.append(("open", kwargs)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager.gui_views,
        "close_bragg_qr_manager_window",
        lambda _view_state: calls.append(("close", _view_state)),
    )

    opened = bragg_qr_manager.open_bragg_qr_toggle_window(
        root=object(),
        view_state=view_state,
        on_qr_selection_changed=lambda _event=None: None,
        on_toggle_qr=lambda _event=None: None,
        on_toggle_l=lambda _event=None: None,
        on_enable_selected_qr=lambda: None,
        on_disable_selected_qr=lambda: None,
        on_enable_all_qr=lambda: None,
        on_disable_all_qr=lambda: None,
        on_enable_selected_l=lambda: None,
        on_disable_selected_l=lambda: None,
        on_enable_all_l=lambda: None,
        on_disable_all_l=lambda: None,
        on_refresh=lambda: calls.append(("refresh", None)),
        on_close=lambda: None,
    )
    bragg_qr_manager.close_bragg_qr_toggle_window(view_state, manager_state)

    assert opened is True
    assert calls[0][0] == "open"
    assert calls[1] == ("refresh", None)
    assert calls[2] == ("close", view_state)
    assert manager_state.qr_index_keys == []
    assert manager_state.l_index_keys == []
    assert manager_state.selected_group_key is None


def test_bragg_qr_runtime_value_helpers_build_live_entries_and_overlay_data() -> None:
    simulation_runtime_state = state.SimulationRuntimeState(
        sim_primary_qr={"1": {"L": [0.0]}},
    )

    primary_a, secondary_a = bragg_qr_manager.current_runtime_bragg_qr_lattice_values(
        primary_candidate=lambda: "5.5",
        primary_fallback=4.0,
        secondary_candidate=lambda: "bad",
    )
    assert primary_a == 5.5
    assert secondary_a == 5.5

    primary_a, secondary_a = bragg_qr_manager.current_runtime_bragg_qr_lattice_values(
        primary_candidate=lambda: "5.5",
        primary_fallback=4.0,
        secondary_candidate=lambda: "6.5",
    )
    assert primary_a == 5.5
    assert secondary_a == 6.5

    entries = bragg_qr_manager.build_active_qr_cylinder_overlay_entries(
        simulation_runtime_state,
        primary_candidate=lambda: "5.0",
        primary_fallback=4.0,
        secondary_candidate=lambda: "6.0",
        primary_miller_all=np.asarray([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
        secondary_miller_all=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
    )

    assert entries == [
        {
            "key": ("primary", 1),
            "source": "primary",
            "m": 1,
            "qr": controllers.qr_value_for_m(1, 5.0),
        },
        {
            "key": ("secondary", 1),
            "source": "secondary",
            "m": 1,
            "qr": controllers.qr_value_for_m(1, 6.0),
        },
    ]


def test_bragg_qr_runtime_binding_builder_reads_live_lattice_values(monkeypatch) -> None:
    simulation_runtime_state = state.SimulationRuntimeState()
    lattice_values = {"primary": "5.0", "secondary": "6.0"}
    calls = []

    monkeypatch.setattr(
        bragg_qr_manager.gui_controllers,
        "build_bragg_qr_entries",
        lambda simulation_state, *, primary_a, secondary_a: (
            calls.append(("entries", simulation_state, primary_a, secondary_a)),
            [{"key": ("primary", 1)}],
        )[-1],
    )
    monkeypatch.setattr(
        bragg_qr_manager.gui_controllers,
        "build_bragg_qr_l_value_map",
        lambda simulation_state, source_label, m_idx: (
            calls.append(("l_map", simulation_state, source_label, m_idx)),
            {1: 0.25},
        )[-1],
    )

    bindings = bragg_qr_manager.build_runtime_bragg_qr_bindings(
        view_state=state.BraggQrManagerViewState(),
        manager_state=state.BraggQrManagerState(),
        simulation_runtime_state=simulation_runtime_state,
        primary_candidate=lambda: lattice_values["primary"],
        primary_fallback=4.0,
        secondary_candidate=lambda: lattice_values["secondary"],
        apply_filters=lambda: None,
        invalid_key=-999,
    )

    lattice_values["primary"] = "7.0"
    assert bindings.get_entries() == [{"key": ("primary", 1)}]
    assert calls[0] == ("entries", simulation_runtime_state, 7.0, 6.0)
    assert bindings.build_l_value_map("primary", 3) == {1: 0.25}
    assert calls[1] == ("l_map", simulation_runtime_state, "primary", 3)
    assert bindings.invalid_key == -999


def test_bragg_qr_runtime_callback_factories_build_zero_arg_delegates(
    monkeypatch,
) -> None:
    calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        bragg_qr_manager,
        "refresh_runtime_bragg_qr_toggle_window",
        lambda bindings: calls.append(("refresh", bindings)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "open_runtime_bragg_qr_toggle_window",
        lambda *, root, bindings: calls.append(("open", root, bindings)) or True,
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    refresh_callback = bragg_qr_manager.make_runtime_bragg_qr_refresh_callback(
        build_bindings
    )
    open_callback = bragg_qr_manager.make_runtime_bragg_qr_open_callback(
        root="root-window",
        bindings_factory=build_bindings,
    )

    assert refresh_callback() is True
    assert open_callback() is True
    assert calls == [
        ("refresh", "bindings-1"),
        ("open", "root-window", "bindings-2"),
    ]


def test_bragg_qr_manager_runtime_refresh_uses_shared_bindings(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(
        qr_listbox=_FakeListbox([0]),
        l_listbox=_FakeListbox([0]),
    )
    manager_state = state.BraggQrManagerState()
    calls = []
    bindings = bragg_qr_manager.BraggQrRuntimeBindings(
        view_state=view_state,
        manager_state=manager_state,
        get_entries=lambda: [{"key": ("primary", 1)}],
        build_l_value_map=lambda source, m_idx: {1: 0.25},
        apply_filters=lambda: calls.append(("filters", None)),
        set_progress_text=lambda text: calls.append(("progress", text)),
        invalid_key=-999,
        tcl_error_types=(_FakeTclError,),
    )

    monkeypatch.setattr(
        bragg_qr_manager,
        "refresh_bragg_qr_toggle_window",
        lambda **kwargs: calls.append(("refresh", kwargs)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "on_bragg_qr_selection_changed",
        lambda **kwargs: calls.append(("selection", kwargs)) or True,
    )

    ok = bragg_qr_manager.refresh_runtime_bragg_qr_toggle_window(bindings)

    assert ok is True
    assert calls[0][0] == "refresh"
    assert calls[0][1]["entries"] == [{"key": ("primary", 1)}]
    assert calls[0][1]["tcl_error_types"] == (_FakeTclError,)
    assert calls[1][0] == "selection"
    assert calls[1][1]["build_l_value_map"]("primary", 1) == {1: 0.25}


def test_bragg_qr_manager_runtime_actions_delegate_with_shared_refresh(monkeypatch) -> None:
    view_state = state.BraggQrManagerViewState(
        qr_listbox=_FakeListbox([0]),
        l_listbox=_FakeListbox([0]),
    )
    manager_state = state.BraggQrManagerState()
    calls = []
    bindings = bragg_qr_manager.BraggQrRuntimeBindings(
        view_state=view_state,
        manager_state=manager_state,
        get_entries=lambda: [{"key": ("primary", 1)}],
        build_l_value_map=lambda source, m_idx: {1: 0.25},
        apply_filters=lambda: calls.append(("filters", None)),
        set_progress_text=lambda text: calls.append(("progress", text)),
        invalid_key=-999,
        tcl_error_types=(_FakeTclError,),
    )

    monkeypatch.setattr(
        bragg_qr_manager,
        "refresh_runtime_bragg_qr_toggle_window",
        lambda bindings_arg: calls.append(("refresh_runtime", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_selected_bragg_qr_groups",
        lambda **kwargs: (
            calls.append(("disable_groups", kwargs)),
            kwargs["refresh_window"](),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_all_bragg_qr_l_values_for_selected_qr",
        lambda **kwargs: (
            calls.append(("disable_all_l", kwargs)),
            kwargs["refresh_window"](),
            True,
        )[-1],
    )

    ok_groups = bragg_qr_manager.disable_selected_bragg_qr_groups_runtime(bindings)
    ok_l = bragg_qr_manager.disable_all_bragg_qr_l_values_for_selected_qr_runtime(
        bindings
    )

    assert ok_groups is True
    assert ok_l is True
    assert calls[0][0] == "disable_groups"
    assert calls[0][1]["qr_listbox"] is view_state.qr_listbox
    assert calls[0][1]["set_progress_text"] is bindings.set_progress_text
    assert calls[1] == ("refresh_runtime", bindings)
    assert calls[2][0] == "disable_all_l"
    assert calls[2][1]["build_l_value_map"]("primary", 1) == {1: 0.25}
    assert calls[3] == ("refresh_runtime", bindings)


def test_bragg_qr_manager_open_runtime_helper_wires_callbacks_from_bindings(
    monkeypatch,
) -> None:
    bindings = bragg_qr_manager.BraggQrRuntimeBindings(
        view_state=state.BraggQrManagerViewState(),
        manager_state=state.BraggQrManagerState(),
        get_entries=lambda: [],
        build_l_value_map=lambda _source, _m_idx: {},
        apply_filters=lambda: None,
    )
    calls = []

    monkeypatch.setattr(
        bragg_qr_manager,
        "open_bragg_qr_toggle_window",
        lambda **kwargs: calls.append(("open", kwargs)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "on_runtime_bragg_qr_selection_changed",
        lambda bindings_arg: calls.append(("selection", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "toggle_selected_bragg_qr_groups_runtime",
        lambda bindings_arg: calls.append(("toggle_qr", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "toggle_selected_bragg_qr_l_values_runtime",
        lambda bindings_arg: calls.append(("toggle_l", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "enable_selected_bragg_qr_groups_runtime",
        lambda bindings_arg: calls.append(("enable_qr", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_selected_bragg_qr_groups_runtime",
        lambda bindings_arg: calls.append(("disable_qr", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "enable_all_bragg_qr_groups_runtime",
        lambda bindings_arg: calls.append(("enable_all_qr", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_all_bragg_qr_groups_runtime",
        lambda bindings_arg: calls.append(("disable_all_qr", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "enable_selected_bragg_qr_l_values_runtime",
        lambda bindings_arg: calls.append(("enable_l", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_selected_bragg_qr_l_values_runtime",
        lambda bindings_arg: calls.append(("disable_l", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "enable_all_bragg_qr_l_values_for_selected_qr_runtime",
        lambda bindings_arg: calls.append(("enable_all_l", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "disable_all_bragg_qr_l_values_for_selected_qr_runtime",
        lambda bindings_arg: calls.append(("disable_all_l", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "refresh_runtime_bragg_qr_toggle_window",
        lambda bindings_arg: calls.append(("refresh", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        bragg_qr_manager,
        "close_runtime_bragg_qr_toggle_window",
        lambda bindings_arg: calls.append(("close", bindings_arg)),
    )

    opened = bragg_qr_manager.open_runtime_bragg_qr_toggle_window(
        root=object(),
        bindings=bindings,
    )

    assert opened is True
    open_kwargs = calls[0][1]
    open_kwargs["on_qr_selection_changed"]()
    open_kwargs["on_toggle_qr"]()
    open_kwargs["on_toggle_l"]()
    open_kwargs["on_enable_selected_qr"]()
    open_kwargs["on_disable_selected_qr"]()
    open_kwargs["on_enable_all_qr"]()
    open_kwargs["on_disable_all_qr"]()
    open_kwargs["on_enable_selected_l"]()
    open_kwargs["on_disable_selected_l"]()
    open_kwargs["on_enable_all_l"]()
    open_kwargs["on_disable_all_l"]()
    open_kwargs["on_refresh"]()
    open_kwargs["on_close"]()

    assert calls[1:] == [
        ("selection", bindings),
        ("toggle_qr", bindings),
        ("toggle_l", bindings),
        ("enable_qr", bindings),
        ("disable_qr", bindings),
        ("enable_all_qr", bindings),
        ("disable_all_qr", bindings),
        ("enable_l", bindings),
        ("disable_l", bindings),
        ("enable_all_l", bindings),
        ("disable_all_l", bindings),
        ("refresh", bindings),
        ("close", bindings),
    ]
