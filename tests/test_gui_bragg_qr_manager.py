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
