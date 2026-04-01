from uuid import UUID

from ra_sim.gui import window_affinity


class _FakeWindow:
    def __init__(self, *, hwnd: int = 200, width: int = 320, height: int = 240) -> None:
        self.hwnd = hwnd
        self.width = width
        self.height = height
        self.geometry_text = None
        self.updated = False

    def update_idletasks(self) -> None:
        self.updated = True

    def winfo_id(self) -> int:
        return self.hwnd

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height

    def geometry(self, text: str) -> None:
        self.geometry_text = text


def test_capture_launch_window_context_collects_desktop_and_monitor_metadata(
    monkeypatch,
) -> None:
    desktop_id = UUID("11111111-2222-3333-4444-555555555555")

    monkeypatch.setattr(window_affinity.os, "name", "nt", raising=False)
    monkeypatch.setattr(window_affinity, "_capture_launch_source_hwnd", lambda: 77)
    monkeypatch.setattr(window_affinity, "_get_window_desktop_id", lambda hwnd: desktop_id)
    monkeypatch.setattr(
        window_affinity,
        "_get_window_rect",
        lambda hwnd: (10, 20, 410, 220),
    )
    monkeypatch.setattr(
        window_affinity,
        "_get_monitor_work_area",
        lambda hwnd: (0, 0, 1920, 1080),
    )

    context = window_affinity.capture_launch_window_context()

    assert context == window_affinity.LaunchWindowContext(
        source_hwnd=77,
        desktop_id=desktop_id,
        source_rect=(10, 20, 410, 220),
        monitor_work_area=(0, 0, 1920, 1080),
    )


def test_apply_window_launch_context_moves_window_and_centers_it_near_launcher(
    monkeypatch,
) -> None:
    desktop_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    moved = []
    window = _FakeWindow(width=320, height=240)
    context = window_affinity.LaunchWindowContext(
        source_hwnd=55,
        desktop_id=desktop_id,
        source_rect=(100, 120, 700, 520),
        monitor_work_area=(0, 0, 1280, 900),
    )

    monkeypatch.setattr(window_affinity.os, "name", "nt", raising=False)
    monkeypatch.setattr(window_affinity, "_resolve_tk_window_hwnd", lambda obj: obj.hwnd)
    monkeypatch.setattr(
        window_affinity,
        "_move_window_to_desktop",
        lambda hwnd, desktop: moved.append((hwnd, desktop)) or True,
    )

    applied = window_affinity.apply_window_launch_context(window, context=context)

    assert applied is True
    assert window.updated is True
    assert moved == [(200, desktop_id)]
    assert window.geometry_text == "320x240+240+200"


def test_apply_window_launch_context_clamps_geometry_to_launcher_monitor(
    monkeypatch,
) -> None:
    window = _FakeWindow(width=900, height=700)
    context = window_affinity.LaunchWindowContext(
        source_hwnd=55,
        desktop_id=None,
        source_rect=(1100, 700, 1300, 900),
        monitor_work_area=(960, 0, 1600, 900),
    )

    monkeypatch.setattr(window_affinity.os, "name", "nt", raising=False)

    applied = window_affinity.apply_window_launch_context(window, context=context)

    assert applied is True
    assert window.geometry_text == "900x700+960+200"
