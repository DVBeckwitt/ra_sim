"""Best-effort Windows launch-window affinity helpers for Tk GUIs."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
import os
from uuid import UUID


HRESULT = ctypes.c_long
HMONITOR = wintypes.HANDLE

COINIT_APARTMENTTHREADED = 0x2
CLSCTX_INPROC_SERVER = 0x1
MONITOR_DEFAULTTONEAREST = 0x2
RPC_E_CHANGED_MODE = ctypes.c_long(0x80010106).value


@dataclass(frozen=True)
class LaunchWindowContext:
    """Launcher-window affinity metadata captured before a Tk window is shown."""

    source_hwnd: int
    desktop_id: UUID | None = None
    source_rect: tuple[int, int, int, int] | None = None
    monitor_work_area: tuple[int, int, int, int] | None = None


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_uuid(cls, value: UUID) -> _GUID:
        raw = value.bytes_le
        return cls.from_buffer_copy(raw)

    def to_uuid(self) -> UUID:
        return UUID(bytes_le=bytes(self))


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


_IID_IVIRTUALDESKTOPMANAGER = _GUID.from_uuid(
    UUID("a5cd92ff-29be-454c-8d04-d82879fb3f1b")
)
_CLSID_VIRTUALDESKTOPMANAGER = _GUID.from_uuid(
    UUID("aa509086-5ca9-4c25-8f95-589d3c07b48a")
)


class _IVirtualDesktopManagerVTable(ctypes.Structure):
    _fields_ = [
        (
            "QueryInterface",
            ctypes.WINFUNCTYPE(
                HRESULT,
                ctypes.c_void_p,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            ),
        ),
        (
            "AddRef",
            ctypes.WINFUNCTYPE(
                wintypes.ULONG,
                ctypes.c_void_p,
            ),
        ),
        (
            "Release",
            ctypes.WINFUNCTYPE(
                wintypes.ULONG,
                ctypes.c_void_p,
            ),
        ),
        (
            "IsWindowOnCurrentVirtualDesktop",
            ctypes.WINFUNCTYPE(
                HRESULT,
                ctypes.c_void_p,
                wintypes.HWND,
                ctypes.POINTER(wintypes.BOOL),
            ),
        ),
        (
            "GetWindowDesktopId",
            ctypes.WINFUNCTYPE(
                HRESULT,
                ctypes.c_void_p,
                wintypes.HWND,
                ctypes.POINTER(_GUID),
            ),
        ),
        (
            "MoveWindowToDesktop",
            ctypes.WINFUNCTYPE(
                HRESULT,
                ctypes.c_void_p,
                wintypes.HWND,
                ctypes.POINTER(_GUID),
            ),
        ),
    ]


class _IVirtualDesktopManager(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_IVirtualDesktopManagerVTable))]


def capture_launch_window_context() -> LaunchWindowContext | None:
    """Capture the console/launcher desktop and monitor used to start the app."""

    if os.name != "nt":
        return None

    source_hwnd = _capture_launch_source_hwnd()
    if source_hwnd <= 0:
        return None

    return LaunchWindowContext(
        source_hwnd=int(source_hwnd),
        desktop_id=_get_window_desktop_id(int(source_hwnd)),
        source_rect=_get_window_rect(int(source_hwnd)),
        monitor_work_area=_get_monitor_work_area(int(source_hwnd)),
    )


def apply_window_launch_context(
    window: object,
    *,
    context: LaunchWindowContext | None = None,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """Move/place a Tk top-level window onto the launcher's desktop when possible."""

    if os.name != "nt":
        return False
    if context is None:
        context = capture_launch_window_context()
    if context is None:
        return False

    _safe_update_idletasks(window)

    applied = False
    hwnd = _resolve_tk_window_hwnd(window)
    if hwnd > 0 and context.desktop_id is not None:
        applied = _move_window_to_desktop(hwnd, context.desktop_id) or applied

    geometry_text = _build_window_geometry(
        window,
        width=width,
        height=height,
        source_rect=context.source_rect,
        monitor_work_area=context.monitor_work_area,
    )
    if geometry_text:
        try:
            geometry = getattr(window, "geometry")
            geometry(geometry_text)
            applied = True
        except Exception:
            pass

    return applied


def _capture_launch_source_hwnd() -> int:
    console_hwnd = _coerce_hwnd(_get_console_window())
    if _window_is_usable(console_hwnd):
        return console_hwnd

    foreground_hwnd = _coerce_hwnd(_get_foreground_window())
    if _window_is_usable(foreground_hwnd):
        return foreground_hwnd
    return 0


def _window_is_usable(hwnd: int) -> bool:
    if hwnd <= 0:
        return False
    try:
        return bool(_get_user32().IsWindow(wintypes.HWND(hwnd))) and bool(
            _get_user32().IsWindowVisible(wintypes.HWND(hwnd))
        )
    except Exception:
        return False


def _resolve_tk_window_hwnd(window: object) -> int:
    _safe_update_idletasks(window)

    try:
        wm_frame = getattr(window, "wm_frame", None)
        if callable(wm_frame):
            frame_value = str(wm_frame()).strip()
            if frame_value:
                return int(frame_value, 0)
    except Exception:
        pass

    try:
        winfo_id = getattr(window, "winfo_id", None)
        if callable(winfo_id):
            return int(winfo_id())
    except Exception:
        pass
    return 0


def _build_window_geometry(
    window: object,
    *,
    width: int | None,
    height: int | None,
    source_rect: tuple[int, int, int, int] | None,
    monitor_work_area: tuple[int, int, int, int] | None,
) -> str | None:
    resolved_width = _positive_int(width)
    resolved_height = _positive_int(height)

    if resolved_width is None:
        resolved_width = _window_dimension(window, "winfo_width", "winfo_reqwidth")
    if resolved_height is None:
        resolved_height = _window_dimension(window, "winfo_height", "winfo_reqheight")
    if resolved_width is None or resolved_height is None:
        return None

    frame_rect = (
        monitor_work_area
        if monitor_work_area is not None
        else source_rect
    )
    if frame_rect is None:
        return None

    frame_left, frame_top, frame_right, frame_bottom = frame_rect
    anchor_left, anchor_top = frame_left, frame_top
    anchor_width = max(frame_right - frame_left, 1)
    anchor_height = max(frame_bottom - frame_top, 1)

    if source_rect is not None:
        anchor_left, anchor_top, source_right, source_bottom = source_rect
        anchor_width = max(source_right - anchor_left, 1)
        anchor_height = max(source_bottom - anchor_top, 1)

    max_left = frame_right - resolved_width
    max_top = frame_bottom - resolved_height
    if max_left < frame_left:
        max_left = frame_left
    if max_top < frame_top:
        max_top = frame_top

    target_left = anchor_left + max((anchor_width - resolved_width) // 2, 0)
    target_top = anchor_top + max((anchor_height - resolved_height) // 2, 0)
    target_left = min(max(target_left, frame_left), max_left)
    target_top = min(max(target_top, frame_top), max_top)

    return f"{resolved_width}x{resolved_height}+{target_left}+{target_top}"


def _window_dimension(window: object, *getter_names: str) -> int | None:
    for getter_name in getter_names:
        getter = getattr(window, getter_name, None)
        if not callable(getter):
            continue
        try:
            value = _positive_int(getter())
        except Exception:
            value = None
        if value is not None and value > 1:
            return value
    return None


def _safe_update_idletasks(window: object) -> None:
    try:
        update_idletasks = getattr(window, "update_idletasks", None)
        if callable(update_idletasks):
            update_idletasks()
    except Exception:
        pass


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    if hwnd <= 0:
        return None
    rect = wintypes.RECT()
    try:
        if not _get_user32().GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
    except Exception:
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _get_monitor_work_area(hwnd: int) -> tuple[int, int, int, int] | None:
    if hwnd <= 0:
        return None
    try:
        monitor = _get_user32().MonitorFromWindow(
            wintypes.HWND(hwnd),
            MONITOR_DEFAULTTONEAREST,
        )
    except Exception:
        return None
    if not monitor:
        return None

    info = _MONITORINFO()
    info.cbSize = ctypes.sizeof(_MONITORINFO)
    try:
        if not _get_user32().GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
    except Exception:
        return None
    return (
        int(info.rcWork.left),
        int(info.rcWork.top),
        int(info.rcWork.right),
        int(info.rcWork.bottom),
    )


def _get_window_desktop_id(hwnd: int) -> UUID | None:
    if hwnd <= 0:
        return None

    init_hr = _co_initialize_ex()
    should_uninitialize = init_hr in (0, 1)
    if _hresult_failed(init_hr) and init_hr != RPC_E_CHANGED_MODE:
        return None

    manager: ctypes.POINTER(_IVirtualDesktopManager) | None = None
    try:
        manager = _co_create_virtual_desktop_manager()
        if manager is None:
            return None
        desktop_id = _GUID()
        hr = manager.contents.lpVtbl.contents.GetWindowDesktopId(
            manager,
            wintypes.HWND(hwnd),
            ctypes.byref(desktop_id),
        )
        if _hresult_failed(hr):
            return None
        return desktop_id.to_uuid()
    finally:
        try:
            if manager is not None:
                manager.contents.lpVtbl.contents.Release(manager)
        except Exception:
            pass
        if should_uninitialize:
            _get_ole32().CoUninitialize()


def _move_window_to_desktop(hwnd: int, desktop_id: UUID) -> bool:
    if hwnd <= 0:
        return False

    init_hr = _co_initialize_ex()
    should_uninitialize = init_hr in (0, 1)
    if _hresult_failed(init_hr) and init_hr != RPC_E_CHANGED_MODE:
        return False

    desktop_guid = _GUID.from_uuid(desktop_id)
    manager: ctypes.POINTER(_IVirtualDesktopManager) | None = None
    try:
        manager = _co_create_virtual_desktop_manager()
        if manager is None:
            return False
        hr = manager.contents.lpVtbl.contents.MoveWindowToDesktop(
            manager,
            wintypes.HWND(hwnd),
            ctypes.byref(desktop_guid),
        )
        return not _hresult_failed(hr)
    finally:
        try:
            if manager is not None:
                manager.contents.lpVtbl.contents.Release(manager)
        except Exception:
            pass
        if should_uninitialize:
            _get_ole32().CoUninitialize()


def _co_create_virtual_desktop_manager() -> ctypes.POINTER(_IVirtualDesktopManager) | None:
    manager_void = ctypes.c_void_p()
    hr = _get_ole32().CoCreateInstance(
        ctypes.byref(_CLSID_VIRTUALDESKTOPMANAGER),
        None,
        CLSCTX_INPROC_SERVER,
        ctypes.byref(_IID_IVIRTUALDESKTOPMANAGER),
        ctypes.byref(manager_void),
    )
    if _hresult_failed(hr) or not manager_void:
        return None
    return ctypes.cast(manager_void, ctypes.POINTER(_IVirtualDesktopManager))


def _co_initialize_ex() -> int:
    return int(
        _get_ole32().CoInitializeEx(
            None,
            COINIT_APARTMENTTHREADED,
        )
    )


def _coerce_hwnd(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _positive_int(value: object) -> int | None:
    try:
        result = int(round(float(value)))
    except Exception:
        return None
    if result <= 0:
        return None
    return result


def _hresult_failed(value: object) -> bool:
    try:
        return int(value) < 0
    except Exception:
        return True


@lru_cache(maxsize=1)
def _get_user32():
    dll = ctypes.WinDLL("user32", use_last_error=True)
    dll.GetForegroundWindow.restype = wintypes.HWND
    dll.IsWindow.argtypes = [wintypes.HWND]
    dll.IsWindow.restype = wintypes.BOOL
    dll.IsWindowVisible.argtypes = [wintypes.HWND]
    dll.IsWindowVisible.restype = wintypes.BOOL
    dll.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    dll.GetWindowRect.restype = wintypes.BOOL
    dll.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    dll.MonitorFromWindow.restype = HMONITOR
    dll.GetMonitorInfoW.argtypes = [HMONITOR, ctypes.POINTER(_MONITORINFO)]
    dll.GetMonitorInfoW.restype = wintypes.BOOL
    return dll


@lru_cache(maxsize=1)
def _get_kernel32():
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.GetConsoleWindow.restype = wintypes.HWND
    return dll


@lru_cache(maxsize=1)
def _get_ole32():
    dll = ctypes.OleDLL("ole32")
    dll.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    dll.CoInitializeEx.restype = HRESULT
    dll.CoUninitialize.argtypes = []
    dll.CoUninitialize.restype = None
    dll.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    dll.CoCreateInstance.restype = HRESULT
    return dll


def _get_console_window() -> int:
    return _coerce_hwnd(_get_kernel32().GetConsoleWindow())


def _get_foreground_window() -> int:
    return _coerce_hwnd(_get_user32().GetForegroundWindow())
