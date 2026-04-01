"""Best-effort Fit2D error sound helpers for editable Tk inputs."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from importlib.resources import as_file, files
from typing import Any

try:
    import winsound as _winsound
except Exception:  # pragma: no cover - non-Windows platforms
    _winsound = None


_BACKSPACE_BIND_SEQUENCE = "<KeyPress-BackSpace>"
_FIT2D_SOUND_PACKAGE = "ra_sim.gui.assets"
_FIT2D_SOUND_RESOURCE = "fit2d_beep.wav"
_ENTRY_CLASS_NAMES = {
    "entry",
    "spinbox",
    "tentry",
    "tspinbox",
    "ttk::entry",
    "ttk::spinbox",
}


class _QueuedFit2DWavePlayer:
    """Play one packaged WAV sequentially on a background worker thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[int] = queue.Queue()
        self._worker_started = False
        self._lock = threading.Lock()

    def play(self, repeat: int = 1) -> int:
        """Queue *repeat* sequential plays of the packaged Fit2D beep."""

        try:
            count = max(0, int(repeat))
        except Exception:
            count = 0
        if count <= 0 or _winsound is None:
            return 0

        if not self._ensure_worker_started():
            return 0

        for _ in range(count):
            self._queue.put(1)
        return count

    def _ensure_worker_started(self) -> bool:
        with self._lock:
            if self._worker_started:
                return True
            try:
                threading.Thread(
                    target=self._worker,
                    name="fit2d-error-sound",
                    daemon=True,
                ).start()
            except Exception:
                return False
            self._worker_started = True
            return True

    def _worker(self) -> None:
        while True:
            self._queue.get()
            try:
                resource = files(_FIT2D_SOUND_PACKAGE).joinpath(_FIT2D_SOUND_RESOURCE)
                with as_file(resource) as sound_path:
                    _winsound.PlaySound(
                        str(sound_path),
                        _winsound.SND_FILENAME,
                    )
            except Exception:
                continue


_FIT2D_PLAYER = _QueuedFit2DWavePlayer()


def _widget_class_name(widget: object | None) -> str:
    if widget is None:
        return ""
    class_getter = getattr(widget, "winfo_class", None)
    if callable(class_getter):
        try:
            return str(class_getter()).strip().lower()
        except Exception:
            return ""
    return str(widget.__class__.__name__).strip().lower()


def _widget_state_text(widget: object | None) -> str:
    if widget is None:
        return ""
    cget = getattr(widget, "cget", None)
    if callable(cget):
        try:
            return str(cget("state")).strip().lower()
        except Exception:
            pass
    state = getattr(widget, "state", None)
    if callable(state):
        try:
            raw_state = state()
        except Exception:
            return ""
        if isinstance(raw_state, (tuple, list, set, frozenset)):
            return " ".join(str(item).strip().lower() for item in raw_state)
        return str(raw_state).strip().lower()
    return ""


def _widget_is_editable(widget: object | None) -> bool:
    state_text = _widget_state_text(widget)
    return "disabled" not in state_text and "readonly" not in state_text


def _entry_has_selection(widget: object) -> bool:
    selection_present = getattr(widget, "selection_present", None)
    if callable(selection_present):
        try:
            return bool(selection_present())
        except Exception:
            return False

    index = getattr(widget, "index", None)
    if callable(index):
        try:
            start = index("sel.first")
            end = index("sel.last")
        except Exception:
            return False
        return start != end
    return False


def _text_has_selection(widget: object) -> bool:
    tag_ranges = getattr(widget, "tag_ranges", None)
    if not callable(tag_ranges):
        return False
    try:
        return len(tuple(tag_ranges("sel"))) >= 2
    except Exception:
        return False


def widget_backspace_would_underflow(widget: object | None) -> bool:
    """Return whether Backspace would go past the start of one editable input."""

    if widget is None or not _widget_is_editable(widget):
        return False

    widget_class = _widget_class_name(widget)
    index = getattr(widget, "index", None)
    if not callable(index):
        return False

    if widget_class == "text" or callable(getattr(widget, "tag_ranges", None)):
        if _text_has_selection(widget):
            return False
        try:
            return str(index("insert")) == "1.0"
        except Exception:
            return False

    if widget_class not in _ENTRY_CLASS_NAMES and not callable(
        getattr(widget, "selection_present", None)
    ):
        return False
    if _entry_has_selection(widget):
        return False
    try:
        return int(float(index("insert"))) <= 0
    except Exception:
        return False


def queue_fit2d_error_sound(
    *,
    repeat: int = 1,
    bell_callback: Callable[[], object] | None = None,
) -> int:
    """Queue one or more Fit2D beeps, falling back to ``bell`` when needed."""

    try:
        count = max(0, int(repeat))
    except Exception:
        count = 0
    if count <= 0:
        return 0

    played = _FIT2D_PLAYER.play(count)
    if played > 0:
        return played

    if callable(bell_callback):
        for _ in range(count):
            try:
                bell_callback()
            except Exception:
                break
        return count
    return 0


def bind_fit2d_backspace_error_sound(
    root: object,
    *,
    enabled_var: object,
    bell_callback: Callable[[], object] | None = None,
) -> bool:
    """Bind a best-effort global Backspace handler for editable Tk inputs."""

    bind_all = getattr(root, "bind_all", None)
    if not callable(bind_all):
        return False
    if getattr(root, "_ra_fit2d_error_sound_bound", False):
        return True

    def _sound_enabled() -> bool:
        getter = getattr(enabled_var, "get", None)
        if not callable(getter):
            return False
        try:
            return bool(getter())
        except Exception:
            return False

    def _on_backspace(event: object) -> None:
        if not _sound_enabled():
            return
        widget = getattr(event, "widget", None)
        if not widget_backspace_would_underflow(widget):
            return
        queue_fit2d_error_sound(
            repeat=1,
            bell_callback=bell_callback,
        )

    bind_all(_BACKSPACE_BIND_SEQUENCE, _on_backspace, add="+")
    try:
        setattr(root, "_ra_fit2d_error_sound_bound", True)
    except Exception:
        return True
    return True
