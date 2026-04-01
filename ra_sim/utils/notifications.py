"""Notification-chime helpers for long-running interactive tasks."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

try:
    import winsound as _winsound
except Exception:  # pragma: no cover - non-Windows platforms
    _winsound = None

_DEFAULT_ALIAS = "SystemAsterisk"
_DEFAULT_TONE_SEQUENCE: tuple[tuple[int, int], ...] = (
    (784, 60),
    (988, 80),
    (1175, 120),
)


def _normalize_tone_step(step: Any) -> tuple[int, int] | None:
    if isinstance(step, Mapping):
        raw_frequency = step.get("frequency_hz", 0)
        raw_duration = step.get("duration_ms", 0)
    elif isinstance(step, Sequence) and not isinstance(step, (str, bytes, bytearray)):
        values = list(step)
        if len(values) != 2:
            return None
        raw_frequency, raw_duration = values
    else:
        return None

    try:
        frequency_hz = int(raw_frequency)
        duration_ms = int(raw_duration)
    except Exception:
        return None
    if not (37 <= frequency_hz <= 32767):
        return None
    if duration_ms <= 0:
        return None
    return frequency_hz, duration_ms


def normalize_completion_chime_tone_sequence(
    raw_sequence: object,
) -> list[tuple[int, int]]:
    """Return a validated tone sequence for one completion chime."""

    if not isinstance(raw_sequence, Sequence) or isinstance(
        raw_sequence,
        (str, bytes, bytearray),
    ):
        return list(_DEFAULT_TONE_SEQUENCE)
    sequence = [
        normalized
        for normalized in (_normalize_tone_step(step) for step in raw_sequence)
        if normalized is not None
    ]
    return sequence or list(_DEFAULT_TONE_SEQUENCE)


def play_completion_chime(config: Mapping[str, object] | None) -> bool:
    """Play one configured completion chime.

    The helper is intentionally best-effort: invalid settings or backend
    failures are ignored so fit completion never fails because of audio.
    """

    if not isinstance(config, Mapping):
        return False
    if not bool(config.get("enabled", False)):
        return False
    if _winsound is None:
        return False

    mode = str(config.get("mode", "alias")).strip().lower()
    if mode in {"off", "none", "disabled"}:
        return False

    if mode in {"alias", "system", "system_alias"}:
        alias = str(config.get("alias", _DEFAULT_ALIAS)).strip() or _DEFAULT_ALIAS
        try:
            _winsound.PlaySound(
                alias,
                _winsound.SND_ALIAS
                | _winsound.SND_ASYNC
                | getattr(_winsound, "SND_NODEFAULT", 0),
            )
            return True
        except Exception:
            return False

    if mode not in {"tone", "beep", "sequence"}:
        return False

    tone_sequence = normalize_completion_chime_tone_sequence(
        config.get("tone_sequence")
    )

    def _play_sequence() -> None:
        for frequency_hz, duration_ms in tone_sequence:
            try:
                _winsound.Beep(int(frequency_hz), int(duration_ms))
            except Exception:
                break

    try:
        threading.Thread(
            target=_play_sequence,
            name="geometry-fit-completion-chime",
            daemon=True,
        ).start()
        return True
    except Exception:
        return False
