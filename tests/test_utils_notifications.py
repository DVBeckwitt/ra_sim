from types import SimpleNamespace

from ra_sim.utils import notifications


def test_play_completion_chime_plays_configured_alias(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    fake_winsound = SimpleNamespace(
        SND_ALIAS=1,
        SND_ASYNC=2,
        SND_NODEFAULT=4,
        PlaySound=lambda sound, flags: calls.append((sound, flags)),
    )
    monkeypatch.setattr(notifications, "_winsound", fake_winsound)

    played = notifications.play_completion_chime(
        {
            "enabled": True,
            "mode": "alias",
            "alias": "SystemAsterisk",
        }
    )

    assert played is True
    assert calls == [("SystemAsterisk", 7)]


def test_play_completion_chime_runs_tone_sequence_in_background(monkeypatch) -> None:
    beep_calls: list[tuple[int, int]] = []
    thread_calls: list[tuple[str, bool]] = []

    class _FakeThread:
        def __init__(self, *, target, name, daemon):
            self._target = target
            thread_calls.append((str(name), bool(daemon)))

        def start(self) -> None:
            self._target()

    fake_winsound = SimpleNamespace(
        Beep=lambda frequency_hz, duration_ms: beep_calls.append(
            (int(frequency_hz), int(duration_ms))
        )
    )
    monkeypatch.setattr(notifications, "_winsound", fake_winsound)
    monkeypatch.setattr(notifications.threading, "Thread", _FakeThread)

    played = notifications.play_completion_chime(
        {
            "enabled": True,
            "mode": "tone",
            "tone_sequence": [
                {"frequency_hz": 740, "duration_ms": 50},
                {"frequency_hz": 988, "duration_ms": 80},
            ],
        }
    )

    assert played is True
    assert thread_calls == [("geometry-fit-completion-chime", True)]
    assert beep_calls == [(740, 50), (988, 80)]
