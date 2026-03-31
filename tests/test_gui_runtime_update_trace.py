from datetime import datetime
from pathlib import Path

from ra_sim.gui import runtime_update_trace


def test_resolve_runtime_update_trace_path_uses_daily_stamp() -> None:
    path = runtime_update_trace.resolve_runtime_update_trace_path(
        "C:/tmp/downloads",
        current_time=datetime(2026, 3, 31, 12, 15, 0),
    )

    assert path == Path("C:/tmp/downloads/runtime_update_trace_20260331.log")


def test_format_runtime_update_trace_line_formats_scalar_and_sequence_fields() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_start",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "phase": "applying",
            "update_id": 7,
            "ready": True,
            "cost": 1.25,
            "values": [1, 2, 3],
            "message": "line1\nline2",
        },
    )

    assert line == (
        "2026-03-31T12:15:00.123 pid=4242 event=do_update_start "
        "cost=1.250000 message=line1\\nline2 phase=applying ready=true "
        "update_id=7 values=[1,2,3]"
    )


def test_append_runtime_update_trace_line_writes_file(tmp_path) -> None:
    trace_path = tmp_path / "runtime_update_trace_20260331.log"

    runtime_update_trace.append_runtime_update_trace_line(
        trace_path,
        "schedule_update",
        queued=True,
        update_id=9,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert "event=schedule_update" in text
    assert "queued=true" in text
    assert "update_id=9" in text
