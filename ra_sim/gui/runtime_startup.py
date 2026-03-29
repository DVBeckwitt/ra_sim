"""Import-safe helpers for staged post-startup task execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StartupTask:
    """One named post-startup task."""

    name: str
    callback: Callable[[], None]


@dataclass(frozen=True)
class RuntimeStartupTaskRunner:
    """Late-bound scheduler for incremental post-startup work."""

    has_tasks: Callable[[], bool]
    schedule: Callable[[], None]


def build_runtime_startup_task_runner(
    *,
    root,
    tasks: Sequence[StartupTask],
    on_error: Callable[[str, Exception], None] | None = None,
    initial_delay_ms: int = 200,
    inter_task_delay_ms: int = 75,
) -> RuntimeStartupTaskRunner:
    """Build a staged post-startup task scheduler."""

    planned_tasks = tuple(tasks)

    def _run_task(index: int = 0) -> None:
        if index >= len(planned_tasks):
            return
        task = planned_tasks[index]
        try:
            task.callback()
        except Exception as exc:
            if callable(on_error):
                on_error(str(task.name), exc)
            return
        if (index + 1) < len(planned_tasks):
            root.after(int(inter_task_delay_ms), lambda: _run_task(index + 1))

    return RuntimeStartupTaskRunner(
        has_tasks=lambda: bool(planned_tasks),
        schedule=lambda: (
            root.after(int(initial_delay_ms), _run_task)
            if planned_tasks
            else None
        ),
    )
