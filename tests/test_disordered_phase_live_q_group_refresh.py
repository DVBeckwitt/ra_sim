from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui.state import GeometryPreviewState, GeometryQGroupState, GeometryQGroupViewState
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL
from tests.test_disordered_phase_live_runtime_regression import (
    _hit_table,
    _prepare_live_runtime,
)


def _install_live_q_group_refresh_path(
    monkeypatch: pytest.MonkeyPatch,
    runtime_session,
    runtime_state,
    *,
    progress_messages: list[str],
    scheduled_updates: list[str],
) -> object:
    q_group_state = GeometryQGroupState()
    preview_state = GeometryPreviewState()
    view_state = GeometryQGroupViewState()
    monkeypatch.setattr(runtime_session, "geometry_q_group_state", q_group_state, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_preview_state", preview_state, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_q_group_view_state", view_state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(config=lambda **kwargs: progress_messages.append(str(kwargs.get("text")))),
        raising=False,
    )

    def build_entries_snapshot() -> list[dict[str, object]]:
        hit_tables = getattr(runtime_state, "stored_max_positions_local", None)
        lattice = getattr(runtime_state, "stored_peak_table_lattice", None)
        if hit_tables is None:
            hit_tables = getattr(runtime_state, "stored_primary_max_positions", None)
            lattice = getattr(runtime_state, "stored_primary_peak_table_lattice", None)
        return geometry_q_group_manager.build_geometry_q_group_entries(
            hit_tables,
            peak_table_lattice=lattice,
            primary_a=4.557,
            primary_c=6.979,
            allow_nominal_hkl_indices=True,
        )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=view_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        build_entries_snapshot=build_entries_snapshot,
        clear_last_simulation_signature=lambda: setattr(
            runtime_state,
            "last_simulation_signature",
            None,
        ),
        schedule_update=lambda: scheduled_updates.append("schedule_update"),
        set_status_text=lambda text: progress_messages.append(str(text)),
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_runtime_bindings_factory",
        lambda: bindings,
        raising=False,
    )
    return bindings


def _decision_lines(log_text: str) -> list[str]:
    return [
        line
        for line in log_text.splitlines()
        if (
            "Disordered Qr refs enabled:" in line
            or "Disordered Qr refs skipped:" in line
            or "Disordered Qr refs pending:" in line
        )
    ]


def _exercise_update_listed_peaks_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    disordered_enabled: bool,
    generate_pbii_ht_shifted_cif=None,
    miller_generator=None,
    capture_trace: bool = True,
    return_messages: bool = False,
) -> str | tuple[list[str], list[str], object]:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=disordered_enabled,
    )
    if generate_pbii_ht_shifted_cif is not None:
        monkeypatch.setattr(
            runtime_session,
            "generate_pbii_ht_shifted_cif",
            generate_pbii_ht_shifted_cif,
            raising=False,
        )
    if miller_generator is not None:
        monkeypatch.setattr(
            runtime_session,
            "miller_generator",
            miller_generator,
            raising=False,
        )
    progress_messages: list[str] = []
    trace_messages: list[str] = []
    scheduled_updates: list[str] = []
    requested_jobs: list[dict[str, object]] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_request_async_simulation_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
        raising=False,
    )
    if capture_trace:
        monkeypatch.setattr(
            runtime_session,
            "_append_runtime_update_trace",
            lambda event, **fields: trace_messages.append(str(fields.get("message") or event)),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            runtime_session,
            "_append_runtime_update_trace",
            lambda *_args, **_kwargs: None,
            raising=False,
        )

    runtime_session.do_update()
    progress_messages.clear()
    trace_messages.clear()
    requested_jobs.clear()

    geometry_q_group_manager.request_runtime_geometry_q_group_window_update(
        runtime_session.geometry_q_group_runtime_bindings_factory()
    )
    assert scheduled_updates == ["schedule_update"]

    runtime_session.do_update()
    first_log_text = "\n".join([*progress_messages, *trace_messages])
    if "Updated listed Qr/Qz peaks:" in first_log_text or not requested_jobs:
        if return_messages:
            return list(progress_messages), list(trace_messages), runtime_state
        return first_log_text

    requested_job = dict(requested_jobs[-1])
    hit_table_signature = requested_job.get("hit_table_signature")
    runtime_state.stored_hit_table_signature = hit_table_signature
    runtime_state.stored_primary_intersection_cache_signature = hit_table_signature
    runtime_state.stored_primary_intersection_cache = [_hit_table()]
    runtime_state.stored_intersection_cache = [_hit_table()]
    if disordered_enabled:
        runtime_state.stored_disordered_phase_max_positions = []
        runtime_state.stored_disordered_phase_source_reflection_indices = []
        runtime_state.stored_disordered_phase_peak_table_lattice = []
    runtime_state.last_simulation_signature = requested_job.get("signature")
    runtime_state.update_running = False
    progress_messages.clear()
    trace_messages.clear()
    requested_jobs.clear()
    scheduled_updates.clear()

    geometry_q_group_manager.request_runtime_geometry_q_group_window_update(
        runtime_session.geometry_q_group_runtime_bindings_factory()
    )
    assert scheduled_updates == ["schedule_update"]

    runtime_session.do_update()

    if return_messages:
        return list(progress_messages), list(trace_messages), runtime_state
    return "\n".join([*progress_messages, *trace_messages])


def _has_primary_only_update(progress_messages: list[str]) -> bool:
    return any(
        "Updated listed Qr/Qz peaks:" in msg
        and "sources: primary=" in msg
        and f"{DISORDERED_PHASE_SOURCE_LABEL}=" not in msg
        for msg in progress_messages
    )


def test_q_group_refresh_always_logs_disordered_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_messages, trace_messages, _runtime_state = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
        return_messages=True,
    )
    progress_text = "\n".join(progress_messages)
    trace_text = "\n".join(trace_messages)
    log_text = "\n".join([progress_text, trace_text])

    assert "Updating listed Qr/Qz peaks from the current simulation..." in log_text
    assert "Updated listed Qr/Qz peaks:" in progress_text
    assert len(_decision_lines(progress_text)) == 1


def test_visible_q_group_refresh_log_includes_disordered_decision_without_trace_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert any(
        "Disordered Qr refs enabled:" in msg
        or "Disordered Qr refs skipped:" in msg
        or "Disordered Qr refs pending:" in msg
        for msg in progress_messages
    )


def test_enabled_disordered_missing_rows_defers_primary_only_q_group_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert any("Disordered Qr refs pending:" in msg for msg in progress_messages)
    assert runtime_state.disordered_phase_hit_table_collection_requested is True
    assert not _has_primary_only_update(progress_messages)


def test_q_group_refresh_logs_enabled_status_when_generated_disorder_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert "Disordered Qr refs enabled: true" in progress_messages
    assert any("Disordered Qr refs pending:" in msg for msg in progress_messages)
    assert not _has_primary_only_update(progress_messages)


def test_q_group_refresh_logs_skip_reason_when_generated_disorder_inactive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_messages, trace_messages, _runtime_state = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
        return_messages=True,
    )
    progress_text = "\n".join(progress_messages)

    assert any("Updated listed Qr/Qz peaks:" in msg for msg in progress_messages)
    assert _decision_lines(progress_text) == [
        "Disordered Qr refs skipped: zero disordered weight"
    ]


def test_disabled_generated_disordered_qr_does_not_log_inventory_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_messages, _trace_messages, _runtime_state = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
        return_messages=True,
    )
    progress_text = "\n".join(progress_messages)

    assert "Disordered Qr refs skipped: zero disordered weight" in progress_messages
    assert "Disordered Qr refs inventory:" not in progress_text


def test_updated_q_group_log_cannot_appear_without_disordered_decision_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inactive_progress, inactive_trace, _runtime_state = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=False,
        return_messages=True,
    )
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch, tmp_path, disordered_enabled=True
    )
    active_progress: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=active_progress,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    for log_text in (
        "\n".join(inactive_progress),
        "\n".join(active_progress),
    ):
        assert _decision_lines(log_text)
    assert any("Updated listed Qr/Qz peaks:" in msg for msg in inactive_progress)
    assert not _has_primary_only_update(active_progress)


def test_pending_disordered_collection_does_not_reschedule_forever(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    first = runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )
    second = runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert first.scheduled_collection is True
    assert second.scheduled_collection is False
    assert runtime_state.disordered_phase_hit_table_collection_requested is True
    assert "Disordered Qr refs pending: collecting generated disordered hit tables" in (
        progress_messages
    )
    assert (
        "Disordered Qr refs pending: waiting for generated disordered hit-table update"
        in progress_messages
    )


def test_terminal_disordered_skip_clears_collection_request_and_does_not_spin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
    )
    progress_messages: list[str] = []
    scheduled_updates: list[str] = []
    _install_live_q_group_refresh_path(
        monkeypatch,
        runtime_session,
        runtime_state,
        progress_messages=progress_messages,
        scheduled_updates=scheduled_updates,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *_args, **_kwargs: (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            None,
            None,
        ),
        raising=False,
    )
    runtime_state.disordered_phase_inventory_cache = None
    runtime_state.disordered_phase_hit_table_collection_requested = True
    runtime_state.disordered_phase_hit_table_collection_request_signature = ("old",)
    runtime_session.geometry_q_group_state.refresh_requested = True
    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    result = runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )

    assert result.can_refresh_selector is False
    assert result.scheduled_collection is False
    assert runtime_state.disordered_phase_hit_table_collection_requested is False
    assert runtime_state.disordered_phase_hit_table_collection_request_signature is None
    assert runtime_session.geometry_q_group_state.refresh_requested is False
    assert any(
        message.startswith(
            "Disordered Qr refs skipped: generated Miller rows empty after explicit-P1 fallback;"
        )
        for message in progress_messages
    )
