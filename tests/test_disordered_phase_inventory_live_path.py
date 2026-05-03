from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_disordered_phase_live_q_group_refresh import (
    _install_live_q_group_refresh_path,
)
from tests.test_disordered_phase_live_runtime_regression import (
    _prepare_live_runtime,
)


def _write_generated_cif(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data_generated\n", encoding="utf-8")
    return path


def _exercise_live_inventory_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    generate_pbii_ht_shifted_cif=None,
    miller_generator=None,
) -> str:
    runtime_session, runtime_state = _prepare_live_runtime(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
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
        lambda event, **fields: trace_messages.append(str(fields.get("message") or event)),
        raising=False,
    )
    status = runtime_session._geometry_disordered_phase_qr_enable_status()
    runtime_session._ensure_generated_disordered_qr_rows_for_live_refresh(
        status,
        primary_a=4.557,
        primary_c=6.979,
    )
    return "\n".join(progress_messages)


def test_live_q_group_refresh_evaluates_disordered_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        calls.append({"source_cif": source_cif, "output_dir": output_dir, "mode": mode})
        return SimpleNamespace(
            cif_path=_write_generated_cif(tmp_path / "generated-disordered.cif"),
            a=4.557,
            c=20.937,
        )

    _exercise_live_inventory_refresh(
        monkeypatch,
        tmp_path,
        generate_pbii_ht_shifted_cif=fake_generate_pbii_ht_shifted_cif,
    )

    assert calls


def test_live_inventory_uses_active_primary_cif(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        calls.append(Path(str(source_cif)))
        return SimpleNamespace(
            cif_path=_write_generated_cif(tmp_path / "generated-disordered.cif"),
            a=4.557,
            c=20.937,
        )

    log_text = _exercise_live_inventory_refresh(
        monkeypatch,
        tmp_path,
        generate_pbii_ht_shifted_cif=fake_generate_pbii_ht_shifted_cif,
    )

    assert calls == [(tmp_path / "active-pbi2.cif").resolve()]
    assert f"source_cif={(tmp_path / 'active-pbi2.cif').resolve()}" in log_text


def test_live_inventory_does_not_use_packaged_6h_cif(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        calls.append(str(source_cif))
        return SimpleNamespace(
            cif_path=_write_generated_cif(tmp_path / "generated-disordered.cif"),
            a=4.557,
            c=20.937,
        )

    _exercise_live_inventory_refresh(
        monkeypatch,
        tmp_path,
        generate_pbii_ht_shifted_cif=fake_generate_pbii_ht_shifted_cif,
    )

    assert calls
    assert all("PbI2_6H.cif" not in path for path in calls)


def test_live_inventory_logs_source_and_generated_cif_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated_cif = tmp_path / "generated-disordered.cif"

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        return SimpleNamespace(cif_path=_write_generated_cif(generated_cif), a=4.557, c=20.937)

    log_text = _exercise_live_inventory_refresh(
        monkeypatch,
        tmp_path,
        generate_pbii_ht_shifted_cif=fake_generate_pbii_ht_shifted_cif,
    )

    assert (
        f"Disordered Qr refs inventory: source_cif={(tmp_path / 'active-pbi2.cif').resolve()}"
        in log_text
    )
    assert f"Disordered Qr refs inventory: generated_cif={generated_cif}" in log_text


def test_live_inventory_logs_unavailable_skip_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_miller_generator(*_args, **_kwargs):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            None,
            None,
        )

    log_text = _exercise_live_inventory_refresh(
        monkeypatch,
        tmp_path,
        miller_generator=fake_miller_generator,
    )

    assert "Disordered Qr refs skipped: generated Miller rows empty after explicit-P1 fallback;" in log_text
