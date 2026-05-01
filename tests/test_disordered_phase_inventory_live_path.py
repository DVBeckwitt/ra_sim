from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_disordered_phase_live_q_group_refresh import (
    _exercise_update_listed_peaks_refresh,
)


def test_live_q_group_refresh_evaluates_disordered_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        calls.append({"source_cif": source_cif, "output_dir": output_dir, "mode": mode})
        return SimpleNamespace(cif_path=tmp_path / "generated-disordered.cif", a=4.557, c=20.937)

    _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
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
        return SimpleNamespace(cif_path=tmp_path / "generated-disordered.cif", a=4.557, c=20.937)

    log_text = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
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
        return SimpleNamespace(cif_path=tmp_path / "generated-disordered.cif", a=4.557, c=20.937)

    _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
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
        return SimpleNamespace(cif_path=generated_cif, a=4.557, c=20.937)

    log_text = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
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

    log_text = _exercise_update_listed_peaks_refresh(
        monkeypatch,
        tmp_path,
        disordered_enabled=True,
        miller_generator=fake_miller_generator,
    )

    assert "Disordered Qr refs skipped: inventory unavailable" in log_text
