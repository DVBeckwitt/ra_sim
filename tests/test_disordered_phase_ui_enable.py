from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace


class _Var:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


def _runtime_session():
    return importlib.import_module("ra_sim.gui._runtime.runtime_session")


def _write_cif(path: Path) -> Path:
    path.write_text("data_pbi2\n_cell_length_a 4.557\n", encoding="utf-8")
    return path


def _patch_status_env(
    monkeypatch,
    tmp_path: Path,
    *,
    weights: tuple[float, float, float],
    active_cif: bool = True,
    packaged_6h_enabled: bool = False,
    generated_disordered_enabled: bool | None = None,
):
    runtime_session = _runtime_session()
    source_cif = _write_cif(tmp_path / "active.cif")
    monkeypatch.setattr(
        runtime_session,
        "stacking_parameter_controls_view_state",
        SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "defaults",
        {"w0": 100.0, "w1": 0.0, "w2": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "w0_var", _Var(weights[0]), raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", _Var(weights[1]), raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", _Var(weights[2]), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_active_primary_cif_path",
        (lambda: str(source_cif)) if active_cif else (lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_include_6h_qr_reference_var",
        _Var(packaged_6h_enabled),
        raising=False,
    )
    if generated_disordered_enabled is None:
        monkeypatch.delattr(
            runtime_session,
            "geometry_include_generated_disordered_qr_var",
            raising=False,
        )
    else:
        monkeypatch.setattr(
            runtime_session,
            "geometry_include_generated_disordered_qr_var",
            _Var(generated_disordered_enabled),
            raising=False,
        )
    monkeypatch.delattr(
        runtime_session,
        "geometry_include_disordered_phase_qr_reference_var",
        raising=False,
    )
    return runtime_session, source_cif


def test_generated_disordered_checkbox_defaults_on(monkeypatch, tmp_path):
    runtime_session, _source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        generated_disordered_enabled=None,
    )

    assert runtime_session._geometry_generated_disordered_qr_reference_checked() is True


def test_generated_disordered_qr_enabled_when_weight_nonzero(monkeypatch, tmp_path):
    runtime_session, source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is True
    assert status.reason == "enabled"
    assert status.stacking_disorder_enabled is True
    assert status.nonzero_disordered_weight is True
    assert status.active_primary_cif_path == str(source_cif.resolve())
    assert runtime_session._geometry_disordered_phase_qr_enabled() is True


def test_generated_disordered_qr_disabled_when_checkbox_off(monkeypatch, tmp_path):
    runtime_session, source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        generated_disordered_enabled=False,
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is False
    assert status.reason == "checkbox disabled"
    assert status.stacking_disorder_enabled is True
    assert status.nonzero_disordered_weight is True
    assert status.active_primary_cif_path == str(source_cif.resolve())


def test_generated_disordered_qr_enabled_when_checkbox_on_and_weight_nonzero(
    monkeypatch,
    tmp_path,
):
    runtime_session, _source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        generated_disordered_enabled=True,
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is True
    assert status.reason == "enabled"


def test_generated_disordered_qr_disabled_when_all_disordered_weights_zero(
    monkeypatch,
    tmp_path,
):
    runtime_session, source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(100.0, 0.0, 0.0),
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is False
    assert status.reason == "zero disordered weight"
    assert status.stacking_disorder_enabled is True
    assert status.nonzero_disordered_weight is False
    assert status.active_primary_cif_path == str(source_cif.resolve())


def test_generated_disordered_qr_disabled_when_stacking_disorder_off(
    monkeypatch,
    tmp_path,
):
    runtime_session, source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 0.0, 0.0),
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is False
    assert status.reason == "stacking disorder disabled"
    assert status.stacking_disorder_enabled is False
    assert status.nonzero_disordered_weight is False
    assert status.active_primary_cif_path == str(source_cif.resolve())


def test_generated_disordered_qr_disabled_without_active_primary_cif(
    monkeypatch,
    tmp_path,
):
    runtime_session, _source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        active_cif=False,
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is False
    assert status.reason == "no active primary CIF"
    assert status.stacking_disorder_enabled is True
    assert status.nonzero_disordered_weight is True
    assert status.active_primary_cif_path is None


def test_packaged_6h_toggle_does_not_control_generated_disordered_phase(
    monkeypatch,
    tmp_path,
):
    runtime_session, _source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        packaged_6h_enabled=False,
    )
    disabled_6h_status = runtime_session._geometry_disordered_phase_qr_enable_status()

    _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        packaged_6h_enabled=True,
    )
    enabled_6h_status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert disabled_6h_status.enabled is True
    assert enabled_6h_status.enabled is True
    assert disabled_6h_status.reason == "enabled"
    assert enabled_6h_status.reason == "enabled"


def test_packaged_6h_toggle_does_not_enable_generated_disordered_phase(
    monkeypatch,
    tmp_path,
):
    runtime_session, _source_cif = _patch_status_env(
        monkeypatch,
        tmp_path,
        weights=(0.0, 100.0, 0.0),
        packaged_6h_enabled=True,
        generated_disordered_enabled=False,
    )

    status = runtime_session._geometry_disordered_phase_qr_enable_status()

    assert status.enabled is False
    assert status.reason == "checkbox disabled"
