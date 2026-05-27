from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ra_sim.gui.state import SimulationRuntimeState
from tests.helpers.gui_fakes import FloatRuntimeVar as _Var


def _write_source(path: Path, marker: str) -> Path:
    path.write_text(
        "\n".join(
            [
                f"data_{marker}",
                "_cell_length_a 4.0",
                "_cell_length_b 4.0",
                "_cell_length_c 7.0",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 120",
                "loop_",
                "_space_group_symop_operation_xyz",
                "'x, y, z'",
                "loop_",
                "_atom_site_label",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_type_symbol",
                "Pb1 1.0 0 0 0 Pb",
                "I1 1.0 0.333333 0.666667 0.2675 I",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_charged_source(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data_charged_pbii",
                "_cell_length_a 4.557",
                "_cell_length_b 4.557",
                "_cell_length_c 6.979",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 120",
                "loop_",
                "_space_group_symop_operation_xyz",
                "'x, y, z'",
                "loop_",
                "_atom_site_label",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_type_symbol",
                "Pb1 1.0 0 0 0 Pb2+",
                "I1 1.0 0.333333 0.666667 0.2675 I1-",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_generated(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data_generated\n", encoding="utf-8")
    return path


class _ProgressLabel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def config(self, **kwargs) -> None:
        text = kwargs.get("text")
        if text is not None:
            self.messages.append(str(text))


def _capture_visible_messages(monkeypatch, runtime_session) -> _ProgressLabel:
    label = _ProgressLabel()
    monkeypatch.setattr(runtime_session, "progress_label_geometry", label, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    return label


def _patch_inventory_env(monkeypatch, tmp_path, source_path: Path):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimulationRuntimeState(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "defaults",
        {"p0": 0.0, "p1": 1.0, "p2": 0.5, "w0": 0.0, "w1": 1.0, "w2": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "p0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "w0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", _Var(100.0), raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "mx", 4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.5406, raising=False)
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 70.0), raising=False)
    monkeypatch.setattr(runtime_session, "intensity_threshold", 0.25, raising=False)
    monkeypatch.setattr(runtime_session, "_active_primary_cif_path", lambda: str(source_path))
    monkeypatch.setattr(
        runtime_session,
        "_disordered_phase_cif_cache_dir",
        lambda: tmp_path / "generated-cache",
    )
    return runtime_session


def test_disordered_inventory_generates_from_active_primary_cif(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    generate_calls = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        generate_calls.append((source_cif, output_dir, mode))
        return SimpleNamespace(
            cif_path=_write_generated(tmp_path / "generated.cif"),
            a=4.557,
            c=20.937,
        )

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        fake_generate_pbii_ht_shifted_cif,
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (
            np.array([[1.0, 0.0, 1.0]]),
            np.array([2.0]),
            None,
            None,
        ),
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert generate_calls == [(source.resolve(), tmp_path / "generated-cache", "compact_6h")]
    assert payload["cif_path"] == str(tmp_path / "generated.cif")
    assert runtime_session.simulation_runtime_state.generated_disordered_phase_cif_path == str(
        tmp_path / "generated.cif"
    )


def test_disordered_inventory_calls_miller_generator_on_generated_cif(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    generated_cif = tmp_path / "generated.cif"
    _write_generated(generated_cif)
    miller_calls = []

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda **_kwargs: SimpleNamespace(cif_path=generated_cif, a=4.557, c=20.937),
    )

    def fake_miller_generator(*args, **kwargs):
        miller_calls.append((args, kwargs))
        return np.array([[2.0, 0.0, 3.0]]), np.array([5.0]), None, None

    monkeypatch.setattr(runtime_session, "miller_generator", fake_miller_generator)

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert miller_calls
    args, kwargs = miller_calls[0]
    assert args[:4] == (4, str(generated_cif), [1.0], 1.5406)
    assert kwargs["intensity_threshold"] == 0.25
    assert kwargs["two_theta_range"] == (0.0, 70.0)
    assert len(miller_calls) == 2
    assert miller_calls[1][1]["intensity_threshold"] == 0.0
    np.testing.assert_allclose(payload["miller"], np.array([[2.0, 0.0, 3.0]]))
    np.testing.assert_allclose(payload["intensities"], np.array([5.0]))


def test_disordered_inventory_accepts_charged_active_pbii_cif(monkeypatch, tmp_path):
    source = _write_charged_source(tmp_path / "charged-active.cif")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (
            np.array([[1.0, 0.0, 1.0]]),
            np.array([2.0]),
            None,
            None,
        ),
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()
    visible_text = "\n".join(label.messages)

    assert payload["available"] is True
    assert payload["diagnostics"]["generated_cif_exists"] is True
    assert "anchor species 'Pb' not found" not in visible_text


def test_disordered_inventory_signature_changes_when_primary_cif_changes(monkeypatch, tmp_path):
    source_a = _write_source(tmp_path / "active_a.cif", "active_a")
    source_b = _write_source(tmp_path / "active_b.cif", "active_b")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source_a)

    sig_a = runtime_session._geometry_disordered_phase_source_signature()
    monkeypatch.setattr(runtime_session, "_active_primary_cif_path", lambda: str(source_b))
    sig_b = runtime_session._geometry_disordered_phase_source_signature()

    assert sig_a != sig_b
    assert str(source_a.resolve()) in sig_a
    assert str(source_b.resolve()) in sig_b


def test_disordered_inventory_signature_uses_charged_species_generator_version(
    monkeypatch,
    tmp_path,
):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)

    sig = runtime_session._geometry_disordered_phase_source_signature()

    assert runtime_session.DISORDERED_PHASE_QR_GENERATOR_VERSION == "ra_sim.pbi2_ht_shift_cif.v2"
    assert "ra_sim.pbi2_ht_shift_cif.v2" in sig


def test_disordered_inventory_signature_changes_when_stacking_params_change(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)

    sig_a = runtime_session._geometry_disordered_phase_source_signature()
    runtime_session.p1_var.value = 0.75
    sig_b = runtime_session._geometry_disordered_phase_source_signature()

    assert sig_a != sig_b


def test_disordered_inventory_is_not_enabled_when_stacking_disorder_disabled(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    runtime_session.w0_var.value = 100.0
    runtime_session.w1_var.value = 0.0
    runtime_session.w2_var.value = 0.0

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not generate")),
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run Miller")),
    )

    assert runtime_session._geometry_disordered_phase_qr_enabled() is False
    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["signature"] == ("disordered_phase_qr_reference", False)
    assert payload["miller"].shape == (0, 3)
    assert runtime_session.simulation_runtime_state.generated_disordered_phase_cif_path is None


def test_disordered_inventory_reports_active_cif_not_found(monkeypatch, tmp_path):
    missing_source = tmp_path / "missing.cif"
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, missing_source)
    label = _capture_visible_messages(monkeypatch, runtime_session)

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is False
    assert payload["skip_reason"] == f"active primary CIF not found: {missing_source.resolve()}"
    assert any("active primary CIF not found:" in message for message in label.messages)


def test_disordered_inventory_reports_generator_exception(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)

    def fake_generate_pbii_ht_shifted_cif(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        fake_generate_pbii_ht_shifted_cif,
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is False
    assert payload["skip_reason"] == "generated CIF failed: RuntimeError: boom"
    assert payload["diagnostics"]["generator_error_type"] == "RuntimeError"
    assert any("generated CIF failed: RuntimeError: boom" in message for message in label.messages)


def test_disordered_inventory_reports_generated_cif_missing(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)
    missing_generated = tmp_path / "missing-generated.cif"

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda **_kwargs: SimpleNamespace(cif_path=missing_generated, a=4.557, c=20.937),
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is False
    assert payload["skip_reason"] == f"generated CIF missing: {missing_generated.resolve()}"
    assert any("generated CIF missing:" in message for message in label.messages)


def test_disordered_inventory_reports_miller_exception(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)
    generated_cif = _write_generated(tmp_path / "generated.cif")

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda **_kwargs: SimpleNamespace(cif_path=generated_cif, a=4.557, c=20.937),
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad cif")),
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is False
    assert payload["skip_reason"] == "generated Miller calculation failed: ValueError: bad cif"
    assert payload["diagnostics"]["miller_error_type"] == "ValueError"
    assert any(
        "generated Miller calculation failed: ValueError: bad cif" in message
        for message in label.messages
    )


def test_disordered_inventory_uses_reference_threshold_when_current_filters_all_rows(
    monkeypatch,
    tmp_path,
):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)
    generated_cif = _write_generated(tmp_path / "generated.cif")

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        lambda **_kwargs: SimpleNamespace(cif_path=generated_cif, a=4.557, c=20.937),
    )

    def fake_miller_generator(*_args, **kwargs):
        if float(kwargs["intensity_threshold"]) > 0.0:
            return np.empty((0, 3)), np.empty((0,)), None, None
        return np.array([[1.0, 0.0, 1.0]]), np.array([2.0]), None, None

    monkeypatch.setattr(runtime_session, "miller_generator", fake_miller_generator)

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is True
    assert payload["diagnostics"]["current_threshold_miller_count"] == 0
    assert payload["diagnostics"]["reference_threshold_miller_count"] == 1
    assert any("generated Miller rows at current threshold=0" in msg for msg in label.messages)
    assert any("reference threshold=0 -> 1" in msg for msg in label.messages)


def test_disordered_inventory_empty_compact_rows_trigger_explicit_p1_fallback(
    monkeypatch,
    tmp_path,
):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)
    generate_modes: list[str] = []

    def fake_generate_pbii_ht_shifted_cif(*, mode, **_kwargs):
        generate_modes.append(str(mode))
        return SimpleNamespace(
            cif_path=_write_generated(tmp_path / f"generated-{mode}.cif"),
            a=4.557,
            c=20.937,
        )

    def fake_miller_generator(_mx, cif_file, *_args, **_kwargs):
        if "explicit_p1" in str(cif_file):
            return np.array([[2.0, 0.0, 1.0]]), np.array([3.0]), None, None
        return np.empty((0, 3)), np.empty((0,)), None, None

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        fake_generate_pbii_ht_shifted_cif,
    )
    monkeypatch.setattr(runtime_session, "miller_generator", fake_miller_generator)

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is True
    assert payload["inventory_mode"] == "explicit_p1"
    assert generate_modes == ["compact_6h", "explicit_p1"]
    assert any("using explicit-P1 fallback because" in msg for msg in label.messages)


def test_disordered_inventory_empty_explicit_p1_fallback_reports_terminal_skip(
    monkeypatch,
    tmp_path,
):
    source = _write_source(tmp_path / "active.cif", "active")
    runtime_session = _patch_inventory_env(monkeypatch, tmp_path, source)
    label = _capture_visible_messages(monkeypatch, runtime_session)

    def fake_generate_pbii_ht_shifted_cif(*, mode, **_kwargs):
        return SimpleNamespace(
            cif_path=_write_generated(tmp_path / f"generated-{mode}.cif"),
            a=4.557,
            c=20.937,
        )

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        fake_generate_pbii_ht_shifted_cif,
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *_args, **_kwargs: (np.empty((0, 3)), np.empty((0,)), None, None),
    )

    payload = runtime_session._geometry_disordered_phase_inventory_payload()

    assert payload["available"] is False
    assert payload["skip_reason"].startswith(
        "generated Miller rows empty after explicit-P1 fallback;"
    )
    assert any(
        "generated Miller rows empty after explicit-P1 fallback;" in msg
        for msg in label.messages
    )
