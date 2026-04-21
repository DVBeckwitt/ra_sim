from __future__ import annotations

from io import StringIO
from pathlib import Path

from ra_sim import config
from ra_sim import dev_doctor


def _write_config(config_dir: Path, *, local: bool) -> None:
    config_dir.mkdir()
    file_paths_name = "file_paths.yaml" if local else "file_paths.example.yaml"
    hbn_paths_name = "hbn_paths.yaml" if local else "hbn_paths.example.yaml"
    (config_dir / file_paths_name).write_text(
        "\n".join(
            [
                'simulation_dark_osc_file: "./data/dark.osc"',
                "simulation_background_osc_files:",
                '  - "./data/sample.osc"',
                'geometry_poni: "./data/geometry.poni"',
                'cif_file: "./data/material.cif"',
                'measured_peaks: "./data/measured_peaks.npy"',
                'gui_geometry_poni: "./data/geometry.poni"',
                'gui_background_image: "./data/background.asc"',
                'overlay_output: "./artifacts/overlay.png"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / hbn_paths_name).write_text(
        "\n".join(
            [
                'calibrant: "./data/hbn_calibrant.osc"',
                'dark: "./data/dark.osc"',
                'bundle: "./artifacts/hbn_ellipse_bundle.npz"',
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "dir_paths.yaml").write_text(
        "\n".join(
            [
                'downloads: "./tmp/downloads"',
                'overlay_dir: "./tmp/overlays"',
                'debug_log_dir: "./tmp/logs"',
                'file_dialog_dir: "./tmp/dialogs"',
                'temp_root: "./tmp"',
            ]
        ),
        encoding="utf-8",
    )


def _tool_ok() -> list[dev_doctor.DoctorFinding]:
    return [dev_doctor.DoctorFinding("OK", "dev tools", "mocked")]


def test_doctor_default_returns_zero_with_missing_local_files(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir, local=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(dev_doctor, "check_dev_tools", _tool_ok)
    config.clear_config_cache()
    out = StringIO()

    exit_code = dev_doctor.run_doctor(strict=False, config_dir=config_dir, out=out)

    assert exit_code == 0
    assert "RA-SIM setup doctor" in out.getvalue()
    assert "missing" in out.getvalue()
    assert "[FAIL]" not in out.getvalue()


def test_doctor_strict_ignores_example_fallback_missing_files(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir, local=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(dev_doctor, "check_dev_tools", _tool_ok)
    config.clear_config_cache()
    out = StringIO()

    exit_code = dev_doctor.run_doctor(strict=True, config_dir=config_dir, out=out)

    assert exit_code == 0
    assert "using example fallback" in out.getvalue()


def test_doctor_strict_fails_for_missing_explicit_local_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir, local=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(dev_doctor, "check_dev_tools", _tool_ok)
    config.clear_config_cache()

    exit_code = dev_doctor.run_doctor(strict=True, config_dir=config_dir, out=StringIO())

    assert exit_code == 1


def test_doctor_strict_fails_for_missing_dev_tools(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir, local=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        dev_doctor,
        "check_dev_tools",
        lambda: [dev_doctor.DoctorFinding("FAIL", "pytest", "missing", True)],
    )
    config.clear_config_cache()

    out = StringIO()
    exit_code = dev_doctor.run_doctor(strict=True, config_dir=config_dir, out=out)

    assert exit_code == 1
    assert "[FAIL] pytest: missing" in out.getvalue()


def test_doctor_default_renders_strict_failures_as_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    _write_config(config_dir, local=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        dev_doctor,
        "check_dev_tools",
        lambda: [dev_doctor.DoctorFinding("FAIL", "pytest", "missing", True)],
    )
    config.clear_config_cache()
    out = StringIO()

    exit_code = dev_doctor.run_doctor(strict=False, config_dir=config_dir, out=out)

    assert exit_code == 0
    assert "[FAIL]" not in out.getvalue()
    assert "[WARN] pytest: missing" in out.getvalue()
