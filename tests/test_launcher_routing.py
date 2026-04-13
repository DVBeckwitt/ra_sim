import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ra_sim import cli
from ra_sim.gui import runtime as gui_runtime
from ra_sim import launcher


def test_cmd_gui_uses_package_launcher(monkeypatch) -> None:
    calls: list[bool | None] = []

    def _fake_launch(*, write_excel_flag: bool | None = None) -> None:
        calls.append(write_excel_flag)

    monkeypatch.setattr(launcher, "launch_simulation_gui", _fake_launch)

    cli._cmd_gui(SimpleNamespace(no_excel=False))
    cli._cmd_gui(SimpleNamespace(no_excel=True))

    assert calls == [None, False]


def test_cmd_calibrant_uses_package_launcher(monkeypatch) -> None:
    calls: list[str | None] = []

    def _fake_launch(*, bundle: str | None = None) -> None:
        calls.append(bundle)

    monkeypatch.setattr(launcher, "launch_calibrant_gui", _fake_launch)

    cli._cmd_calibrant(SimpleNamespace(bundle="bundle.npz"))

    assert calls == ["bundle.npz"]


def test_cmd_mosaic_uses_package_launcher(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_launch() -> None:
        calls.append("mosaic")

    monkeypatch.setattr(launcher, "launch_mosaic_visualizer", _fake_launch)

    cli._cmd_mosaic(SimpleNamespace())

    assert calls == ["mosaic"]


def test_root_launcher_forwards_non_launcher_commands_to_cli(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_cli_main(argv: list[str] | None = None) -> None:
        calls.append(list(argv or []))

    monkeypatch.setattr(cli, "main", _fake_cli_main)

    launcher.main(["simulate", "--out", "output.png"])

    assert calls == [["simulate", "--out", "output.png"]]


def test_launch_startup_mode_uses_mosaic_visualizer(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_launch() -> None:
        calls.append("mosaic")

    monkeypatch.setattr(launcher, "launch_mosaic_visualizer", _fake_launch)

    launcher.launch_startup_mode("mosaic")

    assert calls == ["mosaic"]


def test_launch_simulation_gui_forces_simulation_mode(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        launcher.install_prereqs,
        "require_tkinter",
        lambda entrypoint_label: calls.append(("preflight", entrypoint_label)),
    )

    def _fake_gui_main(
        *,
        write_excel_flag: bool | None = None,
        startup_mode: str = "prompt",
        calibrant_bundle: str | None = None,
    ) -> None:
        calls.append(("launch", write_excel_flag, startup_mode, calibrant_bundle))

    monkeypatch.setattr(gui_runtime, "main", _fake_gui_main)

    launcher.launch_simulation_gui(write_excel_flag=False)

    assert calls == [
        (
            "preflight",
            "The RA-SIM simulation GUI (`python -m ra_sim gui` or `ra-sim gui`)",
        ),
        ("launch", False, "simulation", None),
    ]


def test_launch_simulation_gui_reframes_missing_input_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    error_dialog_calls: list[tuple[str, str]] = []

    class _FakeRoot:
        def withdraw(self) -> None:
            return None

        def attributes(self, *_args) -> None:
            return None

        def destroy(self) -> None:
            return None

    monkeypatch.setattr(
        launcher.install_prereqs,
        "require_tkinter_modules",
        lambda entrypoint_label: SimpleNamespace(
            tk=SimpleNamespace(
                Tk=lambda: _FakeRoot(),
                messagebox=SimpleNamespace(
                    showerror=lambda title, message: error_dialog_calls.append(
                        (title, message)
                    )
                ),
            ),
            ttk=None,
        ),
    )

    monkeypatch.setattr(
        launcher.install_prereqs,
        "require_tkinter",
        lambda entrypoint_label: None,
    )
    monkeypatch.setattr(launcher, "get_config_dir", lambda: cfg)
    monkeypatch.setattr(
        gui_runtime,
        "main",
        lambda **kwargs: (_ for _ in ()).throw(
            FileNotFoundError(2, "No such file or directory", "./data/sample_01.osc")
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        launcher.launch_simulation_gui()

    expected = (
        "RA-SIM simulation GUI startup failed. "
        "Missing file: ./data/sample_01.osc. "
        f"No local config at '{cfg / 'file_paths.yaml'}'. "
        "Copy `config/file_paths.example.yaml` to `config/file_paths.yaml` "
        "and point it at local `.osc`, `.poni`, `.cif`, and measured-peak files. "
        "Or set `RA_SIM_CONFIG_DIR` to your machine-specific config folder."
    )
    assert str(excinfo.value) == expected
    assert error_dialog_calls == [("RA-SIM Startup Error", expected)]


def test_launch_mosaic_visualizer_uses_installed_module(monkeypatch) -> None:
    calls: list[tuple[list[str], object, bool]] = []

    def _fake_run(command: list[str], *, cwd=None, check=False) -> None:
        calls.append((command, cwd, check))

    monkeypatch.setattr(launcher.subprocess, "run", _fake_run)

    launcher.launch_mosaic_visualizer()

    assert calls == [([sys.executable, "-m", "mosaic_sim.unified_app"], None, True)]


def test_launch_mosaic_specular_visualizer_uses_seeded_state(monkeypatch) -> None:
    calls: list[tuple[list[str], object]] = []

    def _fake_popen(command: list[str], *, cwd=None):
        calls.append((command, cwd))
        return object()

    monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(launcher, "_pick_available_local_port", lambda: 8123)

    launcher.launch_mosaic_specular_visualizer(
        {"specular-view": {"H": 1, "K": 0, "L": 2, "theta_i": 8.5}}
    )

    assert calls == [
        (
            [
                sys.executable,
                "-m",
                "mosaic_sim.unified_app",
                "--mode",
                "specular-view",
                "--port",
                "8123",
                "--state-json",
                '{"specular-view": {"H": 1, "K": 0, "L": 2, "theta_i": 8.5}}',
            ],
            None,
        )
    ]
