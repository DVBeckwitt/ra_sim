import sys
from contextlib import contextmanager
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


def test_root_launcher_force_exits_after_flagged_gui_close(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _StreamRecorder:
        def __init__(self, label: str) -> None:
            self._label = label

        def write(self, text: str) -> int:
            calls.append(("write", self._label, text))
            return len(text)

        def flush(self) -> None:
            calls.append(("flush", self._label))

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(launcher, "start_run_bundle", lambda *, entrypoint: calls.append(("bundle", entrypoint)))
    monkeypatch.setattr(
        launcher,
        "launch_startup_mode",
        lambda startup_mode, *, write_excel_flag=None, calibrant_bundle=None: calls.append(
            ("launch", startup_mode, write_excel_flag, calibrant_bundle)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "finalize_run_bundle",
        lambda: calls.append(("finalize",)) or Path("bundle.zip"),
    )
    monkeypatch.setattr(launcher.sys, "stdout", _StreamRecorder("stdout"))
    monkeypatch.setattr(launcher.sys, "stderr", _StreamRecorder("stderr"))
    monkeypatch.setattr(launcher.os, "_exit", lambda code: calls.append(("exit", code)))

    launcher.main(["gui"])

    assert calls == [
        ("bundle", "launcher:simulation"),
        ("launch", "simulation", None, None),
        ("finalize",),
        ("flush", "stdout"),
        ("flush", "stderr"),
        ("exit", 0),
    ]


def test_root_launcher_force_exit_survives_finalize_failure(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _StreamRecorder:
        def __init__(self, label: str) -> None:
            self._label = label

        def write(self, text: str) -> int:
            calls.append(("write", self._label, text))
            return len(text)

        def flush(self) -> None:
            calls.append(("flush", self._label))

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(launcher, "start_run_bundle", lambda *, entrypoint: calls.append(("bundle", entrypoint)))
    monkeypatch.setattr(
        launcher,
        "launch_startup_mode",
        lambda startup_mode, *, write_excel_flag=None, calibrant_bundle=None: calls.append(
            ("launch", startup_mode, write_excel_flag, calibrant_bundle)
        ),
    )

    def _raise_finalize() -> None:
        calls.append(("finalize",))
        raise OSError("zip failed")

    monkeypatch.setattr(launcher, "finalize_run_bundle", _raise_finalize)
    monkeypatch.setattr(launcher.sys, "stdout", _StreamRecorder("stdout"))
    monkeypatch.setattr(launcher.sys, "stderr", _StreamRecorder("stderr"))
    monkeypatch.setattr(launcher.os, "_exit", lambda code: calls.append(("exit", code)))

    launcher.main(["gui"])

    assert calls[0:3] == [
        ("bundle", "launcher:simulation"),
        ("launch", "simulation", None, None),
        ("finalize",),
    ]
    assert ("flush", "stdout") in calls
    assert ("flush", "stderr") in calls
    assert ("exit", 0) in calls
    assert any(
        entry[0] == "write"
        and entry[1] == "stderr"
        and "run bundle finalization failed" in entry[2]
        for entry in calls
    )


def test_root_launcher_force_exit_runs_on_simulation_startup_error(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _StreamRecorder:
        def __init__(self, label: str) -> None:
            self._label = label

        def write(self, text: str) -> int:
            calls.append(("write", self._label, text))
            return len(text)

        def flush(self) -> None:
            calls.append(("flush", self._label))

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(launcher, "start_run_bundle", lambda *, entrypoint: calls.append(("bundle", entrypoint)))

    def _raise_launch(
        startup_mode, *, write_excel_flag=None, calibrant_bundle=None
    ) -> None:
        calls.append(("launch", startup_mode, write_excel_flag, calibrant_bundle))
        raise SystemExit("gui boom")

    monkeypatch.setattr(launcher, "launch_startup_mode", _raise_launch)
    monkeypatch.setattr(
        launcher,
        "finalize_run_bundle",
        lambda: calls.append(("finalize",)) or Path("bundle.zip"),
    )
    monkeypatch.setattr(launcher.sys, "stdout", _StreamRecorder("stdout"))
    monkeypatch.setattr(launcher.sys, "stderr", _StreamRecorder("stderr"))
    monkeypatch.setattr(launcher.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["gui"])

    assert excinfo.value.code == 1
    assert calls[0:3] == [
        ("bundle", "launcher:simulation"),
        ("launch", "simulation", None, None),
        ("write", "stderr", "gui boom"),
    ]
    assert ("finalize",) in calls
    assert ("flush", "stdout") in calls
    assert ("flush", "stderr") in calls


def test_root_launcher_force_exit_prints_traceback_for_unexpected_startup_error(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class _StreamRecorder:
        def __init__(self, label: str) -> None:
            self._label = label

        def write(self, text: str) -> int:
            calls.append(("write", self._label, text))
            return len(text)

        def flush(self) -> None:
            calls.append(("flush", self._label))

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(launcher, "start_run_bundle", lambda *, entrypoint: calls.append(("bundle", entrypoint)))

    def _raise_launch(
        startup_mode, *, write_excel_flag=None, calibrant_bundle=None
    ) -> None:
        calls.append(("launch", startup_mode, write_excel_flag, calibrant_bundle))
        raise RuntimeError("gui boom")

    monkeypatch.setattr(launcher, "launch_startup_mode", _raise_launch)
    monkeypatch.setattr(
        launcher,
        "finalize_run_bundle",
        lambda: calls.append(("finalize",)) or Path("bundle.zip"),
    )
    monkeypatch.setattr(launcher.sys, "stdout", _StreamRecorder("stdout"))
    monkeypatch.setattr(launcher.sys, "stderr", _StreamRecorder("stderr"))
    monkeypatch.setattr(launcher.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["gui"])

    assert excinfo.value.code == 1
    assert calls[0:2] == [
        ("bundle", "launcher:simulation"),
        ("launch", "simulation", None, None),
    ]
    assert ("finalize",) in calls
    assert ("flush", "stdout") in calls
    assert ("flush", "stderr") in calls
    assert any(
        entry[0] == "write" and entry[1] == "stderr" and "RuntimeError: gui boom" in entry[2]
        for entry in calls
    )


@pytest.mark.parametrize("startup_mode", ["calibrant", "mosaic"])
def test_root_launcher_flag_does_not_force_exit_for_non_simulation_launcher_modes(
    monkeypatch,
    startup_mode: str,
) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(launcher, "start_run_bundle", lambda *, entrypoint: calls.append(("bundle", entrypoint)))
    monkeypatch.setattr(
        launcher,
        "launch_startup_mode",
        lambda selected_mode, *, write_excel_flag=None, calibrant_bundle=None: calls.append(
            ("launch", selected_mode, write_excel_flag, calibrant_bundle)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "finalize_run_bundle",
        lambda: calls.append(("finalize",)) or Path("bundle.zip"),
    )
    monkeypatch.setattr(launcher.os, "_exit", lambda code: calls.append(("exit", code)))

    launcher.main([startup_mode])

    assert calls == [
        ("bundle", f"launcher:{startup_mode}"),
        ("launch", startup_mode, None, None),
    ]


def test_root_launcher_flag_does_not_force_exit_for_forwarded_cli(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setenv("RA_SIM_FORCE_EXIT_ON_GUI_CLOSE", "1")
    monkeypatch.setattr(
        cli,
        "main",
        lambda argv=None: calls.append(("cli", list(argv or []))),
    )
    monkeypatch.setattr(
        launcher,
        "finalize_run_bundle",
        lambda: calls.append(("finalize",)) or Path("bundle.zip"),
    )
    monkeypatch.setattr(launcher.os, "_exit", lambda code: calls.append(("exit", code)))

    launcher.main(["simulate", "--out", "output.png"])

    assert calls == [("cli", ["simulate", "--out", "output.png"])]


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
    monkeypatch.setattr(
        launcher.gui_bootstrap,
        "quick_simulation_debug_override_dialog",
        lambda: "inherit",
    )

    @contextmanager
    def _fake_debug_override(mode: str):
        calls.append(("debug-override", mode))
        yield

    monkeypatch.setattr(launcher, "temporary_startup_debug_override", _fake_debug_override)

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
        ("debug-override", "inherit"),
        ("launch", False, "simulation", None),
    ]


def test_launch_simulation_gui_cancelled_before_runtime_start(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        launcher.install_prereqs,
        "require_tkinter",
        lambda entrypoint_label: calls.append(("preflight", entrypoint_label)),
    )
    monkeypatch.setattr(
        launcher.gui_bootstrap,
        "quick_simulation_debug_override_dialog",
        lambda: None,
    )
    monkeypatch.setattr(
        gui_runtime,
        "main",
        lambda **kwargs: calls.append(("launch", kwargs)),
    )

    launcher.launch_simulation_gui()

    assert calls == [
        (
            "preflight",
            "The RA-SIM simulation GUI (`python -m ra_sim gui` or `ra-sim gui`)",
        ),
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
    monkeypatch.setattr(
        launcher.gui_bootstrap,
        "quick_simulation_debug_override_dialog",
        lambda: "disable_all",
    )
    @contextmanager
    def _fake_debug_override(mode: str):
        yield

    monkeypatch.setattr(launcher, "temporary_startup_debug_override", _fake_debug_override)
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
