from types import SimpleNamespace

from ra_sim import cli
from ra_sim.gui import app as gui_app
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


def test_root_launcher_forwards_non_launcher_commands_to_cli(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_cli_main(argv: list[str] | None = None) -> None:
        calls.append(list(argv or []))

    monkeypatch.setattr(cli, "main", _fake_cli_main)

    launcher.main(["simulate", "--out", "output.png"])

    assert calls == [["simulate", "--out", "output.png"]]


def test_launch_simulation_gui_forces_simulation_mode(monkeypatch) -> None:
    calls: list[tuple[bool | None, str, str | None]] = []

    def _fake_gui_main(
        *,
        write_excel_flag: bool | None = None,
        startup_mode: str = "prompt",
        calibrant_bundle: str | None = None,
    ) -> None:
        calls.append((write_excel_flag, startup_mode, calibrant_bundle))

    monkeypatch.setattr(gui_app, "main", _fake_gui_main)

    launcher.launch_simulation_gui(write_excel_flag=False)

    assert calls == [(False, "simulation", None)]
