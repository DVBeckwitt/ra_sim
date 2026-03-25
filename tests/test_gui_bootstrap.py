import os

import pytest

from ra_sim.gui import bootstrap


def test_extract_bundle_arg_supports_inline_and_separate_values() -> None:
    assert bootstrap.extract_bundle_arg(["--bundle=test.npz"]) == "test.npz"
    assert bootstrap.extract_bundle_arg(["--bundle", "test.npz"]) == "test.npz"
    assert bootstrap.extract_bundle_arg(["--bundle"]) is None


def test_should_forward_to_cli_only_for_non_launcher_commands() -> None:
    assert bootstrap.should_forward_to_cli(["simulate", "--out", "output.png"])
    assert not bootstrap.should_forward_to_cli(["gui"])
    assert not bootstrap.should_forward_to_cli(["calibrant"])
    assert not bootstrap.should_forward_to_cli(["--help"])


def test_parse_launch_args_round_trip() -> None:
    args = bootstrap.parse_launch_args(["gui", "--no-excel", "--bundle", "bundle.npz"])

    assert args.command == "gui"
    assert args.no_excel is True
    assert args.bundle == "bundle.npz"


def test_resolve_startup_mode_prefers_explicit_command() -> None:
    assert bootstrap.resolve_startup_mode("gui") == "simulation"
    assert bootstrap.resolve_startup_mode("calibrant", early_mode="simulation") == "calibrant"
    assert bootstrap.resolve_startup_mode(None, early_mode="simulation") == "simulation"
    assert bootstrap.resolve_startup_mode(None) == "prompt"


def test_early_main_bootstrap_marks_gui_mode(monkeypatch) -> None:
    monkeypatch.delenv("RA_SIM_EARLY_STARTUP_MODE", raising=False)

    bootstrap.early_main_bootstrap("__main__", ["gui"])

    assert os.environ["RA_SIM_EARLY_STARTUP_MODE"] == "simulation"


def test_early_main_bootstrap_forwards_cli(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.delenv("RA_SIM_EARLY_STARTUP_MODE", raising=False)

    def _fake_cli_main(argv: list[str]) -> None:
        calls.append(list(argv))

    with pytest.raises(SystemExit) as exc_info:
        bootstrap.early_main_bootstrap(
            "__main__",
            ["simulate", "--out", "output.png"],
            cli_main=_fake_cli_main,
        )

    assert exc_info.value.code == 0
    assert calls == [["simulate", "--out", "output.png"]]
