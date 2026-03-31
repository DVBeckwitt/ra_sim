from __future__ import annotations

from argparse import _SubParsersAction
from types import SimpleNamespace

import pytest

from ra_sim import cli


def _get_subparser_choices(parser):
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return action.choices
    raise AssertionError("parser does not define any subcommands")


def _require_fit_geometry_command():
    cmd = getattr(cli, "_cmd_fit_geometry", None)
    if cmd is None:
        pytest.skip("fit-geometry CLI command is not available in this checkout")
    return cmd


def _patch_first_available(monkeypatch, candidates, replacement):
    for name in candidates:
        if hasattr(cli, name):
            monkeypatch.setattr(cli, name, replacement)
            return name
    pytest.skip("geometry-fit runner helper is not available in this checkout")


def test_cli_build_parser_includes_fit_geometry_command() -> None:
    if getattr(cli, "_cmd_fit_geometry", None) is None:
        pytest.skip("fit-geometry CLI command is not available in this checkout")

    parser = cli._build_parser()
    assert "fit-geometry" in _get_subparser_choices(parser)


def test_cmd_fit_geometry_loads_saved_state_runs_fit_and_saves(monkeypatch, tmp_path) -> None:
    cmd = _require_fit_geometry_command()

    input_path = tmp_path / "saved_gui_state.json"
    output_path = tmp_path / "fit_gui_state.json"
    loaded_payload = {
        "type": "ra_sim.gui_state",
        "state": {
            "files": {"background_files": ["bg0.osc"]},
            "geometry": {"manual_pairs": [{"background_index": 0, "entries": []}]},
        },
    }

    events: list[tuple[str, object]] = []
    saved_payload: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "load_gui_state_file",
        lambda path: events.append(("load", str(path))) or loaded_payload,
    )

    def _fake_runner(*args, **kwargs):
        events.append(("run", {"args": args, "kwargs": kwargs}))
        return {
            "type": "ra_sim.gui_state",
            "state": {
                "files": {"background_files": ["bg0.osc"]},
                "geometry": {
                    "manual_pairs": [{"background_index": 0, "entries": []}],
                    "fit_result": "ok",
                },
            },
        }

    _patch_first_available(
        monkeypatch,
        [
            "run_headless_geometry_fit",
            "run_geometry_fit_from_saved_state",
            "run_geometry_fit_from_state",
        ],
        _fake_runner,
    )

    def _fake_save(path, state, **kwargs):
        events.append(("save", str(path), kwargs))
        saved_payload["path"] = str(path)
        saved_payload["state"] = state
        saved_payload["kwargs"] = kwargs
        return {"path": str(path), "state": state}

    monkeypatch.setattr(cli, "save_gui_state_file", _fake_save)

    args = SimpleNamespace(
        state=str(input_path),
        input_state=str(input_path),
        gui_state=str(input_path),
        source_state=str(input_path),
        out_state=str(output_path),
        output_state=str(output_path),
        output=str(output_path),
        in_place=False,
        overwrite=False,
    )

    cmd(args)

    assert events[0] == ("load", str(input_path))
    assert events[1][0] == "run"
    assert events[2] == ("save", str(output_path), {})
    assert saved_payload["path"] == str(output_path)
    assert saved_payload["state"]["geometry"]["fit_result"] == "ok"


def test_cmd_fit_geometry_supports_in_place_saves(monkeypatch, tmp_path) -> None:
    cmd = _require_fit_geometry_command()

    input_path = tmp_path / "saved_gui_state.json"
    loaded_payload = {"type": "ra_sim.gui_state", "state": {"geometry": {}}}

    monkeypatch.setattr(
        cli,
        "load_gui_state_file",
        lambda path: loaded_payload,
    )

    _patch_first_available(
        monkeypatch,
        [
            "run_headless_geometry_fit",
            "run_geometry_fit_from_saved_state",
            "run_geometry_fit_from_state",
        ],
        lambda *args, **kwargs: loaded_payload,
    )

    save_calls: list[tuple[str, object]] = []

    def _fake_save(path, state, **kwargs):
        save_calls.append((str(path), state))
        return {"path": str(path), "state": state}

    monkeypatch.setattr(cli, "save_gui_state_file", _fake_save)

    args = SimpleNamespace(
        state=str(input_path),
        input_state=str(input_path),
        gui_state=str(input_path),
        source_state=str(input_path),
        out_state=None,
        output_state=None,
        output=None,
        in_place=True,
        overwrite=False,
    )

    cmd(args)

    assert save_calls == [(str(input_path), loaded_payload["state"])]
