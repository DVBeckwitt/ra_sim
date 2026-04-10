"""Package-owned launcher helpers for GUI entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from ra_sim.gui import bootstrap as gui_bootstrap

_MOSAIC_REPO_ENV_VAR = "RA_SIM_MOSAIC_REPO"
_MOSAIC_REPO_DIRNAME = "2D_Mosaic_Sim"
_MOSAIC_SCRIPT_NAMES = (
    "mosaic_simulator.py",
    "simulate_mosaic.py",
)


def _pick_available_local_port() -> int:
    """Return an ephemeral localhost TCP port for a new Dash subprocess."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_simulation_gui(*, write_excel_flag: bool | None = None) -> None:
    """Launch the canonical packaged simulation GUI runtime."""

    from ra_sim.gui.runtime import main as gui_main

    gui_main(
        write_excel_flag=write_excel_flag,
        startup_mode="simulation",
    )


def launch_calibrant_gui(*, bundle: str | None = None) -> None:
    """Launch the calibrant fitter GUI."""

    gui_bootstrap.launch_calibrant_gui(bundle=bundle)


def resolve_mosaic_repo_path() -> Path:
    """Return the configured local 2D_Mosaic_Sim repository path."""

    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []

    override = os.environ.get(_MOSAIC_REPO_ENV_VAR, "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(repo_root.parent / _MOSAIC_REPO_DIRNAME)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    searched = ", ".join(f"`{candidate}`" for candidate in candidates)
    raise FileNotFoundError(
        "Unable to locate the 2D_Mosaic_Sim repository. "
        f"Checked {searched}. Set `{_MOSAIC_REPO_ENV_VAR}` to override the default location."
    )


def resolve_mosaic_launcher_script(repo_path: Path) -> Path:
    """Return the preferred launcher script inside the mosaic-visualizer repo."""

    for script_name in _MOSAIC_SCRIPT_NAMES:
        script_path = repo_path / script_name
        if script_path.is_file():
            return script_path

    expected = ", ".join(f"`{name}`" for name in _MOSAIC_SCRIPT_NAMES)
    raise FileNotFoundError(
        "Unable to locate a supported 2D_Mosaic_Sim launcher script in "
        f"`{repo_path}`. Expected one of {expected}."
    )


def launch_mosaic_visualizer() -> None:
    """Launch the sibling 2D_Mosaic_Sim visualization tool."""

    repo_path = resolve_mosaic_repo_path()
    script_path = resolve_mosaic_launcher_script(repo_path)
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_path,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"2D_Mosaic_Sim exited with status {exc.returncode}."
        ) from exc


def launch_mosaic_specular_visualizer(initial_state: object) -> None:
    """Launch the unified 2D_Mosaic_Sim app directly in seeded specular mode."""

    repo_path = resolve_mosaic_repo_path()
    script_path = resolve_mosaic_launcher_script(repo_path)
    try:
        state_json = json.dumps(initial_state)
    except TypeError as exc:
        raise RuntimeError(f"Unable to serialize 2D_Mosaic_Sim startup state: {exc}") from exc

    port = _pick_available_local_port()
    try:
        subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "specular-view",
                "--port",
                str(port),
                "--state-json",
                state_json,
            ],
            cwd=repo_path,
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to launch 2D_Mosaic_Sim: {exc}") from exc


def launch_startup_mode(
    startup_mode: str | None,
    *,
    write_excel_flag: bool | None = None,
    calibrant_bundle: str | None = None,
) -> None:
    """Launch the selected startup mode."""

    if startup_mode is None:
        return
    if startup_mode == "simulation":
        launch_simulation_gui(write_excel_flag=write_excel_flag)
        return
    if startup_mode == "calibrant":
        launch_calibrant_gui(bundle=calibrant_bundle)
        return
    if startup_mode == "mosaic":
        launch_mosaic_visualizer()
        return
    raise ValueError(
        "startup_mode must resolve to one of: simulation, calibrant, mosaic"
    )


def main(argv: list[str] | None = None) -> None:
    """Run the lightweight packaged GUI launcher."""

    cli_argv = list(sys.argv[1:] if argv is None else argv)
    if gui_bootstrap.should_forward_to_cli(cli_argv):
        from ra_sim.cli import main as cli_main

        cli_main(cli_argv)
        return

    args = gui_bootstrap.parse_launch_args(cli_argv)
    startup_mode = gui_bootstrap.resolve_startup_mode(args.command)
    if startup_mode == "prompt":
        startup_mode = gui_bootstrap.quick_startup_mode_dialog()

    launch_startup_mode(
        startup_mode,
        write_excel_flag=False if args.no_excel else None,
        calibrant_bundle=args.bundle,
    )
