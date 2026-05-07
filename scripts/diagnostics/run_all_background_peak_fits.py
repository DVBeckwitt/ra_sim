"""Execute the all-background peak-fit diagnostic for one or more GUI states.

Examples
--------
python scripts/diagnostics/run_all_background_peak_fits.py path/to/state.json
python scripts/diagnostics/run_all_background_peak_fits.py --out-dir artifacts/background_peak_fits state_a.json state_b.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import TracebackType


DEFAULT_NOTEBOOK = Path(__file__).with_name("all_background_peak_fits.ipynb")
DEFAULT_STATE_PATH = Path.home() / ".local" / "share" / "ra_sim" / "all.json"


def _repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


def _safe_run_name(value: object) -> str:
    raw = str(value).strip().replace("\\", "/")
    text = Path(raw).stem if raw else "state"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "state"


def _state_paths(values: list[str]) -> list[Path]:
    if not values:
        env_value = os.environ.get("RA_SIM_ALL_BACKGROUND_STATE", "")
        values = [env_value] if env_value.strip() else [str(DEFAULT_STATE_PATH)]
    return [Path(value).expanduser().resolve() for value in values]


def _notebook_code_cells(notebook_path: Path) -> list[tuple[int, str]]:
    if notebook_path.suffix.lower() == ".py":
        return [(0, notebook_path.read_text(encoding="utf-8"))]

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"{notebook_path} is not a valid Jupyter notebook")

    code_cells: list[tuple[int, str]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            code = source
        else:
            code = "".join(str(line) for line in source)
        if code.strip():
            code_cells.append((index, code))
    return code_cells


def _output_dir_for_state(
    *,
    base_out_dir: Path | None,
    state_count: int,
    run_name: str,
    repo_root: Path,
) -> Path | None:
    if base_out_dir is None:
        return None

    out_dir = base_out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    if state_count > 1:
        out_dir = out_dir / f"{run_name}_state"
    return out_dir


def _set_or_clear_env(name: str, value: object | None) -> str | None:
    old_value = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)
    return old_value


def _set_env_if_value(name: str, value: object | None) -> str | None:
    old_value = os.environ.get(name)
    if value is not None:
        os.environ[name] = str(value)
    return old_value


def _restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def _execute_notebook(
    *,
    notebook_path: Path,
    state_path: Path,
    run_name: str,
    out_dir: Path | None,
    repo_root: Path,
    fit_workers: int | None,
    numba_threads: int | None,
    fit_backend: str | None,
    process_numba_threads: int | None,
) -> None:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_env = {
        "RA_SIM_ALL_BACKGROUND_STATE": _set_or_clear_env(
            "RA_SIM_ALL_BACKGROUND_STATE",
            state_path,
        ),
        "RA_SIM_ALL_BACKGROUND_RUN_NAME": _set_or_clear_env(
            "RA_SIM_ALL_BACKGROUND_RUN_NAME",
            run_name,
        ),
        "RA_SIM_ALL_BACKGROUND_OUT_DIR": _set_env_if_value(
            "RA_SIM_ALL_BACKGROUND_OUT_DIR",
            out_dir,
        ),
        "BACKGROUND_FIT_WORKERS": _set_env_if_value("BACKGROUND_FIT_WORKERS", fit_workers),
        "NUMBA_NUM_THREADS": _set_env_if_value("NUMBA_NUM_THREADS", numba_threads),
        "BACKGROUND_FIT_BACKEND": _set_env_if_value("BACKGROUND_FIT_BACKEND", fit_backend),
        "BACKGROUND_PROCESS_NUMBA_THREADS": _set_env_if_value(
            "BACKGROUND_PROCESS_NUMBA_THREADS",
            process_numba_threads,
        ),
        "RA_SIM_ALL_BACKGROUND_PROCESS_GUARD": _set_or_clear_env(
            "RA_SIM_ALL_BACKGROUND_PROCESS_GUARD",
            "1",
        ),
    }

    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }
    try:
        os.chdir(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        is_python_diagnostic = notebook_path.suffix.lower() == ".py"
        for cell_index, code in _notebook_code_cells(notebook_path):
            if is_python_diagnostic:
                print(f"[{run_name}] executing python diagnostic script")
                filename = str(notebook_path)
            else:
                print(f"[{run_name}] executing notebook cell {cell_index}")
                filename = f"{notebook_path}#cell-{cell_index}"
            exec(compile(code, filename, "exec"), namespace)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for key, old_value in old_env.items():
            _restore_env(key, old_value)


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _format_exception(exc: BaseException, traceback: TracebackType | None) -> str:
    location = ""
    tb = traceback
    while tb is not None:
        location = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        tb = tb.tb_next
    return f"{type(exc).__name__}: {exc}" + (f" at {location}" if location else "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an all-background peak-fit notebook or Python diagnostic for saved GUI states.",
    )
    parser.add_argument(
        "states",
        nargs="*",
        help="Saved RA-SIM GUI state JSON paths. Defaults to RA_SIM_ALL_BACKGROUND_STATE or all.json.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Notebook or Python diagnostic script to execute.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override. With multiple states, each run writes to a "
            "<state-stem>_state subdirectory under this path."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output run-name override for a single state. Defaults to the state file stem.",
    )
    parser.add_argument(
        "--fit-workers",
        type=_positive_int,
        default=None,
        help="Override BACKGROUND_FIT_WORKERS for the notebook run.",
    )
    parser.add_argument(
        "--numba-threads",
        type=_positive_int,
        default=None,
        help="Override NUMBA_NUM_THREADS for the notebook run.",
    )
    parser.add_argument(
        "--fit-backend",
        choices=("auto", "process", "thread", "serial"),
        default=None,
        help="Override BACKGROUND_FIT_BACKEND for notebooks that support it.",
    )
    parser.add_argument(
        "--process-numba-threads",
        type=_positive_int,
        default=None,
        help="Override BACKGROUND_PROCESS_NUMBA_THREADS for process-pool workers.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next state after a failed notebook execution.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    notebook_path = args.notebook.expanduser().resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    repo_root = _repo_root(notebook_path.parent)
    states = _state_paths(args.states)
    if args.run_name and len(states) != 1:
        raise SystemExit("--run-name can only be used with one state")

    base_out_dir = args.out_dir
    if base_out_dir is None and os.environ.get("RA_SIM_ALL_BACKGROUND_OUT_DIR", "").strip():
        base_out_dir = Path(os.environ["RA_SIM_ALL_BACKGROUND_OUT_DIR"])
    env_run_name = os.environ.get("RA_SIM_ALL_BACKGROUND_RUN_NAME", "").strip()

    failures: list[tuple[Path, str]] = []
    for state_path in states:
        if not state_path.exists():
            message = f"Saved GUI state not found: {state_path}"
            if not args.keep_going:
                raise FileNotFoundError(message)
            print(f"[error] {message}", file=sys.stderr)
            failures.append((state_path, message))
            continue

        run_name = _safe_run_name(
            args.run_name or (env_run_name if len(states) == 1 else "") or state_path.stem
        )
        out_dir = _output_dir_for_state(
            base_out_dir=base_out_dir,
            state_count=len(states),
            run_name=run_name,
            repo_root=repo_root,
        )
        print(f"[{run_name}] state={state_path}")
        print(f"[{run_name}] notebook={notebook_path}")
        if out_dir is not None:
            print(f"[{run_name}] out={out_dir}")
        start = time.perf_counter()
        try:
            _execute_notebook(
                notebook_path=notebook_path,
                state_path=state_path,
                run_name=run_name,
                out_dir=out_dir,
                repo_root=repo_root,
                fit_workers=args.fit_workers,
                numba_threads=args.numba_threads,
                fit_backend=args.fit_backend,
                process_numba_threads=args.process_numba_threads,
            )
        except Exception as exc:
            message = _format_exception(exc, exc.__traceback__)
            print(f"[{run_name}] failed: {message}", file=sys.stderr)
            failures.append((state_path, message))
            if not args.keep_going:
                return 1
        else:
            elapsed = time.perf_counter() - start
            print(f"[{run_name}] completed in {elapsed:.1f}s")

    if failures:
        print("Failed states:", file=sys.stderr)
        for state_path, message in failures:
            print(f"- {state_path}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
