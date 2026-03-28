"""Import-safe compatibility wrapper for the packaged GUI runtime.

The heavy Tk/matplotlib implementation now lives in
``ra_sim/gui/_runtime/runtime_impl.py`` and is loaded only when callers invoke
``main()`` or request implementation attributes explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

write_excel = False

_RUNTIME_MODULE: ModuleType | None = None
_RUNTIME_MODULE_NAME = "ra_sim.gui._runtime_impl"
_RUNTIME_MODULE_PATH = Path(__file__).with_name("_runtime") / "runtime_impl.py"


def _load_runtime_module() -> ModuleType:
    """Load the heavy GUI runtime implementation on demand."""

    global _RUNTIME_MODULE

    module = _RUNTIME_MODULE
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(
        _RUNTIME_MODULE_NAME,
        _RUNTIME_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load GUI runtime implementation from {_RUNTIME_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _RUNTIME_MODULE = module
    return module


def main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Launch the full GUI runtime lazily."""

    global write_excel
    if write_excel_flag is not None:
        write_excel = bool(write_excel_flag)

    runtime = _load_runtime_module()
    runtime.write_excel = write_excel
    runtime.main(
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )


def __getattr__(name: str):
    if name == "write_excel":
        return write_excel
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_load_runtime_module(), name)


if __name__ == "__main__":
    from ra_sim.launcher import main as launcher_main

    launcher_main(["gui", *sys.argv[1:]])
