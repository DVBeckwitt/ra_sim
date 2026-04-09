"""Import-safe wrapper for the packaged Tk runtime session.

The heavy runtime implementation now lives in
``ra_sim.gui._runtime.runtime_session`` and is loaded only when callers invoke
``main()`` or request implementation attributes explicitly.
"""

from __future__ import annotations

import sys
from types import ModuleType

from ra_sim.gui import lazy_runtime as gui_lazy_runtime

write_excel = False

_RUNTIME_MODULE: ModuleType | None = None
_RUNTIME_MODULE_NAME = "ra_sim.gui._runtime.runtime_session"
__all__ = ["main", "write_excel"]


def _load_runtime_module() -> ModuleType:
    """Load the heavy runtime session implementation on demand."""

    global _RUNTIME_MODULE

    module = gui_lazy_runtime.load_cached_imported_module(
        _RUNTIME_MODULE,
        module_name=_RUNTIME_MODULE_NAME,
    )
    _RUNTIME_MODULE = module
    return module


def main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Launch the full GUI runtime session lazily."""

    global write_excel
    write_excel = gui_lazy_runtime.forward_lazy_main(
        current_write_excel=write_excel,
        load_runtime_module=_load_runtime_module,
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )


def __getattr__(name: str):
    return gui_lazy_runtime.lazy_module_getattr(
        name=name,
        module_name=__name__,
        current_write_excel=write_excel,
        load_runtime_module=_load_runtime_module,
    )


def __dir__() -> list[str]:
    return gui_lazy_runtime.lazy_module_dir(
        module_globals=globals(),
        loaded_module=_RUNTIME_MODULE,
    )


if __name__ == "__main__":
    from ra_sim.launcher import main as launcher_main

    launcher_main(["gui", *sys.argv[1:]])
