"""Shared import-safe helpers for GUI runtime wrapper modules."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable


def load_cached_imported_module(
    cached_module: ModuleType | None,
    *,
    module_name: str,
) -> ModuleType:
    """Import one module name lazily while honoring an existing local cache."""

    if cached_module is not None:
        return cached_module
    return importlib.import_module(module_name)


def load_cached_module_from_path(
    cached_module: ModuleType | None,
    *,
    module_name: str,
    module_path: Path,
) -> ModuleType:
    """Load one module from a file path lazily and clean up on failure."""

    if cached_module is not None:
        return cached_module

    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GUI runtime implementation from {module_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def forward_lazy_main(
    *,
    current_write_excel: bool,
    load_runtime_module: Callable[[], ModuleType],
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> bool:
    """Apply wrapper-local flags and forward the GUI launch call lazily."""

    write_excel = bool(current_write_excel)
    if write_excel_flag is not None:
        write_excel = bool(write_excel_flag)

    runtime = load_runtime_module()
    runtime.write_excel = write_excel
    runtime.main(
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )
    return write_excel


def lazy_module_getattr(
    *,
    name: str,
    module_name: str,
    current_write_excel: bool,
    load_runtime_module: Callable[[], ModuleType],
):
    """Resolve one wrapper attribute without importing on dunder lookups."""

    if name == "write_excel":
        return current_write_excel
    if name.startswith("__"):
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    return getattr(load_runtime_module(), name)


def lazy_module_dir(
    *,
    module_globals: dict[str, object],
    loaded_module: ModuleType | None,
) -> list[str]:
    """Return a lazy merged dir surface for one wrapper module."""

    names = set(module_globals.keys())
    if loaded_module is not None:
        names.update(dir(loaded_module))
    return sorted(names)
