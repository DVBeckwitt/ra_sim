import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_NAME = "ra_sim.__main__"
LAUNCHER_MODULE = "ra_sim.launcher"
CLI_MODULE = "ra_sim.cli"
RUNTIME_IMPL_MODULE = "ra_sim.gui._runtime.runtime_impl"
NON_WINDOWS_IMPORT_SAFE_MODULES = (
    "ra_sim.gui.window_affinity",
    "ra_sim.launcher",
    "ra_sim.__main__",
)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_main_entrypoint_imports_launcher_without_eager_cli_or_runtime(monkeypatch) -> None:
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    previous = {
        name: sys.modules.pop(name, None)
        for name in (
            "ra_sim",
            MODULE_NAME,
            LAUNCHER_MODULE,
            CLI_MODULE,
            RUNTIME_IMPL_MODULE,
        )
    }

    try:
        module = importlib.import_module(MODULE_NAME)

        assert module.__name__ == MODULE_NAME
        assert "NUMBA_CACHE_DIR" in os.environ
        assert LAUNCHER_MODULE in sys.modules
        assert CLI_MODULE not in sys.modules
        assert RUNTIME_IMPL_MODULE not in sys.modules
    finally:
        for name in ("ra_sim", MODULE_NAME, LAUNCHER_MODULE, CLI_MODULE, RUNTIME_IMPL_MODULE):
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module


@pytest.mark.parametrize("module_name", NON_WINDOWS_IMPORT_SAFE_MODULES)
def test_non_windows_import_chain_stays_safe_without_windows_ctypes_helpers(
    module_name: str,
) -> None:
    script = """
import ctypes
import importlib
import sys

for attr in ("WINFUNCTYPE", "WinDLL", "OleDLL"):
    if hasattr(ctypes, attr):
        delattr(ctypes, attr)

for name in tuple(sys.modules):
    if name.startswith("ra_sim"):
        sys.modules.pop(name, None)

importlib.import_module(sys.argv[1])
"""

    subprocess.run(
        [sys.executable, "-c", script, module_name],
        check=True,
        cwd=REPO_ROOT,
    )
