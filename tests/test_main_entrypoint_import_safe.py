import importlib
import sys


MODULE_NAME = "ra_sim.__main__"
LAUNCHER_MODULE = "ra_sim.launcher"
CLI_MODULE = "ra_sim.cli"
RUNTIME_IMPL_MODULE = "ra_sim.gui._runtime.runtime_impl"


def test_main_entrypoint_imports_launcher_without_eager_cli_or_runtime() -> None:
    previous = {
        name: sys.modules.pop(name, None)
        for name in (MODULE_NAME, LAUNCHER_MODULE, CLI_MODULE, RUNTIME_IMPL_MODULE)
    }

    try:
        module = importlib.import_module(MODULE_NAME)

        assert module.__name__ == MODULE_NAME
        assert LAUNCHER_MODULE in sys.modules
        assert CLI_MODULE not in sys.modules
        assert RUNTIME_IMPL_MODULE not in sys.modules
    finally:
        for name in (MODULE_NAME, LAUNCHER_MODULE, CLI_MODULE, RUNTIME_IMPL_MODULE):
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module
