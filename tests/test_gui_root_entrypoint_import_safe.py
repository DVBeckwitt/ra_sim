import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT_MAIN_NAME = "ra_sim_root_main"
ROOT_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _load_root_main() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        ROOT_MAIN_NAME,
        ROOT_MAIN_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load root main.py module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_main_module_is_safe_import() -> None:
    previous_main = sys.modules.pop(ROOT_MAIN_NAME, None)
    previous_runtime = sys.modules.pop("ra_sim.gui.runtime", None)
    previous_impl = sys.modules.pop("ra_sim.gui._runtime_impl", None)

    try:
        module = _load_root_main()

        assert module.__name__ == ROOT_MAIN_NAME
        assert "ra_sim.gui.runtime" not in sys.modules
        assert "ra_sim.gui._runtime_impl" not in sys.modules
        assert callable(module.main)
    finally:
        sys.modules.pop(ROOT_MAIN_NAME, None)
        if previous_main is not None:
            sys.modules[ROOT_MAIN_NAME] = previous_main
        sys.modules.pop("ra_sim.gui.runtime", None)
        sys.modules.pop("ra_sim.gui._runtime_impl", None)
        if previous_runtime is not None:
            sys.modules["ra_sim.gui.runtime"] = previous_runtime
        if previous_impl is not None:
            sys.modules["ra_sim.gui._runtime_impl"] = previous_impl


def test_root_main_delegates_to_launcher_without_importing_runtime(monkeypatch) -> None:
    module = _load_root_main()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        module,
        "_launcher_main",
        lambda *args: calls.append(list(args)),
    )

    module.main()
    assert calls == [[]]
