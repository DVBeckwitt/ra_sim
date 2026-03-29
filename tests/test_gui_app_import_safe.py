import importlib
import sys
from types import SimpleNamespace

APP_MODULE_NAME = "ra_sim.gui.app"
RUNTIME_MODULE_NAME = "ra_sim.gui.runtime"
RUNTIME_IMPL_MODULE_NAME = "ra_sim.gui._runtime_impl"


def test_app_import_is_lazy() -> None:
    previous_app = sys.modules.pop(APP_MODULE_NAME, None)
    previous_runtime = sys.modules.pop(RUNTIME_MODULE_NAME, None)
    previous_impl = sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)

    try:
        app = importlib.import_module(APP_MODULE_NAME)

        assert app.__name__ == APP_MODULE_NAME
        assert RUNTIME_MODULE_NAME not in sys.modules
        assert RUNTIME_IMPL_MODULE_NAME not in sys.modules
        assert callable(app.main)
    finally:
        sys.modules.pop(APP_MODULE_NAME, None)
        sys.modules.pop(RUNTIME_MODULE_NAME, None)
        sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)
        if previous_app is not None:
            sys.modules[APP_MODULE_NAME] = previous_app
        if previous_runtime is not None:
            sys.modules[RUNTIME_MODULE_NAME] = previous_runtime
        if previous_impl is not None:
            sys.modules[RUNTIME_IMPL_MODULE_NAME] = previous_impl


def test_app_main_loads_runtime_and_forwards_arguments(monkeypatch) -> None:
    app = importlib.import_module(APP_MODULE_NAME)
    calls: list[dict[str, object]] = []

    fake_runtime = SimpleNamespace(
        write_excel=None,
        main=lambda **kwargs: calls.append(dict(kwargs)),
    )

    monkeypatch.setattr(app, "_load_runtime_module", lambda: fake_runtime)
    app.write_excel = False

    app.main(
        write_excel_flag=True,
        startup_mode="simulation",
        calibrant_bundle="bundle.npz",
    )

    assert fake_runtime.write_excel is True
    assert calls == [
        {
            "write_excel_flag": True,
            "startup_mode": "simulation",
            "calibrant_bundle": "bundle.npz",
        }
    ]
