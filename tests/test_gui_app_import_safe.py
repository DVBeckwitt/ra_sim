import importlib
import pytest
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


def test_app_dunder_attribute_raises_without_loading(monkeypatch) -> None:
    app = importlib.import_module(APP_MODULE_NAME)

    monkeypatch.setattr(
        app,
        "_load_runtime_module",
        lambda: (_ for _ in ()).throw(AssertionError("impl should not be imported")),
    )

    with pytest.raises(AttributeError):
        _ = app.__gui_app_test_guard__


def test_app_dir_is_lazy() -> None:
    app = importlib.import_module(APP_MODULE_NAME)

    available = dir(app)

    assert "main" in available
    assert "write_excel" in available


def test_app_unknown_attr_forwards_to_runtime(monkeypatch) -> None:
    app = importlib.import_module(APP_MODULE_NAME)
    fake_runtime = SimpleNamespace(test_value=321)

    monkeypatch.setattr(app, "_load_runtime_module", lambda: fake_runtime)

    assert app.test_value == 321
