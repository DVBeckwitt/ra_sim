import sys
from types import SimpleNamespace

import pytest

from ra_sim.gui import lazy_runtime


def test_load_cached_module_from_path_cleans_sys_modules_on_failure(
    tmp_path,
) -> None:
    module_name = "ra_sim.gui._lazy_runtime_failure_test"
    module_path = tmp_path / "broken_runtime.py"
    previous_module = sys.modules.pop(module_name, None)
    module_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="boom"):
            lazy_runtime.load_cached_module_from_path(
                None,
                module_name=module_name,
                module_path=module_path,
            )

        assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_forward_lazy_main_updates_write_excel_and_forwards_args() -> None:
    calls: list[dict[str, object]] = []
    fake_runtime = SimpleNamespace(
        write_excel=None,
        main=lambda **kwargs: calls.append(dict(kwargs)),
    )

    write_excel = lazy_runtime.forward_lazy_main(
        current_write_excel=False,
        load_runtime_module=lambda: fake_runtime,
        write_excel_flag=True,
        startup_mode="simulation",
        calibrant_bundle="bundle.npz",
    )

    assert write_excel is True
    assert fake_runtime.write_excel is True
    assert calls == [
        {
            "write_excel_flag": True,
            "startup_mode": "simulation",
            "calibrant_bundle": "bundle.npz",
        }
    ]


def test_lazy_module_getattr_and_dir_keep_local_surface_lazy() -> None:
    load_calls: list[str] = []
    fake_runtime = SimpleNamespace(test_value=123)

    def _load_runtime():
        load_calls.append("load")
        return fake_runtime

    assert (
        lazy_runtime.lazy_module_getattr(
            name="write_excel",
            module_name="ra_sim.gui.runtime",
            current_write_excel=False,
            load_runtime_module=_load_runtime,
        )
        is False
    )
    assert load_calls == []

    with pytest.raises(AttributeError):
        lazy_runtime.lazy_module_getattr(
            name="__lazy_guard__",
            module_name="ra_sim.gui.runtime",
            current_write_excel=False,
            load_runtime_module=_load_runtime,
        )
    assert load_calls == []

    assert (
        lazy_runtime.lazy_module_getattr(
            name="test_value",
            module_name="ra_sim.gui.runtime",
            current_write_excel=False,
            load_runtime_module=_load_runtime,
        )
        == 123
    )
    assert load_calls == ["load"]

    available = lazy_runtime.lazy_module_dir(
        module_globals={"alpha": 1, "write_excel": False},
        loaded_module=fake_runtime,
    )

    assert "alpha" in available
    assert "write_excel" in available
    assert "test_value" in available
