from __future__ import annotations

import types

import pytest

from ra_sim import install_prereqs


def test_require_tkinter_modules_returns_imported_modules(monkeypatch) -> None:
    imports: list[str] = []
    fake_tk = types.SimpleNamespace(name="tk")
    fake_ttk = types.SimpleNamespace(name="ttk")

    def _fake_import_module(module_name: str):
        imports.append(module_name)
        if module_name == "tkinter":
            return fake_tk
        if module_name == "tkinter.ttk":
            return fake_ttk
        raise AssertionError(f"Unexpected module import: {module_name}")

    monkeypatch.setattr(install_prereqs.importlib, "import_module", _fake_import_module)

    modules = install_prereqs.require_tkinter_modules("The RA-SIM simulation GUI")

    assert modules == install_prereqs.TkinterModules(tk=fake_tk, ttk=fake_ttk)
    assert imports == ["tkinter", "tkinter.ttk"]


def test_require_tkinter_modules_raises_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(
        install_prereqs.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ModuleNotFoundError(module_name)),
    )

    with pytest.raises(install_prereqs.MissingPrerequisiteError) as exc_info:
        install_prereqs.require_tkinter_modules("The RA-SIM calibrant GUI")

    message = str(exc_info.value)
    assert "The RA-SIM calibrant GUI requires Tkinter" in message
    assert "python3-tk" in message
    assert "python3.11-tk" in message
    assert "Windows and macOS Python distributions usually bundle Tk already" in message
    assert "`python -m ra_sim simulate`" in message
    assert "`python -m ra_sim hbn-fit`" in message
