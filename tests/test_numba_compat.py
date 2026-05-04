from __future__ import annotations

import builtins
from contextlib import contextmanager
import importlib
import sys
from typing import Any, Iterator

import pytest


_FORMER_DIRECT_IMPORT_MODULES = (
    "ra_sim.simulation.diffraction",
    "ra_sim.simulation.exact_cake",
    "ra_sim.StructureFactor.StructureFactor",
    "ra_sim.utils.calculations",
    "ra_sim.utils.parallel",
)

_CLEAR_FOR_FALLBACK = (
    "ra_sim.utils.numba_compat",
    *_FORMER_DIRECT_IMPORT_MODULES,
)

_OPTIONAL_DEP_NAMES = {"CifFile", "Dans_Diffraction", "xraydb", "spglib"}


def _snapshot_ra_sim_modules() -> dict[str, Any]:
    return {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "ra_sim" or name.startswith("ra_sim.")
    }


def _clear_ra_sim_modules() -> None:
    for name in list(sys.modules):
        if name == "ra_sim" or name.startswith("ra_sim."):
            sys.modules.pop(name, None)


def _clear_fallback_modules() -> None:
    for name in _CLEAR_FOR_FALLBACK:
        sys.modules.pop(name, None)


def _restore_ra_sim_modules(modules: dict[str, Any]) -> None:
    _clear_ra_sim_modules()
    sys.modules.update(modules)


def _block_numba_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("blocked numba for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


@contextmanager
def _blocked_numba_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous_modules = _snapshot_ra_sim_modules()
    try:
        _clear_ra_sim_modules()
        _clear_fallback_modules()
        _block_numba_imports(monkeypatch)
        yield
    finally:
        _restore_ra_sim_modules(previous_modules)


@contextmanager
def _import_fallback_compat(monkeypatch: pytest.MonkeyPatch):
    with _blocked_numba_context(monkeypatch):
        compat = importlib.import_module("ra_sim.utils.numba_compat")
        assert compat.NUMBA_AVAILABLE is False
        assert compat.NUMBA_IMPORT_ERROR is not None
        yield compat


def _is_unrelated_optional_dependency_failure(exc: BaseException) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return str(getattr(exc, "name", "")) in _OPTIONAL_DEP_NAMES
    return any(name in str(exc) for name in _OPTIONAL_DEP_NAMES)


def test_numba_compat_fallback_njit_direct_form(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:

        @compat.njit
        def add(left, right):
            return left + right

        assert add(2, 3) == 5
        assert add.py_func is add


def test_numba_compat_fallback_njit_kwargs_form(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:

        @compat.njit(cache=True, fastmath=True)
        def add(left, right):
            return left + right

        assert add(2, 3) == 5
        assert add.py_func is add


def test_numba_compat_fallback_njit_empty_signature_and_call_forms(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:
        signature = compat.types.float64(compat.types.float64)

        @compat.njit()
        def increment(value):
            return value + 1

        @compat.njit(signature, cache=True)
        def square(value):
            return value * value

        def decrement(value):
            return value - 1

        compiled_decrement = compat.njit(cache=True)(decrement)

        assert increment(2) == 3
        assert increment.py_func is increment
        assert square(4) == 16
        assert square.py_func is square
        assert compiled_decrement(5) == 4
        assert compiled_decrement.py_func is compiled_decrement


def test_numba_compat_fallback_prange_is_range(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:
        assert list(compat.prange(3)) == [0, 1, 2]


def test_numba_compat_fallback_list_empty_list(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:
        lst = compat.List.empty_list(compat.types.float64[:, ::1])

        lst.append([1.0, 2.0])

        assert list(lst) == [[1.0, 2.0]]
        assert isinstance(compat.List(), list)


def test_numba_compat_fallback_types_accept_project_signature_shapes(monkeypatch):
    with _import_fallback_compat(monkeypatch) as compat:
        type_expressions = [
            compat.types.float64[:, ::1],
            compat.types.int64[:],
            compat.types.Tuple((compat.types.float64, compat.types.int64)),
            compat.types.UniTuple(compat.types.float64, 2),
            compat.types.ListType(compat.types.float64),
            compat.types.Array(compat.types.float64, 2, "C"),
        ]

        assert all("fallback" in repr(type_expr) for type_expr in type_expressions)


def test_diffraction_imports_when_numba_import_fails(monkeypatch):
    with _blocked_numba_context(monkeypatch):
        diffraction = importlib.import_module("ra_sim.simulation.diffraction")
        compat = importlib.import_module("ra_sim.utils.numba_compat")

        assert compat.NUMBA_AVAILABLE is False
        assert compat.NUMBA_IMPORT_ERROR is not None
        assert diffraction.NUMBA_AVAILABLE is False
        assert diffraction.NUMBA_IMPORT_ERROR is compat.NUMBA_IMPORT_ERROR
        assert diffraction.attenuation.py_func is diffraction.attenuation


def test_exact_cake_engine_selection_with_numba_import_failure(monkeypatch):
    with _blocked_numba_context(monkeypatch):
        exact_cake = importlib.import_module("ra_sim.simulation.exact_cake")

        assert exact_cake._resolve_engine("auto") == "python"
        assert exact_cake._resolve_engine("python") == "python"
        with pytest.raises(RuntimeError, match="engine='numba' requested"):
            exact_cake._resolve_engine("numba")


def test_numba_direct_import_modules_import_with_fallback(monkeypatch):
    imported_modules = []
    with _blocked_numba_context(monkeypatch):
        for module_name in _FORMER_DIRECT_IMPORT_MODULES:
            try:
                imported_modules.append(importlib.import_module(module_name))
            except ImportError as exc:
                if _is_unrelated_optional_dependency_failure(exc):
                    continue
                raise

        compat = importlib.import_module("ra_sim.utils.numba_compat")
        assert compat.NUMBA_AVAILABLE is False
        assert imported_modules


def test_numba_compat_real_numba_path_when_available():
    compat = importlib.import_module("ra_sim.utils.numba_compat")
    if not compat.NUMBA_AVAILABLE:
        pytest.skip(f"Numba unavailable: {compat.NUMBA_IMPORT_ERROR!r}")

    assert compat.NUMBA_IMPORT_ERROR is None
    assert callable(compat.njit)
    assert compat.prange is not None
    assert hasattr(compat.List, "empty_list")
