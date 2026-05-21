from __future__ import annotations

import builtins
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
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
_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _looks_like_plain_python_function(obj: Any) -> bool:
    return (
        callable(obj)
        and getattr(obj, "__code__", None) is not None
        and isinstance(getattr(obj, "__name__", None), str)
    )


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
        assert compat.NUMBA_JIT_DISABLED is False
        assert compat.NUMBA_COMPILATION_AVAILABLE is False
        yield compat


@contextmanager
def _fake_raw_njit_numba_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous_modules = _snapshot_ra_sim_modules()

    class _FakeNumbaType:
        def __getattr__(self, _name: str):
            return self

        def __getitem__(self, _item):
            return self

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return self

    class _FakeList(list):
        @classmethod
        def empty_list(cls, _item_type=None):
            del _item_type
            return cls()

    def fake_njit(*decorator_args, **decorator_kwargs):
        if (
            len(decorator_args) == 1
            and not decorator_kwargs
            and _looks_like_plain_python_function(decorator_args[0])
        ):
            return decorator_args[0]

        def _decorate(fn):
            return fn

        return _decorate

    fake_numba = ModuleType("numba")
    fake_numba.njit = fake_njit
    fake_numba.prange = range
    fake_numba.types = _FakeNumbaType()
    fake_numba.config = SimpleNamespace(DISABLE_JIT=True)
    fake_numba.get_num_threads = lambda: 1
    fake_numba.set_num_threads = lambda _value: None

    fake_typed = ModuleType("numba.typed")
    fake_typed.List = _FakeList

    try:
        _clear_ra_sim_modules()
        _clear_fallback_modules()
        monkeypatch.setitem(sys.modules, "numba", fake_numba)
        monkeypatch.setitem(sys.modules, "numba.typed", fake_typed)
        yield
    finally:
        _restore_ra_sim_modules(previous_modules)


def _run_python_with_env(code: str, env_updates: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_updates)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 77:
        message = completed.stdout.strip() or completed.stderr.strip()
        pytest.skip(message or "Numba unavailable in subprocess")
    assert completed.returncode == 0, (
        f"subprocess failed with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


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
        assert compat.NUMBA_JIT_DISABLED is False
        assert compat.NUMBA_COMPILATION_AVAILABLE is False
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
        assert compat.NUMBA_JIT_DISABLED is False
        assert compat.NUMBA_COMPILATION_AVAILABLE is False
        assert imported_modules


def test_numba_compat_raw_real_njit_results_get_py_func(monkeypatch):
    with _fake_raw_njit_numba_context(monkeypatch):
        compat = importlib.import_module("ra_sim.utils.numba_compat")

        @compat.njit
        def increment(value):
            return value + 1

        @compat.njit(cache=True, fastmath=True)
        def add_two(value):
            return value + 2

        signature = compat.types.float64(compat.types.float64)

        @compat.njit(signature)
        def add_three(value):
            return value + 3

        def add_four(value):
            return value + 4

        compiled_add_four = compat.njit(cache=True)(add_four)

        assert compat.NUMBA_AVAILABLE is True
        assert compat.NUMBA_JIT_DISABLED is True
        assert compat.NUMBA_COMPILATION_AVAILABLE is False
        assert increment(1) == 2
        assert increment.py_func is increment
        assert add_two(1) == 3
        assert add_two.py_func is add_two
        assert add_three(1) == 4
        assert add_three.py_func is add_three
        assert compiled_add_four(1) == 5
        assert compiled_add_four.py_func is compiled_add_four


def test_numba_disable_jit_diffraction_decorated_functions_keep_py_func():
    completed = _run_python_with_env(
        """
        from ra_sim.utils import numba_compat

        if not numba_compat.NUMBA_AVAILABLE:
            print(f"Numba unavailable: {numba_compat.NUMBA_IMPORT_ERROR!r}")
            raise SystemExit(77)

        from ra_sim.simulation import diffraction

        assert numba_compat.NUMBA_AVAILABLE is True
        assert numba_compat.NUMBA_JIT_DISABLED is True
        assert numba_compat.NUMBA_COMPILATION_AVAILABLE is False

        for name in (
            "compute_intensity_array_serial",
            "_weighted_event_pass1_for_qset",
        ):
            fn = getattr(diffraction, name)
            assert hasattr(fn, "py_func"), name
            assert fn.py_func is fn, name

        print("ok")
        """,
        {"NUMBA_DISABLE_JIT": "1"},
    )

    assert "ok" in completed.stdout


def test_numba_disable_jit_exact_cake_engine_resolution():
    completed = _run_python_with_env(
        """
        from ra_sim.utils import numba_compat

        if not numba_compat.NUMBA_AVAILABLE:
            print(f"Numba unavailable: {numba_compat.NUMBA_IMPORT_ERROR!r}")
            raise SystemExit(77)

        from ra_sim.simulation import exact_cake

        assert numba_compat.NUMBA_AVAILABLE is True
        assert numba_compat.NUMBA_JIT_DISABLED is True
        assert numba_compat.NUMBA_COMPILATION_AVAILABLE is False

        assert exact_cake._resolve_engine("auto") == "python"
        assert exact_cake._resolve_engine("python") == "python"

        try:
            exact_cake._resolve_engine("numba")
        except RuntimeError as exc:
            assert "JIT" in str(exc) or "disabled" in str(exc)
        else:
            raise AssertionError("engine='numba' should fail when NUMBA_DISABLE_JIT=1")

        print("ok")
        """,
        {"NUMBA_DISABLE_JIT": "1"},
    )

    assert "ok" in completed.stdout


def test_numba_compat_real_numba_path_when_available():
    compat = importlib.import_module("ra_sim.utils.numba_compat")
    if not compat.NUMBA_AVAILABLE:
        pytest.skip(f"Numba unavailable: {compat.NUMBA_IMPORT_ERROR!r}")
    if compat.NUMBA_JIT_DISABLED or not compat.NUMBA_COMPILATION_AVAILABLE:
        pytest.skip("real compiled Numba path unavailable because JIT is disabled")

    assert compat.NUMBA_IMPORT_ERROR is None
    assert compat.NUMBA_JIT_DISABLED is False
    assert compat.NUMBA_COMPILATION_AVAILABLE is True
    assert callable(compat.njit)
    assert compat.prange is not None
    assert hasattr(compat.List, "empty_list")
