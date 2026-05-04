"""Compatibility wrappers for optional Numba support."""

from __future__ import annotations

from typing import Any, Callable


NUMBA_AVAILABLE: bool
NUMBA_IMPORT_ERROR: BaseException | None
NUMBA_JIT_DISABLED: bool
NUMBA_COMPILATION_AVAILABLE: bool


def _attach_py_func(obj: Any, py_func: Callable[..., Any] | None = None) -> Any:
    if callable(obj) and not hasattr(obj, "py_func"):
        try:
            obj.py_func = py_func if py_func is not None else obj  # type: ignore[attr-defined]
        except Exception:
            pass
    return obj


def _looks_like_plain_python_function(obj: Any) -> bool:
    return (
        callable(obj)
        and getattr(obj, "__code__", None) is not None
        and isinstance(getattr(obj, "__name__", None), str)
    )


try:  # pragma: no cover - exercised by real-Numba smoke tests
    from numba import config as _numba_config
    from numba import get_num_threads, njit as _numba_njit, prange, set_num_threads, types
    from numba.typed import List

    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = None
    NUMBA_JIT_DISABLED = bool(getattr(_numba_config, "DISABLE_JIT", False))
    NUMBA_COMPILATION_AVAILABLE = NUMBA_AVAILABLE and not NUMBA_JIT_DISABLED

    def njit(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        """Wrapper for :func:`numba.njit` that preserves ``.py_func`` with JIT disabled."""

        if (
            len(decorator_args) == 1
            and not decorator_kwargs
            and _looks_like_plain_python_function(decorator_args[0])
        ):
            fn = decorator_args[0]
            compiled = _numba_njit(fn)
            return _attach_py_func(compiled, fn)

        decorator = _numba_njit(*decorator_args, **decorator_kwargs)

        if callable(decorator) and not hasattr(decorator, "py_func"):
            def _decorate(fn: Callable[..., Any]) -> Any:
                compiled = decorator(fn)
                return _attach_py_func(compiled, fn)

            return _decorate

        return _attach_py_func(decorator)

except Exception as exc:  # pragma: no cover - tested with monkeypatched import
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = exc
    NUMBA_JIT_DISABLED = False
    NUMBA_COMPILATION_AVAILABLE = False

    def njit(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        """Fallback replacement for :func:`numba.njit`.

        Supports direct decorator use, configured decorator use, and signatures:
        ``@njit``, ``@njit()``, ``@njit(cache=True)``,
        ``@njit(signature, cache=True)``, and ``njit(cache=True)(fn)``.
        """

        if (
            len(decorator_args) == 1
            and not decorator_kwargs
            and _looks_like_plain_python_function(decorator_args[0])
        ):
            return _attach_py_func(decorator_args[0])

        def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            return _attach_py_func(fn)

        return _decorate

    prange = range

    get_num_threads = None
    set_num_threads = None

    class _FallbackNumbaType:
        def __init__(self, name: str = "numba_fallback_type") -> None:
            self.name = name

        def __getitem__(self, _item: Any) -> "_FallbackNumbaType":
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> "_FallbackNumbaType":
            del args, kwargs
            return self

        def __getattr__(self, name: str) -> "_FallbackNumbaType":
            return _FallbackNumbaType(f"{self.name}.{name}")

        def __repr__(self) -> str:
            return f"<fallback {self.name}>"

    class _FallbackNumbaTypes:
        def __getattr__(self, name: str) -> _FallbackNumbaType:
            return _FallbackNumbaType(f"types.{name}")

    types = _FallbackNumbaTypes()

    class List(list):  # type: ignore[no-redef]
        @classmethod
        def empty_list(cls, _item_type: Any = None) -> "List":
            del _item_type
            return cls()
