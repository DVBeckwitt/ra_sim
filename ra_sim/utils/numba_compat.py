"""Compatibility wrappers for optional Numba support."""

from __future__ import annotations

import inspect
from typing import Any, Callable


NUMBA_AVAILABLE: bool
NUMBA_IMPORT_ERROR: BaseException | None


def _attach_py_func(fn: Callable[..., Any]) -> Callable[..., Any]:
    if not hasattr(fn, "py_func"):
        try:
            fn.py_func = fn  # type: ignore[attr-defined]
        except Exception:
            pass
    return fn


try:  # pragma: no cover - exercised by real-Numba smoke tests
    from numba import get_num_threads, njit, prange, set_num_threads, types
    from numba.typed import List

    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = None

except Exception as exc:  # pragma: no cover - tested with monkeypatched import
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = exc

    def _looks_like_decorated_function(value: Any) -> bool:
        return inspect.isfunction(value) or inspect.ismethod(value)

    def njit(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        """Fallback replacement for :func:`numba.njit`.

        Supports direct decorator use, configured decorator use, and signatures:
        ``@njit``, ``@njit()``, ``@njit(cache=True)``,
        ``@njit(signature, cache=True)``, and ``njit(cache=True)(fn)``.
        """

        if (
            len(decorator_args) == 1
            and not decorator_kwargs
            and _looks_like_decorated_function(decorator_args[0])
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
