from __future__ import annotations

__all__ = ["ionic_atomic_form_factors", "F_comp"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .factors import F_comp, ionic_atomic_form_factors

    globals().update(
        {
            "ionic_atomic_form_factors": ionic_atomic_form_factors,
            "F_comp": F_comp,
        }
    )
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
