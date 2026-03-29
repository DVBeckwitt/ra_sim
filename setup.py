"""Legacy setuptools shim.

Package metadata and configuration live in ``pyproject.toml``.
This file remains only for compatibility with older setuptools workflows.
"""

from __future__ import annotations

from setuptools import setup


if __name__ == "__main__":
    setup()
