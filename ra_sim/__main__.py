"""Module entry to expose `python -m ra_sim` CLI.

Delegates to `ra_sim.cli.main`.
"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    main()

