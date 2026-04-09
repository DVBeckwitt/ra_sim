"""Module entry to expose ``python -m ra_sim``.

Route through the lightweight launcher first so GUI startup paths avoid paying
the shared CLI import cost before startup mode is resolved.
"""

from .launcher import main

if __name__ == "__main__":  # pragma: no cover
    main()

