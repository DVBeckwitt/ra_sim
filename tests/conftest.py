import sys
from pathlib import Path

import pytest

# Allow importing ra_sim from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ra_sim.debug_controls import reset_startup_debug_log_path


@pytest.fixture(autouse=True)
def _reset_startup_log_path() -> None:
    reset_startup_debug_log_path()
    yield
    reset_startup_debug_log_path()
