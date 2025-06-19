import sys
from pathlib import Path

# Allow importing ra_sim from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip heavy example scripts when running tests
collect_ignore = [
    'run_diffraction_test.py',
    'optimization.py',
    'analyze_simulation_debug.py',
]
