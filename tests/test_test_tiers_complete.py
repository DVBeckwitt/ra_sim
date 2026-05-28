from pathlib import Path

from ra_sim import test_tiers


REPO_ROOT = Path(__file__).resolve().parents[1]


def _top_level_test_files() -> set[str]:
    return {path.name for path in (REPO_ROOT / "tests").glob("test_*.py")}


def test_all_top_level_test_files_are_assigned_to_exactly_one_tier() -> None:
    assignments: dict[str, list[str]] = {}
    tier_files = {
        "fast": test_tiers.FAST_TEST_FILES,
        "integration": test_tiers.INTEGRATION_TEST_FILES,
        "slow": test_tiers.SLOW_TEST_FILES,
        "diagnostic": test_tiers.DIAGNOSTIC_TEST_FILES,
        "benchmark": test_tiers.BENCHMARK_TEST_FILES,
    }

    for tier, filenames in tier_files.items():
        for filename in filenames:
            assignments.setdefault(filename, []).append(tier)

    top_level_tests = _top_level_test_files()
    unknown = sorted(set(assignments) - top_level_tests)
    unclassified = sorted(top_level_tests - set(assignments))
    duplicates = {
        filename: tiers for filename, tiers in sorted(assignments.items()) if len(tiers) != 1
    }

    assert not unknown, f"Tier manifest references missing tests: {unknown}"
    assert not unclassified, f"Unclassified top-level tests: {unclassified}"
    assert not duplicates, f"Tests assigned to multiple tiers: {duplicates}"
