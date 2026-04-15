from pathlib import Path

from ra_sim.user_paths import dev_cache_dir, user_cache_root, user_data_root


def test_user_paths_follow_home_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    assert user_cache_root() == tmp_path / ".cache" / "ra_sim"
    assert user_data_root() == tmp_path / ".local" / "share" / "ra_sim"
    assert dev_cache_dir("pytest") == tmp_path / ".cache" / "ra_sim" / "dev" / "pytest"
