from __future__ import annotations

from pathlib import Path

from ra_sim.hbn_geometry import resolve_hbn_paths


def test_resolve_hbn_paths_uses_active_config_dir_for_auto_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    bundle_path = tmp_path / "bundle_from_env.npz"
    paths_file = config_dir / "hbn_paths.yaml"
    paths_file.write_text(
        "\n".join(
            [
                f"bundle: {bundle_path}",
                "beam_center: [11.5, 22.5]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(config_dir))

    resolved = resolve_hbn_paths()

    assert Path(str(resolved["paths_file"])).resolve() == paths_file.resolve()
    assert resolved["bundle"] == str(bundle_path)
    assert resolved["beam_center"] == (11.5, 22.5)


def test_explicit_paths_file_resolves_bundle_relative_to_yaml_parent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RA_SIM_CONFIG_DIR", raising=False)

    config_dir = tmp_path / "configs" / "nested"
    config_dir.mkdir(parents=True)
    paths_file = config_dir / "custom_hbn_paths.yaml"
    paths_file.write_text("bundle: bundles/test.npz\n", encoding="utf-8")

    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    resolved = resolve_hbn_paths(paths_file=str(paths_file))

    assert Path(resolved["bundle"]) == (config_dir / "bundles" / "test.npz").resolve()


def test_auto_discovered_custom_config_resolves_bundle_relative_to_config_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "custom-config"
    config_dir.mkdir()
    paths_file = config_dir / "hbn_paths.yaml"
    paths_file.write_text("bundle: ./artifacts/x.npz\n", encoding="utf-8")

    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    monkeypatch.setenv("RA_SIM_CONFIG_DIR", str(config_dir))
    monkeypatch.chdir(other_cwd)

    resolved = resolve_hbn_paths()

    assert Path(str(resolved["paths_file"])).resolve() == paths_file.resolve()
    assert Path(resolved["bundle"]) == (config_dir / "artifacts" / "x.npz").resolve()
