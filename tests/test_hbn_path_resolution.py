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
