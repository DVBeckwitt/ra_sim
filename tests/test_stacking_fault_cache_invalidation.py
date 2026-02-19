import os
from pathlib import Path

import numpy as np
import pytest

import ra_sim.utils.stacking_fault as stacking_fault


def _write_mock_cif(path: Path, iodine_z: str) -> None:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 4.0",
                "_cell_length_c 10.0",
                f"I1 I 0 0 {iodine_z}",
            ]
        ),
        encoding="utf-8",
    )


def test_infer_iodine_z_reads_quoted_atom_site_loop(tmp_path):
    cif_path = tmp_path / "quoted_loop.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 4.0",
                "_cell_length_c 10.0",
                "loop_",
                "  _atom_site_label",
                "  _atom_site_fract_z",
                "  'I1' 0.2675",
                "  'Pb1' 0.0",
            ]
        ),
        encoding="utf-8",
    )

    z_val = stacking_fault._infer_iodine_z_like_diffuse(str(cif_path))
    assert z_val == pytest.approx(0.2675)


def test_get_base_curves_recomputes_f2_when_cif_changes(tmp_path, monkeypatch):
    stacking_fault._HT_BASE_CACHE.clear()
    cif_path = tmp_path / "cache_toggle.cif"
    _write_mock_cif(cif_path, "0.1000")

    monkeypatch.setattr(stacking_fault, "_cell_a_c_from_cif", lambda _path: (4.0, 10.0))
    monkeypatch.setattr(
        stacking_fault,
        "_sites_from_cif_with_factors",
        lambda _path, occ_factors=1.0: [
            (0.0, 0.0, 0.0, "Pb", 1.0),
            (0.0, 0.0, 0.2, "I", 1.0),
        ],
    )

    def _fake_f_comp(sym, q_vals, _energy_kev):
        q_arr = np.asarray(q_vals, dtype=float)
        amp = 2.0 if stacking_fault._element_key(sym) == "I" else 1.0
        return np.full(q_arr.shape, amp, dtype=np.complex128)

    monkeypatch.setattr(stacking_fault, "f_comp", _fake_f_comp)

    first = stacking_fault._get_base_curves(
        cif_path=str(cif_path),
        hk_list=[(0, 0)],
        L_step=0.5,
        L_max=1.0,
        occ_factors=1.0,
    )
    f2_first = np.asarray(first[(0, 0)]["F2"], dtype=float).copy()

    prev_mtime_ns = cif_path.stat().st_mtime_ns
    _write_mock_cif(cif_path, "0.2000")
    updated = cif_path.stat()
    if updated.st_mtime_ns <= prev_mtime_ns:
        os.utime(
            cif_path,
            ns=(updated.st_atime_ns, prev_mtime_ns + 1_000_000),
        )

    second = stacking_fault._get_base_curves(
        cif_path=str(cif_path),
        hk_list=[(0, 0)],
        L_step=0.5,
        L_max=1.0,
        occ_factors=1.0,
    )
    f2_second = np.asarray(second[(0, 0)]["F2"], dtype=float).copy()

    assert not np.allclose(f2_first, f2_second)
