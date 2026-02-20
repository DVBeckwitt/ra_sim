import numpy as np

import ra_sim.utils.stacking_fault as stacking_fault


def test_get_base_curves_reuses_form_factor_per_radial_class(monkeypatch):
    """Equivalent HK families should not recompute element form factors."""

    stacking_fault._HT_BASE_CACHE.clear()

    monkeypatch.setattr(stacking_fault, "_cell_a_c_from_cif", lambda _path: (4.0, 10.0))
    monkeypatch.setattr(
        stacking_fault,
        "_sites_from_cif_with_factors",
        lambda _path, occ_factors=1.0: [
            (0.0, 0.0, 0.0, "Pb", 1.0),
            (1.0 / 3.0, 2.0 / 3.0, 0.25, "I", 1.0),
        ],
    )

    call_counts: dict[str, int] = {"Pb": 0, "I": 0}

    def _fake_f_comp(sym, q_vals, _energy_kev):
        q_arr = np.asarray(q_vals, dtype=float)
        element = stacking_fault._element_key(sym)
        call_counts[element] = call_counts.get(element, 0) + 1
        amp = 2.0 if element == "I" else 1.0
        return np.full(q_arr.shape, amp, dtype=np.complex128)

    monkeypatch.setattr(stacking_fault, "f_comp", _fake_f_comp)

    # Six HK pairs share m=1; one pair has m=4.
    hk_list = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0)]
    curves = stacking_fault._get_base_curves(
        cif_path="unused.cif",
        hk_list=hk_list,
        L_step=0.2,
        L_max=1.0,
        two_theta_max=None,
        lambda_=1.5406,
        occ_factors=1.0,
        phase_z_divisor=1.0,
        iodine_single_plane=False,
    )

    assert set(curves.keys()) == set(hk_list)
    assert call_counts["Pb"] == 2
    assert call_counts["I"] == 2

