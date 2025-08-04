import numpy as np
from pathlib import Path
from ra_sim.utils.stacking_fault import (
    ht_Iinf_dict,
    ht_dict_to_qr_dict,
    qr_dict_to_arrays,
)
from ra_sim.simulation import diffraction


def combine_qr_dicts(caches, weights):
    """Minimal stand‑in for :func:`main.combine_qr_dicts`.

    It merges multiple Hendricks–Teller QR dictionaries while preserving the
    ``deg`` value for each rod irrespective of the weights.
    """
    import numpy as np

    out = {}
    for cache, w in zip(caches, weights):
        qr = cache["qr"]
        for m, data in qr.items():
            if m not in out:
                out[m] = {
                    "L": data["L"].copy(),
                    "I": w * data["I"].copy(),
                    "hk": data["hk"],
                    "deg": data.get("deg", 1),
                }
            else:
                entry = out[m]
                if entry["L"].shape != data["L"].shape or not np.allclose(entry["L"], data["L"]):
                    union_L = np.union1d(entry["L"], data["L"])
                    entry_I = np.interp(union_L, entry["L"], entry["I"], left=0.0, right=0.0)
                    add_I = w * np.interp(union_L, data["L"], data["I"], left=0.0, right=0.0)
                    entry["L"] = union_L
                    entry["I"] = entry_I + add_I
                else:
                    entry["I"] += w * data["I"]
                # ``deg`` is unchanged – it depends only on symmetry, not weights
    return out


def test_qr_grouping_matches_manual_sum():
    cif = Path('tests/Diffuse/PbI2_2H.cif')
    hk_list = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    curves = ht_Iinf_dict(
        cif_path=str(cif),
        hk_list=hk_list,
        p=0.1,
        L_step=0.1,
        L_max=1.0,
        lambda_=1.54,
    )
    manual_I = np.sum([curves[hk]['I'] for hk in hk_list], axis=0)
    qr = ht_dict_to_qr_dict(curves)
    assert list(qr.keys()) == [1]
    rod = qr[1]
    assert rod['deg'] == len(hk_list)
    assert np.allclose(rod['I'], manual_I)
    miller, intens, deg_arr, _ = qr_dict_to_arrays(qr)
    assert np.all(deg_arr == len(hk_list))
    assert np.allclose(intens, manual_I)


def test_combine_ht_caches_preserves_deg():
    cif = Path('tests/Diffuse/PbI2_2H.cif')
    hk_list = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    cache0 = {
        "qr": ht_dict_to_qr_dict(
            ht_Iinf_dict(
                cif_path=str(cif),
                hk_list=hk_list,
                p=0.1,
                L_step=0.1,
                L_max=1.0,
                lambda_=1.54,
            )
        )
    }
    cache1 = {
        "qr": ht_dict_to_qr_dict(
            ht_Iinf_dict(
                cif_path=str(cif),
                hk_list=hk_list,
                p=0.2,
                L_step=0.1,
                L_max=1.0,
                lambda_=1.54,
            )
        )
    }

    combined = combine_qr_dicts([cache0, cache1], [0.6, 0.4])
    rod = combined[1]
    assert rod["deg"] == len(hk_list)

    miller, intens, deg_arr, _ = qr_dict_to_arrays(combined)
    assert np.all(deg_arr == len(hk_list))

    # Intensities should be a weighted sum of the individual caches
    manual_I = 0.6 * cache0["qr"][1]["I"] + 0.4 * cache1["qr"][1]["I"]
    assert np.allclose(combined[1]["I"], manual_I)


def test_process_qr_rods_parallel_scales_with_degeneracy(monkeypatch):
    captured = {}

    def fake_process_peaks_parallel(miller, intensities, *args, **kwargs):
        captured["miller"] = miller
        captured["intensities"] = intensities
        return (None, None, None, None, None, None)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_process_peaks_parallel)

    qr_dict = {
        1: {
            "hk": (1, 0),
            "L": np.array([0.0, 0.5]),
            "I": np.array([1.0, 2.0]),
            "deg": 3,
        }
    }

    image = np.zeros((1, 1), dtype=np.float64)
    result = diffraction.process_qr_rods_parallel(
        qr_dict=qr_dict,
        image_size=1,
        av=1.0,
        cv=1.0,
        lambda_=1.0,
        image=image,
        Distance_CoR_to_Detector=1.0,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        zs=0.0,
        zb=0.0,
        n2=1,
        beam_x_array=np.array([0.0]),
        beam_y_array=np.array([0.0]),
        theta_array=np.array([0.0]),
        phi_array=np.array([0.0]),
        sigma_pv_deg=0.0,
        gamma_pv_deg=0.0,
        eta_pv=0.0,
        wavelength_array=np.array([1.0]),
        debye_x=0.0,
        debye_y=0.0,
        center=[0.0, 0.0],
        theta_initial_deg=0.0,
        unit_x=np.array([1.0, 0.0, 0.0]),
        n_detector=np.array([0.0, 1.0, 0.0]),
        save_flag=0,
    )

    assert np.allclose(captured["intensities"], np.array([1.0, 2.0]) * 3)
    assert np.array_equal(result[-1], np.array([3, 3], dtype=np.int32))
