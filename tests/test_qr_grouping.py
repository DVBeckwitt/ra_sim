import numpy as np
from pathlib import Path
from ra_sim.utils.stacking_fault import (
    ht_Iinf_dict,
    ht_dict_to_qr_dict,
    qr_dict_to_arrays,
)


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
