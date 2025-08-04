import numpy as np
from pathlib import Path
from ra_sim.utils.stacking_fault import ht_Iinf_dict, ht_dict_to_qr_dict, qr_dict_to_arrays


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
