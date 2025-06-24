import time
from ra_sim.utils import stacking_fault as sf

CIF = 'tests/local_test.cif'


def run(use_jit=True):
    if not use_jit:
        orig_F2, orig_I_inf = sf._F2, sf._I_inf
        sf._F2 = sf._F2.py_func
        sf._I_inf = sf._I_inf.py_func
    start = time.perf_counter()
    sf.ht_Iinf_dict(cif_path=CIF, mx=2, p=0.1, L_step=0.02)
    elapsed = time.perf_counter() - start
    if not use_jit:
        sf._F2 = orig_F2
        sf._I_inf = orig_I_inf
    return elapsed


def main():
    # warm up JIT compilation
    sf.ht_Iinf_dict(cif_path=CIF, mx=1)
    t_py = run(False)
    t_jit = run(True)
    print(f"Python: {t_py:.4f}s, Numba: {t_jit:.4f}s")


if __name__ == "__main__":
    main()
