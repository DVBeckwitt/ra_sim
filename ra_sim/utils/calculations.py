import numpy as np

def d_spacing(h, k, l, av, cv):
    if (h, k, l) == (0, 0, 0):
        return None
    term1 = 4/3*(h**2 + h*k + k**2)/av**2
    term2 = (l**2)/cv**2
    return 1/np.sqrt(term1 + term2)
