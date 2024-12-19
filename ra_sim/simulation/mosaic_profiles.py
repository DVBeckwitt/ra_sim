import numpy as np

def sample_2d_gaussian(n, sigma):
    return np.random.normal(0, sigma, (n, 2))

def sample_2d_cauchy(n, gamma):
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    x = gamma * np.tan(np.pi*(u1 - 0.5))
    y = gamma * np.tan(np.pi*(u2 - 0.5))
    return np.column_stack((x, y))

def sample_pseudo_voigt_2d(n, eta, sigma, gamma):
    u = np.random.rand(n)
    gaussian_indices = (u >= eta)
    cauchy_indices = (u < eta)

    samples = np.zeros((n, 2))
    g_count = np.sum(gaussian_indices)
    if g_count > 0:
        samples[gaussian_indices,:] = sample_2d_gaussian(g_count, sigma)

    c_count = np.sum(cauchy_indices)
    if c_count > 0:
        samples[cauchy_indices,:] = sample_2d_cauchy(c_count, gamma)

    return samples
