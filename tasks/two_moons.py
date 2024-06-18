import numpy as np

def two_moons(num_samples: int, noise: float = 0.05) -> np.ndarray:
    """Generate two moons data

    Args:
        num_samples: Number of samples to generate
        noise: Noise level

    Returns:
        Data
    """
    from sklearn.datasets import make_moons

    x, labels = make_moons(n_samples=num_samples, noise=noise)
    perm = np.random.permutation(len(x))
    return x[perm], labels[perm]
