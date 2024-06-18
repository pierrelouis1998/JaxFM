from typing import Tuple
import numpy as np


def bimodal_gaussian(
    num_samples: int,
    mean_1: np.ndarray = np.array([0.0, 0.0]),
    mean_2: np.ndarray = np.array([3.0, 0.0]),
    sigma_1: float = 0.2,
    sigma_2: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bimodal gaussian data

    Args:
        num_samples: Number of samples to generate
        mean_1: Mean of the first gaussian
        mean_2: Mean of the second gaussian
        sigma_1: Std of the first gaussian
        sigma_2: Std of the second gaussian

    Returns:
        Data and labels
    """
    x = np.concatenate(
        [
            np.random.randn(num_samples // 2, 2) * sigma_1 + mean_1,
            np.random.randn(num_samples // 2, 2) * sigma_2 + mean_2,
        ],
        axis=0,
    )
    labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])
    perm = np.random.permutation(len(x))
    return x[perm], labels[perm]
