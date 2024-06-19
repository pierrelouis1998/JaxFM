import numpy as np

# Create a function that generates 8 gaussian
def eight_gaussian(num_samples: int, noise: float = 0.05) -> np.ndarray:
    """Generate 8 gaussians data

    Args:
        num_samples: Number of samples to generate
        noise: Noise level

    Returns:
        Data
    """
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]

    dataset = []
    for i in range(num_samples):
        point = np.random.randn(2) * 0.05
        shift = np.array([0.5, 0.5])
        scale = 2
        idx = np.random.choice(len(centers))
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        point = point * scale + shift
        dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)
    perm = np.random.permutation(len(dataset))
    return dataset[perm], None
