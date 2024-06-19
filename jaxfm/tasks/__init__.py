from jaxfm.tasks.bimodal_gaussians import bimodal_gaussian
from jaxfm.tasks.two_moons import two_moons
from jaxfm.tasks.eight_gaussians import eight_gaussian

def get_data(num_samples: int, task_name: str, **kwargs):
    """Return data for the given task

    Args:
        num_samples: Number of samples to generate
        task_name: Name of the task
        **kwargs: Task specific arguments

    Raises:
        ValueError: If the task is unknown

    Returns:
        Data and labels
    """
    if task_name == "bimodal_gaussian":
        return bimodal_gaussian(num_samples, **kwargs)
    if task_name == "two_moons":
        return two_moons(num_samples, **kwargs)
    if task_name == "eight_gaussian":
        return eight_gaussian(num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown task {task_name}")
