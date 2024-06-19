from jaxfm.flows.base import Flow
from jaxfm.utils.time_scheduler import get_t_samples
from jaxtyping import Array, Float
from typing import Tuple
import jax


class CFM(Flow):
    """Conditional Flow model
    Implements Conditional Flow Matching for the Optimal Transport objective i.e. constant vector field.

    Reference:
    Flow Matching for Generative Modeling (https://arxiv.org/abs/2210.02747)
    """

    def __init__(self, model=None, num_steps=100, law: str = "uniform", noise=0.01) -> None:
        """Initialize the Flow Matching procedure with the model and the number of steps

        Args:
            model : The model to learn the vector field
            num_steps : The number of steps for the Euler integration
            noise : The noise to add to the samples
        """
        self.model = model  # The model to learn the vector field
        self.N = num_steps  # The number of steps for the Euler integration
        self.sigma = noise  # The noise to add to the samples
        self.law = law # The law for the time sampling

    def get_train_tuple( # type: ignore
        self,
        z0: Float[Array, "batch_size dim"],
        z1: Float[Array, "batch_size dim"],
        key: int,
    ) -> Tuple[
        Float[Array, "batch_size dim"],
        Float[Array, "batch_size 1"],
        Float[Array, "batch_size dim"],
    ]:
        """Interpolate the samples and get the target.

        Args:
            z0: Source samples
            z1: Target samples
            key: Random key

        Returns:
            Interpolated samples, time samples, target
        """
        t = get_t_samples(key, z0.shape[0], law=self.law)
        z_t = (
            z0 * (1 - t)
            + z1 * t
            + self.sigma * jax.random.normal(key, (z0.shape[0], z0.shape[1]))
        )
        return z_t, t, z1 - z0
