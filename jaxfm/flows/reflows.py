from typing import Tuple

from jaxtyping import Array, Float
from jaxfm.flows.base import Flow

from jaxfm.utils.time_scheduler import get_t_samples


class ReFlow(Flow): 
    """Implement the ReFlow model.
    Reference:
    Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow http://arxiv.org/abs/2209.03003
    """

    def __init__(self, model=None, num_steps=100, law: str = "uniform") -> None:
        super().__init__(model, num_steps, law)

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
        z_t = z0 * (1 - t) + z1 * t
        target = z1 - z0

        return z_t, t, target
