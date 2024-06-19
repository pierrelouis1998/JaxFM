"""Base class for flow models."""

from abc import abstractmethod
import jax

import jax.numpy as jnp
from typing import Tuple

from jaxtyping import Array, Float


class Flow:
    """Base class for flow models."""

    def __init__(self, model=None, num_steps=100, law: str= "uniform") -> None:
        self.model = model
        self.N = num_steps
        self.law = law

    @abstractmethod
    def get_train_tuple(
        self,
        z0: Float[Array, "batch_size dim"],
        z1: Float[Array, "batch_size dim"],
        key: int,
        **kwargs,
    ) -> Tuple[
        Float[Array, "batch_size dim"],
        Float[Array, "batch_size 1"],
        Float[Array, "batch_size dim"],
    ]:
        """Get the interpolated samples and the target."""
        pass

    def sample_ode(self, z0: jnp.ndarray, N=None) -> jnp.ndarray:
        """Sample the ODE"""
        if N is None:
            N = self.N

        dt = 1 / N
        traj = []
        z = z0
        batch_size = z0.shape[0]

        @jax.jit
        def euler_step(z, i, dt):
            t = jnp.ones((batch_size, 1)) * i * dt
            pred = jax.vmap(self.model)(z, t)
            return z + pred * dt

        traj.append(z)
        for i in range(N):
            z = euler_step(z, i, dt)
            traj.append(z)
        return jnp.asarray(traj)
