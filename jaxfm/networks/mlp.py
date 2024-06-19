import equinox as eqx
from jaxtyping import Array, Float
import jax.numpy as jnp
import jax

class MLP(eqx.Module):
    """Module to learn the score function using a MLP
    The output of the model is the a vector of the same size as the input theta
    """

    layers: list

    def __init__(
        self,
        key: Array,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
    ):
        keys = jax.random.split(key, n_layers)
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]),
            jax.nn.relu,
        ]
        for i in range(n_layers - 2):
            self.layers.extend(
                [
                    eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i + 1]),
                    jax.nn.tanh,
                ]
            )
        self.layers.extend(
            [
                eqx.nn.Linear(hidden_dim, output_dim, key=keys[-1]),
            ]
        )

    def __call__(
        self, x_t: Float[Array, "dim_x_t"], t: Float[Array, "dim_t"]
    ) -> Float[Array, "dim_"]:
        inputs = jnp.concatenate([x_t, t], axis=-1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


