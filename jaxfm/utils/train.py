"""Training utilities for rectified flows."""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float
from tqdm import tqdm

from jaxfm.flows.base import Flow


def get_loss_fn(name="l2"):
    if name == "l2":

        @eqx.filter_jit
        def loss_fn(model, z_t, t, target):
            vmapped_model = jax.vmap(model, in_axes=(0, 0))
            return jnp.sum((vmapped_model(z_t, t) - target) ** 2, axis=1).mean()

    return loss_fn


def train_flow(
    flow: Flow,
    optimizer: optax.GradientTransformation,
    x0: Float[Array, "batch_size dim"],
    x1: Float[Array, "batch_size dim"],
    batch_size: int,
    inner_iters: int,
) -> (Flow, List[float]):
    """Train the flow with data points x0 and x1.

    Args:
        flow: Instance of the flow model.
        optimizer: optimizer for training.
        x0: Source points.
        x1: Target points.
        batch_size: Batch size for training.
        inner_iters: Number of training iterations.

    Returns:
        Trained flow and loss curve.
    """
    loss_curve = []
    opt_state = optimizer.init(eqx.filter(flow.model, eqx.is_array))
    key = jax.random.PRNGKey(0)
    loss_fn = get_loss_fn()

    @eqx.filter_jit
    def make_step(model, z_t, t, target, opt_state):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, z_t, t, target)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    trange = tqdm(range(inner_iters))
    for i in trange:
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(x0), (batch_size,), replace=False)
        key, subkey = jax.random.split(key)
        z0 = x0[idx]
        z1 = x1[idx]

        z_t, t, target = flow.get_train_tuple(z0, z1, subkey)
        # print(z_t.shape, t.shape, target.shape)
        flow.model, opt_state, loss = make_step(
            flow.model, z_t, t, target, opt_state
        )
        if i == inner_iters - 1:
            vt = jax.vmap(flow.model, in_axes=(0, 0))(z_t, t)
            fig = plt.figure(figsize=(8, 5))
            angle = jnp.arctan2(vt[:, 1], vt[:, 0])
            plt.quiver(z_t[:, 0], z_t[:, 1], jnp.cos(angle), jnp.sin(angle))
            plt.savefig("fig/vt.pdf")
            plt.close(fig)
        loss_curve.append(loss)

        trange.set_postfix(loss=loss)
    return flow, loss_curve
