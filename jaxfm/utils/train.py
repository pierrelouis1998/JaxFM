import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from jaxfm.flows.base import Flow


def get_loss_fn(name="l2"):
    if name == "l2":

        @eqx.filter_jit
        def loss_fn(model, z_t, t, target):
            vmapped_model = jax.vmap(model, in_axes=(0, 0))
            return jnp.sum((vmapped_model(z_t, t) - target) ** 2, axis=1).mean()

    return loss_fn


def train_rectified_flow(
    rectified_flow: Flow, optimizer, x0, x1, batch_size, inner_iters
):
    loss_curve = []
    opt_state = optimizer.init(eqx.filter(rectified_flow.model, eqx.is_array))
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

        z_t, t, target = rectified_flow.get_train_tuple(z0, z1, subkey)
        # print(z_t.shape, t.shape, target.shape)
        rectified_flow.model, opt_state, loss = make_step(
            rectified_flow.model, z_t, t, target, opt_state
        )
        if i == inner_iters - 1:
            vt = jax.vmap(rectified_flow.model, in_axes=(0, 0))(z_t, t)
            fig = plt.figure(figsize=(8, 5))
            angle = jnp.arctan2(vt[:, 1], vt[:, 0])
            plt.quiver(z_t[:, 0], z_t[:, 1], jnp.cos(angle), jnp.sin(angle))
            plt.savefig("fig/vt.pdf")
            plt.close(fig)
        loss_curve.append(loss)

        trange.set_postfix(loss=loss)
    return rectified_flow, loss_curve
