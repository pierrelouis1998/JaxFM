import equinox as eqx
import jax
import jax.numpy as jnp
from matplotlib.pyplot import plot, show
import numpy as np 
import optax

class MPLTest(eqx.Module):
    layers: list
    def __init__(self, key, input_dim, hidden_dim, output_dim, n_layers=3):
        keys = jax.random.split(key, n_layers)
        self.layers = []
        self.layers.append(eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]))
        for i in range(n_layers - 2):
            self.layers.extend([eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i + 1]), jax.nn.tanh])
        self.layers.append(eqx.nn.Linear(hidden_dim, output_dim, key=keys[-1]))
        # classification head
        self.layers.append(jax.nn.softmax)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def bimodal_gaussian(num_samples, mean_1=np.array([0., 0.]), mean_2=np.array([0., 3.]), var_1=0.2, var_2=0.1):
    x = np.concatenate([np.random.randn(num_samples//2, 2) * var_1 + mean_1,
                            np.random.randn(num_samples//2, 2) * var_2 + mean_2], axis=0)
    return x[np.random.permutation(len(x))]


@eqx.filter_jit
def loss_fn(model, x, y):
    y_pred = jax.vmap(model)(x)
    return -jnp.sum(y * jnp.log(y_pred))

def test_mlp(data, labels, batch_size=100):
    key = jax.random.PRNGKey(0)
    model = MPLTest(key, 2, 100, 2, n_layers=4)
    optimizer = optax.adam(1e-5)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(0)
    inner_iters = 4000
    loss_curve = []
    
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    trange = range(inner_iters)
    for i in trange:
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(data), (batch_size,), replace=False)
        key, subkey = jax.random.split(key)
        x = data[idx]
        y = labels[idx]
        model, opt_state, loss_value = make_step(model, x, y, opt_state)
        loss_curve.append(loss_value)
    return model, loss_curve

if __name__ == "__main__":
    num_samples = 1000
    test_data = np.concatenate([np.random.randn(num_samples//2, 2) * 0.2 + np.array([-5., 0.]),
                           np.random.randn(num_samples//2, 2) * 0.2 + np.array([5., 0.])], axis=0)
    labels = np.concatenate([np.zeros(num_samples//2), np.ones(num_samples//2)])
    labels = jax.nn.one_hot(labels, 2)
    perm = np.random.permutation(len(test_data))
    test_data = test_data[perm]
    labels = labels[perm]

    model, loss_curve = test_mlp(test_data, labels)
    plot(loss_curve)
    show()


