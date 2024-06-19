import jax.numpy as jnp
import jax

def get_t_samples(key: int, num_samples: int, law: str = "uniform", **kwargs) -> jnp.ndarray:
    """Sample the t values"""
    if law == "normal":
        t = 0.3 * jax.random.normal(key, (num_samples, 1)) + 0.5
        # rescale to [0, 1]
        t = jnp.clip(t, 0, 1)
    elif law == "uniform":
        t = jax.random.uniform(key, (num_samples, 1))
    elif law == "power":
        alpha = kwargs.get("alpha", 0.2)
        a = (1 + alpha) * -1 + 1
        t = jax.random.beta(key, a, 1, (num_samples, 1))
    else:
        raise ValueError(f"Unknown law {law}")

    return t


