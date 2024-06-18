import jax
import numpy as np
import optax
import mlxp
from tqdm import tqdm

from flows import ReFlow, get_flow
from networks import MLP
from tasks import get_data
from utils.plots import plot_data, plot_loss_curve, plot_traj
from utils.train import train_rectified_flow

def cfg_to_dict(cfg):
    return {k: v for k, v in cfg.items()}


@mlxp.launch(config_path="./configs")
def run_task(ctx: mlxp.Context) -> None:
    cfg = ctx.config
    name_0 = cfg.data_0.name
    name_1 = cfg.data_1.name
    task_params_0 = cfg_to_dict(cfg.data_0.params)
    task_params_1 = cfg_to_dict(cfg.data_1.params)
    x0_params = task_params_0.get("x0", {})
    x1_params = task_params_1.get("x1", {})
    x0, y0 = get_data(cfg.num_samples, name_0, **x0_params)
    x1, y1 = get_data(cfg.num_samples, name_1, **x1_params)
    plot_data(x0, x1)

    key, subkey = jax.random.split(jax.random.PRNGKey(cfg.seed))
    # Define drift model
    drift = MLP(
        subkey,
        input_dim=x0.shape[-1] + 1,
        hidden_dim=cfg.networks.hidden_dim,
        output_dim=x1.shape[-1],
    )
    key, subkey = jax.random.split(key)

    # Define flow model
    flows_params = cfg_to_dict(cfg.flows.params)
    flow = get_flow(cfg.flows.name, drift, cfg.flows.num_steps, **flows_params)

    # Train the flow
    # schedule = optax.warmup_cosine_decay_schedule(
    #   init_value=0.0,
    #   peak_value=cfg.networks.lr,
    #   warmup_steps=100,
    #   decay_steps=800,
    #   end_value=0.0,
    # )
    #
    # optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.adam(cfg.networks.lr)
    flow, loss_curve = train_rectified_flow(
        flow,
        optimizer,
        x0,
        x1,
        cfg.networks.batch_size,
        cfg.networks.epochs,
    )

    # Iterate the process
    for i in tqdm(range(cfg.flows.k - 1)):
        z0, lab0 = get_data(cfg.num_samples, cfg.task_name, **x0_params)
        z1 = flow.sample_ode(z0)[-1]
        optimizer = optax.adam(cfg.networks.lr)
        flow, loss_curve = train_rectified_flow(
            flow,
            optimizer,
            z0,
            z1,
            cfg.networks.batch_size,
            cfg.networks.epochs,
        )
        plot_traj(flow.sample_ode(z0), x1, ntraj=cfg.plots.n_traj, title=f"traj_{i}", labels=lab0)

    plot_loss_curve(loss_curve)
    plot_traj(flow.sample_ode(x0, N=100), x1, ntraj=cfg.plots.n_traj, title="traj_final", labels=None)


if __name__ == "__main__":
    run_task()
