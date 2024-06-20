"""Plotting functions for Flow Matching."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float


def plot_data(
    z0: Float[Array, "nsamples dim"], z1: Float[Array, "nsamples dim"]
) -> None:
    """Plot initial data points."""
    plt.figure(figsize=(8, 5))
    plt.scatter(z0[:, 0], z0[:, 1], alpha=0.3, label="z0", s=10)
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.3, label="z1", s=10)
    plt.legend()
    plt.axis("equal")
    plt.savefig("fig/data.pdf")


def plot_init_traj(
    z0: Float[Array, "nsamples dim"],
    z1: Float[Array, "nsamples dim"],
    zt: Float[Array, "nsamples dim"],
    n: int = 1000,
) -> None:
    """Plot initial trajectories.

    Args:
        z0: Source points.
        z1: Target points.
        zt: Trajectory points.
        n: Number of trajectories to plot.
    """
    plt.figure(figsize=(8, 5))
    assert len(z0) == len(z1) == len(zt)
    idx = np.random.choice(len(z0), n)
    z0 = z0[idx]  # + np.array([-5., 0.])
    z1 = z1[idx]  # + np.array([5., 0.])
    plt.scatter(z0[:, 0], z0[:, 1], alpha=0.4, label="z0", s=10)
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.4, label="z1", s=10)
    # plt.plot(zt[:, 0], zt[:, 1], 'go', label='zt')
    plt.legend()
    plt.savefig("fig/init_traj.pdf")


def plot_loss_curve(loss_curve: List[float], **kwargs) -> None:
    """Plot the loss curve of the neural vector field.

    Args:
        loss_curve: Loss at each iteration.
        **kwargs: Figure options.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig(f"fig/{kwargs.get('title', 'loss_curve')}.pdf")


def plot_traj(
    traj: Float[Array, "batch_size ntraj dim"],
    x1: Float[Array, "batch_size dim"],
    ntraj: Optional[int] = None,
    **fig_kwargs,
) -> None:
    """Plot a given number of trajectories.

    Args:
        traj: Array of trajectories (batch_size, ntraj, dim).
        x1: Target points.
        ntraj: Number of trajectories to plot.
        **fig_kwargs: Figure options.
    """
    if ntraj is None:
        ntraj = traj.shape[1]
    labels = fig_kwargs.get("labels", None)
    plt.figure(figsize=(6, 6))
    plt.tight_layout()
    for i in range(ntraj):
        if labels is not None:
            idx = np.where(labels == 0)[0]
            plt.plot(
                traj[:, idx[i], 0],
                traj[:, idx[i], 1],
                alpha=0.2,
                color="blue",
                zorder=1,
            )
            idx = np.where(labels == 1)[0]
            plt.plot(
                traj[:, idx[i], 0], traj[:, idx[i], 1], alpha=0.2, color="red", zorder=1
            )
        else:
            plt.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.2, color="olive", zorder=1)
    plt.scatter(
        traj[0, :, 0],
        traj[0, :, 1],
        label=r"$z_0$",
        s=4,
        alpha=0.5,
        zorder=2,
        color="black",
    )
    plt.scatter(
        traj[-1, :, 0],
        traj[-1, :, 1],
        label=r"$T(z_0)$",
        s=4,
        alpha=0.2,
        zorder=3,
        color="blue",
    )
    plot_target = fig_kwargs.get("plot_target", False)
    if plot_target:
        plt.scatter(
            x1[:, 0],
            x1[:, 1],
            label=r"$z_1$",
            s=4,
            alpha=0.2,
            zorder=2,
            color="red",
        )
    plt.legend(fontsize=12)
    plt.axis("equal")
    plt.savefig(f"fig/{fig_kwargs.get('title', 'traj')}.pdf")
