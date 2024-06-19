import matplotlib.pyplot as plt
import numpy as np
def plot_data(z0, z1):
    plt.figure(figsize=(8, 5))
    plt.scatter(z0[:, 0], z0[:, 1], alpha=0.3, label="z0", s=10)
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.3, label="z1", s=10)
    plt.legend()
    plt.axis("equal")
    plt.savefig("fig/data.pdf")


def plot_init_traj(z0, z1, zt, n=1000):
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


def plot_loss_curve(loss_curve, **kwargs):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.savefig(f"fig/{kwargs.get('title', 'loss_curve')}.pdf")


def plot_traj(traj, x1, ntraj=None, **fig_kwargs):
    traj = np.array(traj)
    if ntraj is None:
        ntraj = traj.shape[1]
    labels = fig_kwargs.get("labels", None)
    plt.figure(figsize=(6,6))
    for i in range(ntraj):
        if labels is not None:
            idx = np.where(labels == 0)[0]
            plt.plot(
                traj[:, idx[i], 0], traj[:, idx[i], 1], alpha=0.2, color="blue", zorder=1
            )
            idx = np.where(labels == 1)[0]
            plt.plot(
                traj[:, idx[i], 0], traj[:, idx[i], 1], alpha=0.2, color="red", zorder=1
            )
        else:
            plt.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.2, color="olive", zorder=1)
    plt.scatter(traj[0, :, 0], traj[0, :, 1], label=r"$z_0$", s=4, alpha=0.5, zorder=2, color="black")
    plt.scatter(
        traj[-1, :, 0], traj[-1, :, 1], label=r"$T(z_0)$", s=4, alpha=0.2, zorder=3, color="blue"
    )
    # plt.scatter(x1[:, 0], x1[:, 1], label=r"$z_1$", s=4, alpha=0.2, zorder=2, color="red")
    plt.legend()
    plt.axis("equal")
    plt.savefig(f"fig/{fig_kwargs.get('title', 'traj')}.pdf")
